# Copyright 2026 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the data processing timing callback: step metric, epoch totals,
sanity-check suppression, and console logging at a fixed batch interval."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from autoware_ml.callbacks.data_processing_timer import DataProcessingTimer

_LOGGER_NAME = "autoware_ml.callbacks.data_processing_timer"


def _module() -> MagicMock:
    """Build a stand-in LightningModule that records ``log()`` calls."""
    module = MagicMock()
    module.log = MagicMock()
    return module


def _trainer(sanity_checking: bool = False) -> MagicMock:
    """Build a stand-in Trainer with the sanity-checking flag set."""
    trainer = MagicMock()
    trainer.sanity_checking = sanity_checking
    return trainer


def _batch(io_processing_time: float) -> MagicMock:
    """Build a stand-in for MultiTaskBatchInputs carrying an IO timing."""
    batch = MagicMock(spec=["multi_task_gt_batch"])
    batch.multi_task_gt_batch.io_processing_time = io_processing_time
    return batch


def _logged(module: MagicMock) -> dict[str, float]:
    """Return the metric name to value mapping the callback logged."""
    return {call.args[0]: call.args[1] for call in module.log.call_args_list}


class TestStepMetric(unittest.TestCase):
    """The per-batch IO time is logged for every loop stage."""

    def test_batch_io_time_is_logged(self) -> None:
        """A training batch logs its IO time under the train-prefixed metric name."""
        callback = DataProcessingTimer()
        module = _module()

        callback.on_train_batch_start(_trainer(), module, batch=_batch(0.5), batch_idx=0)

        self.assertAlmostEqual(_logged(module)["train/data_processing_total_time"], 0.5)

    def test_every_stage_is_recorded(self) -> None:
        """Train, validation and test hooks each log under their own prefix."""
        for stage in ("train", "val", "test"):
            with self.subTest(stage=stage):
                callback = DataProcessingTimer()
                module = _module()
                hook = {
                    "train": callback.on_train_batch_start,
                    "val": callback.on_validation_batch_start,
                    "test": callback.on_test_batch_start,
                }[stage]

                hook(_trainer(), module, batch=_batch(0.25), batch_idx=0)

                self.assertAlmostEqual(_logged(module)[f"{stage}/data_processing_total_time"], 0.25)


class TestEpochSummary(unittest.TestCase):
    """The epoch hooks summarise and then reset the accumulated batch timings."""

    def test_epoch_total_mean_and_max(self) -> None:
        """The summary reports the sum, per-batch mean and slowest batch."""
        callback = DataProcessingTimer()
        module = _module()
        trainer = _trainer()

        for io_time in (1.0, 2.0, 3.0):
            callback.on_train_batch_start(trainer, module, batch=_batch(io_time), batch_idx=0)
        callback.on_train_epoch_end(trainer, module)

        logged = _logged(module)
        self.assertAlmostEqual(logged["train/data_processing_total_time_sum"], 6.0)
        self.assertAlmostEqual(logged["train/data_processing_total_time_mean"], 2.0)
        self.assertAlmostEqual(logged["train/data_processing_total_time_max"], 3.0)

    def test_epoch_end_resets_the_accumulator(self) -> None:
        """Timings from a finished epoch do not leak into the next one."""
        callback = DataProcessingTimer()
        module = _module()
        trainer = _trainer()

        callback.on_train_batch_start(trainer, module, batch=_batch(10.0), batch_idx=0)
        callback.on_train_epoch_end(trainer, module)
        module.log.reset_mock()

        callback.on_train_batch_start(trainer, module, batch=_batch(1.0), batch_idx=0)
        callback.on_train_epoch_end(trainer, module)

        self.assertAlmostEqual(_logged(module)["train/data_processing_total_time_sum"], 1.0)

    def test_stages_accumulate_independently(self) -> None:
        """A validation epoch summary does not consume the training timings."""
        callback = DataProcessingTimer()
        module = _module()
        trainer = _trainer()

        callback.on_train_batch_start(trainer, module, batch=_batch(4.0), batch_idx=0)
        callback.on_validation_batch_start(trainer, module, batch=_batch(1.0), batch_idx=0)
        callback.on_validation_epoch_end(trainer, module)
        callback.on_train_epoch_end(trainer, module)

        logged = _logged(module)
        self.assertAlmostEqual(logged["val/data_processing_total_time_sum"], 1.0)
        self.assertAlmostEqual(logged["train/data_processing_total_time_sum"], 4.0)

    def test_summary_is_skipped_without_batches(self) -> None:
        """An epoch that recorded no batches logs nothing."""
        callback = DataProcessingTimer()
        module = _module()

        callback.on_test_epoch_end(_trainer(), module)

        self.assertEqual(module.log.call_args_list, [])


class TestSanityCheck(unittest.TestCase):
    """Sanity-check batches are excluded from the metrics."""

    def test_sanity_check_batches_are_not_recorded(self) -> None:
        """Neither the batch metric nor the epoch summary is logged while sanity checking."""
        callback = DataProcessingTimer()
        module = _module()
        sanity = _trainer(sanity_checking=True)

        callback.on_validation_batch_start(sanity, module, batch=_batch(1.0), batch_idx=0)
        callback.on_validation_epoch_end(sanity, module)

        self.assertEqual(module.log.call_args_list, [])


class TestLogInterval(unittest.TestCase):
    """Console logging happens every ``log_interval`` batches, counted per stage."""

    def _run_batches(self, callback: DataProcessingTimer, count: int) -> None:
        """Feed ``count`` training batches through the callback."""
        trainer, module = _trainer(), _module()
        for batch_idx in range(count):
            callback.on_train_batch_start(trainer, module, batch=_batch(0.5), batch_idx=batch_idx)

    def test_every_nth_batch_is_logged(self) -> None:
        """Only the batches landing on the interval reach the console."""
        callback = DataProcessingTimer(log_interval=2)

        with self.assertLogs(_LOGGER_NAME, level="INFO") as captured:
            self._run_batches(callback, count=5)

        # Batches 2 and 4 of 5 land on the interval.
        lines = [
            record.message for record in captured.records if "IO processing batch" in record.message
        ]
        self.assertEqual(len(lines), 2)
        self.assertIn("batch 2", lines[0])
        self.assertIn("batch 4", lines[1])

    def test_zero_disables_console_logging(self) -> None:
        """A zero interval leaves only the per-epoch summary."""
        callback = DataProcessingTimer(log_interval=0)

        with self.assertNoLogs(_LOGGER_NAME, level="INFO"):
            self._run_batches(callback, count=5)

    def test_interval_counts_per_stage(self) -> None:
        """Batches from different stages are counted against separate intervals."""
        callback = DataProcessingTimer(log_interval=2)
        trainer, module = _trainer(), _module()

        with self.assertNoLogs(_LOGGER_NAME, level="INFO"):
            # One train batch and one val batch must not add up to the interval.
            callback.on_train_batch_start(trainer, module, batch=_batch(0.5), batch_idx=0)
            callback.on_validation_batch_start(trainer, module, batch=_batch(0.5), batch_idx=0)

    def test_negative_interval_is_rejected(self) -> None:
        """Construction fails rather than silently accepting a negative interval."""
        with self.assertRaisesRegex(ValueError, "log_interval"):
            DataProcessingTimer(log_interval=-1)


if __name__ == "__main__":
    unittest.main()
