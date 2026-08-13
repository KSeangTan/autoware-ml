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

"""Unit tests for the GPU stats callback: per-rank metric names, the gather across
ranks, the logging interval, and sanity-check suppression."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

import torch

from autoware_ml.callbacks.gpu_stats_monitor import GPUSample, GPUStatsMonitor


def _module(world_size: int = 1, global_rank: int = 0) -> MagicMock:
    """Build a stand-in LightningModule whose all_gather stacks per-rank rows."""
    module = MagicMock()
    module.log = MagicMock()
    module.device = torch.device("cpu")
    module.trainer.world_size = world_size
    module.trainer.global_rank = global_rank

    def all_gather(tensor: torch.Tensor) -> torch.Tensor:
        # Rank n reports n times the rank-zero reading, so rows stay distinguishable.
        return torch.stack([tensor * (rank + 1) for rank in range(world_size)])

    module.all_gather = MagicMock(side_effect=all_gather)
    return module


def _trainer(
    sanity_checking: bool = False, is_global_zero: bool = True, world_size: int = 1
) -> MagicMock:
    """Build a stand-in Trainer with the flags the callback reads."""
    trainer = MagicMock()
    trainer.sanity_checking = sanity_checking
    trainer.is_global_zero = is_global_zero
    trainer.world_size = world_size
    return trainer


def _logged(module: MagicMock) -> dict[str, float]:
    """Return the metric name to value mapping the callback logged."""
    return {call.args[0]: call.args[1] for call in module.log.call_args_list}


def _monitor(samples: list[GPUSample], **kwargs: object) -> GPUStatsMonitor:
    """Build a monitor whose device readings are replayed from a list."""
    monitor = GPUStatsMonitor(**kwargs)  # type: ignore[arg-type]
    monitor._handle = object()  # stand in for the resolved NVML handle
    readings = iter(samples)
    monitor._read_device = lambda: next(readings)  # type: ignore[method-assign]
    return monitor


class TestPerRankMetrics(unittest.TestCase):
    """Every rank's device is reported under its own metric name."""

    def test_single_rank_logs_rank_zero_metrics(self) -> None:
        """Without distribution the reading is still tagged with the rank id."""
        monitor = _monitor([GPUSample(utilization_percent=87.0, memory_gb=4.5)])
        module = _module()

        monitor.on_train_batch_end(_trainer(), module, outputs=None, batch=None, batch_idx=0)

        logged = _logged(module)
        self.assertAlmostEqual(logged["train/batch_end/gpu_util_rank0"], 87.0)
        self.assertAlmostEqual(logged["train/batch_end/gpu_memory_gb_rank0"], 4.5)

    def test_every_rank_is_logged_separately(self) -> None:
        """Two ranks produce two distinct pairs of metrics."""
        monitor = _monitor([GPUSample(utilization_percent=50.0, memory_gb=2.0)])
        module = _module(world_size=2)

        monitor.on_train_batch_end(
            _trainer(world_size=2), module, outputs=None, batch=None, batch_idx=0
        )

        logged = _logged(module)
        self.assertAlmostEqual(logged["train/batch_end/gpu_util_rank0"], 50.0)
        self.assertAlmostEqual(logged["train/batch_end/gpu_memory_gb_rank0"], 2.0)
        self.assertAlmostEqual(logged["train/batch_end/gpu_util_rank1"], 100.0)
        self.assertAlmostEqual(logged["train/batch_end/gpu_memory_gb_rank1"], 4.0)

    def test_readings_are_gathered_across_ranks(self) -> None:
        """The callback gathers rather than relying on rank zero's device alone."""
        monitor = _monitor([GPUSample(utilization_percent=10.0, memory_gb=1.0)])
        module = _module(world_size=4)

        monitor.on_train_batch_end(
            _trainer(world_size=4), module, outputs=None, batch=None, batch_idx=0
        )

        module.all_gather.assert_called_once()
        self.assertIn("train/batch_end/gpu_util_rank3", _logged(module))


class TestBatchStartHook(unittest.TestCase):
    """Both ends of an iteration are sampled, under their own metric names."""

    def test_batch_start_is_recorded(self) -> None:
        """The batch-start reading lands under the batch_start prefix."""
        monitor = _monitor([GPUSample(utilization_percent=12.0, memory_gb=1.5)])
        module = _module()

        monitor.on_train_batch_start(_trainer(), module, batch=None, batch_idx=0)

        logged = _logged(module)
        self.assertAlmostEqual(logged["train/batch_start/gpu_util_rank0"], 12.0)
        self.assertAlmostEqual(logged["train/batch_start/gpu_memory_gb_rank0"], 1.5)

    def test_both_hooks_of_one_iteration_are_separate(self) -> None:
        """The two readings of an iteration do not overwrite each other."""
        monitor = _monitor(
            [
                GPUSample(utilization_percent=12.0, memory_gb=1.0),
                GPUSample(utilization_percent=95.0, memory_gb=8.0),
            ]
        )
        module = _module()
        trainer = _trainer()

        monitor.on_train_batch_start(trainer, module, batch=None, batch_idx=0)
        monitor.on_train_batch_end(trainer, module, outputs=None, batch=None, batch_idx=0)

        logged = _logged(module)
        self.assertAlmostEqual(logged["train/batch_start/gpu_util_rank0"], 12.0)
        self.assertAlmostEqual(logged["train/batch_end/gpu_util_rank0"], 95.0)

    def test_every_stage_records_both_hooks(self) -> None:
        """Train, validation and test each sample at both ends."""
        for stage in ("train", "val", "test"):
            with self.subTest(stage=stage):
                monitor = _monitor(
                    [
                        GPUSample(utilization_percent=20.0, memory_gb=2.0),
                        GPUSample(utilization_percent=80.0, memory_gb=6.0),
                    ]
                )
                module = _module()
                start, end = {
                    "train": (monitor.on_train_batch_start, monitor.on_train_batch_end),
                    "val": (monitor.on_validation_batch_start, monitor.on_validation_batch_end),
                    "test": (monitor.on_test_batch_start, monitor.on_test_batch_end),
                }[stage]

                start(_trainer(), module, batch=None, batch_idx=0)
                end(_trainer(), module, outputs=None, batch=None, batch_idx=0)

                logged = _logged(module)
                self.assertAlmostEqual(logged[f"{stage}/batch_start/gpu_util_rank0"], 20.0)
                self.assertAlmostEqual(logged[f"{stage}/batch_end/gpu_util_rank0"], 80.0)


class TestCollectiveSafety(unittest.TestCase):
    """The gather must run on every rank, and only the logging is rank-scoped."""

    def test_non_zero_rank_gathers_but_does_not_log(self) -> None:
        """A worker rank participates in the collective and logs nothing."""
        monitor = _monitor([GPUSample(utilization_percent=30.0, memory_gb=1.5)])
        module = _module(world_size=2, global_rank=1)

        monitor.on_train_batch_end(
            _trainer(is_global_zero=False, world_size=2),
            module,
            outputs=None,
            batch=None,
            batch_idx=0,
        )

        module.all_gather.assert_called_once()
        self.assertEqual(module.log.call_args_list, [])

    def test_logging_is_marked_rank_zero_only(self) -> None:
        """Gathered values must not be reduced across ranks a second time."""
        monitor = _monitor([GPUSample(utilization_percent=30.0, memory_gb=1.5)])
        module = _module()

        monitor.on_train_batch_end(_trainer(), module, outputs=None, batch=None, batch_idx=0)

        for call in module.log.call_args_list:
            self.assertTrue(call.kwargs["rank_zero_only"])


class TestLogInterval(unittest.TestCase):
    """Recording is gated on batch_idx so every rank makes the same decision."""

    def test_only_matching_batches_are_sampled(self) -> None:
        """With an interval of two, the odd batches are skipped."""
        monitor = _monitor(
            [
                GPUSample(utilization_percent=10.0, memory_gb=1.0),
                GPUSample(utilization_percent=20.0, memory_gb=2.0),
            ],
            log_interval=2,
        )
        module = _module()
        trainer = _trainer()

        for batch_idx in range(4):
            monitor.on_train_batch_end(
                trainer, module, outputs=None, batch=None, batch_idx=batch_idx
            )

        # Batches 0 and 2 sampled: the last logged value is the second reading.
        self.assertAlmostEqual(_logged(module)["train/batch_end/gpu_util_rank0"], 20.0)
        self.assertEqual(module.all_gather.call_count, 0)  # single rank takes the fast path


class TestInactiveOrSanityChecking(unittest.TestCase):
    """Nothing is recorded without a device or during sanity checking."""

    def test_sanity_check_iterations_are_not_recorded(self) -> None:
        """No metric is logged while the trainer is sanity checking."""
        monitor = _monitor([GPUSample(utilization_percent=50.0, memory_gb=2.0)])
        module = _module()
        sanity = _trainer(sanity_checking=True)

        monitor.on_validation_batch_end(sanity, module, outputs=None, batch=None, batch_idx=0)

        self.assertEqual(module.log.call_args_list, [])

    def test_without_a_device_handle_nothing_is_read(self) -> None:
        """A CPU run leaves the callback inert instead of failing."""
        monitor = GPUStatsMonitor()
        module = _module()

        monitor.on_train_batch_end(_trainer(), module, outputs=None, batch=None, batch_idx=0)

        self.assertEqual(module.log.call_args_list, [])


class TestConstruction(unittest.TestCase):
    """Invalid configuration is rejected up front."""

    def test_non_positive_log_interval_is_rejected(self) -> None:
        """A zero interval would divide by zero, so it is refused."""
        with self.assertRaisesRegex(ValueError, "log_interval"):
            GPUStatsMonitor(log_interval=0)


if __name__ == "__main__":
    unittest.main()
