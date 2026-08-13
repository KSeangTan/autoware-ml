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

"""Unit tests for the CPU stats callback: readings at both batch hooks, the
logging interval, and sanity-check suppression."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from autoware_ml.callbacks.cpu_stats_monitor import CPUStatsMonitor


class _Readings:
    """Deterministic replacement for ``psutil.cpu_percent(interval=None)``."""

    def __init__(self, readings: list[float]) -> None:
        self._readings = iter(readings)

    def __call__(self) -> float:
        return next(self._readings)


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


def _logged(module: MagicMock) -> dict[str, float]:
    """Return the metric name to value mapping the callback logged."""
    return {call.args[0]: call.args[1] for call in module.log.call_args_list}


def _monitor(readings: list[float], **kwargs: object) -> CPUStatsMonitor:
    """Build a monitor whose utilization readings are replayed from a list."""
    monitor = CPUStatsMonitor(memory_stats=False, **kwargs)  # type: ignore[arg-type]
    monitor._cpu_percent = _Readings(readings)  # type: ignore[method-assign]
    return monitor


def _run_train_iters(
    monitor: CPUStatsMonitor, trainer: MagicMock, module: MagicMock, iterations: int
) -> None:
    """Drive whole train iterations, consuming two readings each."""
    for batch_idx in range(iterations):
        monitor.on_train_batch_start(trainer, module, batch=None, batch_idx=batch_idx)
        monitor.on_train_batch_end(trainer, module, outputs=None, batch=None, batch_idx=batch_idx)


class TestHookMetrics(unittest.TestCase):
    """Each hook logs its own reading under its own metric name."""

    def test_both_hooks_are_recorded(self) -> None:
        """The batch-start and batch-end readings land under separate names."""
        monitor = _monitor([12.0, 75.0])
        module = _module()

        _run_train_iters(monitor, _trainer(), module, iterations=1)

        logged = _logged(module)
        self.assertAlmostEqual(logged["train/batch_start/cpu_percent"], 12.0)
        self.assertAlmostEqual(logged["train/batch_end/cpu_percent"], 75.0)

    def test_first_iteration_is_recorded(self) -> None:
        """Every hook records, including the first batch-start of an epoch."""
        monitor = _monitor([42.0, 20.0])
        module = _module()

        _run_train_iters(monitor, _trainer(), module, iterations=1)

        self.assertAlmostEqual(_logged(module)["train/batch_start/cpu_percent"], 42.0)

    def test_every_stage_is_recorded(self) -> None:
        """Train, validation and test hooks each log under their own prefix."""
        for stage in ("train", "val", "test"):
            with self.subTest(stage=stage):
                monitor = _monitor([33.0, 66.0])
                module = _module()
                start, end = {
                    "train": (monitor.on_train_batch_start, monitor.on_train_batch_end),
                    "val": (monitor.on_validation_batch_start, monitor.on_validation_batch_end),
                    "test": (monitor.on_test_batch_start, monitor.on_test_batch_end),
                }[stage]

                start(_trainer(), module, batch=None, batch_idx=0)
                end(_trainer(), module, outputs=None, batch=None, batch_idx=0)

                logged = _logged(module)
                self.assertAlmostEqual(logged[f"{stage}/batch_start/cpu_percent"], 33.0)
                self.assertAlmostEqual(logged[f"{stage}/batch_end/cpu_percent"], 66.0)

    def test_memory_is_sampled_at_both_hooks_when_enabled(self) -> None:
        """RAM and swap accompany each reading unless disabled."""
        monitor = CPUStatsMonitor(memory_stats=True)
        monitor._cpu_percent = _Readings([10.0, 20.0])  # type: ignore[method-assign]
        module = _module()

        _run_train_iters(monitor, _trainer(), module, iterations=1)

        logged = _logged(module)
        for name in (
            "train/batch_start/cpu_vm_percent",
            "train/batch_start/cpu_swap_percent",
            "train/batch_end/cpu_vm_percent",
            "train/batch_end/cpu_swap_percent",
        ):
            self.assertIn(name, logged)


class TestLogInterval(unittest.TestCase):
    """Only the batches landing on the interval are recorded."""

    def test_off_interval_batches_are_not_logged(self) -> None:
        """With an interval of two, batch 1 records nothing."""
        monitor = _monitor([10.0, 20.0, 30.0, 40.0], log_interval=2)
        module = _module()
        trainer = _trainer()

        monitor.on_train_batch_start(trainer, module, batch=None, batch_idx=1)
        monitor.on_train_batch_end(trainer, module, outputs=None, batch=None, batch_idx=1)

        self.assertEqual(module.log.call_args_list, [])

    def test_on_interval_batches_are_logged(self) -> None:
        """Batch 2 lands on an interval of two and is recorded."""
        monitor = _monitor([10.0, 20.0], log_interval=2)
        module = _module()
        trainer = _trainer()

        monitor.on_train_batch_start(trainer, module, batch=None, batch_idx=2)
        monitor.on_train_batch_end(trainer, module, outputs=None, batch=None, batch_idx=2)

        logged = _logged(module)
        self.assertAlmostEqual(logged["train/batch_start/cpu_percent"], 10.0)
        self.assertAlmostEqual(logged["train/batch_end/cpu_percent"], 20.0)

    def test_the_cpu_is_still_read_on_skipped_batches(self) -> None:
        """Sampling every hook keeps each reading bound to one hook-to-hook interval."""
        readings = _Readings([1.0, 2.0, 3.0, 4.0])
        monitor = CPUStatsMonitor(memory_stats=False, log_interval=2)
        monitor._cpu_percent = readings  # type: ignore[method-assign]
        module = _module()
        trainer = _trainer()

        # Batch 1 is skipped but must still consume its two readings.
        monitor.on_train_batch_start(trainer, module, batch=None, batch_idx=1)
        monitor.on_train_batch_end(trainer, module, outputs=None, batch=None, batch_idx=1)
        monitor.on_train_batch_start(trainer, module, batch=None, batch_idx=2)
        monitor.on_train_batch_end(trainer, module, outputs=None, batch=None, batch_idx=2)

        logged = _logged(module)
        self.assertAlmostEqual(logged["train/batch_start/cpu_percent"], 3.0)
        self.assertAlmostEqual(logged["train/batch_end/cpu_percent"], 4.0)


class TestSanityCheck(unittest.TestCase):
    """Sanity-check iterations are excluded from the metrics."""

    def test_sanity_check_iterations_are_not_recorded(self) -> None:
        """No metric is logged while the trainer is sanity checking."""
        monitor = _monitor([10.0, 20.0])
        module = _module()
        sanity = _trainer(sanity_checking=True)

        monitor.on_validation_batch_start(sanity, module, batch=None, batch_idx=0)
        monitor.on_validation_batch_end(sanity, module, outputs=None, batch=None, batch_idx=0)

        self.assertEqual(module.log.call_args_list, [])


class TestConstruction(unittest.TestCase):
    """Invalid configuration is rejected up front."""

    def test_non_positive_log_interval_is_rejected(self) -> None:
        """A zero interval would divide by zero, so it is refused."""
        with self.assertRaisesRegex(ValueError, "log_interval"):
            CPUStatsMonitor(log_interval=0)


if __name__ == "__main__":
    unittest.main()
