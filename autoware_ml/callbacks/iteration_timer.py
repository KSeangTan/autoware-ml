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

"""Per-iteration wall-clock timing for the train, validation, and test loops."""

from __future__ import annotations

import logging
import statistics
import time
from typing import Any

import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback

from autoware_ml.dataclasses.multi_task_batch_inputs import MultiTaskBatchInputs

logger = logging.getLogger(__name__)

TRAIN = "train"
VAL = "val"
TEST = "test"


class _StageTimer:
    """Accumulate per-iteration timings for a single loop stage.

    Attributes:
        iter_times: Iteration durations collected in the current epoch.
        data_times: Batch-fetch durations collected in the current epoch.
    """

    def __init__(self) -> None:
        self.iter_times: list[float] = []
        self.data_times: list[float] = []
        self._batch_start: float | None = None
        self._batch_end: float | None = None

    def reset(self) -> None:
        """Drop all timings and pending timestamps, ready for a new epoch."""
        self.iter_times.clear()
        self.data_times.clear()
        self._batch_start = None
        self._batch_end = None

    def start_batch(self, now: float) -> float | None:
        """Mark the beginning of an iteration.

        Args:
            now: Current timestamp, in seconds.

        Returns:
            Seconds spent fetching this batch, or ``None`` for the first
            iteration of an epoch, where there is no previous batch to
            measure against.
        """
        self._batch_start = now
        if self._batch_end is None:
            return None
        data_time = now - self._batch_end
        self.data_times.append(data_time)
        return data_time

    def end_batch(self, now: float) -> float | None:
        """Mark the end of an iteration.

        Args:
            now: Current timestamp, in seconds.

        Returns:
            Seconds the iteration took, or ``None`` when no matching
            :meth:`start_batch` was recorded.
        """
        self._batch_end = now
        if self._batch_start is None:
            return None
        iter_time = now - self._batch_start
        self._batch_start = None
        self.iter_times.append(iter_time)
        return iter_time


class IterationTimer(Callback):
    """Record how long each train, validation, and test iteration takes.

    Every iteration contributes two step metrics, ``{stage}/iter_time`` (the
    time spent between the batch-start and batch-end hooks) and
    ``{stage}/data_time`` (the gap since the previous iteration ended, which
    is dominated by waiting on the dataloader). At the end of each epoch the
    callback logs ``{stage}/iter_time_mean``, ``{stage}/iter_time_max`` and
    ``{stage}/data_time_mean``, all in seconds.

    Because CUDA kernels are launched asynchronously, a batch-end hook can be
    reached while the GPU is still busy; ``sync_cuda`` inserts a device
    synchronization before each timestamp so the measurement reflects real
    device work, at the cost of removing CPU/GPU overlap.

    Args:
        sync_cuda: Synchronize CUDA before taking a timestamp. Timings become
            accurate per iteration, but the loop loses CPU/GPU overlap.
        warmup_iters: Number of leading iterations per epoch excluded from the
            epoch summary. The first iterations pay for dataloader worker
            startup and kernel autotuning and are not representative.
        log_interval: Log an iteration timing line to the Python logger every
            this many iterations. ``0`` disables console logging.
    """

    def __init__(
        self,
        sync_cuda: bool = False,
        warmup_iters: int = 1,
        log_interval: int = 0,
    ) -> None:
        if warmup_iters < 0:
            raise ValueError(f"warmup_iters must be non-negative, got {warmup_iters}.")
        if log_interval < 0:
            raise ValueError(f"log_interval must be non-negative, got {log_interval}.")
        self.sync_cuda = sync_cuda
        self.warmup_iters = warmup_iters
        self.log_interval = log_interval
        self._timers = {stage: _StageTimer() for stage in (TRAIN, VAL, TEST)}

    def _now(self) -> float:
        """Return the current time, optionally after draining CUDA work.

        Returns:
            Monotonic timestamp in seconds.
        """
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()

    def _batch_start(self, stage: str, trainer: Trainer, pl_module: LightningModule) -> None:
        """Time the batch fetch and open a new iteration for ``stage``.

        Args:
            stage: Loop stage being timed.
            trainer: Active trainer.
            pl_module: Module used to log metrics.
        """
        if trainer.sanity_checking:
            return
        data_time = self._timers[stage].start_batch(self._now())
        if data_time is not None:
            self._log(pl_module, f"{stage}/data_time", data_time)

    def _batch_end(self, stage: str, trainer: Trainer, pl_module: LightningModule) -> None:
        """Close the open iteration for ``stage`` and log its duration.

        Args:
            stage: Loop stage being timed.
            trainer: Active trainer.
            pl_module: Module used to log metrics.
        """
        if trainer.sanity_checking:
            return
        timer = self._timers[stage]
        iter_time = timer.end_batch(self._now())
        if iter_time is None:
            return
        self._log(pl_module, f"{stage}/iter_time", iter_time)
        count = len(timer.iter_times)
        if self.log_interval and count % self.log_interval == 0:
            logger.info("%s iteration %d: %.4fs", stage, count, iter_time)

    def _epoch_end(self, stage: str, trainer: Trainer, pl_module: LightningModule) -> None:
        """Summarize the finished epoch for ``stage`` and reset its timer.

        Args:
            stage: Loop stage being timed.
            trainer: Active trainer.
            pl_module: Module used to log metrics.
        """
        if trainer.sanity_checking:
            self._timers[stage].reset()
            return
        timer = self._timers[stage]
        iter_times = timer.iter_times[self.warmup_iters :]
        data_times = timer.data_times[self.warmup_iters :]
        if iter_times:
            self._log(
                pl_module, f"{stage}/iter_time_mean", statistics.fmean(iter_times), on_step=False
            )
            self._log(pl_module, f"{stage}/iter_time_max", max(iter_times), on_step=False)
            logger.info(
                "%s epoch timing over %d iterations (%d warmup excluded): mean %.4fs, max %.4fs",
                stage,
                len(iter_times),
                min(self.warmup_iters, len(timer.iter_times)),
                statistics.fmean(iter_times),
                max(iter_times),
            )
        if data_times:
            self._log(
                pl_module, f"{stage}/data_time_mean", statistics.fmean(data_times), on_step=False
            )
        timer.reset()

    @staticmethod
    def _log(pl_module: LightningModule, name: str, value: float, on_step: bool = True) -> None:
        """Log a timing metric through the module's loggers.

        The fixed ``batch_size`` keeps Lightning from inferring a reduction
        weight from the batch: a timing is a property of the iteration, not of
        the samples in it.

        Args:
            pl_module: Module used to log metrics.
            name: Metric name.
            value: Metric value, in seconds.
            on_step: Log per step rather than per epoch.
        """
        pl_module.log(
            name,
            value,
            on_step=on_step,
            on_epoch=not on_step,
            prog_bar=False,
            batch_size=1,
        )

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: MultiTaskBatchInputs,
        batch_idx: int,
    ) -> None:
        """Open the timing window for a training iteration."""
        self._batch_start(TRAIN, trainer, pl_module)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: MultiTaskBatchInputs,
        batch_idx: int,
    ) -> None:
        """Close the timing window for a training iteration."""
        self._batch_end(TRAIN, trainer, pl_module)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Log the training epoch timing summary."""
        self._epoch_end(TRAIN, trainer, pl_module)

    def on_validation_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: MultiTaskBatchInputs,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Open the timing window for a validation iteration."""
        self._batch_start(VAL, trainer, pl_module)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: MultiTaskBatchInputs,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Close the timing window for a validation iteration."""
        self._batch_end(VAL, trainer, pl_module)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Log the validation epoch timing summary."""
        self._epoch_end(VAL, trainer, pl_module)

    def on_test_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: MultiTaskBatchInputs,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Open the timing window for a test iteration."""
        self._batch_start(TEST, trainer, pl_module)

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: MultiTaskBatchInputs,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Close the timing window for a test iteration."""
        self._batch_end(TEST, trainer, pl_module)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Log the test epoch timing summary."""
        self._epoch_end(TEST, trainer, pl_module)
