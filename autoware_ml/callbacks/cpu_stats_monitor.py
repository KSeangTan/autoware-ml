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

"""CPU utilization sampled at the batch-start and batch-end hooks."""

from __future__ import annotations

import logging
from typing import Any

import psutil
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback

from autoware_ml.dataclasses.multi_task_batch_inputs import MultiTaskBatchInputs

logger = logging.getLogger(__name__)

TRAIN = "train"
VAL = "val"
TEST = "test"

BATCH_START = "batch_start"
BATCH_END = "batch_end"


class CPUStatsMonitor(Callback):
    """Record CPU utilization at the batch-start and batch-end hooks.

    Modelled on Lightning's ``DeviceStatsMonitor``, but reading
    ``psutil.cpu_percent(interval=None)`` at both ends of every iteration. That
    call reports system-wide utilization since the previous call, so each
    reading covers the interval that just elapsed:

    - ``{stage}/batch_start/cpu_percent`` is read when a batch starts, so it
      covers the period since the previous batch ended, in which the dataloader
      workers build the next batch.
    - ``{stage}/batch_end/cpu_percent`` is read when the batch ends, so it
      covers the model's forward and backward pass. A low value here means the
      CPU sat idle while the GPU worked, so the workers were not preparing the
      next batch alongside it.

    Both are percentages of total system capacity across all logical cores, and
    both cover every process on the machine, not just this job.

    System memory is sampled alongside them as
    ``{stage}/{hook}/cpu_vm_percent`` and ``{stage}/{hook}/cpu_swap_percent``,
    matching the fields ``DeviceStatsMonitor`` reports.

    The CPU is read at every hook so that each reading covers one hook-to-hook
    interval, but only the readings landing on ``log_interval`` are logged.
    Sampling less often would stretch each interval across several iterations
    and leave both hooks reporting the same span.

    ``psutil.cpu_percent`` keeps a single module-wide comparison point, so any
    other caller in the same process consumes the interval this callback is
    measuring. In particular, enabling Lightning's ``DeviceStatsMonitor`` with
    ``cpu_stats=True`` calls it once per hook as well and will split these
    readings.

    Args:
        log_interval: Record the metrics every this many batches. The batch
            index drives the decision, so every rank records the same batches.
        memory_stats: Sample system-wide RAM and swap alongside utilization.
    """

    def __init__(self, log_interval: int = 1, memory_stats: bool = True) -> None:
        if log_interval < 1:
            raise ValueError(f"log_interval must be positive, got {log_interval}.")
        self.log_interval = log_interval
        self.memory_stats = memory_stats
        # The first call has no comparison point and always returns 0.0, so it is
        # spent here rather than on the first recorded reading.
        psutil.cpu_percent(interval=None)

    def _cpu_percent(self) -> float:
        """Return system-wide CPU utilization since the previous reading.

        Returns:
            Utilization percentage across all logical cores.
        """
        return psutil.cpu_percent(interval=None)

    def _record(
        self,
        stage: str,
        hook: str,
        trainer: Trainer,
        pl_module: LightningModule,
        batch_idx: int,
    ) -> None:
        """Sample the CPU at one hook and log the reading on the interval.

        The reading is taken on every call so that consecutive readings bound
        one hook-to-hook interval each, and discarded unless this batch lands on
        ``log_interval``.

        Args:
            stage: Loop stage being measured.
            hook: Either ``batch_start`` or ``batch_end``.
            trainer: Active trainer.
            pl_module: Module used to log metrics.
            batch_idx: Index of the current batch.
        """
        if trainer.sanity_checking:
            return
        cpu_percent = self._cpu_percent()
        if batch_idx % self.log_interval != 0:
            return
        self._log(pl_module, f"{stage}/{hook}/cpu_percent", cpu_percent)
        if self.memory_stats:
            self._log(pl_module, f"{stage}/{hook}/cpu_vm_percent", psutil.virtual_memory().percent)
            self._log(pl_module, f"{stage}/{hook}/cpu_swap_percent", psutil.swap_memory().percent)

    @staticmethod
    def _log(pl_module: LightningModule, name: str, value: float) -> None:
        """Log a CPU metric through the module's loggers.

        Readings are logged per step only: an epoch average of a machine-wide
        utilization sample says little, and ``on_epoch`` would also make
        Lightning suffix the key with ``_step``/``_epoch``.

        The fixed ``batch_size`` keeps Lightning from inferring a reduction
        weight from the batch: utilization is a property of the iteration, not
        of the samples in it.

        Args:
            pl_module: Module used to log metrics.
            name: Metric name.
            value: Metric value, as a percentage.
        """
        pl_module.log(
            name,
            value,
            on_step=True,
            on_epoch=False,
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
        """Sample the CPU as a training iteration starts."""
        self._record(TRAIN, BATCH_START, trainer, pl_module, batch_idx)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: MultiTaskBatchInputs,
        batch_idx: int,
    ) -> None:
        """Sample the CPU as a training iteration ends."""
        self._record(TRAIN, BATCH_END, trainer, pl_module, batch_idx)

    def on_validation_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: MultiTaskBatchInputs,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Sample the CPU as a validation iteration starts."""
        self._record(VAL, BATCH_START, trainer, pl_module, batch_idx)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: MultiTaskBatchInputs,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Sample the CPU as a validation iteration ends."""
        self._record(VAL, BATCH_END, trainer, pl_module, batch_idx)

    def on_test_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: MultiTaskBatchInputs,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Sample the CPU as a test iteration starts."""
        self._record(TEST, BATCH_START, trainer, pl_module, batch_idx)

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: MultiTaskBatchInputs,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Sample the CPU as a test iteration ends."""
        self._record(TEST, BATCH_END, trainer, pl_module, batch_idx)
