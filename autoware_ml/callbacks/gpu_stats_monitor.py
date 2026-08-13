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

"""Per-rank GPU utilization and memory, read through NVML and gathered to rank zero."""

from __future__ import annotations

import logging
from typing import Any, NamedTuple

import pynvml
import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback

from autoware_ml.dataclasses.multi_task_batch_inputs import MultiTaskBatchInputs

logger = logging.getLogger(__name__)

TRAIN = "train"
VAL = "val"
TEST = "test"

BATCH_START = "batch_start"
BATCH_END = "batch_end"

_BYTES_PER_GB = 1024**3


class GPUSample(NamedTuple):
    """One device's utilization and memory footprint.

    Attributes:
        utilization_percent: Share of the last sampling period during which one
            or more kernels were resident on the device.
        memory_gb: Device memory in use, in GiB, matching what ``nvidia-smi``
            reports.
    """

    utilization_percent: float
    memory_gb: float


class GPUStatsMonitor(Callback):
    """Record GPU utilization and memory for every rank, not only rank zero.

    Each rank reads its own device through NVML. Because Lightning writes
    metrics from rank zero alone, the readings are gathered across ranks and
    then logged there under one metric per rank:

    - ``{stage}/{hook}/gpu_util_rank{n}`` -- utilization percentage of rank ``n``'s device.
    - ``{stage}/{hook}/gpu_memory_gb_rank{n}`` -- memory in use on rank ``n``'s device, in GiB.

    ``{hook}`` is ``batch_start`` or ``batch_end``. Reading at both ends of an
    iteration separates the device's state while the dataloader is feeding it
    from its state right after the forward and backward pass.

    Comparing ranks is the point: a rank that consistently reports lower
    utilization than its peers is the straggler holding up every collective.

    NVML utilization is a coarse measure: the driver reports the fraction of a
    sampling period (roughly one second) in which any kernel was running, not
    how much of the device those kernels used. A short iteration can therefore
    read 100% while leaving most of the GPU idle.

    Devices are resolved by CUDA UUID rather than by index, so the reading stays
    correct when ``CUDA_VISIBLE_DEVICES`` reorders or restricts the devices
    visible to the process.

    The gather is a collective: every rank must reach it the same number of
    times, which is why the interval is driven by ``batch_idx`` rather than by
    anything rank-dependent. Reading the gathered values on the host also forces
    a CUDA synchronization, so a wider interval costs less throughput.

    Args:
        log_interval: Sample, gather and log every this many batches.
    """

    def __init__(self, log_interval: int = 1) -> None:
        if log_interval < 1:
            raise ValueError(f"log_interval must be positive, got {log_interval}.")
        self.log_interval = log_interval
        self._handle: Any = None
        self._nvml_initialized = False

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Resolve this rank's NVML handle from the device it trains on.

        Args:
            trainer: Active trainer.
            pl_module: Module being run.
            stage: Lightning stage name.
        """
        if self._handle is not None:
            return
        device = trainer.strategy.root_device
        if device.type != "cuda":
            logger.info("GPUStatsMonitor is inactive: the strategy runs on %s.", device.type)
            return
        pynvml.nvmlInit()
        self._nvml_initialized = True
        index = device.index if device.index is not None else torch.cuda.current_device()
        uuid = getattr(torch.cuda.get_device_properties(index), "uuid", None)
        if uuid is None:
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            return
        try:
            self._handle = pynvml.nvmlDeviceGetHandleByUUID(f"GPU-{uuid}".encode())
        except pynvml.NVMLError:
            # Older drivers reject the UUID lookup; the index is right whenever
            # CUDA_VISIBLE_DEVICES has not reordered the devices.
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(index)

    def teardown(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Release NVML.

        Args:
            trainer: Active trainer.
            pl_module: Module being run.
            stage: Lightning stage name.
        """
        if self._nvml_initialized:
            pynvml.nvmlShutdown()
            self._nvml_initialized = False
            self._handle = None

    def _read_device(self) -> GPUSample:
        """Read this rank's device.

        Returns:
            The device's utilization and memory use.
        """
        utilization = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
        memory = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
        return GPUSample(
            utilization_percent=float(utilization.gpu),
            memory_gb=memory.used / _BYTES_PER_GB,
        )

    def _gather(self, pl_module: LightningModule, sample: GPUSample) -> torch.Tensor:
        """Collect every rank's reading on every rank.

        Args:
            pl_module: Module used for the collective.
            sample: This rank's reading.

        Returns:
            Tensor of shape ``(world_size, 2)`` holding utilization and memory
            per rank.
        """
        values = torch.tensor(
            [sample.utilization_percent, sample.memory_gb],
            dtype=torch.float32,
            device=pl_module.device,
        )
        world_size = pl_module.trainer.world_size
        if world_size <= 1:
            return values.unsqueeze(0)
        return pl_module.all_gather(values).reshape(world_size, 2)

    def _record(
        self,
        stage: str,
        hook: str,
        trainer: Trainer,
        pl_module: LightningModule,
        batch_idx: int,
    ) -> None:
        """Sample every rank's device and log the readings from rank zero.

        Args:
            stage: Loop stage being measured.
            hook: Either ``batch_start`` or ``batch_end``.
            trainer: Active trainer.
            pl_module: Module used to log metrics.
            batch_idx: Index of the current batch, identical on every rank, so
                the gather below stays collective-safe.
        """
        if trainer.sanity_checking or self._handle is None:
            return
        if batch_idx % self.log_interval != 0:
            return

        gathered = self._gather(pl_module, self._read_device())
        utilizations = gathered[:, 0].tolist()
        memories = gathered[:, 1].tolist()

        if not trainer.is_global_zero:
            return
        for rank, (utilization, memory) in enumerate(zip(utilizations, memories)):
            self._log(pl_module, f"{stage}/{hook}/gpu_util_rank{rank}", utilization)
            self._log(pl_module, f"{stage}/{hook}/gpu_memory_gb_rank{rank}", memory)

    @staticmethod
    def _log(pl_module: LightningModule, name: str, value: float) -> None:
        """Log one rank's GPU metric from rank zero.

        ``rank_zero_only`` marks the call as non-collective: the metric belongs
        to one rank and must not be reduced across them, and the values have
        already been gathered.

        Args:
            pl_module: Module used to log metrics.
            name: Metric name.
            value: Metric value, a percentage or GiB.
        """
        pl_module.log(
            name,
            value,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            batch_size=1,
            rank_zero_only=True,
        )

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: MultiTaskBatchInputs,
        batch_idx: int,
    ) -> None:
        """Sample every rank's device as a training iteration starts."""
        self._record(TRAIN, BATCH_START, trainer, pl_module, batch_idx)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: MultiTaskBatchInputs,
        batch_idx: int,
    ) -> None:
        """Sample every rank's device after a training iteration."""
        self._record(TRAIN, BATCH_END, trainer, pl_module, batch_idx)

    def on_validation_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: MultiTaskBatchInputs,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Sample every rank's device as a validation iteration starts."""
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
        """Sample every rank's device after a validation iteration."""
        self._record(VAL, BATCH_END, trainer, pl_module, batch_idx)

    def on_test_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: MultiTaskBatchInputs,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Sample every rank's device as a test iteration starts."""
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
        """Sample every rank's device after a test iteration."""
        self._record(TEST, BATCH_END, trainer, pl_module, batch_idx)
