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

"""Loss scale logging for mixed precision training."""

from __future__ import annotations

from typing import Any

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
import torch


class GradScalerMonitor(Callback):
    """Log the loss scale of the AMP gradient scaler as ``train/loss_scale``.

    Under ``precision: 16-mixed`` the loss is multiplied by the scale before the backward pass,
    so the half precision gradients do not underflow. The scaler halves the scale and skips the
    optimizer step whenever a gradient is not finite, and doubles it after ``growth_interval``
    consecutive steps with finite gradients. The curve therefore shows the skipped steps: every
    drop is at least one optimizer step that did not happen, and a scale that keeps falling is a
    run whose gradients overflow in half precision.

    The scale is read after the scaler update of the batch, so the logged value is the scale the
    next batch trains with. Precisions without a scaler log nothing: ``bf16-mixed`` keeps the
    exponent range of single precision and needs no loss scaling, and neither do the full
    precisions.

    The metric is a step metric: it is logged on every training batch, and the trainer's
    ``log_every_n_steps`` decides which steps reach the logger.
    """

    LOG_KEY = "train/loss_scale"

    @staticmethod
    def _scaler(trainer: Trainer) -> torch.amp.GradScaler | None:
        """Return the gradient scaler of the trainer's precision plugin.

        Args:
            trainer: Trainer running the fit.

        Returns:
            The scaler, or ``None`` when the precision trains without one.
        """
        scaler = getattr(trainer.precision_plugin, "scaler", None)
        return scaler if isinstance(scaler, torch.amp.GradScaler) else None

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Log the scale the scaler holds after the update of this batch."""
        scaler = self._scaler(trainer)
        if scaler is None or not scaler.is_enabled():
            return
        pl_module.log(
            self.LOG_KEY,
            scaler.get_scale(),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            batch_size=1,
        )
