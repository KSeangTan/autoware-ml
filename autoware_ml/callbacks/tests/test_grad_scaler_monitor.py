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

"""Unit tests for the loss scale callback: the scale of the mixed precision scaler, the
precisions that train without a scaler, the step-metric logging options, and a fit that logs
the scale through the trainer."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

import lightning as L
from lightning.pytorch.plugins.precision import MixedPrecision, Precision
import torch
from torch.utils.data import DataLoader, TensorDataset

from autoware_ml.callbacks.grad_scaler_monitor import GradScalerMonitor


def _trainer(precision_plugin: Precision) -> MagicMock:
    """Build a stand-in Trainer holding ``precision_plugin``."""
    trainer = MagicMock()
    trainer.precision_plugin = precision_plugin
    return trainer


def _run(precision_plugin: Precision) -> MagicMock:
    """Run the batch-end hook under ``precision_plugin`` and return the recording module."""
    module = MagicMock()
    GradScalerMonitor().on_train_batch_end(_trainer(precision_plugin), module, None, None, 0)
    return module


class TestLoggedScale(unittest.TestCase):
    """The scale is logged only when the precision trains with a scaler."""

    def test_logs_the_scale_of_the_scaler(self) -> None:
        """A 16-mixed plugin logs the scale its scaler holds."""
        scaler = torch.amp.GradScaler(device="cpu", init_scale=1024.0)
        module = _run(MixedPrecision("16-mixed", "cpu", scaler=scaler))

        module.log.assert_called_once()
        self.assertEqual(module.log.call_args.args, ("train/loss_scale", 1024.0))

    def test_logs_as_a_step_metric_with_unit_batch_size(self) -> None:
        """The scale is logged per step, not per epoch, weighted by one."""
        scaler = torch.amp.GradScaler(device="cpu")
        kwargs = _run(MixedPrecision("16-mixed", "cpu", scaler=scaler)).log.call_args.kwargs

        self.assertTrue(kwargs["on_step"])
        self.assertFalse(kwargs["on_epoch"])
        self.assertFalse(kwargs["prog_bar"])
        self.assertEqual(kwargs["batch_size"], 1)

    def test_bf16_mixed_logs_nothing(self) -> None:
        """bf16-mixed trains without a scaler, so there is no scale to log."""
        _run(MixedPrecision("bf16-mixed", "cpu")).log.assert_not_called()

    def test_full_precision_logs_nothing(self) -> None:
        """The plain precision plugin has no scaler attribute."""
        _run(Precision()).log.assert_not_called()

    def test_disabled_scaler_logs_nothing(self) -> None:
        """A disabled scaler never scales the loss, its constant 1.0 is not logged."""
        scaler = torch.amp.GradScaler(device="cpu", enabled=False)
        _run(MixedPrecision("16-mixed", "cpu", scaler=scaler)).log.assert_not_called()


class _Module(L.LightningModule):
    """One linear layer trained on a squared output."""

    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(2, 1)

    def training_step(self, batch: list[torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self.layer(batch[0]).pow(2).mean()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=0.01)


class TestFit(unittest.TestCase):
    """The callback logs through a real trainer."""

    def test_fit_logs_the_scale_after_the_scaler_update(self) -> None:
        """Two finite steps with a growth interval of one double the scale twice."""
        scaler = torch.amp.GradScaler(device="cpu", init_scale=8.0, growth_interval=1)
        trainer = L.Trainer(
            accelerator="cpu",
            devices=1,
            max_steps=2,
            plugins=[MixedPrecision("16-mixed", "cpu", scaler=scaler)],
            callbacks=[GradScalerMonitor()],
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        trainer.fit(_Module(), DataLoader(TensorDataset(torch.ones(4, 2)), batch_size=2))

        self.assertEqual(trainer.callback_metrics["train/loss_scale"].item(), 32.0)


if __name__ == "__main__":
    unittest.main()
