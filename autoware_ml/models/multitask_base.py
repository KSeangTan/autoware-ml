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

"""Base model classes for Autoware-ML.

This module defines shared Lightning model interfaces and helper abstractions
used by task-specific model wrappers throughout the framework.
"""

from __future__ import annotations

import inspect
from abc import abstractmethod
from collections.abc import Callable, Mapping, Sequence
from typing import Any, final

import lightning as L
from lightning.pytorch.utilities.data import extract_batch_size
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from autoware_ml.metrics.base import MetricSuite
from autoware_ml.metrics.eval_mixin import MetricEvalMixin
from autoware_ml.preprocessing.base import DataPreprocessing
from autoware_ml.utils.optimizer import build_lightning_optimizer_config
from autoware_ml.datamodule.multi_task.dataclasses.multi_task_samples import (
    MultiTaskGTSample,
)


class MultiTaskBaseModel(MetricEvalMixin, L.LightningModule):
    """Base Lightning Module for all Autoware-ML models.

    Provides common functionality for training, validation, and testing with
    built-in support for flexible optimizer and scheduler configuration.
    All parameters are explicitly typed for IDE support and type checking.
    """

    def __init__(
        self,
        data_preprocessing: DataPreprocessing,
        optimizer: Callable[..., Optimizer] | None = None,
        scheduler: Callable[[Optimizer], LRScheduler] | None = None,
        optimizer_group_overrides: Mapping[str, Mapping[str, Any]] | None = None,
        scheduler_config: Mapping[str, Any] | None = None,
        metrics: Sequence[MetricSuite] | None = None,
    ):
        """Initialize base model.

        Args:
            optimizer: Callable that returns an optimizer when given model parameters.
            scheduler: Callable that returns a scheduler when given the optimizer.
            optimizer_group_overrides: Optional optimizer overrides keyed by
                model-defined optimizer group name.
            scheduler_config: Optional Lightning scheduler metadata such as
                ``interval`` or ``monitor``.
            metrics: Task metrics accumulated during validation and test. Empty
                or ``None`` logs only losses.
        """
        super().__init__(metrics=metrics)
        self._data_preprocessing = data_preprocessing
        self.forward_signature = inspect.signature(self.forward)
        self.optimizer_partial = optimizer
        self.scheduler_partial = scheduler
        self.optimizer_group_overrides = (
            dict(optimizer_group_overrides) if optimizer_group_overrides else None
        )
        self.scheduler_config = dict(scheduler_config) if scheduler_config else {}
        self._data_preprocessing = DataPreprocessing()

    def on_after_batch_transfer(
        self, batch: MultiTaskGTSample, dataloader_idx: int
    ) -> MultiTaskGTSample:
        """Apply runtime preprocessing after Lightning moves a batch to device.

        Args:
            batch: Collated batch of type :class:`MultiTaskGTSample` on the target device.
            dataloader_idx: Lightning dataloader index.

        Returns:
            Batch of type :class:`MultiTaskGTSample` after runtime preprocessing.
        """
        return self._data_preprocessing(batch)

    def predict_outputs(self, batch_inputs_dict: Mapping[str, Any], outputs: Any) -> Any:
        """Convert raw model outputs into task-level predictions.

        The default implementation returns the model outputs unchanged. Task
        wrappers should override this when prediction-time outputs differ from
        training-time outputs, for example to convert logits into probabilities
        and labels.

        Args:
            batch_inputs_dict: Full batch dictionary after runtime preprocessing.
            outputs: Raw outputs returned by :meth:`forward`.

        Returns:
            Task-level predictions.
        """
        del batch_inputs_dict
        return outputs

    @torch.no_grad()
    def predict(self, *args: Any, **kwargs: Any) -> Any:
        """Run inference and return task-level predictions.

        Args:
            *args: Positional arguments forwarded to :meth:`forward`.
            **kwargs: Keyword arguments forwarded to :meth:`forward`.

        Returns:
            Task-level predictions produced by :meth:`predict_outputs`.
        """
        return self.predict_outputs(kwargs, self(*args, **kwargs))

    def get_export_output_names(self) -> list[str] | None:
        """Return output names used by the generic export wrapper.

        Models that export structured prediction dictionaries should override
        this hook and return the tensor names in the exported output order.

        Returns:
            Export output names or ``None`` when the generic wrapper should keep
            the model outputs unnamed.
        """
        return None

    def prepare_export_outputs(self, predictions: Any) -> Any:
        """Convert prediction outputs into an ONNX-exportable structure.

        Args:
            predictions: Task-level predictions produced by
                :meth:`predict_outputs`.

        Returns:
            Tensor, tuple of tensors, or another ONNX-exportable structure.

        Raises:
            ValueError: Raised when prediction outputs are a mapping but the
                model does not define explicit export output names.
        """
        if isinstance(predictions, Mapping):
            output_names = self.get_export_output_names()
            if output_names is None:
                raise ValueError(
                    "Structured prediction outputs require explicit export output names."
                )
            return tuple(predictions[name] for name in output_names)
        return predictions

    def build_optimizer_groups(self) -> Mapping[str, Sequence[torch.nn.Parameter]]:
        """Return structural optimizer groups for the model.

        Models that do not need custom grouping use a single ``default`` group.
        Models with optimizer-group-specific tuning can override this hook.

        Returns:
            Mapping from optimizer group names to parameter sequences.
        """
        return {
            "default": [parameter for parameter in self.parameters() if parameter.requires_grad]
        }

    @abstractmethod
    def forward(self, **kwargs: Any) -> Any:
        """Forward pass of the model.

        Subclasses can define this method with any signature. The base class
        automatically filters batch inputs to match the method signature using
        signature inspection.

        Args:
            **kwargs: Keyword arguments (subclass-specific).

        Returns:
            Model outputs.
        """
        pass

    @abstractmethod
    def compute_metrics(
        self, batch_inputs_dict: Mapping[str, Any], outputs: Any
    ) -> dict[str, torch.Tensor]:
        """Compute metrics.

        Args:
            batch_inputs_dict: Full batch dictionary after runtime preprocessing.
            outputs: Model outputs from forward().

        Returns:
            Dictionary of metric tensors. A ``"loss"`` key is required.
        """
        pass

    def get_log_batch_size(self, batch_inputs_dict: Mapping[str, Any]) -> int | None:
        """Infer the effective sample batch size for logging.

        The default implementation tries Lightning's recursive batch-size
        inference on the actual model inputs. Models with ragged point-cloud
        batches should override this hook to provide an explicit sample count.

        Args:
            batch_inputs_dict: Full batch dictionary from the dataloader.

        Returns:
            Sample batch size when it can be inferred, otherwise ``None``.
        """
        forward_inputs = {
            key: batch_inputs_dict[key]
            for key in self.forward_signature.parameters
            if key in batch_inputs_dict
        }
        return extract_batch_size(forward_inputs)

    def _shared_step(
        self, batch_inputs_dict: Mapping[str, Any], step_prefix: str, **kwargs: Any
    ) -> tuple[dict[str, torch.Tensor], Any]:
        """Run one forward pass, compute metrics, and log them.

        Args:
            batch_inputs_dict: Dictionary with input data.
            step_prefix: Prefix for logging (train, val, test).
            **kwargs: Keyword arguments forwarded to ``self.log_dict``.

        Returns:
            Tuple of the metric dictionary and the raw model outputs.
            The metric dictionary contains at least a ``"loss"`` key.
        """
        forward_inputs = {
            key: batch_inputs_dict[key]
            for key in self.forward_signature.parameters
            if key in batch_inputs_dict
        }
        outputs = self(**forward_inputs)
        metrics = self.compute_metrics(batch_inputs_dict, outputs)
        if "loss" not in metrics:
            raise ValueError("compute_metrics() must return a dict containing a 'loss' key.")
        batch_size = self.get_log_batch_size(batch_inputs_dict)
        self.log_dict(
            {f"{step_prefix}/{k}": v for k, v in metrics.items()},
            batch_size=batch_size,
            **kwargs,
        )
        return metrics, outputs

    @final
    def training_step(self, batch_inputs_dict: Mapping[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step.

        Args:
            batch_inputs_dict: Dictionary with input data.
            batch_idx: Batch index.

        Returns:
            Total loss tensor required by Lightning for backpropagation.
        """
        metrics, _ = self._shared_step(
            batch_inputs_dict,
            "train",
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return metrics["loss"]

    @final
    def validation_step(
        self, batch_inputs_dict: Mapping[str, Any], batch_idx: int
    ) -> dict[str, Any]:
        """Validation step.

        Args:
            batch_inputs_dict: Dictionary with input data.
            batch_idx: Batch index.

        Returns:
            Dictionary with at least a ``"loss"`` key and a ``"model_outputs"``
            key containing the raw forward outputs. The raw outputs are available
            to ``on_validation_batch_end`` for epoch-level metric accumulation.
        """
        metrics, outputs = self._shared_step(
            batch_inputs_dict,
            "val",
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return {**metrics, "model_outputs": outputs}

    @final
    def test_step(self, batch_inputs_dict: Mapping[str, Any], batch_idx: int) -> dict[str, Any]:
        """Test step.

        Args:
            batch_inputs_dict: Dictionary with input data.
            batch_idx: Batch index.

        Returns:
            Dictionary with at least a ``"loss"`` key and a ``"model_outputs"``
            key containing the raw forward outputs.
        """
        metrics, outputs = self._shared_step(
            batch_inputs_dict,
            "test",
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return {**metrics, "model_outputs": outputs}

    @final
    def predict_step(self, batch_inputs_dict: Mapping[str, Any], batch_idx: int) -> Any:
        """Prediction step.

        Args:
            batch_inputs_dict: Dictionary with input data.
            batch_idx: Batch index.

        Returns:
            Predictions.
        """
        del batch_idx
        forward_inputs = {
            key: batch_inputs_dict[key]
            for key in self.forward_signature.parameters
            if key in batch_inputs_dict
        }
        outputs = self(**forward_inputs)
        return self.predict_outputs(batch_inputs_dict, outputs)

    def configure_optimizers(self) -> Optimizer | dict[str, Any]:
        """Configure optimizers and schedulers.

        Scheduler behavior such as ``interval``, ``frequency``, and ``monitor``
        is configured explicitly through ``scheduler_config``. The framework
        only auto-fills ``total_steps`` when the configured scheduler declares
        that argument and it was not already bound in the scheduler factory.

        Returns:
            Optimizer instance or Lightning optimizer configuration dictionary.
        """
        if self.optimizer_partial is None:
            raise ValueError("Optimizer must be provided.")
        return build_lightning_optimizer_config(
            self,
            self.optimizer_partial,
            self.scheduler_partial,
            optimizer_group_overrides=self.optimizer_group_overrides,
            scheduler_config=self.scheduler_config,
            estimated_stepping_batches=self.trainer.estimated_stepping_batches
            if self._trainer is not None
            else None,
        )
