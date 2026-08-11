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

import logging

from hydra.utils import instantiate
from omegaconf import DictConfig
import torch

from autoware_ml.models.multi_task_base_model import MultiTaskBaseModel
from autoware_ml.preprocessing.data_preprocessor import DataPreprocessor
from autoware_ml.utils.checkpoints import apply_matching_weights

logger = logging.getLogger(__name__)


def build_data_preprocessor(cfg: DictConfig) -> DataPreprocessor:
    """
    Build a data preprocessor from the Hydra configuration.

    Args:
        cfg: Hydra configuration.
    """
    logger.info("Building data preprocessor...")
    data_preprocessor = instantiate(cfg.data_preprocessor)
    logger.info(f"Data preprocessor built successfully with {data_preprocessor}.")
    return data_preprocessor


def build_model(
    cfg: DictConfig, data_preprocessor: DataPreprocessor, resume_checkpoint_path: str | None
) -> MultiTaskBaseModel:
    """
    Build a model from the Hydra configuration.

    Args:
        cfg: Hydra configuration.

    Returns:
        Pytorch-Lightning MultiTaskBaseModel for multi-task learning/inference.
    """
    logger.info("Building model...")
    model = instantiate(cfg.model, data_preprocessor=data_preprocessor)

    # Resume checkpoint
    weights_path = cfg.get("weights", None)
    if resume_checkpoint_path is not None and weights_path is not None:
        raise ValueError("'--resume-checkpoint' and '--weights' are mutually exclusive.")

    if weights_path is not None:
        apply_matching_weights(model, weights_path, map_location="cpu", logger=logger)

    if resume_checkpoint_path is not None:
        progress = torch.load(
            resume_checkpoint_path, map_location="cpu", weights_only=False, mmap=True
        )
        logger.info(
            "Resuming from '%s': checkpoint saved at epoch %d (global step %d), "
            "training continues at epoch %d.",
            resume_checkpoint_path,
            progress["epoch"],
            progress["global_step"],
            progress["epoch"] + 1,
        )
    logger.info(f"Model built successfully with {model}.")
    return model
