# Copyright (c) 2024 Horizon Robotics. MIT License.
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
#
# Adapted from projects/mmdet3d_plugin/models/detection3d/losses.py of
# https://github.com/HorizonRobotics/Sparse4D (MIT License). The mmdet losses it built from
# configs (L1Loss, sigmoid CrossEntropyLoss, GaussianFocalLoss) are written out here with
# their ``avg_factor`` normalization.

"""Box regression loss of the Sparse4D head, with the centerness and yawness quality losses."""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

from jaxtyping import Float, Int64
import torch
import torch.nn as nn
import torch.nn.functional as F

from autoware_ml.models.detection3d.heads.sparse4d.box3d import CENTER, YAW, QualityIndex

# Focal exponents of the yawness loss (mmdet GaussianFocalLoss defaults).
_YAWNESS_ALPHA = 2.0
_YAWNESS_GAMMA = 4.0
_EPS = 1e-12


class SparseBox3DLosses(NamedTuple):
    """Losses of the matched boxes.

    Attributes:
        loss_box: Weighted L1 loss of the box fields.
        loss_centerness: Loss of the centerness estimate, ``None`` without quality.
        loss_yawness: Loss of the yawness estimate, ``None`` without quality.
    """

    loss_box: torch.Tensor
    loss_centerness: torch.Tensor | None
    loss_yawness: torch.Tensor | None


class SparseBox3DLoss(nn.Module):
    """Box regression loss of the Sparse4D head.

    The box loss is an L1 over the encoded box fields. With a quality estimate, the
    centerness logit is trained with a sigmoid cross-entropy towards ``exp(-distance)`` of
    the predicted to the target center, and the yawness logit with a focal loss towards
    whether the predicted heading points the way of the target.
    """

    def __init__(
        self,
        box_weight: float = 0.25,
        centerness_weight: float = 1.0,
        yawness_weight: float = 1.0,
        cls_allow_reverse: Sequence[int] | None = None,
    ) -> None:
        """Initialize the loss.

        Args:
            box_weight: Weight of the box loss.
            centerness_weight: Weight of the centerness loss.
            yawness_weight: Weight of the yawness loss.
            cls_allow_reverse: Classes whose heading has no front, such as barriers; the
                target heading of such a box is flipped when the prediction points the other
                way.
        """
        super().__init__()
        self.box_weight = box_weight
        self.centerness_weight = centerness_weight
        self.yawness_weight = yawness_weight
        self.cls_allow_reverse = None if cls_allow_reverse is None else list(cls_allow_reverse)

    def forward(
        self,
        box: Float[torch.Tensor, "num_matched box_dims"],
        box_target: Float[torch.Tensor, "num_matched box_dims"],
        weight: Float[torch.Tensor, "num_matched box_dims"] | None = None,
        avg_factor: float | torch.Tensor | None = None,
        quality: Float[torch.Tensor, "num_matched 2"] | None = None,
        cls_target: Int64[torch.Tensor, " num_matched"] | None = None,
    ) -> SparseBox3DLosses:
        """Compute the losses of the matched boxes.

        Args:
            box: Predicted boxes in the anchor layout.
            box_target: Their encoded targets.
            weight: Per-box, per-field weights of the box loss, ``None`` for one.
            avg_factor: Number the summed losses are divided by, ``None`` for the number of
                boxes.
            quality: Centerness and yawness logits, see ``QualityIndex``; ``None`` to skip
                their losses.
            cls_target: Class of every box, needed by ``cls_allow_reverse``.

        Returns:
            The box loss and, with quality, the centerness and yawness losses.
        """
        if avg_factor is None:
            avg_factor = max(box.shape[0], 1)
        avg_factor = torch.as_tensor(avg_factor, dtype=box.dtype, device=box.device)

        heading = box[..., YAW]
        heading_target = box_target[..., YAW]
        # Some classes do not tell their front from their back, for example barriers.
        if self.cls_allow_reverse is not None and cls_target is not None:
            reverse = F.cosine_similarity(heading_target, heading, dim=-1) < 0
            reverse = reverse & torch.isin(
                cls_target, cls_target.new_tensor(self.cls_allow_reverse)
            )
            heading_target = torch.where(reverse[..., None], -heading_target, heading_target)
            box_target = box_target.clone()
            box_target[..., YAW] = heading_target

        box_loss = (box - box_target).abs()
        if weight is not None:
            box_loss = box_loss * weight
        loss_box = box_loss.sum() / avg_factor * self.box_weight
        if quality is None:
            return SparseBox3DLosses(loss_box, None, None)

        centerness = quality[..., QualityIndex.CENTERNESS]
        centerness_target = torch.exp(
            -torch.norm(box_target[..., CENTER] - box[..., CENTER], p=2, dim=-1)
        )
        loss_centerness = (
            F.binary_cross_entropy_with_logits(
                centerness, centerness_target, reduction="none"
            ).sum()
            / avg_factor
            * self.centerness_weight
        )

        yawness = quality[..., QualityIndex.YAWNESS].sigmoid()
        yawness_target = (F.cosine_similarity(heading_target, heading, dim=-1) > 0).to(box.dtype)
        positive = yawness_target.eq(1).to(box.dtype)
        negative_weights = (1 - yawness_target).pow(_YAWNESS_GAMMA)
        pos_loss = -(yawness + _EPS).log() * (1 - yawness).pow(_YAWNESS_ALPHA) * positive
        neg_loss = -(1 - yawness + _EPS).log() * yawness.pow(_YAWNESS_ALPHA) * negative_weights
        loss_yawness = (pos_loss + neg_loss).sum() / avg_factor * self.yawness_weight
        return SparseBox3DLosses(loss_box, loss_centerness, loss_yawness)
