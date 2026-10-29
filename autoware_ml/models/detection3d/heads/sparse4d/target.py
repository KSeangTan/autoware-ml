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
# Adapted from projects/mmdet3d_plugin/models/detection3d/target.py and base_target.py of
# https://github.com/HorizonRobotics/Sparse4D (MIT License). The targets are returned as
# dataclasses, the boxes are read with ``Box3DFieldIndex`` and the regression weights follow
# the width of the predicted boxes.

"""Target assignment and denoising of the Sparse4D head.

:class:`SparseBox3DTarget` matches the anchors of a decoder layer to the ground truth boxes
with the Hungarian algorithm over a focal classification cost and a weighted L1 box cost, and
builds the noised copies of the ground truth boxes that train the decoder to recover them. Some of
the noised groups are cached and brought back in the next frame as temporal denoising groups,
matched again by instance id.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import NamedTuple

from jaxtyping import Bool, Float, Int64
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn.functional as F

from autoware_ml.dataclasses.models.detection3d.head_targets import (
    Sparse4DDenoisingTargets,
    Sparse4DMatchedTargets,
)
from autoware_ml.models.detection3d.heads.sparse4d.box3d import ANCHOR_DIMS
from autoware_ml.types.geometry import Box3DFieldIndex

# Class target of a negative noised anchor; the padding of a sample with fewer boxes is -1.
NEGATIVE_DN_CLASS = -3
# Cost of a pair the costs leave undefined, large enough to never be assigned.
_UNDEFINED_COST = 1e8


class TemporalDenoisingCache(NamedTuple):
    """Denoising groups kept for the next frame.

    Attributes:
        instance_feature: Features of the noised anchors after the last decoder.
        anchor: Their anchors.
        cls_target: Class every noised anchor was matched to.
        valid_mask: True for the noised anchors that come from a real box.
        id_target: Instance id every noised anchor was matched to, ``None`` without ids.
    """

    instance_feature: Float[torch.Tensor, "batch_size num_temp_groups num_noised embed_dims"]
    anchor: Float[torch.Tensor, "batch_size num_temp_groups num_noised box_dims"]
    cls_target: Int64[torch.Tensor, "batch_size num_temp_groups num_noised"]
    valid_mask: Bool[torch.Tensor, "batch_size num_temp_groups num_noised"]
    id_target: Int64[torch.Tensor, "batch_size num_temp_groups num_noised"] | None


class TemporalDenoisingUpdate(NamedTuple):
    """Instances and denoising targets after the cached groups replaced the current ones.

    Attributes:
        instance_feature: Learnable and noised instances, the noised ones last.
        anchor: Their anchors.
        box_target: Encoded box every noised anchor is matched to.
        cls_target: Class every noised anchor is matched to.
        valid_mask: True for the noised anchors that come from a real box.
        id_target: Instance id every noised anchor is matched to.
    """

    instance_feature: Float[torch.Tensor, "batch_size num_instances embed_dims"]
    anchor: Float[torch.Tensor, "batch_size num_instances box_dims"]
    box_target: Float[torch.Tensor, "batch_size num_noised box_dims"]
    cls_target: Int64[torch.Tensor, "batch_size num_noised"]
    valid_mask: Bool[torch.Tensor, "batch_size num_noised"]
    id_target: Int64[torch.Tensor, "batch_size num_noised"]


def encode_reg_target(
    boxes: Float[torch.Tensor, "num_boxes num_fields"], box_dims: int = ANCHOR_DIMS
) -> Float[torch.Tensor, "num_boxes box_dims"]:
    """Encode metric boxes into the anchor layout.

    Args:
        boxes: Metric boxes, see ``Box3DFieldIndex``; the fields after the yaw are the
            velocity components.
        box_dims: Number of leading anchor fields to keep.

    Returns:
        ``[x, y, z, log l, log w, log h, sin yaw, cos yaw, velocity...]`` cut to ``box_dims``.
    """
    yaw = boxes[..., Box3DFieldIndex.YAW]
    encoded = torch.cat(
        [
            boxes[..., Box3DFieldIndex.X : Box3DFieldIndex.Z + 1],
            boxes[..., Box3DFieldIndex.LENGTH : Box3DFieldIndex.HEIGHT + 1].log(),
            torch.sin(yaw).unsqueeze(-1),
            torch.cos(yaw).unsqueeze(-1),
            boxes[..., Box3DFieldIndex.YAW + 1 :],
        ],
        dim=-1,
    )
    return encoded[..., :box_dims]


def _hungarian(cost: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Assign rows to columns at the lowest total cost; undefined costs never match."""
    cost_array = cost.detach().cpu().numpy()
    cost_array = np.where(
        np.isneginf(cost_array) | np.isnan(cost_array), _UNDEFINED_COST, cost_array
    )
    row_indices, col_indices = linear_sum_assignment(cost_array)
    return (
        torch.as_tensor(row_indices, dtype=torch.int64, device=cost.device),
        torch.as_tensor(col_indices, dtype=torch.int64, device=cost.device),
    )


class SparseBox3DTarget:
    """Hungarian matching and denoising of the Sparse4D anchors."""

    def __init__(
        self,
        cls_weight: float = 2.0,
        alpha: float = 0.25,
        gamma: float = 2.0,
        eps: float = 1e-12,
        box_weight: float = 0.25,
        reg_weights: Sequence[float] | None = None,
        cls_wise_reg_weights: Mapping[int, Sequence[float]] | None = None,
        num_dn_groups: int = 0,
        dn_noise_scale: float | Sequence[float] = 0.5,
        max_dn_gt: int = 32,
        add_neg_dn: bool = True,
        num_temp_dn_groups: int = 0,
    ) -> None:
        """Initialize the target.

        Args:
            cls_weight: Weight of the classification cost.
            alpha: Focal balance of the classification cost.
            gamma: Focal exponent of the classification cost.
            eps: Added to the probabilities before their logarithm.
            box_weight: Weight of the box cost.
            reg_weights: Weight of every box field in the box cost, ``None`` for one on the
                center, the sizes and the yaw and zero on the velocity.
            cls_wise_reg_weights: Per-field weights replacing the ones of a box of the given
                class in the box cost and in the matched regression weights.
            num_dn_groups: Number of noised copies of the ground truth, ``0`` for none.
            dn_noise_scale: Half-width of the uniform noise added to every field of a box, one
                value for all fields or one per field.
            max_dn_gt: Number of boxes per sample the noised copies are made of, ``0`` for all.
            add_neg_dn: Whether every group also holds a copy noised by one to two scales,
                trained as a negative.
            num_temp_dn_groups: Number of groups cached for the next frame.
        """
        self.cls_weight = cls_weight
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.box_weight = box_weight
        self.reg_weights = None if reg_weights is None else list(reg_weights)
        self.cls_wise_reg_weights = (
            None
            if cls_wise_reg_weights is None
            else {cls: list(weights) for cls, weights in cls_wise_reg_weights.items()}
        )
        self.num_dn_groups = num_dn_groups
        self.dn_noise_scale = dn_noise_scale
        self.max_dn_gt = max_dn_gt
        self.add_neg_dn = add_neg_dn
        self.num_temp_dn_groups = num_temp_dn_groups
        self.dn_cache: TemporalDenoisingCache | None = None

    def _reg_weights(self, box_dims: int, like: torch.Tensor) -> torch.Tensor:
        """Weights of the box fields in the box cost."""
        if self.reg_weights is None:
            weights = [1.0] * min(box_dims, 8) + [0.0] * max(box_dims - 8, 0)
        else:
            weights = self.reg_weights
        if len(weights) != box_dims:
            raise ValueError(f"reg_weights must have {box_dims} entries, got {len(weights)}.")
        return like.new_tensor(weights)

    def _cls_cost(
        self,
        cls_pred: Float[torch.Tensor, "num_anchors num_classes"],
        cls_target: Int64[torch.Tensor, " num_boxes"],
    ) -> Float[torch.Tensor, "num_anchors num_boxes"]:
        """Focal cost of predicting the class of every box with every anchor."""
        prob = cls_pred.sigmoid()
        neg_cost = -(1 - prob + self.eps).log() * (1 - self.alpha) * prob.pow(self.gamma)
        pos_cost = -(prob + self.eps).log() * self.alpha * (1 - prob).pow(self.gamma)
        return (pos_cost[:, cls_target] - neg_cost[:, cls_target]) * self.cls_weight

    def _box_cost(
        self,
        box_pred: Float[torch.Tensor, "num_anchors box_dims"],
        box_target: Float[torch.Tensor, "num_boxes box_dims"],
        instance_reg_weights: Float[torch.Tensor, "num_boxes box_dims"],
    ) -> Float[torch.Tensor, "num_anchors num_boxes"]:
        """Weighted L1 distance of every anchor to every box."""
        reg_weights = self._reg_weights(box_pred.shape[-1], box_pred)
        # An undefined (NaN) field has a zero instance weight; keep it out of the sum.
        distance = (box_pred[:, None] - box_target[None]).abs().nan_to_num(0.0)
        return (distance * instance_reg_weights[None] * reg_weights).sum(dim=-1) * self.box_weight

    def _instance_reg_weights(
        self,
        box_target: Float[torch.Tensor, "num_boxes box_dims"],
        cls_target: Int64[torch.Tensor, " num_boxes"],
    ) -> Float[torch.Tensor, "num_boxes box_dims"]:
        """Per-box field weights: zero on undefined fields, class-wise weights where given."""
        weights = torch.logical_not(box_target.isnan()).to(dtype=box_target.dtype)
        if self.cls_wise_reg_weights is not None:
            for cls, cls_weights in self.cls_wise_reg_weights.items():
                weights = torch.where(
                    (cls_target == cls)[:, None], weights.new_tensor(cls_weights), weights
                )
        return weights

    def sample(
        self,
        cls_pred: Float[torch.Tensor, "batch_size num_anchors num_classes"],
        box_pred: Float[torch.Tensor, "batch_size num_anchors box_dims"],
        cls_target: Sequence[Int64[torch.Tensor, " num_boxes"]],
        box_target: Sequence[Float[torch.Tensor, "num_boxes num_fields"]],
    ) -> Sparse4DMatchedTargets:
        """Match the anchors of a decoder layer to the ground truth of every sample.

        Args:
            cls_pred: Class logits of every anchor.
            box_pred: Predicted boxes in the anchor layout.
            cls_target: Classes of the boxes of every sample.
            box_target: Metric boxes of every sample, see ``Box3DFieldIndex``.

        Returns:
            The class, the encoded box and the regression weights of every anchor; an
            unmatched anchor gets the class ``num_classes``, a zero box and zero weights, and
            an undefined (NaN) field of a matched box is zeroed with its weight.
        """
        batch_size, num_anchors, num_classes = cls_pred.shape
        box_dims = box_pred.shape[-1]

        output_cls_target = cls_pred.new_full(
            (batch_size, num_anchors), num_classes, dtype=torch.long
        )
        output_box_target = box_pred.new_zeros(box_pred.shape)
        output_reg_weights = box_pred.new_zeros(box_pred.shape)
        for index in range(batch_size):
            if len(cls_target[index]) == 0:
                continue
            sample_cls_target = cls_target[index].to(device=cls_pred.device, dtype=torch.long)
            sample_box_target = encode_reg_target(box_target[index], box_dims).to(
                device=box_pred.device, dtype=box_pred.dtype
            )
            instance_reg_weights = self._instance_reg_weights(sample_box_target, sample_cls_target)
            cost = self._cls_cost(cls_pred[index], sample_cls_target) + self._box_cost(
                box_pred[index], sample_box_target, instance_reg_weights
            )
            pred_idx, target_idx = _hungarian(cost)
            output_cls_target[index, pred_idx] = sample_cls_target[target_idx]
            output_box_target[index, pred_idx] = sample_box_target[target_idx].nan_to_num(0.0)
            output_reg_weights[index, pred_idx] = instance_reg_weights[target_idx]
        return Sparse4DMatchedTargets(
            cls_target=output_cls_target,
            box_target=output_box_target,
            reg_weights=output_reg_weights,
        )

    def get_dn_anchors(
        self,
        cls_target: Sequence[Int64[torch.Tensor, " num_boxes"]],
        box_target: Sequence[Float[torch.Tensor, "num_boxes num_fields"]],
        gt_instance_ids: Sequence[Int64[torch.Tensor, " num_boxes"]] | None = None,
        box_dims: int = ANCHOR_DIMS,
    ) -> Sparse4DDenoisingTargets | None:
        """Build the noised anchors of a training frame.

        The first ``max_dn_gt`` boxes of every sample are padded to a common count, encoded
        and replicated ``num_dn_groups`` times with uniform noise of ``dn_noise_scale``; with
        ``add_neg_dn`` every group also holds a copy noised by one to two scales. Every group
        is then matched to its boxes by box cost, so a noised copy that drifted closer to
        another box is trained on that one.

        Args:
            cls_target: Classes of the boxes of every sample.
            box_target: Metric boxes of every sample, see ``Box3DFieldIndex``.
            gt_instance_ids: Instance ids of the boxes of every sample, read only with
                temporal denoising groups.
            box_dims: Width of the noised anchors.

        Returns:
            The noised anchors and their targets, ``None`` without denoising groups or boxes.
        """
        if self.num_dn_groups <= 0:
            return None
        if self.num_temp_dn_groups <= 0:
            gt_instance_ids = None

        cls_list = list(cls_target)
        box_list = list(box_target)
        id_list = None if gt_instance_ids is None else list(gt_instance_ids)
        if self.max_dn_gt > 0:
            cls_list = [x[: self.max_dn_gt] for x in cls_list]
            box_list = [x[: self.max_dn_gt] for x in box_list]
            if id_list is not None:
                id_list = [x[: self.max_dn_gt] for x in id_list]

        max_dn_gt = max(len(x) for x in cls_list)
        if max_dn_gt == 0:
            return None
        device = box_list[0].device
        cls_padded = torch.stack(
            [F.pad(x.long(), (0, max_dn_gt - x.shape[0]), value=-1) for x in cls_list]
        ).to(device)
        box_padded = torch.stack(
            [
                F.pad(encode_reg_target(x, box_dims), (0, 0, 0, max_dn_gt - x.shape[0]))
                for x in box_list
            ]
        )
        box_padded = torch.where(
            cls_padded[..., None] == -1, box_padded.new_tensor(0.0), box_padded
        )
        id_padded = None
        if id_list is not None:
            id_padded = torch.stack(
                [F.pad(x.long(), (0, max_dn_gt - x.shape[0]), value=-1) for x in id_list]
            ).to(device)

        batch_size, num_gt, state_dims = box_padded.shape
        num_groups = self.num_dn_groups
        # Group g of sample b is row g * batch_size + b.
        cls_padded = cls_padded.tile(num_groups, 1)
        box_padded = box_padded.tile(num_groups, 1, 1)
        if id_padded is not None:
            id_padded = id_padded.tile(num_groups, 1)

        noise_scale = box_padded.new_tensor(self.dn_noise_scale)
        if noise_scale.ndim == 1 and noise_scale.numel() != state_dims:
            raise ValueError(
                f"dn_noise_scale must have one value or {state_dims} values, "
                f"got {noise_scale.numel()}."
            )
        noise = (torch.rand_like(box_padded) * 2 - 1) * noise_scale
        dn_anchor = box_padded + noise
        if self.add_neg_dn:
            noise_neg = torch.rand_like(box_padded) + 1
            sign = torch.where(
                torch.rand_like(box_padded) > 0.5,
                noise_neg.new_tensor(1.0),
                noise_neg.new_tensor(-1.0),
            )
            dn_anchor = torch.cat([dn_anchor, box_padded + noise_neg * sign * noise_scale], dim=1)
            num_gt = num_gt * 2

        dn_box_target = torch.zeros_like(dn_anchor)
        dn_cls_target = torch.full(
            dn_anchor.shape[:2], NEGATIVE_DN_CLASS, dtype=torch.long, device=device
        )
        dn_id_target = None if id_padded is None else torch.full_like(dn_cls_target, -1)
        for row in range(dn_anchor.shape[0]):
            cost = self._box_cost(dn_anchor[row], box_padded[row], torch.ones_like(box_padded[row]))
            anchor_idx, gt_idx = _hungarian(cost)
            dn_box_target[row, anchor_idx] = box_padded[row, gt_idx]
            dn_cls_target[row, anchor_idx] = cls_padded[row, gt_idx]
            if dn_id_target is not None:
                dn_id_target[row, anchor_idx] = id_padded[row, gt_idx]

        def groups_last(tensor: torch.Tensor) -> torch.Tensor:
            # (groups * batch, num_gt, ...) -> (batch, groups * num_gt, ...)
            grouped = tensor.reshape(num_groups, batch_size, num_gt, -1).transpose(0, 1)
            return grouped.reshape(batch_size, num_groups * num_gt, -1)

        dn_anchor = groups_last(dn_anchor)
        dn_box_target = groups_last(dn_box_target)
        dn_cls_target = groups_last(dn_cls_target).squeeze(-1)
        if dn_id_target is not None:
            dn_id_target = groups_last(dn_id_target).squeeze(-1)

        valid_mask = dn_cls_target >= 0
        if self.add_neg_dn:
            source_cls = groups_last(torch.cat([cls_padded, cls_padded], dim=1)).squeeze(-1)
            # A negative copy of a real box counts too; the copies of the padding do not.
            valid_mask = valid_mask | ((source_cls >= 0) & (dn_cls_target == NEGATIVE_DN_CLASS))

        num_noised = num_gt * num_groups
        attn_mask = torch.ones(num_noised, num_noised, dtype=torch.bool, device=device)
        for group in range(num_groups):
            start = num_gt * group
            attn_mask[start : start + num_gt, start : start + num_gt] = False
        return Sparse4DDenoisingTargets(
            anchors=dn_anchor,
            box_target=dn_box_target,
            cls_target=dn_cls_target,
            attn_mask=attn_mask,
            valid_mask=valid_mask,
            id_target=dn_id_target,
        )

    def update_dn(
        self,
        instance_feature: Float[torch.Tensor, "batch_size num_instances embed_dims"],
        anchor: Float[torch.Tensor, "batch_size num_instances box_dims"],
        dn_box_target: Float[torch.Tensor, "batch_size num_noised box_dims"],
        dn_cls_target: Int64[torch.Tensor, "batch_size num_noised"],
        valid_mask: Bool[torch.Tensor, "batch_size num_noised"],
        dn_id_target: Int64[torch.Tensor, "batch_size num_noised"],
        num_normal_anchor: int,
        temporal_valid_mask: Bool[torch.Tensor, " batch_size"] | None,
    ) -> TemporalDenoisingUpdate:
        """Replace the leading denoising groups with the groups cached from the previous frame.

        The cached groups keep their features and anchors (moved into the current ego frame
        by the instance bank) and get the targets of the current boxes that carry the same
        instance id; a cached anchor whose box left the frame becomes a negative. The lanes
        whose stream ended keep their current groups.

        Args:
            instance_feature: Learnable and noised instances, the noised ones last.
            anchor: Their anchors.
            dn_box_target: Encoded box of every noised anchor.
            dn_cls_target: Class of every noised anchor.
            valid_mask: True for the noised anchors that come from a real box.
            dn_id_target: Instance id of every noised anchor.
            num_normal_anchor: Number of learnable instances leading the set.
            temporal_valid_mask: True for every lane whose temporal instances carried over,
                ``None`` to drop the cache.

        Returns:
            The updated instances and denoising targets; the inputs when there is no cache.
        """
        batch_size, num_instances = instance_feature.shape[:2]
        if temporal_valid_mask is None:
            self.dn_cache = None
        if self.dn_cache is None or num_normal_anchor >= num_instances:
            return TemporalDenoisingUpdate(
                instance_feature, anchor, dn_box_target, dn_cls_target, valid_mask, dn_id_target
            )

        num_dn = num_instances - num_normal_anchor
        dn_instance_feature = instance_feature[:, num_normal_anchor:]
        dn_anchor = anchor[:, num_normal_anchor:]
        instance_feature = instance_feature[:, :num_normal_anchor]
        anchor = anchor[:, :num_normal_anchor]

        num_groups = self.num_dn_groups
        num_dn = num_dn // num_groups
        current = [
            dn_instance_feature.reshape(batch_size, num_groups, num_dn, -1),
            dn_anchor.reshape(batch_size, num_groups, num_dn, -1),
            dn_box_target.reshape(batch_size, num_groups, num_dn, -1),
            dn_cls_target.reshape(batch_size, num_groups, num_dn),
            valid_mask.reshape(batch_size, num_groups, num_dn),
            dn_id_target.reshape(batch_size, num_groups, num_dn),
        ]
        dn_id = current[-1]

        cache = self.dn_cache
        num_temp_groups, num_temp_dn = cache.instance_feature.shape[1:3]
        # (batch, temp groups, temp noised, current noised)
        match = cache.id_target[..., None] == dn_id[:, :num_temp_groups, None]
        temp_box_target = (match[..., None] * current[2][:, :num_temp_groups, None]).sum(dim=3)
        temp_cls_target = torch.where(
            torch.all(torch.logical_not(match), dim=-1),
            cache.cls_target.new_tensor(-1),
            cache.cls_target,
        )
        cached = [
            cache.instance_feature,
            cache.anchor,
            temp_box_target,
            temp_cls_target,
            cache.valid_mask,
            cache.id_target,
        ]

        output = []
        for temp_meta, meta in zip(cached, current):
            # The box count may have changed since the cache was taken.
            if num_temp_dn < num_dn:
                pad = (0, num_dn - num_temp_dn)
                if temp_meta.dim() == 4:
                    pad = (0, 0, 0, num_dn - num_temp_dn)
                temp_meta = F.pad(temp_meta, pad, value=0)
            else:
                temp_meta = temp_meta[:, :, :num_dn]
            lane_mask = temporal_valid_mask[:, None, None]
            if meta.dim() == 4:
                lane_mask = lane_mask.unsqueeze(-1)
            temp_meta = torch.where(lane_mask, temp_meta, meta[:, :num_temp_groups])
            meta = torch.cat([temp_meta, meta[:, num_temp_groups:]], dim=1)
            output.append(meta.flatten(1, 2))
        return TemporalDenoisingUpdate(
            instance_feature=torch.cat([instance_feature, output[0]], dim=1),
            anchor=torch.cat([anchor, output[1]], dim=1),
            box_target=output[2],
            cls_target=output[3],
            valid_mask=output[4],
            id_target=output[5],
        )

    def cache_dn(
        self,
        dn_instance_feature: Float[torch.Tensor, "batch_size num_noised embed_dims"],
        dn_anchor: Float[torch.Tensor, "batch_size num_noised box_dims"],
        dn_cls_target: Int64[torch.Tensor, "batch_size num_noised"],
        valid_mask: Bool[torch.Tensor, "batch_size num_noised"],
        dn_id_target: Int64[torch.Tensor, "batch_size num_noised"] | None,
    ) -> None:
        """Keep ``num_temp_dn_groups`` random denoising groups for the next frame.

        Args:
            dn_instance_feature: Features of the noised anchors after the last decoder.
            dn_anchor: Their anchors.
            dn_cls_target: Class of every noised anchor.
            valid_mask: True for the noised anchors that come from a real box.
            dn_id_target: Instance id of every noised anchor, ``None`` without ids.
        """
        if self.num_temp_dn_groups < 0:
            return
        num_groups = self.num_dn_groups
        batch_size, num_dn = dn_instance_feature.shape[:2]
        num_temp_dn = num_dn // num_groups
        group_mask = (torch.randperm(num_groups) < self.num_temp_dn_groups).to(dn_anchor.device)
        self.dn_cache = TemporalDenoisingCache(
            instance_feature=dn_instance_feature.detach().reshape(
                batch_size, num_groups, num_temp_dn, -1
            )[:, group_mask],
            anchor=dn_anchor.detach().reshape(batch_size, num_groups, num_temp_dn, -1)[
                :, group_mask
            ],
            cls_target=dn_cls_target.reshape(batch_size, num_groups, num_temp_dn)[:, group_mask],
            valid_mask=valid_mask.reshape(batch_size, num_groups, num_temp_dn)[:, group_mask],
            id_target=None
            if dn_id_target is None
            else dn_id_target.reshape(batch_size, num_groups, num_temp_dn)[:, group_mask],
        )
