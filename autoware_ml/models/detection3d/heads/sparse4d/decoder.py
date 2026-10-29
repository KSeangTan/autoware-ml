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
# Adapted from projects/mmdet3d_plugin/models/detection3d/decoder.py of
# https://github.com/HorizonRobotics/Sparse4D (MIT License). The decoder emits
# ``Detection3DSamplePredictions`` with ``(l, w, h)`` sizes and a planar velocity, and keeps
# the boxes on their device.

"""Decoding of the Sparse4D anchors into boxes."""

from __future__ import annotations

from jaxtyping import Float, Int64
import torch

from autoware_ml.dataclasses.models.detection3d.predictions import Detection3DSamplePredictions
from autoware_ml.models.detection3d.heads.sparse4d.box3d import (
    CENTER,
    SIZE,
    AnchorFieldIndex,
    QualityIndex,
)

# The decoded boxes carry a planar velocity: (x, y, z, l, w, h, yaw, vx, vy).
_NUM_BOX_PARAMS = 9


def decode_box(
    box: Float[torch.Tensor, "num_boxes anchor_dims"],
) -> Float[torch.Tensor, "num_boxes 9"]:
    """Decode anchors into metric boxes.

    Args:
        box: Anchors, see ``AnchorFieldIndex``.

    Returns:
        ``(x, y, z, l, w, h, yaw, vx, vy)`` boxes.
    """
    yaw = torch.atan2(box[:, AnchorFieldIndex.SIN_YAW], box[:, AnchorFieldIndex.COS_YAW])
    return torch.cat(
        [
            box[:, CENTER],
            box[:, SIZE].exp(),
            yaw[:, None],
            box[:, AnchorFieldIndex.VELOCITY_X : AnchorFieldIndex.VELOCITY_Y + 1],
        ],
        dim=-1,
    )[:, :_NUM_BOX_PARAMS]


class SparseBox3DDecoder:
    """Pick the most confident boxes of a frame.

    Without instance ids every (anchor, class) pair competes, so an anchor may yield boxes of
    several classes. With instance ids every anchor yields one box of its best class, which
    keeps one id per box. With a quality estimate the scores are multiplied by the
    centerness before the final ordering.
    """

    def __init__(
        self, num_output: int = 300, score_threshold: float | None = None, sorted: bool = True
    ) -> None:
        """Initialize the decoder.

        Args:
            num_output: Number of boxes kept per sample before the threshold.
            score_threshold: Score a box needs to be kept, ``None`` for all.
            sorted: Whether the boxes are ordered by score.
        """
        self.num_output = num_output
        self.score_threshold = score_threshold
        self.sorted = sorted

    def decode(
        self,
        cls_scores: Float[torch.Tensor, "batch_size num_anchors num_classes"],
        box_preds: Float[torch.Tensor, "batch_size num_anchors anchor_dims"],
        quality: Float[torch.Tensor, "batch_size num_anchors 2"] | None = None,
        instance_ids: Int64[torch.Tensor, "batch_size num_anchors"] | None = None,
    ) -> list[Detection3DSamplePredictions]:
        """Decode the outputs of one decoder layer.

        Args:
            cls_scores: Class logits of every anchor.
            box_preds: Anchors, see ``AnchorFieldIndex``.
            quality: Centerness and yawness logits of every anchor, see ``QualityIndex``.
            instance_ids: Tracking id of every anchor.

        Returns:
            The boxes of every sample.
        """
        cls_scores = cls_scores.float().sigmoid()
        box_preds = box_preds.float()
        if instance_ids is not None:
            cls_scores, cls_ids = cls_scores.max(dim=-1)
            cls_scores = cls_scores.unsqueeze(-1)
        batch_size, num_anchors, num_cls = cls_scores.shape
        num_output = min(self.num_output, num_anchors * num_cls)
        cls_scores, indices = cls_scores.flatten(start_dim=1).topk(
            num_output, dim=1, sorted=self.sorted
        )
        if instance_ids is None:
            cls_ids = indices % num_cls
        anchor_indices = indices // num_cls
        keep = None
        if self.score_threshold is not None:
            keep = cls_scores >= self.score_threshold

        if quality is not None:
            centerness = quality[..., QualityIndex.CENTERNESS].float()
            centerness = torch.gather(centerness, 1, anchor_indices)
            cls_scores = cls_scores * centerness.sigmoid()
            cls_scores, order = torch.sort(cls_scores, dim=1, descending=True)
            if instance_ids is None:
                cls_ids = torch.gather(cls_ids, 1, order)
            if keep is not None:
                keep = torch.gather(keep, 1, order)
            anchor_indices = torch.gather(anchor_indices, 1, order)

        predictions = []
        for index in range(batch_size):
            category_ids = cls_ids[index]
            if instance_ids is not None:
                category_ids = category_ids[anchor_indices[index]]
            scores = cls_scores[index]
            box = box_preds[index, anchor_indices[index]]
            ids = None if instance_ids is None else instance_ids[index, anchor_indices[index]]
            if keep is not None:
                category_ids = category_ids[keep[index]]
                scores = scores[keep[index]]
                box = box[keep[index]]
                ids = None if ids is None else ids[keep[index]]
            predictions.append(
                Detection3DSamplePredictions(
                    bboxes_3d=decode_box(box),
                    scores_3d=scores,
                    labels_3d=category_ids.long(),
                    instance_ids_3d=ids,
                )
            )
        return predictions
