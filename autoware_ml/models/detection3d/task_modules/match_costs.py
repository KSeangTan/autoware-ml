"""Reusable matching costs for detection3d assignment.

This module contains pairwise matching costs used by Hungarian assignment.
"""

from __future__ import annotations

from dataclasses import dataclass

from jaxtyping import Float32, Int64, Bool
import torch


@dataclass(frozen=True)
class ClassificationCost:
    """Compute focal-style classification cost for Hungarian matching.

    The cost mirrors the classification term used by query-based detection
    assigners.
    """

    weight: float = 1.0
    alpha: float = 0.25
    gamma: float = 2.0
    eps: float = 1e-12

    def __call__(
        self,
        cls_logits: Float32[torch.Tensor, "batch_size num_queries num_classes"],
        gt_labels: Int64[torch.Tensor, "batch_size max_num_gt_bboxes"],
        valid_masks: Bool[torch.Tensor, "batch_size max_num_gt_bboxes"],
    ) -> Float32[torch.Tensor, "batch_size num_queries max_num_gt_bboxes"]:
        """
        Compute pairwise classification cost between queries and labels. Since it's
        batch-wise implemnatation, the cost is computed for invalid gt_bboxes, but the cost will be
        highest for invalid gt_bboxes, so the assignment will not match to invalid gt_bboxes.dense_

        Args:
            cls_logits: Classification logits for each query.
            gt_labels: Ground-truth class labels.
            valid_masks: Mask indicating valid ground-truth labels.

        Returns:
            Pairwise classification cost matrix in:
            [
                [cost(pred_box_0, gt_box_0), cost(pred_box_0, gt_box_1), ...]
                [cost(pred_box_1, gt_box_0), cost(pred_box_1, gt_box_1), ...]
            ]
            for every batch.
        """
        num_queries = cls_logits.shape[1]
        probs = cls_logits.sigmoid().clamp(min=self.eps, max=1.0 - self.eps)
        neg_cost = -(1.0 - probs + self.eps).log() * (1.0 - self.alpha) * probs.pow(self.gamma)
        pos_cost = -(probs + self.eps).log() * self.alpha * (1.0 - probs).pow(self.gamma)

        # (batch_size, max_num_bboxes) -> (batch_size, num_queries, max_num_gt_bboxes)
        class_indices = gt_labels.unsqueeze(1).expand(-1, num_queries, -1)
        cost = (pos_cost.gather(2, class_indices) - neg_cost.gather(2, class_indices)) * self.weight
        cost = (pos_cost[:, gt_labels] - neg_cost[:, gt_labels]) * self.weight
        # valid_mask: (batch_size, max_num_gt_bboxes) -> (batch_size, 1, max_num_gt_bboxes)
        # Broadcast to every column (gt_bboxes)
        cost = cost.masked_fill(~valid_masks.unsqueeze(1), torch.finfo(cost.dtype).max)
        return cost


@dataclass(frozen=True)
class BBoxBEVL1Cost:
    """Compute normalized BEV L1 cost between proposal and target boxes.

    The cost operates on BEV box centers normalized by the detector range.
    """

    weight: float = 1.0

    def __call__(
        self, bboxes: torch.Tensor, gt_bboxes: torch.Tensor, point_cloud_range: list[float]
    ) -> torch.Tensor:
        """Compute pairwise BEV L1 cost in normalized coordinates.

        Args:
            bboxes: Predicted boxes.
            gt_bboxes: Ground-truth boxes.
            point_cloud_range: Detector point-cloud range.

        Returns:
            Pairwise BEV L1 cost matrix.
        """
        pc_start = bboxes.new_tensor(point_cloud_range[0:2])
        pc_extent = bboxes.new_tensor(point_cloud_range[3:5]) - pc_start
        norm_bboxes = (bboxes[:, :2] - pc_start) / pc_extent
        norm_gt_bboxes = (gt_bboxes[:, :2] - pc_start) / pc_extent
        return torch.cdist(norm_bboxes, norm_gt_bboxes, p=1) * self.weight


@dataclass(frozen=True)
class IoU3DCost:
    """Compute a negative-IoU matching cost.

    Higher overlaps produce lower matching costs, making the term suitable for
    Hungarian assignment.
    """

    weight: float = 1.0

    def __call__(self, iou: torch.Tensor) -> torch.Tensor:
        """Convert IoU values into a minimization cost.

        Args:
            iou: Pairwise IoU matrix.

        Returns:
            Pairwise minimization cost derived from IoU.
        """
        return -iou * self.weight
