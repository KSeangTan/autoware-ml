"""Reusable box coders for detection3d heads.

This module implements reusable box encoding and decoding logic for 3D
detection heads and deployment paths.
"""

from __future__ import annotations

from dataclasses import dataclass

from jaxtyping import Float32
import torch


@dataclass
class TransFusionBBoxCoder:
    """Encode and decode boxes for TransFusion-style query heads.

    Attributes:
        pc_range: Point-cloud range used by the detector.
        out_size_factor: BEV downsampling factor between point space and feature space.
        voxel_size: Voxel size along each spatial axis.
        post_center_range: Optional metric-space range used to filter predictions.
        score_threshold: Optional score threshold applied during decoding. A
            scalar applies to every class; a sequence provides one threshold
            per class index.
        code_size: Number of regression channels produced by the head.
    """

    pc_range: list[float]
    out_size_factor: int
    voxel_size: list[float]
    # post_center_range: list[float] | None = None
    # score_threshold: float | Sequence[float] | None = None
    code_size: int = 8

    def encode(
        self,
        gt_boxes: Float32[torch.Tensor, "batch_size max_num_boxes 9"],
    ) -> Float32[torch.Tensor, "batch_size max_num_boxes code_size"]:
        """Encode metric-space boxes into normalized regression targets.

        Args:
            gt_boxes: Ground-truth boxes in metric coordinates, where each box is represented as
                (center_x, center_y, center_z, length, width, height, yaw, velocity_x, velocity_y).
                Note that it will also encode invalid boxes (e.g., zero-padded boxes) into
                regression targets, so the caller should mask out invalid boxes if necessary.

        Returns:
            Encoded regression targets aligned with the TransFusion head layout.
        """
        batch_size, max_num_boxes, _ = gt_boxes.shape
        targets = torch.zeros(
            (batch_size, max_num_boxes, self.code_size),
            device=gt_boxes.device,
            dtype=gt_boxes.dtype,
        )
        targets[:, :, 0] = (gt_boxes[:, :, 0] - self.pc_range[0]) / (
            self.out_size_factor * self.voxel_size[0]
        )
        targets[:, :, 1] = (gt_boxes[:, :, 1] - self.pc_range[1]) / (
            self.out_size_factor * self.voxel_size[1]
        )
        dims = gt_boxes[:, :, 3:6]
        log_dims = dims.log()
        targets[:, :, 3] = log_dims[:, :, 0]
        targets[:, :, 4] = log_dims[:, :, 1]
        targets[:, :, 5] = log_dims[:, :, 2]
        targets[:, :, 2] = gt_boxes[:, :, 2] + gt_boxes[:, :, 5] * 0.5
        targets[:, :, 6] = torch.sin(gt_boxes[:, :, 6])
        targets[:, :, 7] = torch.cos(gt_boxes[:, :, 6])
        if self.code_size == 10:
            targets[:, :, 8:10] = gt_boxes[:, :, 7:9]
        return targets

    def decode_heatmaps(
        self,
        heatmaps: Float32[torch.Tensor, "batch_size num_classes num_proposals"],
    ) -> tuple[
        Float32[torch.Tensor, "batch_size num_proposals"],
        Float32[torch.Tensor, "batch_size num_proposals"],
    ]:
        """Decode heatmaps into class predictions and scores.

        Args:
            heatmaps: Class confidence heatmap.
        """
        final_scores, final_preds = heatmaps.max(dim=1)
        return final_scores, final_preds

    def decode_boxes(
        self,
        rots: Float32[torch.Tensor, "batch_size 2 num_proposals"],
        dims: Float32[torch.Tensor, "batch_size 3 num_proposals"],
        centers: Float32[torch.Tensor, "batch_size 2 num_proposals"],
        heights: Float32[torch.Tensor, "batch_size 1 num_proposals"],
        vels: Float32[torch.Tensor, "batch_size 2 num_proposals"] | None,
    ) -> Float32[torch.Tensor, "batch_size code_size num_proposals"]:
        """
        Decode regression channels into metric-space boxes, where each box is represented as
        (center_x, center_y, center_z, length, width, height, yaw,  velocity_x, velocity_y).

        Args:
            rots: Rotation channels storing sine and cosine values.
            dims: Log-space box dimensions.
            centers: Predicted BEV center offsets.
            heights: Predicted box gravity heights, where the box center
            is already at height + 0.5 * box_height.
            vels: Optional velocity channels.

        Returns:
            Decoded boxes in metric coordinates.
        """
        centers = centers.clone()
        dims = dims.clone()
        heights = heights.clone()
        centers[:, 0, :] = (
            centers[:, 0, :] * self.out_size_factor * self.voxel_size[0] + self.pc_range[0]
        )
        centers[:, 1, :] = (
            centers[:, 1, :] * self.out_size_factor * self.voxel_size[1] + self.pc_range[1]
        )
        dims = dims.exp()
        # heights = heights - dims[:, 2:3, :] * 0.5
        yaw = torch.atan2(rots[:, 0:1, :], rots[:, 1:2, :])
        if vels is None:
            final_boxes = torch.cat([centers, heights, dims, yaw], dim=1)
        else:
            final_boxes = torch.cat([centers, heights, dims, yaw, vels], dim=1)
        return final_boxes
