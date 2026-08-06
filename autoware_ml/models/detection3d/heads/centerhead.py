"""Detection heads used by CenterPoint-style models.

This module implements dense prediction heads, target generation, decoding,
and training losses used by CenterPoint-style detectors.
"""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from types import MappingProxyType

from jaxtyping import Bool, Float32, Int64
import torch
import torch.nn as nn
import torch.nn.functional as F

from autoware_ml.losses.detection3d.gaussian_focal import GaussianFocalLoss
from autoware_ml.models.common.layers.conv import ConvModule
from autoware_ml.models.dataclasses.multi_task_predictions import MultiTaskPredictions
from autoware_ml.models.detection3d.task_modules.heatmap import (
    batch_circle_nms,
    vectorize_gaussian_radii,
    create_gaussian_heatmaps,
)
from autoware_ml.models.detection3d.dataclasses.detection3d import Detection3DPredictions
from autoware_ml.models.detection3d.dataclasses.outputs import Detection3DOutputs, CenterHeadOutputs
from autoware_ml.models.detection3d.dataclasses.targets import CenterHeadTargets
from autoware_ml.types.geometry import Box3DFieldIndex


def _gather_feat(
    features: Float32[torch.Tensor, "batch_size height*width channels"],
    indices: Int64[torch.Tensor, "batch_size max_num_bboxes"],
) -> Float32[torch.Tensor, "batch_size max_num_bboxes channels"]:
    """Gather flattened features at the requested indices."""
    channels = features.shape[-1]
    expanded_indices = indices.unsqueeze(-1).expand(*indices.shape, channels)
    return features.gather(dim=1, index=expanded_indices)


def _transpose_and_gather_feat(
    features: Float32[torch.Tensor, "batch_size channels height width"],
    indices: Int64[torch.Tensor, "batch_size max_num_bboxes"],
) -> Float32[torch.Tensor, "batch_size height*width channels"]:
    """Transpose a feature map and gather flattened features."""
    features = features.permute(0, 2, 3, 1).contiguous()
    features = features.view(features.shape[0], -1, features.shape[-1])
    return _gather_feat(features, indices)


class CenterHead(nn.Module):
    """Predict dense heatmaps and regression maps for CenterPoint.

    The head uses a shared BEV tower followed by lightweight prediction
    branches for heatmap, center offsets, dimensions, rotation, and velocity.
    It also owns the CenterPoint target generation, loss computation, and
    decode logic so the model wrapper stays reusable and task-agnostic.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        shared_channels: int,
        point_cloud_range: list[float],
        voxel_size: list[float],
        out_size_factor: int,
        max_objs: int,
        min_radius: int,
        score_threshold: float,
        post_max_size: int,
        nms_min_radius: float,
        class_names: Sequence[str] | None = None,
        gaussian_overlap: float = 0.1,
        loss_bbox_weight: float = 0.25,
        heatmap_init_bias: float = -2.19,
        use_velocity: bool = True,
    ) -> None:
        """Initialize the CenterPoint head.

        Args:
            in_channels: Input feature channels.
            num_classes: Number of detection classes.
            shared_channels: Channel count for the shared tower.
            point_cloud_range: Detector point-cloud range.
            voxel_size: Voxel size used by preprocessing.
            out_size_factor: Downsampling factor between BEV cells and head outputs.
            max_objs: Maximum number of targets kept per sample.
            min_radius: Minimum Gaussian radius for heatmap targets.
            score_threshold: Score threshold applied during decoding.
            post_max_size: Maximum number of predictions kept after decoding.
            nms_min_radius: Minimum center distance used by circle NMS.
            class_names: Optional ordered class names for metric logging.
            gaussian_overlap: Minimum Gaussian overlap with the target box.
            loss_bbox_weight: Weight applied to the box regression loss.
            heatmap_init_bias: Initial bias used by the heatmap prediction branch.
            use_velocity: Whether to predict velocity components.
        """
        super().__init__()
        self.num_classes = num_classes
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.out_size_factor = out_size_factor
        self.max_objs = max_objs
        self.min_radius = min_radius
        self.score_threshold = score_threshold
        self.post_max_size = post_max_size
        self.nms_min_radius = nms_min_radius
        self.class_names = tuple(class_names) if class_names is not None else None
        self.gaussian_overlap = gaussian_overlap
        self.loss_bbox_weight = loss_bbox_weight
        self.heatmap_init_bias = heatmap_init_bias
        self.use_velocity = use_velocity
        self.box_code_size = 10 if use_velocity else 8

        self.shared_conv = ConvModule(in_channels, shared_channels)
        self.heatmap = self._build_head(shared_channels, num_classes, init_bias=heatmap_init_bias)
        self.reg = self._build_head(shared_channels, 2)
        self.height = self._build_head(shared_channels, 1)
        self.dim = self._build_head(shared_channels, 3)
        self.rot = self._build_head(shared_channels, 2)
        self.vel = self._build_head(shared_channels, 2) if use_velocity else None

        self.loss_heatmap = GaussianFocalLoss()
        self.loss_bbox = nn.L1Loss(reduction="none")

    def _build_head(
        self, in_channels: int, out_channels: int, init_bias: float | None = None
    ) -> nn.Sequential:
        """Build one CenterPoint prediction branch."""
        head = nn.Sequential(
            ConvModule(in_channels, in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )
        if init_bias is not None:
            nn.init.constant_(head[-1].bias, init_bias)
        return head

    def forward(
        self, x: Float32[torch.Tensor, "batch_size neck_feature_channels height width"]
    ) -> CenterHeadOutputs:
        """Predict dense heatmap and regression maps."""
        shared = self.shared_conv(x)  # (Batch_size, shared_channels, height, width)
        heatmaps = self.heatmap(shared)  # (Batch_size, num_classes, height, width)
        regs = self.reg(shared)  # (Batch_size, 2, height, width)
        heights = self.height(shared)  # (Batch_size, 1, height, width)
        dims = self.dim(shared)  # (Batch_size, 3, height, width)
        rots = self.rot(shared)  # (Batch_size, 2, height, width)
        vels = (
            self.vel(shared) if self.vel is not None else None
        )  # (Batch_size, 2, height, width) or None

        return CenterHeadOutputs(
            heatmaps=heatmaps, regs=regs, heights=heights, dims=dims, rots=rots, vels=vels
        )

    def get_targets(
        self,
        gt_bboxes_3d: Float32[torch.Tensor, "batch_size max_num_3d_gt_bboxes num_Box3DFieldIndex"],
        gt_labels_3d: Float32[torch.Tensor, "batch_size max_num_3d_gt_bboxes"],
        gt_valid_bboxes: Float32[torch.Tensor, " batch_size"],
        feature_map_size: tuple[int, int],
        device: torch.device,
    ) -> CenterHeadTargets:
        """Build heatmap and regression targets for one batch."""
        batch_size = len(gt_bboxes_3d)
        max_num_bboxes = gt_bboxes_3d.shape[1]
        feature_height, feature_width = feature_map_size

        # Movement of tensors to the correct device and type
        # Get only the first K params for ground truths
        gt_bboxes_3d[:, :, : self.box_code_size] = gt_bboxes_3d[:, :, : self.box_code_size].to(
            device=device
        )
        gt_labels_3d = gt_labels_3d.to(device=device, dtype=torch.long)
        gt_valid_bboxes = gt_valid_bboxes.to(device=device)

        # Vectorization implementation instead of for-loops
        center_x = (
            (gt_bboxes_3d[:, :, Box3DFieldIndex.X] - self.point_cloud_range[0])
            / self.voxel_size[0]
            / self.out_size_factor
        )
        center_y = (
            (gt_bboxes_3d[:, :, Box3DFieldIndex.Y] - self.point_cloud_range[1])
            / self.voxel_size[1]
            / self.out_size_factor
        )

        # (batch_size, max_num_bboxes) boolean mask for valid boxes based on the distance
        valid_distance_masks = (
            (center_x >= 0)
            & (center_x < feature_width)
            & (center_y >= 0)
            & (center_y < feature_height)
        )

        # (batch_size, max_num_bboxes) boolean mask for valid boxes based on the number of valid boxes per sample
        # (max_num_bboxes) -> (1, max_num_bboxes) -> (batch_size, max_num_bboxes) < gt_valid_bboxes.unsqueeze(1) (batch_size, 1)
        # -> (batch_size, max_num_bboxes)
        valid_num_bboxes_masks = torch.arange(max_num_bboxes, device=device).unsqueeze(0).expand(
            batch_size, -1
        ) < gt_valid_bboxes.unsqueeze(1)
        # (batch_size, max_num_bboxes) boolean mask for valid boxes based on both distance and number of valid boxes
        valid_bbox_masks = valid_distance_masks & valid_num_bboxes_masks

        lengths = (
            gt_bboxes_3d[:, :, Box3DFieldIndex.LENGTH] / self.voxel_size[0] / self.out_size_factor
        )
        widths = (
            gt_bboxes_3d[:, :, Box3DFieldIndex.WIDTH] / self.voxel_size[1] / self.out_size_factor
        )

        # (batch_size, max_num_bboxes)
        gaussian_radii = vectorize_gaussian_radii(
            widths=lengths,
            heights=widths,
            min_overlap=self.gaussian_overlap,
        )
        center = torch.stack((center_x, center_y), dim=-1)
        # (batch_size, max_num_bboxes, 2)
        center_int = center.floor().to(torch.long)
        heatmaps = create_gaussian_heatmaps(
            heatmap_width=feature_width,
            heatmap_height=feature_height,
            num_classes=self.num_classes,
            batch_size=batch_size,
            centers=center_int,
            gaussian_radii=gaussian_radii.long(),
            gt_bboxes_labels=gt_labels_3d,
            valid_masks=valid_bbox_masks,
            device=device,
        )
        center_targets = torch.stack(
            (center_x - center_int[:, :, 0].floor(), center_y - center_int[:, :, 1].floor()), dim=-1
        )
        dim_targets = gt_bboxes_3d[:, :, Box3DFieldIndex.LENGTH : Box3DFieldIndex.HEIGHT].log()
        heading_targets = torch.stack(
            (
                torch.sin(gt_bboxes_3d[:, :, Box3DFieldIndex.YAW]),
                torch.cos(gt_bboxes_3d[:, :, Box3DFieldIndex.YAW]),
            ),
            dim=-1,
        )
        # (batch_size, max_num_bboxes, 2 + 1 + 3 + 2) if not velocity else
        # (batch_size, max_num_bboxes, 2 + 1 + 3 + 2 + 2)
        reg_targets = torch.cat(
            [center_targets, gt_bboxes_3d[:, :, Box3DFieldIndex.Z], dim_targets, heading_targets],
            dim=-1,
        )

        if self.use_velocity:
            vel_targets = gt_bboxes_3d[
                :, :, Box3DFieldIndex.VELOCITY_X : Box3DFieldIndex.VELOCITY_Y + 1
            ]
            reg_targets = torch.cat([reg_targets, vel_targets], dim=-1)

        # (batch_size, max_num_bboxes) -> (batch_size, max_num_bboxes)
        reg_indices = (center_int[:, :, 1] * feature_width + center_int[:, :, 0]).long()
        return CenterHeadTargets(
            heatmaps=heatmaps,
            reg_targets=reg_targets,
            valid_masks=valid_bbox_masks,
            reg_indices=reg_indices,
        )

    def loss(
        self,
        outputs: Detection3DOutputs,
        gt_bboxes_3d: Float32[torch.Tensor, "batch_size max_num_3d_gt_bboxes num_Box3DFieldIndex"],
        gt_labels_3d: Float32[torch.Tensor, "batch_size max_num_3d_gt_bboxes"],
        gt_valid_bboxes: Float32[torch.Tensor, " batch_size"],
    ) -> MappingProxyType[str, torch.Tensor]:
        """
        Compute CenterPoint heatmap and box losses.

        Args:
            gt_bboxes_3d: Ground truth 3D bounding boxes for the batch.
            gt_labels_3d: Ground truth class labels for the 3D bounding boxes.
            gt_valid_bboxes: Number of valid bounding boxes for each sample in the batch.
            outputs: CenterHeadOutputs containing the predicted heatmap and regression maps.

        Returns:
            MappingProxyType[str, torch.Tensor]: A read-only dictionary containing the total loss,
                heatmap loss, and box regression loss.
        """
        if outputs.center_head_outputs is None:
            raise ValueError(
                "CenterHeadOutputs must be provided in Detection3DOutputs for loss computation."
            )

        output_heatmaps = outputs.center_head_outputs.heatmaps
        heatmap_size = (int(output_heatmaps.shape[-2]), int(output_heatmaps.shape[-1]))
        targets = self.get_targets(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            feature_map_size=heatmap_size,
            gt_valid_bboxes=gt_valid_bboxes,
            device=output_heatmaps.device,
        )
        loss_heatmap = self.loss_heatmap(output_heatmaps, targets.heatmaps)

        bbox_predictions = [
            outputs.center_head_outputs.regs,
            outputs.center_head_outputs.heights,
            outputs.center_head_outputs.dims,
            outputs.center_head_outputs.rots,
        ]
        if self.use_velocity and outputs.center_head_outputs.vels is not None:
            bbox_predictions.append(outputs.center_head_outputs.vels)

        bbox_predictions = torch.cat(bbox_predictions, dim=1)

        # Gather the predicted bounding box parameters across channels at the target indices
        # (batch_size, channels, height, width) -> (batch_size, max_num_bboxes, output_channels)
        flatten_bbox_predictions = _transpose_and_gather_feat(bbox_predictions, targets.reg_indices)
        # (batch_size, max_num_bboxes) -> (batch_size, max_num_bboxes, output_channels)
        bbox_valid_masks = (
            targets.valid_masks.unsqueeze(-1).expand_as(flatten_bbox_predictions).float()
        )
        bbox_losses = (
            self.loss_bbox(flatten_bbox_predictions, targets.reg_targets) * bbox_valid_masks
        )

        # Average over the number of valid bounding boxes and avoid division by zero
        bbox_losses = bbox_losses.sum() / bbox_valid_masks.sum().clamp_min(1.0)
        total_loss = loss_heatmap + self.loss_bbox_weight * bbox_losses
        # MappingProxyType is used to create a read-only dictionary for the loss outputs
        return MappingProxyType(
            {"loss": total_loss, "loss_heatmap": loss_heatmap, "loss_bbox": bbox_losses}
        )

    def _decode_regression_outputs(
        self,
        center_head_outputs: CenterHeadOutputs,
        flatten_indices: Int64[torch.Tensor, "batch_size max_num_bboxes"],
        batch_size: int,
        width: int,
    ) -> Float32[torch.Tensor, "batch_size num_classes*max_num_bboxes box_code_size"]:
        """
        Decode the regression outputs to convert it to physical coordinates
        from the CenterHeadOutputs.
        Args:
          center_head_outputs: Outputs from the CenterHead head.
          flatten_indices: Flattened indices to gather the regression outputs.
          batch_size: Batch size of the input.
          width: Width of the feature map.
          feature_map_size: Tuple of (height, width) of the feature map.

        Returns:
          bbox_predictions: Decoded bounding box predictions.
        """
        ys = torch.div(flatten_indices, width, rounding_mode="floor")
        xs = flatten_indices % width

        # (batch_size, 2, height, width) -> (batch_size, height, width, 2) -> (batch_size, height*width, 2) -> (batch_size, num_classes*max_num_bboxes, 2)
        regs = center_head_outputs.regs.permute(0, 2, 3, 1).reshape(batch_size, -1, 2)[
            flatten_indices
        ]
        # (batch_size, num_classes*max_num_bboxes, 1)
        heights = center_head_outputs.heights.permute(0, 2, 3, 1).reshape(batch_size, -1, 1)[
            flatten_indices
        ]
        # (batch_size, num_classes*max_num_bboxes, 3)
        dims = center_head_outputs.dims.permute(0, 2, 3, 1).reshape(batch_size, -1, 3)[
            flatten_indices
        ]
        # Convert log-dimensions back to actual dimensions
        dims = dims.exp()
        # (batch_size, num_classes*max_num_bboxes, 2)
        rots = center_head_outputs.rots.permute(0, 2, 3, 1).reshape(batch_size, -1, 2)[
            flatten_indices
        ]
        vels = center_head_outputs.vels if self.use_velocity else None
        if vels is not None:
            # (batch_size, num_classes*max_num_bboxes, 2)
            vels = vels.permute(0, 2, 3, 1).reshape(batch_size, -1, 2)[flatten_indices]

        # Compute yaws with atan2
        # (batch_size, num_classes*max_num_bboxes, 1)
        batch_yaws = torch.atan2(rots[:, :, 0], rots[:, :, 1]).unsqueeze(1)
        # Add center translation offsets to their x and y grid and convert them from bev-grid representation to the lidar physical representation
        # (batch_size, num_classes*max_num_bboxes, 1)
        batch_xs = (xs.to(regs.dtype) + regs[:, :, 0]).unsqueeze(
            2
        ) * self.out_size_factor * self.voxel_size[0] + self.point_cloud_range[0]
        batch_ys = (ys.to(regs.dtype) + regs[:, :, 1]).unsqueeze(
            2
        ) * self.out_size_factor * self.voxel_size[1] + self.point_cloud_range[1]

        # (1+1+1+3+1) = 7 or (1+1+1+3+1+2) = 9
        bbox_predictions = [batch_xs, batch_ys, heights, dims, batch_yaws]
        if vels is not None:
            bbox_predictions.append(vels)
        # (batch_size, num_classes*max_num_bboxes, 7 or 9)
        bbox_predictions = torch.cat(bbox_predictions, dim=2)

        assert bbox_predictions.shape[2] == self.box_code_size, (
            f"Expected bbox_predictions to have shape[2] == {self.box_code_size}, "
            f"but got {bbox_predictions.shape[2]}"
        )
        return bbox_predictions

    def _filter_bbox_predictions(
        self,
        bbox_predictions: Float32[
            torch.Tensor, "batch_size num_classes max_num_bboxes box_code_size"
        ],
        scores: Float32[torch.Tensor, "batch_size num_classes max_num_bboxes"],
        class_ids: Int64[torch.Tensor, "batch_size num_classes max_num_bboxes"],
        keep_masks: Bool[torch.Tensor, "batch_size num_classes max_num_bboxes"],
        max_num_bboxes: int,
        batch_size: int,
    ) -> MultiTaskPredictions:
        """
        Filter the predictions based on the keep_masks and return a MultiTaskPredictions object.
        """
        # (batch_size, num_classes, max_num_bboxes) -> (batch_size, num_classes*max_num_bboxes)
        flatten_keep_masks = keep_masks.view(batch_size, -1)
        # Return empty list of predictions if no valid bboxes remain after filtering
        if flatten_keep_masks.sum() == 0:
            return MultiTaskPredictions(detection3d_predictions=[])

        # (batch_size, num_classes*max_num_bboxes)
        valid_flatten_scores = scores.view(batch_size, -1)[flatten_keep_masks]
        # Get the top-k indices based on the valid scores across classes
        # (batch_size, num_topk_indices)
        _, topk_indices = torch.topk(
            valid_flatten_scores, k=max_num_bboxes, largest=True, sorted=True, dim=1
        )
        # Select the scores corresponding to the top-k indices
        # (batch_size, num_topk_indices)
        valid_keep_flatten_scores = torch.gather(valid_flatten_scores, dim=1, index=topk_indices)

        # (batch_size, num_classes*max_num_bboxes)
        valid_flatten_class_ids = class_ids.view(batch_size, -1)[flatten_keep_masks]
        # Select the class ids corresponding to the top-k indices
        # (batch_size, num_topk_indices)
        valid_keep_flatten_class_ids = torch.gather(
            valid_flatten_class_ids, dim=1, index=topk_indices
        )

        # (batch_size, num_classes*max_num_bboxes, box_code_size)
        valid_bbox_prediction_masks = flatten_keep_masks.unsqueeze(-1).expand_as(bbox_predictions)
        valid_flatten_bbox_predictions = bbox_predictions.view(batch_size, -1, self.box_code_size)[
            valid_bbox_prediction_masks
        ]
        # (batch_size, num_topk_indices, box_code_size)
        valid_keep_flatten_bbox_predictions = torch.gather(
            valid_flatten_bbox_predictions,
            dim=1,
            index=topk_indices.unsqueeze(-1).expand(-1, -1, self.box_code_size),
        )

        # Iterate over the batch and create a list of Detection3dPredictions for each sample
        detection3d_predictions = []
        for batch_index in range(batch_size):
            keep_flatten_scores = valid_keep_flatten_scores[batch_index]
            keep_flatten_class_ids = valid_keep_flatten_class_ids[batch_index]
            keep_flatten_bbox_predictions = valid_keep_flatten_bbox_predictions[batch_index]

            detection3d_predictions.append(
                Detection3DPredictions(
                    bboxes_3d=keep_flatten_bbox_predictions,
                    scores_3d=keep_flatten_scores,
                    labels_3d=keep_flatten_class_ids,
                )
            )

        return MultiTaskPredictions(detection3d_predictions=detection3d_predictions)

    def _decode_heatmap_outputs(
        self, center_head_outputs: CenterHeadOutputs
    ) -> Float32[torch.Tensor, "batch_size num_classes height width"]:
        """
        Decode the heatmap outputs to apply sigmoid activation and non-maximum suppression.

        Args:
            center_head_outputs: Outputs from the CenterHead head.

        Returns:
            heatmaps: Decoded heatmaps after applying sigmoid and NMS.
        """
        heatmaps = center_head_outputs.heatmaps.sigmoid()
        pooled = F.max_pool2d(heatmaps, kernel_size=3, stride=1, padding=1)
        heatmaps = heatmaps * (pooled == heatmaps)
        return heatmaps

    def decode_outputs(self, outputs: Detection3DOutputs) -> MultiTaskPredictions:
        """
        Decode dense head outputs into 3D boxes, scores, and labels.
        """
        if outputs.center_head_outputs is None:
            raise ValueError(
                "CenterHeadOutputs must be provided in Detection3DOutputs for centerhead decoding."
            )

        heatmaps = self._decode_heatmap_outputs(outputs.center_head_outputs)
        batch_size, num_classes, height, width = heatmaps.shape
        max_num_bboxes = min(self.post_max_size, height * width)

        # (batch_size, num_classes, height*width)
        batch_scores = heatmaps.reshape(batch_size, num_classes, -1)

        # Get the top-k scores and their corresponding indices for each class in the batch
        top_scores, top_indices = batch_scores.topk(
            k=max_num_bboxes, dim=2
        )  # (batch_size, num_classes, max_num_bboxes)

        # (num_classes) -> (1, num_classes) -> (1, num_classes, 1) -> (batch_size, num_classes, max_num_bboxes)
        class_ids = (
            torch.arange(num_classes, device=heatmaps.device)
            .unsqueeze(0)
            .unsqueeze(2)
            .expand_as(top_indices)
        )
        # (batch_size, num_classes*max_num_bboxes)
        flatten_indices = top_indices.reshape(batch_size, -1)

        bbox_predictions = self._decode_regression_outputs(
            center_head_outputs=outputs.center_head_outputs,
            flatten_indices=flatten_indices,
            batch_size=batch_size,
            width=width,
        )
        valid_bboxes_masks = top_scores > self.score_threshold
        bbox_centers = bbox_predictions[:, :, :2]
        # (batch_size, num_classes, max_num_bboxes)
        # Note that batch_keep_masks includes the valid_bboxes_masks, so it doesn't need to apply it
        # again after NMS
        keep_masks = batch_circle_nms(
            bboxes_centers=bbox_centers,
            scores=top_scores,
            bboxes_labels=class_ids,
            valid_bboxes_masks=valid_bboxes_masks,
            post_max_size=self.post_max_size,
            min_radius=self.nms_min_radius,
        )
        # Filter the predictions based on the keep_masks and return MultiTaskPredictions
        multi_task_predictions = self._filter_bbox_predictions(
            bbox_predictions=bbox_predictions,
            scores=top_scores,
            class_ids=class_ids,
            keep_masks=keep_masks,
            max_num_bboxes=max_num_bboxes,
            batch_size=batch_size,
        )
        return multi_task_predictions

    def prepare_for_export(self) -> CenterHead:
        """Return an export-ready copy of the head.

        Returns:
            Deep copy of the head in evaluation mode.
        """
        return deepcopy(self).eval()
