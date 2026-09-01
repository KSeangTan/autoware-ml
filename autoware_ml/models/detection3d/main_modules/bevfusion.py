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

"""Native BEVFusion detector.

This module contains the high-level BEVFusion detector wrapper and export ABI.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from typing import Any

from jaxtyping import Float32, Int32, Int64


import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


from autoware_ml.metrics.base import MetricSuite
from autoware_ml.metrics.detection3d.eval_output import detection_eval_output
from autoware_ml.models.base import BaseModel
from autoware_ml.models.detection3d.main_modules.bevfusions.bevfusion_lidar import (
    BEVFusionLidar,
    BEVFusionLidarExportWrapper,
)
from autoware_ml.utils.deploy import ExportSpec
from autoware_ml.utils.point_cloud.batching import infer_batch_size_from_voxel_coords
from autoware_ml.preprocessing.data_preprocessor import DataPreprocessor


class _BEVFusionExportWrapper(nn.Module):
    """Wrap the camera-lidar main body export.

    The wrapper exposes the exact tensor-only signature expected by the
    deployment runtime and returns runtime-decodable detection outputs.
    """

    def __init__(self, model: BEVFusionDetectionModel) -> None:
        """Initialize the export wrapper.

        Args:
            model: BEVFusion model instance.
        """
        super().__init__()
        self.model = model

    def forward(
        self,
        voxels: Float32[torch.Tensor, "num_voxels max_num_points num_point_features"],
        coors: Int32[torch.Tensor, "num_voxels 3"],
        num_points_per_voxel: Int32[torch.Tensor, " num_voxels"],
        points: torch.Tensor,
        lidar2image: torch.Tensor,
        img_aug_matrix: torch.Tensor,
        geom_feats: torch.Tensor,
        kept: torch.Tensor,
        ranks: torch.Tensor,
        indices: torch.Tensor,
        image_feats: torch.Tensor,
    ) -> tuple[
        Float32[torch.Tensor, "num_box_code num_proposals"],
        Float32[torch.Tensor, " num_proposals"],
        Int64[torch.Tensor, " num_proposals"],
    ]:
        """Run export-time inference with precomputed BEV-pool metadata.

        Args:
            voxels: Lidar voxel tensor.
            coors: Voxel coordinates in ``(z, y, x)`` order without batch column.
            num_points_per_voxel: Number of points per voxel.
            points: Raw point tensor used for lidar depth guidance.
            lidar2image: Raw lidar-to-image projection matrices.
            img_aug_matrix: Image augmentation matrices.
            geom_feats: Precomputed BEV-pool geometry features.
            kept: Keep mask for pooled features.
            ranks: Sorted BEV ranks.
            indices: Sorting indices aligned with ``ranks``.
            image_feats: Precomputed image features.

        Returns:
            Tuple of ``bbox_pred``, ``score``, and ``label_pred``.
        """
        outputs = self.model._forward_export(
            voxels=voxels,
            coors=coors,
            num_points_per_voxel=num_points_per_voxel,
            points=points,
            lidar2image=lidar2image,
            img_aug_matrix=img_aug_matrix,
            geom_feats=geom_feats,
            kept=kept,
            ranks=ranks,
            indices=indices,
            image_feats=image_feats,
        )
        return export_detection_outputs(self.model.bbox_head, outputs)


class BEVFusionDetectionModel(BaseModel):
    """Compose a BEVFusion detector with camera and lidar branches.

    The model fuses image and lidar features in BEV space and exposes the
    shared Autoware-ML training, prediction, and export interfaces.
    """

    def __init__(
        self,
        data_preprocessor: DataPreprocessor,
        bevfusion_lidar: BEVFusionLidar,
        # TODO(KokSeang): Encoder and middle_encoder should be standardized to a common interface for all voxel encoders
        pts_voxel_encoder: nn.Module | None,
        pts_middle_encoder: nn.Module | None,
        pts_backbone: nn.Module | None,
        pts_neck: nn.Module | None,
        bbox_head: nn.Module,
        img_backbone: nn.Module | None = None,
        img_neck: nn.Module | None = None,
        view_transform: nn.Module | None = None,
        fusion_layer: nn.Module | None = None,
        optimizer: Callable[..., Optimizer] | None = None,
        scheduler: Callable[[Optimizer], LRScheduler] | None = None,
        metrics: Sequence[MetricSuite] | None = None,
    ) -> None:
        """Initialize BEVFusion.

        Args:
            pts_voxel_encoder: Lidar voxel encoder.
            pts_middle_encoder: Lidar BEV encoder.
            pts_backbone: BEV backbone of the lidar branch.
            pts_neck: BEV neck of the lidar branch.
            bbox_head: Detection head.
            img_backbone: Image backbone.
            img_neck: Image neck.
            view_transform: View transform from image features to BEV.
            fusion_layer: BEV fusion layer for multi-branch inputs.
            optimizer: Optimizer factory.
            scheduler: Scheduler factory.
            metrics: Detection metrics accumulated during validation and test.
        """
        super().__init__(optimizer=optimizer, scheduler=scheduler, metrics=metrics)
        self.lidar = (
            BEVFusionLidar(
                pts_voxel_encoder=pts_voxel_encoder,
                pts_middle_encoder=pts_middle_encoder,
                pts_backbone=pts_backbone,
                pts_neck=pts_neck,
            )
            if pts_voxel_encoder is not None and pts_middle_encoder is not None
            else None
        )
        self.bbox_head = bbox_head
        self.fusion_layer = fusion_layer
        self.camera = (
            BEVFusionCamera(
                img_backbone=img_backbone,
                img_neck=img_neck,
                view_transform=view_transform,
            )
            if img_backbone is not None and img_neck is not None and view_transform is not None
            else None
        )
        self._validate_geometry_contract()

    def _validate_geometry_contract(self) -> None:
        """Validate static geometry contracts between camera and lidar branches.

        Raises:
            ValueError: If configured lidar and image BEV branches do not
                agree on the BEV spatial shape.
        """
        if self.camera is None or self.lidar is None:
            return
        image_bev_shape = self.camera.expected_bev_shape
        if image_bev_shape is None:
            return
        if not hasattr(self.lidar.pts_middle_encoder, "output_shape"):
            return

        lidar_bev_shape = tuple(int(value) for value in self.lidar.pts_middle_encoder.output_shape)
        if image_bev_shape != lidar_bev_shape:
            raise ValueError(
                "BEVFusion image and lidar branches must share the same BEV shape. "
                f"Got image BEV shape {image_bev_shape} and lidar BEV shape {lidar_bev_shape}."
            )

    def _validate_runtime_bev_shapes(
        self,
        bev_features: Sequence[Float32[torch.Tensor, "batch_size channels height width"]],
    ) -> None:
        """Validate runtime BEV tensor shapes before multi-branch fusion.

        Args:
            bev_features: Sequence of BEV feature maps to fuse.

        Raises:
            ValueError: If the BEV branches do not share the same spatial shape.
        """
        if len(bev_features) < 2:
            return
        reference_shape = tuple(bev_features[0].shape[-2:])
        for feature in bev_features[1:]:
            feature_shape = tuple(feature.shape[-2:])
            if feature_shape != reference_shape:
                raise ValueError(
                    "BEVFusion branches must share the same runtime BEV shape before fusion. "
                    f"Expected {reference_shape}, got {feature_shape}."
                )

    def _forward_export(
        self,
        voxels: Float32[torch.Tensor, "num_voxels max_num_points num_point_features"],
        coors: Int32[torch.Tensor, "num_voxels 3"],
        num_points_per_voxel: Int32[torch.Tensor, " num_voxels"],
        points: torch.Tensor,
        lidar2image: torch.Tensor,
        img_aug_matrix: torch.Tensor,
        geom_feats: torch.Tensor,
        kept: torch.Tensor,
        ranks: torch.Tensor,
        indices: torch.Tensor,
        image_feats: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Run the export-time main body with runtime-compatible inputs.

        Args:
            voxels: Lidar voxel tensor.
            coors: Voxel coordinates in ``(z, y, x)`` order without batch column.
            num_points_per_voxel: Number of points per voxel.
            points: Raw point tensor used for lidar depth guidance.
            lidar2image: Raw lidar-to-image projection matrices.
            img_aug_matrix: Image augmentation matrices.
            geom_feats: Precomputed BEV-pool geometry features.
            kept: Keep mask for pooled features.
            ranks: Sorted BEV ranks.
            indices: Sorting indices aligned with ``ranks``.
            image_feats: Precomputed image features.

        Returns:
            Detection head outputs produced by the export path.
        """
        if self.camera is None:
            raise ValueError("Image branch is not configured.")
        image_bev = self.camera.forward_export(
            points=points,
            lidar2image=lidar2image,
            img_aug_matrix=img_aug_matrix,
            geom_feats=geom_feats,
            kept=kept,
            ranks=ranks,
            indices=indices,
            image_feats=image_feats,
        )

        return self._forward_with_batch_size(
            voxels=voxels,
            num_points=num_points_per_voxel,
            voxel_coords=BEVFusionLidar.runtime_coors_to_voxel_coords(coors),
            batch_size=1,
            image_bev=image_bev,
        )

    def _forward_with_batch_size(
        self,
        voxels: Float32[torch.Tensor, "num_voxels max_num_points num_point_features"] | None = None,
        num_points: Int32[torch.Tensor, " num_voxels"] | None = None,
        voxel_coords: Int32[torch.Tensor, "num_voxels 4"] | None = None,
        img: Sequence[torch.Tensor] | None = None,
        points: Sequence[torch.Tensor] | None = None,
        lidar2img: Sequence[torch.Tensor] | None = None,
        camera_intrinsics: Sequence[torch.Tensor] | None = None,
        lidar2cam: Sequence[torch.Tensor] | None = None,
        batch_size: int | None = None,
        image_bev: Float32[torch.Tensor, "batch_size channels height width"] | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Run the configured BEV branches and dense head.

        Args:
            voxels: Optional lidar voxel tensor.
            num_points: Optional number of points per voxel.
            voxel_coords: Optional voxel coordinates with batch indices.
            img: Optional multiview image tensors.
            points: Optional per-sample lidar points for depth guidance.
            lidar2img: Optional lidar-to-image projection matrices.
            camera_intrinsics: Optional camera intrinsic matrices.
            lidar2cam: Optional lidar-to-camera extrinsics.
            batch_size: Optional explicit batch size.
            image_bev: Optional precomputed image BEV tensor.
            **kwargs: Unused extra keyword arguments.

        Returns:
            Detection head outputs for the configured branches.
        """
        del kwargs
        bev_features: list[Float32[torch.Tensor, "batch_size channels height width"]] = []

        if voxels is not None and num_points is not None and voxel_coords is not None:
            if batch_size is None:
                batch_size = infer_batch_size_from_voxel_coords(voxel_coords)
            if self.lidar is None:
                raise ValueError("Lidar branch is not configured.")
            bev_features.append(self.lidar(voxels, num_points, voxel_coords, batch_size=batch_size))

        if image_bev is not None:
            bev_features.append(image_bev)
        elif img is not None and camera_intrinsics is not None and lidar2cam is not None:
            if points is None or lidar2img is None:
                raise ValueError(
                    "BEVFusion image branch requires points and lidar2img for depth guidance."
                )
            if self.camera is None:
                raise ValueError("Image branch is not configured.")
            bev_features.append(self.camera(img, points, lidar2img, camera_intrinsics, lidar2cam))

        if not bev_features:
            raise ValueError("At least one BEV branch must be provided.")
        self._validate_runtime_bev_shapes(bev_features)

        if len(bev_features) == 1:
            fused = bev_features[0]
        else:
            if self.fusion_layer is None:
                raise ValueError(
                    "Fusion layer must be configured when multiple BEV branches are used."
                )
            fused = self.fusion_layer(bev_features)

        return self.bbox_head(fused)

    def forward(
        self,
        voxels: Float32[torch.Tensor, "num_voxels max_num_points num_point_features"] | None = None,
        num_points: Int32[torch.Tensor, " num_voxels"] | None = None,
        voxel_coords: Int32[torch.Tensor, "num_voxels 4"] | None = None,
        img: Sequence[torch.Tensor] | None = None,
        points: Sequence[torch.Tensor] | None = None,
        lidar2img: Sequence[torch.Tensor] | None = None,
        camera_intrinsics: Sequence[torch.Tensor] | None = None,
        lidar2cam: Sequence[torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Run the detector on lidar, image, or fused BEV inputs.

        Args:
            voxels: Optional lidar voxel tensor.
            num_points: Optional number of points per voxel.
            voxel_coords: Optional voxel coordinates with batch indices.
            img: Optional multiview image tensors.
            points: Optional per-sample lidar points for depth guidance.
            lidar2img: Optional lidar-to-image projection matrices.
            camera_intrinsics: Optional camera intrinsic matrices.
            lidar2cam: Optional lidar-to-camera extrinsics.
            **kwargs: Additional arguments forwarded to the shared forward path.

        Returns:
            Detection head outputs.
        """
        return self._forward_with_batch_size(
            voxels=voxels,
            num_points=num_points,
            voxel_coords=voxel_coords,
            img=img,
            points=points,
            lidar2img=lidar2img,
            camera_intrinsics=camera_intrinsics,
            lidar2cam=lidar2cam,
            **kwargs,
        )

    def compute_metrics(
        self,
        batch_inputs_dict: dict[str, Any],
        outputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute BEVFusion training losses.

        Args:
            batch_inputs_dict: Full batch dictionary.
            outputs: Detection head outputs.

        Returns:
            Loss dictionary produced by the detection head.
        """
        return self.bbox_head.loss(
            outputs, batch_inputs_dict["gt_boxes"], batch_inputs_dict["gt_labels"]
        )

    def predict_outputs(
        self, batch_inputs_dict: dict[str, Any], outputs: dict[str, torch.Tensor]
    ) -> Any:
        """Decode predictions for inference.

        Args:
            batch_inputs_dict: Full batch dictionary.
            outputs: Detection head outputs.

        Returns:
            Decoded prediction results.
        """
        del batch_inputs_dict
        return self.bbox_head.predict(outputs)

    def build_eval_output(self, batch: Mapping[str, Any], outputs: Any) -> dict[str, Any]:
        """Decode detections and pair them with ground truth for metrics."""
        return detection_eval_output(self.bbox_head.predict(outputs), batch)

    def get_log_batch_size(self, batch_inputs_dict: dict[str, Any]) -> int | None:
        """Log the sample count for fusion detection batches."""
        if "gt_boxes" in batch_inputs_dict:
            return len(batch_inputs_dict["gt_boxes"])
        if "img" in batch_inputs_dict:
            return len(batch_inputs_dict["img"])
        if "points" in batch_inputs_dict:
            return len(batch_inputs_dict["points"])
        return super().get_log_batch_size(batch_inputs_dict)

    def _prepare_export_model(self) -> "BEVFusionDetectionModel":
        """Return an export-ready model copy with exportable submodules.

        Returns:
            Deep copy of the model with the sparse middle encoder and the
            detection head replaced by their ONNX-exportable variants.
        """
        model = deepcopy(self).eval()
        if model.lidar is not None:
            model.lidar.prepare_for_export()
        if hasattr(model.bbox_head, "prepare_for_export"):
            model.bbox_head = model.bbox_head.prepare_for_export()
        return model

    def build_export_specs(self, batch_inputs_dict: dict[str, Any]) -> dict[str, ExportSpec]:
        """Build the ONNX export specifications for the runtime-compatible ABI.

        Lidar-only models export one ``bevfusion_lidar`` main body. Camera-lidar
        models export the ``bevfusion_image_backbone`` (raw ``uint8`` images to
        neck features) and the ``bevfusion_camera_lidar`` main body consuming
        those features together with precomputed BEV-pool metadata.

        Args:
            batch_inputs_dict: Batched model inputs used to derive export tensors.

        Returns:
            Ordered mapping of module name to export specification.
        """
        voxels, coors, num_points_per_voxel = BEVFusionLidar.first_sample_voxel_inputs(
            batch_inputs_dict
        )
        export_model = self._prepare_export_model()

        if self.camera is None:
            return {
                "bevfusion_lidar": ExportSpec(
                    module=BEVFusionLidarExportWrapper(export_model),
                    args=(voxels, coors, num_points_per_voxel),
                    input_param_names=["voxels", "coors", "num_points_per_voxel"],
                )
            }

        img = torch.stack(batch_inputs_dict["img"], dim=0).float()[:1]
        camera_intrinsics = torch.stack(batch_inputs_dict["camera_intrinsics"], dim=0).float()[:1]
        lidar2cam = torch.stack(batch_inputs_dict["lidar2cam"], dim=0).float()[:1]
        lidar2img = torch.stack(batch_inputs_dict["lidar2img"], dim=0).float()[:1]
        img_aug_matrix = torch.stack(batch_inputs_dict["img_aug_matrix"], dim=0).float()[:1]
        points = batch_inputs_dict["points"][0].float()

        # The pipeline bakes the image augmentation into lidar2img; the
        # runtime provides the raw projection and augmentation separately, so
        # split them back apart for the export sample.
        lidar2image_raw = torch.inverse(img_aug_matrix).matmul(lidar2img)

        imgs_uint8 = (img[0] * 255.0).round().clamp(0.0, 255.0).to(torch.uint8)
        image_feats = self.camera.extract_image_features(img)[0]

        geom_feats, kept, ranks, indices = self.camera.build_export_geometry(
            camera_intrinsics, lidar2cam
        )

        return {
            "bevfusion_image_backbone": ExportSpec(
                module=BEVFusionImageBackboneExportWrapper(export_model.camera),
                args=(imgs_uint8,),
                input_param_names=["imgs"],
            ),
            "bevfusion_camera_lidar": ExportSpec(
                module=_BEVFusionExportWrapper(export_model),
                args=(
                    voxels,
                    coors,
                    num_points_per_voxel,
                    points,
                    lidar2image_raw[0],
                    img_aug_matrix[0],
                    geom_feats.float(),
                    kept.bool(),
                    ranks.long(),
                    indices.long(),
                    image_feats,
                ),
                input_param_names=[
                    "voxels",
                    "coors",
                    "num_points_per_voxel",
                    "points",
                    "lidar2image",
                    "img_aug_matrix",
                    "geom_feats",
                    "kept",
                    "ranks",
                    "indices",
                    "image_feats",
                ],
            ),
        }
