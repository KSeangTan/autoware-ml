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

"""Camera branch of the native BEVFusion detector."""

from __future__ import annotations

from collections.abc import Sequence

from jaxtyping import Float32
import torch
import torch.nn as nn


class BEVFusionCamera(nn.Module):
    """Encode multiview images into the BEV feature map BEVFusion fuses.

    The branch owns the whole camera path: the image backbone and neck that encode the multiview
    images, and the view transform that lifts the resulting features into BEV space. That mirrors
    the lidar branch, which carries its own encoders, backbone and neck, so each modality is
    complete before the two are fused.
    """

    def __init__(
        self,
        img_backbone: nn.Module,
        img_neck: nn.Module,
        view_transform: nn.Module,
    ) -> None:
        """Initialize the camera branch.

        Args:
            img_backbone: Image backbone.
            img_neck: Image neck applied after the backbone.
            view_transform: View transform lifting image features into BEV.
        """
        super().__init__()
        self.img_backbone = img_backbone
        self.img_neck = img_neck
        self.view_transform = view_transform

    def extract_image_features(
        self, image_batch: Float32[torch.Tensor, "batch_size num_cams 3 height width"]
    ) -> Float32[torch.Tensor, "batch_size num_cams channels feature_height feature_width"]:
        """Encode multiview images into the neck features expected by the view transform.

        Args:
            image_batch: Image batch with shape ``(B, N, C, H, W)``.

        Returns:
            Neck feature tensor consumed by the view transform.
        """
        batch_size, num_cams, channels, image_height, image_width = image_batch.shape[:2]
        flat_images = image_batch.view(batch_size * num_cams, channels, image_height, image_width)
        image_features = self.img_backbone(flat_images)
        if isinstance(image_features, torch.Tensor):
            image_features = (image_features,)
        image_features = self.img_neck(image_features)
        primary_feature = (
            image_features[0] if isinstance(image_features, (list, tuple)) else image_features
        )
        _, primary_feature_channels, primary_feature_height, primary_feature_width = (
            primary_feature.shape
        )
        return primary_feature.view(
            batch_size,
            num_cams,
            primary_feature_channels,
            primary_feature_height,
            primary_feature_width,
        )

    def forward(
        self,
        image_batch: Float32[torch.Tensor, "batch_size num_cams 3 height width"],
        points: Sequence[Float32[torch.Tensor, ""]],
        lidar2img: Sequence[torch.Tensor],
        camera_intrinsics: Sequence[torch.Tensor],
        lidar2cam: Sequence[torch.Tensor],
        geom_feats_precomputed: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        | None = None,
        image_feature: torch.Tensor | None = None,
        img_aug_matrix: torch.Tensor | None = None,
    ) -> Float32[torch.Tensor, "batch_size channels height width"]:
        """Encode multiview images into a BEV feature map.

        Args:
            img: Multiview image tensors.
            points: Per-sample lidar points used for depth guidance.
            lidar2img: Lidar-to-image projection matrices. The training pipeline bakes image
                augmentation into these, so the default identity ``img_aug_matrix`` keeps the
                projection consistent.
            camera_intrinsics: Camera intrinsic matrices.
            lidar2cam: Lidar-to-camera extrinsics expressed in the lidar frame the BEV grid
                is built in. The pipeline composes the inverse of the lidar augmentation into
                these, so their inverse maps the camera into the augmented lidar frame.
            geom_feats_precomputed: Optional precomputed BEV-pool metadata.
            image_feature: Optional precomputed image feature tensor.
            img_aug_matrix: Optional image augmentation matrices.

        Returns:
            Image BEV feature map.
        """
        if image_feature is None:
            image_feature = self.extract_image_features(img)
        batch_size, num_cams = image_feature.shape[:2]

        intrinsics = (
            torch.stack(list(camera_intrinsics), dim=0).float()
            if isinstance(camera_intrinsics, (list, tuple))
            else camera_intrinsics.float()
        )
        lidar2cam_tensor = (
            torch.stack(list(lidar2cam), dim=0).float()
            if isinstance(lidar2cam, (list, tuple))
            else lidar2cam.float()
        )
        lidar2image = (
            torch.stack(list(lidar2img), dim=0).float()
            if isinstance(lidar2img, (list, tuple))
            else lidar2img.float()
        )
        camera2aug_lidar = torch.inverse(lidar2cam_tensor)
        if img_aug_matrix is None:
            img_aug_matrix = (
                torch.eye(4, device=image_feature.device)
                .view(1, 1, 4, 4)
                .repeat(batch_size, num_cams, 1, 1)
            )
        return self.view_transform(
            image_feature,
            points,
            lidar2image,
            intrinsics,
            camera2aug_lidar,
            img_aug_matrix,
            geom_feats_precomputed=geom_feats_precomputed,
        )

    def forward_export(
        self,
        points: torch.Tensor,
        lidar2image: torch.Tensor,
        img_aug_matrix: torch.Tensor,
        geom_feats: torch.Tensor,
        kept: torch.Tensor,
        ranks: torch.Tensor,
        indices: torch.Tensor,
        image_feats: torch.Tensor,
    ) -> Float32[torch.Tensor, "1 channels height width"]:
        """Encode a single sample into a BEV feature map with precomputed BEV-pool metadata.

        The exported main body is a single-sample graph, so the runtime passes the camera inputs
        without a batch dimension; they are unsqueezed back into the batched layout the view
        transform expects.

        Args:
            points: Raw point tensor used for lidar depth guidance.
            lidar2image: Raw lidar-to-image projection matrices.
            img_aug_matrix: Image augmentation matrices.
            geom_feats: Precomputed BEV-pool geometry features.
            kept: Keep mask for pooled features.
            ranks: Sorted BEV ranks.
            indices: Sorting indices aligned with ``ranks``.
            image_feats: Precomputed image features.

        Returns:
            Image BEV feature map for the single exported sample.
        """
        return self.view_transform.forward_precomputed(
            image_feats.unsqueeze(0),
            [points],
            lidar2image.unsqueeze(0),
            img_aug_matrix.unsqueeze(0),
            geom_feats.long(),
            kept,
            ranks,
            indices,
        )

    def build_export_geometry(
        self,
        camera_intrinsics: Float32[torch.Tensor, "1 num_cams 4 4"],
        lidar2cam: Float32[torch.Tensor, "1 num_cams 4 4"],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Precompute the BEV-pool metadata baked into the exported graph.

        The runtime resolves the pooling geometry once from the calibration, so the export sample
        uses an identity image augmentation to keep the metadata free of training-time augmentation.

        Args:
            camera_intrinsics: Single-sample camera intrinsic matrices.
            lidar2cam: Single-sample lidar-to-camera extrinsics.

        Returns:
            Tuple of geometry features, keep mask, sorted ranks, and sorting indices.
        """
        device = camera_intrinsics.device
        num_cams = camera_intrinsics.shape[1]
        camera2aug_lidar = torch.inverse(lidar2cam)
        identity_img_aug = torch.eye(4, device=device).view(1, 1, 4, 4).repeat(1, num_cams, 1, 1)
        geom = self.view_transform.camera_to_lidar_geometry(
            camera2aug_lidar,
            camera_intrinsics,
            identity_img_aug,
        )
        return self.view_transform.bev_pool_aux(geom)


class BEVFusionImageBackboneExportWrapper(nn.Module):
    """Wrap the multiview image backbone export.

    The wrapper consumes raw ``uint8`` images and bakes the training-time ``1 / 255``
    normalization into the graph.
    """

    def __init__(self, camera: BEVFusionCamera) -> None:
        """Initialize the image backbone export wrapper.

        Args:
            camera: BEVFusion camera branch instance.
        """
        super().__init__()
        self.camera = camera

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """Encode raw multiview images into neck features.

        Args:
            imgs: Raw multiview images of shape ``(N, 3, H, W)`` with ``uint8`` values.

        Returns:
            Image neck features of shape ``(N, C, fH, fW)``.
        """
        images = imgs.float() / 255.0
        return self.camera.extract_image_features(images.unsqueeze(0)).squeeze(0)
