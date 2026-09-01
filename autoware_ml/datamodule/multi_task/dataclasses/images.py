from __future__ import annotations

from typing import Sequence, NamedTuple

from jaxtyping import Float32
import torch
from torch import Tensor

from autoware_ml.geometry.cameras.base_images import BaseImages


class ImageGTBatch(NamedTuple):
    """Named tuple to represent pointcloud features in a batch size with their batch indices."""

    images: Float32[Tensor, "batch_size num_cameras num_channels height width"]
    camera_intrinsics: Float32[Tensor, "batch_size num_cameras 3 3"]
    augmented_camera_intrinsics: Float32[Tensor, "batch_size num_cameras 3 3"]
    lidar2images: Float32[Tensor, "batch_size num_cameras 4 4"]
    lidar2cams: Float32[Tensor, "batch_size num_cameras 4 4"]

    @staticmethod
    def collate_gt_samples(
        images_gt_samples: Sequence[BaseImages],
    ) -> ImageGTBatch | None:
        """
        Collate a sequence of points (BasePoints) into a single PointCloudGTBatch.

        Args:
          Images_gt_samples: Sequence of images (BaseImage) to be collated.

        Returns:
          ImageGTBatch: Collated point cloud GT batch.
        """
        if len(images_gt_samples) == 0:
            return None

        # Concatenate all images from the sequence of BaseImages
        images = torch.cat([sample.images for sample in images_gt_samples], dim=0)
        camera_intrinsics = torch.cat(
            [sample.camera_intrinsics for sample in images_gt_samples], dim=0
        )
        lidar2images = torch.cat([sample.lidar2images for sample in images_gt_samples], dim=0)
        lidar2cams = torch.cat([sample.lidar2cams for sample in images_gt_samples], dim=0)
        augmented_camera_intrinsics = torch.cat(
            [sample.augmented_camera_intrinsics for sample in images_gt_samples], dim=0
        )

        return ImageGTBatch(
            images=images,
            camera_intrinsics=camera_intrinsics,
            augmented_camera_intrinsics=augmented_camera_intrinsics,
            lidar2cams=lidar2cams,
            lidar2images=lidar2images,
        )

    def to_device(self, device: torch.device) -> ImageGTBatch:
        """
        Move the ImageGtBatch to the specified device.

        Args:
          device: The target device to move the batch to.

        Returns:
          ImageGtBatch: The batch moved to the specified device.
        """
        return ImageGTBatch(
            images=self.images.to(device),
            camera_intrinsics=self.camera_intrinsics.to(device),
            augmented_camera_intrinsics=self.augmented_camera_intrinsics.to(device),
            lidar2cams=self.lidar2cams.to(device),
            lidar2images=self.lidar2images.to(device),
        )
