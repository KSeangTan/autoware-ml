"""Camera normalization transforms."""

from __future__ import annotations

from typing import Sequence

import torch
from torchvision.transforms import v2

from autoware_ml.geometry.cameras.base_images import BaseImages
from autoware_ml.transforms.multi_task.base import MultiTaskBaseTransform
from autoware_ml.datamodule.multi_task.dataclasses.multi_task_samples import MultiTaskGTSample


class NormalizeMultiviewImage(MultiTaskBaseTransform):
    """Normalize multiview images channel-wise."""

    _required_keys = ["camera_image_data"]

    def __init__(
        self,
        mean: Sequence[float],
        std: Sequence[float],
    ) -> None:
        """Initialize the NormalizeMultiviewImage transform.

        Args:
            mean: Per-channel mean subtracted from each image.
            std: Per-channel standard deviation used for scaling.
            probability: Probability of applying the transform, None to always run.
        """
        super().__init__(probability=None)
        self.mean = mean
        self.std = std
        self._normalize = v2.Normalize(mean=list(mean), std=list(std))

    def transform(self, multi_task_gt_sample: MultiTaskGTSample) -> MultiTaskGTSample:
        """Normalize every camera image channel-wise.

        Args:
            multi_task_gt_sample: MultiTaskGTSample instance containing `camera_image_data`.

        Returns:
            Updated MultiTaskGTSample instance with updated `camera_image_data`.
        """
        assert multi_task_gt_sample.camera_image_data is not None
        camera_image_data = multi_task_gt_sample.camera_image_data

        # v2.Normalize expects float (..., C, H, W), so the leading num_cameras
        # dimension is broadcast over automatically.
        images = self._normalize(camera_image_data.images.to(torch.float32))

        normalized_camera_image_data = BaseImages(
            images=images,
            timestamps=camera_image_data.timestamps,
            camera_intrinsics=camera_image_data.camera_intrinsics,
            camera_names=camera_image_data.camera_names,
            lidar2images=camera_image_data.lidar2images,
            lidar2cams=camera_image_data.lidar2cams,
            augmented_camera_intrinsics=camera_image_data.augmented_camera_intrinsics,
            noises=camera_image_data.noises,
            distortion_models=camera_image_data.distortion_models,
            distortion_coefficients=camera_image_data.distortion_coefficients,
        )
        return multi_task_gt_sample._replace(camera_image_data=normalized_camera_image_data)
