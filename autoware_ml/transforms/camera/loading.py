"""Camera loading transforms."""

from __future__ import annotations

from typing import Sequence

import torch
from torchvision.io import decode_image

from autoware_ml.geometry.cameras.base_images import BaseImages
from autoware_ml.transforms.multi_task.base import MultiTaskBaseTransform
from autoware_ml.datamodule.multi_task.dataclasses.multi_task_samples import (
    ImageSample,
    MultiTaskGTSample,
)
from autoware_ml.types.geometry import ImageChannel


class LoadImageFromFile(MultiTaskBaseTransform):
    """Load one RGB image from a metadata path."""

    _required_keys = ["image_samples"]

    def __init__(self, color_type: ImageChannel, normalize_to_unit: bool) -> None:
        """Initialize the LoadImageFromFile transform.

        Args:
            color_type: Output color format, only rgb is supported now.
            normalize_to_unit: Whether to divide pixel values by ``255``.
        """
        super().__init__(probability=None)
        self.color_type = color_type
        self.normalize_to_unit = normalize_to_unit

    def transform(self, multi_task_gt_sample: MultiTaskGTSample) -> MultiTaskGTSample:
        """Load only the first image data from the current sample.

        Args:
            multi_task_gt_sample: MultiTaskGTSample instance containing `image_samples`.

        Returns:
            Updated MultiTaskGTSample instance with a loaded `camera_image_data`.
        """
        assert multi_task_gt_sample.image_samples is not None
        image_sample = multi_task_gt_sample.image_samples[0]
        decoded_image = decode_image(
            image_sample.image_path,
            mode=self.color_type.value,  # type: ignore
        )
        if self.normalize_to_unit:
            decoded_image = decoded_image / 255.0
        # (num_cameras, num_channels, height, width)
        decoded_image = decoded_image.view(
            1, decoded_image.shape[0], decoded_image.shape[1], decoded_image.shape[2]
        ).to(torch.float32)
        # (num_cameras)
        timestamp = torch.tensor([image_sample.timestamp], dtype=torch.float32)

        camera_image_data = BaseImages(
            images=decoded_image,
            timestamps=timestamp,
            # The leading num_cameras dimension is kept so every field stays indexable
            # per camera by the downstream transforms.
            camera_intrinsics=image_sample.camera_intrinsic.unsqueeze(0),
            lidar2images=image_sample.lidar2image.unsqueeze(0),
            lidar2cams=image_sample.lidar2cam.unsqueeze(0),
            camera_names=[image_sample.camera_name],
            distortion_models=[image_sample.distortion_model],
            distortion_coefficients=[image_sample.distortion_coefficients],
            noises=None,  # Initially, set to None
            augmented_camera_intrinsics=None,  # Initially, set to None
            image_augmentation_matrices=None,  # Initially, set to None
        )
        return multi_task_gt_sample._replace(camera_image_data=camera_image_data)


class LoadMultiViewImagesFromFiles(MultiTaskBaseTransform):
    """Load synchronized multiview images and camera matrices."""

    _required_keys = ["image_samples"]

    def __init__(
        self, normalize_to_unit: bool, color_type: ImageChannel, camera_order: Sequence[str]
    ) -> None:
        """Initialize the LoadMultiViewImagesFromFiles transform.

        Args:
            normalize_to_unit: Whether to divide pixel values by ``255``.
            color_type: Output color format, only rgb is supported now.
            camera_order: Loading order for each camera.
        """
        super().__init__(probability=None)
        self.normalize_to_unit = normalize_to_unit
        self.color_type = color_type
        self.camera_order = camera_order

    def transform(self, multi_task_gt_sample: MultiTaskGTSample) -> MultiTaskGTSample:
        """Load only the first image data from the current sample.

        Args:
            multi_task_gt_sample: MultiTaskGTSample instance containing `image_samples`.

        Returns:
            Updated MultiTaskGTSample instance with a loaded `camera_image_data`.
        """
        assert multi_task_gt_sample.image_samples is not None
        # Reorder image_samples based on camera_order.
        image_samples_by_camera_name = {
            image_sample.camera_name: image_sample
            for image_sample in multi_task_gt_sample.image_samples
        }
        image_samples: Sequence[ImageSample] = []
        for camera_name in self.camera_order:
            if camera_name not in image_samples_by_camera_name:
                raise ValueError(
                    f"Missing camera_name: {camera_name} from the "
                    f"sample: {multi_task_gt_sample.image_samples}"
                )
            image_samples.append(image_samples_by_camera_name[camera_name])

        images = []
        camera_intrinsics = []
        lidar2cams = []
        lidar2images = []
        timestamps = []
        distortion_models = []
        distortion_coefficients = []
        for image_sample in image_samples:
            decoded_image = decode_image(
                image_sample.image_path,
                mode=self.color_type.value,  # type: ignore
            )

            decoded_image = decoded_image.to(torch.float32)
            if self.normalize_to_unit:
                decoded_image = decoded_image / 255.0

            images.append(decoded_image)
            camera_intrinsics.append(image_sample.camera_intrinsic)
            lidar2cams.append(image_sample.lidar2cam)
            lidar2images.append(image_sample.lidar2image)
            timestamps.append(image_sample.timestamp)
            distortion_models.append(image_sample.distortion_model)
            distortion_coefficients.append(image_sample.distortion_coefficients)

        camera_image_data = BaseImages(
            images=torch.stack(images, dim=0),
            camera_intrinsics=torch.stack(camera_intrinsics, dim=0),
            lidar2images=torch.stack(lidar2images, dim=0),
            lidar2cams=torch.stack(lidar2cams, dim=0),
            timestamps=torch.tensor(timestamps, dtype=torch.float32),
            camera_names=self.camera_order,
            distortion_models=distortion_models,
            distortion_coefficients=distortion_coefficients,
            noises=None,  # Initially set to Empty
            augmented_camera_intrinsics=None,  # Initially set to Empty
            image_augmentation_matrices=None,  # Initially set to Empty
        )
        return multi_task_gt_sample._replace(camera_image_data=camera_image_data)
