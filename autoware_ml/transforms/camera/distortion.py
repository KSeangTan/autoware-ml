"""Camera distortion correction transforms."""

from __future__ import annotations

import cv2
import numpy as np
import torch

from autoware_ml.geometry.cameras.base_images import BaseImages
from autoware_ml.transforms.multi_task.base import MultiTaskBaseTransform
from autoware_ml.datamodule.multi_task.dataclasses.multi_task_samples import MultiTaskGTSample


class UndistortImage(MultiTaskBaseTransform):
    """Undistort every camera image using its distortion coefficients.

    The distortion coefficients are expressed in the frame of the raw image, so this
    transform must run before any other image-space transform. Undistorted cameras have
    their entry in `augmented_camera_intrinsics` replaced by the optimal new camera matrix
    and their distortion coefficients zeroed, so applying the transform twice is a no-op.
    """

    _required_keys = ["camera_image_data"]

    def __init__(self, alpha: float = 0.0) -> None:
        """Initialize the UndistortImage transform.

        Args:
            alpha: Free scaling parameter passed to OpenCV undistortion. ``0.0``
                crops invalid pixels, while ``1.0`` retains the full field of view.
        """
        super().__init__(probability=None)
        self.alpha = alpha

    def transform(self, multi_task_gt_sample: MultiTaskGTSample) -> MultiTaskGTSample:
        """Undistort every camera image and update its intrinsics.

        Args:
            multi_task_gt_sample: MultiTaskGTSample instance containing `camera_image_data`.

        Returns:
            Updated MultiTaskGTSample instance with an undistorted `camera_image_data`.
        """
        assert multi_task_gt_sample.camera_image_data is not None
        camera_image_data = multi_task_gt_sample.camera_image_data

        images = camera_image_data.images
        camera_intrinsics = camera_image_data.camera_intrinsics
        augmented_camera_intrinsics = camera_image_data.augmented_camera_intrinsics.clone()

        undistorted_images = []
        undistorted_coefficients = []
        for index, coefficients in enumerate(camera_image_data.distortion_coefficients):
            # Nothing to correct for pre-undistorted cameras, keep the image as it is.
            if coefficients.numel() == 0 or not torch.any(coefficients):
                undistorted_images.append(images[index])
                undistorted_coefficients.append(coefficients)
                continue

            # OpenCV expects (height, width, num_channels) contiguous arrays.
            image = images[index].permute(1, 2, 0).contiguous().numpy()
            height, width = image.shape[:2]
            camera_matrix = camera_intrinsics[index].numpy().astype(np.float64)
            distortion_coefficients = coefficients.numpy().astype(np.float64)

            new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
                camera_matrix,
                distortion_coefficients,
                (width, height),
                self.alpha,
                (width, height),
            )
            image = cv2.undistort(
                image,
                camera_matrix,
                distortion_coefficients,
                newCameraMatrix=new_camera_matrix,
            )

            undistorted_images.append(torch.from_numpy(image).permute(2, 0, 1).to(images.dtype))
            augmented_camera_intrinsics[index] = torch.from_numpy(new_camera_matrix).to(
                augmented_camera_intrinsics.dtype
            )
            undistorted_coefficients.append(torch.zeros_like(coefficients))

        undistorted_camera_image_data = BaseImages(
            images=torch.stack(undistorted_images, dim=0),
            timestamps=camera_image_data.timestamps,
            camera_intrinsics=camera_intrinsics,
            camera_names=camera_image_data.camera_names,
            lidar2images=camera_image_data.lidar2images,
            lidar2cams=camera_image_data.lidar2cams,
            distortion_models=camera_image_data.distortion_models,
            distortion_coefficients=undistorted_coefficients,
            noises=camera_image_data.noises,
            augmented_camera_intrinsics=augmented_camera_intrinsics,
        )
        return multi_task_gt_sample._replace(camera_image_data=undistorted_camera_image_data)
