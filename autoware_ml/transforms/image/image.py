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

"""Image-specific transforms.

This module contains reusable image-domain augmentations and preprocessing
transforms used by detection and fusion models.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

from autoware_ml.datamodule.multi_task.dataclasses.multi_task_samples import MultiTaskGTSample
from autoware_ml.geometry.cameras.base_images import BaseImages
from autoware_ml.transforms.multi_task.base import MultiTaskBaseTransform


class PhotometricDistortion(MultiTaskBaseTransform):
    """Apply random brightness, contrast, saturation, and hue to RGB channels.

    Operates on the camera images (num_cameras, 3, H, W). Assumes float32 [0, 255] input in
    RGB format, as produced by the loading transforms. The images are handed to OpenCV as
    uint8 (H, W, 3) arrays and converted back to a float32 tensor at the end. One set of
    distortions is sampled per call and applied to every camera.

    Required keys:
        - camera_image_data: (num_cameras, 3, H, W) float32 RGB images in [0, 255].

    Optional keys:
        - None

    Generated keys:
        - camera_image_data: With the photometrically distorted images (when applied).
    """

    _required_keys = ["camera_image_data"]

    def __init__(
        self,
        probability: float | None = 0.5,
        brightness: float = 0.0,
        contrast: float = 0.0,
        saturation: float = 0.0,
        hue: float = 0.0,
    ) -> None:
        """Initialize the PhotometricDistortion transform.

        Args:
            probability: Probability of applying the transform, None to always run.
            brightness: Max brightness deviation [0, 1].
            contrast: Max contrast deviation [0, 1].
            saturation: Max saturation deviation [0, 1].
            hue: Max hue deviation [0, 0.5].
        """
        super().__init__(probability=probability)
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def transform(self, multi_task_gt_sample: MultiTaskGTSample) -> MultiTaskGTSample:
        """Apply photometric distortion to the RGB channels of every camera image.

        Args:
            multi_task_gt_sample: MultiTaskGTSample instance containing `camera_image_data`.

        Returns:
            Updated MultiTaskGTSample instance with distorted `camera_image_data`.
        """
        assert multi_task_gt_sample.camera_image_data is not None
        camera_image_data = multi_task_gt_sample.camera_image_data
        images = camera_image_data.images

        # TODO(Kok Seang): Consider to implement torch version to make it faster
        # Convert from torch (num_cameras, 3, H, W) float32 to cv2 (num_cameras, H, W, 3) uint8
        opencv_images = (
            images.round()
            .clamp(0, 255)
            .to(torch.uint8)
            .permute(0, 2, 3, 1)
            .contiguous()
            .cpu()
            .numpy()
        )

        # Sample the distortions once so every camera of the sample is distorted the same way
        brightness_factor = (
            np.random.uniform(1 - self.brightness, 1 + self.brightness)
            if self.brightness > 0
            else 1.0
        )
        saturation_factor = (
            np.random.uniform(1 - self.saturation, 1 + self.saturation)
            if self.saturation > 0
            else 1.0
        )
        contrast_factor = (
            np.random.uniform(1 - self.contrast, 1 + self.contrast) if self.contrast > 0 else 1.0
        )
        hue_shift = np.random.uniform(-self.hue, self.hue) * 179.0 if self.hue > 0 else 0.0

        augmented_images = []
        for img in opencv_images:
            # img is (H, W, 3) uint8
            # Convert to HSV
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)

            # Apply distortions
            if self.brightness > 0:
                hsv[..., 2] *= brightness_factor

            if self.saturation > 0:
                hsv[..., 1] *= saturation_factor

            if self.contrast > 0:
                # Simple contrast: scale V around 127.5
                hsv[..., 2] = (hsv[..., 2] - 127.5) * contrast_factor + 127.5

            if self.hue > 0:
                hsv[..., 0] += hue_shift
                hsv[..., 0] = np.mod(hsv[..., 0], 180.0)

            # Clip and convert back
            hsv = np.clip(hsv, 0, 255).astype(np.uint8)
            augmented_images.append(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))

        # Convert from cv2 (num_cameras, H, W, 3) uint8 back to torch (num_cameras, 3, H, W) float32
        distorted_images = (
            torch.from_numpy(np.stack(augmented_images, axis=0))
            .to(images.device)
            .permute(0, 3, 1, 2)
            .to(torch.float32)
        )

        # Only the pixel values change, every other field is carried over unchanged.
        # model_copy does not validate what it is given, so the copy is validated explicitly.
        distorted_camera_image_data = BaseImages.model_validate(
            camera_image_data.model_copy(update={"images": distorted_images})
        )
        return multi_task_gt_sample._replace(camera_image_data=distorted_camera_image_data)
