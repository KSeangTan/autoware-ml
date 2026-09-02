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

"""Unit tests for the NormalizeMultiviewImage transform."""

import unittest

from jaxtyping import Float32
from pydantic import ValidationError
import torch
from torch import Tensor

from autoware_ml.datamodule.multi_task.dataclasses.multi_task_samples import MultiTaskGTSample
from autoware_ml.geometry.cameras.base_images import BaseImages
from autoware_ml.transforms.camera.normalize import NormalizeMultiviewImage


class TestNormalizeMultiviewImage(unittest.TestCase):
    """Unit tests for the NormalizeMultiviewImage transform."""

    def build_multi_task_gt_sample(
        self,
        images: Float32[Tensor, "num_cameras num_channels height width"],
    ) -> MultiTaskGTSample:
        """Build a minimal sample holding only camera image data.

        Args:
            images: Images the sample holds.

        Returns:
            MultiTaskGTSample holding the given images and identity camera matrices.
        """
        num_cameras = images.shape[0]
        camera_image_data = BaseImages(
            images=images,
            timestamps=torch.zeros(num_cameras, dtype=torch.float32),
            camera_intrinsics=torch.eye(3).repeat(num_cameras, 1, 1),
            camera_names=[f"camera{index}" for index in range(num_cameras)],
            lidar2images=torch.eye(4).repeat(num_cameras, 1, 1),
            lidar2cams=torch.eye(4).repeat(num_cameras, 1, 1),
            distortion_models=["plumb_bob"] * num_cameras,
            distortion_coefficients=[torch.zeros(5) for _ in range(num_cameras)],
            augmented_camera_intrinsics=torch.eye(3).repeat(num_cameras, 1, 1),
            image_augmentation_matrices=torch.eye(3).repeat(num_cameras, 1, 1),
        )
        return MultiTaskGTSample(
            lidar_point_cloud_samples=None,
            image_samples=None,
            point_cloud_data=None,
            camera_image_data=camera_image_data,
            detection3d_gt_bboxes_3d=None,
            segmentation3d_gt_sample=None,
        )

    def test_handles_chw_stack(self) -> None:
        """Test that the leading num_cameras dimension is broadcast over."""
        images = torch.ones((2, 3, 4, 5), dtype=torch.float32)

        output = NormalizeMultiviewImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(
            self.build_multi_task_gt_sample(images)
        )

        assert output.camera_image_data is not None
        normalized_images = output.camera_image_data.images
        self.assertEqual(normalized_images.shape, (2, 3, 4, 5))
        # (1.0 - 0.5) / 0.5 for every pixel of every camera.
        self.assertTrue(torch.allclose(normalized_images, torch.ones_like(images)))

    def test_normalizes_per_channel(self) -> None:
        """Test that each channel is normalized with its own mean and standard deviation."""
        images = torch.ones((1, 3, 2, 2), dtype=torch.float32)
        mean = [0.0, 0.5, 1.0]
        std = [1.0, 0.5, 2.0]

        output = NormalizeMultiviewImage(mean=mean, std=std)(
            self.build_multi_task_gt_sample(images)
        )

        assert output.camera_image_data is not None
        normalized_images = output.camera_image_data.images
        expected_channel_values = [
            (1.0 - channel_mean) / channel_std for channel_mean, channel_std in zip(mean, std)
        ]
        for channel, expected_value in enumerate(expected_channel_values):
            self.assertTrue(
                torch.allclose(
                    normalized_images[0, channel],
                    torch.full((2, 2), expected_value, dtype=torch.float32),
                )
            )

    def test_keeps_float32(self) -> None:
        """Test that the normalized images keep the float32 dtype BaseImages requires."""
        images = torch.full((1, 3, 2, 2), 255.0, dtype=torch.float32)

        output = NormalizeMultiviewImage(mean=[0.0, 0.0, 0.0], std=[255.0, 255.0, 255.0])(
            self.build_multi_task_gt_sample(images)
        )

        assert output.camera_image_data is not None
        normalized_images = output.camera_image_data.images
        self.assertEqual(normalized_images.dtype, torch.float32)
        self.assertTrue(torch.allclose(normalized_images, torch.ones_like(normalized_images)))

    def test_rejects_non_float32_images(self) -> None:
        """Test that BaseImages refuses images that are not float32."""
        with self.assertRaises(ValidationError):
            self.build_multi_task_gt_sample(torch.full((1, 3, 2, 2), 255, dtype=torch.uint8))

    def test_geometry_preserved(self) -> None:
        """Test that only pixel values change, geometric attributes are carried over."""
        sample = self.build_multi_task_gt_sample(torch.ones((2, 3, 4, 5), dtype=torch.float32))
        assert sample.camera_image_data is not None

        output = NormalizeMultiviewImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(sample)

        assert output.camera_image_data is not None
        camera_image_data = output.camera_image_data
        self.assertEqual(camera_image_data.camera_names, sample.camera_image_data.camera_names)
        self.assertEqual(
            camera_image_data.distortion_models, sample.camera_image_data.distortion_models
        )
        self.assertTrue(
            torch.equal(
                camera_image_data.camera_intrinsics,
                sample.camera_image_data.camera_intrinsics,
            )
        )
        self.assertTrue(
            torch.equal(
                camera_image_data.augmented_camera_intrinsics,
                sample.camera_image_data.augmented_camera_intrinsics,
            )
        )
        self.assertTrue(
            torch.equal(camera_image_data.lidar2images, sample.camera_image_data.lidar2images)
        )
        self.assertTrue(
            torch.equal(camera_image_data.lidar2cams, sample.camera_image_data.lidar2cams)
        )
        self.assertTrue(
            torch.equal(
                camera_image_data.image_augmentation_matrices,
                sample.camera_image_data.image_augmentation_matrices,
            )
        )

    def test_missing_camera_image_data_key(self) -> None:
        """Test that missing 'camera_image_data' raises KeyError."""
        sample = self.build_multi_task_gt_sample(
            torch.ones((1, 3, 4, 5), dtype=torch.float32)
        )._replace(camera_image_data=None)

        with self.assertRaises(KeyError):
            NormalizeMultiviewImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(sample)


if __name__ == "__main__":
    unittest.main()
