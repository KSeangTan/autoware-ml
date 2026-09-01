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

"""Unit tests for the GridMask transform."""

import unittest

from jaxtyping import Float32
import numpy as np
import torch
from torch import Tensor

from autoware_ml.datamodule.multi_task.dataclasses.multi_task_samples import MultiTaskGTSample
from autoware_ml.geometry.cameras.base_images import BaseImages
from autoware_ml.transforms.camera.masking import GridMask


class TestGridMask(unittest.TestCase):
    """Unit tests for the GridMask transform."""

    def setUp(self) -> None:
        """Seed the mask sampling so every test draws the same grid."""
        np.random.seed(0)

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
        )
        return MultiTaskGTSample(
            lidar_point_cloud_samples=None,
            image_samples=None,
            point_cloud_data=None,
            camera_image_data=camera_image_data,
            detection3d_gt_bboxes_3d=None,
            segmentation3d_gt_sample=None,
        )

    def test_masks_every_camera(self) -> None:
        """Test that each camera image is masked while its shape is preserved."""
        images = torch.ones((2, 3, 64, 64), dtype=torch.float32)
        sample = self.build_multi_task_gt_sample(images)

        output = GridMask(probability=1.0, ratio=0.5, rotate=0)(sample)

        assert output.camera_image_data is not None
        masked_images = output.camera_image_data.images
        self.assertEqual(masked_images.shape, images.shape)
        for masked_image in masked_images:
            self.assertTrue((masked_image == 0).any())
            self.assertTrue((masked_image == 1).any())

    def test_masks_channels_consistently(self) -> None:
        """Test that the same mask is applied to every channel of an image."""
        images = torch.ones((1, 3, 64, 64), dtype=torch.float32)

        output = GridMask(probability=1.0, ratio=0.5, rotate=0)(
            self.build_multi_task_gt_sample(images)
        )

        assert output.camera_image_data is not None
        masked_image = output.camera_image_data.images[0]
        self.assertTrue(torch.equal(masked_image[0], masked_image[1]))
        self.assertTrue(torch.equal(masked_image[0], masked_image[2]))

    def test_geometry_preserved(self) -> None:
        """Test that only pixel values change, geometric attributes are carried over."""
        sample = self.build_multi_task_gt_sample(torch.ones((2, 3, 64, 64), dtype=torch.float32))
        assert sample.camera_image_data is not None

        output = GridMask(probability=1.0, ratio=0.5, rotate=1)(sample)

        assert output.camera_image_data is not None
        camera_image_data = output.camera_image_data
        self.assertEqual(camera_image_data.camera_names, sample.camera_image_data.camera_names)
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

    def test_zero_probability_skips(self) -> None:
        """Test that a zero probability returns the sample unchanged."""
        sample = self.build_multi_task_gt_sample(torch.ones((2, 3, 64, 64), dtype=torch.float32))

        output = GridMask(probability=0.0)(sample)

        self.assertIs(output, sample)

    def test_missing_camera_image_data_key(self) -> None:
        """Test that missing 'camera_image_data' raises KeyError."""
        sample = self.build_multi_task_gt_sample(
            torch.ones((1, 3, 64, 64), dtype=torch.float32)
        )._replace(camera_image_data=None)

        with self.assertRaises(KeyError):
            GridMask(probability=1.0)(sample)


if __name__ == "__main__":
    unittest.main()
