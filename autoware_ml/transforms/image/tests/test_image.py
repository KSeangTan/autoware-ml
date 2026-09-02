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

"""Unit tests for the image transforms."""

import unittest
from unittest import mock

from jaxtyping import Float32
import numpy as np
import torch
from torch import Tensor

from autoware_ml.datamodule.multi_task.dataclasses.multi_task_samples import MultiTaskGTSample
from autoware_ml.geometry.cameras.base_images import BaseImages
from autoware_ml.transforms.image.image import PhotometricDistortion


def solid_color_images(
    rgb: tuple[float, float, float], num_cameras: int = 1, height: int = 2, width: int = 3
) -> Float32[Tensor, "num_cameras 3 height width"]:
    """Build images filled with one RGB color.

    Args:
        rgb: The color of every pixel, in [0, 255].
        num_cameras: Number of cameras.
        height: Image height.
        width: Image width.

    Returns:
        Images of the given size filled with the color.
    """
    color = torch.tensor(rgb, dtype=torch.float32).view(1, 3, 1, 1)
    return color.expand(num_cameras, 3, height, width).clone()


class TestPhotometricDistortion(unittest.TestCase):
    """Unit tests for the PhotometricDistortion transform."""

    def setUp(self) -> None:
        """Seed the random generators for reproducible sampling."""
        np.random.seed(0)
        torch.manual_seed(0)

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
            image_augmentation_matrices=torch.eye(4).repeat(num_cameras, 1, 1),
        )
        return MultiTaskGTSample(
            lidar_point_cloud_samples=None,
            image_samples=None,
            point_cloud_data=None,
            camera_image_data=camera_image_data,
            detection3d_gt_bboxes_3d=None,
            segmentation3d_gt_sample=None,
        )

    def build_random_images(self, num_cameras: int = 2) -> Float32[Tensor, "num_cameras 3 4 5"]:
        """Build random RGB images with whole pixel values in [0, 255].

        Args:
            num_cameras: Number of cameras.

        Returns:
            Random images that survive the uint8 round trip exactly.
        """
        return torch.randint(0, 256, (num_cameras, 3, 4, 5)).to(torch.float32)

    def distort(
        self,
        images: Float32[Tensor, "num_cameras 3 height width"],
        transform: PhotometricDistortion,
    ) -> Float32[Tensor, "num_cameras 3 height width"]:
        """Run the transform on images and return the distorted images.

        Args:
            images: The images to distort.
            transform: The transform to run.

        Returns:
            The distorted images.
        """
        output = transform(self.build_multi_task_gt_sample(images))
        assert output.camera_image_data is not None
        return output.camera_image_data.images

    def test_all_distortions_modify_the_images(self) -> None:
        """Test that enabling every distortion changes the pixels but keeps shape and dtype."""
        images = self.build_random_images()
        sample = self.build_multi_task_gt_sample(images)
        transform = PhotometricDistortion(
            probability=1.0, brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1
        )

        output = transform(sample)

        assert output.camera_image_data is not None
        assert sample.camera_image_data is not None
        distorted_images = output.camera_image_data.images
        self.assertEqual(distorted_images.shape, images.shape)
        self.assertEqual(distorted_images.dtype, torch.float32)
        self.assertFalse(torch.equal(distorted_images, images))
        # The output is a new container, the input is left untouched.
        self.assertIsNot(output.camera_image_data, sample.camera_image_data)
        self.assertTrue(torch.equal(sample.camera_image_data.images, images))

    def test_output_stays_in_uint8_range(self) -> None:
        """Test that strong distortions never push the pixels outside [0, 255]."""
        images = self.build_random_images()
        transform = PhotometricDistortion(probability=None, brightness=1.0, contrast=1.0)

        distorted_images = self.distort(images, transform)

        self.assertGreaterEqual(float(distorted_images.min()), 0.0)
        self.assertLessEqual(float(distorted_images.max()), 255.0)
        # Values come back from uint8, hence are whole numbers.
        self.assertTrue(torch.equal(distorted_images, distorted_images.round()))

    def test_no_distortion_keeps_the_images(self) -> None:
        """Test that a transform with every deviation at zero only round-trips through OpenCV."""
        images = self.build_random_images()

        distorted_images = self.distort(images, PhotometricDistortion(probability=None))

        # OpenCV quantizes the hue to [0, 180) and the saturation to uint8, so saturated colors
        # move by a few levels on the way back. Gray levels, which carry no hue, are exact.
        self.assertTrue(torch.allclose(distorted_images, images, atol=6.0))
        gray_images = solid_color_images((100.0, 100.0, 100.0))
        self.assertTrue(
            torch.equal(
                self.distort(gray_images, PhotometricDistortion(probability=None)), gray_images
            )
        )

    def test_fractional_pixels_are_rounded_and_clamped(self) -> None:
        """Test that non-integer pixel values are rounded to the nearest uint8 and clamped."""
        transform = PhotometricDistortion(probability=None)
        cases = {10.4: 10.0, 10.6: 11.0, 300.0: 255.0, -5.0: 0.0}

        for value, expected_value in cases.items():
            with self.subTest(value=value):
                gray_images = solid_color_images((value, value, value))
                distorted_images = self.distort(gray_images, transform)
                self.assertTrue(
                    torch.equal(
                        distorted_images,
                        solid_color_images((expected_value, expected_value, expected_value)),
                    )
                )

    def test_brightness_scales_the_pixels(self) -> None:
        """Test that the brightness factor scales every channel of a color."""
        images = solid_color_images((200.0, 100.0, 50.0))
        transform = PhotometricDistortion(probability=None, brightness=0.5)

        with mock.patch("autoware_ml.transforms.image.image.np.random.uniform", return_value=0.5):
            distorted_images = self.distort(images, transform)

        self.assertTrue(torch.allclose(distorted_images, images * 0.5, atol=1.0))

    def test_contrast_stretches_around_mid_gray(self) -> None:
        """Test that the contrast factor moves gray levels away from mid-gray."""
        # (191 - 127.5) * 2 + 127.5 = 254.5, which rounds up to the maximum.
        images = solid_color_images((191.0, 191.0, 191.0))
        transform = PhotometricDistortion(probability=None, contrast=1.0)

        with mock.patch("autoware_ml.transforms.image.image.np.random.uniform", return_value=2.0):
            distorted_images = self.distort(images, transform)

        self.assertTrue(torch.allclose(distorted_images, torch.full_like(images, 255.0), atol=1.0))

    def test_saturation_leaves_gray_untouched_and_desaturates_colors(self) -> None:
        """Test that the saturation factor has no effect on gray and pulls colors towards it."""
        transform = PhotometricDistortion(probability=None, saturation=1.0)
        gray_images = solid_color_images((100.0, 100.0, 100.0))
        red_images = solid_color_images((255.0, 0.0, 0.0))

        with mock.patch("autoware_ml.transforms.image.image.np.random.uniform", return_value=0.0):
            distorted_gray_images = self.distort(gray_images, transform)
            distorted_red_images = self.distort(red_images, transform)

        self.assertTrue(torch.equal(distorted_gray_images, gray_images))
        # A fully desaturated red keeps its value, hence becomes white.
        self.assertTrue(torch.equal(distorted_red_images, torch.full_like(red_images, 255.0)))

    def test_hue_shift_rotates_the_colors(self) -> None:
        """Test that a third-of-a-turn hue shift maps red onto green in RGB order."""
        red_images = solid_color_images((255.0, 0.0, 0.0))
        transform = PhotometricDistortion(probability=None, hue=0.5)

        # The sampled deviation is scaled by 179, so 60 / 179 lands on OpenCV's green hue of 60.
        with mock.patch(
            "autoware_ml.transforms.image.image.np.random.uniform", return_value=60.0 / 179.0
        ):
            distorted_images = self.distort(red_images, transform)

        self.assertTrue(
            torch.allclose(distorted_images, solid_color_images((0.0, 255.0, 0.0)), atol=1.0)
        )

    def test_same_distortion_for_every_camera(self) -> None:
        """Test that identical camera images are distorted identically."""
        images = self.build_random_images(num_cameras=1).repeat(3, 1, 1, 1)
        transform = PhotometricDistortion(
            probability=None, brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2
        )

        distorted_images = self.distort(images, transform)

        self.assertTrue(torch.equal(distorted_images[1], distorted_images[0]))
        self.assertTrue(torch.equal(distorted_images[2], distorted_images[0]))

    def test_geometry_preserved(self) -> None:
        """Test that every non-pixel field of the camera data is carried over unchanged."""
        sample = self.build_multi_task_gt_sample(self.build_random_images())
        assert sample.camera_image_data is not None

        output = PhotometricDistortion(probability=None, brightness=0.5, hue=0.1)(sample)

        assert output.camera_image_data is not None
        original = sample.camera_image_data
        distorted = output.camera_image_data
        self.assertTrue(torch.equal(distorted.camera_intrinsics, original.camera_intrinsics))
        self.assertTrue(
            torch.equal(distorted.augmented_camera_intrinsics, original.augmented_camera_intrinsics)
        )
        self.assertTrue(
            torch.equal(distorted.image_augmentation_matrices, original.image_augmentation_matrices)
        )
        self.assertTrue(torch.equal(distorted.lidar2cams, original.lidar2cams))
        self.assertTrue(torch.equal(distorted.lidar2images, original.lidar2images))
        self.assertTrue(torch.equal(distorted.timestamps, original.timestamps))
        self.assertEqual(distorted.camera_names, original.camera_names)

    def test_zero_probability_skips(self) -> None:
        """Test that a zero probability returns the sample untouched."""
        sample = self.build_multi_task_gt_sample(self.build_random_images())

        output = PhotometricDistortion(probability=0.0, brightness=0.5, hue=0.1)(sample)

        self.assertIs(output, sample)

    def test_missing_camera_image_data_key(self) -> None:
        """Test that missing 'camera_image_data' raises KeyError."""
        sample = self.build_multi_task_gt_sample(self.build_random_images())._replace(
            camera_image_data=None
        )

        with self.assertRaises(KeyError):
            PhotometricDistortion(probability=None)(sample)


if __name__ == "__main__":
    unittest.main()
