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

"""Unit tests for the camera image resize, crop, and padding transforms."""

import unittest

import numpy as np
import torch

from autoware_ml.datamodule.multi_task.dataclasses.multi_task_samples import MultiTaskGTSample
from autoware_ml.geometry.cameras.base_images import BaseImages
from autoware_ml.transforms.camera.resize import (
    CropAndScale,
    PadMultiViewImage,
    ResizeCropFlipRotImage,
    ResizeMultiviewImages,
)


class CameraImageDataTestCase(unittest.TestCase):
    """Base test case building a sample holding only camera image data."""

    def setUp(self) -> None:
        """Set up the camera calibration shared by all tests."""
        self.image_height = 32
        self.image_width = 48
        self.num_cameras = 2
        # A realistic pinhole intrinsic matrix for the image size under test.
        self.camera_intrinsic = torch.tensor(
            [[100.0, 0.0, 24.0], [0.0, 100.0, 16.0], [0.0, 0.0, 1.0]], dtype=torch.float32
        )
        # A camera looking forward, one metre above and behind the lidar.
        self.lidar2cam = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, -1.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )

    def build_multi_task_gt_sample(self) -> MultiTaskGTSample:
        """Build a minimal sample holding only camera image data.

        Returns:
            MultiTaskGTSample holding random images and the test camera calibration.
        """
        homogeneous_intrinsic = torch.eye(4, dtype=torch.float32)
        homogeneous_intrinsic[:3, :3] = self.camera_intrinsic

        camera_image_data = BaseImages(
            images=torch.rand(
                (self.num_cameras, 3, self.image_height, self.image_width), dtype=torch.float32
            ),
            timestamps=torch.zeros(self.num_cameras, dtype=torch.float32),
            camera_intrinsics=self.camera_intrinsic.repeat(self.num_cameras, 1, 1),
            camera_names=[f"camera{index}" for index in range(self.num_cameras)],
            lidar2images=(homogeneous_intrinsic @ self.lidar2cam).repeat(self.num_cameras, 1, 1),
            lidar2cams=self.lidar2cam.repeat(self.num_cameras, 1, 1),
            distortion_models=["plumb_bob"] * self.num_cameras,
            distortion_coefficients=[torch.zeros(5)] * self.num_cameras,
        )
        return MultiTaskGTSample(
            lidar_point_cloud_samples=None,
            image_samples=None,
            point_cloud_data=None,
            camera_image_data=camera_image_data,
            detection3d_gt_bboxes_3d=None,
            segmentation3d_gt_sample=None,
        )

    def assert_projection_is_consistent(self, camera_image_data: BaseImages) -> None:
        """Assert `lidar2images` projects like the augmented intrinsics and `lidar2cams`.

        Args:
            camera_image_data: Camera image data returned by a transform.
        """
        homogeneous_intrinsics = torch.eye(4, dtype=torch.float32).repeat(self.num_cameras, 1, 1)
        homogeneous_intrinsics[:, :3, :3] = camera_image_data.augmented_camera_intrinsics
        self.assertTrue(
            torch.allclose(
                camera_image_data.lidar2images,
                homogeneous_intrinsics @ camera_image_data.lidar2cams,
                atol=1e-5,
            )
        )


class TestCropAndScale(CameraImageDataTestCase):
    """Unit tests for the CropAndScale transform."""

    def test_instantiation(self) -> None:
        """Test instantiation with default and custom parameters."""
        self.assertEqual(CropAndScale().crop_ratio, 0.8)
        self.assertEqual(CropAndScale(probability=0.9, crop_ratio=0.7).crop_ratio, 0.7)

    def test_missing_camera_image_data_key(self) -> None:
        """Test that missing 'camera_image_data' raises KeyError."""
        sample = self.build_multi_task_gt_sample()._replace(camera_image_data=None)

        with self.assertRaises(KeyError):
            CropAndScale(probability=1.0)(sample)

    def test_zero_probability_is_a_no_op(self) -> None:
        """Test that a zero probability leaves the sample untouched."""
        sample = self.build_multi_task_gt_sample()

        output = CropAndScale(probability=0.0)(sample)

        self.assertIs(output.camera_image_data, sample.camera_image_data)

    def test_image_size_preserved_and_intrinsics_updated(self) -> None:
        """Test that the crop is scaled back to the source size and zooms the intrinsics."""
        sample = self.build_multi_task_gt_sample()
        assert sample.camera_image_data is not None

        output = CropAndScale(probability=1.0, crop_ratio=0.8)(sample)

        assert output.camera_image_data is not None
        camera_image_data = output.camera_image_data
        self.assertEqual(camera_image_data.images.shape, sample.camera_image_data.images.shape)
        # Scaling a crop back to the source size can only magnify the focal lengths.
        self.assertTrue(torch.all(camera_image_data.augmented_camera_intrinsics[:, 0, 0] >= 100.0))
        self.assertTrue(torch.all(camera_image_data.augmented_camera_intrinsics[:, 1, 1] >= 100.0))
        # The raw intrinsics stay untouched, only the augmented ones follow the transform.
        self.assertTrue(
            torch.equal(
                camera_image_data.camera_intrinsics, sample.camera_image_data.camera_intrinsics
            )
        )
        self.assert_projection_is_consistent(camera_image_data)

    def test_cameras_are_cropped_independently(self) -> None:
        """Test that every camera of a sample is given its own crop box."""
        np.random.seed(0)
        sample = self.build_multi_task_gt_sample()

        output = CropAndScale(probability=1.0, crop_ratio=0.5)(sample)

        assert output.camera_image_data is not None
        intrinsics = output.camera_image_data.augmented_camera_intrinsics
        self.assertFalse(torch.allclose(intrinsics[0], intrinsics[1]))


class TestResizeMultiviewImages(CameraImageDataTestCase):
    """Unit tests for the ResizeMultiviewImages transform."""

    def test_missing_camera_image_data_key(self) -> None:
        """Test that missing 'camera_image_data' raises KeyError."""
        sample = self.build_multi_task_gt_sample()._replace(camera_image_data=None)

        with self.assertRaises(KeyError):
            ResizeMultiviewImages(target_size=[16, 24])(sample)

    def test_images_and_intrinsics_are_scaled(self) -> None:
        """Test that halving the image size halves the focal lengths and the center."""
        sample = self.build_multi_task_gt_sample()

        output = ResizeMultiviewImages(target_size=[self.image_height // 2, self.image_width // 2])(
            sample
        )

        assert output.camera_image_data is not None
        camera_image_data = output.camera_image_data
        self.assertEqual(
            camera_image_data.images.shape,
            (self.num_cameras, 3, self.image_height // 2, self.image_width // 2),
        )
        expected_intrinsic = torch.tensor(
            [[50.0, 0.0, 12.0], [0.0, 50.0, 8.0], [0.0, 0.0, 1.0]], dtype=torch.float32
        )
        self.assertTrue(
            torch.allclose(
                camera_image_data.augmented_camera_intrinsics,
                expected_intrinsic.repeat(self.num_cameras, 1, 1),
            )
        )
        self.assert_projection_is_consistent(camera_image_data)

    def test_anisotropic_resize(self) -> None:
        """Test that the width and the height are scaled independently."""
        sample = self.build_multi_task_gt_sample()

        output = ResizeMultiviewImages(target_size=[self.image_height, self.image_width * 2])(
            sample
        )

        assert output.camera_image_data is not None
        intrinsics = output.camera_image_data.augmented_camera_intrinsics
        self.assertTrue(torch.allclose(intrinsics[:, 0, 0], torch.full((2,), 200.0)))
        self.assertTrue(torch.allclose(intrinsics[:, 1, 1], torch.full((2,), 100.0)))

    def test_projected_point_follows_the_resize(self) -> None:
        """Test that a lidar point projects onto the pixel the resize maps it to."""
        sample = self.build_multi_task_gt_sample()
        assert sample.camera_image_data is not None
        # (4, ), a point five metres in front of the lidar.
        point = torch.tensor([1.0, 0.5, 5.0, 1.0], dtype=torch.float32)
        source_projection = sample.camera_image_data.lidar2images[0] @ point
        source_pixel = source_projection[:2] / source_projection[2]

        output = ResizeMultiviewImages(target_size=[self.image_height // 2, self.image_width // 2])(
            sample
        )

        assert output.camera_image_data is not None
        projection = output.camera_image_data.lidar2images[0] @ point
        self.assertTrue(
            torch.allclose(projection[:2] / projection[2], source_pixel / 2.0, atol=1e-5)
        )


class TestPadMultiViewImage(CameraImageDataTestCase):
    """Unit tests for the PadMultiViewImage transform."""

    def test_requires_exactly_one_size_argument(self) -> None:
        """Test that giving none or both size arguments raises ValueError."""
        with self.assertRaises(ValueError):
            PadMultiViewImage()
        with self.assertRaises(ValueError):
            PadMultiViewImage(size=[64, 64], size_divisor=32)

    def test_pads_to_size_divisor(self) -> None:
        """Test that the images are rounded up to a multiple of the size divisor."""
        sample = self.build_multi_task_gt_sample()
        assert sample.camera_image_data is not None

        output = PadMultiViewImage(size_divisor=32, pad_value=0.0)(sample)

        assert output.camera_image_data is not None
        camera_image_data = output.camera_image_data
        self.assertEqual(camera_image_data.images.shape, (self.num_cameras, 3, 32, 64))
        # The padding is appended at the bottom and the right, the source pixels stay put.
        self.assertTrue(
            torch.equal(
                camera_image_data.images[:, :, : self.image_height, : self.image_width],
                sample.camera_image_data.images,
            )
        )
        self.assertTrue(torch.all(camera_image_data.images[:, :, :, self.image_width :] == 0.0))

    def test_pads_to_fixed_size_and_keeps_intrinsics(self) -> None:
        """Test that padding to a fixed size leaves the intrinsics untouched."""
        sample = self.build_multi_task_gt_sample()
        assert sample.camera_image_data is not None

        output = PadMultiViewImage(size=[64, 64], pad_value=1.0)(sample)

        assert output.camera_image_data is not None
        camera_image_data = output.camera_image_data
        self.assertEqual(camera_image_data.images.shape, (self.num_cameras, 3, 64, 64))
        self.assertTrue(torch.all(camera_image_data.images[:, :, self.image_height :, :] == 1.0))
        self.assertTrue(
            torch.allclose(
                camera_image_data.augmented_camera_intrinsics,
                sample.camera_image_data.augmented_camera_intrinsics,
            )
        )
        self.assert_projection_is_consistent(camera_image_data)

    def test_smaller_target_size_raises(self) -> None:
        """Test that padding to a smaller size than the images raises ValueError."""
        sample = self.build_multi_task_gt_sample()

        with self.assertRaises(ValueError):
            PadMultiViewImage(size=[16, 16])(sample)


class TestResizeCropFlipRotImage(CameraImageDataTestCase):
    """Unit tests for the ResizeCropFlipRotImage transform."""

    def test_missing_camera_image_data_key(self) -> None:
        """Test that missing 'camera_image_data' raises KeyError."""
        sample = self.build_multi_task_gt_sample()._replace(camera_image_data=None)

        with self.assertRaises(KeyError):
            ResizeCropFlipRotImage(
                target_size=[16, 24],
                resize_range=[0.5, 0.5],
                bottom_crop_ratio_range=[0.0, 0.0],
                training=False,
            )(sample)

    def test_target_size_and_intrinsics_in_validation(self) -> None:
        """Test that a plain half resize is deterministic and only scales the intrinsics."""
        sample = self.build_multi_task_gt_sample()

        output = ResizeCropFlipRotImage(
            target_size=[self.image_height // 2, self.image_width // 2],
            resize_range=[0.5, 0.5],
            bottom_crop_ratio_range=[0.0, 0.0],
            training=False,
        )(sample)

        assert output.camera_image_data is not None
        camera_image_data = output.camera_image_data
        self.assertEqual(
            camera_image_data.images.shape,
            (self.num_cameras, 3, self.image_height // 2, self.image_width // 2),
        )
        expected_intrinsic = torch.tensor(
            [[50.0, 0.0, 12.0], [0.0, 50.0, 8.0], [0.0, 0.0, 1.0]], dtype=torch.float32
        )
        self.assertTrue(
            torch.allclose(
                camera_image_data.augmented_camera_intrinsics,
                expected_intrinsic.repeat(self.num_cameras, 1, 1),
                atol=1e-5,
            )
        )
        self.assert_projection_is_consistent(camera_image_data)

    def test_scalar_resize_range_fits_the_target_size(self) -> None:
        """Test that a scalar resize range is read as a tolerance around a fitting resize."""
        transform = ResizeCropFlipRotImage(
            target_size=[self.image_height // 2, self.image_width // 2],
            resize_range=0.0,
            bottom_crop_ratio_range=[0.0, 0.0],
            training=True,
        )

        parameters = transform.sample_augmentation(self.image_height, self.image_width)

        self.assertAlmostEqual(parameters.resize, 0.5)

    def test_bottom_crop_keeps_the_bottom_of_the_image(self) -> None:
        """Test that the crop box is taken from the bottom of the resized image."""
        transform = ResizeCropFlipRotImage(
            target_size=[self.image_height, self.image_width],
            resize_range=[1.0, 1.0],
            bottom_crop_ratio_range=[0.5, 0.5],
            training=False,
        )

        parameters = transform.sample_augmentation(self.image_height, self.image_width)

        self.assertEqual(parameters.crop_height, self.image_height // 2)
        self.assertEqual(parameters.crop_top, self.image_height - self.image_height // 2)

    def test_horizontal_flip_matches_the_intrinsics(self) -> None:
        """Test that a flipped image is described by the flipped augmented intrinsics."""
        np.random.seed(0)
        sample = self.build_multi_task_gt_sample()
        assert sample.camera_image_data is not None
        image = sample.camera_image_data.images[0]

        transform = ResizeCropFlipRotImage(
            target_size=[self.image_height, self.image_width],
            resize_range=[1.0, 1.0],
            bottom_crop_ratio_range=[0.0, 0.0],
            training=True,
            random_horizontal_flip=True,
        )
        parameters = transform.sample_augmentation(self.image_height, self.image_width)
        parameters = parameters._replace(horizontal_flip=True)
        augmented_image, image_transform = transform.apply_augmentation(image, parameters)

        self.assertTrue(torch.allclose(augmented_image, torch.flip(image, dims=[-1])))
        # The principal point is mirrored about the center of the image width.
        principal_point = image_transform @ torch.tensor([24.0, 16.0, 1.0])
        self.assertAlmostEqual(float(principal_point[0]), self.image_width - 1.0 - 24.0)
        self.assertAlmostEqual(float(principal_point[1]), 16.0)

    def test_rotation_matches_the_intrinsics(self) -> None:
        """Test that the image transform predicts where a rotation moves a pixel."""
        sample = self.build_multi_task_gt_sample()
        assert sample.camera_image_data is not None
        # A single lit pixel whose location can be recovered from the rotated image.
        image = torch.zeros_like(sample.camera_image_data.images[0])
        image[:, 8, 12] = 1.0

        transform = ResizeCropFlipRotImage(
            target_size=[self.image_height, self.image_width],
            resize_range=[1.0, 1.0],
            bottom_crop_ratio_range=[0.0, 0.0],
            training=False,
        )
        parameters = transform.sample_augmentation(self.image_height, self.image_width)
        parameters = parameters._replace(rotation=90.0)
        augmented_image, image_transform = transform.apply_augmentation(image, parameters)

        rotated_pixel = image_transform @ torch.tensor([12.0, 8.0, 1.0])
        lit_index = int(torch.argmax(augmented_image[0].flatten()))
        lit_y, lit_x = divmod(lit_index, self.image_width)
        self.assertAlmostEqual(float(rotated_pixel[0]), lit_x, places=4)
        self.assertAlmostEqual(float(rotated_pixel[1]), lit_y, places=4)

    def test_cameras_are_augmented_independently(self) -> None:
        """Test that every camera of a sample is given its own augmentation."""
        np.random.seed(0)
        sample = self.build_multi_task_gt_sample()

        output = ResizeCropFlipRotImage(
            target_size=[self.image_height, self.image_width],
            resize_range=[0.5, 1.5],
            bottom_crop_ratio_range=[0.0, 0.3],
            training=True,
        )(sample)

        assert output.camera_image_data is not None
        intrinsics = output.camera_image_data.augmented_camera_intrinsics
        self.assertFalse(torch.allclose(intrinsics[0], intrinsics[1]))
        self.assert_projection_is_consistent(output.camera_image_data)


if __name__ == "__main__":
    unittest.main()
