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

"""Unit tests for the camera image loading transforms."""

from pathlib import Path
import tempfile
import unittest

import torch
from torchvision.io import write_png

from autoware_ml.datamodule.multi_task.dataclasses.multi_task_samples import (
    ImageSample,
    MultiTaskGTSample,
)
from autoware_ml.geometry.cameras.base_images import BaseImages
from autoware_ml.transforms.camera.loading import (
    LoadImageFromFile,
    LoadMultiViewImagesFromFiles,
)
from autoware_ml.types.geometry import ImageChannel


class BaseCameraLoadingTestCase(unittest.TestCase):
    """Shared sample builders for the camera image loading transforms."""

    def setUp(self) -> None:
        """Create the temporary directory the test images are written to."""
        self.image_height = 4
        self.image_width = 6
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.image_root = Path(self._temporary_directory.name)
        self.addCleanup(self._temporary_directory.cleanup)

    def build_image_sample(
        self,
        camera_name: str,
        pixel_value: int,
        timestamp: float = 1.5,
    ) -> ImageSample:
        """Write a uniform PNG to disk and build the sample pointing at it.

        Args:
            camera_name: Name of the camera the sample belongs to.
            pixel_value: Value every pixel of the written image holds, which lets a test
                identify which camera an image was loaded from.
            timestamp: Timestamp of the sample.

        Returns:
            ImageSample referencing the image written to the temporary directory.
        """
        image_path = self.image_root / f"{camera_name}.png"
        write_png(
            torch.full((3, self.image_height, self.image_width), pixel_value, dtype=torch.uint8),
            str(image_path),
        )
        return ImageSample(
            image_path=str(image_path),
            camera_name=camera_name,
            timestamp=timestamp,
            camera_intrinsic=torch.tensor(
                [[100.0, 0.0, 3.0], [0.0, 100.0, 2.0], [0.0, 0.0, 1.0]],
                dtype=torch.float32,
            ),
            lidar2cam=torch.eye(4, dtype=torch.float32),
            lidar2image=torch.eye(4, dtype=torch.float32),
            distortion_model="plumb_bob",
            distortion_coefficients=torch.zeros(5, dtype=torch.float32),
        )

    def build_multi_task_gt_sample(self, image_samples: list[ImageSample]) -> MultiTaskGTSample:
        """Build a sample holding only the given image metadata.

        Args:
            image_samples: Image metadata the loading transforms read.

        Returns:
            MultiTaskGTSample with no camera image data loaded yet.
        """
        return MultiTaskGTSample(
            lidar_point_cloud_samples=None,
            image_samples=image_samples,
            point_cloud_data=None,
            camera_image_data=None,
            detection3d_gt_bboxes_3d=None,
            segmentation3d_gt_sample=None,
        )


class TestLoadImageFromFile(BaseCameraLoadingTestCase):
    """Unit tests for the LoadImageFromFile transform."""

    def test_instantiation(self) -> None:
        """Test instantiation stores the color type and normalization flag."""
        transform = LoadImageFromFile(color_type=ImageChannel.RGB, normalize_to_unit=True)

        self.assertEqual(transform.color_type, ImageChannel.RGB)
        self.assertTrue(transform.normalize_to_unit)

    def test_missing_image_samples_key(self) -> None:
        """Test that missing 'image_samples' raises KeyError."""
        sample = self.build_multi_task_gt_sample([])._replace(image_samples=None)

        with self.assertRaises(KeyError):
            LoadImageFromFile(color_type=ImageChannel.RGB, normalize_to_unit=True)(sample)

    def test_loads_only_the_first_image(self) -> None:
        """Test that a single camera is loaded even when the sample holds several."""
        sample = self.build_multi_task_gt_sample(
            [self.build_image_sample("camera0", 10), self.build_image_sample("camera1", 20)]
        )

        output = LoadImageFromFile(color_type=ImageChannel.RGB, normalize_to_unit=False)(sample)

        assert output.camera_image_data is not None
        camera_image_data = output.camera_image_data
        self.assertIsInstance(camera_image_data, BaseImages)
        self.assertEqual(camera_image_data.camera_names, ["camera0"])
        self.assertTrue(torch.allclose(camera_image_data.images, torch.tensor(10.0)))

    def test_keeps_the_leading_num_cameras_dimension(self) -> None:
        """Test that every per-camera field stays indexable by camera."""
        sample = self.build_multi_task_gt_sample([self.build_image_sample("camera0", 10)])
        assert sample.image_samples is not None
        image_sample = sample.image_samples[0]

        output = LoadImageFromFile(color_type=ImageChannel.RGB, normalize_to_unit=False)(sample)

        assert output.camera_image_data is not None
        camera_image_data = output.camera_image_data
        self.assertEqual(
            camera_image_data.images.shape, (1, 3, self.image_height, self.image_width)
        )
        self.assertEqual(camera_image_data.timestamps.shape, (1,))
        self.assertEqual(camera_image_data.camera_intrinsics.shape, (1, 3, 3))
        self.assertEqual(camera_image_data.lidar2cams.shape, (1, 4, 4))
        self.assertEqual(camera_image_data.lidar2images.shape, (1, 4, 4))
        self.assertEqual(camera_image_data.augmented_camera_intrinsics.shape, (1, 3, 3))
        self.assertTrue(
            torch.equal(camera_image_data.camera_intrinsics[0], image_sample.camera_intrinsic)
        )

    def test_normalize_to_unit(self) -> None:
        """Test that pixel values are divided by 255 only when requested."""
        sample = self.build_multi_task_gt_sample([self.build_image_sample("camera0", 255)])

        normalized = LoadImageFromFile(color_type=ImageChannel.RGB, normalize_to_unit=True)(sample)
        raw = LoadImageFromFile(color_type=ImageChannel.RGB, normalize_to_unit=False)(sample)

        assert normalized.camera_image_data is not None
        assert raw.camera_image_data is not None
        self.assertEqual(normalized.camera_image_data.images.dtype, torch.float32)
        self.assertTrue(torch.allclose(normalized.camera_image_data.images, torch.tensor(1.0)))
        self.assertTrue(torch.allclose(raw.camera_image_data.images, torch.tensor(255.0)))

    def test_metadata_carried_over(self) -> None:
        """Test that the calibration metadata of the sample is carried over."""
        sample = self.build_multi_task_gt_sample(
            [self.build_image_sample("camera0", 10, timestamp=2.5)]
        )

        output = LoadImageFromFile(color_type=ImageChannel.RGB, normalize_to_unit=False)(sample)

        assert output.camera_image_data is not None
        camera_image_data = output.camera_image_data
        self.assertEqual(camera_image_data.distortion_models, ["plumb_bob"])
        self.assertTrue(torch.allclose(camera_image_data.timestamps, torch.tensor([2.5])))
        self.assertTrue(torch.equal(camera_image_data.distortion_coefficients[0], torch.zeros(5)))
        # Both are set only once the augmentation transforms have run.
        self.assertIsNone(camera_image_data.noises)
        self.assertTrue(
            torch.equal(
                camera_image_data.augmented_camera_intrinsics,
                camera_image_data.camera_intrinsics,
            )
        )
        self.assertTrue(
            torch.equal(
                camera_image_data.image_augmentation_matrices,
                torch.eye(4, dtype=torch.float32).repeat(
                    camera_image_data.camera_intrinsics.shape[0], 1, 1
                ),
            )
        )


class TestLoadMultiViewImagesFromFiles(BaseCameraLoadingTestCase):
    """Unit tests for the LoadMultiViewImagesFromFiles transform."""

    def test_instantiation(self) -> None:
        """Test instantiation stores the color type, normalization flag and camera order."""
        transform = LoadMultiViewImagesFromFiles(
            normalize_to_unit=True,
            color_type=ImageChannel.RGB,
            camera_order=["camera0", "camera1"],
        )

        self.assertEqual(transform.color_type, ImageChannel.RGB)
        self.assertTrue(transform.normalize_to_unit)
        self.assertEqual(transform.camera_order, ["camera0", "camera1"])

    def test_missing_image_samples_key(self) -> None:
        """Test that missing 'image_samples' raises KeyError."""
        sample = self.build_multi_task_gt_sample([])._replace(image_samples=None)

        with self.assertRaises(KeyError):
            LoadMultiViewImagesFromFiles(
                normalize_to_unit=False,
                color_type=ImageChannel.RGB,
                camera_order=["camera0"],
            )(sample)

    def test_loads_every_camera(self) -> None:
        """Test that every camera is stacked along the leading dimension."""
        sample = self.build_multi_task_gt_sample(
            [self.build_image_sample("camera0", 10), self.build_image_sample("camera1", 20)]
        )

        output = LoadMultiViewImagesFromFiles(
            normalize_to_unit=False,
            color_type=ImageChannel.RGB,
            camera_order=["camera0", "camera1"],
        )(sample)

        assert output.camera_image_data is not None
        camera_image_data = output.camera_image_data
        self.assertIsInstance(camera_image_data, BaseImages)
        self.assertEqual(
            camera_image_data.images.shape, (2, 3, self.image_height, self.image_width)
        )
        self.assertEqual(camera_image_data.timestamps.shape, (2,))
        self.assertEqual(camera_image_data.camera_intrinsics.shape, (2, 3, 3))
        self.assertEqual(camera_image_data.lidar2cams.shape, (2, 4, 4))
        self.assertEqual(camera_image_data.lidar2images.shape, (2, 4, 4))
        self.assertEqual(camera_image_data.distortion_models, ["plumb_bob"] * 2)
        self.assertEqual(len(camera_image_data.distortion_coefficients), 2)

    def test_images_follow_the_camera_order(self) -> None:
        """Test that the images are reordered to match the configured camera order."""
        # The metadata is given in the reverse of the requested order.
        sample = self.build_multi_task_gt_sample(
            [self.build_image_sample("camera1", 20), self.build_image_sample("camera0", 10)]
        )

        output = LoadMultiViewImagesFromFiles(
            normalize_to_unit=False,
            color_type=ImageChannel.RGB,
            camera_order=["camera0", "camera1"],
        )(sample)

        assert output.camera_image_data is not None
        camera_image_data = output.camera_image_data
        self.assertEqual(camera_image_data.camera_names, ["camera0", "camera1"])
        # Every image holds its camera's pixel value, hence the images follow the order.
        self.assertTrue(torch.allclose(camera_image_data.images[0], torch.tensor(10.0)))
        self.assertTrue(torch.allclose(camera_image_data.images[1], torch.tensor(20.0)))

    def test_loads_a_subset_of_the_cameras(self) -> None:
        """Test that only the cameras of the configured order are loaded."""
        sample = self.build_multi_task_gt_sample(
            [self.build_image_sample("camera0", 10), self.build_image_sample("camera1", 20)]
        )

        output = LoadMultiViewImagesFromFiles(
            normalize_to_unit=False,
            color_type=ImageChannel.RGB,
            camera_order=["camera1"],
        )(sample)

        assert output.camera_image_data is not None
        camera_image_data = output.camera_image_data
        self.assertEqual(camera_image_data.camera_names, ["camera1"])
        self.assertTrue(torch.allclose(camera_image_data.images, torch.tensor(20.0)))

    def test_missing_camera_raises(self) -> None:
        """Test that a camera of the order missing from the sample raises ValueError."""
        sample = self.build_multi_task_gt_sample([self.build_image_sample("camera0", 10)])

        with self.assertRaises(ValueError):
            LoadMultiViewImagesFromFiles(
                normalize_to_unit=False,
                color_type=ImageChannel.RGB,
                camera_order=["camera0", "camera1"],
            )(sample)

    def test_normalize_to_unit(self) -> None:
        """Test that pixel values are divided by 255 only when requested."""
        sample = self.build_multi_task_gt_sample(
            [self.build_image_sample("camera0", 255), self.build_image_sample("camera1", 255)]
        )
        camera_order = ["camera0", "camera1"]

        normalized = LoadMultiViewImagesFromFiles(
            normalize_to_unit=True, color_type=ImageChannel.RGB, camera_order=camera_order
        )(sample)
        raw = LoadMultiViewImagesFromFiles(
            normalize_to_unit=False, color_type=ImageChannel.RGB, camera_order=camera_order
        )(sample)

        assert normalized.camera_image_data is not None
        assert raw.camera_image_data is not None
        self.assertEqual(normalized.camera_image_data.images.dtype, torch.float32)
        self.assertTrue(torch.allclose(normalized.camera_image_data.images, torch.tensor(1.0)))
        self.assertTrue(torch.allclose(raw.camera_image_data.images, torch.tensor(255.0)))

    def test_augmented_intrinsics_initialized_from_the_raw_ones(self) -> None:
        """Test that the augmented intrinsics start out as a copy of the raw ones."""
        sample = self.build_multi_task_gt_sample(
            [self.build_image_sample("camera0", 10), self.build_image_sample("camera1", 20)]
        )

        output = LoadMultiViewImagesFromFiles(
            normalize_to_unit=False,
            color_type=ImageChannel.RGB,
            camera_order=["camera0", "camera1"],
        )(sample)

        assert output.camera_image_data is not None
        camera_image_data = output.camera_image_data
        self.assertIsNone(camera_image_data.noises)
        self.assertTrue(
            torch.equal(
                camera_image_data.augmented_camera_intrinsics,
                camera_image_data.camera_intrinsics,
            )
        )
        self.assertTrue(
            torch.equal(
                camera_image_data.image_augmentation_matrices,
                torch.eye(4, dtype=torch.float32).repeat(
                    camera_image_data.camera_intrinsics.shape[0], 1, 1
                ),
            )
        )


if __name__ == "__main__":
    unittest.main()
