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

"""Unit tests for the camera-only geometric augmentations."""

import unittest

from jaxtyping import Float32
import numpy as np
import torch
from torch import Tensor

from autoware_ml.datamodule.multi_task.dataclasses.multi_task_samples import MultiTaskGTSample
from autoware_ml.geometry.bbox_3d.base_bbox3d import BaseBBoxes3D
from autoware_ml.geometry.bbox_3d.lidar_bbox3d import LidarBBoxes3D
from autoware_ml.geometry.cameras.base_images import BaseImages
from autoware_ml.transforms.camera.geometry import GlobalBEVRandomFlip, GlobalRotScaleTrans
from autoware_ml.types.geometry import (
    Box3DCenterCoordinateType,
    Box3DFieldIndex,
    TransformationName,
)


class BaseCameraGeometryTestCase(unittest.TestCase):
    """Shared sample builders and assertions for the camera-only geometric augmentations."""

    def build_bboxes_3d(self) -> LidarBBoxes3D:
        """Build a single 3D bounding box with a non-zero yaw and velocity.

        Returns:
            LidarBBoxes3D holding one bounding box.
        """
        return LidarBBoxes3D(
            bbox_params=torch.tensor(
                [[1.0, 2.0, 0.0, 4.0, 2.0, 1.0, 0.3, 1.5, -0.5, 0.0]], dtype=torch.float32
            ),
            bbox_labels=torch.tensor([0], dtype=torch.int32),
            bbox_label_names=["car"],
            bbox_num_lidar_points=torch.tensor([10], dtype=torch.int32),
            bbox_center_coordinate_type=Box3DCenterCoordinateType.GRAVITY_CENTER,
        )

    def build_multi_task_gt_sample(
        self,
        num_cameras: int = 2,
        with_bboxes_3d: bool = True,
    ) -> MultiTaskGTSample:
        """Build a minimal sample holding camera image data and optional bboxes.

        Args:
            num_cameras: Number of cameras the sample holds.
            with_bboxes_3d: Whether the sample holds 3D bounding boxes.

        Returns:
            MultiTaskGTSample with identity camera matrices, hence every update applied by a
            transform is the transformation itself.
        """
        camera_image_data = BaseImages(
            images=torch.ones((num_cameras, 3, 4, 4), dtype=torch.float32),
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
            detection3d_gt_bboxes_3d=self.build_bboxes_3d() if with_bboxes_3d else None,
            segmentation3d_gt_sample=None,
        )

    def apply_inverse_transformation(
        self,
        bboxes_3d: BaseBBoxes3D,
        transformation_matrix: Float32[Tensor, "4 4"],
    ) -> None:
        """Apply the inverse of a saved augmentation to the bboxes in place.

        The 4x4 matrix is decomposed back into the rotation, scale, and translation the
        bounding box API expects, which are applied in the same order the transforms use.

        Args:
            bboxes_3d: The augmented bounding boxes to restore.
            transformation_matrix: The 4x4 augmentation saved by the transform.
        """
        inverse_transformation_matrix = torch.linalg.inv(transformation_matrix)
        # The upper-left block is rotation * scale, so any of its columns has the scale as norm.
        scale_factor = float(torch.linalg.norm(inverse_transformation_matrix[:3, 0]))
        rotation_matrix = inverse_transformation_matrix[:3, :3] / scale_factor

        # Rotation is expressed in the column vector convention, the bboxes expect a row one.
        bboxes_3d.rotate(rotation_matrix.T)
        bboxes_3d.scale(scale_factor)
        bboxes_3d.translate(inverse_transformation_matrix[:3, 3].reshape(1, 3))

    def assert_bbox_params_close(
        self,
        bbox_params: Float32[Tensor, "num_bboxes num_Box3DFieldIndex"],
        expected_bbox_params: Float32[Tensor, "num_bboxes num_Box3DFieldIndex"],
    ) -> None:
        """Assert that two sets of bbox parameters match, comparing yaw modulo a full turn.

        Args:
            bbox_params: The bbox parameters under test.
            expected_bbox_params: The bbox parameters they are expected to match.
        """
        # A flip mirrors the yaw twice, which restores the heading but can offset it by 2 pi.
        yaw_difference = (
            torch.remainder(
                bbox_params[:, Box3DFieldIndex.YAW]
                - expected_bbox_params[:, Box3DFieldIndex.YAW]
                + torch.pi,
                2 * torch.pi,
            )
            - torch.pi
        )
        self.assertTrue(torch.allclose(yaw_difference, torch.zeros_like(yaw_difference), atol=1e-5))

        remaining_fields = [
            index for index in range(bbox_params.shape[1]) if index != Box3DFieldIndex.YAW
        ]
        self.assertTrue(
            torch.allclose(
                bbox_params[:, remaining_fields],
                expected_bbox_params[:, remaining_fields],
                atol=1e-5,
            )
        )


class TestGlobalRotScaleTrans(BaseCameraGeometryTestCase):
    """Unit tests for the camera-only GlobalRotScaleTrans transform."""

    def setUp(self) -> None:
        """Set up the same sample and transform parameters for all tests."""
        np.random.seed(0)
        self.sample = self.build_multi_task_gt_sample()
        assert self.sample.detection3d_gt_bboxes_3d is not None
        assert self.sample.camera_image_data is not None
        self.original_bbox_params = self.sample.detection3d_gt_bboxes_3d.bbox_params.clone()
        self.original_lidar2cams = self.sample.camera_image_data.lidar2cams.clone()
        self.original_lidar2images = self.sample.camera_image_data.lidar2images.clone()
        self.yaw_rot_range = [-0.5, 0.5]
        self.scale_ratio_range = [0.9, 1.1]
        self.translation_std = [0.5, 0.5, 0.2]

    def test_saves_transformation_matrix(self) -> None:
        """Test that the sampled augmentation is saved as a 4x4 transformation matrix."""
        transform = GlobalRotScaleTrans(
            yaw_rot_range=self.yaw_rot_range,
            scale_ratio_range=self.scale_ratio_range,
            translation_std=self.translation_std,
        )

        # Both draws start from the same seed, so the transform samples the expected values.
        np.random.seed(0)
        expected_lidar_transformation_sample, _ = transform.sample_rot_scale_trans()
        np.random.seed(0)
        output = transform(self.sample)

        assert output.lidar_transformation_sample is not None
        lidar_transformation_sample = output.lidar_transformation_sample
        self.assertTrue(
            torch.allclose(
                lidar_transformation_sample.transformation_matrix,
                expected_lidar_transformation_sample.transformation_matrix,
            )
        )
        self.assertEqual(
            lidar_transformation_sample.transformation_order,
            [
                TransformationName.ROTATION,
                TransformationName.SCALING,
                TransformationName.TRANSLATION,
            ],
        )

    def test_camera_matrices_follow_the_augmentation(self) -> None:
        """Test that the camera extrinsics are re-expressed in the augmented frame."""
        output = GlobalRotScaleTrans(
            yaw_rot_range=self.yaw_rot_range,
            scale_ratio_range=self.scale_ratio_range,
            translation_std=self.translation_std,
        )(self.sample)

        assert output.lidar_transformation_sample is not None
        assert output.camera_image_data is not None
        augmentation = output.lidar_transformation_sample.transformation_matrix
        camera_image_data = output.camera_image_data
        self.assertTrue(
            torch.allclose(
                camera_image_data.lidar2cams,
                self.original_lidar2cams @ torch.linalg.inv(augmentation),
                atol=1e-5,
            )
        )

        # lidar2image is recomputed from the (unchanged) intrinsics and the new lidar2cam.
        expected_camera_intrinsics = torch.eye(4).repeat(camera_image_data.images.shape[0], 1, 1)
        expected_camera_intrinsics[:, :3, :3] = camera_image_data.augmented_camera_intrinsics
        self.assertTrue(
            torch.allclose(
                camera_image_data.lidar2images,
                expected_camera_intrinsics @ camera_image_data.lidar2cams,
                atol=1e-5,
            )
        )

    def test_bboxes_transformed(self) -> None:
        """Test that the bboxes are rotated, scaled, and translated."""
        output = GlobalRotScaleTrans(yaw_rot_range=[0.3, 0.3], scale_ratio_range=[2.0, 2.0])(
            self.sample
        )

        assert output.detection3d_gt_bboxes_3d is not None
        bbox_params = output.detection3d_gt_bboxes_3d.bbox_params
        # Yaw is offset by the sampled rotation and the dimensions scaled by the sampled scale.
        self.assertTrue(
            torch.allclose(bbox_params[:, 6], self.original_bbox_params[:, 6] + 0.3, atol=1e-5)
        )
        self.assertTrue(
            torch.allclose(bbox_params[:, 3:6], self.original_bbox_params[:, 3:6] * 2.0, atol=1e-5)
        )

    def test_points_never_touched(self) -> None:
        """Test that the camera-only variant leaves the point cloud alone."""
        output = GlobalRotScaleTrans(yaw_rot_range=[0.3, 0.3], scale_ratio_range=[2.0, 2.0])(
            self.sample
        )

        self.assertIsNone(output.point_cloud_data)

    def test_composes_with_previous_transformation(self) -> None:
        """Test that the augmentation is composed with an earlier one in the pipeline."""
        transform = GlobalRotScaleTrans(yaw_rot_range=[0.3, 0.3], scale_ratio_range=[2.0, 2.0])

        first_output = transform(self.sample)
        assert first_output.lidar_transformation_sample is not None
        first_transformation_matrix = (
            first_output.lidar_transformation_sample.transformation_matrix.clone()
        )
        second_output = transform(first_output)

        assert second_output.lidar_transformation_sample is not None
        self.assertTrue(
            torch.allclose(
                second_output.lidar_transformation_sample.transformation_matrix,
                first_transformation_matrix @ first_transformation_matrix,
                atol=1e-5,
            )
        )
        self.assertEqual(
            second_output.lidar_transformation_sample.transformation_order,
            [
                TransformationName.ROTATION,
                TransformationName.SCALING,
                TransformationName.TRANSLATION,
            ]
            * 2,
        )

    def test_inverse_transformation_recovers_original_bboxes(self) -> None:
        """Test that inversely applying the saved matrix restores the original bboxes."""
        output = GlobalRotScaleTrans(
            yaw_rot_range=self.yaw_rot_range,
            scale_ratio_range=self.scale_ratio_range,
            translation_std=self.translation_std,
        )(self.sample)

        assert output.detection3d_gt_bboxes_3d is not None
        assert output.lidar_transformation_sample is not None
        bboxes_3d = output.detection3d_gt_bboxes_3d
        self.assertFalse(torch.allclose(bboxes_3d.bbox_params, self.original_bbox_params))

        self.apply_inverse_transformation(
            bboxes_3d, output.lidar_transformation_sample.transformation_matrix
        )

        self.assert_bbox_params_close(bboxes_3d.bbox_params, self.original_bbox_params)

    def test_inverse_transformation_recovers_original_camera_matrices(self) -> None:
        """Test that inversely applying the saved matrix restores the camera matrices."""
        output = GlobalRotScaleTrans(
            yaw_rot_range=self.yaw_rot_range,
            scale_ratio_range=self.scale_ratio_range,
            translation_std=self.translation_std,
        )(self.sample)

        assert output.camera_image_data is not None
        assert output.lidar_transformation_sample is not None
        camera_image_data = output.camera_image_data
        transformation_matrix = output.lidar_transformation_sample.transformation_matrix
        self.assertFalse(torch.allclose(camera_image_data.lidar2cams, self.original_lidar2cams))

        # The extrinsics were composed with the inverse of the augmentation, so composing
        # them with the augmentation itself brings them back.
        self.assertTrue(
            torch.allclose(
                camera_image_data.lidar2cams @ transformation_matrix,
                self.original_lidar2cams,
                atol=1e-5,
            )
        )
        self.assertTrue(
            torch.allclose(
                camera_image_data.lidar2images @ transformation_matrix,
                self.original_lidar2images,
                atol=1e-5,
            )
        )

    def test_missing_camera_image_data_key(self) -> None:
        """Test that missing 'camera_image_data' raises KeyError."""
        sample = self.sample._replace(camera_image_data=None)
        transform = GlobalRotScaleTrans(yaw_rot_range=[0.0, 0.0], scale_ratio_range=[1.0, 1.0])

        with self.assertRaises(KeyError):
            transform(sample)


class TestGlobalBEVRandomFlip(BaseCameraGeometryTestCase):
    """Unit tests for the camera-only GlobalBEVRandomFlip transform."""

    def setUp(self) -> None:
        """Set up the same sample for all tests."""
        self.sample = self.build_multi_task_gt_sample()
        assert self.sample.detection3d_gt_bboxes_3d is not None
        assert self.sample.camera_image_data is not None
        self.original_bbox_params = self.sample.detection3d_gt_bboxes_3d.bbox_params.clone()
        self.original_lidar2cams = self.sample.camera_image_data.lidar2cams.clone()
        self.original_lidar2images = self.sample.camera_image_data.lidar2images.clone()

    def test_flips_bboxes_and_camera_matrices(self) -> None:
        """Test that both BEV flips negate x and y for the bboxes and the cameras."""
        output = GlobalBEVRandomFlip(horizontal_flip_ratio=1.0, vertical_flip_ratio=1.0)(
            self.sample
        )

        expected_flip = torch.diag(torch.tensor([-1.0, -1.0, 1.0, 1.0]))
        assert output.lidar_transformation_sample is not None
        assert output.detection3d_gt_bboxes_3d is not None
        assert output.camera_image_data is not None
        lidar_transformation_sample = output.lidar_transformation_sample
        self.assertTrue(
            torch.allclose(lidar_transformation_sample.transformation_matrix, expected_flip)
        )
        self.assertEqual(
            lidar_transformation_sample.transformation_order,
            [TransformationName.HORIZONTAL_FLIP, TransformationName.VERTICAL_FLIP],
        )
        self.assertTrue(
            torch.allclose(
                output.detection3d_gt_bboxes_3d.bbox_params[:, :2],
                -self.original_bbox_params[:, :2],
                atol=1e-5,
            )
        )
        self.assertTrue(
            torch.allclose(
                output.camera_image_data.lidar2cams,
                self.original_lidar2cams @ torch.linalg.inv(expected_flip),
                atol=1e-5,
            )
        )

    def test_no_flip_keeps_identity(self) -> None:
        """Test that a zero flip ratio saves an identity transformation."""
        output = GlobalBEVRandomFlip(horizontal_flip_ratio=0.0, vertical_flip_ratio=0.0)(
            self.sample
        )

        assert output.lidar_transformation_sample is not None
        assert output.detection3d_gt_bboxes_3d is not None
        assert output.camera_image_data is not None
        self.assertTrue(
            torch.allclose(output.lidar_transformation_sample.transformation_matrix, torch.eye(4))
        )
        self.assertEqual(output.lidar_transformation_sample.transformation_order, [])
        self.assertTrue(
            torch.allclose(output.detection3d_gt_bboxes_3d.bbox_params, self.original_bbox_params)
        )
        self.assertTrue(
            torch.allclose(output.camera_image_data.lidar2cams, self.original_lidar2cams)
        )

    def test_horizontal_flip_only_negates_lateral_axis(self) -> None:
        """Test that a horizontal flip mirrors the lateral (y) axis only."""
        output = GlobalBEVRandomFlip(horizontal_flip_ratio=1.0, vertical_flip_ratio=0.0)(
            self.sample
        )

        assert output.lidar_transformation_sample is not None
        assert output.detection3d_gt_bboxes_3d is not None
        self.assertTrue(
            torch.allclose(
                output.lidar_transformation_sample.transformation_matrix,
                torch.diag(torch.tensor([1.0, -1.0, 1.0, 1.0])),
            )
        )
        bbox_params = output.detection3d_gt_bboxes_3d.bbox_params
        self.assertTrue(torch.allclose(bbox_params[:, 0], self.original_bbox_params[:, 0]))
        self.assertTrue(torch.allclose(bbox_params[:, 1], -self.original_bbox_params[:, 1]))

    def test_composes_with_previous_transformation(self) -> None:
        """Test that the flip is composed with an earlier transformation in the pipeline."""
        transform = GlobalBEVRandomFlip(horizontal_flip_ratio=1.0, vertical_flip_ratio=0.0)

        output = transform(transform(self.sample))

        assert output.lidar_transformation_sample is not None
        # Flipping twice along the same axis returns to the identity.
        self.assertTrue(
            torch.allclose(output.lidar_transformation_sample.transformation_matrix, torch.eye(4))
        )
        self.assertEqual(
            output.lidar_transformation_sample.transformation_order,
            [TransformationName.HORIZONTAL_FLIP, TransformationName.HORIZONTAL_FLIP],
        )

    def test_inverse_transformation_recovers_original_bboxes(self) -> None:
        """Test that inversely applying the saved matrix restores the original bboxes."""
        output = GlobalBEVRandomFlip(horizontal_flip_ratio=1.0, vertical_flip_ratio=1.0)(
            self.sample
        )

        assert output.detection3d_gt_bboxes_3d is not None
        assert output.lidar_transformation_sample is not None
        bboxes_3d = output.detection3d_gt_bboxes_3d
        self.assertFalse(torch.allclose(bboxes_3d.bbox_params, self.original_bbox_params))

        self.apply_inverse_transformation(
            bboxes_3d, output.lidar_transformation_sample.transformation_matrix
        )

        self.assert_bbox_params_close(bboxes_3d.bbox_params, self.original_bbox_params)

    def test_inverse_transformation_recovers_original_camera_matrices(self) -> None:
        """Test that inversely applying the saved matrix restores the camera matrices."""
        output = GlobalBEVRandomFlip(horizontal_flip_ratio=1.0, vertical_flip_ratio=0.0)(
            self.sample
        )

        assert output.camera_image_data is not None
        assert output.lidar_transformation_sample is not None
        camera_image_data = output.camera_image_data
        transformation_matrix = output.lidar_transformation_sample.transformation_matrix
        self.assertFalse(torch.allclose(camera_image_data.lidar2cams, self.original_lidar2cams))

        # The extrinsics were composed with the inverse of the flip, so composing them with
        # the flip itself brings them back.
        self.assertTrue(
            torch.allclose(
                camera_image_data.lidar2cams @ transformation_matrix,
                self.original_lidar2cams,
                atol=1e-5,
            )
        )
        self.assertTrue(
            torch.allclose(
                camera_image_data.lidar2images @ transformation_matrix,
                self.original_lidar2images,
                atol=1e-5,
            )
        )

    def test_missing_camera_image_data_key(self) -> None:
        """Test that missing 'camera_image_data' raises KeyError."""
        sample = self.sample._replace(camera_image_data=None)

        with self.assertRaises(KeyError):
            GlobalBEVRandomFlip()(sample)


if __name__ == "__main__":
    unittest.main()
