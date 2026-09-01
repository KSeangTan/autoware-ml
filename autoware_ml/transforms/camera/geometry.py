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

"""Camera-only geometric augmentations.

For camera-only detectors (e.g. StreamPETR) there is no point cloud: the scene is
represented by `detection3d_gt_bboxes_3d` and the camera projection matrices. These
transforms apply the same global rotation/scale/translation and BEV flip math as the
lidar variants in `transforms.multi_task.point_cloud.geometry` to the bboxes and the
camera matrices only. They require `camera_image_data` and deliberately never touch
points.

The augmentation rewrites the coordinates of the scene, it does not move the cameras and
it never re-renders the images. The camera extrinsics are therefore re-expressed in the
augmented frame, by composing them with the inverse of the augmentation, so that a point
keeps projecting onto the same pixel of the unchanged image.

The sampled augmentation is saved as a 4x4 matrix in `lidar_transformation_sample`,
composed with any transformation applied earlier in the pipeline, so it can be reversed
later (e.g. when doing LSS).
"""

from __future__ import annotations

from typing import Sequence, Tuple

from jaxtyping import Float32
import numpy as np
import torch
from torch import Tensor

from autoware_ml.datamodule.multi_task.dataclasses.multi_task_samples import (
    MultiTaskGTSample,
)
from autoware_ml.datamodule.multi_task.dataclasses.transformation import LiDARTransformationSample
from autoware_ml.geometry.cameras.base_images import BaseImages
from autoware_ml.transforms.multi_task.base import MultiTaskBaseTransform
from autoware_ml.transforms.multi_task.point_cloud.geometry import RotationScaleTranslationData
from autoware_ml.transforms.geometry3d import rotation_matrix
from autoware_ml.types.spatial import RotationAxis, BEVDirection
from autoware_ml.types.geometry import TransformationName


class GlobalRotScaleTrans(MultiTaskBaseTransform):
    """Apply global rotation, scaling, and optional translation to bboxes and camera matrices."""

    _required_keys = ["camera_image_data"]

    def __init__(
        self,
        yaw_rot_range: Sequence[float],
        scale_ratio_range: Sequence[float],
        translation_std: Sequence[float] | None = None,
    ) -> None:
        """Initialize the GlobalRotScaleTrans transform.

        Args:
            yaw_rot_range: Min and max rotation angles in radians around yaw.
            scale_ratio_range: Min and max scale factors.
            translation_std: Optional per-axis Gaussian translation std ``[x, y, z]``.
        """
        super().__init__(probability=None)
        self.yaw_rot_range = yaw_rot_range
        self.scale_ratio_range = scale_ratio_range
        self.translation_std = (
            torch.tensor(translation_std, dtype=torch.float32)
            if translation_std is not None
            else None
        )

    def sample_rot_scale_trans(
        self,
    ) -> Tuple[LiDARTransformationSample, RotationScaleTranslationData]:
        """
        Sample random rotation, scale, and translation parameters.
        """
        rotation = float(np.random.uniform(self.yaw_rot_range[0], self.yaw_rot_range[1]))
        matrix = rotation_matrix(str(RotationAxis.Z.name).lower(), rotation)
        scale_factor = float(
            np.random.uniform(self.scale_ratio_range[0], self.scale_ratio_range[1])
        )
        if self.translation_std is not None:
            translation = np.random.normal(0.0, self.translation_std, size=(1, 3)).astype(
                np.float32
            )
        else:
            translation = np.zeros((1, 3), dtype=np.float32)

        # Convert to torch tensor
        rotation_matrix_tensor = torch.tensor(matrix, dtype=torch.float32)
        translation_tensor = torch.tensor(translation, dtype=torch.float32)
        transformation_order = [
            TransformationName.ROTATION,
            TransformationName.SCALING,
            TransformationName.TRANSLATION,
        ]

        rotation_scale_translation_data = RotationScaleTranslationData(
            rotation_matrix=rotation_matrix_tensor,
            scale_factor=scale_factor,
            translation_vector=translation_tensor,
        )
        lidar_transformation_sample = LiDARTransformationSample.create_lidar_transformation_sample(
            rotation_matrix=rotation_matrix_tensor,
            scale_factor=scale_factor,
            translation_vector=translation_tensor,
            transformation_order=transformation_order,
        )
        return lidar_transformation_sample, rotation_scale_translation_data

    def transform(self, multi_task_gt_sample: MultiTaskGTSample) -> MultiTaskGTSample:
        """Rotate, scale, and translate bboxes and camera matrices."""
        # This is checked in the _validate_required_keys()
        camera_image_data: BaseImages = multi_task_gt_sample.camera_image_data  # type: ignore[reportOptionalMemberAccess]

        # Sample rotation, scale, and translation parameters
        lidar_transformation_sample, rotation_scale_translation_data = self.sample_rot_scale_trans()

        # Convert to row vector convention for bounding box transformation
        row_vector_rotation_matrix = rotation_scale_translation_data.rotation_matrix.T
        scale_factor = rotation_scale_translation_data.scale_factor
        translation_vector = rotation_scale_translation_data.translation_vector

        # Rotate, scale, and translate the 3D bounding boxes
        if multi_task_gt_sample.detection3d_gt_bboxes_3d is not None:
            multi_task_gt_sample.detection3d_gt_bboxes_3d.rotate(row_vector_rotation_matrix)
            multi_task_gt_sample.detection3d_gt_bboxes_3d.scale(scale_factor)
            multi_task_gt_sample.detection3d_gt_bboxes_3d.translate(translation_vector)

        # Keep the camera projection consistent with the augmented scene. The image pixels
        # are never re-rendered, so a point must still project onto the same pixel:
        # lidar2cam_augmented @ (augmentation @ point) == lidar2cam @ point. The camera
        # extrinsics are therefore re-expressed in the augmented frame by composing them
        # with the inverse of the augmentation.
        camera_image_data.update_lidar_transformation_matrices(
            torch.linalg.inv(lidar_transformation_sample.transformation_matrix)
        )

        # Create the composed transformation matrix in the MultiTaskGTSample if it exists
        if multi_task_gt_sample.lidar_transformation_sample is not None:
            lidar_transformation_sample = lidar_transformation_sample.create_composed_lidar_transformation_sample(
                previous_lidar_transformation_sample=multi_task_gt_sample.lidar_transformation_sample
            )

        return multi_task_gt_sample._replace(
            lidar_transformation_sample=lidar_transformation_sample
        )


class GlobalBEVRandomFlip(MultiTaskBaseTransform):
    """Globally and randomly flip bboxes and camera matrices along the BEV axes."""

    _required_keys = ["camera_image_data"]

    def __init__(
        self, horizontal_flip_ratio: float = 0.5, vertical_flip_ratio: float = 0.5
    ) -> None:
        """Initialize the GlobalBEVRandomFlip transform.

        Args:
            horizontal_flip_ratio: Ratio of flipping horizontally.
            vertical_flip_ratio: Ratio of flipping vertically.
        """
        super().__init__(probability=None)
        self.horizontal_flip_ratio = horizontal_flip_ratio
        self.vertical_flip_ratio = vertical_flip_ratio

    def sample_flip(self) -> Tuple[bool, bool]:
        """
        Sample random horizontal and vertical flips.
        """
        horizontal_flip = np.random.rand() < self.horizontal_flip_ratio
        vertical_flip = np.random.rand() < self.vertical_flip_ratio
        return horizontal_flip, vertical_flip

    def apply_flip(
        self,
        multi_task_gt_sample: MultiTaskGTSample,
        rotation_matrix: Float32[Tensor, "3 3"],
        bev_flip_direction: BEVDirection,
    ) -> Float32[Tensor, "3 3"]:
        """
        Apply the specified flip to the bboxes.

        Args:
            multi_task_gt_sample: The MultiTaskGTSample to apply the flip to.
            rotation_matrix: The rotation matrix accumulated by the previous flips.
            bev_flip_direction: The direction of the flip (horizontal (lateral) or vertical (longitudinal)).

        Returns:
            The rotation matrix with the flip applied.
        """
        if bev_flip_direction == BEVDirection.HORIZONTAL:
            rotation_matrix = (
                torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=torch.float32)
                @ rotation_matrix
            )
        elif bev_flip_direction == BEVDirection.VERTICAL:
            rotation_matrix = (
                torch.tensor([[-1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32)
                @ rotation_matrix
            )
        else:
            raise ValueError(
                f"Invalid flip direction: {bev_flip_direction}. Must be 'horizontal' or 'vertical'."
            )

        # Flip the 3D bounding boxes along the direction if they exist
        if multi_task_gt_sample.detection3d_gt_bboxes_3d is not None:
            multi_task_gt_sample.detection3d_gt_bboxes_3d.flip_bev(bev_direction=bev_flip_direction)

        return rotation_matrix

    def transform(self, multi_task_gt_sample: MultiTaskGTSample) -> MultiTaskGTSample:
        """Flip bboxes and camera matrices along the specified axis."""
        # This is checked in the _validate_required_keys()
        camera_image_data: BaseImages = multi_task_gt_sample.camera_image_data  # type: ignore[reportOptionalMemberAccess]

        rotation_matrix = torch.eye(3, dtype=torch.float32)
        horizontal_flip, vertical_flip = self.sample_flip()
        transformation_order = []

        if horizontal_flip:
            rotation_matrix = self.apply_flip(
                multi_task_gt_sample, rotation_matrix, bev_flip_direction=BEVDirection.HORIZONTAL
            )
            # Add the horizontal flip transformation to the transformation order
            transformation_order.append(TransformationName.HORIZONTAL_FLIP)

        if vertical_flip:
            rotation_matrix = self.apply_flip(
                multi_task_gt_sample, rotation_matrix, bev_flip_direction=BEVDirection.VERTICAL
            )
            # Add the vertical flip transformation to the transformation order
            transformation_order.append(TransformationName.VERTICAL_FLIP)

        # Create the lidar transformation sample
        lidar_transformation_sample = LiDARTransformationSample.create_lidar_transformation_sample(
            rotation_matrix=rotation_matrix,
            scale_factor=1.0,  # No scaling applied
            translation_vector=torch.zeros((1, 3), dtype=torch.float32),  # No translation applied
            transformation_order=transformation_order,
        )

        # Keep the camera projection consistent with the flipped scene. The image pixels are
        # never re-rendered, so the camera extrinsics are re-expressed in the flipped frame by
        # composing them with the inverse of the flip.
        camera_image_data.update_lidar_transformation_matrices(
            torch.linalg.inv(lidar_transformation_sample.transformation_matrix)
        )

        # Update the lidar transformation sample in the MultiTaskGTSample if it exists
        if multi_task_gt_sample.lidar_transformation_sample is not None:
            lidar_transformation_sample = lidar_transformation_sample.create_composed_lidar_transformation_sample(
                previous_lidar_transformation_sample=multi_task_gt_sample.lidar_transformation_sample
            )

        return multi_task_gt_sample._replace(
            lidar_transformation_sample=lidar_transformation_sample
        )
