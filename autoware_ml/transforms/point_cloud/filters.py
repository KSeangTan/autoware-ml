"""
Point cloud transforms operating on ``point_cloud_data`` only (range filtering).

The geometric augmentations (global rotation/scale/translation and BEV flips) live in
``transforms.camera_lidar.geometry``; they handle lidar-only, camera-only and fusion samples.
"""

from typing import Tuple

import torch

from autoware_ml.datamodule.multi_task.dataclasses.multi_task_samples import (
    MultiTaskGTSample,
)
from autoware_ml.geometry.points.base_points import BasePoints
from autoware_ml.transforms.multi_task.base import MultiTaskBaseTransform


class PointsRangeFilter(MultiTaskBaseTransform):
    """Filter points based on their range."""

    _required_keys = ["point_cloud_data"]

    def __init__(self, points_range: Tuple[float, float, float, float, float, float]) -> None:
        """Initialize the PointsRangeFilter transform.

        Args:
            points_range: The range of points to keep in the format (x_min, y_min, z_min, x_max, y_max, z_max).
        """
        super().__init__(probability=None)
        self.points_range = torch.tensor(points_range, dtype=torch.float32)

    def transform(self, multi_task_gt_sample: MultiTaskGTSample) -> MultiTaskGTSample:
        """Filter points based on the specified range."""
        # This is checked in the _validate_required_keys()
        point_cloud_data: BasePoints = multi_task_gt_sample.point_cloud_data  # type: ignore[reportOptionalMemberAccess]
        if not len(point_cloud_data):
            return multi_task_gt_sample

        point_cloud_range_mask = point_cloud_data.in_range_3d(self.points_range)

        # TODO(Kok Seang): Consider to make it immutable and return a new instance
        # instead of modifying in place.
        # TODO(Kok Seang): Need to remove labels outside of range for 3D semantic segmentation.
        point_cloud_data.remove_points(point_cloud_range_mask)
        return multi_task_gt_sample
