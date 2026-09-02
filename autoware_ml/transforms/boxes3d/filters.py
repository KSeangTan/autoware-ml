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

"""
Bboxes 3d transforms for filtering bboxes (for example, label name filter).
The code is modified based on https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/datasets/transforms/transforms_3d.py.
"""

from typing import Mapping, Sequence, Tuple

import torch

from autoware_ml.datamodule.multi_task.dataclasses.multi_task_samples import (
    MultiTaskGTSample,
)
from autoware_ml.geometry.bbox_3d.base_bbox3d import BaseBBoxes3D
from autoware_ml.geometry.points.base_points import BasePoints
from autoware_ml.transforms.multi_task.base import MultiTaskBaseTransform


class BBoxesLabelNameFilter(MultiTaskBaseTransform):
    """Filter 3D bounding boxes by label names."""

    _required_keys = ["detection3d_gt_bboxes_3d"]

    def __init__(self, label_names_to_keep: Sequence[str]) -> None:
        """Initialize the BBoxesLabelNameFilter transform."""
        super().__init__(probability=None)
        self.label_names_to_keep = label_names_to_keep

    def transform(self, multi_task_gt_sample: MultiTaskGTSample) -> MultiTaskGTSample:
        """Filter 3D bounding boxes by label names."""
        # This is checked in the _validate_required_keys()
        detection3d_gt_bboxes_3d: BaseBBoxes3D = multi_task_gt_sample.detection3d_gt_bboxes_3d  # type: ignore[reportOptionalMemberAccess]
        if not len(detection3d_gt_bboxes_3d):
            return multi_task_gt_sample

        bboxes_to_keep_mask = torch.tensor(
            [
                True if label_name in self.label_names_to_keep else False
                for label_name in detection3d_gt_bboxes_3d.bbox_label_names
            ],
            dtype=torch.bool,
        )

        # TODO(Kok Seang): Consider to make it immutable and return a new instance
        # instead of modifying in place.
        detection3d_gt_bboxes_3d.remove_bboxes(bboxes_to_keep_mask)
        return multi_task_gt_sample


class BBoxesAttributeFilter(MultiTaskBaseTransform):
    """Filter out 3D bounding boxes by their attributes, on a per label name basis.

    For example, the following configuration removes every parked or stopped vehicle, and every
    sitting pedestrian, while every other bounding box is kept:

    .. code-block:: python

        BBoxesAttributeFilter(
            attributes_to_filter={
                "vehicle": ["vehicle_state.parked", "vehicle_state.stopped"],
                "pedestrian": ["pedestrian_state.sitting"],
            }
        )
    """

    _required_keys = ["detection3d_gt_bboxes_3d"]

    def __init__(self, attributes_to_filter: Mapping[str, Sequence[str]]) -> None:
        """
        Initialize the BBoxesAttributeFilter transform.

        Args:
            attributes_to_filter (Mapping[str, Sequence[str]]): Mapping from a bbox label name to
                the attributes to filter out for that label name. A bounding box is removed when
                its label name is in the mapping and it carries at least one of the attributes
                listed for that label name. Label names that are absent from the mapping are
                left untouched.
        """
        super().__init__(probability=None)
        self.attributes_to_filter = {
            label_name: set(attributes) for label_name, attributes in attributes_to_filter.items()
        }

    def transform(self, multi_task_gt_sample: MultiTaskGTSample) -> MultiTaskGTSample:
        """Filter out 3D bounding boxes carrying the configured attributes."""
        # This is checked in the _validate_required_keys()
        detection3d_gt_bboxes_3d: BaseBBoxes3D = multi_task_gt_sample.detection3d_gt_bboxes_3d  # type: ignore[reportOptionalMemberAccess]
        if not len(detection3d_gt_bboxes_3d):
            return multi_task_gt_sample

        bbox_attributes = detection3d_gt_bboxes_3d.bbox_attributes
        if bbox_attributes is None:
            raise ValueError(
                f"{self.__class__.__name__}: The 3D bounding boxes do not carry any attribute, "
                "therefore they cannot be filtered by attributes."
            )

        bboxes_to_keep_mask = torch.tensor(
            [
                not self.attributes_to_filter.get(label_name, set()).intersection(attributes)
                for label_name, attributes in zip(
                    detection3d_gt_bboxes_3d.bbox_label_names, bbox_attributes
                )
            ],
            dtype=torch.bool,
        )

        # TODO(Kok Seang): Consider to make it immutable and return a new instance
        # instead of modifying in place.
        detection3d_gt_bboxes_3d.remove_bboxes(bboxes_to_keep_mask)
        return multi_task_gt_sample


class BBoxesMinPointsFilter(MultiTaskBaseTransform):
    """Filter 3D bounding boxes by minimum number of points and distance of bboxes."""

    _required_keys = ["detection3d_gt_bboxes_3d", "point_cloud_data"]

    def __init__(
        self,
        min_points: int,
        bev_range: Tuple[float, float, float, float],
    ) -> None:
        """
        Initialize the BBoxesMinPointsFilter transform.

        Args:
            min_points (int): The minimum number of points required for a bounding box to be kept.
            bev_range (Tuple[float, float, float, float]): The distance ([x_min, y_min, x_max, y_max]) of bounding boxes
                to apply the minimum number of points filtering.
        """
        super().__init__(probability=None)
        self.min_points = min_points
        self.bev_range = torch.tensor(bev_range, dtype=torch.float32)

    def transform(self, multi_task_gt_sample: MultiTaskGTSample) -> MultiTaskGTSample:
        """Filter 3D bounding boxes by label names."""
        # This is checked in the _validate_required_keys()
        detection3d_gt_bboxes_3d: BaseBBoxes3D = multi_task_gt_sample.detection3d_gt_bboxes_3d  # type: ignore[reportOptionalMemberAccess]
        if not len(detection3d_gt_bboxes_3d):
            return multi_task_gt_sample

        # This is checked in the _validate_required_keys()
        point_cloud_data: BasePoints = multi_task_gt_sample.point_cloud_data  # type: ignore[reportOptionalMemberAccess]

        distance_in_range_masks = detection3d_gt_bboxes_3d.in_range_bev(self.bev_range)
        points_in_bboxes = detection3d_gt_bboxes_3d.compute_points_in_bboxes(
            points=point_cloud_data.coords,
        )

        # Filter bboxes that are either within the specified distance range and
        # have at least `min_points` points,
        # or are outside the distance range (to keep them).
        keep_bboxes_mask = (
            points_in_bboxes.sum(dim=1) >= self.min_points
        ) & distance_in_range_masks | (~distance_in_range_masks)
        detection3d_gt_bboxes_3d.remove_bboxes(keep_bboxes_mask)
        return multi_task_gt_sample


class BBoxesBEVDistanceFilter(MultiTaskBaseTransform):
    """Filter 3D bounding boxes by their bev distance."""

    _required_keys = ["detection3d_gt_bboxes_3d"]

    def __init__(
        self,
        bev_range: Tuple[float, float, float, float],
    ) -> None:
        """
        Initialize the BBoxesBEVDistanceFilter transform.

        Args:
            bev_range (Tuple[float]): The distance ([x_min, y_min, x_max, y_max]) of bounding boxes
                to apply the BEV distance filtering.
        """
        super().__init__(probability=None)
        self.bev_range = torch.tensor(bev_range, dtype=torch.float32)

    def transform(self, multi_task_gt_sample: MultiTaskGTSample) -> MultiTaskGTSample:
        """Filter 3D bounding boxes by BEV distance."""
        # This is checked in the _validate_required_keys()
        detection3d_gt_bboxes_3d: BaseBBoxes3D = multi_task_gt_sample.detection3d_gt_bboxes_3d  # type: ignore[reportOptionalMemberAccess]
        if not len(detection3d_gt_bboxes_3d):
            return multi_task_gt_sample

        distance_in_range_masks = detection3d_gt_bboxes_3d.in_range_bev(self.bev_range)
        detection3d_gt_bboxes_3d.remove_bboxes(distance_in_range_masks)

        return multi_task_gt_sample


class BBoxesPhysicalFilter(MultiTaskBaseTransform):
    """Remove 3D bounding boxes that cannot become a valid training target.

    A physically invalid box is not a geometry outlier, it is pipeline garbage that would
    poison the regression losses: non-finite box parameters, non-positive dimensions (the
    box size targets are log-encoded, so a zero or negative extent is not representable),
    or a ground-plane speed beyond the physical bound (the velocity is never range-filtered
    downstream, so an absurd speed silently explodes the velocity loss).

    Note that the bbox parameters are already float32 here, so a float64 value that overflows
    the float32 cast has become non-finite by this point and is still caught.
    """

    _required_keys = ["detection3d_gt_bboxes_3d"]

    def __init__(
        self,
        max_absolute_speed: float = 150.0,  # m/s, 540 km/h
    ) -> None:
        """
        Initialize the BBoxesPhysicalFilter transform.

        Args:
            max_absolute_speed (float): The maximum ground-plane speed, in m/s, that a bounding
                box may carry before it is considered non-physical. Defaults to 150.0 m/s
        """
        super().__init__(probability=None)
        self.max_absolute_speed = max_absolute_speed

    def transform(self, multi_task_gt_sample: MultiTaskGTSample) -> MultiTaskGTSample:
        """Remove non-physical 3D bounding boxes."""
        # This is checked in the _validate_required_keys()
        detection3d_gt_bboxes_3d: BaseBBoxes3D = multi_task_gt_sample.detection3d_gt_bboxes_3d  # type: ignore[reportOptionalMemberAccess]
        if not len(detection3d_gt_bboxes_3d):
            return multi_task_gt_sample

        is_finite_mask = torch.isfinite(detection3d_gt_bboxes_3d.bbox_params).all(dim=1)
        has_positive_dims_mask = (detection3d_gt_bboxes_3d.dims > 0.0).all(dim=1)
        # Only the ground-plane velocity is bounded.
        is_speed_physical_mask = (
            torch.linalg.norm(detection3d_gt_bboxes_3d.velocity[:, :2], dim=1)
            <= self.max_absolute_speed
        )

        keep_bboxes_mask = is_finite_mask & has_positive_dims_mask & is_speed_physical_mask

        # TODO(Kok Seang): Consider to make it immutable and return a new instance
        # instead of modifying in place.
        detection3d_gt_bboxes_3d.remove_bboxes(keep_bboxes_mask)
        return multi_task_gt_sample
