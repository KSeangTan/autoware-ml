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

"""Unit tests for the 3D bounding box filters."""

from typing import Sequence

import unittest

import numpy as np
import torch

from autoware_ml.datamodule.multi_task.dataclasses.multi_task_samples import MultiTaskGTSample
from autoware_ml.geometry.bbox_3d.base_bbox3d import BaseBBoxes3D
from autoware_ml.geometry.bbox_3d.lidar_bbox3d import LidarBBoxes3D
from autoware_ml.geometry.points.base_points import BasePoints
from autoware_ml.geometry.points.lidar_points import LiDARPoints
from autoware_ml.transforms.boxes3d.annotations import (
    MAX_ABSOLUTE_SPEED,
    normalize_filter_attributes,
)
from autoware_ml.transforms.boxes3d.filters import (
    BBoxesAttributeFilter,
    BBoxesBEVDistanceFilter,
    BBoxesLabelNameFilter,
    BBoxesMinPointsFilter,
    BBoxesPhysicalFilter,
)
from autoware_ml.types.geometry import (
    Box3DCenterCoordinateType,
    Box3DFieldIndex,
    PointFeatureName,
)

# A BEV range wide enough that every bounding box built by these tests falls inside it, so a
# filter that is range-aware behaves as an unconditional filter.
UNBOUNDED_BEV_RANGE = (-1000.0, -1000.0, 1000.0, 1000.0)


class BaseBBoxesFilterTestCase(unittest.TestCase):
    """Shared sample builders and optional field assertions for the 3D bounding box filters."""

    def build_bboxes_3d(
        self,
        bbox_params: Sequence[Sequence[float]],
        bbox_label_names: Sequence[str],
        bbox_num_lidar_points: Sequence[int] | None = None,
        bbox_attributes: Sequence[Sequence[str]] | None = None,
    ) -> LidarBBoxes3D:
        """Build bounding boxes from rows of ``[x, y, z, l, w, h, yaw, vx, vy, vz]``.

        Args:
            bbox_params: One row of the ten box parameters per bounding box.
            bbox_label_names: The label name of every bounding box.
            bbox_num_lidar_points: The annotated point count of every bounding box. Defaults to
                zero for every box, since most filters ignore it.
            bbox_attributes: The attributes of every bounding box, or None when the source
                dataset carries none.

        Returns:
            LidarBBoxes3D holding the requested bounding boxes.
        """
        num_bboxes = len(bbox_params)
        if bbox_num_lidar_points is None:
            bbox_num_lidar_points = [0] * num_bboxes

        return LidarBBoxes3D(
            # The reshape keeps an empty sequence a (0, num_fields) tensor rather than a (0,) one.
            bbox_params=torch.tensor(bbox_params, dtype=torch.float32).reshape(
                -1, len(Box3DFieldIndex)
            ),
            bbox_labels=torch.arange(num_bboxes, dtype=torch.int32),
            bbox_label_names=list(bbox_label_names),
            bbox_num_lidar_points=torch.tensor(bbox_num_lidar_points, dtype=torch.int32),
            bbox_center_coordinate_type=Box3DCenterCoordinateType.GRAVITY_CENTER,
            bbox_attributes=bbox_attributes,
        )

    def build_point_cloud_data(self, coords: Sequence[Sequence[float]]) -> LiDARPoints:
        """Build a point cloud holding only the xyz coordinates of every point.

        Args:
            coords: One ``[x, y, z]`` row per point.

        Returns:
            LiDARPoints holding the requested points.
        """
        return LiDARPoints(
            points=torch.tensor(coords, dtype=torch.float32),
            point_feature_names=[PointFeatureName.X, PointFeatureName.Y, PointFeatureName.Z],
            timestamp=0.0,
        )

    def build_multi_task_gt_sample(
        self,
        detection3d_gt_bboxes_3d: LidarBBoxes3D | None = None,
        point_cloud_data: LiDARPoints | None = None,
    ) -> MultiTaskGTSample:
        """Build a minimal sample holding only what the box filters read.

        Args:
            detection3d_gt_bboxes_3d: The bounding boxes to filter.
            point_cloud_data: The point cloud the point-count filter reads.

        Returns:
            MultiTaskGTSample carrying the given bounding boxes and points.
        """
        return MultiTaskGTSample(
            lidar_point_cloud_samples=None,
            image_samples=None,
            point_cloud_data=point_cloud_data,
            camera_image_data=None,
            detection3d_gt_bboxes_3d=detection3d_gt_bboxes_3d,
            segmentation3d_gt_sample=None,
        )

    def build_two_bboxes_sample(self) -> MultiTaskGTSample:
        """Build the sample shared by the label name and BEV distance filter tests.

        Returns:
            MultiTaskGTSample holding a car at the origin and a pedestrian ten metres ahead,
            with seven and three annotated points respectively.
        """
        detection3d_gt_bboxes_3d = self.build_bboxes_3d(
            bbox_params=[
                [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ],
            bbox_label_names=["car", "pedestrian"],
            bbox_num_lidar_points=[7, 3],
        )
        point_cloud_data = self.build_point_cloud_data(
            coords=[
                [0.1, 0.1, 0.0],
                [-0.2, 0.0, 0.0],
                [0.0, -0.2, 0.0],
                [10.1, 0.0, 0.0],
            ],
        )
        return self.build_multi_task_gt_sample(
            detection3d_gt_bboxes_3d=detection3d_gt_bboxes_3d,
            point_cloud_data=point_cloud_data,
        )

    def assert_bboxes_3d(self, multi_task_gt_sample: MultiTaskGTSample) -> BaseBBoxes3D:
        """Assert the sample still carries bounding boxes, and return them.

        ``detection3d_gt_bboxes_3d`` is optional on the sample, so every dereference below
        goes through this helper: a filter that removes every bounding box must still leave an
        empty container behind, never None, otherwise the transforms downstream of it would
        fail their own required key validation.

        Args:
            multi_task_gt_sample: The sample a filter returned.

        Returns:
            The 3D bounding boxes the sample carries.
        """
        detection3d_gt_bboxes_3d = multi_task_gt_sample.detection3d_gt_bboxes_3d
        self.assertIsNotNone(
            detection3d_gt_bboxes_3d,
            "The filter dropped the detection3d_gt_bboxes_3d field from the sample.",
        )
        # Narrow the optional, so what follows is checked against the concrete type.
        assert detection3d_gt_bboxes_3d is not None
        return detection3d_gt_bboxes_3d

    def assert_point_cloud_data(self, multi_task_gt_sample: MultiTaskGTSample) -> BasePoints:
        """Assert the sample still carries its point cloud, and return it.

        The box filters only ever read the points, so a filter that leaves ``point_cloud_data``
        as None has consumed an input the rest of the pipeline still needs.

        Args:
            multi_task_gt_sample: The sample a filter returned.

        Returns:
            The point cloud the sample carries.
        """
        point_cloud_data = multi_task_gt_sample.point_cloud_data
        self.assertIsNotNone(
            point_cloud_data, "The filter dropped the point_cloud_data field from the sample."
        )
        # Narrow the optional, so what follows is checked against the concrete type.
        assert point_cloud_data is not None
        return point_cloud_data

    def assert_bbox_attributes(self, bboxes_3d: BaseBBoxes3D) -> Sequence[Sequence[str]]:
        """Assert the bounding boxes still carry attributes, and return them.

        ``bbox_attributes`` is optional because not every dataset provides attributes, so a
        filter must keep the attributes of the boxes it kept rather than discarding the field.

        Args:
            bboxes_3d: The bounding boxes a filter returned.

        Returns:
            The attributes of every bounding box.
        """
        bbox_attributes = bboxes_3d.bbox_attributes
        self.assertIsNotNone(
            bbox_attributes, "The filter dropped the attributes of the bounding boxes."
        )
        # Narrow the optional, so what follows is checked against the concrete type.
        assert bbox_attributes is not None
        return bbox_attributes


class BBoxesLabelNameFilterTest(BaseBBoxesFilterTestCase):
    """Tests for BBoxesLabelNameFilter."""

    def test_keeps_only_the_configured_label_names(self) -> None:
        """Every bounding box whose label name is not configured is removed."""
        multi_task_gt_sample = self.build_two_bboxes_sample()

        output = BBoxesLabelNameFilter(label_names_to_keep=["car"])(multi_task_gt_sample)

        filtered_bboxes_3d = self.assert_bboxes_3d(output)
        self.assertEqual(list(filtered_bboxes_3d.bbox_label_names), ["car"])

    def test_keeps_the_per_bbox_fields_aligned(self) -> None:
        """The labels and point counts follow the bounding boxes that survive the filter."""
        multi_task_gt_sample = self.build_two_bboxes_sample()

        output = BBoxesLabelNameFilter(label_names_to_keep=["car"])(multi_task_gt_sample)

        filtered_bboxes_3d = self.assert_bboxes_3d(output)
        self.assertEqual(len(filtered_bboxes_3d), 1)
        self.assertEqual(filtered_bboxes_3d.bbox_labels.tolist(), [0])
        self.assertEqual(filtered_bboxes_3d.bbox_num_lidar_points.tolist(), [7])

    def test_leaves_empty_bboxes_untouched(self) -> None:
        """An empty set of bounding boxes is returned as is instead of failing."""
        multi_task_gt_sample = self.build_multi_task_gt_sample(
            detection3d_gt_bboxes_3d=self.build_bboxes_3d(
                bbox_params=[],
                bbox_label_names=[],
            ),
        )

        output = BBoxesLabelNameFilter(label_names_to_keep=["car"])(multi_task_gt_sample)

        self.assertEqual(len(self.assert_bboxes_3d(output)), 0)

    def test_rejects_a_sample_without_bboxes(self) -> None:
        """A sample missing the required bounding boxes is reported instead of skipped."""
        multi_task_gt_sample = self.build_multi_task_gt_sample()

        with self.assertRaisesRegex(KeyError, "detection3d_gt_bboxes_3d"):
            BBoxesLabelNameFilter(label_names_to_keep=["car"])(multi_task_gt_sample)


class BBoxesAttributeFilterTest(BaseBBoxesFilterTestCase):
    """Tests for BBoxesAttributeFilter."""

    def build_attributed_sample(self) -> MultiTaskGTSample:
        """Build a sample holding one parked, one moving, and one sitting bounding box.

        Returns:
            MultiTaskGTSample whose bounding boxes each carry a single attribute.
        """
        detection3d_gt_bboxes_3d = self.build_bboxes_3d(
            bbox_params=[
                [0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0, 0.0, 0.0, 0.0],
                [5.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0, 0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0, 0.8, 0.8, 1.7, 0.0, 0.0, 0.0, 0.0],
            ],
            bbox_label_names=["vehicle", "vehicle", "pedestrian"],
            bbox_attributes=[
                ["vehicle_state.parked"],
                ["vehicle_state.moving"],
                ["pedestrian_state.sitting"],
            ],
        )
        return self.build_multi_task_gt_sample(detection3d_gt_bboxes_3d=detection3d_gt_bboxes_3d)

    def test_removes_the_configured_attributes_per_label_name(self) -> None:
        """Only the boxes carrying a filtered attribute for their own label name are removed."""
        multi_task_gt_sample = self.build_attributed_sample()

        output = BBoxesAttributeFilter(
            attributes_to_filter={
                "vehicle": ["vehicle_state.parked", "vehicle_state.stopped"],
                "pedestrian": ["pedestrian_state.sitting"],
            }
        )(multi_task_gt_sample)

        filtered_bboxes_3d = self.assert_bboxes_3d(output)
        self.assertEqual(list(filtered_bboxes_3d.bbox_label_names), ["vehicle"])
        self.assertEqual(
            self.assert_bbox_attributes(filtered_bboxes_3d), [["vehicle_state.moving"]]
        )

    def test_leaves_label_names_absent_from_the_mapping_untouched(self) -> None:
        """A bounding box is never removed for an attribute configured under another label."""
        multi_task_gt_sample = self.build_attributed_sample()

        output = BBoxesAttributeFilter(
            attributes_to_filter={"pedestrian": ["vehicle_state.parked"]}
        )(multi_task_gt_sample)

        filtered_bboxes_3d = self.assert_bboxes_3d(output)
        self.assertEqual(len(filtered_bboxes_3d), 3)
        self.assertEqual(len(self.assert_bbox_attributes(filtered_bboxes_3d)), 3)

    def test_rejects_bboxes_without_attributes(self) -> None:
        """Filtering by attributes on boxes that carry none is an error, not a silent no-op."""
        multi_task_gt_sample = self.build_two_bboxes_sample()

        with self.assertRaisesRegex(ValueError, "do not carry any attribute"):
            BBoxesAttributeFilter(attributes_to_filter={"car": ["vehicle_state.parked"]})(
                multi_task_gt_sample
            )


class BBoxesBEVDistanceFilterTest(BaseBBoxesFilterTestCase):
    """Tests for BBoxesBEVDistanceFilter."""

    def test_keeps_only_the_bboxes_centred_inside_the_bev_range(self) -> None:
        """Every bounding box whose BEV centre is outside the range is removed."""
        multi_task_gt_sample = self.build_two_bboxes_sample()

        output = BBoxesBEVDistanceFilter(bev_range=(-1.0, -1.0, 5.0, 5.0))(multi_task_gt_sample)

        filtered_bboxes_3d = self.assert_bboxes_3d(output)
        self.assertEqual(list(filtered_bboxes_3d.bbox_label_names), ["car"])

    def test_keeps_the_per_bbox_fields_aligned(self) -> None:
        """The labels and point counts follow the bounding boxes that survive the filter."""
        multi_task_gt_sample = self.build_two_bboxes_sample()

        output = BBoxesBEVDistanceFilter(bev_range=(-1.0, -1.0, 5.0, 5.0))(multi_task_gt_sample)

        filtered_bboxes_3d = self.assert_bboxes_3d(output)
        self.assertEqual(filtered_bboxes_3d.bbox_labels.tolist(), [0])
        self.assertEqual(filtered_bboxes_3d.bbox_num_lidar_points.tolist(), [7])


class BBoxesMinPointsFilterTest(BaseBBoxesFilterTestCase):
    """Tests for BBoxesMinPointsFilter."""

    def test_removes_bboxes_holding_too_few_points(self) -> None:
        """Inside the BEV range, a bounding box needs at least ``min_points`` points."""
        multi_task_gt_sample = self.build_two_bboxes_sample()

        output = BBoxesMinPointsFilter(min_points=3, bev_range=UNBOUNDED_BEV_RANGE)(
            multi_task_gt_sample
        )

        # The car holds three points, the pedestrian only one.
        filtered_bboxes_3d = self.assert_bboxes_3d(output)
        self.assertEqual(list(filtered_bboxes_3d.bbox_label_names), ["car"])

    def test_leaves_the_point_cloud_untouched(self) -> None:
        """The filter reads the points to count them, it never consumes them."""
        multi_task_gt_sample = self.build_two_bboxes_sample()
        num_points = len(self.assert_point_cloud_data(multi_task_gt_sample))

        output = BBoxesMinPointsFilter(min_points=3, bev_range=UNBOUNDED_BEV_RANGE)(
            multi_task_gt_sample
        )

        self.assertEqual(len(self.assert_point_cloud_data(output)), num_points)

    def test_keeps_the_per_bbox_fields_aligned(self) -> None:
        """The labels and point counts follow the bounding boxes that survive the filter."""
        detection3d_gt_bboxes_3d = self.build_bboxes_3d(
            bbox_params=[
                [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ],
            bbox_label_names=["car", "pedestrian"],
            bbox_num_lidar_points=[7, 3],
        )
        multi_task_gt_sample = self.build_multi_task_gt_sample(
            detection3d_gt_bboxes_3d=detection3d_gt_bboxes_3d,
            point_cloud_data=self.build_point_cloud_data(coords=[[0.1, 0.1, 0.0]]),
        )

        output = BBoxesMinPointsFilter(min_points=1, bev_range=UNBOUNDED_BEV_RANGE)(
            multi_task_gt_sample
        )

        filtered_bboxes_3d = self.assert_bboxes_3d(output)
        self.assertEqual(filtered_bboxes_3d.bbox_labels.tolist(), [0])
        self.assertEqual(filtered_bboxes_3d.bbox_num_lidar_points.tolist(), [7])

    def test_keeps_bboxes_outside_the_bev_range_regardless_of_their_points(self) -> None:
        """A bounding box outside the range is never judged on its point count."""
        multi_task_gt_sample = self.build_two_bboxes_sample()

        # The pedestrian at x = 10 falls outside the range, so its single point is irrelevant.
        output = BBoxesMinPointsFilter(min_points=3, bev_range=(-1.0, -1.0, 5.0, 5.0))(
            multi_task_gt_sample
        )

        filtered_bboxes_3d = self.assert_bboxes_3d(output)
        self.assertEqual(list(filtered_bboxes_3d.bbox_label_names), ["car", "pedestrian"])

    def test_counts_points_inside_rotated_bboxes(self) -> None:
        """The point count honours the box yaw rather than its axis-aligned extent."""
        # The box is two metres long and one metre wide, yawed by a quarter turn, so its long
        # axis lies along y and the point at y = 0.6 is inside. An axis-aligned count would
        # place that point outside and leave only two points in the box.
        detection3d_gt_bboxes_3d = self.build_bboxes_3d(
            bbox_params=[[0.0, 0.0, 0.0, 2.0, 1.0, 2.0, np.pi / 2, 0.0, 0.0, 0.0]],
            bbox_label_names=["car"],
        )
        multi_task_gt_sample = self.build_multi_task_gt_sample(
            detection3d_gt_bboxes_3d=detection3d_gt_bboxes_3d,
            point_cloud_data=self.build_point_cloud_data(
                coords=[
                    [0.0, 0.4, 0.0],
                    [0.0, -0.4, 0.0],
                    [0.0, 0.6, 0.0],
                ],
            ),
        )

        output = BBoxesMinPointsFilter(min_points=3, bev_range=UNBOUNDED_BEV_RANGE)(
            multi_task_gt_sample
        )

        self.assertEqual(len(self.assert_bboxes_3d(output)), 1)

    def test_chained_filters_apply_a_distance_specific_threshold(self) -> None:
        """Chaining two ranges raises the point threshold for the closer bounding boxes."""
        detection3d_gt_bboxes_3d = self.build_bboxes_3d(
            bbox_params=[
                [10.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0],
                [70.0, 0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0],
            ],
            bbox_label_names=["car", "car"],
        )
        multi_task_gt_sample = self.build_multi_task_gt_sample(
            detection3d_gt_bboxes_3d=detection3d_gt_bboxes_3d,
            point_cloud_data=self.build_point_cloud_data(
                coords=[
                    [10.0, 0.0, 0.0],
                    [10.1, 0.0, 0.0],
                    [10.2, 0.0, 0.0],
                    [10.3, 0.0, 0.0],
                    [70.0, 0.0, 0.0],
                    [70.1, 0.0, 0.0],
                    [70.2, 0.0, 0.0],
                ],
            ),
        )

        # Within 60 metres a box needs five points, so the near box and its four points go.
        # The far box is outside that range, hence only judged by the wider filter's threshold
        # of three points, which it meets.
        near_filtered = BBoxesMinPointsFilter(min_points=5, bev_range=(-60.0, -60.0, 60.0, 60.0))(
            multi_task_gt_sample
        )
        output = BBoxesMinPointsFilter(min_points=3, bev_range=(-130.0, -130.0, 130.0, 130.0))(
            near_filtered
        )

        filtered_bboxes_3d = self.assert_bboxes_3d(output)
        self.assertEqual(filtered_bboxes_3d.center[:, 0].tolist(), [70.0])


class BBoxesPhysicalFilterTest(BaseBBoxesFilterTestCase):
    """Tests for BBoxesPhysicalFilter."""

    def build_sane_bboxes_3d(self, num_bboxes: int = 1) -> LidarBBoxes3D:
        """Build bounding boxes that every physical rule accepts.

        Args:
            num_bboxes: Number of identical bounding boxes to build.

        Returns:
            LidarBBoxes3D holding physically valid bounding boxes.
        """
        return self.build_bboxes_3d(
            bbox_params=[[1.0, 2.0, 3.0, 4.0, 1.5, 1.7, 0.1, 0.5, -0.1, 0.0]] * num_bboxes,
            bbox_label_names=["car"] * num_bboxes,
        )

    def test_keeps_physically_valid_bboxes(self) -> None:
        """A sane bounding box survives the filter untouched."""
        multi_task_gt_sample = self.build_multi_task_gt_sample(
            detection3d_gt_bboxes_3d=self.build_sane_bboxes_3d(),
        )

        output = BBoxesPhysicalFilter()(multi_task_gt_sample)

        self.assertEqual(len(self.assert_bboxes_3d(output)), 1)

    def test_removes_bboxes_holding_non_finite_parameters(self) -> None:
        """A NaN or infinite box parameter removes the bounding box."""
        detection3d_gt_bboxes_3d = self.build_sane_bboxes_3d(num_bboxes=3)
        detection3d_gt_bboxes_3d.bbox_params[1, Box3DFieldIndex.X] = float("inf")
        detection3d_gt_bboxes_3d.bbox_params[2, Box3DFieldIndex.YAW] = float("nan")
        multi_task_gt_sample = self.build_multi_task_gt_sample(
            detection3d_gt_bboxes_3d=detection3d_gt_bboxes_3d,
        )

        output = BBoxesPhysicalFilter()(multi_task_gt_sample)

        self.assertEqual(self.assert_bboxes_3d(output).bbox_labels.tolist(), [0])

    def test_removes_bboxes_holding_non_positive_dimensions(self) -> None:
        """A zero or negative extent removes the bounding box, since sizes are log-encoded."""
        # The bounding box constructor rejects non-positive dimensions, so the corruption is
        # injected afterwards, the way an in-place augmentation would produce it.
        detection3d_gt_bboxes_3d = self.build_sane_bboxes_3d(num_bboxes=3)
        detection3d_gt_bboxes_3d.bbox_params[1, Box3DFieldIndex.WIDTH] = -0.8
        detection3d_gt_bboxes_3d.bbox_params[2, Box3DFieldIndex.HEIGHT] = 0.0
        multi_task_gt_sample = self.build_multi_task_gt_sample(
            detection3d_gt_bboxes_3d=detection3d_gt_bboxes_3d,
        )

        output = BBoxesPhysicalFilter()(multi_task_gt_sample)

        self.assertEqual(self.assert_bboxes_3d(output).bbox_labels.tolist(), [0])

    def test_bounds_the_ground_plane_speed_not_its_components(self) -> None:
        """The speed bound applies to the norm, so a fast but physical box is kept."""
        detection3d_gt_bboxes_3d = self.build_bboxes_3d(
            bbox_params=[
                # Norm is about 143 m/s, under the bound, even though vx alone is near it.
                [4.0, 4.0, 0.5, 4.5, 1.9, 1.4, 0.3, 140.0, 30.0, 0.0],
                # Both components are under the bound but the norm is about 170 m/s.
                [1.0, 2.0, 3.0, 4.0, 1.5, 1.7, 0.1, 120.0, 120.0, 0.0],
            ],
            bbox_label_names=["car", "car"],
        )
        multi_task_gt_sample = self.build_multi_task_gt_sample(
            detection3d_gt_bboxes_3d=detection3d_gt_bboxes_3d,
        )

        output = BBoxesPhysicalFilter()(multi_task_gt_sample)

        self.assertEqual(self.assert_bboxes_3d(output).bbox_labels.tolist(), [0])

    def test_ignores_the_vertical_velocity_in_the_speed_bound(self) -> None:
        """Only the ground-plane velocity is bounded, matching ``box_is_physical``."""
        detection3d_gt_bboxes_3d = self.build_bboxes_3d(
            bbox_params=[[1.0, 2.0, 3.0, 4.0, 1.5, 1.7, 0.1, 0.5, -0.1, 1000.0]],
            bbox_label_names=["car"],
        )
        multi_task_gt_sample = self.build_multi_task_gt_sample(
            detection3d_gt_bboxes_3d=detection3d_gt_bboxes_3d,
        )

        output = BBoxesPhysicalFilter()(multi_task_gt_sample)

        self.assertEqual(len(self.assert_bboxes_3d(output)), 1)

    def test_keeps_a_bbox_sitting_exactly_on_the_speed_bound(self) -> None:
        """The bound is inclusive, so a box travelling at exactly the maximum speed is kept."""
        detection3d_gt_bboxes_3d = self.build_bboxes_3d(
            bbox_params=[[1.0, 2.0, 3.0, 4.0, 1.5, 1.7, 0.1, MAX_ABSOLUTE_SPEED, 0.0, 0.0]],
            bbox_label_names=["car"],
        )
        multi_task_gt_sample = self.build_multi_task_gt_sample(
            detection3d_gt_bboxes_3d=detection3d_gt_bboxes_3d,
        )

        output = BBoxesPhysicalFilter()(multi_task_gt_sample)

        self.assertEqual(len(self.assert_bboxes_3d(output)), 1)

    def test_honours_a_configured_speed_bound(self) -> None:
        """A tighter configured bound removes boxes the default bound would keep."""
        detection3d_gt_bboxes_3d = self.build_bboxes_3d(
            bbox_params=[[1.0, 2.0, 3.0, 4.0, 1.5, 1.7, 0.1, 40.0, 0.0, 0.0]],
            bbox_label_names=["car"],
        )
        multi_task_gt_sample = self.build_multi_task_gt_sample(
            detection3d_gt_bboxes_3d=detection3d_gt_bboxes_3d,
        )

        output = BBoxesPhysicalFilter(max_absolute_speed=30.0)(multi_task_gt_sample)

        self.assertEqual(len(self.assert_bboxes_3d(output)), 0)


class NormalizeFilterAttributesTest(unittest.TestCase):
    """Tests for the normalize_filter_attributes helper."""

    def test_rejects_invalid_entries(self) -> None:
        """A malformed attribute exclusion is reported at configuration time."""
        with self.assertRaisesRegex(ValueError, "filter_attributes entries"):
            normalize_filter_attributes([["bicycle"]])

        with self.assertRaisesRegex(TypeError, "filter_attributes entries"):
            normalize_filter_attributes(["bicycle"])


if __name__ == "__main__":
    unittest.main()
