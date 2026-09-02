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

"""Unit tests for the point-cloud filter transforms."""

import unittest

import torch

from autoware_ml.datamodule.multi_task.dataclasses.multi_task_samples import MultiTaskGTSample
from autoware_ml.geometry.points.lidar_points import LiDARPoints
from autoware_ml.transforms.point_cloud.filters import PointsRangeFilter
from autoware_ml.types.geometry import PointFeatureName


class TestPointsRangeFilter(unittest.TestCase):
    """Unit tests for the PointsRangeFilter transform."""

    # Symmetric range of +-10 m laterally and longitudinally, and -3 m to 5 m vertically.
    POINTS_RANGE = (-10.0, -10.0, -3.0, 10.0, 10.0, 5.0)

    def build_point_cloud_data(self, points: torch.Tensor) -> LiDARPoints:
        """Build a point cloud with xyz and intensity features.

        Args:
            points: A (num_points, 4) tensor of ``[x, y, z, intensity]`` rows.

        Returns:
            LiDARPoints holding the given points.
        """
        return LiDARPoints(
            points=points,
            point_feature_names=[
                PointFeatureName.X,
                PointFeatureName.Y,
                PointFeatureName.Z,
                PointFeatureName.INTENSITY,
            ],
            timestamp=0.0,
        )

    def build_multi_task_gt_sample(self, point_cloud_data: LiDARPoints | None) -> MultiTaskGTSample:
        """Build a minimal sample holding only the given point cloud.

        Args:
            point_cloud_data: The point cloud to put in the sample, or ``None``.

        Returns:
            MultiTaskGTSample with the point cloud and every other field unset.
        """
        return MultiTaskGTSample(
            lidar_point_cloud_samples=None,
            image_samples=None,
            point_cloud_data=point_cloud_data,
            camera_image_data=None,
            detection3d_gt_bboxes_3d=None,
            segmentation3d_gt_sample=None,
        )

    def test_keeps_only_points_inside_the_range(self) -> None:
        """Test that points outside the range on any axis are removed and the rest kept."""
        points = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.1],  # inside
                [11.0, 0.0, 0.0, 0.2],  # x above max
                [-11.0, 0.0, 0.0, 0.3],  # x below min
                [0.0, 12.0, 0.0, 0.4],  # y above max
                [0.0, -12.0, 0.0, 0.5],  # y below min
                [0.0, 0.0, 6.0, 0.6],  # z above max
                [0.0, 0.0, -4.0, 0.7],  # z below min
                [5.0, -5.0, 2.0, 0.8],  # inside
            ],
            dtype=torch.float32,
        )
        sample = self.build_multi_task_gt_sample(self.build_point_cloud_data(points))

        output = PointsRangeFilter(points_range=self.POINTS_RANGE)(sample)

        assert output.point_cloud_data is not None
        self.assertTrue(torch.equal(output.point_cloud_data.points, points[[0, 7]]))

    def test_range_bounds_are_inclusive(self) -> None:
        """Test that points exactly on the range boundaries are kept."""
        points = torch.tensor(
            [
                [-10.0, -10.0, -3.0, 0.1],  # on the min corner
                [10.0, 10.0, 5.0, 0.2],  # on the max corner
                [10.0, 0.0, 0.0, 0.3],  # on the x max face
            ],
            dtype=torch.float32,
        )
        sample = self.build_multi_task_gt_sample(self.build_point_cloud_data(points))

        output = PointsRangeFilter(points_range=self.POINTS_RANGE)(sample)

        assert output.point_cloud_data is not None
        self.assertTrue(torch.equal(output.point_cloud_data.points, points))

    def test_kept_rows_are_unchanged(self) -> None:
        """Test that filtering only drops rows and never alters the surviving features."""
        points = torch.tensor(
            [
                [1.0, 2.0, 0.5, 0.9],
                [100.0, 0.0, 0.0, 0.1],
                [-3.0, 4.0, -1.0, 0.4],
            ],
            dtype=torch.float32,
        )
        sample = self.build_multi_task_gt_sample(self.build_point_cloud_data(points))

        output = PointsRangeFilter(points_range=self.POINTS_RANGE)(sample)

        assert output.point_cloud_data is not None
        kept_points = output.point_cloud_data.points
        self.assertEqual(kept_points.shape, (2, 4))
        # Rows keep their relative order and every feature, including intensity.
        self.assertTrue(torch.equal(kept_points, points[[0, 2]]))

    def test_all_points_inside_keeps_everything(self) -> None:
        """Test that a range enclosing every point leaves the point cloud untouched."""
        points = torch.tensor(
            [[1.0, 2.0, 0.5, 0.1], [-3.0, 0.5, -1.0, 0.2], [4.0, -2.0, 1.5, 0.3]],
            dtype=torch.float32,
        )
        sample = self.build_multi_task_gt_sample(self.build_point_cloud_data(points))

        output = PointsRangeFilter(points_range=self.POINTS_RANGE)(sample)

        assert output.point_cloud_data is not None
        self.assertTrue(torch.equal(output.point_cloud_data.points, points))

    def test_all_points_outside_leaves_an_empty_point_cloud(self) -> None:
        """Test that a range excluding every point leaves an empty cloud with the same width."""
        points = torch.tensor([[50.0, 0.0, 0.0, 0.1], [0.0, -50.0, 0.0, 0.2]], dtype=torch.float32)
        sample = self.build_multi_task_gt_sample(self.build_point_cloud_data(points))

        output = PointsRangeFilter(points_range=self.POINTS_RANGE)(sample)

        assert output.point_cloud_data is not None
        self.assertEqual(len(output.point_cloud_data), 0)
        self.assertEqual(output.point_cloud_data.points.shape, (0, 4))

    def test_filters_in_place_and_returns_same_sample(self) -> None:
        """Test that the point cloud is filtered in place and the sample is returned as is."""
        points = torch.tensor([[0.0, 0.0, 0.0, 0.1], [50.0, 0.0, 0.0, 0.2]], dtype=torch.float32)
        point_cloud_data = self.build_point_cloud_data(points)
        sample = self.build_multi_task_gt_sample(point_cloud_data)

        output = PointsRangeFilter(points_range=self.POINTS_RANGE)(sample)

        self.assertIs(output, sample)
        self.assertIs(output.point_cloud_data, point_cloud_data)
        self.assertEqual(len(point_cloud_data), 1)

    def test_empty_point_cloud_is_returned_unchanged(self) -> None:
        """Test that an empty point cloud is passed through without error."""
        point_cloud_data = self.build_point_cloud_data(torch.zeros((0, 4), dtype=torch.float32))
        sample = self.build_multi_task_gt_sample(point_cloud_data)

        output = PointsRangeFilter(points_range=self.POINTS_RANGE)(sample)

        assert output.point_cloud_data is not None
        self.assertIs(output, sample)
        self.assertEqual(output.point_cloud_data.points.shape, (0, 4))

    def test_points_range_is_stored_as_float_tensor(self) -> None:
        """Test that the configured range is kept as a float32 tensor of six values."""
        transform = PointsRangeFilter(points_range=self.POINTS_RANGE)

        self.assertEqual(transform.points_range.dtype, torch.float32)
        self.assertTrue(
            torch.equal(
                transform.points_range, torch.tensor(self.POINTS_RANGE, dtype=torch.float32)
            )
        )

    def test_missing_point_cloud_data_key(self) -> None:
        """Test that missing 'point_cloud_data' raises KeyError."""
        sample = self.build_multi_task_gt_sample(point_cloud_data=None)

        with self.assertRaises(KeyError):
            PointsRangeFilter(points_range=self.POINTS_RANGE)(sample)


if __name__ == "__main__":
    unittest.main()
