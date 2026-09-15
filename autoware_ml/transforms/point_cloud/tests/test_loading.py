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

"""Unit tests for the point-cloud loading transforms."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from autoware_ml.dataclasses.geometry.point_clouds import LiDARPointCloudSample
from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample
from autoware_ml.transforms.point_cloud.loading import (
    LoadMultiSweepPointsFromFile,
    LoadPointsFromFile,
)
from autoware_ml.types.geometry import PointFeatureName


class TestLoadPointsFromFile(unittest.TestCase):
    """Tests for ``LoadPointsFromFile``."""

    def setUp(self) -> None:
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp_dir.cleanup)
        self.tmp_path = Path(self._tmp_dir.name)

    def build_lidar_point_cloud_sample(
        self, point_cloud_path: Path, timestamp: float
    ) -> LiDARPointCloudSample:
        """Build a LiDAR sample pointing at ``point_cloud_path`` with identity transforms."""
        return LiDARPointCloudSample(
            point_cloud_path=str(point_cloud_path),
            timestamp=timestamp,
            sensor_to_ego_pose_matrix=torch.eye(4),
            lidar_to_ego_pose_to_global_matrix=torch.eye(4),
            lidar_sensor_to_lidar_sweep_matrix=torch.eye(4),
        )

    def build_multi_task_gt_sample(
        self, lidar_point_cloud_samples: list[LiDARPointCloudSample]
    ) -> ModelGTSample:
        """Build a minimal sample holding only the LiDAR file metadata the loader reads."""
        return ModelGTSample(
            lidar_point_cloud_samples=lidar_point_cloud_samples,
            image_samples=None,
            point_cloud_data=None,
            camera_image_data=None,
            detection3d_gt_bboxes_3d=None,
            segmentation3d_gt_sample=None,
        )

    def test_load_points_from_file_loads_selected_dims(self) -> None:
        """Loading a single point keeps the selected dims and sample timestamp."""
        points_path = self.tmp_path / "points.bin"
        raw_points = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        raw_points.tofile(points_path)

        sample = self.build_multi_task_gt_sample(
            [self.build_lidar_point_cloud_sample(points_path, timestamp=10.0)]
        )
        output = LoadPointsFromFile(load_dim=4, use_dim=[0, 1, 2, 3])(sample)

        self.assertIsNotNone(output.point_cloud_data)
        self.assertEqual(output.point_cloud_data.shape, (1, 4))
        np.testing.assert_allclose(output.point_cloud_data.to_numpy(), raw_points)
        self.assertEqual(
            list(output.point_cloud_data.point_feature_names),
            [
                PointFeatureName.X,
                PointFeatureName.Y,
                PointFeatureName.Z,
                PointFeatureName.INTENSITY,
            ],
        )
        self.assertEqual(output.point_cloud_data.timestamp, 10.0)
        # The loader only fills ``point_cloud_data``; the remaining fields are passed through.
        self.assertIs(output.lidar_point_cloud_samples, sample.lidar_point_cloud_samples)
        self.assertIsNone(output.detection3d_gt_bboxes_3d)


class TestLoadMultiSweepPointsFromFile(TestLoadPointsFromFile):
    """Tests for ``LoadMultiSweepPointsFromFile``."""

    def write_points(self, name: str, points: np.ndarray) -> Path:
        path = self.tmp_path / name
        points.astype(np.float32).tofile(path)
        return path

    def build_sweep_sample(
        self, point_cloud_path: Path, timestamp: float, translation: tuple[float, float, float]
    ) -> LiDARPointCloudSample:
        """Build a sweep whose lidar frame is offset from the main lidar by ``translation``."""
        matrix = torch.eye(4)
        matrix[:3, 3] = torch.tensor(translation)
        return self.build_lidar_point_cloud_sample(point_cloud_path, timestamp)._replace(
            lidar_sensor_to_lidar_sweep_matrix=matrix
        )

    def build_loaded_sample(self, samples: list[LiDARPointCloudSample]) -> ModelGTSample:
        """Run the single-frame loader so ``point_cloud_data`` holds the current frame."""
        return LoadPointsFromFile(load_dim=4, use_dim=[0, 1, 2, 3])(
            self.build_multi_task_gt_sample(samples)
        )

    def test_appends_sweeps_in_the_main_lidar_frame_with_time_lag(self) -> None:
        current = self.write_points("current.bin", np.array([[1.0, 2.0, 3.0, 4.0]]))
        sweep = self.write_points("sweep.bin", np.array([[5.0, 0.0, 0.0, 7.0]]))
        sample = self.build_loaded_sample(
            [
                self.build_lidar_point_cloud_sample(current, timestamp=10.0),
                self.build_sweep_sample(sweep, timestamp=9.9, translation=(1.0, 0.0, 0.0)),
            ]
        )

        output = LoadMultiSweepPointsFromFile(
            sweeps_num=1, test_mode=True, load_dim=4, use_dim=[0, 1, 2, 3], bev_remove_radius=0.0
        )(sample)

        points = output.point_cloud_data.to_numpy()
        self.assertEqual(points.shape, (2, 5))
        np.testing.assert_allclose(points[0], [1.0, 2.0, 3.0, 4.0, 0.0])
        # The sweep point is moved into the main lidar frame and tagged with the time lag.
        np.testing.assert_allclose(points[1], [4.0, 0.0, 0.0, 7.0, 0.1], atol=1e-6)
        self.assertEqual(
            output.point_cloud_data.point_feature_names[-1], PointFeatureName.TIMESTAMP_DIFFERENCE
        )
        self.assertEqual(output.point_cloud_data.timestamp_difference_dim, 4)

    def test_test_mode_takes_nearest_sweeps_and_caps_at_available(self) -> None:
        current = self.write_points("current.bin", np.zeros((1, 4)))
        sweeps = [
            self.write_points(f"sweep{i}.bin", np.full((1, 4), 10.0 * (i + 1))) for i in range(3)
        ]
        samples = [self.build_lidar_point_cloud_sample(current, 10.0)] + [
            self.build_sweep_sample(path, 10.0 - 0.1 * (i + 1), (0.0, 0.0, 0.0))
            for i, path in enumerate(sweeps)
        ]

        nearest = LoadMultiSweepPointsFromFile(
            sweeps_num=2,
            test_mode=True,
            use_timestamp_difference=False,
            load_dim=4,
            bev_remove_radius=0.0,
        )(self.build_loaded_sample(samples)).point_cloud_data.to_numpy()
        capped = LoadMultiSweepPointsFromFile(
            sweeps_num=10,
            test_mode=True,
            use_timestamp_difference=False,
            load_dim=4,
            bev_remove_radius=0.0,
        )(self.build_loaded_sample(samples)).point_cloud_data.to_numpy()

        self.assertEqual(nearest.shape, (3, 4))
        np.testing.assert_allclose(nearest[1:, 0], [10.0, 20.0])
        self.assertEqual(capped.shape, (4, 4))

    def test_random_mode_draws_without_replacement(self) -> None:
        current = self.write_points("current.bin", np.zeros((1, 4)))
        sweeps = [
            self.write_points(f"sweep{i}.bin", np.full((1, 4), 10.0 * (i + 1))) for i in range(3)
        ]
        samples = [self.build_lidar_point_cloud_sample(current, 10.0)] + [
            self.build_sweep_sample(path, 9.0, (0.0, 0.0, 0.0)) for path in sweeps
        ]
        transform = LoadMultiSweepPointsFromFile(
            sweeps_num=2,
            test_mode=False,
            use_timestamp_difference=False,
            load_dim=4,
            bev_remove_radius=0.0,
        )

        torch.manual_seed(0)
        drawn = set()
        for _ in range(20):
            points = transform(self.build_loaded_sample(samples)).point_cloud_data.to_numpy()
            self.assertEqual(points.shape, (3, 4))
            self.assertEqual(len(set(points[1:, 0].tolist())), 2)
            drawn.update(points[1:, 0].tolist())
        self.assertEqual(drawn, {10.0, 20.0, 30.0})

    def test_bev_remove_radius_only_prunes_sweep_points(self) -> None:
        current = self.write_points("current.bin", np.array([[0.1, 0.1, 0.0, 1.0]]))
        sweep = self.write_points(
            "sweep.bin",
            np.array([[0.5, -0.5, 0.0, 2.0], [0.9, 0.9, 0.0, 3.0], [1.5, 0.0, 0.0, 4.0]]),
        )
        sample = self.build_loaded_sample(
            [
                self.build_lidar_point_cloud_sample(current, 10.0),
                self.build_sweep_sample(sweep, 9.9, (0.0, 0.0, 0.0)),
            ]
        )

        points = LoadMultiSweepPointsFromFile(
            sweeps_num=1,
            test_mode=True,
            use_timestamp_difference=False,
            load_dim=4,
            bev_remove_radius=1.0,
        )(sample).point_cloud_data.to_numpy()

        # The current-frame point near the origin stays; only the sweep box |x|,|y| < 1 is pruned.
        np.testing.assert_allclose(points[:, 3], [1.0, 4.0])

    def test_zero_sweeps_keeps_current_frame_with_time_column(self) -> None:
        current = self.write_points("current.bin", np.array([[1.0, 2.0, 3.0, 4.0]]))
        sample = self.build_loaded_sample([self.build_lidar_point_cloud_sample(current, 10.0)])

        points = LoadMultiSweepPointsFromFile(
            sweeps_num=0, test_mode=True, load_dim=4, bev_remove_radius=1.0
        )(sample).point_cloud_data.to_numpy()

        np.testing.assert_allclose(points, [[1.0, 2.0, 3.0, 4.0, 0.0]])

    def test_pad_empty_sweeps_repeats_current_frame_only_without_sweeps(self) -> None:
        current = self.write_points(
            "current.bin", np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
        )
        sweep = self.write_points("sweep.bin", np.array([[9.0, 0.0, 0.0, 9.0]]))
        no_sweeps = self.build_loaded_sample([self.build_lidar_point_cloud_sample(current, 10.0)])
        one_sweep = self.build_loaded_sample(
            [
                self.build_lidar_point_cloud_sample(current, 10.0),
                self.build_sweep_sample(sweep, 9.9, (0.0, 0.0, 0.0)),
            ]
        )
        transform = LoadMultiSweepPointsFromFile(
            sweeps_num=3, test_mode=True, load_dim=4, bev_remove_radius=0.0, pad_empty_sweeps=True
        )

        padded = transform(no_sweeps).point_cloud_data.to_numpy()
        partial = transform(one_sweep).point_cloud_data.to_numpy()

        # The current frame is repeated sweeps_num times, each copy tagged with a zero time lag.
        self.assertEqual(padded.shape, (8, 5))
        np.testing.assert_allclose(
            padded[:, :4], np.tile([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], (4, 1))
        )
        np.testing.assert_allclose(padded[:, 4], 0.0)
        # A frame with fewer sweeps than requested is not padded.
        self.assertEqual(partial.shape, (3, 5))

    def test_pad_empty_sweeps_disabled_by_default(self) -> None:
        current = self.write_points("current.bin", np.array([[1.0, 2.0, 3.0, 4.0]]))
        sample = self.build_loaded_sample([self.build_lidar_point_cloud_sample(current, 10.0)])

        points = LoadMultiSweepPointsFromFile(
            sweeps_num=3, test_mode=True, load_dim=4, bev_remove_radius=0.0
        )(sample).point_cloud_data.to_numpy()

        self.assertEqual(points.shape, (1, 5))

    def test_requires_loaded_current_frame(self) -> None:
        current = self.write_points("current.bin", np.zeros((1, 4)))
        sample = self.build_multi_task_gt_sample(
            [self.build_lidar_point_cloud_sample(current, 10.0)]
        )

        with self.assertRaisesRegex(KeyError, "Missing required key 'point_cloud_data'"):
            LoadMultiSweepPointsFromFile(sweeps_num=1, test_mode=True, load_dim=4)(sample)


if __name__ == "__main__":
    unittest.main()
