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

from autoware_ml.datamodule.multi_task.dataclasses.multi_task_samples import (
    LiDARPointCloudSample,
    MultiTaskGTSample,
)
from autoware_ml.transforms.point_cloud.loading import LoadPointsFromFile
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
    ) -> MultiTaskGTSample:
        """Build a minimal sample holding only the LiDAR file metadata the loader reads."""
        return MultiTaskGTSample(
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


if __name__ == "__main__":
    unittest.main()
