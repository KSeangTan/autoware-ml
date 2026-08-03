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

"""Unit tests for point pillar preprocessing."""

from __future__ import annotations

import unittest

import torch

from autoware_ml.preprocessing.detection3d.point_pillar import PointPillarPreprocessor
from autoware_ml.datamodule.multi_task.dataclasses.multi_task_samples import (
    MultiTaskGTBatch,
    PointCloudGTBatch,
    Detection3DGTBatch,
)
from autoware_ml.datamodule.multi_task.dataclasses.multi_task_features import MultiTaskFeatures


class TestPointPillarPreprocessor(unittest.TestCase):
    def setUp(self) -> None:
        """Set up the same PointPillarPreprocessor instance for all tests. Note that this class will
        be called in each test case.
        """
        torch.manual_seed(0)
        self.point_pillar_preprocessor = PointPillarPreprocessor(
            voxel_size=[1.0, 1.0, 4.0],
            point_cloud_range=[0.0, 0.0, -2.0, 4.0, 4.0, 2.0],
            max_num_points=2,
            max_voxels=8,
            voxelization_z_order_first=True,  # This is used for backward-compatible, and will be removed very soon.
        )
        self.multi_task_gt_batch = MultiTaskGTBatch(
            point_cloud_gt_batch=None, detection3d_gt_batch=None
        )

    def test_forward_builds_padded_pillars(self) -> None:
        """
        Test that the forward method correctly builds padded pillars from a batch of point
        clouds.
        """
        multi_task_gt_batch = MultiTaskGTBatch(
            point_cloud_gt_batch=PointCloudGTBatch(
                points=torch.tensor(
                    [
                        [0.1, 0.1, 0.0, 1.0],
                        [0.2, 0.2, 0.0, 2.0],
                        [1.1, 1.1, 0.0, 3.0],
                    ],
                    dtype=torch.float32,
                ),
                batch_indices=torch.tensor([0, 0, 0], dtype=torch.int32),
            ),
            detection3d_gt_batch=None,
        )
        multi_task_features = MultiTaskFeatures(
            detection3d_features=None, multi_task_gt_batch=multi_task_gt_batch
        )

        outputs = self.point_pillar_preprocessor(multi_task_features)
        self.assertIsNotNone(outputs.detection3d_features)
        voxels_data = outputs.detection3d_features.voxels_data
        self.assertEqual(voxels_data.voxels.shape, (2, 2, 4))
        self.assertEqual(voxels_data.num_points.tolist(), [2, 1])
        self.assertEqual(voxels_data.coords.shape, (2, 3))
        self.assertEqual(voxels_data.coords[:, 0].tolist(), [0, 0])

    def test_per_sample_coords(self) -> None:
        """Test that the voxel coordinates are correctly computed for each sample in the batch."""
        multi_task_gt_batch = MultiTaskGTBatch(
            point_cloud_gt_batch=PointCloudGTBatch(
                points=torch.tensor(
                    [
                        [0.5, 0.5, 0.0, 1.0],
                        [0.5, 0.5, 0.0, 1.0],
                        [0.5, 0.5, 0.0, 1.0],
                    ],
                    dtype=torch.float32,
                ),
                batch_indices=torch.tensor([0, 1, 2], dtype=torch.int32),
            ),
            detection3d_gt_batch=None,
        )
        multi_task_features = MultiTaskFeatures(
            detection3d_features=None, multi_task_gt_batch=multi_task_gt_batch
        )
        outputs = self.point_pillar_preprocessor(multi_task_features)
        self.assertIsNotNone(outputs.detection3d_features)
        voxels_data = outputs.detection3d_features.voxels_data
        self.assertTrue(
            torch.allclose(
                voxels_data.coords,
                torch.tensor(
                    [
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                    ],
                    dtype=torch.int32,
                ),
            )
        )

    def test_per_sample_batch_indices(self) -> None:
        """Test that the batch indices are correctly computed for each sample in the batch."""
        multi_task_gt_batch = MultiTaskGTBatch(
            point_cloud_gt_batch=PointCloudGTBatch(
                points=torch.tensor(
                    [
                        [0.5, 0.5, 0.0, 1.0],
                        [0.5, 0.5, 0.0, 1.0],
                        [0.5, 0.5, 0.0, 1.0],
                    ],
                    dtype=torch.float32,
                ),
                batch_indices=torch.tensor([0, 2, 4], dtype=torch.int32),
            ),
            detection3d_gt_batch=None,
        )
        multi_task_features = MultiTaskFeatures(
            detection3d_features=None, multi_task_gt_batch=multi_task_gt_batch
        )
        outputs = self.point_pillar_preprocessor(multi_task_features)
        self.assertIsNotNone(outputs.detection3d_features)
        voxels_data = outputs.detection3d_features.voxels_data
        self.assertTrue(
            torch.allclose(voxels_data.batch_indices, torch.tensor([0, 2, 4], dtype=torch.int32))
        )

    def test_empty_sample_in_batch(self) -> None:
        """
        Test that the PointPillarPreprocessor correctly handles a batch containing an empty
        sample.
        """
        point = torch.tensor([[0.5, 0.5, 0.0, 1.0]], dtype=torch.float32)
        empty = torch.zeros((0, 4), dtype=torch.float32)
        points = torch.cat([point, empty, point], dim=0)
        multi_task_gt_batch = MultiTaskGTBatch(
            point_cloud_gt_batch=PointCloudGTBatch(
                points=points,
                batch_indices=torch.tensor([0, 0, 1], dtype=torch.int32),
            ),
            detection3d_gt_batch=None,
        )
        multi_task_features = MultiTaskFeatures(
            detection3d_features=None, multi_task_gt_batch=multi_task_gt_batch
        )
        # Raise a ValueError because the length of points list does not match the length of
        # batch indices in MultiTaskGTBatch.
        with self.assertRaises(ValueError):
            self.point_pillar_preprocessor(multi_task_features)

    def test_empty_batch_returns_empty_pillar_tensors(self) -> None:
        """
        Test that the PointPillarPreprocessor returns empty pillar tensors when given an
        empty batch.
        """
        multi_task_gt_batch = MultiTaskGTBatch(
            point_cloud_gt_batch=PointCloudGTBatch(
                points=torch.zeros((0, 4), dtype=torch.float32),
                batch_indices=torch.tensor([], dtype=torch.int32),
            ),
            detection3d_gt_batch=None,
        )
        multi_task_features = MultiTaskFeatures(
            detection3d_features=None, multi_task_gt_batch=multi_task_gt_batch
        )
        outputs = self.point_pillar_preprocessor(multi_task_features)
        self.assertIsNotNone(outputs.detection3d_features)
        voxels_data = outputs.detection3d_features.voxels_data
        self.assertEqual(voxels_data.voxels.shape, (0, 2, 4))
        self.assertEqual(voxels_data.num_points.shape, (0,))
        self.assertEqual(voxels_data.coords.shape, (0, 3))
        self.assertEqual(voxels_data.batch_indices.shape, (0,))

    def test_passthrough_of_existing_keys(self) -> None:
        """
        Test that the PointPillarPreprocessor correctly passes through existing
        keys in the input batch dictionary.
        """
        gt_bboxes_3d = torch.tensor(
            [
                [[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]],
                [[1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.57, 0.0, 1.0, 1.0]],
            ],
            dtype=torch.float32,
        )
        gt_labels_3d = torch.tensor([[1], [2]], dtype=torch.int32)
        gt_valid_bboxes = torch.tensor([1, 1], dtype=torch.int32)
        multi_task_gt_batch = MultiTaskGTBatch(
            point_cloud_gt_batch=PointCloudGTBatch(
                points=torch.tensor(
                    [
                        [0.5, 0.5, 0.0, 1.0],
                        [0.5, 0.5, 0.0, 1.0],
                        [0.5, 0.5, 0.0, 1.0],
                    ],
                    dtype=torch.float32,
                ),
                batch_indices=torch.tensor([0, 2, 4], dtype=torch.int32),
            ),
            detection3d_gt_batch=Detection3DGTBatch(
                gt_bboxes_3d=gt_bboxes_3d,
                gt_labels_3d=gt_labels_3d,
                gt_valid_bboxes=gt_valid_bboxes,
            ),
        )
        multi_task_features = MultiTaskFeatures(
            detection3d_features=None, multi_task_gt_batch=multi_task_gt_batch
        )
        outputs = self.point_pillar_preprocessor(multi_task_features)
        self.assertIsNotNone(outputs.detection3d_features)
        self.assertIsNotNone(outputs.multi_task_gt_batch)
        self.assertIsNotNone(outputs.multi_task_gt_batch.detection3d_gt_batch)

        self.assertTrue(
            torch.allclose(
                outputs.multi_task_gt_batch.detection3d_gt_batch.gt_bboxes_3d, gt_bboxes_3d
            )
        )
        self.assertTrue(
            torch.allclose(
                outputs.multi_task_gt_batch.detection3d_gt_batch.gt_labels_3d, gt_labels_3d
            )
        )
        self.assertTrue(
            torch.allclose(
                outputs.multi_task_gt_batch.detection3d_gt_batch.gt_valid_bboxes, gt_valid_bboxes
            )
        )


if __name__ == "__main__":
    unittest.main()
