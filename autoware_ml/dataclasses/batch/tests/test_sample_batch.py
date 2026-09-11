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

"""Unit tests for collating ``ModelGTSample`` sequences into a ``ModelGTBatch``."""

from __future__ import annotations

from collections.abc import Sequence
import unittest

import torch

from autoware_ml.dataclasses.batch.detection3d import Detection3DGTBatch
from autoware_ml.dataclasses.batch.sample_batch import ModelGTBatch, ModelGTSample
from autoware_ml.geometry.bbox_3d.lidar_bbox3d import LidarBBoxes3D
from autoware_ml.types.geometry import Box3DCenterCoordinateType, Box3DFieldIndex


class TestModelGTBatchDetection3DCollation(unittest.TestCase):
    """Unit tests for the 3D detection part of ``ModelGTBatch.collate_gt_samples``."""

    def setUp(self) -> None:
        """Set up the padding size shared by every test."""
        self.max_num_3d_gt_bboxes = 3

    def _build_bboxes_3d(self, num_bboxes: int) -> LidarBBoxes3D:
        """Build ``num_bboxes`` unit boxes spread one metre apart along x."""
        bbox_params = torch.zeros((num_bboxes, len(Box3DFieldIndex)), dtype=torch.float32)
        bbox_params[:, Box3DFieldIndex.X] = torch.arange(num_bboxes, dtype=torch.float32)
        bbox_params[:, Box3DFieldIndex.LENGTH : Box3DFieldIndex.HEIGHT + 1] = 1.0
        return LidarBBoxes3D(
            bbox_params=bbox_params,
            bbox_labels=torch.zeros(num_bboxes, dtype=torch.int32),
            bbox_label_names=["car"] * num_bboxes,
            bbox_num_lidar_points=torch.full((num_bboxes,), 5, dtype=torch.int32),
            bbox_center_coordinate_type=Box3DCenterCoordinateType.GRAVITY_CENTER,
        )

    def _build_sample(
        self, num_bboxes: int, traffic_cone_barrier_bbox_status: bool | None
    ) -> ModelGTSample:
        """Build a sample holding only 3D detection ground truth and the cone/barrier flag."""
        return ModelGTSample(
            lidar_point_cloud_samples=None,
            image_samples=None,
            point_cloud_data=None,
            camera_image_data=None,
            detection3d_gt_bboxes_3d=self._build_bboxes_3d(num_bboxes),
            segmentation3d_gt_sample=None,
            detection3d_traffic_cone_barrier_bbox_status=traffic_cone_barrier_bbox_status,
        )

    def _collate(self, samples: Sequence[ModelGTSample]) -> Detection3DGTBatch:
        """Collate the samples and return the 3D detection batch, which must exist."""
        batch = ModelGTBatch.collate_gt_samples(
            gt_samples=samples, max_num_3d_gt_bboxes=self.max_num_3d_gt_bboxes
        )
        assert batch.detection3d_gt_batch is not None
        return batch.detection3d_gt_batch

    def test_sample_defaults_to_unknown_status(self) -> None:
        """Test that a sample built without the flag reports it as unknown."""
        sample = ModelGTSample(
            lidar_point_cloud_samples=None,
            image_samples=None,
            point_cloud_data=None,
            camera_image_data=None,
            detection3d_gt_bboxes_3d=self._build_bboxes_3d(1),
            segmentation3d_gt_sample=None,
        )

        self.assertIsNone(sample.detection3d_traffic_cone_barrier_bbox_status)

    def test_collate_stacks_one_status_per_sample(self) -> None:
        """Test that the per-sample flags become one boolean tensor aligned with the boxes."""
        detection3d_gt_batch = self._collate(
            [
                self._build_sample(num_bboxes=2, traffic_cone_barrier_bbox_status=True),
                self._build_sample(num_bboxes=1, traffic_cone_barrier_bbox_status=False),
                self._build_sample(num_bboxes=3, traffic_cone_barrier_bbox_status=True),
            ]
        )

        status = detection3d_gt_batch.gt_traffic_cone_barrier_bbox_status
        assert status is not None
        self.assertEqual(status.dtype, torch.bool)
        self.assertEqual(status.device, detection3d_gt_batch.gt_bboxes_3d.device)
        self.assertEqual(status.tolist(), [True, False, True])
        self.assertEqual(detection3d_gt_batch.gt_valid_bboxes.tolist(), [2, 1, 3])

    def test_collate_without_any_status_leaves_it_none(self) -> None:
        """Test that datasets without the flag collate to a batch without the flag."""
        detection3d_gt_batch = self._collate(
            [
                self._build_sample(num_bboxes=1, traffic_cone_barrier_bbox_status=None),
                self._build_sample(num_bboxes=2, traffic_cone_barrier_bbox_status=None),
            ]
        )

        self.assertIsNone(detection3d_gt_batch.gt_traffic_cone_barrier_bbox_status)
        self.assertEqual(detection3d_gt_batch.gt_valid_bboxes.tolist(), [1, 2])

    def test_collate_rejects_partially_known_status(self) -> None:
        """Test that a batch mixing known and unknown flags is rejected."""
        samples = [
            self._build_sample(num_bboxes=1, traffic_cone_barrier_bbox_status=True),
            self._build_sample(num_bboxes=1, traffic_cone_barrier_bbox_status=None),
        ]

        with self.assertRaisesRegex(ValueError, "detection3d_traffic_cone_barrier_bbox_status"):
            self._collate(samples)

    def test_collate_gt_samples_rejects_misaligned_status(self) -> None:
        """Test that the batch collation rejects a flag list not matching the sample count."""
        bboxes = [self._build_bboxes_3d(1), self._build_bboxes_3d(2)]

        with self.assertRaisesRegex(ValueError, "one entry per sample"):
            Detection3DGTBatch.collate_gt_samples(
                detection3d_gt_bboxes_3d=bboxes,
                max_num_3d_gt_bboxes=self.max_num_3d_gt_bboxes,
                detection3d_traffic_cone_barrier_bbox_status=[True],
            )

    def test_collate_gt_samples_without_status_argument_keeps_it_none(self) -> None:
        """Test that callers not passing the flags get a batch without them."""
        detection3d_gt_batch = Detection3DGTBatch.collate_gt_samples(
            detection3d_gt_bboxes_3d=[self._build_bboxes_3d(1)],
            max_num_3d_gt_bboxes=self.max_num_3d_gt_bboxes,
        )

        assert detection3d_gt_batch is not None
        self.assertIsNone(detection3d_gt_batch.gt_traffic_cone_barrier_bbox_status)

    def test_to_device_moves_status_and_keeps_none(self) -> None:
        """Test that moving the batch carries the flag along and keeps a missing flag missing."""
        with_status = self._collate(
            [self._build_sample(num_bboxes=1, traffic_cone_barrier_bbox_status=True)]
        )
        without_status = self._collate(
            [self._build_sample(num_bboxes=1, traffic_cone_barrier_bbox_status=None)]
        )
        device = torch.device("cpu")

        moved_with_status = with_status.to_device(device)
        moved_without_status = without_status.to_device(device)

        assert moved_with_status.gt_traffic_cone_barrier_bbox_status is not None
        self.assertEqual(moved_with_status.gt_traffic_cone_barrier_bbox_status.device, device)
        self.assertEqual(moved_with_status.gt_traffic_cone_barrier_bbox_status.tolist(), [True])
        self.assertIsNone(moved_without_status.gt_traffic_cone_barrier_bbox_status)


if __name__ == "__main__":
    unittest.main()
