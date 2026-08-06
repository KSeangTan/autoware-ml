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

"""Unit tests for CenterHead."""

import math
import unittest

import torch

from autoware_ml.models.detection3d.heads.centerhead import CenterHead
from autoware_ml.models.detection3d.dataclasses.outputs import CenterHeadOutputs, Detection3DOutputs
from autoware_ml.datamodule.multi_task.dataclasses.detection3d import (
    Detection3DGTBatch,
)


class TestCenterHead(unittest.TestCase):
    """Unit tests for the CenterHead."""

    def setUp(self) -> None:
        """Set up the common classes/inputs for the tests."""
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(0)
        self.center_head = CenterHead(
            in_channels=384,
            num_classes=2,
            shared_channels=64,
            point_cloud_range=[0.0, 0.0, -2.0, 8.0, 8.0, 2.0],
            voxel_size=[0.5, 0.5, 4.0],
            out_size_factor=2,
            max_objs=16,
            min_radius=1,
            score_threshold=0.1,
            post_max_size=10,
            nms_min_radius=1.0,
            use_velocity=True,
        ).to(self.device)

        # Dummy inputs
        self.gt_bboxes_3d = torch.tensor(
            [[[2.2, 3.3, 0.2, 4.0, 1.6, 1.5, 0.25, 0.5, -0.1, -0.2]]],
            device=self.device,
            dtype=torch.float32,
        )

        self.gt_labels_3d = torch.tensor([0], dtype=torch.int32, device=self.device)
        self.gt_valid_bboxes = torch.tensor([1], dtype=torch.int32, device=self.device)
        self.gt_bboxes_num_points = torch.tensor([[100]], dtype=torch.int32, device=self.device)
        self.detection3d_gt_batch = Detection3DGTBatch(
            gt_bboxes_3d=self.gt_bboxes_3d,
            gt_labels_3d=self.gt_labels_3d,
            gt_valid_bboxes=self.gt_valid_bboxes,
            gt_bboxes_num_points=self.gt_bboxes_num_points,
        )

        # self.multi_task_predictions = MultiTaskPredictions(
        #     detection3d_predictions=[
        #         Detection3DPredictions(
        #             bboxes_3d=torch.tensor(
        #                 [
        #                     [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 20.0, 30.0],
        #                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        #                 ],
        #                 dtype=torch.float32,
        #                 device=self.device,
        #             ),
        #             scores_3d=torch.tensor([0.9, 0.0], dtype=torch.float32, device=self.device),
        #             labels_3d=torch.tensor([1, 0], dtype=torch.int64, device=self.device),
        #         ),
        #         Detection3DPredictions(
        #             bboxes_3d=torch.tensor(
        #                 [[4.9, 0.2, 0.1, 6.8, 7.2, 9.2, 40.0, 50.0, 60.0]], device=self.device
        #             ),
        #             scores_3d=torch.tensor([0.9], dtype=torch.float32, device=self.device),
        #             labels_3d=torch.tensor([1, 2], dtype=torch.int64, device=self.device),
        #         ),
        #     ]
        # )
        # # (batch_size, num_boxes, box_dim) = (2, 2, 10)
        # gt_bboxes_3d = torch.tensor(
        #     [
        #         [
        #             [0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 5.0, 6.0, 7.0, 10.0],
        #             [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 20.0, 30.0, 40.0, 50.0],
        #         ],
        #         [
        #             [20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 30.0, 40.0, 50.0, 60.0],
        #             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        #         ],
        #     ],
        #     dtype=torch.float32,
        #     device=self.device,
        # )
        # gt_labels_3d = torch.tensor([[1, 2], [3, -1]], dtype=torch.int32, device=self.device)
        # gt_valid_bboxes = torch.tensor([2, 1], dtype=torch.int32, device=self.device)
        # gt_bboxes_num_points = torch.tensor(
        #     [[100, 200], [300, 0]], dtype=torch.int32, device=self.device
        # )

        # # Inputs
        # detection3d_gt_batch = Detection3DGTBatch(
        #     gt_bboxes_3d=gt_bboxes_3d,
        #     gt_labels_3d=gt_labels_3d,
        #     gt_valid_bboxes=gt_valid_bboxes,
        #     gt_bboxes_num_points=gt_bboxes_num_points,
        # )
        # self.multi_task_features = MultiTaskFeatures(
        #     multi_task_gt_batch=MultiTaskGTBatch(
        #         point_cloud_gt_batch=None, detection3d_gt_batch=detection3d_gt_batch
        #     ),
        #     detection3d_features=None,
        # )

    def test_centerhead_weights_mean_std(self) -> None:
        """Test that the CenterHead weights are initialized with near means and std."""
        weights = []
        biases = []
        for name, param in self.center_head.named_parameters():
            if "weight" in name:
                weights.append(param.data)
            if "bias" in name:
                biases.append(param.data)

        weights = torch.cat([w.flatten() for w in weights])
        biases = torch.cat([b.flatten() for b in biases])
        weight_mean = weights.mean().item()
        weight_std = weights.std().item()
        bias_mean = biases.mean().item()
        bias_std = biases.std().item()

        expected_weight_mean = 0.0
        expected_weight_std = 0.03679
        # Biases should be almost similar since the bias for the heatmap head is initialized
        # to a negative value and the rest are initialized to zero.
        expected_bias_mean = -0.008480
        expected_bias_std = 0.144860
        self.assertAlmostEqual(weight_mean, expected_weight_mean, places=2)
        self.assertAlmostEqual(weight_std, expected_weight_std, places=2)
        self.assertAlmostEqual(bias_mean, expected_bias_mean, places=2)
        self.assertAlmostEqual(bias_std, expected_bias_std, places=2)

    def test_center_head_zero_forward(self) -> None:
        """
        Test that the CenterHead forward pass with all zero features
        returns outputs of the expected shape.
        """
        # Initialize a dummy feature map with the expected input shape for the CenterHead
        dummy_input_features = torch.zeros((1, 384, 4, 4), device=self.device, dtype=torch.float32)
        outputs = self.center_head(dummy_input_features)

        self.assertEqual(outputs.heatmaps.shape, (1, 2, 4, 4))
        self.assertEqual(outputs.centers.shape, (1, 2, 4, 4))
        self.assertEqual(outputs.dims.shape, (1, 3, 4, 4))
        self.assertEqual(outputs.rots.shape, (1, 2, 4, 4))
        self.assertEqual(outputs.vels.shape, (1, 2, 4, 4))

        # All values are the same since the input features are zeros and biases for heatmap heads
        # are set to -2.19.
        expected_heatmaps = torch.tensor(-2.1900, device=self.device).expand_as(outputs.heatmaps)
        self.assertTrue(torch.allclose(outputs.heatmaps, expected_heatmaps))

        expected_centers = (
            torch.tensor([[0.0541, 0.1058]], device=self.device)
            .view(1, 2, 1, 1)
            .expand_as(outputs.centers)
        )
        self.assertTrue(torch.allclose(outputs.centers, expected_centers, atol=1e-4))

        expected_dims = (
            torch.tensor([[0.0855, -0.0303, 0.0646]], device=self.device)
            .view(1, 3, 1, 1)
            .expand_as(outputs.dims)
        )
        self.assertTrue(torch.allclose(outputs.dims, expected_dims, atol=1e-4))

        expected_rots = (
            torch.tensor([[0.1088, -0.1206]], device=self.device)
            .view(1, 2, 1, 1)
            .expand_as(outputs.rots)
        )
        self.assertTrue(torch.allclose(outputs.rots, expected_rots, atol=1e-4))

        expected_vels = (
            torch.tensor([[0.0809, 0.0154]], device=self.device)
            .view(1, 2, 1, 1)
            .expand_as(outputs.vels)
        )
        self.assertTrue(torch.allclose(outputs.vels, expected_vels, atol=1e-4))

    def test_build_targets_populates_heatmap_and_boxes(self) -> None:
        """Test that build_targets populates the heatmap and boxes correctly."""
        targets = self.center_head.get_targets(
            gt_bboxes_3d=self.gt_bboxes_3d,
            gt_labels_3d=self.gt_labels_3d,
            gt_valid_bboxes=self.gt_valid_bboxes,
            feature_map_size=(4, 4),
            device=self.device,
        )

        self.assertEqual(targets.heatmaps.shape, (1, 2, 4, 4))
        self.assertTrue(targets.valid_masks[0, 0].item())
        self.assertEqual(targets.reg_indices[0, 0].item(), 14)
        self.assertEqual(targets.heatmaps[0, 0, 3, 2].item(), 1.0)
        self.assertEqual(targets.reg_targets.shape, (1, 1, 10))
        expected_reg_targets = torch.cat(
            [
                torch.tensor([0.2, 0.3, 0.2], device=self.device),
                torch.tensor([4.0, 1.6, 1.5], device=self.device).log(),
                torch.tensor([math.sin(0.25), math.cos(0.25)], device=self.device),
                torch.tensor([0.5, -0.1], device=self.device),
            ],
            dim=-1,
        )
        self.assertTrue(
            torch.allclose(
                targets.reg_targets,
                expected_reg_targets.view(1, 1, -1),
            )
        )

    def test_decode_outputs_returns_length_width_height_after_unified_dim_order(self) -> None:
        """
        Test that decode_outputs returns the correct length, width, and height after
        applying the unified dimension order.
        """
        # Create a different CenterHead
        center_head = CenterHead(
            in_channels=4,
            num_classes=5,
            shared_channels=4,
            point_cloud_range=[0.0, 0.0, -2.0, 8.0, 8.0, 2.0],
            voxel_size=[0.5, 0.5, 4.0],
            out_size_factor=2,
            max_objs=16,
            min_radius=1,
            score_threshold=0.1,
            post_max_size=10,
            nms_min_radius=1.0,
            use_velocity=False,
        ).to(self.device)

        # Dummy outputs from CenterHead.forward
        dummy_outputs = CenterHeadOutputs(
            heatmaps=torch.full((1, 2, 4, 4), -20.0, device=self.device),
            centers=torch.zeros((1, 2, 4, 4), device=self.device),
            heights=torch.zeros((1, 1, 4, 4), device=self.device),
            dims=torch.zeros((1, 3, 4, 4), device=self.device),
            rots=torch.zeros((1, 2, 4, 4), device=self.device),
            vels=None,
        )
        dummy_outputs.heatmaps[0, 0, 3, 2] = 20.0
        dummy_outputs.heights[0, 0, 3, 2] = 0.2
        dummy_outputs.dims[0, :, 3, 2] = torch.tensor([4.0, 1.6, 1.5], device=self.device).log()
        dummy_outputs.rots[0, 1, 3, 2] = 1.0
        dummy_detection3d_outputs = Detection3DOutputs(
            center_head_outputs=dummy_outputs, transfusion_head_outputs=None
        )

        decoded_outputs = center_head.decode_outputs(dummy_detection3d_outputs)
        self.assertEqual(decoded_outputs.detection3d_predictions[0].bboxes_3d.shape, (1, 7))  # type: ignore
        self.assertTrue(
            torch.allclose(
                decoded_outputs.detection3d_predictions[0].bboxes_3d[0, 3:6],  # type: ignore
                torch.tensor([4.0, 1.6, 1.5], device=self.device),
            )
        )

    def test_loss_function(self) -> None:
        """Test that the loss function computes expected and non-negative losses."""
        # Modify dummy_outputs to have velocity values for testing
        # Dummy outputs from CenterHead.forward
        dummy_outputs = CenterHeadOutputs(
            heatmaps=torch.full((1, 2, 4, 4), -20.0, device=self.device),
            centers=torch.zeros((1, 2, 4, 4), device=self.device),
            heights=torch.zeros((1, 1, 4, 4), device=self.device),
            dims=torch.zeros((1, 3, 4, 4), device=self.device),
            rots=torch.zeros((1, 2, 4, 4), device=self.device),
            vels=torch.zeros((1, 2, 4, 4), device=self.device),
        )
        dummy_outputs.heatmaps[0, 0, 3, 2] = 20.0
        dummy_outputs.heights[0, 0, 3, 2] = 0.2
        dummy_outputs.dims[0, :, 3, 2] = torch.tensor([4.0, 1.6, 1.5], device=self.device).log()
        dummy_outputs.rots[0, 1, 3, 2] = 1.0
        dummy_outputs.vels[0, :, 3, 2] = torch.tensor([0.5, -0.1], device=self.device)
        dummy_detection3d_outputs = Detection3DOutputs(
            center_head_outputs=dummy_outputs, transfusion_head_outputs=None
        )

        losses = self.center_head.loss(
            outputs=dummy_detection3d_outputs,
            gt_bboxes_3d=self.gt_bboxes_3d,
            gt_labels_3d=self.gt_labels_3d,
            gt_valid_bboxes=self.gt_valid_bboxes,
        )
        self.assertIn("loss_heatmap", losses)
        self.assertIn("loss_bbox", losses)
        self.assertIn("loss", losses)

        self.assertTrue(losses["loss_heatmap"].item() >= 0.0)
        self.assertTrue(losses["loss_bbox"].item() >= 0.0)
        self.assertTrue(torch.isclose(losses["loss"], losses["loss_heatmap"] + losses["loss_bbox"]))


if __name__ == "__main__":
    unittest.main()
