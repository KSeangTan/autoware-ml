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

"""Unit tests for heatmap utilities."""

import unittest

import torch

from autoware_ml.models.detection3d.task_modules.heatmap import (
    _vectorize_gaussian2d,
    vectorize_gaussian_radii,
    create_gaussian_heatmaps,
)


class TestVectorizeGaussianRadii(unittest.TestCase):
    """Unit tests for the vectorize_gaussian_radii function."""

    def setUp(self) -> None:
        """Set up the same input tensors for all tests."""
        # (batch_size, 3)
        self.widths = torch.tensor([[1.0, 2.0, 3.2], [3.0, 4.4, 2.8]])
        self.heights = torch.tensor([[1.0, 2.0, 3.2], [3.0, 4.8, 5.0]])
        self.min_overlap = 0.1

    def test_vectorize_gaussian_radii(self) -> None:
        """Test the vectorize_gaussian_radii function."""
        gaussian_radii = vectorize_gaussian_radii(
            widths=self.widths,
            heights=self.heights,
            min_overlap=self.min_overlap,
        )

        self.assertEqual(gaussian_radii.shape, self.widths.shape)
        expected_radii = torch.tensor([[0, 0, 1], [1, 1, 1]], dtype=torch.int32)
        self.assertTrue(torch.allclose(gaussian_radii, expected_radii))


class TestVectorizeGaussian2D(unittest.TestCase):
    """Unit tests for the _vectorize_gaussian2d function."""

    def setUp(self) -> None:
        """Set up the same input tensors for all tests."""
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        # (batch_size, 3)
        self.widths = torch.tensor([[1, 2, 4], [3, 4, 5]], device=self.device, dtype=torch.int32)
        self.heights = torch.tensor([[1, 2, 4], [3, 4, 5]], device=self.device, dtype=torch.int32)
        self.min_overlap = 0.1
        # (batch_size, max_num_boxes, max_height, max_width)
        self.expected = torch.zeros(2, 3, 5, 5, device=self.device, dtype=torch.float32)

        # batch 0, box 0: 1x1
        self.expected[0, 0, :1, :1] = torch.tensor([[1.0]])

        # batch 0, box 1: 2x2
        self.expected[0, 1, :2, :2] = torch.tensor(
            [
                [0.0019, 0.0019],
                [0.0019, 0.0019],
            ]
        )

        # batch 0, box 2: 4x4
        self.expected[0, 2, :4, :4] = torch.tensor(
            [
                [0.0000e00, 9.2925e-07, 9.2925e-07, 0.0000e00],
                [9.2925e-07, 6.2177e-02, 6.2177e-02, 9.2925e-07],
                [9.2925e-07, 6.2177e-02, 6.2177e-02, 9.2925e-07],
                [0.0000e00, 9.2925e-07, 9.2925e-07, 0.0000e00],
            ]
        )

        # batch 1, box 0: 3x3
        self.expected[1, 0, :3, :3] = torch.tensor(
            [
                [1.4945e-05, 3.8659e-03, 1.4945e-05],
                [3.8659e-03, 1.0000e00, 3.8659e-03],
                [1.4945e-05, 3.8659e-03, 1.4945e-05],
            ]
        )

        # batch 1, box 1: 4x4
        self.expected[1, 1, :4, :4] = torch.tensor(
            [
                [7.8115e-07, 4.0465e-04, 4.0465e-04, 7.8115e-07],
                [4.0465e-04, 2.0961e-01, 2.0961e-01, 4.0465e-04],
                [4.0465e-04, 2.0961e-01, 2.0961e-01, 4.0465e-04],
                [7.8115e-07, 4.0465e-04, 4.0465e-04, 7.8115e-07],
            ]
        )

        # batch 1, box 2: 5x5
        self.expected[1, 2] = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 3.7267e-06, 0.0, 0.0],
                [0.0, 3.7267e-06, 1.0000e00, 3.7267e-06, 0.0],
                [0.0, 0.0, 3.7267e-06, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

    def test_vectorize_gaussian2d(self) -> None:
        """Test the _vectorize_gaussian2d function with 5x5 output but different sigma for each inputs."""
        gaussian_2d = _vectorize_gaussian2d(
            heights=self.heights,
            widths=self.widths,
            sigmas=torch.tensor(
                [[0.1, 0.2, 0.3], [0.3, 0.4, 0.2]], device=self.device, dtype=torch.float32
            ),  # Example sigmas
            valid_masks=torch.tensor([[1, 1, 1], [1, 1, 1]], device=self.device, dtype=torch.bool),
            device=self.device,
            dtype=torch.float32,
        )

        self.assertTrue(torch.allclose(gaussian_2d, self.expected, atol=1e-4))
        self.assertEqual(gaussian_2d.shape[:2], self.widths.shape)

    def test_vectorize_gaussian2d_invalid_mask(self) -> None:
        """Test the _vectorize_gaussian2d function with 5x5 output but the last box is invalid."""
        gaussian_2d = _vectorize_gaussian2d(
            heights=self.heights,
            widths=self.widths,
            sigmas=torch.tensor(
                [[0.1, 0.2, 0.3], [0.3, 0.4, 0.2]], device=self.device, dtype=torch.float32
            ),  # Example sigmas
            valid_masks=torch.tensor([[1, 1, 1], [1, 1, 0]], device=self.device, dtype=torch.bool),
            device=self.device,
            dtype=torch.float32,
        )

        # batch 1, box 2: 5x5
        # Set to zeros since valid_masks indicates this box is invalid
        self.expected[1, 2] = torch.zeros((5, 5), device=self.device, dtype=torch.float32)
        self.assertTrue(torch.allclose(gaussian_2d, self.expected, atol=1e-4))
        self.assertEqual(gaussian_2d.shape[:2], self.widths.shape)


class TestCreateGaussianHeatmap(unittest.TestCase):
    """Unit tests for the create_gaussian_heatmap function."""

    def setUp(self) -> None:
        """Set up the same input tensors for all tests."""
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.heatmap_width = 8
        self.heatmap_height = 8
        self.num_classes = 5
        self.batch_size = 2
        self.gt_bboxes_labels = torch.tensor(
            [[0, 0, 2], [1, 2, 2]], device=self.device, dtype=torch.int64
        )
        # (batch_size, 3, 2)
        self.centers = torch.tensor(
            [[[1, 2], [3, 4], [5, 6]], [[2, 3], [4, 5], [6, 7]]],
            device=self.device,
            dtype=torch.int64,
        )
        self.gaussian_radii = torch.tensor(
            [[1, 2, 1], [4, 1, 3]], device=self.device, dtype=torch.int32
        )
        self.expected_heatmap = torch.zeros(
            self.batch_size,
            self.num_classes,
            self.heatmap_height,
            self.heatmap_width,
            device=self.device,
            dtype=torch.float32,
        )

        # Manually draw the expected heatmaps for each valid box in the batch
        # Batch 0, Class 0, center (1, 2) and radius 1, center (3, 4) and radius 2
        self.expected_heatmap[0, 0] = torch.tensor(
            [
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                [0.0183, 0.1353, 0.0183, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                [0.1353, 1.0000, 0.1353, 0.0561, 0.0273, 0.0032, 0.0000, 0.0000],
                [0.0183, 0.1353, 0.2369, 0.4868, 0.2369, 0.0273, 0.0000, 0.0000],
                [0.0000, 0.0561, 0.4868, 1.0000, 0.4868, 0.0561, 0.0000, 0.0000],
                [0.0000, 0.0273, 0.2369, 0.4868, 0.2369, 0.0273, 0.0000, 0.0000],
                [0.0000, 0.0032, 0.0273, 0.0561, 0.0273, 0.0032, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            ],
            dtype=torch.float32,
            device=self.device,
        )

        # Batch 0, Class 2, center (5, 6) and radius 1
        self.expected_heatmap[0, 2, 5:8, :] = torch.tensor(
            [
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0183, 0.1353, 0.0183, 0.0000],
                [0.0000, 0.0000, 0.0000, 0.0000, 0.1353, 1.0000, 0.1353, 0.0000],
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0183, 0.1353, 0.0183, 0.0000],
            ],
            dtype=torch.float32,
            device=self.device,
        )

        # Batch 1, Class 1, center (2, 3) and radius 4
        self.expected_heatmap[1, 1] = torch.tensor(
            [
                [
                    5.5638e-02,
                    1.0837e-01,
                    1.3534e-01,
                    1.0837e-01,
                    5.5638e-02,
                    1.8316e-02,
                    3.8659e-03,
                    0.0000,
                ],
                [
                    1.6901e-01,
                    3.2919e-01,
                    4.1111e-01,
                    3.2919e-01,
                    1.6901e-01,
                    5.5638e-02,
                    1.1744e-02,
                    0.0000,
                ],
                [
                    3.2919e-01,
                    6.4118e-01,
                    8.0074e-01,
                    6.4118e-01,
                    3.2919e-01,
                    1.0837e-01,
                    2.2873e-02,
                    0.0000,
                ],
                [
                    4.1111e-01,
                    8.0074e-01,
                    1.0000e00,
                    8.0074e-01,
                    4.1111e-01,
                    1.3534e-01,
                    2.8566e-02,
                    0.0000,
                ],
                [
                    3.2919e-01,
                    6.4118e-01,
                    8.0074e-01,
                    6.4118e-01,
                    3.2919e-01,
                    1.0837e-01,
                    2.2873e-02,
                    0.0000,
                ],
                [
                    1.6901e-01,
                    3.2919e-01,
                    4.1111e-01,
                    3.2919e-01,
                    1.6901e-01,
                    5.5638e-02,
                    1.1744e-02,
                    0.0000,
                ],
                [
                    5.5638e-02,
                    1.0837e-01,
                    1.3534e-01,
                    1.0837e-01,
                    5.5638e-02,
                    1.8316e-02,
                    3.8659e-03,
                    0.0000,
                ],
                [
                    1.1744e-02,
                    2.2873e-02,
                    2.8566e-02,
                    2.2873e-02,
                    1.1744e-02,
                    3.8659e-03,
                    8.1599e-04,
                    0.0000,
                ],
            ],
            dtype=torch.float32,
            device=self.device,
        )

        # Batch 1, Class 2, center (4, 5) and radius 1, center (6, 7) and radius 3
        self.expected_heatmap[1, 2, 4:8, 3:8] = torch.tensor(
            [
                [0.0183, 0.1353, 0.0254, 0.0367, 0.0254],
                [0.1353, 1.0000, 0.1593, 0.2301, 0.1593],
                [0.0254, 0.1593, 0.4797, 0.6926, 0.4797],
                [0.0367, 0.2301, 0.6926, 1.0000, 0.6926],
            ],
            dtype=torch.float32,
            device=self.device,
        )

    def test_create_gaussian_heatmaps(self) -> None:
        """Test create_gaussian_heatmap function with 8x8 output but different sigma for each inputs."""
        gaussian_heatmaps = create_gaussian_heatmaps(
            heatmap_width=self.heatmap_width,
            heatmap_height=self.heatmap_height,
            num_classes=self.num_classes,
            centers=self.centers,
            gaussian_radii=self.gaussian_radii,
            gt_bboxes_labels=self.gt_bboxes_labels,
            valid_masks=torch.tensor([[1, 1, 1], [1, 1, 1]], device=self.device, dtype=torch.bool),
            batch_size=self.batch_size,
            device=self.device,
        )

        self.assertEqual(
            gaussian_heatmaps.shape,
            (self.batch_size, self.num_classes, self.heatmap_height, self.heatmap_width),
        )
        self.assertTrue(torch.allclose(gaussian_heatmaps, self.expected_heatmap, atol=1e-4))

    def test_create_gaussian_heatmaps_with_invalid_mask(self) -> None:
        """Test create_gaussian_heatmap function with 8x8 output with invalid bboxes."""
        gaussian_heatmaps = create_gaussian_heatmaps(
            heatmap_width=self.heatmap_width,
            heatmap_height=self.heatmap_height,
            num_classes=self.num_classes,
            centers=self.centers,
            gaussian_radii=self.gaussian_radii,
            gt_bboxes_labels=self.gt_bboxes_labels,
            valid_masks=torch.tensor([[1, 0, 1], [0, 1, 1]], device=self.device, dtype=torch.bool),
            batch_size=self.batch_size,
            device=self.device,
        )

        self.assertEqual(
            gaussian_heatmaps.shape,
            (self.batch_size, self.num_classes, self.heatmap_height, self.heatmap_width),
        )
        # For batch 1, the first box is invalid, so the heatmap for class 1 should be all zeros
        self.expected_heatmap[1, 1] = torch.zeros(
            (self.heatmap_height, self.heatmap_width), device=self.device, dtype=torch.float32
        )
        # For batch 0, the second box is invalid, so the heatmap for class 0 should only have
        # the first box's contribution, where 1.0 from the second box (center at 3, 4) is removed
        self.expected_heatmap[
            0,
            0,
        ] = torch.tensor(
            [
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                [0.0183, 0.1353, 0.0183, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                [0.1353, 1.0000, 0.1353, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                [0.0183, 0.1353, 0.0183, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            ],
            dtype=torch.float32,
            device=self.device,
        )
        self.assertTrue(torch.allclose(gaussian_heatmaps, self.expected_heatmap, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
