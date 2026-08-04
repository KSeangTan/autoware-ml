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

from autoware_ml.models.detection3d.task_modules.heatmap import vectorize_gaussian_radii


class TestHeatmapUtilities(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
