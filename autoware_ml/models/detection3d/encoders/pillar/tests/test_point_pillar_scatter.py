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

"""Unit tests for PointPillarsScatter."""

import unittest

import torch

from autoware_ml.models.detection3d.encoders.pillar.point_pillar_scatter import PointPillarsScatter


class TestPointPillarScatter(unittest.TestCase):
    def setUp(self) -> None:
        """Set up the same PFNLayer instance for all tests. Note that this class will
        be called in each test case.
        """
        torch.manual_seed(0)
        self.point_pillars_scatter = PointPillarsScatter(
            in_channels=8,
            output_shape=[16, 16],
            batch_size=2,
        )
        self.pillar_features = torch.tensor(
            [
                [0.4582, 0.5374, 1.9813, 0.5429, 1.9834, 0.6387, 2.1574, 0.5094],
                [0.5929, 0.5374, 0.3992, 0.5730, 0.0000, 0.6387, 0.0000, 0.5764],
            ],
            dtype=torch.float32,
        )
        self.batch_indices = torch.tensor([0, 1], dtype=torch.int32)
        self.coords = torch.tensor([[1, 2, 0], [3, 4, 0]], dtype=torch.int32)

    def test_forward(self) -> None:
        """Test the forward function of PFNLayer."""

        outputs = self.point_pillars_scatter(
            pillar_features=self.pillar_features,
            coords=self.coords,
            batch_indices=self.batch_indices,
        )
        self.assertEqual(outputs.shape, (2, 8, 16, 16))
        # x moves the fastest, y moves the second fastest, and batch_index moves the slowest
        # x = 1, y = 2, batch_index = 0, where x is width, y is height, and batch_index is the index of the batch
        self.assertTrue(torch.allclose(outputs[0, :, 2, 1], self.pillar_features[0]))
        # x = 3, y = 4, batch_index = 1, where x is width, y is height, and batch_index is the index of the batch
        self.assertTrue(torch.allclose(outputs[1, :, 4, 3], self.pillar_features[1]))


if __name__ == "__main__":
    unittest.main()
