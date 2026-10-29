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

"""Unit tests for the flattened feature map dataclass."""

from __future__ import annotations

import unittest

from pydantic import ValidationError
import torch

from autoware_ml.dataclasses.models.detection2d.feature_maps import FlattenedFeatureMaps


class TestFlattenedFeatureMaps(unittest.TestCase):
    """Unit tests for FlattenedFeatureMaps."""

    def setUp(self) -> None:
        """Set up two cameras with a 2x3 level and a 1x2 level each."""
        self.spatial_shapes = torch.tensor([[[2, 3], [1, 2]], [[2, 3], [1, 2]]])
        self.level_start_index = torch.tensor([[0, 6], [8, 14]])
        self.features = torch.zeros(1, 16, 4)

    def test_consistent_layout_is_accepted(self) -> None:
        """
        Input: 16 tokens, the level shapes of two cameras and the matching offsets.
        Expected: the dataclass keeps the tensors as given.
        Check: every attribute is the tensor it was built from.
        """
        flattened = FlattenedFeatureMaps(
            features=self.features,
            spatial_shapes=self.spatial_shapes,
            level_start_index=self.level_start_index,
        )

        self.assertIs(flattened.features, self.features)
        self.assertIs(flattened.spatial_shapes, self.spatial_shapes)
        self.assertIs(flattened.level_start_index, self.level_start_index)

    def test_different_pyramid_per_camera_is_accepted(self) -> None:
        """
        Input: camera 0 with a 1x2 level and camera 1 with a 2x1 level, 4 tokens, offsets 0 and 2.
        Expected: the layout allows a different pyramid per camera.
        Check: construction succeeds.
        """
        flattened = FlattenedFeatureMaps(
            features=torch.zeros(1, 4, 1),
            spatial_shapes=torch.tensor([[[1, 2]], [[2, 1]]]),
            level_start_index=torch.tensor([[0], [2]]),
        )

        self.assertEqual(flattened.features.shape, (1, 4, 1))

    def test_wrong_offsets_are_rejected(self) -> None:
        """
        Input: the offsets of the second camera shifted by one token.
        Expected: the offsets must be the cumulative sum of the level areas.
        Check: a ValidationError mentions level_start_index.
        """
        with self.assertRaisesRegex(ValidationError, "level_start_index"):
            FlattenedFeatureMaps(
                features=self.features,
                spatial_shapes=self.spatial_shapes,
                level_start_index=torch.tensor([[0, 6], [9, 15]]),
            )

    def test_wrong_offset_shape_is_rejected(self) -> None:
        """
        Input: offsets for one camera only while the shapes describe two cameras.
        Expected: the offsets must cover every level of every camera.
        Check: a ValidationError mentions level_start_index.
        """
        with self.assertRaisesRegex(ValidationError, "level_start_index"):
            FlattenedFeatureMaps(
                features=self.features,
                spatial_shapes=self.spatial_shapes,
                level_start_index=torch.tensor([[0, 6]]),
            )

    def test_wrong_token_count_is_rejected(self) -> None:
        """
        Input: 15 tokens while the level shapes add up to 16.
        Expected: the token axis must match the sum of the level areas.
        Check: a ValidationError mentions the 16 expected tokens.
        """
        with self.assertRaisesRegex(ValidationError, "16 tokens"):
            FlattenedFeatureMaps(
                features=torch.zeros(1, 15, 4),
                spatial_shapes=self.spatial_shapes,
                level_start_index=self.level_start_index,
            )

    def test_int32_shapes_are_rejected(self) -> None:
        """
        Input: the level shapes as int32.
        Expected: the dataclass holds int64 shapes; the kernel wrapper casts them itself.
        Check: a ValidationError is raised.
        """
        with self.assertRaises(ValidationError):
            FlattenedFeatureMaps(
                features=self.features,
                spatial_shapes=self.spatial_shapes.int(),
                level_start_index=self.level_start_index,
            )

    def test_is_frozen(self) -> None:
        """
        Input: a valid instance.
        Expected: the dataclass is immutable.
        Check: assigning features raises a ValidationError.
        """
        flattened = FlattenedFeatureMaps(
            features=self.features,
            spatial_shapes=self.spatial_shapes,
            level_start_index=self.level_start_index,
        )

        with self.assertRaises(ValidationError):
            flattened.features = torch.zeros(1, 16, 4)
