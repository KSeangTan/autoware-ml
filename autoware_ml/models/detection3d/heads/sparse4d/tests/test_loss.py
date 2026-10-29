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

"""Unit tests for the Sparse4D box loss."""

from __future__ import annotations

import math
import unittest

import torch

from autoware_ml.models.detection3d.heads.sparse4d.box3d import ANCHOR_DIMS, AnchorFieldIndex
from autoware_ml.models.detection3d.heads.sparse4d.loss import SparseBox3DLoss


def _heading(sin: float, cos: float) -> torch.Tensor:
    """An anchor at the origin with the given heading."""
    box = torch.zeros(1, ANCHOR_DIMS)
    box[0, AnchorFieldIndex.SIN_YAW] = sin
    box[0, AnchorFieldIndex.COS_YAW] = cos
    return box


class TestSparseBox3DLoss(unittest.TestCase):
    """The box, centerness and yawness losses."""

    def test_box_loss_is_the_weighted_l1_over_the_average_factor(self) -> None:
        """
        Input: a zero box, a target of ones on every field, unit weights and ``avg_factor=2``.
        Expected: 11 fields off by one, summed, halved and scaled by 0.25.
        Check: the box loss equals 1.375 and the quality losses are None.
        """
        loss = SparseBox3DLoss(box_weight=0.25)

        losses = loss(
            torch.zeros(1, ANCHOR_DIMS),
            torch.ones(1, ANCHOR_DIMS),
            weight=torch.ones(1, ANCHOR_DIMS),
            avg_factor=2.0,
        )

        torch.testing.assert_close(losses.loss_box, torch.tensor(1.375))
        self.assertIsNone(losses.loss_centerness)
        self.assertIsNone(losses.loss_yawness)

    def test_zero_weights_drop_fields_and_the_default_factor_is_the_box_count(self) -> None:
        """
        Input: two zero boxes with targets of ones, weights that keep only the first field,
            no average factor and a box weight of one.
        Expected: two fields off by one over two boxes.
        Check: the box loss equals 1.
        """
        loss = SparseBox3DLoss(box_weight=1.0)
        weight = torch.zeros(2, ANCHOR_DIMS)
        weight[:, 0] = 1.0

        losses = loss(torch.zeros(2, ANCHOR_DIMS), torch.ones(2, ANCHOR_DIMS), weight=weight)

        torch.testing.assert_close(losses.loss_box, torch.tensor(1.0))

    def test_reversible_classes_flip_the_target_heading(self) -> None:
        """
        Input: a prediction heading (0, -1) against a target (0, 1), for a box of class 3
            with class 3 reversible, and for a box of class 1.
        Expected: the reversible target flips to (0, -1) and costs nothing; the other costs
            2 on the cosine.
        Check: box losses 0 and 2 with a box weight of one.
        """
        loss = SparseBox3DLoss(box_weight=1.0, cls_allow_reverse=[3])
        box = _heading(0.0, -1.0)
        target = _heading(0.0, 1.0)

        reversible = loss(box, target, cls_target=torch.tensor([3]))
        fixed = loss(box, target, cls_target=torch.tensor([1]))

        torch.testing.assert_close(reversible.loss_box, torch.tensor(0.0))
        torch.testing.assert_close(fixed.loss_box, torch.tensor(2.0))
        torch.testing.assert_close(target, _heading(0.0, 1.0))

    def test_centerness_targets_the_exponential_of_the_center_error(self) -> None:
        """
        Input: a prediction at the target center with a zero centerness logit.
        Expected: the target is exp(0) = 1, so the loss is the cross-entropy of probability
            0.5 against 1.
        Check: the centerness loss equals log 2.
        """
        loss = SparseBox3DLoss()
        box = torch.zeros(1, ANCHOR_DIMS)
        quality = torch.zeros(1, 2)

        losses = loss(box, box, quality=quality, avg_factor=1.0)

        torch.testing.assert_close(losses.loss_centerness, torch.tensor(math.log(2.0)))

    def test_centerness_target_follows_the_distance(self) -> None:
        """
        Input: a prediction 1 m off the target center and a centerness logit of 0.
        Expected: the target exp(-1), so the loss is the cross-entropy of 0.5 against it,
            which is log 2 regardless of the target.
        Check: the loss equals log 2, and a logit of 20 gives (1 - exp(-1)) * 20.
        """
        loss = SparseBox3DLoss()
        box = torch.zeros(1, ANCHOR_DIMS)
        target = torch.zeros(1, ANCHOR_DIMS)
        target[0, AnchorFieldIndex.X] = 1.0
        quality = torch.zeros(1, 2)
        quality[0, 0] = 20.0

        losses = loss(box, target, quality=quality, avg_factor=1.0)

        torch.testing.assert_close(
            losses.loss_centerness,
            torch.tensor((1.0 - math.exp(-1.0)) * 20.0),
            atol=1e-4,
            rtol=1e-4,
        )

    def test_yawness_targets_whether_the_heading_agrees(self) -> None:
        """
        Input: an aligned heading with a yawness logit of 0, and an opposed heading with the
            same logit.
        Expected: a focal loss of -log(0.5) * 0.5^2 in both cases, as the positive and the
            negative branches mirror each other at probability 0.5.
        Check: both yawness losses equal 0.25 * log 2, and a confident aligned logit gives
            nearly zero.
        """
        loss = SparseBox3DLoss()
        quality = torch.zeros(1, 2)

        aligned = loss(_heading(0.0, 1.0), _heading(0.0, 1.0), quality=quality, avg_factor=1.0)
        opposed = loss(_heading(0.0, -1.0), _heading(0.0, 1.0), quality=quality, avg_factor=1.0)
        confident = loss(
            _heading(0.0, 1.0),
            _heading(0.0, 1.0),
            quality=torch.tensor([[0.0, 20.0]]),
            avg_factor=1.0,
        )

        expected = torch.tensor(0.25 * math.log(2.0))
        torch.testing.assert_close(aligned.loss_yawness, expected)
        torch.testing.assert_close(opposed.loss_yawness, expected)
        self.assertLess(confident.loss_yawness.item(), 1e-6)

    def test_quality_weights_scale_their_losses(self) -> None:
        """
        Input: the aligned case with centerness and yawness weights of 2.
        Expected: both quality losses double.
        Check: 2 log 2 and 0.5 log 2.
        """
        loss = SparseBox3DLoss(centerness_weight=2.0, yawness_weight=2.0)

        losses = loss(
            _heading(0.0, 1.0), _heading(0.0, 1.0), quality=torch.zeros(1, 2), avg_factor=1.0
        )

        torch.testing.assert_close(losses.loss_centerness, torch.tensor(2.0 * math.log(2.0)))
        torch.testing.assert_close(losses.loss_yawness, torch.tensor(0.5 * math.log(2.0)))
