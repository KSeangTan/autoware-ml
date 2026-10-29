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

"""Unit tests for the Sparse4D target assignment and denoising."""

from __future__ import annotations

import math
import unittest

import torch

from autoware_ml.models.detection3d.heads.sparse4d.box3d import ANCHOR_DIMS, AnchorFieldIndex
from autoware_ml.models.detection3d.heads.sparse4d.target import (
    NEGATIVE_DN_CLASS,
    SparseBox3DTarget,
    TemporalDenoisingCache,
    encode_reg_target,
)
from autoware_ml.types.geometry import Box3DFieldIndex

NUM_CLASSES = 2


def _metric_box(x: float, yaw: float = 0.0) -> torch.Tensor:
    """A unit box at (x, 0, 0) with the given yaw and no velocity, in the Box3DFieldIndex layout."""
    box = torch.zeros(len(Box3DFieldIndex))
    box[Box3DFieldIndex.X] = x
    box[Box3DFieldIndex.LENGTH] = 1.0
    box[Box3DFieldIndex.WIDTH] = 1.0
    box[Box3DFieldIndex.HEIGHT] = 1.0
    box[Box3DFieldIndex.YAW] = yaw
    return box


def _anchor(x: float) -> torch.Tensor:
    """The encoding of ``_metric_box(x)``."""
    return encode_reg_target(_metric_box(x).unsqueeze(0)).squeeze(0)


def _logits(class_per_anchor: list[int]) -> torch.Tensor:
    """Logits of one sample confident in the given class of every anchor."""
    logits = torch.full((len(class_per_anchor), NUM_CLASSES), -5.0)
    for index, cls in enumerate(class_per_anchor):
        logits[index, cls] = 5.0
    return logits


class TestEncodeRegTarget(unittest.TestCase):
    """The box encoding."""

    def test_encodes_sizes_in_log_and_yaw_as_sin_cos(self) -> None:
        """
        Input: a box at (1, 2, 3) of sizes (e, e^2, 1), yaw pi/2 and velocity (4, 5, 6).
        Expected: the log sizes, sin 1, cos 0 and the velocity.
        Check: the 11 encoded fields.
        """
        box = torch.tensor([1.0, 2.0, 3.0, math.e, math.e**2, 1.0, math.pi / 2, 4.0, 5.0, 6.0])

        encoded = encode_reg_target(box.unsqueeze(0))

        expected = torch.tensor([[1.0, 2.0, 3.0, 1.0, 2.0, 0.0, 1.0, 0.0, 4.0, 5.0, 6.0]])
        torch.testing.assert_close(encoded, expected)

    def test_cuts_to_the_asked_width(self) -> None:
        """
        Input: the same box and ``box_dims=10``.
        Expected: the vertical velocity is dropped.
        Check: ten fields ending with the planar velocity.
        """
        box = torch.tensor([1.0, 2.0, 3.0, 1.0, 1.0, 1.0, 0.0, 4.0, 5.0, 6.0])

        encoded = encode_reg_target(box.unsqueeze(0), box_dims=10)

        self.assertEqual(encoded.shape, (1, 10))
        torch.testing.assert_close(encoded[0, 8:], torch.tensor([4.0, 5.0]))


class TestSparseBox3DTargetSample(unittest.TestCase):
    """The Hungarian matching of the anchors."""

    def setUp(self) -> None:
        """Set up three anchors at x = 0, 5, 10 and two boxes at x = 10 (class 1) and x = 0."""
        self.target = SparseBox3DTarget()
        self.box_pred = torch.stack([_anchor(0.0), _anchor(5.0), _anchor(10.0)]).unsqueeze(0)
        self.cls_pred = _logits([0, 0, 1]).unsqueeze(0)
        self.gt_boxes = [torch.stack([_metric_box(10.0), _metric_box(0.0)])]
        self.gt_labels = [torch.tensor([1, 0])]

    def test_matches_the_nearest_anchor_of_the_right_class(self) -> None:
        """
        Input: the anchors and the boxes of the fixture.
        Expected: anchor 2 takes the class 1 box at x = 10, anchor 0 the class 0 box at
            x = 0 and anchor 1 stays unmatched.
        Check: the class targets, the box targets and the regression weights.
        """
        matched = self.target.sample(self.cls_pred, self.box_pred, self.gt_labels, self.gt_boxes)

        torch.testing.assert_close(matched.cls_target, torch.tensor([[0, NUM_CLASSES, 1]]))
        torch.testing.assert_close(matched.box_target[0, 0], _anchor(0.0))
        torch.testing.assert_close(matched.box_target[0, 2], _anchor(10.0))
        torch.testing.assert_close(matched.box_target[0, 1], torch.zeros(ANCHOR_DIMS))
        torch.testing.assert_close(matched.reg_weights[0, 0], torch.ones(ANCHOR_DIMS))
        torch.testing.assert_close(matched.reg_weights[0, 1], torch.zeros(ANCHOR_DIMS))

    def test_undefined_fields_get_zero_weight_and_target(self) -> None:
        """
        Input: a box whose velocity is NaN.
        Expected: the matched anchor gets a zero target and weight on those fields.
        Check: the velocity target and weight of the anchor matched to it.
        """
        boxes = self.gt_boxes[0].clone()
        boxes[0, Box3DFieldIndex.VELOCITY_X :] = float("nan")

        matched = self.target.sample(self.cls_pred, self.box_pred, self.gt_labels, [boxes])

        velocity = slice(AnchorFieldIndex.VELOCITY_X, ANCHOR_DIMS)
        torch.testing.assert_close(matched.box_target[0, 2, velocity], torch.zeros(3))
        torch.testing.assert_close(matched.reg_weights[0, 2, velocity], torch.zeros(3))
        torch.testing.assert_close(matched.reg_weights[0, 2, :8], torch.ones(8))

    def test_class_wise_weights_replace_the_weights_of_that_class(self) -> None:
        """
        Input: class-wise weights of 2 for class 0.
        Expected: the anchor matched to the class 0 box carries weights 2, the other ones.
        Check: the regression weights of anchors 0 and 2.
        """
        target = SparseBox3DTarget(cls_wise_reg_weights={0: [2.0] * ANCHOR_DIMS})

        matched = target.sample(self.cls_pred, self.box_pred, self.gt_labels, self.gt_boxes)

        torch.testing.assert_close(matched.reg_weights[0, 0], torch.full((ANCHOR_DIMS,), 2.0))
        torch.testing.assert_close(matched.reg_weights[0, 2], torch.ones(ANCHOR_DIMS))

    def test_a_sample_without_boxes_is_all_background(self) -> None:
        """
        Input: a batch of two samples, the second without boxes.
        Expected: every anchor of the second sample is background with zero targets.
        Check: the class targets and weights of the second sample.
        """
        matched = self.target.sample(
            self.cls_pred.repeat(2, 1, 1),
            self.box_pred.repeat(2, 1, 1),
            [self.gt_labels[0], torch.zeros(0, dtype=torch.long)],
            [self.gt_boxes[0], torch.zeros(0, len(Box3DFieldIndex))],
        )

        torch.testing.assert_close(matched.cls_target[1], torch.full((3,), NUM_CLASSES))
        torch.testing.assert_close(matched.reg_weights[1], torch.zeros(3, ANCHOR_DIMS))
        torch.testing.assert_close(matched.cls_target[0], torch.tensor([0, NUM_CLASSES, 1]))

    def test_narrower_predictions_cut_the_targets(self) -> None:
        """
        Input: predictions of ten fields.
        Expected: the targets and weights have ten fields.
        Check: the shapes.
        """
        matched = self.target.sample(
            self.cls_pred, self.box_pred[..., :10], self.gt_labels, self.gt_boxes
        )

        self.assertEqual(matched.box_target.shape, (1, 3, 10))
        self.assertEqual(matched.reg_weights.shape, (1, 3, 10))

    def test_rejects_reg_weights_of_the_wrong_width(self) -> None:
        """
        Input: ten regression weights for eleven-field predictions.
        Expected: the mismatch is refused.
        Check: ValueError.
        """
        target = SparseBox3DTarget(reg_weights=[1.0] * 10)

        with self.assertRaisesRegex(ValueError, "reg_weights"):
            target.sample(self.cls_pred, self.box_pred, self.gt_labels, self.gt_boxes)


class TestSparseBox3DTargetDenoising(unittest.TestCase):
    """The noised anchors."""

    def setUp(self) -> None:
        """Set up two samples with two and one boxes far apart."""
        self.gt_boxes = [
            torch.stack([_metric_box(0.0), _metric_box(20.0)]),
            torch.stack([_metric_box(40.0)]),
        ]
        self.gt_labels = [torch.tensor([0, 1]), torch.tensor([1])]
        self.gt_ids = [torch.tensor([7, 8]), torch.tensor([9])]

    def test_without_noise_the_anchors_are_the_boxes(self) -> None:
        """
        Input: two groups, no negatives and a noise scale of zero.
        Expected: every group holds the encoded boxes, matched to themselves; the padding of
            the second sample is invalid with class -1 and the groups cannot attend to each
            other.
        Check: the anchors, the targets, the valid mask and the attention mask.
        """
        target = SparseBox3DTarget(num_dn_groups=2, dn_noise_scale=0.0, add_neg_dn=False)

        denoising = target.get_dn_anchors(self.gt_labels, self.gt_boxes)

        self.assertEqual(denoising.anchors.shape, (2, 4, ANCHOR_DIMS))
        torch.testing.assert_close(denoising.anchors[0, 1], _anchor(20.0))
        torch.testing.assert_close(denoising.anchors[0, 3], _anchor(20.0))
        torch.testing.assert_close(denoising.box_target, denoising.anchors)
        torch.testing.assert_close(
            denoising.cls_target, torch.tensor([[0, 1, 0, 1], [1, -1, 1, -1]])
        )
        torch.testing.assert_close(
            denoising.valid_mask,
            torch.tensor([[True, True, True, True], [True, False, True, False]]),
        )
        expected_mask = torch.tensor(
            [
                [False, False, True, True],
                [False, False, True, True],
                [True, True, False, False],
                [True, True, False, False],
            ]
        )
        torch.testing.assert_close(denoising.attn_mask, expected_mask)
        self.assertIsNone(denoising.id_target)

    def test_noise_stays_within_the_scale_and_negatives_beyond_it(self) -> None:
        """
        Input: one group with negatives and a noise scale of 0.1 on boxes 20 m apart.
        Expected: the two positives stay within 0.1 of their box and are matched to it, the
            two negatives sit between 0.1 and 0.2 away per field and get the class -3; all
            four come from real boxes.
        Check: the distances, the class targets and the valid mask of the first sample.
        """
        torch.manual_seed(0)
        target = SparseBox3DTarget(num_dn_groups=1, dn_noise_scale=0.1, add_neg_dn=True)

        denoising = target.get_dn_anchors([self.gt_labels[0]], [self.gt_boxes[0]])

        self.assertEqual(denoising.anchors.shape, (1, 4, ANCHOR_DIMS))
        positives = denoising.cls_target[0] >= 0
        torch.testing.assert_close(positives, torch.tensor([True, True, False, False]))
        torch.testing.assert_close(denoising.cls_target[0, 2:], torch.full((2,), NEGATIVE_DN_CLASS))
        distance = (denoising.anchors[0, :2] - denoising.box_target[0, :2]).abs()
        self.assertLessEqual(distance.max().item(), 0.1)
        boxes = torch.stack([_anchor(0.0), _anchor(20.0)])
        negative_distance = (denoising.anchors[0, 2:] - boxes).abs()
        self.assertGreaterEqual(negative_distance.min().item(), 0.1)
        self.assertLessEqual(negative_distance.max().item(), 0.2)
        self.assertTrue(bool(denoising.valid_mask.all()))

    def test_per_field_noise_scale_and_box_limit(self) -> None:
        """
        Input: a per-field noise scale that only moves x, and ``max_dn_gt=1``.
        Expected: only the first box of every sample is noised, and only along x.
        Check: one noised anchor per sample whose non-x fields equal the box.
        """
        scale = [1.0] + [0.0] * (ANCHOR_DIMS - 1)
        target = SparseBox3DTarget(
            num_dn_groups=1, dn_noise_scale=scale, add_neg_dn=False, max_dn_gt=1
        )

        denoising = target.get_dn_anchors(self.gt_labels, self.gt_boxes)

        self.assertEqual(denoising.anchors.shape, (2, 1, ANCHOR_DIMS))
        torch.testing.assert_close(denoising.anchors[0, 0, 1:], _anchor(0.0)[1:])
        torch.testing.assert_close(denoising.anchors[1, 0, 1:], _anchor(40.0)[1:])

    def test_rejects_a_noise_scale_of_the_wrong_width(self) -> None:
        """
        Input: ten noise scales for eleven-field anchors.
        Expected: the mismatch is refused.
        Check: ValueError.
        """
        target = SparseBox3DTarget(num_dn_groups=1, dn_noise_scale=[0.5] * 10)

        with self.assertRaisesRegex(ValueError, "dn_noise_scale"):
            target.get_dn_anchors(self.gt_labels, self.gt_boxes)

    def test_instance_ids_follow_the_matching(self) -> None:
        """
        Input: temporal groups enabled and instance ids 7, 8 and 9.
        Expected: every noised anchor carries the id of its box, -1 on the padding.
        Check: the id targets without noise.
        """
        target = SparseBox3DTarget(
            num_dn_groups=1, dn_noise_scale=0.0, add_neg_dn=False, num_temp_dn_groups=1
        )

        denoising = target.get_dn_anchors(self.gt_labels, self.gt_boxes, self.gt_ids)

        torch.testing.assert_close(denoising.id_target, torch.tensor([[7, 8], [9, -1]]))

    def test_instance_ids_are_ignored_without_temporal_groups(self) -> None:
        """
        Input: instance ids but no temporal groups.
        Expected: no id targets.
        Check: None.
        """
        target = SparseBox3DTarget(num_dn_groups=1, dn_noise_scale=0.0, add_neg_dn=False)

        denoising = target.get_dn_anchors(self.gt_labels, self.gt_boxes, self.gt_ids)

        self.assertIsNone(denoising.id_target)

    def test_no_groups_or_no_boxes_give_nothing(self) -> None:
        """
        Input: a target without groups, and one with groups but samples without boxes.
        Expected: no denoising in both cases.
        Check: None.
        """
        self.assertIsNone(SparseBox3DTarget().get_dn_anchors(self.gt_labels, self.gt_boxes))
        target = SparseBox3DTarget(num_dn_groups=2)
        empty = [torch.zeros(0, dtype=torch.long)]
        self.assertIsNone(target.get_dn_anchors(empty, [torch.zeros(0, len(Box3DFieldIndex))]))


class TestSparseBox3DTargetTemporalDenoising(unittest.TestCase):
    """The denoising groups carried between frames."""

    def setUp(self) -> None:
        """Set up two groups of two noised anchors per sample after two learnable instances."""
        self.target = SparseBox3DTarget(num_dn_groups=2, num_temp_dn_groups=1)
        self.embed_dims = 3
        self.instance_feature = torch.arange(2 * 6 * self.embed_dims, dtype=torch.float32).reshape(
            2, 6, self.embed_dims
        )
        self.anchor = torch.arange(2 * 6 * ANCHOR_DIMS, dtype=torch.float32).reshape(
            2, 6, ANCHOR_DIMS
        )
        self.box_target = torch.stack([_anchor(0.0), _anchor(20.0)]).repeat(2, 2, 1)
        self.cls_target = torch.tensor([[0, 1, 0, 1], [0, 1, 0, 1]])
        self.valid_mask = torch.ones(2, 4, dtype=torch.bool)
        self.id_target = torch.tensor([[7, 8, 7, 8], [7, 8, 7, 8]])

    def test_cache_dn_keeps_one_group_per_sample(self) -> None:
        """
        Input: the four noised anchors of every sample.
        Expected: one of the two groups is cached, with its two anchors and targets.
        Check: the cache shapes and that the cached anchors are one of the groups.
        """
        self.target.cache_dn(
            self.instance_feature[:, 2:],
            self.anchor[:, 2:],
            self.cls_target,
            self.valid_mask,
            self.id_target,
        )

        cache = self.target.dn_cache
        self.assertEqual(cache.instance_feature.shape, (2, 1, 2, self.embed_dims))
        self.assertEqual(cache.anchor.shape, (2, 1, 2, ANCHOR_DIMS))
        self.assertEqual(cache.id_target.shape, (2, 1, 2))
        groups = self.anchor[:, 2:].reshape(2, 2, 2, ANCHOR_DIMS)
        matches_a_group = any(
            torch.equal(cache.anchor[:, 0], groups[:, group]) for group in range(2)
        )
        self.assertTrue(matches_a_group)

    def test_update_dn_brings_back_the_cached_group_by_instance_id(self) -> None:
        """
        Input: a cache whose two anchors carry the ids 8 and 7 (swapped) and the current
            frame whose first group carries 7 and 8.
        Expected: the first group becomes the cached one with the box targets re-ordered by
            id, the second group stays, and the learnable instances lead.
        Check: the features, the anchors, the box targets, the classes and the ids.
        """
        cached_feature = torch.full((2, 1, 2, self.embed_dims), -1.0)
        cached_anchor = torch.full((2, 1, 2, ANCHOR_DIMS), -2.0)
        self.target.dn_cache = TemporalDenoisingCache(
            instance_feature=cached_feature,
            anchor=cached_anchor,
            cls_target=torch.tensor([[[1, 0]], [[1, 0]]]),
            valid_mask=torch.ones(2, 1, 2, dtype=torch.bool),
            id_target=torch.tensor([[[8, 7]], [[8, 7]]]),
        )

        updated = self.target.update_dn(
            self.instance_feature,
            self.anchor,
            self.box_target,
            self.cls_target,
            self.valid_mask,
            self.id_target,
            num_normal_anchor=2,
            temporal_valid_mask=torch.tensor([True, True]),
        )

        torch.testing.assert_close(updated.instance_feature[:, :2], self.instance_feature[:, :2])
        torch.testing.assert_close(updated.instance_feature[:, 2:4], cached_feature[:, 0])
        torch.testing.assert_close(updated.instance_feature[:, 4:], self.instance_feature[:, 4:])
        torch.testing.assert_close(updated.anchor[:, 2:4], cached_anchor[:, 0])
        torch.testing.assert_close(updated.box_target[0, 0], _anchor(20.0))
        torch.testing.assert_close(updated.box_target[0, 1], _anchor(0.0))
        torch.testing.assert_close(updated.cls_target, torch.tensor([[1, 0, 0, 1], [1, 0, 0, 1]]))
        torch.testing.assert_close(updated.id_target, torch.tensor([[8, 7, 7, 8], [8, 7, 7, 8]]))

    def test_a_cached_box_that_left_the_frame_becomes_background(self) -> None:
        """
        Input: a cache with the id 9 that no current box carries.
        Expected: that anchor gets the class -1 and a zero box target.
        Check: the first anchor of the first group.
        """
        self.target.dn_cache = TemporalDenoisingCache(
            instance_feature=torch.zeros(2, 1, 2, self.embed_dims),
            anchor=torch.zeros(2, 1, 2, ANCHOR_DIMS),
            cls_target=torch.tensor([[[1, 0]], [[1, 0]]]),
            valid_mask=torch.ones(2, 1, 2, dtype=torch.bool),
            id_target=torch.tensor([[[9, 7]], [[9, 7]]]),
        )

        updated = self.target.update_dn(
            self.instance_feature,
            self.anchor,
            self.box_target,
            self.cls_target,
            self.valid_mask,
            self.id_target,
            num_normal_anchor=2,
            temporal_valid_mask=torch.tensor([True, True]),
        )

        self.assertEqual(updated.cls_target[0, 0].item(), -1)
        torch.testing.assert_close(updated.box_target[0, 0], torch.zeros(ANCHOR_DIMS))
        self.assertEqual(updated.cls_target[0, 1].item(), 0)

    def test_an_ended_lane_keeps_its_current_group(self) -> None:
        """
        Input: a cache and a temporal mask that is False for the second lane.
        Expected: the second lane keeps its current first group.
        Check: the features of the second lane are unchanged.
        """
        self.target.dn_cache = TemporalDenoisingCache(
            instance_feature=torch.full((2, 1, 2, self.embed_dims), -1.0),
            anchor=torch.full((2, 1, 2, ANCHOR_DIMS), -2.0),
            cls_target=torch.tensor([[[1, 0]], [[1, 0]]]),
            valid_mask=torch.ones(2, 1, 2, dtype=torch.bool),
            id_target=torch.tensor([[[8, 7]], [[8, 7]]]),
        )

        updated = self.target.update_dn(
            self.instance_feature,
            self.anchor,
            self.box_target,
            self.cls_target,
            self.valid_mask,
            self.id_target,
            num_normal_anchor=2,
            temporal_valid_mask=torch.tensor([True, False]),
        )

        torch.testing.assert_close(updated.instance_feature[1], self.instance_feature[1])
        torch.testing.assert_close(updated.box_target[1], self.box_target[1])
        torch.testing.assert_close(
            updated.instance_feature[0, 2:4], torch.full((2, self.embed_dims), -1.0)
        )

    def test_a_smaller_cache_is_padded(self) -> None:
        """
        Input: a cache with one noised anchor per group while the frame has two.
        Expected: the cached group is padded with a zero anchor that has no box.
        Check: the padded anchor, its class 0 and its invalid mask.
        """
        self.target.dn_cache = TemporalDenoisingCache(
            instance_feature=torch.full((2, 1, 1, self.embed_dims), -1.0),
            anchor=torch.full((2, 1, 1, ANCHOR_DIMS), -2.0),
            cls_target=torch.tensor([[[1]], [[1]]]),
            valid_mask=torch.ones(2, 1, 1, dtype=torch.bool),
            id_target=torch.tensor([[[8]], [[8]]]),
        )

        updated = self.target.update_dn(
            self.instance_feature,
            self.anchor,
            self.box_target,
            self.cls_target,
            self.valid_mask,
            self.id_target,
            num_normal_anchor=2,
            temporal_valid_mask=torch.tensor([True, True]),
        )

        torch.testing.assert_close(updated.anchor[0, 2], torch.full((ANCHOR_DIMS,), -2.0))
        torch.testing.assert_close(updated.anchor[0, 3], torch.zeros(ANCHOR_DIMS))
        self.assertFalse(bool(updated.valid_mask[0, 1]))
        torch.testing.assert_close(updated.box_target[0, 0], _anchor(20.0))

    def test_without_cache_or_temporal_mask_the_inputs_pass_through(self) -> None:
        """
        Input: no cache, and a cache with a None temporal mask.
        Expected: the inputs come back unchanged and the cache is dropped.
        Check: identity of the outputs and the None cache.
        """
        updated = self.target.update_dn(
            self.instance_feature,
            self.anchor,
            self.box_target,
            self.cls_target,
            self.valid_mask,
            self.id_target,
            num_normal_anchor=2,
            temporal_valid_mask=torch.tensor([True, True]),
        )
        self.assertIs(updated.instance_feature, self.instance_feature)

        self.target.dn_cache = TemporalDenoisingCache(
            instance_feature=torch.zeros(2, 1, 2, self.embed_dims),
            anchor=torch.zeros(2, 1, 2, ANCHOR_DIMS),
            cls_target=torch.zeros(2, 1, 2, dtype=torch.long),
            valid_mask=torch.ones(2, 1, 2, dtype=torch.bool),
            id_target=torch.zeros(2, 1, 2, dtype=torch.long),
        )
        updated = self.target.update_dn(
            self.instance_feature,
            self.anchor,
            self.box_target,
            self.cls_target,
            self.valid_mask,
            self.id_target,
            num_normal_anchor=2,
            temporal_valid_mask=None,
        )
        self.assertIs(updated.anchor, self.anchor)
        self.assertIsNone(self.target.dn_cache)
