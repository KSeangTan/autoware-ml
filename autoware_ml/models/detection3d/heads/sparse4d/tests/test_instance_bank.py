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

"""Unit tests for the Sparse4D instance bank."""

from __future__ import annotations

import unittest

import torch

from autoware_ml.models.detection3d.heads.sparse4d.box3d import ANCHOR_DIMS, AnchorFieldIndex
from autoware_ml.models.detection3d.heads.sparse4d.instance_bank import InstanceBank

NUM_ANCHOR = 4
EMBED_DIMS = 2


def _anchors() -> torch.Tensor:
    """Four anchors at x = 0, 1, 2, 3 with unit sizes, heading along x and no velocity."""
    anchors = torch.zeros(NUM_ANCHOR, ANCHOR_DIMS)
    anchors[:, AnchorFieldIndex.X] = torch.arange(NUM_ANCHOR, dtype=torch.float32)
    anchors[:, AnchorFieldIndex.COS_YAW] = 1.0
    return anchors


def _translation(x: float) -> torch.Tensor:
    """A pose translated by x along the first axis."""
    pose = torch.eye(4)
    pose[0, 3] = x
    return pose


def _logits(confidence_per_anchor: list[float]) -> torch.Tensor:
    """Class logits of one sample whose per-anchor maximum is the given value."""
    logits = torch.full((1, len(confidence_per_anchor), 2), -10.0)
    logits[0, :, 0] = torch.tensor(confidence_per_anchor)
    return logits


class TestInstanceBank(unittest.TestCase):
    """The instances a frame starts from and the ones carried over."""

    def setUp(self) -> None:
        """Set up a bank of four anchors that keeps two temporal instances."""
        self.bank = InstanceBank(
            num_anchor=NUM_ANCHOR,
            embed_dims=EMBED_DIMS,
            anchor=_anchors(),
            num_temp_instances=2,
            default_time_interval=0.5,
            confidence_decay=0.6,
            max_time_interval=2.0,
        )
        with torch.no_grad():
            self.bank.instance_feature.copy_(torch.arange(8).reshape(NUM_ANCHOR, EMBED_DIMS))
        self.timestamp = torch.tensor([10.0], dtype=torch.float64)
        self.pose = torch.eye(4).unsqueeze(0)

    def _cache_frame(self, confidences: list[float], timestamp: float, x_offset: float) -> None:
        """Cache a frame whose anchors are the bank's ones moved by ``x_offset``."""
        state = self.bank.get(torch.tensor([timestamp], dtype=torch.float64), self.pose)
        anchor = state.anchor.clone()
        anchor[..., AnchorFieldIndex.X] += x_offset
        self.bank.cache(
            state.instance_feature,
            anchor,
            _logits(confidences),
            torch.tensor([timestamp], dtype=torch.float64),
            self.pose,
        )

    def test_first_frame_replicates_the_learnable_instances(self) -> None:
        """
        Input: the first frame of a batch of two lanes.
        Expected: every lane starts from the learnable anchors and features, without temporal
            instances and with the default time interval.
        Check: the replicated tensors, the None temporal entries and the interval 0.5.
        """
        state = self.bank.get(
            torch.tensor([1.0, 2.0], dtype=torch.float64), self.pose.repeat(2, 1, 1)
        )

        torch.testing.assert_close(state.anchor, _anchors().unsqueeze(0).repeat(2, 1, 1))
        torch.testing.assert_close(
            state.instance_feature, self.bank.instance_feature.detach().unsqueeze(0).repeat(2, 1, 1)
        )
        self.assertIsNone(state.temp_instance_feature)
        self.assertIsNone(state.temp_anchor)
        self.assertIsNone(state.temp_dn_anchor)
        torch.testing.assert_close(state.time_interval, torch.tensor([0.5, 0.5]))

    def test_cache_keeps_the_most_confident_instances(self) -> None:
        """
        Input: a frame whose anchors 3 and 1 are the most confident.
        Expected: the bank keeps those two, most confident first, with sigmoid confidences.
        Check: the cached anchors sit at x = 3 and x = 1 and the confidences match.
        """
        self._cache_frame([0.0, 2.0, -1.0, 3.0], timestamp=10.0, x_offset=0.0)

        torch.testing.assert_close(
            self.bank.cached_anchor[0, :, AnchorFieldIndex.X], torch.tensor([3.0, 1.0])
        )
        torch.testing.assert_close(self.bank.confidence, torch.tensor([3.0, 2.0]).sigmoid()[None])
        torch.testing.assert_close(self.bank.cached_feature[0, 0], torch.tensor([6.0, 7.0]))

    def test_temporal_anchors_follow_the_ego_motion_and_the_velocity(self) -> None:
        """
        Input: a cached frame at t = 10 whose anchors all have velocity 1 m/s along x, then a
            frame at t = 11 whose ego moved 2 m along x.
        Expected: a cached anchor at x = 3 travels 1 m with its velocity and loses 2 m of ego
            motion, so it sits at x = 2 in the new ego frame; the time interval is 1 s.
        Check: the temporal anchor center and the interval.
        """
        state = self.bank.get(self.timestamp, self.pose)
        anchor = state.anchor.clone()
        anchor[..., AnchorFieldIndex.VELOCITY_X] = 1.0
        self.bank.cache(
            state.instance_feature,
            anchor,
            _logits([0.0, 2.0, -1.0, 3.0]),
            self.timestamp,
            self.pose,
        )

        state = self.bank.get(
            torch.tensor([11.0], dtype=torch.float64), _translation(2.0).unsqueeze(0)
        )

        torch.testing.assert_close(state.temp_anchor[0, 0, AnchorFieldIndex.X], torch.tensor(2.0))
        torch.testing.assert_close(
            state.temp_anchor[0, 0, AnchorFieldIndex.VELOCITY_X], torch.tensor(1.0)
        )
        torch.testing.assert_close(state.time_interval, torch.tensor([1.0]))
        self.assertTrue(bool(self.bank.mask[0]))

    def test_cached_denoising_anchors_are_projected_too(self) -> None:
        """
        Input: a cached frame and a group of cached denoising anchors at x = 5, then a frame
            whose ego moved 2 m along x.
        Expected: the denoising anchors move like the temporal ones.
        Check: the returned denoising anchor sits at x = 3 with its group layout kept.
        """
        self._cache_frame([0.0, 2.0, -1.0, 3.0], timestamp=10.0, x_offset=0.0)
        dn_anchor = torch.zeros(1, 1, 2, ANCHOR_DIMS)
        dn_anchor[..., AnchorFieldIndex.X] = 5.0
        dn_anchor[..., AnchorFieldIndex.COS_YAW] = 1.0

        state = self.bank.get(
            torch.tensor([11.0], dtype=torch.float64),
            _translation(2.0).unsqueeze(0),
            temp_dn_anchor=dn_anchor,
        )

        self.assertEqual(state.temp_dn_anchor.shape, (1, 1, 2, ANCHOR_DIMS))
        torch.testing.assert_close(
            state.temp_dn_anchor[..., AnchorFieldIndex.X], torch.full((1, 1, 2), 3.0)
        )

    def test_update_merges_the_temporal_and_the_best_new_instances(self) -> None:
        """
        Input: a cached frame (anchors 3 and 1 kept), then the next frame whose instances have
            confidences 5, 4, 0, 1.
        Expected: the temporal instances lead and the two most confident new ones (0 and 1)
            follow.
        Check: the merged anchors sit at x = 3, 1, 0, 1 and the features follow.
        """
        self._cache_frame([0.0, 2.0, -1.0, 3.0], timestamp=10.0, x_offset=0.0)
        state = self.bank.get(torch.tensor([10.5], dtype=torch.float64), self.pose)

        feature, anchor = self.bank.update(
            state.instance_feature, state.anchor, _logits([5.0, 4.0, 0.0, 1.0])
        )

        torch.testing.assert_close(
            anchor[0, :, AnchorFieldIndex.X], torch.tensor([3.0, 1.0, 0.0, 1.0])
        )
        torch.testing.assert_close(feature[0, 2], torch.tensor([0.0, 1.0]))
        torch.testing.assert_close(feature[0, 0], torch.tensor([6.0, 7.0]))

    def test_update_passes_denoising_instances_through(self) -> None:
        """
        Input: a cached frame, then instances with two denoising ones appended.
        Expected: the learnable instances are merged and the denoising ones stay last.
        Check: the output keeps six instances and the last two are untouched.
        """
        self._cache_frame([0.0, 2.0, -1.0, 3.0], timestamp=10.0, x_offset=0.0)
        state = self.bank.get(torch.tensor([10.5], dtype=torch.float64), self.pose)
        dn_anchor = torch.full((1, 2, ANCHOR_DIMS), 9.0)
        dn_feature = torch.full((1, 2, EMBED_DIMS), 8.0)

        feature, anchor = self.bank.update(
            torch.cat([state.instance_feature, dn_feature], dim=1),
            torch.cat([state.anchor, dn_anchor], dim=1),
            torch.cat([_logits([5.0, 4.0, 0.0, 1.0]), torch.zeros(1, 2, 2)], dim=1),
        )

        self.assertEqual(anchor.shape, (1, 6, ANCHOR_DIMS))
        torch.testing.assert_close(anchor[:, 4:], dn_anchor)
        torch.testing.assert_close(feature[:, 4:], dn_feature)
        torch.testing.assert_close(anchor[0, :2, AnchorFieldIndex.X], torch.tensor([3.0, 1.0]))

    def test_a_long_gap_or_an_ended_stream_keeps_the_new_instances(self) -> None:
        """
        Input: a cached frame at t = 10, then a frame at t = 13 (gap above 2 s) and a frame
            at t = 10.5 whose stream ended.
        Expected: both lanes get the default interval and update leaves their instances.
        Check: the interval 0.5 and the unchanged anchors.
        """
        self._cache_frame([0.0, 2.0, -1.0, 3.0], timestamp=10.0, x_offset=0.0)

        state = self.bank.get(torch.tensor([13.0], dtype=torch.float64), self.pose)
        _, anchor = self.bank.update(
            state.instance_feature, state.anchor, _logits([5.0, 4.0, 0.0, 1.0])
        )
        torch.testing.assert_close(state.time_interval, torch.tensor([0.5]))
        torch.testing.assert_close(anchor, state.anchor)

        state = self.bank.get(
            torch.tensor([10.5], dtype=torch.float64), self.pose, prev_exists=torch.tensor([False])
        )
        _, anchor = self.bank.update(
            state.instance_feature, state.anchor, _logits([5.0, 4.0, 0.0, 1.0])
        )
        torch.testing.assert_close(state.time_interval, torch.tensor([0.5]))
        torch.testing.assert_close(anchor, state.anchor)

    def test_batch_size_change_resets_the_bank(self) -> None:
        """
        Input: a cached frame of one lane, then a frame of two lanes.
        Expected: the bank forgets the cache.
        Check: no temporal instances and a None cache.
        """
        self._cache_frame([0.0, 2.0, -1.0, 3.0], timestamp=10.0, x_offset=0.0)

        state = self.bank.get(
            torch.tensor([10.5, 10.5], dtype=torch.float64), self.pose.repeat(2, 1, 1)
        )

        self.assertIsNone(state.temp_anchor)
        self.assertIsNone(self.bank.cached_anchor)

    def test_confidence_decays_for_the_temporal_instances(self) -> None:
        """
        Input: a cached frame with confidences sigmoid(3) and sigmoid(2), then a frame whose
            first two instances (the temporal slots) score sigmoid(-5).
        Expected: the temporal slots keep 0.6 times their previous confidence, which beats
            the new score, so they stay the most confident.
        Check: the stored confidences equal the decayed ones.
        """
        self._cache_frame([0.0, 2.0, -1.0, 3.0], timestamp=10.0, x_offset=0.0)
        state = self.bank.get(torch.tensor([10.5], dtype=torch.float64), self.pose)
        self.bank.cache(
            state.instance_feature,
            state.anchor,
            _logits([-5.0, -5.0, -6.0, -6.0]),
            torch.tensor([10.5], dtype=torch.float64),
            self.pose,
        )

        # 0.6 * sigmoid(3) and 0.6 * sigmoid(2).
        expected = torch.tensor([0.5715445, 0.5284782])
        torch.testing.assert_close(self.bank.confidence[0], expected)

    def test_instance_ids_follow_the_cached_instances(self) -> None:
        """
        Input: a first frame whose instances 3 and 1 are cached, then a second frame.
        Expected: the first frame gets ids 0 to 3; in the second frame the two temporal
            slots carry the ids of instances 3 and 1 and the others get new ids 4 and 5.
        Check: the id tensors of both frames.
        """
        self._cache_frame([0.0, 2.0, -1.0, 3.0], timestamp=10.0, x_offset=0.0)
        first_ids = self.bank.get_instance_id(_logits([0.0, 2.0, -1.0, 3.0]))
        torch.testing.assert_close(first_ids, torch.tensor([[0, 1, 2, 3]]))

        state = self.bank.get(torch.tensor([10.5], dtype=torch.float64), self.pose)
        self.bank.update(state.instance_feature, state.anchor, _logits([5.0, 4.0, 0.0, 1.0]))
        self.bank.cache(
            state.instance_feature,
            state.anchor,
            _logits([5.0, 4.0, 0.0, 1.0]),
            torch.tensor([10.5], dtype=torch.float64),
            self.pose,
        )
        second_ids = self.bank.get_instance_id(_logits([5.0, 4.0, 0.0, 1.0]))

        torch.testing.assert_close(second_ids, torch.tensor([[3, 1, 4, 5]]))

    def test_instance_ids_below_the_threshold_stay_unset(self) -> None:
        """
        Input: a frame with confidences sigmoid(0) and sigmoid(2) on two instances and a
            threshold of 0.6.
        Expected: only the instance above the threshold gets an id.
        Check: ids -1, 0, -1, 1 for confidences 0, 2, -1, 3.
        """
        self._cache_frame([0.0, 2.0, -1.0, 3.0], timestamp=10.0, x_offset=0.0)

        ids = self.bank.get_instance_id(_logits([0.0, 2.0, -1.0, 3.0]), threshold=0.6)

        torch.testing.assert_close(ids, torch.tensor([[-1, 0, -1, 1]]))

    def test_init_weights_restores_the_initial_anchors(self) -> None:
        """
        Input: anchors moved after construction.
        Expected: init_weights puts the initial anchors back.
        Check: the anchor parameter equals the initial anchors.
        """
        with torch.no_grad():
            self.bank.anchor.add_(1.0)

        self.bank.init_weights()

        torch.testing.assert_close(self.bank.anchor.detach(), _anchors())

    def test_rejects_malformed_anchors(self) -> None:
        """
        Input: anchors with ten fields, and fewer anchors than asked.
        Expected: both are refused.
        Check: ValueError in both cases.
        """
        with self.assertRaisesRegex(ValueError, "11"):
            InstanceBank(num_anchor=2, embed_dims=EMBED_DIMS, anchor=torch.zeros(2, 10))
        with self.assertRaisesRegex(ValueError, "exceeds"):
            InstanceBank(num_anchor=8, embed_dims=EMBED_DIMS, anchor=_anchors())
        with self.assertRaisesRegex(ValueError, "num_temp_instances"):
            InstanceBank(
                num_anchor=4, embed_dims=EMBED_DIMS, anchor=_anchors(), num_temp_instances=4
            )
