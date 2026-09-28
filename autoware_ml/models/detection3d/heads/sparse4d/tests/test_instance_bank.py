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

"""Unit tests for InstanceBank."""

import math
import unittest

from jaxtyping import Float32
import torch

from autoware_ml.models.detection3d.heads.sparse4d.anchor_layout import AnchorIndex
from autoware_ml.models.detection3d.heads.sparse4d.instance_bank import (
    InstanceBank,
    Sparse4DFrameMetas,
    topk_instances,
)


class TestTopkInstances(unittest.TestCase):
    """Unit tests for the batched top-k gather."""

    def test_gathers_per_sample_in_descending_confidence(self) -> None:
        """Test that each sample keeps its own most confident rows, best first."""
        confidence = torch.tensor([[0.1, 0.9, 0.5], [0.7, 0.2, 0.8]])
        features = torch.arange(6, dtype=torch.float32).reshape(2, 3, 1)

        selected_confidence, outputs = topk_instances(confidence, 2, [features])

        expected_confidence = torch.tensor([[0.9, 0.5], [0.8, 0.7]])
        expected_features = torch.tensor([[[1.0], [2.0]], [[5.0], [3.0]]])
        self.assertTrue(torch.equal(selected_confidence, expected_confidence))
        self.assertEqual(len(outputs), 1)
        self.assertTrue(torch.equal(outputs[0], expected_features))


class TestInstanceBank(unittest.TestCase):
    """Unit tests for the InstanceBank."""

    def setUp(self) -> None:
        """Set up the common classes/inputs for the tests."""
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(0)

        self.batch_size = 2
        self.num_anchor = 6
        self.num_temp_instances = 2
        self.embed_dims = 8
        self.num_classes = 3
        self.anchor_dims = 11
        self.default_time_interval = 0.5
        self.max_time_interval = 2.0

        self.instance_bank = self._build_instance_bank()

    def _build_initial_anchors(self) -> Float32[torch.Tensor, "num_anchor anchor_dims"]:
        """Build unit boxes with zero yaw and zero velocity spread along x."""
        anchors = torch.zeros(self.num_anchor, self.anchor_dims)
        anchors[:, AnchorIndex.X] = torch.arange(self.num_anchor, dtype=torch.float32)
        anchors[:, AnchorIndex.COS_YAW] = 1.0
        return anchors

    def _build_instance_bank(self) -> InstanceBank:
        """Build the module under test on the shared device."""
        return InstanceBank(
            num_anchor=self.num_anchor,
            embed_dims=self.embed_dims,
            anchor=self._build_initial_anchors().tolist(),
            num_temp_instances=self.num_temp_instances,
            default_time_interval=self.default_time_interval,
            max_time_interval=self.max_time_interval,
        ).to(self.device)

    def _build_metas(self, timestamp: float, ego_x: float = 0.0) -> Sparse4DFrameMetas:
        """Build frame metadata with the ego translated along x by ``ego_x`` in the map."""
        ego2global = torch.eye(4, device=self.device)
        ego2global[0, 3] = ego_x
        return Sparse4DFrameMetas(
            timestamps=torch.full((self.batch_size,), timestamp, device=self.device),
            ego2globals=ego2global[None].tile(self.batch_size, 1, 1),
        )

    def _build_frame_outputs(
        self,
    ) -> tuple[
        Float32[torch.Tensor, "batch_size num_anchor embed_dims"],
        Float32[torch.Tensor, "batch_size num_anchor anchor_dims"],
        Float32[torch.Tensor, "batch_size num_anchor num_classes"],
    ]:
        """Build head outputs where anchor i has confidence increasing with i."""
        instance_feature = torch.randn(
            self.batch_size, self.num_anchor, self.embed_dims, device=self.device
        )
        anchor = self._build_initial_anchors().to(self.device)[None].tile(self.batch_size, 1, 1)
        confidence = torch.full(
            (self.batch_size, self.num_anchor, self.num_classes), -10.0, device=self.device
        )
        confidence[:, :, 0] = torch.arange(self.num_anchor, dtype=torch.float32, device=self.device)
        return instance_feature, anchor, confidence

    def test_get_without_history_returns_tiled_initials_and_default_interval(self) -> None:
        """Test that a fresh bank hands out the learned initials and no cache."""
        outputs = self.instance_bank.get(self.batch_size, self._build_metas(timestamp=10.0))

        self.assertEqual(
            outputs.instance_feature.shape, (self.batch_size, self.num_anchor, self.embed_dims)
        )
        self.assertEqual(outputs.anchor.shape, (self.batch_size, self.num_anchor, self.anchor_dims))
        self.assertIsNone(outputs.cached_feature)
        self.assertIsNone(outputs.cached_anchor)
        expected_interval = torch.full(
            (self.batch_size,), self.default_time_interval, device=self.device
        )
        self.assertTrue(torch.equal(outputs.time_interval, expected_interval))

    def test_cache_keeps_most_confident_instances(self) -> None:
        """Test that the cache holds the top instances in descending confidence order."""
        instance_feature, anchor, confidence = self._build_frame_outputs()

        self.instance_bank.cache(instance_feature, anchor, confidence, self._build_metas(10.0))

        cached_anchor = self.instance_bank.cached_anchor
        cached_feature = self.instance_bank.cached_feature
        self.assertIsNotNone(cached_anchor)
        self.assertIsNotNone(cached_feature)
        expected_x = torch.tensor([5.0, 4.0], device=self.device)
        self.assertTrue(torch.equal(cached_anchor[0, :, AnchorIndex.X], expected_x))
        self.assertTrue(torch.equal(cached_feature[0, 0], instance_feature[0, 5]))
        self.assertFalse(cached_feature.requires_grad)

    def test_get_projects_cached_anchors_into_the_current_ego_frame(self) -> None:
        """Test that a forward ego motion shifts the cached anchors backwards."""
        instance_feature, anchor, confidence = self._build_frame_outputs()
        self.instance_bank.cache(instance_feature, anchor, confidence, self._build_metas(10.0))

        outputs = self.instance_bank.get(
            self.batch_size, self._build_metas(timestamp=10.5, ego_x=1.0)
        )

        self.assertIsNotNone(outputs.cached_anchor)
        expected_x = torch.tensor([4.0, 3.0], device=self.device)
        self.assertTrue(torch.allclose(outputs.cached_anchor[0, :, AnchorIndex.X], expected_x))
        expected_interval = torch.full((self.batch_size,), 0.5, device=self.device)
        self.assertTrue(torch.allclose(outputs.time_interval, expected_interval))
        self.assertTrue(bool(self.instance_bank.mask.all()))

    def test_get_masks_stale_history_and_falls_back_to_default_interval(self) -> None:
        """Test that a gap above max_time_interval disables the history for that sample."""
        instance_feature, anchor, confidence = self._build_frame_outputs()
        self.instance_bank.cache(instance_feature, anchor, confidence, self._build_metas(10.0))

        outputs = self.instance_bank.get(self.batch_size, self._build_metas(timestamp=15.0))

        self.assertFalse(bool(self.instance_bank.mask.any()))
        expected_interval = torch.full(
            (self.batch_size,), self.default_time_interval, device=self.device
        )
        self.assertTrue(torch.equal(outputs.time_interval, expected_interval))

    def test_get_drops_history_on_batch_size_change(self) -> None:
        """Test that a different batch size starts a new stream without a cache."""
        instance_feature, anchor, confidence = self._build_frame_outputs()
        self.instance_bank.cache(instance_feature, anchor, confidence, self._build_metas(10.0))

        outputs = self.instance_bank.get(self.batch_size + 1)

        self.assertIsNone(outputs.cached_feature)
        self.assertIsNone(outputs.cached_anchor)
        self.assertIsNone(self.instance_bank.confidence)

    def test_update_prepends_cached_instances_for_fresh_samples(self) -> None:
        """Test that cached instances lead and the best current ones fill the rest."""
        instance_feature, anchor, confidence = self._build_frame_outputs()
        self.instance_bank.cache(instance_feature, anchor, confidence, self._build_metas(10.0))
        self.instance_bank.get(self.batch_size, self._build_metas(timestamp=10.5))

        updated_feature, updated_anchor = self.instance_bank.update(
            instance_feature, anchor, confidence
        )

        num_current = self.num_anchor - self.num_temp_instances
        self.assertTrue(torch.equal(updated_feature[:, :2], self.instance_bank.cached_feature))
        self.assertTrue(torch.equal(updated_anchor[:, :2], self.instance_bank.cached_anchor))
        expected_current_x = torch.tensor([5.0, 4.0, 3.0, 2.0], device=self.device)
        self.assertEqual(updated_anchor[:, 2:].shape[1], num_current)
        self.assertTrue(torch.equal(updated_anchor[0, 2:, AnchorIndex.X], expected_current_x))

    def test_update_without_history_is_identity(self) -> None:
        """Test that update leaves the inputs alone when nothing is cached."""
        instance_feature, anchor, confidence = self._build_frame_outputs()

        updated_feature, updated_anchor = self.instance_bank.update(
            instance_feature, anchor, confidence
        )

        self.assertIs(updated_feature, instance_feature)
        self.assertIs(updated_anchor, anchor)

    def test_cache_decays_previous_confidence_of_temporal_instances(self) -> None:
        """Test that a carried-over instance keeps a decayed score if it beats the new one."""
        instance_feature, anchor, confidence = self._build_frame_outputs()
        self.instance_bank.cache(instance_feature, anchor, confidence, self._build_metas(10.0))
        previous_confidence = self.instance_bank.confidence.clone()

        # The leading slots hold the temporal instances. Give them a very weak new score so the
        # decayed previous score wins.
        weak_confidence = torch.full_like(confidence, -10.0)
        self.instance_bank.cache(instance_feature, anchor, weak_confidence, self._build_metas(10.5))

        expected = previous_confidence * self.instance_bank.confidence_decay
        self.assertTrue(torch.allclose(self.instance_bank.confidence, expected))

    def test_reset_clears_history(self) -> None:
        """Test that reset drops the cache so the next get sees a fresh stream."""
        instance_feature, anchor, confidence = self._build_frame_outputs()
        self.instance_bank.cache(instance_feature, anchor, confidence, self._build_metas(10.0))

        self.instance_bank.reset()
        outputs = self.instance_bank.get(self.batch_size, self._build_metas(timestamp=10.5))

        self.assertIsNone(outputs.cached_feature)
        self.assertIsNone(self.instance_bank.previous_metas)

    def test_init_weights_restores_initial_anchors(self) -> None:
        """Test that the anchors start from the provided values."""
        expected = self._build_initial_anchors().to(self.device)
        self.assertTrue(torch.equal(self.instance_bank.anchor.detach(), expected))
        self.assertFalse(
            math.isclose(float(self.instance_bank.instance_feature.detach().abs().sum()), 0.0)
        )


if __name__ == "__main__":
    unittest.main()
