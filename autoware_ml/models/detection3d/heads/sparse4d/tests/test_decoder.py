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

"""Unit tests for the Sparse4D decoder."""

from __future__ import annotations

import math
import unittest

import torch

from autoware_ml.dataclasses.models.detection3d.predictions import Detection3DSamplePredictions
from autoware_ml.models.detection3d.heads.sparse4d.box3d import ANCHOR_DIMS
from autoware_ml.models.detection3d.heads.sparse4d.decoder import SparseBox3DDecoder, decode_box


def _anchors() -> torch.Tensor:
    """Two anchors at x = 1 and x = 2 of unit size, heading along x, no velocity."""
    anchors = torch.zeros(1, 2, ANCHOR_DIMS)
    anchors[0, :, 0] = torch.tensor([1.0, 2.0])
    anchors[0, :, 7] = 1.0
    return anchors


class TestDecodeBox(unittest.TestCase):
    """The anchor to box decoding."""

    def test_decodes_sizes_yaw_and_planar_velocity(self) -> None:
        """
        Input: an anchor at (1, 2, 3) with log sizes (0, log 2, log 3), heading (sin 1,
            cos 0) and velocity (4, 5, 6).
        Expected: sizes (1, 2, 3), yaw pi/2 and the planar velocity (4, 5).
        Check: the nine box parameters.
        """
        anchor = torch.tensor(
            [[1.0, 2.0, 3.0, 0.0, math.log(2.0), math.log(3.0), 1.0, 0.0, 4.0, 5.0, 6.0]]
        )

        box = decode_box(anchor)

        expected = torch.tensor([[1.0, 2.0, 3.0, 1.0, 2.0, 3.0, math.pi / 2, 4.0, 5.0]])
        torch.testing.assert_close(box, expected)


class TestSparseBox3DDecoder(unittest.TestCase):
    """The selection of the boxes."""

    def setUp(self) -> None:
        """Set up two anchors with two classes: scores 0.9/0.2 on the first, 0.6/0.7 on the second."""
        probabilities = torch.tensor([[[0.9, 0.2], [0.6, 0.7]]])
        self.cls_scores = torch.logit(probabilities)
        self.box_preds = _anchors()

    def test_ranks_every_anchor_class_pair(self) -> None:
        """
        Input: the four pairs and ``num_output=3``.
        Expected: the three best pairs in order: anchor 0 class 0, anchor 1 class 1, anchor 1
            class 0.
        Check: the scores, the labels and the box centers of the one sample.
        """
        predictions = SparseBox3DDecoder(num_output=3).decode(self.cls_scores, self.box_preds)

        self.assertEqual(len(predictions), 1)
        prediction = predictions[0]
        self.assertIsInstance(prediction, Detection3DSamplePredictions)
        torch.testing.assert_close(prediction.scores_3d, torch.tensor([0.9, 0.7, 0.6]))
        torch.testing.assert_close(prediction.labels_3d, torch.tensor([0, 1, 0]))
        torch.testing.assert_close(prediction.bboxes_3d[:, 0], torch.tensor([1.0, 2.0, 2.0]))
        self.assertEqual(prediction.bboxes_3d.shape, (3, 9))
        self.assertIsNone(prediction.instance_ids_3d)

    def test_num_output_is_clamped_to_the_pairs(self) -> None:
        """
        Input: ``num_output=300`` on four pairs.
        Expected: all four pairs come out.
        Check: four boxes.
        """
        predictions = SparseBox3DDecoder(num_output=300).decode(self.cls_scores, self.box_preds)

        self.assertEqual(predictions[0].scores_3d.shape, (4,))

    def test_score_threshold_drops_weak_pairs(self) -> None:
        """
        Input: a threshold of 0.65.
        Expected: only the 0.9 and 0.7 pairs remain.
        Check: the scores and the labels.
        """
        decoder = SparseBox3DDecoder(num_output=4, score_threshold=0.65)

        prediction = decoder.decode(self.cls_scores, self.box_preds)[0]

        torch.testing.assert_close(prediction.scores_3d, torch.tensor([0.9, 0.7]))
        torch.testing.assert_close(prediction.labels_3d, torch.tensor([0, 1]))

    def test_centerness_scales_the_scores_and_reorders(self) -> None:
        """
        Input: a quality whose centerness logit is very negative on anchor 0 and very
            positive on anchor 1, with ``num_output=2``.
        Expected: the top two pairs are picked by class score first (0.9 and 0.7), then the
            0.9 of anchor 0 collapses to 0 and the 0.7 of anchor 1 leads.
        Check: the scores (0.7, 0) and the labels (1, 0).
        """
        quality = torch.tensor([[[-50.0, 0.0], [50.0, 0.0]]])

        prediction = SparseBox3DDecoder(num_output=2).decode(
            self.cls_scores, self.box_preds, quality=quality
        )[0]

        torch.testing.assert_close(prediction.scores_3d, torch.tensor([0.7, 0.0]))
        torch.testing.assert_close(prediction.labels_3d, torch.tensor([1, 0]))
        torch.testing.assert_close(prediction.bboxes_3d[:, 0], torch.tensor([2.0, 1.0]))

    def test_instance_ids_give_one_box_per_anchor(self) -> None:
        """
        Input: instance ids 10 and 11 for the two anchors.
        Expected: every anchor yields one box of its best class with its id, best anchor
            first.
        Check: the scores (0.9, 0.7), the labels (0, 1) and the ids (10, 11).
        """
        ids = torch.tensor([[10, 11]])

        prediction = SparseBox3DDecoder(num_output=5).decode(
            self.cls_scores, self.box_preds, instance_ids=ids
        )[0]

        torch.testing.assert_close(prediction.scores_3d, torch.tensor([0.9, 0.7]))
        torch.testing.assert_close(prediction.labels_3d, torch.tensor([0, 1]))
        torch.testing.assert_close(prediction.instance_ids_3d, torch.tensor([10, 11]))

    def test_threshold_and_quality_apply_with_instance_ids(self) -> None:
        """
        Input: instance ids, the anchor 0 centerness collapsed and a threshold of 0.8.
        Expected: the threshold reads the class score, so anchor 0 (0.9) stays and anchor 1
            (0.7) goes, even though the centerness puts anchor 0 at 0.
        Check: one box with id 10 and score 0.
        """
        ids = torch.tensor([[10, 11]])
        quality = torch.tensor([[[-50.0, 0.0], [50.0, 0.0]]])
        decoder = SparseBox3DDecoder(num_output=5, score_threshold=0.8)

        prediction = decoder.decode(
            self.cls_scores, self.box_preds, quality=quality, instance_ids=ids
        )[0]

        torch.testing.assert_close(prediction.instance_ids_3d, torch.tensor([10]))
        torch.testing.assert_close(prediction.scores_3d, torch.tensor([0.0]))

    def test_decodes_every_sample_of_a_batch(self) -> None:
        """
        Input: a batch of two samples.
        Expected: one prediction per sample.
        Check: two predictions with the same scores.
        """
        predictions = SparseBox3DDecoder(num_output=1).decode(
            self.cls_scores.repeat(2, 1, 1), self.box_preds.repeat(2, 1, 1)
        )

        self.assertEqual(len(predictions), 2)
        torch.testing.assert_close(predictions[1].scores_3d, torch.tensor([0.9]))
