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

"""Unit tests for the Sparse4D anchor modules."""

from __future__ import annotations

import math
import unittest

import torch
import torch.nn as nn

from autoware_ml.models.detection3d.heads.sparse4d.box3d import (
    ANCHOR_DIMS,
    AnchorFieldIndex,
    QualityIndex,
    SparseBox3DEncoder,
    SparseBox3DKeyPointsGenerator,
    SparseBox3DRefinementModule,
)
from autoware_ml.types.geometry import Box3DFieldIndex


def _zero_linear(linear: nn.Module) -> None:
    """Zero the weight and the bias of a linear layer."""
    nn.init.constant_(linear.weight, 0.0)
    nn.init.constant_(linear.bias, 0.0)


class TestAnchorFieldIndex(unittest.TestCase):
    """The anchor layout agrees with the box layout on the shared fields."""

    def test_center_and_sizes_follow_the_box_layout(self) -> None:
        """
        Input: the anchor and the box field indices.
        Expected: the center and the (length, width, height) sit at the same indices in both,
            and the anchor has 11 fields with the velocity last.
        Check: the six shared indices, the field count and the velocity index.
        """
        for field in ("X", "Y", "Z", "LENGTH", "WIDTH", "HEIGHT"):
            self.assertEqual(AnchorFieldIndex[field], Box3DFieldIndex[field])
        self.assertEqual(ANCHOR_DIMS, 11)
        self.assertEqual(AnchorFieldIndex.VELOCITY_X, 8)
        self.assertEqual(len(QualityIndex), 2)


class TestSparseBox3DEncoder(unittest.TestCase):
    """The encoder embeds the anchor fields and combines the embeddings."""

    def setUp(self) -> None:
        """Draw a batch of anchors."""
        torch.manual_seed(0)
        self.anchors = torch.randn(2, 5, ANCHOR_DIMS)

    def test_add_mode_returns_the_embedding_width(self) -> None:
        """
        Input: anchors and an ``add`` encoder of width 32 with an output MLP.
        Expected: the four embeddings are summed into one 32-wide embedding per anchor.
        Check: the output shape.
        """
        encoder = SparseBox3DEncoder(embed_dims=32, vel_dims=3, mode="add")

        output = encoder(self.anchors)

        self.assertEqual(output.shape, (2, 5, 32))

    def test_cat_mode_concatenates_the_four_embeddings(self) -> None:
        """
        Input: anchors and a ``cat`` encoder with widths [16, 8, 8, 4] and no output MLP, the
            decoupled-attention recipe.
        Expected: the embedding is the concatenation, 36 wide, and no output MLP exists.
        Check: the output shape and the absent output MLP.
        """
        encoder = SparseBox3DEncoder(
            embed_dims=[16, 8, 8, 4], vel_dims=3, mode="cat", output_fc=False
        )

        output = encoder(self.anchors)

        self.assertEqual(output.shape, (2, 5, 36))
        self.assertIsNone(encoder.output_fc)

    def test_without_velocity_the_velocity_fields_are_ignored(self) -> None:
        """
        Input: a ``cat`` encoder with ``vel_dims=0``, no output MLP, and anchors differing
            only in velocity.
        Expected: no velocity MLP exists and the two anchors get the same 32-wide embedding.
        Check: the absent MLP, the output shape and the equal outputs.
        """
        encoder = SparseBox3DEncoder(
            embed_dims=[16, 8, 8, 4], vel_dims=0, mode="cat", output_fc=False
        )
        moved = self.anchors.clone()
        moved[..., AnchorFieldIndex.VELOCITY_X :] = 7.0

        self.assertIsNone(encoder.vel_fc)
        self.assertEqual(encoder(self.anchors).shape, (2, 5, 32))
        torch.testing.assert_close(encoder(moved), encoder(self.anchors))

    def test_anchors_are_embedded_independently(self) -> None:
        """
        Input: the anchors and the same anchors with the first two swapped.
        Expected: every anchor is embedded on its own, so the embeddings swap too.
        Check: the swapped embeddings equal the embeddings of the swapped anchors.
        """
        encoder = SparseBox3DEncoder(embed_dims=32)
        swapped = self.anchors[:, [1, 0, 2, 3, 4]]

        torch.testing.assert_close(encoder(swapped), encoder(self.anchors)[:, [1, 0, 2, 3, 4]])

    def test_rejects_an_output_mlp_that_does_not_fit_the_embedding(self) -> None:
        """
        Input: a ``cat`` encoder with widths [16, 8, 8, 4] and an output MLP, whose input would
            have to be 36 wide, and an ``add`` encoder whose widths differ.
        Expected: the encoder refuses both, the widths cannot be combined.
        Check: a ValueError is raised twice.
        """
        with self.assertRaises(ValueError):
            SparseBox3DEncoder(embed_dims=[16, 8, 8, 4], mode="cat", output_fc=True)
        with self.assertRaises(ValueError):
            SparseBox3DEncoder(embed_dims=[16, 8, 8, 4], mode="add", output_fc=False)

    def test_a_five_width_list_names_the_output_width(self) -> None:
        """
        Input: a ``cat`` encoder with widths [16, 8, 8, 4, 36] and an output MLP.
        Expected: the fifth width is the output MLP's, so the concatenation feeds it.
        Check: the output shape is (2, 5, 36).
        """
        encoder = SparseBox3DEncoder(embed_dims=[16, 8, 8, 4, 36], mode="cat", output_fc=True)

        self.assertEqual(encoder(self.anchors).shape, (2, 5, 36))

    def test_rejects_an_unknown_mode(self) -> None:
        """
        Input: a mode other than ``add`` and ``cat``.
        Expected: the encoder refuses.
        Check: a ValueError is raised.
        """
        with self.assertRaises(ValueError):
            SparseBox3DEncoder(embed_dims=32, mode="mul")


class TestSparseBox3DKeyPointsGenerator(unittest.TestCase):
    """The generator spreads key points over the anchors and moves anchors between frames."""

    def test_fixed_points_are_scaled_by_the_size_and_rotated_by_the_yaw(self) -> None:
        """
        Input: one anchor at (1, 2, 3) with sizes (2, 4, 6) and a 90 degree yaw, and the
            offsets (0, 0, 0), (0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5).
        Expected: the offsets become (1, 0, 0), (0, 2, 0), (0, 0, 3) in the box frame, rotate
            to (0, 1, 0), (-2, 0, 0), (0, 0, 3), and shift by the center.
        Check: the four key points equal (1, 2, 3), (1, 3, 3), (-1, 2, 3), (1, 2, 6).
        """
        generator = SparseBox3DKeyPointsGenerator(
            fix_scale=[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]]
        )
        anchor = torch.tensor(
            [
                [
                    [
                        1.0,
                        2.0,
                        3.0,
                        math.log(2.0),
                        math.log(4.0),
                        math.log(6.0),
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ]
                ]
            ]
        )

        key_points = generator(anchor)

        expected = torch.tensor(
            [[[[1.0, 2.0, 3.0], [1.0, 3.0, 3.0], [-1.0, 2.0, 3.0], [1.0, 2.0, 6.0]]]]
        )
        self.assertEqual(generator.num_pts, 4)
        torch.testing.assert_close(key_points, expected)

    def test_learnable_points_start_at_the_center(self) -> None:
        """
        Input: a generator with two learnable points whose predictor is zeroed, one anchor at
            (1, 2, 3), and an instance feature.
        Expected: a zero prediction gives ``sigmoid(0) - 0.5 = 0`` offsets, so the learnable
            points sit at the center after the fixed one.
        Check: three key points, all equal to the center.
        """
        generator = SparseBox3DKeyPointsGenerator(embed_dims=8, num_learnable_pts=2)
        _zero_linear(generator.learnable_fc)
        anchor = torch.tensor([[[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]])

        key_points = generator(anchor, torch.randn(1, 1, 8))

        self.assertEqual(generator.num_pts, 3)
        torch.testing.assert_close(key_points, torch.tensor([[1.0, 2.0, 3.0]]).expand(1, 1, 3, 3))

    def test_without_instance_features_only_the_fixed_points_are_returned(self) -> None:
        """
        Input: a generator with learnable points called without instance features.
        Expected: only the fixed key point is returned.
        Check: the key points shape is (1, 1, 1, 3).
        """
        generator = SparseBox3DKeyPointsGenerator(embed_dims=8, num_learnable_pts=2)
        anchor = torch.zeros(1, 1, ANCHOR_DIMS)

        self.assertEqual(generator(anchor).shape, (1, 1, 1, 3))

    def test_anchor_projection_moves_the_center_heading_and_velocity(self) -> None:
        """
        Input: one anchor at (1, 0, 0) heading along x with velocity (1, 0, 0), a transform
            rotating 90 degrees about z and lifting by 1, and a time interval of 0.5.
        Expected: the center travels back to (0.5, 0, 0), rotates to (0, 0.5, 0) and lifts to
            (0, 0.5, 1); the heading turns to +y, stored as sin 1 and cos 0; the velocity turns
            to (0, 1, 0); the sizes stay.
        Check: the projected anchor equals [0, 0.5, 1, 0, 0, 0, 1, 0, 0, 1, 0].
        """
        anchor = torch.tensor([[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]]])
        src2dst = torch.tensor(
            [
                [
                    [0.0, -1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ]
        )

        projected = SparseBox3DKeyPointsGenerator.anchor_projection(
            anchor, src2dst, torch.tensor([0.5])
        )

        expected = torch.tensor([[[0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]]])
        torch.testing.assert_close(projected, expected)

    def test_anchor_projection_with_the_identity_keeps_the_anchor(self) -> None:
        """
        Input: random anchors, the identity transform and no time interval.
        Expected: nothing moves.
        Check: the projected anchors equal the input.
        """
        torch.manual_seed(0)
        anchor = torch.randn(2, 4, ANCHOR_DIMS)

        projected = SparseBox3DKeyPointsGenerator.anchor_projection(
            anchor, torch.eye(4).expand(2, 4, 4)
        )

        torch.testing.assert_close(projected, anchor)

    def test_distance_is_measured_in_the_ground_plane(self) -> None:
        """
        Input: anchors at (3, 4, 10) and (0, 0, 5).
        Expected: the height is ignored, so the distances are 5 and 0.
        Check: the distances.
        """
        anchor = torch.zeros(1, 2, ANCHOR_DIMS)
        anchor[0, 0, :3] = torch.tensor([3.0, 4.0, 10.0])
        anchor[0, 1, 2] = 5.0

        torch.testing.assert_close(
            SparseBox3DKeyPointsGenerator.distance(anchor), torch.tensor([[5.0, 0.0]])
        )


class TestSparseBox3DRefinementModule(unittest.TestCase):
    """The refinement is a residual on the anchor with class scores and quality on the side."""

    def setUp(self) -> None:
        """Draw features and anchors, and build a module whose regression output is zero."""
        torch.manual_seed(0)
        self.instance_feature = torch.randn(2, 3, 16)
        self.anchor_embed = torch.randn(2, 3, 16)
        self.anchor = torch.randn(2, 3, ANCHOR_DIMS)

    def _zeroed_module(
        self, refine_yaw: bool, with_quality_estimation: bool = False
    ) -> SparseBox3DRefinementModule:
        """Build a module whose last regression layer is zeroed, so it predicts no residual."""
        module = SparseBox3DRefinementModule(
            embed_dims=16,
            num_cls=4,
            refine_yaw=refine_yaw,
            with_quality_estimation=with_quality_estimation,
        )
        _zero_linear(module.layers[-2])
        return module

    def test_a_zero_residual_returns_the_refined_fields_of_the_anchor(self) -> None:
        """
        Input: a module without yaw refinement whose regression output is zero, and a time
            interval of 0.5.
        Expected: the center and the sizes are copied from the anchor, the yaw fields are the
            zero residual, and the velocity is the zero translation over 0.5 plus the anchor
            velocity.
        Check: the three groups of fields of the output.
        """
        module = self._zeroed_module(refine_yaw=False)

        output, _, _ = module(
            self.instance_feature, self.anchor, self.anchor_embed, torch.tensor([0.5, 0.5])
        )

        torch.testing.assert_close(output[..., :6], self.anchor[..., :6])
        torch.testing.assert_close(output[..., 6:8], torch.zeros(2, 3, 2))
        torch.testing.assert_close(output[..., 8:], self.anchor[..., 8:])

    def test_yaw_refinement_copies_the_yaw_too(self) -> None:
        """
        Input: a module with yaw refinement whose regression output is zero.
        Expected: the sine and cosine of the yaw are copied from the anchor as well.
        Check: the first eight fields of the output equal the anchor's.
        """
        module = self._zeroed_module(refine_yaw=True)

        output, _, _ = module(self.instance_feature, self.anchor, self.anchor_embed)

        torch.testing.assert_close(output[..., :8], self.anchor[..., :8])

    def test_the_velocity_is_a_translation_over_the_time_interval(self) -> None:
        """
        Input: a zeroed module with the bias of its velocity outputs set to 1, zero anchors and
            time intervals 0.5 and 2.
        Expected: the velocity is the translation divided by the time of its sample: 2 for
            the first sample and 0.5 for the second.
        Check: the velocity fields of both samples.
        """
        module = self._zeroed_module(refine_yaw=False)
        with torch.no_grad():
            module.layers[-2].bias[AnchorFieldIndex.VELOCITY_X :] = 1.0

        output, _, _ = module(
            self.instance_feature,
            torch.zeros(2, 3, ANCHOR_DIMS),
            self.anchor_embed,
            torch.tensor([0.5, 2.0]),
        )

        torch.testing.assert_close(output[0, :, 8:], torch.full((3, 3), 2.0))
        torch.testing.assert_close(output[1, :, 8:], torch.full((3, 3), 0.5))

    def test_normalize_yaw_puts_the_heading_on_the_unit_circle(self) -> None:
        """
        Input: a module with ``normalize_yaw`` and random weights.
        Expected: the sine and cosine of every refined anchor have unit norm.
        Check: the norm of the yaw fields.
        """
        module = SparseBox3DRefinementModule(embed_dims=16, num_cls=4, normalize_yaw=True)

        output, _, _ = module(self.instance_feature, self.anchor, self.anchor_embed)

        torch.testing.assert_close(output[..., 6:8].norm(dim=-1), torch.ones(2, 3))

    def test_class_scores_start_at_the_foreground_prior(self) -> None:
        """
        Input: a freshly built module with four classes.
        Expected: the class bias is the focal-loss prior ``-log(99)`` and the scores have one
            entry per class.
        Check: the bias of the last classification layer and the scores shape.
        """
        module = SparseBox3DRefinementModule(embed_dims=16, num_cls=4)

        _, cls, _ = module(self.instance_feature, self.anchor, self.anchor_embed)

        torch.testing.assert_close(module.cls_layers[-1].bias, torch.full((4,), -math.log(99.0)))
        self.assertEqual(cls.shape, (2, 3, 4))

    def test_quality_is_predicted_only_when_asked(self) -> None:
        """
        Input: a module with quality estimation, called with and without ``return_cls``.
        Expected: the quality has a centerness and a yawness per anchor when asked, and both
            the scores and the quality are ``None`` otherwise.
        Check: the quality shape and the two ``None`` results.
        """
        module = SparseBox3DRefinementModule(embed_dims=16, num_cls=4, with_quality_estimation=True)

        _, cls, quality = module(self.instance_feature, self.anchor, self.anchor_embed)
        _, no_cls, no_quality = module(
            self.instance_feature, self.anchor, self.anchor_embed, return_cls=False
        )

        self.assertEqual(quality.shape, (2, 3, 2))
        self.assertIsNotNone(cls)
        self.assertIsNone(no_cls)
        self.assertIsNone(no_quality)

    def test_without_velocity_outputs_the_anchor_is_eight_wide(self) -> None:
        """
        Input: a module with ``output_dim=8``.
        Expected: no velocity is predicted, the output stops after the yaw.
        Check: the output shape.
        """
        module = SparseBox3DRefinementModule(embed_dims=16, output_dim=8, num_cls=4)

        output, _, _ = module(self.instance_feature, self.anchor, self.anchor_embed)

        self.assertEqual(output.shape, (2, 3, 8))

    def test_rejects_class_scores_without_a_class_branch(self) -> None:
        """
        Input: a module built without a classification branch, asked for class scores.
        Expected: the module refuses.
        Check: a ValueError is raised.
        """
        module = SparseBox3DRefinementModule(embed_dims=16, num_cls=4, with_cls_branch=False)

        with self.assertRaises(ValueError):
            module(self.instance_feature, self.anchor, self.anchor_embed)


if __name__ == "__main__":
    unittest.main()
