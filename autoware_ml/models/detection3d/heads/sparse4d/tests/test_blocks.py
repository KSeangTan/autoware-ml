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

"""Unit tests for the Sparse4D decoder blocks."""

from __future__ import annotations

import math
import unittest

import torch
import torch.nn as nn

from autoware_ml.models.detection3d.heads.sparse4d.blocks import (
    AsymmetricFFN,
    DeformableFeatureAggregation,
    DenseDepthNet,
)
from autoware_ml.models.detection3d.heads.sparse4d.box3d import (
    ANCHOR_DIMS,
    SparseBox3DKeyPointsGenerator,
)
from autoware_ml.models.detection3d.heads.sparse4d.layers import Scale, linear_relu_ln
from autoware_ml.ops.deformable_aggregation import flatten_feature_maps


def _zero_linear(linear: nn.Module) -> None:
    """Zero the weight and the bias of a linear layer."""
    nn.init.constant_(linear.weight, 0.0)
    nn.init.constant_(linear.bias, 0.0)


def _orthographic_projection(batch_size: int, num_cams: int) -> torch.Tensor:
    """Projection matrices mapping a point (x, y, z) to the pixel (x, y) at unit depth."""
    projection = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]
    )
    return projection.expand(batch_size, num_cams, 4, 4).clone()


def _anchors_inside_the_image(batch_size: int, num_anchors: int) -> torch.Tensor:
    """Anchors whose key points, half a meter wide, stay inside a 10 x 8 image."""
    anchors = torch.zeros(batch_size, num_anchors, ANCHOR_DIMS)
    anchors[..., 0] = torch.rand(batch_size, num_anchors).mul(6.0).add(2.0)
    anchors[..., 1] = torch.rand(batch_size, num_anchors).mul(4.0).add(2.0)
    anchors[..., 3:6] = math.log(0.5)
    yaw = torch.rand(batch_size, num_anchors).mul(6.0)
    anchors[..., 6] = yaw.sin()
    anchors[..., 7] = yaw.cos()
    return anchors


class TestLayers(unittest.TestCase):
    """The shared layers of the blocks."""

    def test_linear_relu_ln_stacks_the_layers_in_order(self) -> None:
        """
        Input: ``embed_dims=8``, one inner loop, two outer loops and a 12-wide input.
        Expected: two blocks of Linear, ReLU and LayerNorm, the first Linear taking 12 inputs
            and the second 8.
        Check: the layer types and the input widths.
        """
        layers = linear_relu_ln(8, in_loops=1, out_loops=2, input_dims=12)

        self.assertEqual(
            [type(layer) for layer in layers],
            [nn.Linear, nn.ReLU, nn.LayerNorm, nn.Linear, nn.ReLU, nn.LayerNorm],
        )
        self.assertEqual(layers[0].in_features, 12)
        self.assertEqual(layers[3].in_features, 8)

    def test_scale_multiplies_every_channel_by_its_own_factor(self) -> None:
        """
        Input: a scale of (1, 2, 3) and a row of ones.
        Expected: the channels are multiplied by their own factor.
        Check: the output equals (1, 2, 3).
        """
        scale = Scale([1.0, 2.0, 3.0])

        torch.testing.assert_close(scale(torch.ones(3)), torch.tensor([1.0, 2.0, 3.0]))


class TestAsymmetricFFN(unittest.TestCase):
    """The feed-forward block with a projected residual."""

    def setUp(self) -> None:
        """Draw an input batch."""
        torch.manual_seed(0)
        self.x = torch.randn(2, 3, 16)

    def test_a_zeroed_block_returns_the_identity(self) -> None:
        """
        Input: a block whose input and output are 16 wide, with its last linear zeroed.
        Expected: the residual is the plain identity, so the output equals the input.
        Check: the identity projection type and the output.
        """
        ffn = AsymmetricFFN(embed_dims=16, feedforward_channels=32)
        _zero_linear(ffn.layers[1])

        self.assertIsInstance(ffn.identity_fc, nn.Identity)
        torch.testing.assert_close(ffn(self.x), self.x)

    def test_the_pre_norm_feeds_the_residual_too(self) -> None:
        """
        Input: a block with a pre-norm and its last linear zeroed.
        Expected: the residual is the normalized input.
        Check: the output equals the LayerNorm of the input.
        """
        ffn = AsymmetricFFN(pre_norm=True, embed_dims=16, feedforward_channels=32)
        _zero_linear(ffn.layers[1])

        torch.testing.assert_close(ffn(self.x), ffn.pre_norm(self.x))

    def test_a_wider_input_is_projected_onto_the_output(self) -> None:
        """
        Input: a block taking 32 channels and returning 16, the decoder recipe.
        Expected: the residual goes through a 32 to 16 linear projection and the output is
            16 wide.
        Check: the projection widths and the output shape.
        """
        ffn = AsymmetricFFN(in_channels=32, embed_dims=16, feedforward_channels=64)

        output = ffn(torch.randn(2, 3, 32))

        self.assertEqual((ffn.identity_fc.in_features, ffn.identity_fc.out_features), (32, 16))
        self.assertEqual(output.shape, (2, 3, 16))

    def test_without_the_identity_a_zeroed_block_returns_zero(self) -> None:
        """
        Input: a block without the residual and its last linear zeroed.
        Expected: nothing is added back, so the output is zero.
        Check: the output.
        """
        ffn = AsymmetricFFN(embed_dims=16, feedforward_channels=32, add_identity=False)
        _zero_linear(ffn.layers[1])

        torch.testing.assert_close(ffn(self.x), torch.zeros_like(self.x))

    def test_rejects_a_single_linear_layer(self) -> None:
        """
        Input: ``num_fcs=1``.
        Expected: the block refuses, it needs a hidden layer.
        Check: a ValueError is raised.
        """
        with self.assertRaises(ValueError):
            AsymmetricFFN(embed_dims=16, num_fcs=1)


class TestDeformableFeatureAggregation(unittest.TestCase):
    """The aggregation projects the key points, weighs the samples and fuses them."""

    def setUp(self) -> None:
        """Build a small aggregation over two cameras and two levels with 8 channels."""
        torch.manual_seed(0)
        self.batch_size = 2
        self.num_cams = 2
        self.num_levels = 2
        self.embed_dims = 8
        self.num_groups = 2
        self.num_anchors = 3
        self.generator = SparseBox3DKeyPointsGenerator(
            embed_dims=self.embed_dims,
            num_learnable_pts=1,
            fix_scale=[[0.0, 0.0, 0.0], [0.45, 0.0, 0.0], [0.0, 0.45, 0.0]],
        )
        self.aggregation = DeformableFeatureAggregation(
            self.generator,
            embed_dims=self.embed_dims,
            num_groups=self.num_groups,
            num_levels=self.num_levels,
            num_cams=self.num_cams,
        ).eval()
        self.instance_feature = torch.randn(self.batch_size, self.num_anchors, self.embed_dims)
        self.anchor_embed = torch.randn(self.batch_size, self.num_anchors, self.embed_dims)
        self.anchors = _anchors_inside_the_image(self.batch_size, self.num_anchors)
        self.projection_mat = _orthographic_projection(self.batch_size, self.num_cams)
        self.image_wh = torch.tensor([10.0, 8.0]).expand(self.batch_size, self.num_cams, 2)
        self.feature_maps = [
            torch.randn(self.batch_size, self.num_cams, self.embed_dims, 6, 7),
            torch.randn(self.batch_size, self.num_cams, self.embed_dims, 3, 4),
        ]

    def test_project_points_applies_the_projection_and_normalizes_by_the_image_size(self) -> None:
        """
        Input: a pinhole projection with focal 100 and principal point (50, 50), a point at
            (1, 2, 10) and one at (1, 2, -10), and an image of 100 x 200 pixels.
        Expected: the first point lands on pixel (600 / 10, 700 / 10) = (60, 70), normalized
            to (0.6, 0.35); the second is behind the camera and lands far outside the image.
        Check: the normalized coordinates of the first point and the sign of the second's.
        """
        projection = torch.tensor(
            [
                [
                    [
                        [100.0, 0.0, 50.0, 0.0],
                        [0.0, 100.0, 50.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ]
            ]
        )
        key_points = torch.tensor([[[[1.0, 2.0, 10.0], [1.0, 2.0, -10.0]]]])
        image_wh = torch.tensor([[[100.0, 200.0]]])

        points_2d = DeformableFeatureAggregation.project_points(key_points, projection, image_wh)

        self.assertEqual(points_2d.shape, (1, 1, 1, 2, 2))
        torch.testing.assert_close(points_2d[0, 0, 0, 0], torch.tensor([0.6, 0.35]))
        self.assertTrue((points_2d[0, 0, 0, 1] < 0).all())

    def test_weights_start_uniform_and_sum_to_one_per_group(self) -> None:
        """
        Input: the freshly built aggregation, whose weight predictor is zeroed at init.
        Expected: a softmax of zeros, every (camera, level, point) sample weighs
            1 / (cameras x levels x points) in every group.
        Check: the weights shape and values.
        """
        weights = self.aggregation._get_weights(
            self.instance_feature, self.anchor_embed, self.projection_mat
        )

        num_samples = math.prod([self.num_cams, self.num_levels, self.generator.num_pts])
        self.assertEqual(
            weights.shape,
            (
                self.batch_size,
                self.num_anchors,
                self.num_cams,
                self.num_levels,
                self.generator.num_pts,
                self.num_groups,
            ),
        )
        torch.testing.assert_close(weights, torch.full_like(weights, 1.0 / num_samples))

    def test_attention_dropout_drops_whole_samples_and_rescales_the_rest(self) -> None:
        """
        Input: the aggregation in training mode with ``attn_drop=0.5`` and random weight
            predictor.
        Expected: a dropped (camera, point) sample is zero on every level and group, and the
            kept ones are the evaluation weights divided by 0.5.
        Check: every weight is zero or twice its evaluation value, and the zeros cover all
            levels and groups of their sample.
        """
        nn.init.normal_(self.aggregation.weights_fc.weight)
        eval_weights = self.aggregation._get_weights(
            self.instance_feature, self.anchor_embed, self.projection_mat
        )
        self.aggregation.attn_drop = 0.5
        self.aggregation.train()

        weights = self.aggregation._get_weights(
            self.instance_feature, self.anchor_embed, self.projection_mat
        )

        dropped = weights == 0
        torch.testing.assert_close(
            weights, torch.where(dropped, torch.zeros_like(weights), eval_weights.mul(2.0))
        )
        per_sample = dropped.any(dim=3).any(dim=-1)
        torch.testing.assert_close(dropped.all(dim=3).all(dim=-1), per_sample)
        self.assertTrue(per_sample.any())
        self.assertFalse(per_sample.all())

    def test_the_camera_embedding_predicts_the_weights_per_camera(self) -> None:
        """
        Input: an aggregation with ``use_camera_embed``.
        Expected: the predictor outputs (levels x points x groups) weights per camera, and
            the weights still sum to one over the samples of every group.
        Check: the predictor width and the sums.
        """
        aggregation = DeformableFeatureAggregation(
            self.generator,
            embed_dims=self.embed_dims,
            num_groups=self.num_groups,
            num_levels=self.num_levels,
            num_cams=self.num_cams,
            use_camera_embed=True,
        ).eval()
        nn.init.normal_(aggregation.weights_fc.weight)

        weights = aggregation._get_weights(
            self.instance_feature, self.anchor_embed, self.projection_mat
        )

        self.assertEqual(
            aggregation.weights_fc.out_features,
            math.prod([self.num_levels, self.generator.num_pts, self.num_groups]),
        )
        torch.testing.assert_close(
            weights.sum(dim=(2, 3, 4)),
            torch.ones(self.batch_size, self.num_anchors, self.num_groups),
        )

    def test_the_grid_sample_path_returns_the_embedding_width(self) -> None:
        """
        Input: the aggregation with the pyramid as a list of levels.
        Expected: the fused features are added to the instance features, 8 wide.
        Check: the output shape.
        """
        output = self.aggregation(
            self.instance_feature,
            self.anchors,
            self.anchor_embed,
            self.feature_maps,
            self.projection_mat,
            self.image_wh,
        )

        self.assertEqual(output.shape, (self.batch_size, self.num_anchors, self.embed_dims))

    def test_the_cat_residual_doubles_the_width(self) -> None:
        """
        Input: an aggregation with ``residual_mode="cat"`` and a zeroed output projection.
        Expected: the output is the fused features (zero) followed by the instance features.
        Check: the output shape and its two halves.
        """
        aggregation = DeformableFeatureAggregation(
            self.generator,
            embed_dims=self.embed_dims,
            num_groups=self.num_groups,
            num_levels=self.num_levels,
            num_cams=self.num_cams,
            residual_mode="cat",
        ).eval()
        _zero_linear(aggregation.output_proj)

        output = aggregation(
            self.instance_feature,
            self.anchors,
            self.anchor_embed,
            self.feature_maps,
            self.projection_mat,
            self.image_wh,
        )

        self.assertEqual(output.shape, (self.batch_size, self.num_anchors, 16))
        torch.testing.assert_close(output[..., :8], torch.zeros_like(self.instance_feature))
        torch.testing.assert_close(output[..., 8:], self.instance_feature)

    def test_points_outside_every_image_sample_nothing(self) -> None:
        """
        Input: anchors 100 meters outside the image and a zeroed output projection bias.
        Expected: every sample is zero, so the fused feature is zero and the output is the
            instance feature alone.
        Check: the output equals the instance features.
        """
        far_anchors = self.anchors.clone()
        far_anchors[..., 0] = 100.0

        output = self.aggregation(
            self.instance_feature,
            far_anchors,
            self.anchor_embed,
            self.feature_maps,
            self.projection_mat,
            self.image_wh,
        )

        torch.testing.assert_close(output, self.instance_feature)

    def test_rejects_inconsistent_construction(self) -> None:
        """
        Input: groups that do not divide the channels, a generator of another width, and an
            unknown residual mode.
        Expected: the aggregation refuses each.
        Check: a ValueError is raised three times.
        """
        with self.assertRaises(ValueError):
            DeformableFeatureAggregation(self.generator, embed_dims=8, num_groups=3)
        with self.assertRaises(ValueError):
            DeformableFeatureAggregation(SparseBox3DKeyPointsGenerator(embed_dims=4), embed_dims=8)
        with self.assertRaises(ValueError):
            DeformableFeatureAggregation(self.generator, embed_dims=8, residual_mode="mul")


@unittest.skipUnless(torch.cuda.is_available(), "the deformable aggregation kernel requires CUDA")
class TestDeformableFeatureAggregationKernelPath(unittest.TestCase):
    """The CUDA kernel path and the grid_sample path give the same features."""

    def test_the_flattened_pyramid_takes_the_kernel_and_matches_grid_sample(self) -> None:
        """
        Input: an aggregation with random weight predictor and learnable points, anchors
            inside a two-camera, two-level pyramid, given once as a list of levels and once
            flattened for the kernel.
        Expected: both paths sample the same key points with the same weights.
        Check: the two outputs agree to 1e-4.
        """
        torch.manual_seed(0)
        device = torch.device("cuda:0")
        generator = SparseBox3DKeyPointsGenerator(
            embed_dims=8,
            num_learnable_pts=2,
            fix_scale=[[0.0, 0.0, 0.0], [0.45, 0.0, 0.0], [0.0, 0.0, -0.45]],
        )
        aggregation = (
            DeformableFeatureAggregation(
                generator, embed_dims=8, num_groups=2, num_levels=2, num_cams=2
            )
            .to(device)
            .eval()
        )
        nn.init.normal_(aggregation.weights_fc.weight)
        instance_feature = torch.randn(2, 5, 8, device=device)
        anchor_embed = torch.randn(2, 5, 8, device=device)
        anchors = _anchors_inside_the_image(2, 5).to(device)
        projection_mat = _orthographic_projection(2, 2).to(device)
        image_wh = torch.tensor([10.0, 8.0], device=device).expand(2, 2, 2)
        feature_maps = [
            torch.randn(2, 2, 8, 6, 7, device=device),
            torch.randn(2, 2, 8, 3, 4, device=device),
        ]

        grid_sample_output = aggregation(
            instance_feature, anchors, anchor_embed, feature_maps, projection_mat, image_wh
        )
        kernel_output = aggregation(
            instance_feature,
            anchors,
            anchor_embed,
            flatten_feature_maps(feature_maps),
            projection_mat,
            image_wh,
        )

        torch.testing.assert_close(kernel_output, grid_sample_output, rtol=1e-4, atol=1e-4)


class TestDenseDepthNet(unittest.TestCase):
    """The auxiliary depth head and its loss."""

    def setUp(self) -> None:
        """Build a two-level head whose convolutions are zeroed, so it predicts depth 1."""
        torch.manual_seed(0)
        self.head = DenseDepthNet(
            embed_dims=4, num_depth_layers=2, equal_focal=100.0, max_depth=60.0, loss_weight=0.2
        )
        for layer in self.head.depth_layers:
            nn.init.constant_(layer.weight, 0.0)
            nn.init.constant_(layer.bias, 0.0)
        self.feature_maps = [
            torch.randn(1, 2, 4, 2, 3),
            torch.randn(1, 2, 4, 1, 2),
            torch.randn(1, 2, 4, 1, 1),
        ]

    def test_predicts_one_map_per_depth_layer_at_the_level_resolution(self) -> None:
        """
        Input: three pyramid levels and a two-layer head without focal lengths.
        Expected: two depth maps, one per predicted level, of the level resolution per
            camera, all equal to ``exp(0) = 1``.
        Check: the number of maps, their shapes and values.
        """
        depths = self.head(self.feature_maps)

        self.assertEqual(len(depths), 2)
        self.assertEqual(depths[0].shape, (1, 2, 2, 3))
        self.assertEqual(depths[1].shape, (1, 2, 1, 2))
        torch.testing.assert_close(depths[0], torch.ones(1, 2, 2, 3))

    def test_the_focal_length_scales_the_depth(self) -> None:
        """
        Input: focal lengths of 200 and 50 for the two cameras, with ``equal_focal=100``.
        Expected: the depth of the first camera doubles and that of the second halves.
        Check: the depth maps of both cameras.
        """
        depths = self.head(self.feature_maps, torch.tensor([[200.0, 50.0]]))

        torch.testing.assert_close(depths[0][0, 0], torch.full((2, 3), 2.0))
        torch.testing.assert_close(depths[0][0, 1], torch.full((2, 3), 0.5))

    def test_the_loss_averages_the_error_over_the_supervised_pixels_and_levels(self) -> None:
        """
        Input: predicted depth 1 everywhere, a first level with ground truth [2, 0, 5] on one
            camera and nothing on the other, and an empty second level.
        Expected: only the two positive pixels count: (1 + 4) / (2 pixels x 2 levels) x 0.2 =
            0.25, and the empty level adds 0 / max(1, 0) = 0.
        Check: the loss equals 0.25.
        """
        depths = self.head(self.feature_maps)
        gt_level0 = torch.zeros(1, 2, 2, 3)
        gt_level0[0, 0, 0] = torch.tensor([2.0, 0.0, 5.0])
        gt_level1 = torch.zeros(1, 2, 1, 2)

        loss = self.head.loss(depths, [gt_level0, gt_level1])

        torch.testing.assert_close(loss, torch.tensor(0.25))

    def test_the_loss_clips_the_prediction_at_the_maximum_depth(self) -> None:
        """
        Input: a prediction of 100 everywhere (bias ``log(100)``) and a ground truth of 70.
        Expected: the prediction is clipped to 60 before the error, so the error is 10 on
            one pixel over two levels: 10 / 2 x 0.2 = 1.
        Check: the loss equals 1.
        """
        for layer in self.head.depth_layers:
            nn.init.constant_(layer.bias, math.log(100.0))
        depths = self.head(self.feature_maps)
        gt_level0 = torch.zeros(1, 2, 2, 3)
        gt_level0[0, 1, 1, 2] = 70.0
        gt_level1 = torch.zeros(1, 2, 1, 2)

        loss = self.head.loss(depths, [gt_level0, gt_level1])

        torch.testing.assert_close(loss, torch.tensor(1.0))


if __name__ == "__main__":
    unittest.main()
