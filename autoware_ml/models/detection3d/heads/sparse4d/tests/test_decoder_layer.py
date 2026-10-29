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

"""Unit tests for the Sparse4D decoder layer."""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from autoware_ml.models.detection3d.heads.sparse4d.blocks import (
    AsymmetricFFN,
    DeformableFeatureAggregation,
)
from autoware_ml.models.detection3d.heads.sparse4d.box3d import (
    ANCHOR_DIMS,
    SparseBox3DKeyPointsGenerator,
    SparseBox3DRefinementModule,
)
from autoware_ml.models.detection3d.heads.sparse4d.decoder_layer import Sparse4DDecoderLayer

EMBED_DIMS = 16
NUM_CAMS = 2
NUM_ANCHOR = 3


def _blocks() -> tuple[AsymmetricFFN, DeformableFeatureAggregation, SparseBox3DRefinementModule]:
    """Blocks of a tiny layer whose aggregation keeps the width."""
    kps_generator = SparseBox3DKeyPointsGenerator(
        embed_dims=EMBED_DIMS, fix_scale=[[0.0, 0.0, 0.0]]
    )
    return (
        AsymmetricFFN(embed_dims=EMBED_DIMS, feedforward_channels=32),
        DeformableFeatureAggregation(
            kps_generator, embed_dims=EMBED_DIMS, num_groups=2, num_levels=1, num_cams=NUM_CAMS
        ),
        SparseBox3DRefinementModule(embed_dims=EMBED_DIMS, num_cls=2, with_quality_estimation=True),
    )


def _inputs(batch_size: int = 1) -> dict:
    """Instances and one feature level of two cameras."""
    torch.manual_seed(0)
    projection = torch.tensor(
        [
            [10.0, 0.0, 12.0, 0.0],
            [0.0, 10.0, 8.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    anchor = torch.zeros(batch_size, NUM_ANCHOR, ANCHOR_DIMS)
    anchor[..., 2] = 5.0
    anchor[..., 7] = 1.0
    return {
        "instance_feature": torch.randn(batch_size, NUM_ANCHOR, EMBED_DIMS),
        "anchor": anchor,
        "anchor_embed": torch.randn(batch_size, NUM_ANCHOR, EMBED_DIMS),
        "feature_maps": [torch.randn(batch_size, NUM_CAMS, EMBED_DIMS, 4, 6)],
        "projection_mat": projection.expand(batch_size, NUM_CAMS, 4, 4).clone(),
        "image_wh": torch.tensor([24.0, 16.0]).expand(batch_size, NUM_CAMS, 2).clone(),
        "time_interval": torch.full((batch_size,), 0.5),
    }


def _run(
    layer: Sparse4DDecoderLayer, inputs: dict, temp_feature=None, temp_embed=None, attn_mask=None
):
    """Run a layer on the inputs."""
    return layer(
        inputs["instance_feature"],
        inputs["anchor"],
        inputs["anchor_embed"],
        inputs["feature_maps"],
        inputs["projection_mat"],
        inputs["image_wh"],
        inputs["time_interval"],
        temp_instance_feature=temp_feature,
        temp_anchor_embed=temp_embed,
        attn_mask=attn_mask,
    )


class TestSparse4DDecoderLayer(unittest.TestCase):
    """One refinement of the anchors."""

    def setUp(self) -> None:
        """Set up the blocks and the inputs."""
        self.ffn, self.deformable, self.refine = _blocks()
        self.inputs = _inputs()

    def _layer(
        self, with_self_attention: bool, with_temporal_attention: bool
    ) -> Sparse4DDecoderLayer:
        return Sparse4DDecoderLayer(
            EMBED_DIMS,
            2,
            self.ffn,
            self.deformable,
            self.refine,
            with_self_attention=with_self_attention,
            with_temporal_attention=with_temporal_attention,
            attn_drop=0.0,
            decouple_attn=True,
        )

    def test_first_layer_has_no_attention(self) -> None:
        """
        Input: a layer without attentions.
        Expected: the attention blocks and the norm after them are absent, and the layer
            still refines.
        Check: the None blocks and the output shapes.
        """
        layer = self._layer(with_self_attention=False, with_temporal_attention=False)

        feature, anchor, cls, quality = _run(layer, self.inputs)

        self.assertIsNone(layer.gnn)
        self.assertIsNone(layer.temp_gnn)
        self.assertIsNone(layer.norm_attn)
        self.assertEqual(feature.shape, (1, NUM_ANCHOR, EMBED_DIMS))
        self.assertEqual(anchor.shape, (1, NUM_ANCHOR, ANCHOR_DIMS))
        self.assertEqual(cls.shape, (1, NUM_ANCHOR, 2))
        self.assertEqual(quality.shape, (1, NUM_ANCHOR, 2))

    def test_temporal_attention_reads_the_temporal_instances(self) -> None:
        """
        Input: a full layer with and without temporal instances.
        Expected: the temporal instances change the output, and without them the temporal
            attention attends among the instances under the mask.
        Check: different outputs, and a run under a full mask.
        """
        layer = self._layer(with_self_attention=True, with_temporal_attention=True).eval()
        temp_feature = torch.randn(1, 2, EMBED_DIMS)
        temp_embed = torch.randn(1, 2, EMBED_DIMS)

        with_temporal, _, _, _ = _run(layer, self.inputs, temp_feature, temp_embed)
        without_temporal, _, _, _ = _run(layer, self.inputs)

        self.assertFalse(torch.allclose(with_temporal, without_temporal))
        mask = torch.eye(NUM_ANCHOR, dtype=torch.bool).logical_not()
        masked, _, _, _ = _run(layer, self.inputs, attn_mask=mask)
        self.assertEqual(masked.shape, (1, NUM_ANCHOR, EMBED_DIMS))

    def test_holds_the_given_blocks(self) -> None:
        """
        Input: a layer built from the blocks.
        Expected: the layer owns exactly those modules; cloning is the head's job.
        Check: identity of the three blocks.
        """
        layer = self._layer(with_self_attention=True, with_temporal_attention=False)

        self.assertIs(layer.ffn, self.ffn)
        self.assertIs(layer.deformable, self.deformable)
        self.assertIs(layer.refine, self.refine)

    def test_init_weights_keeps_the_refinement_bias(self) -> None:
        """
        Input: a layer whose aggregation weights were set to one, then init_weights.
        Expected: Xavier does not touch the refinement, and the aggregation's own init puts
            its sampling weights back to zero.
        Check: the class bias of the refinement and the zero sampling weights.
        """
        layer = self._layer(with_self_attention=True, with_temporal_attention=True)
        bias = layer.refine.cls_layers[-1].bias.detach().clone()
        nn.init.ones_(layer.deformable.weights_fc.weight)

        layer.init_weights()

        torch.testing.assert_close(layer.refine.cls_layers[-1].bias, bias)
        weights = layer.deformable.weights_fc.weight
        torch.testing.assert_close(weights, torch.zeros_like(weights))
