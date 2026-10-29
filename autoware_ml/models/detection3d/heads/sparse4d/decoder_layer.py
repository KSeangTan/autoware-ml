# Copyright (c) 2024 Horizon Robotics. MIT License.
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
#
# Adapted from the ``operation_order`` loop of projects/mmdet3d_plugin/models/sparse4d_head.py
# of https://github.com/HorizonRobotics/Sparse4D (MIT License). The operations of one layer
# of the v3 recipe are one module here, instead of a flat list of operation names; the head
# clones its prototype blocks for every layer.

"""One decoder layer of the Sparse4D head."""

from __future__ import annotations

from collections.abc import Sequence

from jaxtyping import Bool, Float
import torch
import torch.nn as nn

from autoware_ml.dataclasses.models.detection2d.feature_maps import FlattenedFeatureMaps
from autoware_ml.models.detection3d.heads.sparse4d.attention import GraphAttention
from autoware_ml.models.detection3d.heads.sparse4d.blocks import (
    AsymmetricFFN,
    DeformableFeatureAggregation,
)
from autoware_ml.models.detection3d.heads.sparse4d.box3d import SparseBox3DRefinementModule


class Sparse4DDecoderLayer(nn.Module):
    """One refinement of the anchors, in the order of the v3 recipe.

    A temporal graph attention over the instances of the previous frame, a graph attention
    among the instances followed by a norm, the deformable aggregation of the image
    features, the feed-forward network followed by a norm, and the refinement that moves
    the anchors and scores them. The first layer of a frame has no attention, since every
    instance starts from the same feature, and the single-frame layers have no temporal
    attention.
    """

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        ffn: AsymmetricFFN,
        deformable_model: DeformableFeatureAggregation,
        refine_layer: SparseBox3DRefinementModule,
        with_self_attention: bool = True,
        with_temporal_attention: bool = True,
        attn_drop: float = 0.1,
        decouple_attn: bool = True,
        value_proj: nn.Linear | None = None,
        output_proj: nn.Linear | None = None,
    ) -> None:
        """Initialize the layer.

        Args:
            embed_dims: Width of the instance features.
            num_heads: Attention heads of the graph attentions.
            ffn: The feed-forward network of the layer.
            deformable_model: The deformable aggregation of the layer.
            refine_layer: The refinement of the layer.
            with_self_attention: Whether the graph attention among the instances exists.
            with_temporal_attention: Whether the temporal graph attention exists.
            attn_drop: Dropout of the attention weights.
            decouple_attn: Whether the graph attentions attend over the instance feature and
                its anchor embedding concatenated, see :class:`GraphAttention`.
            value_proj: Value projection shared by the decoupled attentions, see
                :class:`GraphAttention`.
            output_proj: Output projection shared by the decoupled attentions.
        """
        super().__init__()
        self.temp_gnn: GraphAttention | None = None
        self.gnn: GraphAttention | None = None
        self.norm_attn: nn.LayerNorm | None = None
        if with_temporal_attention:
            self.temp_gnn = GraphAttention(
                embed_dims,
                num_heads,
                attn_drop,
                decouple_attn=decouple_attn,
                value_proj=value_proj,
                output_proj=output_proj,
            )
        if with_self_attention:
            self.gnn = GraphAttention(
                embed_dims,
                num_heads,
                attn_drop,
                decouple_attn=decouple_attn,
                value_proj=value_proj,
                output_proj=output_proj,
            )
            self.norm_attn = nn.LayerNorm(embed_dims)
        self.deformable = deformable_model
        self.ffn = ffn
        self.norm_ffn = nn.LayerNorm(embed_dims)
        self.refine = refine_layer

    def init_weights(self) -> None:
        """Draw the parameters: Xavier on every block but the refinement, then the blocks' own."""
        for name, block in self.named_children():
            if name == "refine":
                continue
            for parameter in block.parameters():
                if parameter.dim() > 1:
                    nn.init.xavier_uniform_(parameter)
        for block in self.children():
            if hasattr(block, "init_weights"):
                block.init_weights()

    def forward(
        self,
        instance_feature: Float[torch.Tensor, "batch_size num_instances embed_dims"],
        anchor: Float[torch.Tensor, "batch_size num_instances anchor_dims"],
        anchor_embed: Float[torch.Tensor, "batch_size num_instances embed_dims"],
        feature_maps: FlattenedFeatureMaps
        | Sequence[Float[torch.Tensor, "batch_size num_cams embed_dims height width"]],
        projection_mat: Float[torch.Tensor, "batch_size num_cams 4 4"],
        image_wh: Float[torch.Tensor, "batch_size num_cams 2"],
        time_interval: Float[torch.Tensor, " batch_size"],
        temp_instance_feature: Float[torch.Tensor, "batch_size num_temp embed_dims"] | None = None,
        temp_anchor_embed: Float[torch.Tensor, "batch_size num_temp embed_dims"] | None = None,
        attn_mask: Bool[torch.Tensor, "num_instances num_instances"] | None = None,
    ) -> tuple[
        Float[torch.Tensor, "batch_size num_instances embed_dims"],
        Float[torch.Tensor, "batch_size num_instances anchor_dims"],
        Float[torch.Tensor, "batch_size num_instances num_classes"],
        Float[torch.Tensor, "batch_size num_instances 2"] | None,
    ]:
        """Refine the anchors once.

        Args:
            instance_feature: Features of the instances, the noised ones last.
            anchor: Their anchors.
            anchor_embed: The embedding of the anchors.
            feature_maps: Image feature pyramid of every camera.
            projection_mat: Transform from the ego frame to the pixels of every camera.
            image_wh: Width and height of every camera image.
            time_interval: Time since the previous frame of every lane.
            temp_instance_feature: Features of the temporal instances, ``None`` on the first
                frame of a stream, when the temporal attention attends among the instances.
            temp_anchor_embed: The embedding of the temporal anchors.
            attn_mask: True where an instance must not attend to another, which keeps the
                noised groups apart.

        Returns:
            The refined features, the refined anchors, their class logits and their quality
            (``None`` without quality estimation).
        """
        if self.temp_gnn is not None:
            instance_feature = self.temp_gnn(
                instance_feature,
                temp_instance_feature,
                temp_instance_feature,
                query_pos=anchor_embed,
                key_pos=temp_anchor_embed,
                attn_mask=attn_mask if temp_instance_feature is None else None,
            )
        if self.gnn is not None:
            instance_feature = self.gnn(
                instance_feature,
                value=instance_feature,
                query_pos=anchor_embed,
                attn_mask=attn_mask,
            )
            instance_feature = self.norm_attn(instance_feature)
        instance_feature = self.deformable(
            instance_feature, anchor, anchor_embed, feature_maps, projection_mat, image_wh
        )
        instance_feature = self.norm_ffn(self.ffn(instance_feature))
        anchor, cls, quality = self.refine(
            instance_feature, anchor, anchor_embed, time_interval=time_interval, return_cls=True
        )
        return instance_feature, anchor, cls, quality
