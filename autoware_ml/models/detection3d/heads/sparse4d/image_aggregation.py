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

"""Image block of the Sparse4D decoder, the deformable aggregation over the camera pyramids.

The block reads the image features of a frame and nothing of its lidar features, which the
lidar block of the decoder reads. The two blocks share the instance features alone.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from autoware_ml.models.detection3d.heads.sparse4d.sampling_weights import keypoint_dropout
from autoware_ml.ops.deformable_aggregation.bev_deformable_aggregation import (
    bev_deformable_aggregation,
)
from autoware_ml.ops.deformable_aggregation.image_deformable_aggregation import (
    ImageFeatures,
    project_key_points,
)

# Entries of the 3x4 projection of a camera, the input of the camera embedding
PROJECTION_DIM = 12


class ImageDeformableAggregation(nn.Module):
    """Sample the pyramid of every camera at the projected keypoints of the anchors.

    The deformable aggregation of Sparse4D (arXiv 2211.10581): the keypoints of an anchor
    project into every camera and read every stride of its pyramid bilinearly, and weights
    predicted from the instance fuse the samples over the cameras, the strides and the
    keypoints of every feature group. An embedding of the projection of a camera tells the
    weights of that camera which one it is.

    The samples are aggregated by :func:`bev_deformable_aggregation`, the function of the lidar
    BEV block, with the map of every camera and stride as one of its levels.

    The weights normalize over the samples the cameras see. A keypoint behind a camera or
    outside its image reads zeros, and most anchors lie in one or two of the cameras, so a
    softmax over every sample would leave most of the weight on zeros. An anchor no camera
    sees gets no image features.
    """

    def __init__(
        self,
        *,
        embed_dims: int,
        image_channels: int,
        num_groups: int,
        num_levels: int,
        kps_generator: nn.Module,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        camera_embed: bool = True,
    ) -> None:
        """Initialize the image block.

        Args:
            embed_dims: Instance feature dimension.
            image_channels: Feature dimension of the image pyramid.
            num_groups: Number of feature groups sharing one weight.
            num_levels: Number of strides of the pyramid.
            kps_generator: Keypoint generator of the block, its own and not the one of the
                lidar block.
            proj_drop: Dropout on the output projection.
            attn_drop: Probability of dropping whole samples in training.
            camera_embed: Whether the weights of a camera read an embedding of its projection.

        Raises:
            ValueError: If the image channels do not split into the groups, or without a level.
        """
        super().__init__()
        if image_channels % num_groups != 0:
            raise ValueError(
                f"image_channels {image_channels} not divisible by {num_groups} groups."
            )
        if num_levels < 1:
            raise ValueError(f"The image block samples at least one stride, got {num_levels}.")
        self.num_groups = num_groups
        self.num_levels = num_levels
        self.kps_generator = kps_generator
        self.num_pts = kps_generator.num_pts
        self.attn_drop = attn_drop
        self.camera_encoder = (
            nn.Sequential(
                nn.Linear(PROJECTION_DIM, embed_dims),
                nn.ReLU(inplace=True),
                nn.LayerNorm(embed_dims),
                nn.Linear(embed_dims, embed_dims),
                nn.ReLU(inplace=True),
                nn.LayerNorm(embed_dims),
            )
            if camera_embed
            else None
        )
        self.weights_fc = nn.Linear(embed_dims, num_groups * num_levels * self.num_pts)
        self.output_proj = nn.Linear(image_channels, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop)
        nn.init.zeros_(self.weights_fc.weight)
        nn.init.zeros_(self.weights_fc.bias)

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
        image_features: ImageFeatures,
    ) -> torch.Tensor:
        """Aggregate the image features of the cameras into the instance features.

        Args:
            instance_feature: Instance features ``(B, N, C)``.
            anchor: Anchor states ``(B, N, 10)`` in the lidar frame.
            anchor_embed: Anchor embeddings ``(B, N, C)``.
            image_features: Pyramid and projection of the cameras of the frame.

        Returns:
            Updated instance features ``(B, N, C)``.

        Raises:
            ValueError: If the pyramid does not have the strides of the block.
        """
        levels = image_features.levels
        if len(levels) != self.num_levels:
            raise ValueError(
                f"The image block samples {self.num_levels} strides, got {len(levels)}."
            )
        key_points = self.kps_generator(anchor, instance_feature)
        _, visible = project_key_points(
            key_points, image_features.lidar2image, image_features.image_size
        )
        weights = self._weights(instance_feature + anchor_embed, image_features, visible)
        # The aggregation of the lidar BEV block, over the map of every camera and stride
        features = bev_deformable_aggregation(
            key_points, weights.flatten(2, 4), image_features.camera_levels()
        )
        return instance_feature + self.proj_drop(self.output_proj(features))

    def _weights(
        self, query: torch.Tensor, image_features: ImageFeatures, visible: torch.Tensor
    ) -> torch.Tensor:
        """Weights of every sample ``(B, N, V, L, P, G)``, normalized over the seen samples.

        Args:
            query: Instance features with the anchor embedding ``(B, N, C)``.
            image_features: Pyramid and projection of the cameras of the frame.
            visible: Whether every camera sees every keypoint ``(B, V, N, P)``.

        Returns:
            The weights, zero on the samples no camera sees.
        """
        batch_size, num_views, num_anchor, num_points = visible.shape
        query = query[:, :, None]
        if self.camera_encoder is not None:
            projection = image_features.normalized_projection().flatten(-2)
            query = query + self.camera_encoder(projection)[:, None].to(query.dtype)
        else:
            query = query.expand(-1, -1, num_views, -1)
        logits = self.weights_fc(query).reshape(
            batch_size, num_anchor, num_views, self.num_levels, num_points, self.num_groups
        )
        # (B, V, N, P) to (B, N, V, 1, P, 1), a keypoint is seen in every stride of a camera
        seen = visible.permute(0, 2, 1, 3)[:, :, :, None, :, None]
        logits = logits.masked_fill(~seen, torch.finfo(logits.dtype).min)
        weights = logits.flatten(2, 4).softmax(dim=2) * seen.expand_as(logits).flatten(2, 4)
        weights = keypoint_dropout(weights, self.attn_drop, self.training)
        return weights.reshape(logits.shape)
