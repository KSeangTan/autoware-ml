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

"""Lidar block of the Sparse4D decoder, the deformable aggregation over the lidar features.

The block reads the lidar features of a frame, the sparse voxel stages, the BEV maps or both,
and nothing of its image features, which the image block of the decoder reads. The dense
attention over the BEV maps, the other lidar block, is :class:`BEVCrossAttention` of the head.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Protocol

import torch
import torch.nn as nn

from autoware_ml.utils.point_cloud.batching import offset_to_batch

from autoware_ml.models.detection3d.heads.sparse4d.sampling_weights import (
    group_softmax as _group_softmax,
    keypoint_dropout,
)
from autoware_ml.ops.deformable_aggregation.bev_deformable_aggregation import (
    BEVLevel,
    bev_deformable_aggregation,
)
from autoware_ml.ops.deformable_aggregation.sparse_deformable_aggregation import (
    EMPTY_ROW,
    SparseLevel,
    sparse_deformable_aggregation,
)

# Features, voxel coordinates and batch offsets of one sparse encoder stage
SparseStage = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
# The lidar features a head samples: the sparse stages, or the dense BEV map of a BEV only head
LidarFeatures = Sequence[SparseStage] | torch.Tensor
# Below this occupied weight the aggregated features fade to zero instead of amplifying a sliver
MIN_OCCUPANCY = 0.05


class KeyPointsGenerator(Protocol):
    """What the deformable aggregation needs from a keypoint generator."""

    num_pts: int

    def __call__(self, anchor: torch.Tensor, instance_feature: torch.Tensor) -> torch.Tensor:
        """Keypoints ``(B, N, num_pts, 3)`` of anchors ``(B, N, D)``."""
        ...


class SparseDeformableAggregation(nn.Module):
    """Sample sparse voxel features at anchor keypoints with learned weights.

    Keypoints interpolate trilinearly over every sampled stage, the weights normalize over the
    stages and the keypoints of every feature group. Keypoints in empty space read zeros, so the
    weighted sum is divided by the weight that landed on occupied voxels. A BEV branch samples
    dense BEV maps at the same keypoints and adds to the voxel features, or replaces them. A
    subclass joins the features of its frame context, see :meth:`join_context`.
    """

    def __init__(
        self,
        *,
        embed_dims: int,
        num_groups: int,
        num_levels: int,
        kps_generator: KeyPointsGenerator,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        bev_aggregation: BEVDeformableAggregation | None = None,
        sample_sparse: bool = True,
    ) -> None:
        """Initialize the sparse deformable aggregation branch.

        Args:
            embed_dims: Instance and voxel feature dimension.
            num_groups: Number of feature groups sharing one weight.
            num_levels: Number of sampled encoder stages.
            kps_generator: Shared keypoint generator.
            proj_drop: Dropout on the output projection.
            attn_drop: Probability of dropping whole keypoints in training.
            bev_aggregation: BEV map sampling at the same keypoints, None without BEV maps.
            sample_sparse: Whether the block samples the sparse voxels.

        Raises:
            ValueError: If the block samples no lidar features.
        """
        super().__init__()
        if embed_dims % num_groups != 0:
            raise ValueError(f"embed_dims {embed_dims} not divisible by num_groups {num_groups}.")
        if not sample_sparse and bev_aggregation is None:
            raise ValueError("The block samples the sparse voxels, BEV maps or both.")
        self.embed_dims = embed_dims
        self.num_groups = num_groups
        self.num_levels = num_levels
        self.kps_generator = kps_generator
        self.num_pts = kps_generator.num_pts
        self.attn_drop = attn_drop
        self.sample_sparse = sample_sparse
        # A BEV only block has no voxel weights, every parameter must train under DDP
        self.weights_fc = (
            nn.Linear(embed_dims, num_groups * num_levels * self.num_pts) if sample_sparse else None
        )
        self.output_proj = nn.Linear(embed_dims, embed_dims) if sample_sparse else None
        self.proj_drop = nn.Dropout(proj_drop)
        self.bev_aggregation = bev_aggregation
        if self.weights_fc is not None:
            nn.init.zeros_(self.weights_fc.weight)
            nn.init.zeros_(self.weights_fc.bias)

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
        levels: Sequence[SparseLevel],
        bev_levels: Sequence[BEVLevel] = (),
        context: Any = None,
    ) -> torch.Tensor:
        """Aggregate sparse voxel features into the instance features.

        Args:
            instance_feature: Instance features ``(B, N, C)``.
            anchor: Anchor states ``(B, N, 10)``.
            anchor_embed: Anchor embeddings ``(B, N, C)``.
            levels: Sampled encoder stages with their voxel lookup.
            bev_levels: Dense BEV maps of the BEV branch.
            context: Frame context of a subclass block, None for the lidar block.

        Returns:
            Updated instance features ``(B, N, C)``.
        """
        key_points = self.kps_generator(anchor, instance_feature)
        query = instance_feature + anchor_embed
        bev = (
            None
            if self.bev_aggregation is None
            else self.bev_aggregation(query, key_points, bev_levels)
        )
        if self.sample_sparse:
            weights = keypoint_dropout(
                _group_softmax(self.weights_fc(query), self.num_levels * self.num_pts),
                self.attn_drop,
                self.training,
            )
            features = sparse_deformable_aggregation(key_points, weights, levels, MIN_OCCUPANCY)
            output = self.output_proj(features) if bev is None else self.output_proj(features) + bev
        else:
            output = bev
        output = self.join_context(output, query, key_points, context)
        return instance_feature + self.proj_drop(output)

    def join_context(
        self, output: torch.Tensor, query: torch.Tensor, key_points: torch.Tensor, context: Any
    ) -> torch.Tensor:
        """Join the features of the frame context to the lidar features ``(B, N, C)``.

        Args:
            output: Lidar features of every instance ``(B, N, C)``.
            query: Instance features with the anchor embedding ``(B, N, C)``.
            key_points: Metric keypoints ``(B, N, P, 3)``.
            context: Frame context of the block.

        Returns:
            The lidar features, the lidar block reads no context.

        Raises:
            ValueError: If the lidar block gets a context.
        """
        if context is not None:
            raise ValueError("The lidar aggregation block reads no frame context.")
        return output


class BEVDeformableAggregation(nn.Module):
    """Sample dense BEV maps at the keypoints' (x, y) with learned weights per level and point."""

    def __init__(
        self,
        *,
        embed_dims: int,
        bev_channels: int,
        num_groups: int,
        num_levels: int,
        num_points: int,
        attn_drop: float = 0.0,
    ) -> None:
        """Initialize the BEV branch.

        Args:
            embed_dims: Instance feature dimension.
            bev_channels: Feature dimension of the BEV maps.
            num_groups: Number of feature groups sharing one weight.
            num_levels: Number of sampled BEV levels.
            num_points: Keypoints per instance.
            attn_drop: Probability of dropping whole samples in training.

        Raises:
            ValueError: If the BEV channels do not split into the groups.
        """
        super().__init__()
        if bev_channels % num_groups != 0:
            raise ValueError(f"bev_channels {bev_channels} not divisible by {num_groups} groups.")
        self.num_levels = num_levels
        self.num_samples = num_levels * num_points
        self.attn_drop = attn_drop
        self.weights_fc = nn.Linear(embed_dims, num_groups * self.num_samples)
        self.output_proj = nn.Linear(bev_channels, embed_dims)
        nn.init.zeros_(self.weights_fc.weight)
        nn.init.zeros_(self.weights_fc.bias)

    def forward(
        self, query: torch.Tensor, key_points: torch.Tensor, levels: Sequence[BEVLevel]
    ) -> torch.Tensor:
        """Aggregate the BEV samples of the keypoints into instance features.

        Args:
            query: Instance features with the anchor embedding ``(B, N, C)``.
            key_points: Metric keypoints ``(B, N, P, 3)``.
            levels: BEV levels, finest first.

        Returns:
            BEV features of the instances ``(B, N, C)``.

        Raises:
            ValueError: If the number of levels does not match the branch.
        """
        if len(levels) != self.num_levels:
            raise ValueError(f"The BEV branch samples {self.num_levels} levels, got {len(levels)}.")
        weights = keypoint_dropout(
            _group_softmax(self.weights_fc(query), self.num_samples), self.attn_drop, self.training
        )
        return self.output_proj(bev_deformable_aggregation(key_points, weights, levels))


class SparseLevelGrid(nn.Module):
    """Dense voxel grid of one sparse encoder stage, it builds the lookup of the stage voxels."""

    def __init__(self, point_cloud_range: Sequence[float], grid_size: float, feature_stride: int):
        """Initialize the stage grid.

        Args:
            point_cloud_range: Metric range of the voxel grid.
            grid_size: Metric size of the input voxel grid.
            feature_stride: Total pooling stride of the stage.

        Raises:
            ValueError: If the range or the stride is invalid.
        """
        super().__init__()
        if feature_stride < 1:
            raise ValueError(f"feature_stride must be positive, got {feature_stride}.")
        pc_range = [float(value) for value in point_cloud_range]
        if len(pc_range) != 6:
            raise ValueError(f"point_cloud_range must have 6 values, got {pc_range}.")
        base_cells = [round((pc_range[axis + 3] - pc_range[axis]) / grid_size) for axis in range(3)]
        self.grid_dims = tuple(math.ceil(cells / feature_stride) for cells in base_cells)
        self.num_cells = self.grid_dims[0] * self.grid_dims[1] * self.grid_dims[2]
        self.voxel_size = grid_size * feature_stride
        self.grid_origin = (pc_range[0], pc_range[1], pc_range[2])

    def forward(
        self, feat: torch.Tensor, grid_coord: torch.Tensor, offset: torch.Tensor
    ) -> SparseLevel:
        """Look up the stage voxels in the dense grid of every sample.

        Args:
            feat: Stage features ``(V, C)``.
            grid_coord: Stage voxel coordinates ``(V, 3)`` as ``(x, y, z)``.
            offset: Cumulative voxel offsets of the samples ``(B,)``.

        Returns:
            The stage with its voxel lookup.
        """
        size_x, size_y, size_z = self.grid_dims
        batch_index = offset_to_batch(offset, grid_coord)
        flat_cells = (
            (batch_index * size_z + grid_coord[:, 2].long()) * size_y + grid_coord[:, 1].long()
        ) * size_x + grid_coord[:, 0].long()
        index_grid = torch.full(
            (offset.shape[0] * self.num_cells,),
            EMPTY_ROW,
            dtype=torch.long,
            device=feat.device,
        )
        index_grid[flat_cells] = torch.arange(1, feat.shape[0] + 1, device=feat.device)
        padded = torch.cat([feat.new_zeros((1, feat.shape[1])), feat])
        return SparseLevel(padded, index_grid, self.grid_dims, self.grid_origin, self.voxel_size)
