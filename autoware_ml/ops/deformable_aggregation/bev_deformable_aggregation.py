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

"""Deformable aggregation of dense BEV features at the (x, y) of keypoints."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn.functional as F


class KeyPointLevel(Protocol):
    """Dense features that metric keypoints sample, what the aggregation reads of a level.

    A level places the keypoints on its own map: :class:`BEVLevel` by their ``(x, y)`` on the
    BEV grid, and the image level of a camera by their projection into its image.
    """

    feat: torch.Tensor

    def sample(self, key_points: torch.Tensor) -> torch.Tensor:
        """Features ``(B, N, P, C)`` at the keypoints ``(B, N, P, 3)``, zeros off the map."""


@dataclass(frozen=True)
class BEVLevel:
    """Dense BEV features of one scale with the metric placement of their cells.

    Attributes:
        feat: BEV features ``(B, C, H, W)``, row ``y`` and column ``x`` of the cell grid.
        grid_origin: Metric ``(x, y)`` of the grid corner.
        cell_size: Metric size of a BEV cell.
    """

    feat: torch.Tensor
    grid_origin: tuple[float, float]
    cell_size: float

    def sample(self, key_points: torch.Tensor) -> torch.Tensor:
        """Sample the map bilinearly at the keypoints' (x, y), outside the map reads zeros.

        Args:
            key_points: Metric keypoints ``(B, N, P, 3)``, the height is not read.

        Returns:
            Sampled features ``(B, N, P, C)``.
        """
        height, width = self.feat.shape[-2:]
        extent = key_points.new_tensor([width * self.cell_size, height * self.cell_size])
        origin = key_points.new_tensor(self.grid_origin)
        grid = (key_points[..., :2] - origin) / extent * 2.0 - 1.0
        sampled = F.grid_sample(
            self.feat, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        return sampled.permute(0, 2, 3, 1)


def bev_deformable_aggregation(
    key_points: torch.Tensor, weights: torch.Tensor, levels: Sequence[KeyPointLevel]
) -> torch.Tensor:
    """Sum the samples of every level and keypoint with their weights per feature group.

    The aggregation of the lidar and of the image blocks: the levels are the BEV maps of the
    lidar branch, or the map of every camera and stride of the camera branch. Exports as one
    ONNX ``GridSample`` per level and plain arithmetic.

    Args:
        key_points: Metric keypoints ``(B, N, P, 3)``.
        weights: Sampling weights ``(B, N, L * P, G)``, level major.
        levels: Sampled levels with the same channels, in the order of the weights.

    Returns:
        Aggregated features ``(B, N, C)``.
    """
    batch_size, num_anchor, num_samples, num_groups = weights.shape
    channels = levels[0].feat.shape[1]
    sampled = torch.cat([level.sample(key_points) for level in levels], dim=2).reshape(
        batch_size, num_anchor, num_samples, num_groups, channels // num_groups
    )
    return (sampled * weights[..., None]).sum(dim=2).reshape(batch_size, num_anchor, channels)
