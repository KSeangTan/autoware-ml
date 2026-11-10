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

"""Positional embedding of CMT (arXiv 2301.01283) shared by the BEV cells and the queries."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn

from autoware_ml.ops.deformable_aggregation.bev_deformable_aggregation import BEVLevel


def sincos_position_embedding(
    position: torch.Tensor, num_feats: int, temperature: float = 10000.0
) -> torch.Tensor:
    """Sine and cosine embedding of unit range ``(x, y)`` positions, the ``pos2embed`` of CMT.

    Every axis gets ``num_feats`` features over the wavelengths of the transformer embedding,
    the sine and the cosine of every wavelength alternating. The y features come first.

    Args:
        position: Positions ``(..., 2)`` in the unit range.
        num_feats: Features per axis, even.
        temperature: Base of the wavelength progression.

    Returns:
        Embedding ``(..., 2 * num_feats)``.
    """
    scaled = position * (2.0 * math.pi)
    wavelength = torch.arange(num_feats, dtype=scaled.dtype, device=scaled.device)
    wavelength = temperature ** (2.0 * wavelength.div(2, rounding_mode="floor") / num_feats)
    embeddings = []
    for axis in (1, 0):
        phase = scaled[..., axis, None] / wavelength
        embeddings.append(
            torch.stack((phase[..., 0::2].sin(), phase[..., 1::2].cos()), dim=-1).flatten(-2)
        )
    return torch.cat(embeddings, dim=-1)


class CMTPositionEmbedding(nn.Module):
    """Positional embedding of metric ``(x, y)`` shared by the BEV cells and the queries.

    CMT (arXiv 2301.01283) puts the queries and the BEV map in one positional space: the
    position, normalized to the unit range over the point cloud range, becomes a sine and cosine
    embedding that one MLP maps to the feature dimension. A query reads it at its anchor center,
    a BEV cell at its center, so a query and the cell under it share one embedding.
    """

    def __init__(
        self, embed_dims: int, point_cloud_range: Sequence[float], temperature: float = 10000.0
    ) -> None:
        """Initialize the shared embedding.

        Args:
            embed_dims: Feature dimension of the embedding, even.
            point_cloud_range: Metric range ``(x_min, y_min, z_min, x_max, y_max, z_max)`` that
                normalizes the positions.
            temperature: Base of the wavelength progression of the sine and cosine embedding.

        Raises:
            ValueError: If the feature dimension is odd.
        """
        super().__init__()
        if embed_dims % 2 != 0:
            raise ValueError(f"The sine and cosine embedding needs even dims, got {embed_dims}.")
        self.embed_dims = embed_dims
        self.temperature = temperature
        self.register_buffer(
            "origin",
            torch.tensor([float(point_cloud_range[0]), float(point_cloud_range[1])]),
            persistent=False,
        )
        self.register_buffer(
            "extent",
            torch.tensor(
                [
                    float(point_cloud_range[3] - point_cloud_range[0]),
                    float(point_cloud_range[4] - point_cloud_range[1]),
                ]
            ),
            persistent=False,
        )
        self.mlp = nn.Sequential(
            nn.Linear(embed_dims * 2, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims),
        )

    def forward(self, position: torch.Tensor) -> torch.Tensor:
        """Embed metric ``(x, y)`` positions ``(..., 2)`` into ``(..., C)``."""
        unit = (position - self.origin) / self.extent
        return self.mlp(sincos_position_embedding(unit, self.embed_dims, self.temperature))

    def cells(self, level: BEVLevel) -> torch.Tensor:
        """Embed the cell centers of a BEV level ``(H * W, C)``, row major like its flattened map."""
        height, width = level.feat.shape[-2:]
        origin = self.origin.new_tensor(level.grid_origin)
        xs = (torch.arange(width, device=origin.device) + 0.5) * level.cell_size + origin[0]
        ys = (torch.arange(height, device=origin.device) + 0.5) * level.cell_size + origin[1]
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        return self(torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2))
