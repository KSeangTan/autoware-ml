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
# ``linear_relu_ln`` is adapted from projects/mmdet3d_plugin/models/blocks.py of
# https://github.com/HorizonRobotics/Sparse4D (MIT License); ``Scale`` replaces the mmcv layer
# of the same name the reference used.

"""Small layers shared by the Sparse4D blocks."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


def linear_relu_ln(
    embed_dims: int, in_loops: int, out_loops: int, input_dims: int | None = None
) -> nn.Sequential:
    """Build the ``(Linear, ReLU) x in_loops, LayerNorm`` x out_loops stack.

    Args:
        embed_dims: Width of every layer after the first.
        in_loops: Linear and ReLU pairs before every LayerNorm.
        out_loops: Number of LayerNorm-terminated blocks.
        input_dims: Width of the input, ``embed_dims`` when ``None``.

    Returns:
        The stack, which a caller may extend with ``append``.
    """
    if input_dims is None:
        input_dims = embed_dims
    layers = nn.Sequential()
    for _ in range(out_loops):
        for _ in range(in_loops):
            layers.append(nn.Linear(input_dims, embed_dims))
            layers.append(nn.ReLU(inplace=True))
            input_dims = embed_dims
        layers.append(nn.LayerNorm(embed_dims))
    return layers


class Scale(nn.Module):
    """Learnable per-channel scale of the last axis."""

    def __init__(self, scale: Sequence[float]) -> None:
        """Initialize the scale.

        Args:
            scale: Initial scale of every channel.
        """
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(list(scale), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scale the last axis."""
        return x * self.scale
