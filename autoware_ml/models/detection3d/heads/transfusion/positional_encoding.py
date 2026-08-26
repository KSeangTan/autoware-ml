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

from jaxtyping import Float32
import torch.nn as nn
import torch


class LearnedPositionalEncoding(nn.Module):
    """Learn positional embeddings from 2D BEV coordinates.

    The module maps BEV cell coordinates into query or key embeddings used by
    the TransFusion decoder.
    """

    def __init__(self, input_channels: int, embed_dims: int) -> None:
        """Initialize the positional encoding module.

        Args:
            input_channels: Number of input coordinate channels.
            embed_dims: Output embedding dimension.
        """
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_channels, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims),
        )

    def forward(
        self, bev_positions: Float32[torch.Tensor, "B H W"]
    ) -> Float32[torch.Tensor, "B H W"]:
        """Encode BEV positions into query embeddings.

        Args:
            bev_positions: BEV coordinate tensor.

        Returns:
            Learned positional embeddings.
        """
        return self.proj(bev_positions)
