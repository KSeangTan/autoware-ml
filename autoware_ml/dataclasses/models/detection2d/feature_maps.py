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

"""Dataclasses for the image feature maps a 2D encoder hands to a sparse decoder."""

from __future__ import annotations

from jaxtyping import Float, Int64
from pydantic import BaseModel, ConfigDict, model_validator
import torch


class FlattenedFeatureMaps(BaseModel):
    """
    Feature pyramids of every camera in the token layout of a deformable aggregation kernel.

    Attributes:
        features: Tokens of every camera, level by level, pixels row-major:
            ``(batch, num_tokens, channels)`` with
            ``num_tokens = cameras * sum(height_l * width_l)``.
        spatial_shapes: ``(height, width)`` of every level of every camera,
            ``(cameras, levels, 2)``.
        level_start_index: Token offset of every level of every camera, ``(cameras, levels)``.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    features: Float[torch.Tensor, "batch_size num_tokens channels"]
    spatial_shapes: Int64[torch.Tensor, "num_cams num_levels 2"]
    level_start_index: Int64[torch.Tensor, "num_cams num_levels"]

    @model_validator(mode="after")
    def _check_token_layout(self) -> FlattenedFeatureMaps:
        """Check that the level offsets and shapes describe exactly the tokens of features."""
        if self.level_start_index.shape != self.spatial_shapes.shape[:2]:
            raise ValueError(
                f"level_start_index must have shape {tuple(self.spatial_shapes.shape[:2])} "
                f"to match spatial_shapes, got {tuple(self.level_start_index.shape)}."
            )
        level_sizes = (self.spatial_shapes[..., 0] * self.spatial_shapes[..., 1]).flatten()
        expected_start_index = level_sizes.cumsum(dim=0) - level_sizes
        if not torch.equal(self.level_start_index.flatten(), expected_start_index):
            raise ValueError(
                "level_start_index must be the cumulative sum of the level areas, camera by "
                f"camera, expected {expected_start_index.tolist()}, "
                f"got {self.level_start_index.flatten().tolist()}."
            )
        num_tokens = int(level_sizes.sum().item())
        if self.features.shape[1] != num_tokens:
            raise ValueError(
                f"features must hold {num_tokens} tokens to match spatial_shapes, "
                f"got {self.features.shape[1]}."
            )
        return self
