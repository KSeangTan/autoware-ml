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
# Adapted from the ``graph_model`` method of projects/mmdet3d_plugin/models/sparse4d_head.py
# of https://github.com/HorizonRobotics/Sparse4D (MIT License), which wrapped mmcv's
# MultiheadAttention. The decoupling and the value projections live in the module here.

"""The graph attention among the instances of Sparse4D."""

from __future__ import annotations

from jaxtyping import Bool, Float
import torch
import torch.nn as nn


class GraphAttention(nn.Module):
    """Multi-head attention among instances, with their anchor embeddings as positions.

    Coupled, the positions are added to the queries and the keys before attending and the
    output is added to the raw queries, as mmcv's ``MultiheadAttention`` does. Decoupled, the
    queries and the keys are the features concatenated with their positions, the attention
    runs at twice the width, the values are mapped into that width by ``value_proj`` and the
    result back by ``output_proj``. The two projections may be shared by several attentions,
    as the reference shares them across the layers of the head.
    """

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        decouple_attn: bool = False,
        value_proj: nn.Linear | None = None,
        output_proj: nn.Linear | None = None,
    ) -> None:
        """Initialize the attention.

        Args:
            embed_dims: Width of the instance features and of their positions.
            num_heads: Number of attention heads.
            attn_drop: Dropout of the attention weights.
            proj_drop: Dropout of the attended output before the residual.
            decouple_attn: Whether to attend over the features concatenated with the
                positions.
            value_proj: Map of the values into the decoupled width, a new one when ``None``;
                read only when decoupled.
            output_proj: Map of the decoupled output back to ``embed_dims``, a new one when
                ``None``; read only when decoupled.
        """
        super().__init__()
        self.embed_dims = embed_dims
        self.decouple_attn = decouple_attn
        attn_dims = embed_dims * 2 if decouple_attn else embed_dims
        self.attn = nn.MultiheadAttention(attn_dims, num_heads, dropout=attn_drop, batch_first=True)
        self.proj_drop = nn.Dropout(proj_drop)
        self.value_proj: nn.Linear | None = None
        self.output_proj: nn.Linear | None = None
        if decouple_attn:
            self.value_proj = (
                nn.Linear(embed_dims, attn_dims, bias=False) if value_proj is None else value_proj
            )
            self.output_proj = (
                nn.Linear(attn_dims, embed_dims, bias=False) if output_proj is None else output_proj
            )

    def forward(
        self,
        query: Float[torch.Tensor, "batch_size num_queries embed_dims"],
        key: Float[torch.Tensor, "batch_size num_keys embed_dims"] | None = None,
        value: Float[torch.Tensor, "batch_size num_keys embed_dims"] | None = None,
        query_pos: Float[torch.Tensor, "batch_size num_queries embed_dims"] | None = None,
        key_pos: Float[torch.Tensor, "batch_size num_keys embed_dims"] | None = None,
        attn_mask: Bool[torch.Tensor, "num_queries num_keys"] | None = None,
    ) -> Float[torch.Tensor, "batch_size num_queries embed_dims"]:
        """Attend from the queries to the keys.

        Args:
            query: Queries, which are also the residual.
            key: Keys, the queries when ``None``.
            value: Values, the keys when ``None``; decoupled, given values are projected
                into the attention width while defaulted ones are the concatenated keys.
            query_pos: Position embedding of the queries, required when decoupled.
            key_pos: Position embedding of the keys, ``query_pos`` when ``None`` and the keys
                are the queries.
            attn_mask: True where a query must not attend to a key.

        Returns:
            The queries plus the attended values, in ``embed_dims``.
        """
        if self.decouple_attn:
            if query_pos is None:
                raise ValueError("Decoupled attention needs query_pos.")
            query = torch.cat([query, query_pos], dim=-1)
            if key is not None:
                if key_pos is None:
                    raise ValueError("Decoupled attention needs key_pos with a key.")
                key = torch.cat([key, key_pos], dim=-1)
            query_pos, key_pos = None, None
            if value is not None:
                value = self.value_proj(value)

        identity = query
        if key is None:
            key = query
        if value is None:
            value = key
        if key_pos is None and query_pos is not None and query_pos.shape == key.shape:
            key_pos = query_pos
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos
        out, _ = self.attn(query, key, value, attn_mask=attn_mask, need_weights=False)
        out = identity + self.proj_drop(out)
        if self.output_proj is not None:
            out = self.output_proj(out)
        return out
