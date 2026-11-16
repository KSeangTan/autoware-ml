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

"""Sampling weights shared by the deformable aggregations of the lidar and the image blocks."""

from __future__ import annotations

import torch


def group_softmax(logits: torch.Tensor, num_samples: int) -> torch.Tensor:
    """Softmax over the samples of every feature group, ``(B, N, S * G)`` to ``(B, N, S, G)``."""
    batch_size, num_anchor = logits.shape[:2]
    return logits.reshape(batch_size, num_anchor, num_samples, -1).softmax(dim=-2)


def keypoint_dropout(weights: torch.Tensor, drop: float, training: bool) -> torch.Tensor:
    """Drop whole samples of the aggregation weights in training and rescale the rest."""
    if not training or drop == 0.0:
        return weights
    mask = torch.rand_like(weights[..., :1]) > drop
    return weights * mask / (1.0 - drop)
