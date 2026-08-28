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

"""Shared export output packing for the BEVFusion branches.

Every BEVFusion export wrapper, lidar-only and camera-lidar alike, ends by packing the raw head
outputs into the runtime detection interface. It lives in its own module rather than in one
branch's module so that neither branch has to import the other to reach it.
"""

from __future__ import annotations

from jaxtyping import Float32, Int64
import torch
import torch.nn as nn
import torch.nn.functional as F


def export_detection_outputs(
    head: nn.Module,
    outputs: dict[str, torch.Tensor],
) -> tuple[
    Float32[torch.Tensor, "num_box_code num_proposals"],
    Float32[torch.Tensor, " num_proposals"],
    Int64[torch.Tensor, " num_proposals"],
]:
    """Pack raw head outputs into the runtime detection interface.

    The runtime consumes the raw regression channels and decodes them with
    its own parameters, so no metric-space decoding happens in the graph.

    Args:
        head: TransFusion detection head producing the output dictionary.
        outputs: Raw prediction tensors from the head forward pass.

    Returns:
        Tuple of ``bbox_pred`` with the concatenated regression channels of
        shape ``(10, num_proposals)``, ``score`` of shape ``(num_proposals,)``,
        and ``label_pred`` of shape ``(num_proposals,)``.

    Raises:
        ValueError: If the detection head has no velocity branch.
    """
    num_proposals = head.num_proposals
    query_labels = outputs["query_labels"]
    heatmap = outputs["heatmap"][..., -num_proposals:].sigmoid()
    one_hot = (
        F.one_hot(query_labels, num_classes=head.num_classes).permute(0, 2, 1).to(heatmap.dtype)
    )
    score = (heatmap * outputs["query_heatmap_score"] * one_hot)[0].max(dim=0).values

    if outputs.get("vel") is None:
        raise ValueError("BEVFusion export requires a velocity branch in the detection head.")
    bbox_pred = torch.cat(
        [outputs[key][0, :, -num_proposals:] for key in ("center", "height", "dim", "rot", "vel")],
        dim=0,
    )
    return bbox_pred, score, query_labels[0]
