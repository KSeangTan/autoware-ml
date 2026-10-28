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
# Adapted from projects/mmdet3d_plugin/ops/__init__.py and deformable_aggregation.py of
# https://github.com/HorizonRobotics/Sparse4D (MIT License). The flattening keeps the token
# layout of the reference ``feature_maps_format`` so the deployed graphs of the reference and
# its TensorRT plugin read the same tensors; it returns a named tuple instead of a list.

"""Efficient deformable aggregation of Sparse4D.

The Sparse4D decoder refines a set of 3D anchors. Every anchor owns a few 3D key points,
which are projected into every camera and sampled from every level of that camera's feature
pyramid. The samples are then summed with learned weights over the points, the cameras and
the levels into one feature per anchor. Done with ``grid_sample`` the sampled tensor is
``(batch, anchors, points, cameras, levels, channels)`` before the weighted sum; the CUDA
extension fuses the sampling and the sum and never materializes it.

:func:`flatten_feature_maps` turns a feature pyramid into the token layout the kernel reads,
and :func:`deformable_aggregation` runs the kernel with autograd support.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

from jaxtyping import Float, Float32, Int32, Int64
import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from . import deformable_aggregation_ext


class FlattenedFeatureMaps(NamedTuple):
    """Feature pyramids of every camera in the token layout of the aggregation kernel.

    Attributes:
        features: Tokens of every camera, level by level, pixels row-major:
            ``(batch, num_tokens, channels)`` with
            ``num_tokens = cameras * sum(height_l * width_l)``.
        spatial_shapes: ``(height, width)`` of every level of every camera,
            ``(cameras, levels, 2)``.
        level_start_index: Token offset of every level of every camera, ``(cameras, levels)``.
    """

    features: Float[torch.Tensor, "batch_size num_tokens channels"]
    spatial_shapes: Int64[torch.Tensor, "num_cams num_levels 2"]
    level_start_index: Int64[torch.Tensor, "num_cams num_levels"]


def flatten_feature_maps(
    feature_maps: Sequence[Float[torch.Tensor, "batch_size num_cams channels height width"]],
) -> FlattenedFeatureMaps:
    """Flatten a multi-camera feature pyramid into the token layout of the kernel.

    Every camera has the same pyramid, so the shapes and the offsets repeat per camera. The
    kernel itself allows a different pyramid per camera; a caller with such cameras builds
    the named tuple by hand.

    Args:
        feature_maps: One ``(batch, cameras, channels, height, width)`` tensor per level, in
            the order the levels are indexed by the weights.

    Returns:
        The flattened pyramid.

    Raises:
        ValueError: If no level is given, or the levels differ in batch size, cameras or
            channels.
    """
    if len(feature_maps) == 0:
        raise ValueError("flatten_feature_maps needs at least one level.")
    batch_size, num_cams, num_channels = feature_maps[0].shape[:3]
    for level, feature_map in enumerate(feature_maps):
        if feature_map.dim() != 5 or feature_map.shape[:3] != (batch_size, num_cams, num_channels):
            raise ValueError(
                f"Level {level} must have shape ({batch_size}, {num_cams}, {num_channels}, "
                f"height, width), got {tuple(feature_map.shape)}."
            )

    level_shapes = [feature_map.shape[-2:] for feature_map in feature_maps]
    # (batch, cameras, channels, sum of the level areas) -> (batch, cameras * tokens, channels)
    features = torch.cat([feature_map.flatten(3) for feature_map in feature_maps], dim=-1).permute(
        0, 1, 3, 2
    )
    features = features.flatten(1, 2)

    spatial_shapes = torch.tensor(
        [level_shapes] * num_cams, dtype=torch.int64, device=features.device
    )
    level_sizes = spatial_shapes[..., 0] * spatial_shapes[..., 1]
    level_start_index = level_sizes.flatten().cumsum(dim=0) - level_sizes.flatten()
    return FlattenedFeatureMaps(
        features=features,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index.reshape(num_cams, -1),
    )


class DeformableAggregationFunction(Function):
    """Autograd wrapper of the CUDA deformable aggregation.

    The kernel works in float32 on contiguous tensors, so the inputs are cast on the way in;
    the gradients come back in the dtype of the cast inputs.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        features: Float[torch.Tensor, "batch_size num_tokens channels"],
        spatial_shapes: Int32[torch.Tensor, "num_cams num_levels 2"]
        | Int64[torch.Tensor, "num_cams num_levels 2"],
        level_start_index: Int32[torch.Tensor, "num_cams num_levels"]
        | Int64[torch.Tensor, "num_cams num_levels"],
        sampling_locations: Float[torch.Tensor, "batch_size num_anchors num_points num_cams 2"],
        weights: Float[
            torch.Tensor, "batch_size num_anchors num_points num_cams num_levels num_groups"
        ],
    ) -> Float32[torch.Tensor, "batch_size num_anchors channels"]:
        """Run the forward kernel and keep the inputs for the backward.

        Args:
            ctx: Autograd context.
            features: Flattened feature pyramids, see :class:`FlattenedFeatureMaps`.
            spatial_shapes: Heights and widths of the levels of every camera.
            level_start_index: Token offsets of the levels of every camera.
            sampling_locations: ``(x, y)`` of every key point in every camera, normalized to
                ``[0, 1]`` of the image; a point outside ``(0, 1)`` on either axis is skipped.
            weights: Aggregation weight of every (point, camera, level) sample per channel
                group; the channels are split into ``num_groups`` consecutive groups.

        Returns:
            The aggregated features of every anchor.
        """
        features = features.contiguous().float()
        spatial_shapes = spatial_shapes.contiguous().int()
        level_start_index = level_start_index.contiguous().int()
        sampling_locations = sampling_locations.contiguous().float()
        weights = weights.contiguous().float()
        output = deformable_aggregation_ext.deformable_aggregation_forward(
            features, spatial_shapes, level_start_index, sampling_locations, weights
        )
        ctx.save_for_backward(
            features, spatial_shapes, level_start_index, sampling_locations, weights
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: Float[torch.Tensor, "batch_size num_anchors channels"],
    ) -> tuple[torch.Tensor, None, None, torch.Tensor, torch.Tensor]:
        """Run the backward kernel.

        Args:
            ctx: Autograd context holding the forward inputs.
            grad_output: Gradient of the aggregated features.

        Returns:
            The gradients of ``features``, ``sampling_locations`` and ``weights``; the two
            index tensors get ``None``.
        """
        features, spatial_shapes, level_start_index, sampling_locations, weights = ctx.saved_tensors
        grad_features, grad_sampling_locations, grad_weights = (
            deformable_aggregation_ext.deformable_aggregation_backward(
                features,
                spatial_shapes,
                level_start_index,
                sampling_locations,
                weights,
                grad_output.contiguous().float(),
            )
        )
        return grad_features, None, None, grad_sampling_locations, grad_weights


def deformable_aggregation(
    features: Float[torch.Tensor, "batch_size num_tokens channels"],
    spatial_shapes: Int32[torch.Tensor, "num_cams num_levels 2"]
    | Int64[torch.Tensor, "num_cams num_levels 2"],
    level_start_index: Int32[torch.Tensor, "num_cams num_levels"]
    | Int64[torch.Tensor, "num_cams num_levels"],
    sampling_locations: Float[torch.Tensor, "batch_size num_anchors num_points num_cams 2"],
    weights: Float[
        torch.Tensor, "batch_size num_anchors num_points num_cams num_levels num_groups"
    ],
) -> Float32[torch.Tensor, "batch_size num_anchors channels"]:
    """Aggregate the multi-camera, multi-level features sampled at the key points of anchors.

    See :class:`DeformableAggregationFunction` for the arguments. The output of every anchor is
    ``sum over (point, camera, level) of weight * bilinear_sample(features, location)``, where
    the sample follows ``grid_sample`` with ``align_corners=False`` and zero padding, and a
    location on or outside the image border contributes nothing.

    Returns:
        The aggregated features of every anchor, float32.
    """
    return DeformableAggregationFunction.apply(
        features, spatial_shapes, level_start_index, sampling_locations, weights
    )
