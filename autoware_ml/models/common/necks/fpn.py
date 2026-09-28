# Copyright (c) OpenMMLab. All rights reserved.
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

"""Feature pyramid neck modules.

This module provides the top-down feature pyramid neck of
`Feature Pyramid Networks for Object Detection <https://arxiv.org/abs/1612.03144>`_, used to
turn a multi-scale backbone pyramid into feature maps of one common width.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum

import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float32
from torch import Tensor


class ExtraConvsSource(str, Enum):
    """How the pyramid levels beyond the backbone are built.

    ``MAX_POOLING`` halves the last output without new weights, the other members name the
    feature map the first strided extra block is fed with.
    """

    MAX_POOLING = "max_pooling"
    ON_INPUT = "on_input"
    ON_LATERAL = "on_lateral"
    ON_OUTPUT = "on_output"


class FPN(nn.Module):
    """Fuse a multi-scale feature pyramid in a top-down FPN neck.

    Every used backbone level is projected to ``out_channels`` by a 1x1 lateral block, deeper
    levels are upsampled with nearest interpolation and added to the shallower ones, and each
    fused level is smoothed by a 3x3 block. Extra levels beyond the backbone are appended
    either by strided 3x3 blocks or by max pooling. Every block is a convolution optionally
    followed by batch normalization and a ReLU activation.

    With three backbone levels (``C2`` the shallowest at the highest resolution) and
    ``num_outs=4``, the data flows as follows. ``lateral_convs[i]`` is the 1x1 block of
    level ``i`` and ``fpn_convs[i]`` its 3x3 smoothing block. The lateral path runs left to
    right at every level, the top-down path runs downward from the deepest lateral to the
    shallowest::

        backbone     lateral_convs (1x1)         top-down path           fpn_convs (3x3)   outputs

        C4 ------> lateral_convs[2] --> L4 -------------------------> fpn_convs[2] --> P4
                                        |
                                   upsample x2
                                        v
        C3 ------> lateral_convs[1] --> L3 = L3 + up(L4) -----------> fpn_convs[1] --> P3
                                        |
                                   upsample x2
                                        v
        C2 ------> lateral_convs[0] --> L2 = L2 + up(L3) -----------> fpn_convs[0] --> P2

    The fourth output is an extra level built from the source named by ``add_extra_convs``.
    ``fpn_convs[3]`` is a strided 3x3 block for the ``ON_*`` sources and does not exist for
    ``MAX_POOLING``::

        ON_INPUT     C4 --> fpn_convs[3] (stride 2) --> P5
        ON_LATERAL   L4 --> fpn_convs[3] (stride 2) --> P5
        ON_OUTPUT    P4 --> fpn_convs[3] (stride 2) --> P5
        MAX_POOLING  P4 --> max_pool2d   (stride 2) --> P5

    Further extra levels chain off the previous extra output, ``P5 --> fpn_convs[4] --> P6``,
    with a ReLU in between when ``relu_before_extra_convs`` is set.
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int,
        num_outs: int,
        start_level: int = 0,
        end_level: int = -1,
        add_extra_convs: ExtraConvsSource = ExtraConvsSource.MAX_POOLING,
        relu_before_extra_convs: bool = False,
        conv_bias: bool | None = None,
        with_norm: bool = False,
        with_activation: bool = False,
    ) -> None:
        """Initialize the feature pyramid neck.

        Args:
            in_channels: Input channel dimensions for each backbone level, from high to low
                resolution.
            out_channels: Output channel dimension used at every pyramid level.
            num_outs: Number of output pyramid levels.
            start_level: Index of the first backbone level used to build the pyramid.
            end_level: Index of the last backbone level used to build the pyramid, inclusive.
                ``-1`` means the last level. When it is not the last level, no extra level is
                allowed and ``num_outs`` must cover exactly the selected levels.
            add_extra_convs: How to build the levels beyond the backbone when ``num_outs``
                exceeds the used backbone levels. ``MAX_POOLING`` max pools the last output,
                the other sources add strided extra blocks fed by the last backbone level
                (``ON_INPUT``), the last lateral (``ON_LATERAL``) or the last output
                (``ON_OUTPUT``).
            relu_before_extra_convs: Whether to apply a ReLU to the source of every extra
                block before it is convolved, as RetinaNet does.
            conv_bias: Whether the convolutions have a bias term. None picks the usual default,
                no bias when a normalization layer follows the convolution and a bias
                otherwise.
            with_norm: Whether every block follows its convolution with batch normalization.
            with_activation: Whether every block ends with a ReLU activation.

        Raises:
            ValueError: If the level range or the number of outputs is inconsistent.
        """
        super().__init__()
        self.in_channels = list(in_channels)
        self.out_channels = out_channels
        self.num_ins = len(self.in_channels)
        self.num_outs = num_outs

        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins
            if num_outs < self.num_ins - start_level:
                raise ValueError(
                    f"num_outs ({num_outs}) must cover the {self.num_ins - start_level} "
                    "backbone levels from start_level."
                )
        else:
            # If end_level is not the last level, no extra level is allowed
            self.backbone_end_level = end_level + 1
            if end_level >= self.num_ins:
                raise ValueError(
                    f"end_level ({end_level}) must be below the number of inputs ({self.num_ins})."
                )
            if num_outs != end_level - start_level + 1:
                raise ValueError(
                    f"num_outs ({num_outs}) must equal the {end_level - start_level + 1} "
                    "selected backbone levels when end_level is not the last level."
                )
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = ExtraConvsSource(add_extra_convs)
        self.relu_before_extra_convs = relu_before_extra_convs
        self.conv_bias = conv_bias if conv_bias is not None else not with_norm
        self.with_norm = with_norm
        self.with_activation = with_activation

        lateral_convs = []
        fpn_convs = []
        for level in range(self.start_level, self.backbone_end_level):
            lateral_convs.append(
                self._build_block(self.in_channels[level], out_channels, kernel_size=1)
            )
            fpn_convs.append(self._build_block(out_channels, out_channels, kernel_size=3))

        # Extra strided blocks on top of the pyramid (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs != ExtraConvsSource.MAX_POOLING and extra_levels >= 1:
            for extra_level in range(extra_levels):
                if extra_level == 0 and self.add_extra_convs == ExtraConvsSource.ON_INPUT:
                    extra_in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    extra_in_channels = out_channels
                fpn_convs.append(
                    self._build_block(extra_in_channels, out_channels, kernel_size=3, stride=2)
                )

        self.lateral_convs = nn.ModuleList(lateral_convs)
        self.fpn_convs = nn.ModuleList(fpn_convs)
        self._init_weights()

    def _build_block(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1
    ) -> nn.Sequential:
        """Build one pyramid block with the bias, normalization and activation of this neck.

        Args:
            in_channels: Input channel count.
            out_channels: Output channel count.
            kernel_size: Convolution kernel size, padded so the spatial size only follows the
                stride.
            stride: Convolution stride.

        Returns:
            The convolution block, with the convolution first and the optional normalization
            and activation after it.
        """
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                bias=self.conv_bias,
            )
        ]
        if self.with_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        if self.with_activation:
            layers.append(nn.ReLU(inplace=False))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        """Initialize every convolution with Xavier uniform weights."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self, inputs: Sequence[Float32[Tensor, "batch_size in_channels height width"]]
    ) -> tuple[Float32[Tensor, "batch_size out_channels height width"], ...]:
        """Fuse the backbone pyramid into ``num_outs`` feature maps.

        Args:
            inputs: Backbone feature maps ordered from high to low resolution, one per entry of
                ``in_channels``.

        Returns:
            Tuple of ``num_outs`` feature maps of ``out_channels`` width, ordered from high to
            low resolution.

        Raises:
            ValueError: If the number of inputs does not match ``in_channels``.
        """
        if len(inputs) != len(self.in_channels):
            raise ValueError(
                f"Expected {len(self.in_channels)} input feature maps, got {len(inputs)}."
            )

        # Build laterals
        laterals = [
            lateral_conv(inputs[level + self.start_level])
            for level, lateral_conv in enumerate(self.lateral_convs)
        ]

        # Build the top-down path, upsampling to the size of the shallower level
        used_backbone_levels = len(laterals)
        for level in range(used_backbone_levels - 1, 0, -1):
            laterals[level - 1] = laterals[level - 1] + F.interpolate(
                laterals[level], size=laterals[level - 1].shape[2:], mode="nearest"
            )

        # Build outputs, part 1: from the backbone levels
        outs = [self.fpn_convs[level](laterals[level]) for level in range(used_backbone_levels)]

        # Part 2: extra levels on top of the pyramid
        if self.num_outs > len(outs):
            if self.add_extra_convs == ExtraConvsSource.MAX_POOLING:
                # Max pool to get more levels on top of the outputs (e.g., Faster R-CNN)
                for _ in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            else:
                # Strided blocks on top of the chosen source (e.g., RetinaNet)
                if self.add_extra_convs == ExtraConvsSource.ON_INPUT:
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == ExtraConvsSource.ON_LATERAL:
                    extra_source = laterals[-1]
                else:
                    extra_source = outs[-1]
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for level in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[level](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[level](outs[-1]))
        return tuple(outs)
