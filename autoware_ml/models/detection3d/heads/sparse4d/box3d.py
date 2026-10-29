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
# Adapted from projects/mmdet3d_plugin/models/detection3d/detection3d_blocks.py and
# core/box3d.py of https://github.com/HorizonRobotics/Sparse4D (MIT License). The mmcv
# registries and BaseModule are replaced by plain torch modules; the parameter names are kept
# so the reference checkpoints load. The anchor projection writes the rotated yaw back in the
# (sin, cos) order of the anchor, which the reference left swapped behind a "fix bug" note (the
# fix SparseDrive applied); the temporal key points of the generator, unused by Sparse4D v3,
# are not ported.

"""Anchor boxes of Sparse4D and the modules that embed, sample and refine them.

An anchor is an 11-vector, see :class:`AnchorFieldIndex`: a center, log-scaled sizes, the
sine and cosine of the yaw, and a velocity. :class:`SparseBox3DEncoder` embeds it,
:class:`SparseBox3DKeyPointsGenerator` spreads key points over it and moves it between frames,
and :class:`SparseBox3DRefinementModule` predicts its refinement, class scores and quality.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum
import math
from typing import Literal

from jaxtyping import Float
import torch
import torch.nn as nn

from autoware_ml.models.detection3d.heads.sparse4d.layers import Scale, linear_relu_ln


class AnchorFieldIndex(IntEnum):
    """Field index of a Sparse4D anchor.

    The first six fields follow ``Box3DFieldIndex``: the reference names the size fields
    ``W, L, H`` but its converter stores ``[length, width, height]`` there, so the length,
    along the heading of the box, comes first. The sizes are log-scaled and the yaw is
    carried as its sine and cosine, so the anchor can be refined without wrapping.

    Attributes:
      X: X coordinate of the center.
      Y: Y coordinate of the center.
      Z: Z coordinate of the center.
      LENGTH: Log of the length, along the heading.
      WIDTH: Log of the width.
      HEIGHT: Log of the height.
      SIN_YAW: Sine of the yaw.
      COS_YAW: Cosine of the yaw.
      VELOCITY_X: Velocity along X.
      VELOCITY_Y: Velocity along Y.
      VELOCITY_Z: Velocity along Z.
    """

    X = 0
    Y = 1
    Z = 2
    LENGTH = 3
    WIDTH = 4
    HEIGHT = 5
    SIN_YAW = 6
    COS_YAW = 7
    VELOCITY_X = 8
    VELOCITY_Y = 9
    VELOCITY_Z = 10


class QualityIndex(IntEnum):
    """Field index of the quality estimate of an anchor.

    Attributes:
      CENTERNESS: Logit of the closeness of the center to the matched box.
      YAWNESS: Logit of the heading pointing the way of the matched box.
    """

    CENTERNESS = 0
    YAWNESS = 1


ANCHOR_DIMS = len(AnchorFieldIndex)
CENTER = [AnchorFieldIndex.X, AnchorFieldIndex.Y, AnchorFieldIndex.Z]
SIZE = [AnchorFieldIndex.LENGTH, AnchorFieldIndex.WIDTH, AnchorFieldIndex.HEIGHT]
YAW = [AnchorFieldIndex.SIN_YAW, AnchorFieldIndex.COS_YAW]
VELOCITY_START = AnchorFieldIndex.VELOCITY_X


class SparseBox3DEncoder(nn.Module):
    """Embed the anchors.

    The center, the sizes, the yaw and the velocity get their own small MLP; the four
    embeddings are summed (``add``) or concatenated (``cat``), then optionally passed through
    an output MLP.
    """

    def __init__(
        self,
        embed_dims: int | Sequence[int],
        vel_dims: int = 3,
        mode: Literal["add", "cat"] = "add",
        output_fc: bool = True,
        in_loops: int = 1,
        out_loops: int = 2,
    ) -> None:
        """Initialize the encoder.

        Args:
            embed_dims: Width of the four embeddings and of the output, or one width per
                embedding (center, sizes, yaw, velocity) followed, when an output MLP is
                asked, by the output width.
            vel_dims: Velocity components to embed, ``0`` for none.
            mode: ``add`` sums the embeddings, ``cat`` concatenates them.
            output_fc: Whether an output MLP follows.
            in_loops: Linear and ReLU pairs before every LayerNorm of the MLPs.
            out_loops: LayerNorm-terminated blocks of the MLPs.

        Raises:
            ValueError: If the mode is unknown, or the widths cannot be combined: ``add``
                needs equal widths, and ``cat`` needs the output width (the last one) to be the
                sum of the concatenated widths when an output MLP follows.
        """
        super().__init__()
        if mode not in ("add", "cat"):
            raise ValueError(f"mode must be 'add' or 'cat', got {mode!r}.")
        self.vel_dims = vel_dims
        self.mode = mode
        if isinstance(embed_dims, int):
            embed_dims = [embed_dims] * 5
        self.embed_dims = list(embed_dims)
        used_dims = self.embed_dims[:4] if vel_dims > 0 else self.embed_dims[:3]
        combined_dims = sum(used_dims) if mode == "cat" else used_dims[0]
        if mode == "add" and any(dims != combined_dims for dims in used_dims):
            raise ValueError(f"mode 'add' needs equal embedding widths, got {used_dims}.")
        if output_fc and combined_dims != self.embed_dims[-1]:
            raise ValueError(
                f"The output MLP takes {self.embed_dims[-1]} inputs but the combined embedding "
                f"is {combined_dims} wide; give the output width as a fifth value."
            )

        self.pos_fc = linear_relu_ln(self.embed_dims[0], in_loops, out_loops, len(CENTER))
        self.size_fc = linear_relu_ln(self.embed_dims[1], in_loops, out_loops, len(SIZE))
        self.yaw_fc = linear_relu_ln(self.embed_dims[2], in_loops, out_loops, len(YAW))
        self.vel_fc = (
            linear_relu_ln(self.embed_dims[3], in_loops, out_loops, vel_dims)
            if vel_dims > 0
            else None
        )
        self.output_fc = (
            linear_relu_ln(self.embed_dims[-1], in_loops, out_loops) if output_fc else None
        )

    def forward(
        self, box_3d: Float[torch.Tensor, "batch_size num_anchors anchor_dims"]
    ) -> Float[torch.Tensor, "batch_size num_anchors embed_dims"]:
        """Embed the anchors.

        Args:
            box_3d: Anchors, see :class:`AnchorFieldIndex`.

        Returns:
            The embeddings.
        """
        pos_feat = self.pos_fc(box_3d[..., CENTER])
        size_feat = self.size_fc(box_3d[..., SIZE])
        yaw_feat = self.yaw_fc(box_3d[..., YAW])
        if self.mode == "add":
            output = pos_feat + size_feat + yaw_feat
        else:
            output = torch.cat([pos_feat, size_feat, yaw_feat], dim=-1)

        if self.vel_fc is not None:
            vel_feat = self.vel_fc(box_3d[..., VELOCITY_START : VELOCITY_START + self.vel_dims])
            output = output + vel_feat if self.mode == "add" else torch.cat([output, vel_feat], -1)
        if self.output_fc is not None:
            output = self.output_fc(output)
        return output


class SparseBox3DKeyPointsGenerator(nn.Module):
    """Key points of every anchor, and the projection of anchors between frames.

    The key points are the box center and the fixed ``fix_scale`` offsets scaled by the box
    sizes and rotated by its yaw, plus ``num_learnable_pts`` offsets predicted from the
    instance feature inside the box.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_learnable_pts: int = 0,
        fix_scale: Sequence[Sequence[float]] | None = None,
    ) -> None:
        """Initialize the generator.

        Args:
            embed_dims: Width of the instance features the learnable points are predicted from.
            num_learnable_pts: Number of learnable key points.
            fix_scale: Fixed key point offsets as fractions of the box sizes along (length,
                width, height); the center alone when ``None``.
        """
        super().__init__()
        self.embed_dims = embed_dims
        self.num_learnable_pts = num_learnable_pts
        if fix_scale is None:
            fix_scale = ((0.0, 0.0, 0.0),)
        self.register_buffer(
            "fix_scale", torch.tensor([list(scale) for scale in fix_scale], dtype=torch.float32)
        )
        self.num_pts = len(fix_scale) + num_learnable_pts
        self.learnable_fc: nn.Linear | None = None
        if num_learnable_pts > 0:
            self.learnable_fc = nn.Linear(embed_dims, num_learnable_pts * 3)
            self.init_weights()

    def init_weights(self) -> None:
        """Xavier-initialize the learnable offsets with a zero bias."""
        if self.learnable_fc is not None:
            nn.init.xavier_uniform_(self.learnable_fc.weight)
            nn.init.constant_(self.learnable_fc.bias, 0.0)

    def forward(
        self,
        anchor: Float[torch.Tensor, "batch_size num_anchors anchor_dims"],
        instance_feature: Float[torch.Tensor, "batch_size num_anchors embed_dims"] | None = None,
    ) -> Float[torch.Tensor, "batch_size num_anchors num_pts 3"]:
        """Spread the key points over every anchor.

        Args:
            anchor: Anchors, see :class:`AnchorFieldIndex`.
            instance_feature: Instance features the learnable points are predicted from;
                without them only the fixed points are returned.

        Returns:
            The key points in the frame of the anchors.
        """
        batch_size, num_anchors = anchor.shape[:2]
        size = anchor[..., None, SIZE].exp()
        key_points = self.fix_scale * size
        if self.learnable_fc is not None and instance_feature is not None:
            learnable_scale = (
                self.learnable_fc(instance_feature)
                .reshape(batch_size, num_anchors, self.num_learnable_pts, 3)
                .sigmoid()
                - 0.5
            )
            key_points = torch.cat([key_points, learnable_scale * size], dim=-2)

        rotation_mat = anchor.new_zeros([batch_size, num_anchors, 3, 3])
        rotation_mat[:, :, 0, 0] = anchor[:, :, AnchorFieldIndex.COS_YAW]
        rotation_mat[:, :, 0, 1] = -anchor[:, :, AnchorFieldIndex.SIN_YAW]
        rotation_mat[:, :, 1, 0] = anchor[:, :, AnchorFieldIndex.SIN_YAW]
        rotation_mat[:, :, 1, 1] = anchor[:, :, AnchorFieldIndex.COS_YAW]
        rotation_mat[:, :, 2, 2] = 1

        key_points = torch.matmul(rotation_mat[:, :, None], key_points[..., None]).squeeze(-1)
        return key_points + anchor[..., None, CENTER]

    @staticmethod
    def anchor_projection(
        anchor: Float[torch.Tensor, "batch_size num_anchors anchor_dims"],
        src2dst: Float[torch.Tensor, "batch_size 4 4"],
        time_interval: Float[torch.Tensor, " batch_size"] | None = None,
    ) -> Float[torch.Tensor, "batch_size num_anchors anchor_dims"]:
        """Move the anchors of a frame into another frame.

        The centers first travel back along the velocity for ``time_interval`` (the time from
        the destination frame to the source frame), then the centers, headings and velocities
        are transformed by ``src2dst``; the sizes stay.

        Args:
            anchor: Anchors in the source frame.
            src2dst: Transform from the source frame to the destination frame.
            time_interval: Time from the destination to the source frame of every sample,
                ``None`` to leave the velocity out.

        Returns:
            The anchors in the destination frame.
        """
        src2dst = src2dst.to(dtype=anchor.dtype).unsqueeze(1)
        vel = anchor[..., VELOCITY_START:]
        vel_dim = vel.shape[-1]

        center = anchor[..., CENTER]
        if time_interval is not None:
            center = center - vel * time_interval.to(dtype=vel.dtype)[:, None, None]
        center = (
            torch.matmul(src2dst[..., :3, :3], center[..., None]).squeeze(-1) + src2dst[..., :3, 3]
        )
        size = anchor[..., SIZE]
        # The heading (cos, sin) rotates like a direction; it is stored back as (sin, cos).
        heading = torch.matmul(
            src2dst[..., :2, :2],
            anchor[..., [AnchorFieldIndex.COS_YAW, AnchorFieldIndex.SIN_YAW], None],
        ).squeeze(-1)
        yaw = heading[..., [1, 0]]
        vel = torch.matmul(src2dst[..., :vel_dim, :vel_dim], vel[..., None]).squeeze(-1)
        return torch.cat([center, size, yaw, vel], dim=-1)

    @staticmethod
    def distance(
        anchor: Float[torch.Tensor, "batch_size num_anchors anchor_dims"],
    ) -> Float[torch.Tensor, "batch_size num_anchors"]:
        """Distance of every anchor center from the origin in the ground plane."""
        return torch.norm(anchor[..., :2], p=2, dim=-1)


class SparseBox3DRefinementModule(nn.Module):
    """Refine the anchors and predict their class scores and quality.

    The refinement is a residual on the center, the sizes and, with ``refine_yaw``, the yaw;
    the velocity output is a translation over ``time_interval``, added to the anchor velocity.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        output_dim: int = ANCHOR_DIMS,
        num_cls: int = 10,
        normalize_yaw: bool = False,
        refine_yaw: bool = False,
        with_cls_branch: bool = True,
        with_quality_estimation: bool = False,
    ) -> None:
        """Initialize the module.

        Args:
            embed_dims: Width of the instance features.
            output_dim: Fields of the refined anchor; below 9 no velocity is predicted.
            num_cls: Number of classes.
            normalize_yaw: Whether to normalize the refined (sin, cos) to the unit circle.
            refine_yaw: Whether the yaw is refined as a residual too.
            with_cls_branch: Whether to predict class scores.
            with_quality_estimation: Whether to predict the centerness and yawness.
        """
        super().__init__()
        self.embed_dims = embed_dims
        self.output_dim = output_dim
        self.num_cls = num_cls
        self.normalize_yaw = normalize_yaw
        self.refine_yaw = refine_yaw
        self.refine_state = list(CENTER) + list(SIZE)
        if refine_yaw:
            self.refine_state += list(YAW)

        self.layers = linear_relu_ln(embed_dims, 2, 2)
        self.layers.append(nn.Linear(embed_dims, output_dim))
        self.layers.append(Scale([1.0] * output_dim))
        self.with_cls_branch = with_cls_branch
        self.cls_layers: nn.Sequential | None = None
        if with_cls_branch:
            self.cls_layers = linear_relu_ln(embed_dims, 1, 2)
            self.cls_layers.append(nn.Linear(embed_dims, num_cls))
        self.with_quality_estimation = with_quality_estimation
        self.quality_layers: nn.Sequential | None = None
        if with_quality_estimation:
            self.quality_layers = linear_relu_ln(embed_dims, 1, 2)
            self.quality_layers.append(nn.Linear(embed_dims, len(QualityIndex)))
        self.init_weights()

    def init_weights(self) -> None:
        """Start the class scores at a 1% foreground prior, the focal-loss convention."""
        if self.cls_layers is not None:
            bias_init = -math.log((1.0 - 0.01) / 0.01)
            nn.init.constant_(self.cls_layers[-1].bias, bias_init)

    def forward(
        self,
        instance_feature: Float[torch.Tensor, "batch_size num_anchors embed_dims"],
        anchor: Float[torch.Tensor, "batch_size num_anchors anchor_dims"],
        anchor_embed: Float[torch.Tensor, "batch_size num_anchors embed_dims"],
        time_interval: Float[torch.Tensor, " batch_size"] | float = 1.0,
        return_cls: bool = True,
    ) -> tuple[
        Float[torch.Tensor, "batch_size num_anchors output_dim"],
        Float[torch.Tensor, "batch_size num_anchors num_cls"] | None,
        Float[torch.Tensor, "batch_size num_anchors 2"] | None,
    ]:
        """Refine the anchors.

        Args:
            instance_feature: Instance features of the anchors.
            anchor: Anchors, see :class:`AnchorFieldIndex`.
            anchor_embed: Embedding of the anchors.
            time_interval: Time since the previous frame of every sample, which turns the
                predicted translation into a velocity.
            return_cls: Whether to predict the class scores and the quality.

        Returns:
            The refined anchors, the class scores (``None`` when not asked) and the quality
            (``None`` when not asked or not estimated).

        Raises:
            ValueError: If the class scores are asked of a module without a class branch.
        """
        feature = instance_feature + anchor_embed
        output = self.layers(feature)
        residual = torch.zeros_like(output)
        residual[..., self.refine_state] = anchor[..., self.refine_state]
        output = output + residual
        if self.normalize_yaw:
            normalized = torch.nn.functional.normalize(output[..., YAW], dim=-1)
            output = torch.cat([output[..., : YAW[0]], normalized, output[..., YAW[-1] + 1 :]], -1)
        if self.output_dim > VELOCITY_START:
            if not isinstance(time_interval, torch.Tensor):
                time_interval = instance_feature.new_tensor(time_interval)
            time_interval = time_interval.reshape(-1, 1, 1)
            velocity = output[..., VELOCITY_START:] / time_interval + anchor[..., VELOCITY_START:]
            output = torch.cat([output[..., :VELOCITY_START], velocity], dim=-1)

        cls = None
        quality = None
        if return_cls:
            if self.cls_layers is None:
                raise ValueError("The module has no classification branch.")
            cls = self.cls_layers(instance_feature)
            if self.quality_layers is not None:
                quality = self.quality_layers(feature)
        return output, cls, quality
