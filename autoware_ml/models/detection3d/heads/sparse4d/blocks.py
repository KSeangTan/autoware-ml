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
# Adapted from projects/mmdet3d_plugin/models/blocks.py of
# https://github.com/HorizonRobotics/Sparse4D (MIT License). The mmcv registries, config
# dicts and BaseModule are replaced by plain torch modules taking their submodules as
# arguments; the parameter names are kept so the reference checkpoints load. The unused
# temporal fusion hook of the deformable aggregation is dropped, and its grid_sample path skips
# the key points outside the image as the CUDA kernel does.

"""Decoder blocks of Sparse4D.

The Sparse4D decoder refines a set of 3D anchors with their instance features. Its layers
are built from the blocks here: :class:`AsymmetricFFN`, the feed-forward block whose input is
wider than its output, :class:`DeformableFeatureAggregation`, which samples the image feature
pyramids at the key points of every anchor and sums the samples with predicted weights, and
:class:`DenseDepthNet`, the auxiliary depth head that supervises the image features.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from jaxtyping import Float
import torch
import torch.nn as nn
import torch.nn.functional as F

from autoware_ml.models.detection3d.heads.sparse4d.box3d import SparseBox3DKeyPointsGenerator
from autoware_ml.models.detection3d.heads.sparse4d.layers import linear_relu_ln
from autoware_ml.dataclasses.models.detection2d.feature_maps import FlattenedFeatureMaps
from autoware_ml.ops.deformable_aggregation import deformable_aggregation

# Entries of a 4x4 projection matrix the camera embedding reads: the three rows that map a
# homogeneous point to the image.
_PROJECTION_ROWS = 3
_CAMERA_EMBED_INPUT_DIMS = _PROJECTION_ROWS * 4


class AsymmetricFFN(nn.Module):
    """Feed-forward block whose input can be wider than its output.

    The decoder concatenates the aggregated image features with the instance features, so the
    block takes ``in_channels`` (twice the embedding) and returns ``embed_dims``. The residual
    goes through ``identity_fc``, a linear projection when the widths differ.
    """

    def __init__(
        self,
        in_channels: int | None = None,
        pre_norm: bool = False,
        embed_dims: int = 256,
        feedforward_channels: int = 1024,
        num_fcs: int = 2,
        ffn_drop: float = 0.0,
        add_identity: bool = True,
    ) -> None:
        """Initialize the feed-forward block.

        Args:
            in_channels: Width of the input, ``embed_dims`` when ``None``.
            pre_norm: Whether to normalize the input with a LayerNorm first.
            embed_dims: Width of the output.
            feedforward_channels: Width of the hidden layers.
            num_fcs: Number of linear layers, at least 2.
            ffn_drop: Dropout probability after every layer.
            add_identity: Whether to add the (projected) input to the output.

        Raises:
            ValueError: If ``num_fcs`` is below 2.
        """
        super().__init__()
        if num_fcs < 2:
            raise ValueError(f"num_fcs must be at least 2, got {num_fcs}.")
        if in_channels is None:
            in_channels = embed_dims
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.pre_norm = nn.LayerNorm(in_channels) if pre_norm else None

        self.layers = nn.Sequential()
        hidden_in = in_channels
        for _ in range(num_fcs - 1):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(hidden_in, feedforward_channels),
                    nn.ReLU(inplace=True),
                    nn.Dropout(ffn_drop),
                )
            )
            hidden_in = feedforward_channels
        self.layers.append(nn.Linear(feedforward_channels, embed_dims))
        self.layers.append(nn.Dropout(ffn_drop))

        self.add_identity = add_identity
        self.identity_fc: nn.Module | None = None
        if add_identity:
            self.identity_fc = (
                nn.Identity() if in_channels == embed_dims else nn.Linear(in_channels, embed_dims)
            )

    def forward(
        self,
        x: Float[torch.Tensor, "batch_size num_anchors in_channels"],
        identity: Float[torch.Tensor, "batch_size num_anchors in_channels"] | None = None,
    ) -> Float[torch.Tensor, "batch_size num_anchors embed_dims"]:
        """Run the block.

        Args:
            x: Input features.
            identity: Features of the residual, ``x`` (after the pre-norm) when ``None``.

        Returns:
            The output features.
        """
        if self.pre_norm is not None:
            x = self.pre_norm(x)
        out = self.layers(x)
        if self.identity_fc is None:
            return out
        if identity is None:
            identity = x
        return self.identity_fc(identity) + out


class DeformableFeatureAggregation(nn.Module):
    """Sample the image feature pyramids at the key points of every anchor and fuse them.

    Every anchor owns the key points of its ``kps_generator``. Each key point is projected
    into every camera, sampled from every level of the camera's pyramid, and the samples are
    summed with weights predicted from the instance feature, grouped along the channels. The
    fused feature is projected and combined with the instance feature by ``residual_mode``.

    Two sampling paths give the same result: the CUDA kernel of
    :mod:`autoware_ml.ops.deformable_aggregation`, taken when the pyramids come flattened as a
    :class:`FlattenedFeatureMaps`, and ``grid_sample`` per level otherwise.
    """

    def __init__(
        self,
        kps_generator: SparseBox3DKeyPointsGenerator,
        embed_dims: int = 256,
        num_groups: int = 8,
        num_levels: int = 4,
        num_cams: int = 6,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        use_camera_embed: bool = False,
        residual_mode: Literal["add", "cat"] = "add",
    ) -> None:
        """Initialize the aggregation.

        Args:
            kps_generator: Key points of every anchor; its ``embed_dims`` must match.
            embed_dims: Width of the instance features and of the pyramid channels.
            num_groups: Channel groups sharing one weight.
            num_levels: Levels of every pyramid.
            num_cams: Cameras of every frame.
            proj_drop: Dropout probability on the output projection.
            attn_drop: Probability of dropping a (camera, point) sample in training.
            use_camera_embed: Whether the weights also depend on an embedding of every
                camera's projection matrix; the weights are then predicted per camera.
            residual_mode: ``add`` sums the fused and the instance features, ``cat``
                concatenates them (the output is then ``2 * embed_dims`` wide).

        Raises:
            ValueError: If the groups do not divide the channels, the key points generator
                has another width, or the residual mode is unknown.
        """
        super().__init__()
        if embed_dims % num_groups != 0:
            raise ValueError(
                f"embed_dims must be divisible by num_groups, got {embed_dims} and {num_groups}."
            )
        if kps_generator.embed_dims != embed_dims:
            raise ValueError(
                f"kps_generator has embed_dims {kps_generator.embed_dims}, expected {embed_dims}."
            )
        if residual_mode not in ("add", "cat"):
            raise ValueError(f"residual_mode must be 'add' or 'cat', got {residual_mode!r}.")
        self.embed_dims = embed_dims
        self.num_groups = num_groups
        self.group_dims = embed_dims // num_groups
        self.num_levels = num_levels
        self.num_cams = num_cams
        self.attn_drop = attn_drop
        self.residual_mode = residual_mode
        self.proj_drop = nn.Dropout(proj_drop)
        self.kps_generator = kps_generator
        self.num_pts = kps_generator.num_pts
        self.output_proj = nn.Linear(embed_dims, embed_dims)

        self.camera_encoder: nn.Sequential | None = None
        if use_camera_embed:
            self.camera_encoder = linear_relu_ln(embed_dims, 1, 2, _CAMERA_EMBED_INPUT_DIMS)
            self.weights_fc = nn.Linear(embed_dims, num_groups * num_levels * self.num_pts)
        else:
            self.weights_fc = nn.Linear(
                embed_dims, num_groups * num_cams * num_levels * self.num_pts
            )
        self.init_weights()

    def init_weights(self) -> None:
        """Start from uniform weights (zeroed ``weights_fc``) and a xavier output projection."""
        nn.init.constant_(self.weights_fc.weight, 0.0)
        nn.init.constant_(self.weights_fc.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

    def forward(
        self,
        instance_feature: Float[torch.Tensor, "batch_size num_anchors embed_dims"],
        anchor: Float[torch.Tensor, "batch_size num_anchors anchor_dims"],
        anchor_embed: Float[torch.Tensor, "batch_size num_anchors embed_dims"],
        feature_maps: FlattenedFeatureMaps
        | Sequence[Float[torch.Tensor, "batch_size num_cams embed_dims height width"]],
        projection_mat: Float[torch.Tensor, "batch_size num_cams 4 4"],
        image_wh: Float[torch.Tensor, "batch_size num_cams 2"],
    ) -> Float[torch.Tensor, "batch_size num_anchors out_dims"]:
        """Aggregate the image features of every anchor.

        Args:
            instance_feature: Instance features of the anchors.
            anchor: Anchors, see :class:`~.box3d.AnchorFieldIndex`.
            anchor_embed: Embedding of the anchors.
            feature_maps: The pyramids of every camera, flattened for the CUDA kernel or one
                tensor per level for the ``grid_sample`` path.
            projection_mat: Lidar-to-image projection of every camera.
            image_wh: Width and height of every camera image, in the pixels the projection
                maps to.

        Returns:
            The fused features, ``embed_dims`` wide for ``add`` and twice that for ``cat``.
        """
        batch_size, num_anchors = instance_feature.shape[:2]
        key_points = self.kps_generator(anchor, instance_feature)
        weights = self._get_weights(instance_feature, anchor_embed, projection_mat)
        points_2d = self.project_points(key_points, projection_mat, image_wh)

        if isinstance(feature_maps, FlattenedFeatureMaps):
            # (batch, cameras, anchors, points, 2) -> (batch, anchors, points, cameras, 2)
            points_2d = points_2d.permute(0, 2, 3, 1, 4).reshape(
                batch_size, num_anchors, self.num_pts, self.num_cams, 2
            )
            # (..., cameras, levels, points, groups) -> (..., points, cameras, levels, groups)
            kernel_weights = (
                weights.permute(0, 1, 4, 2, 3, 5)
                .contiguous()
                .reshape(
                    batch_size,
                    num_anchors,
                    self.num_pts,
                    self.num_cams,
                    self.num_levels,
                    self.num_groups,
                )
            )
            features = deformable_aggregation(
                feature_maps.features,
                feature_maps.spatial_shapes,
                feature_maps.level_start_index,
                points_2d,
                kernel_weights,
            ).reshape(batch_size, num_anchors, self.embed_dims)
        else:
            features = self.feature_sampling(feature_maps, points_2d)
            features = self.multi_view_level_fusion(features, weights)
            features = features.sum(dim=2)  # fuse the key points

        output = self.proj_drop(self.output_proj(features))
        if self.residual_mode == "add":
            return output + instance_feature
        return torch.cat([output, instance_feature], dim=-1)

    def _get_weights(
        self,
        instance_feature: Float[torch.Tensor, "batch_size num_anchors embed_dims"],
        anchor_embed: Float[torch.Tensor, "batch_size num_anchors embed_dims"],
        projection_mat: Float[torch.Tensor, "batch_size num_cams 4 4"],
    ) -> Float[torch.Tensor, "batch_size num_anchors num_cams num_levels num_pts num_groups"]:
        """Predict the aggregation weights, a softmax over the cameras, levels and points.

        In training, ``attn_drop`` zeroes random (camera, point) samples and rescales the rest.
        """
        batch_size, num_anchors = instance_feature.shape[:2]
        feature = instance_feature + anchor_embed
        if self.camera_encoder is not None:
            camera_embed = self.camera_encoder(
                projection_mat[:, :, :_PROJECTION_ROWS].reshape(
                    batch_size, self.num_cams, _CAMERA_EMBED_INPUT_DIMS
                )
            )
            feature = feature[:, :, None] + camera_embed[:, None]

        weights = (
            self.weights_fc(feature)
            .reshape(batch_size, num_anchors, -1, self.num_groups)
            .softmax(dim=-2)
            .reshape(
                batch_size,
                num_anchors,
                self.num_cams,
                self.num_levels,
                self.num_pts,
                self.num_groups,
            )
        )
        if self.training and self.attn_drop > 0:
            mask = torch.rand(
                batch_size,
                num_anchors,
                self.num_cams,
                1,
                self.num_pts,
                1,
                device=weights.device,
                dtype=weights.dtype,
            )
            weights = ((mask > self.attn_drop) * weights) / (1 - self.attn_drop)
        return weights

    @staticmethod
    def project_points(
        key_points: Float[torch.Tensor, "batch_size num_anchors num_pts 3"],
        projection_mat: Float[torch.Tensor, "batch_size num_cams 4 4"],
        image_wh: Float[torch.Tensor, "batch_size num_cams 2"] | None = None,
    ) -> Float[torch.Tensor, "batch_size num_cams num_anchors num_pts 2"]:
        """Project the key points into every camera.

        Args:
            key_points: Key points in the frame of the projection matrices.
            projection_mat: Lidar-to-image projection of every camera.
            image_wh: Width and height of every image; when given, the pixels are normalized
                to ``[0, 1]`` of the image.

        Returns:
            The ``(x, y)`` of every key point in every camera. A point behind a camera lands
            far outside its image, since the depth is clamped to a small positive value.
        """
        pts_extend = torch.cat([key_points, torch.ones_like(key_points[..., :1])], dim=-1)
        points_2d = torch.matmul(
            projection_mat[:, :, None, None], pts_extend[:, None, ..., None]
        ).squeeze(-1)
        points_2d = points_2d[..., :2] / torch.clamp(points_2d[..., 2:3], min=1e-5)
        if image_wh is not None:
            points_2d = points_2d / image_wh[:, :, None, None]
        return points_2d

    @staticmethod
    def feature_sampling(
        feature_maps: Sequence[Float[torch.Tensor, "batch_size num_cams embed_dims height width"]],
        points_2d: Float[torch.Tensor, "batch_size num_cams num_anchors num_pts 2"],
    ) -> Float[torch.Tensor, "batch_size num_anchors num_cams num_levels num_pts embed_dims"]:
        """Sample every level of every camera at the projected key points with ``grid_sample``.

        A point on or outside the image border samples zero, as in the CUDA kernel.

        Args:
            feature_maps: One tensor per level.
            points_2d: Key points in ``[0, 1]`` of every image.

        Returns:
            The sampled features.
        """
        num_levels = len(feature_maps)
        batch_size, num_cams, num_anchors, num_pts = points_2d.shape[:4]

        inside = torch.logical_and((points_2d > 0).all(dim=-1), (points_2d < 1).all(dim=-1))
        grid = (points_2d * 2 - 1).flatten(end_dim=1)
        features = []
        for feature_map in feature_maps:
            features.append(
                F.grid_sample(
                    feature_map.flatten(end_dim=1),
                    grid,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False,
                )
            )
        # (batch * cameras, levels, channels, anchors, points)
        sampled = torch.stack(features, dim=1)
        sampled = sampled.reshape(
            batch_size, num_cams, num_levels, -1, num_anchors, num_pts
        ).permute(0, 4, 1, 2, 5, 3)
        inside = inside.permute(0, 2, 1, 3)[:, :, :, None, :, None]
        return sampled * inside.to(sampled.dtype)

    def multi_view_level_fusion(
        self,
        features: Float[
            torch.Tensor, "batch_size num_anchors num_cams num_levels num_pts embed_dims"
        ],
        weights: Float[
            torch.Tensor, "batch_size num_anchors num_cams num_levels num_pts num_groups"
        ],
    ) -> Float[torch.Tensor, "batch_size num_anchors num_pts embed_dims"]:
        """Sum the samples over the cameras and the levels with the group weights.

        Args:
            features: Sampled features.
            weights: Aggregation weights.

        Returns:
            The fused features of every key point.
        """
        batch_size, num_anchors = weights.shape[:2]
        features = weights[..., None] * features.reshape(
            features.shape[:-1] + (self.num_groups, self.group_dims)
        )
        features = features.sum(dim=2).sum(dim=2)
        return features.reshape(batch_size, num_anchors, self.num_pts, self.embed_dims)


class DenseDepthNet(nn.Module):
    """Auxiliary dense depth head on the first levels of the image pyramids.

    One 1x1 convolution per level predicts a log depth, scaled by the focal length of the
    camera relative to ``equal_focal``. The head supervises the image features in training
    only and plays no part in the detections.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_depth_layers: int = 1,
        equal_focal: float = 100.0,
        max_depth: float = 60.0,
        loss_weight: float = 1.0,
    ) -> None:
        """Initialize the depth head.

        Args:
            embed_dims: Channels of the pyramid levels.
            num_depth_layers: Number of levels, from the finest, with a depth prediction.
            equal_focal: Focal length the predicted depth is expressed at.
            max_depth: Largest depth the loss supervises.
            loss_weight: Weight of the depth loss.
        """
        super().__init__()
        self.embed_dims = embed_dims
        self.equal_focal = equal_focal
        self.num_depth_layers = num_depth_layers
        self.max_depth = max_depth
        self.loss_weight = loss_weight
        self.depth_layers = nn.ModuleList(
            [nn.Conv2d(embed_dims, 1, kernel_size=1) for _ in range(num_depth_layers)]
        )

    def forward(
        self,
        feature_maps: Sequence[Float[torch.Tensor, "batch_size num_cams embed_dims height width"]],
        focal: Float[torch.Tensor, "batch_size num_cams"] | None = None,
    ) -> list[Float[torch.Tensor, "batch_size num_cams height width"]]:
        """Predict the depth of every pixel of the first ``num_depth_layers`` levels.

        Args:
            feature_maps: One tensor per level, from the finest.
            focal: Focal length of every camera, ``equal_focal`` for all when ``None``.

        Returns:
            One depth map per predicted level.
        """
        depths = []
        for depth_layer, feature_map in zip(self.depth_layers, feature_maps):
            batch_size, num_cams = feature_map.shape[:2]
            depth = depth_layer(feature_map.flatten(end_dim=1).float()).exp()
            depth = depth.reshape(batch_size, num_cams, depth.shape[-2], depth.shape[-1])
            if focal is not None:
                depth = depth * (focal / self.equal_focal)[:, :, None, None]
            depths.append(depth)
        return depths

    def loss(
        self,
        depth_preds: Sequence[Float[torch.Tensor, "batch_size num_cams height width"]],
        gt_depths: Sequence[Float[torch.Tensor, "batch_size num_cams height width"]],
    ) -> Float[torch.Tensor, ""]:
        """L1 depth loss over the pixels with a ground truth depth.

        Every level is averaged over its supervised pixels and the number of levels, so the
        levels weigh alike whatever their resolution.

        Args:
            depth_preds: Predicted depth of every level.
            gt_depths: Ground truth depth of every level, non-positive where unknown.

        Returns:
            The weighted loss.
        """
        loss = depth_preds[0].new_zeros(())
        for pred, gt in zip(depth_preds, gt_depths):
            pred = pred.reshape(-1).float()
            gt = gt.reshape(-1).float()
            fg_mask = torch.logical_and(gt > 0.0, torch.logical_not(torch.isnan(pred)))
            gt = gt[fg_mask]
            pred = torch.clip(pred[fg_mask], 0.0, self.max_depth)
            error = torch.abs(pred - gt).sum()
            loss = loss + error / max(1.0, len(gt) * len(depth_preds)) * self.loss_weight
        return loss
