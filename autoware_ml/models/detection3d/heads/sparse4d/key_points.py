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

"""Key point generation for Sparse4D 3D box anchors."""

from __future__ import annotations

from collections.abc import Sequence

from jaxtyping import Float32
import torch
import torch.nn as nn

from autoware_ml.models.detection3d.heads.sparse4d.anchor_layout import AnchorIndex

X = AnchorIndex.X
Y = AnchorIndex.Y
Z = AnchorIndex.Z
LOG_L = AnchorIndex.LOG_L
LOG_W = AnchorIndex.LOG_W
LOG_H = AnchorIndex.LOG_H
SIN_YAW = AnchorIndex.SIN_YAW
COS_YAW = AnchorIndex.COS_YAW
VX = AnchorIndex.VX


class SparseBox3DKeyPointsGenerator(nn.Module):
    """Generate sampling key points from 3D box anchors.

    Each anchor yields a fixed set of key points defined by ``fix_scale`` in the box frame plus an
    optional set of learnable offsets predicted from the instance feature. The key points are
    scaled by the box size, rotated by the box yaw and translated to the box center. When temporal
    metadata is given, the key points are additionally projected into each temporal frame using
    the anchor velocity and the ego transforms.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_learnable_pts: int = 0,
        fix_scale: Sequence[Sequence[float]] | None = None,
    ) -> None:
        """Initialize the key point generator.

        Args:
            embed_dims: Dimension of the instance feature used to predict learnable key points.
            num_learnable_pts: Number of learnable key points per anchor.
            fix_scale: Fixed key point offsets in the normalized box frame, one ``(x, y, z)``
                triple per point in ``[-0.5, 0.5]``. Defaults to the box center only.
        """
        super().__init__()
        self.embed_dims = embed_dims
        self.num_learnable_pts = num_learnable_pts
        if fix_scale is None:
            fix_scale = ((0.0, 0.0, 0.0),)
        self.register_buffer(
            "fix_scale", torch.tensor(fix_scale, dtype=torch.float32), persistent=False
        )
        self.num_pts = len(fix_scale) + num_learnable_pts
        if num_learnable_pts > 0:
            self.learnable_fc = nn.Linear(self.embed_dims, num_learnable_pts * 3)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize the learnable key point projection with Xavier uniform weights."""
        if self.num_learnable_pts > 0:
            nn.init.xavier_uniform_(self.learnable_fc.weight)
            nn.init.zeros_(self.learnable_fc.bias)

    def forward(
        self,
        anchor: Float32[torch.Tensor, "batch_size num_anchor anchor_dims"],
        instance_feature: Float32[torch.Tensor, "batch_size num_anchor embed_dims"] | None = None,
        T_cur2temp_list: Sequence[Float32[torch.Tensor, "batch_size 4 4"]] | None = None,
        cur_timestamp: Float32[torch.Tensor, " batch_size"] | None = None,
        temp_timestamps: Sequence[Float32[torch.Tensor, " batch_size"]] | None = None,
    ) -> (
        Float32[torch.Tensor, "batch_size num_anchor num_pts 3"]
        | tuple[
            Float32[torch.Tensor, "batch_size num_anchor num_pts 3"],
            list[Float32[torch.Tensor, "batch_size num_anchor num_pts 3"]],
        ]
    ):
        """Generate key points for every anchor.

        Args:
            anchor: Anchor boxes in the current ego frame.
            instance_feature: Instance features used to predict learnable key points. Ignored when
                ``num_learnable_pts`` is zero.
            T_cur2temp_list: Transforms from the current ego frame to each temporal ego frame.
            cur_timestamp: Timestamp of the current frame.
            temp_timestamps: Timestamps of each temporal frame.

        Returns:
            Key points in the current frame. When all temporal arguments are given and non-empty,
            a tuple of the current key points and the list of key points projected into each
            temporal frame.
        """
        batch_size, num_anchor = anchor.shape[:2]

        # Step 1: Broadcast the fixed normalized offsets to every anchor. Each offset lives in the
        # unit box frame, so a value of 0.5 lands on a box face and 0 lands on the box center.
        # scale: (batch_size, num_anchor, num_fix_pts, 3)
        scale = self.fix_scale[None, None].tile([batch_size, num_anchor, 1, 1])

        # Step 2: Predict extra normalized offsets from the instance feature. The sigmoid keeps
        # each coordinate in (0, 1) and the shift recenters it to (-0.5, 0.5), so learnable points
        # stay inside the box like the fixed ones.
        # learnable_scale: (batch_size, num_anchor, num_learnable_pts, 3)
        # scale: (batch_size, num_anchor, num_pts, 3)
        if self.num_learnable_pts > 0 and instance_feature is not None:
            learnable_scale = (
                self.learnable_fc(instance_feature)
                .reshape(batch_size, num_anchor, self.num_learnable_pts, 3)
                .sigmoid()
                - 0.5
            )
            scale = torch.cat([scale, learnable_scale], dim=-2)

        # Step 3: Scale the normalized offsets by the box size. The anchor stores log sizes, so
        # exp() recovers the metric width, length and height.
        # key_points: (batch_size, num_anchor, num_pts, 3)
        key_points = scale * anchor[..., None, [LOG_L, LOG_W, LOG_H]].exp()

        # Step 4: Rotate the offsets by the box yaw about the z axis. The anchor stores the yaw as
        # (sin, cos), which fills a standard 2D rotation in the top-left block of a 3x3 matrix.
        # rotation_mat: (batch_size, num_anchor, 3, 3)
        # key_points: (batch_size, num_anchor, num_pts, 3)
        rotation_mat = anchor.new_zeros([batch_size, num_anchor, 3, 3])
        rotation_mat[:, :, 0, 0] = anchor[:, :, COS_YAW]
        rotation_mat[:, :, 0, 1] = -anchor[:, :, SIN_YAW]
        rotation_mat[:, :, 1, 0] = anchor[:, :, SIN_YAW]
        rotation_mat[:, :, 1, 1] = anchor[:, :, COS_YAW]
        rotation_mat[:, :, 2, 2] = 1
        key_points = torch.matmul(rotation_mat[:, :, None], key_points[..., None]).squeeze(-1)

        # Step 5: Translate the rotated offsets to the box center, giving key points in the
        # current ego frame.
        # key_points: (batch_size, num_anchor, num_pts, 3)
        key_points = key_points + anchor[..., None, [X, Y, Z]]

        # Step 6: Without temporal metadata there is nothing to project, so return the current
        # frame key points alone.
        if (
            cur_timestamp is None
            or temp_timestamps is None
            or T_cur2temp_list is None
            or len(temp_timestamps) == 0
        ):
            return key_points

        # Step 7: Project the key points into each temporal frame. The anchor velocity moves the
        # points back to where the object was at the temporal timestamp, then the ego transform
        # maps them from the current ego frame into that frame's ego coordinates.
        # velocity: (batch_size, num_anchor, vel_dims)
        temp_key_points_list = []
        velocity = anchor[..., VX:]
        for T_cur2temp, temp_timestamp in zip(T_cur2temp_list, temp_timestamps):
            # Step 7a: Undo the object motion over the time gap using constant velocity.
            # time_interval: (batch_size,)
            # translation: (batch_size, num_anchor, vel_dims)
            # temp_key_points: (batch_size, num_anchor, num_pts, 3)
            time_interval = cur_timestamp - temp_timestamp
            translation = velocity * time_interval.to(dtype=velocity.dtype)[:, None, None]
            temp_key_points = key_points - translation[:, :, None]

            # Step 7b: Apply the rigid ego transform in homogeneous coordinates. Only the top
            # three rows are needed since the output is a 3D point.
            # T_cur2temp: (batch_size, 4, 4)
            # homogeneous_key_points: (batch_size, num_anchor, num_pts, 4)
            # temp_key_points: (batch_size, num_anchor, num_pts, 3, 1) -> squeezed to
            # (batch_size, num_anchor, num_pts, 3)
            T_cur2temp = T_cur2temp.to(dtype=key_points.dtype)
            homogeneous_key_points = torch.cat(
                [temp_key_points, torch.ones_like(temp_key_points[..., :1])], dim=-1
            )
            temp_key_points = T_cur2temp[:, None, None, :3] @ homogeneous_key_points.unsqueeze(-1)
            temp_key_points_list.append(temp_key_points.squeeze(-1))
        return key_points, temp_key_points_list

    @staticmethod
    def anchor_projection(
        anchor: Float32[torch.Tensor, "batch_size num_anchor anchor_dims"],
        T_src2dst_list: Sequence[Float32[torch.Tensor, "batch_size 4 4"]],
        src_timestamp: Float32[torch.Tensor, " batch_size"] | None = None,
        dst_timestamps: Sequence[Float32[torch.Tensor, " batch_size"]] | None = None,
    ) -> list[Float32[torch.Tensor, "batch_size num_anchor anchor_dims"]]:
        """Project anchors from the source ego frame into each destination ego frame.

        The box center is propagated with the anchor velocity over the time gap when timestamps
        are given, then the center, yaw and velocity are rotated and translated into the
        destination frame. Box sizes are unchanged.

        Args:
            anchor: Anchor boxes in the source ego frame.
            T_src2dst_list: Transforms from the source ego frame to each destination ego frame.
            src_timestamp: Timestamp of the source frame.
            dst_timestamps: Timestamps of each destination frame.

        Returns:
            List of projected anchors, one per destination transform.
        """
        dst_anchors = []

        # Step 1: Slice the anchor velocity once since every destination frame reuses it. The
        # velocity may be 2D or 3D depending on the anchor layout, so its width is read here.
        # velocity: (batch_size, num_anchor, vel_dim)
        velocity = anchor[..., VX:]
        vel_dim = velocity.shape[-1]
        for i, T_src2dst in enumerate(T_src2dst_list):
            # Step 2: Start from a copy so the source anchor is untouched, and add an anchor axis
            # to the transform so it broadcasts against every anchor in the batch.
            # dst_anchor: (batch_size, num_anchor, anchor_dims)
            # T_src2dst: (batch_size, 1, 4, 4)
            dst_anchor = anchor.clone()
            T_src2dst = T_src2dst.to(dtype=anchor.dtype).unsqueeze(dim=1)

            # Step 3: Compensate object motion. With timestamps, the center is moved back along
            # the anchor velocity by the time gap so it sits where the object was at the
            # destination timestamp. Without timestamps only ego motion is applied.
            # center: (batch_size, num_anchor, 3)
            # time_interval: (batch_size,)
            # translation: (batch_size, num_anchor, vel_dim)
            center = dst_anchor[..., [X, Y, Z]]
            if src_timestamp is not None and dst_timestamps is not None:
                time_interval = (src_timestamp - dst_timestamps[i]).to(dtype=velocity.dtype)
                translation = velocity * time_interval[:, None, None]
                center = center - translation

            # Step 4: Compensate ego motion on the center with the full rigid transform, rotating
            # by the 3x3 block and shifting by the translation column.
            # dst_anchor[..., [X, Y, Z]]: (batch_size, num_anchor, 3)
            dst_anchor[..., [X, Y, Z]] = (
                torch.matmul(T_src2dst[..., :3, :3], center[..., None]).squeeze(dim=-1)
                + T_src2dst[..., :3, 3]
            )

            # Step 5: Rotate the yaw. The (cos, sin) pair is a unit vector in the BEV plane, so
            # the 2x2 rotation block turns it by the ego yaw change. Translation does not apply.
            # dst_anchor[..., [COS_YAW, SIN_YAW]]: (batch_size, num_anchor, 2)
            dst_anchor[..., [COS_YAW, SIN_YAW]] = torch.matmul(
                T_src2dst[..., :2, :2], dst_anchor[..., [COS_YAW, SIN_YAW], None]
            ).squeeze(-1)

            # Step 6: Rotate the velocity. Velocity is a direction vector, so only the rotation
            # block applies and the translation column is skipped. Box sizes are frame invariant
            # and stay untouched.
            # dst_anchor[..., VX:]: (batch_size, num_anchor, vel_dim)
            dst_anchor[..., VX:] = torch.matmul(
                T_src2dst[..., :vel_dim, :vel_dim], velocity[..., None]
            ).squeeze(-1)

            dst_anchors.append(dst_anchor)
        return dst_anchors

    @staticmethod
    def distance(
        anchor: Float32[torch.Tensor, "batch_size num_anchor anchor_dims"],
    ) -> Float32[torch.Tensor, "batch_size num_anchor"]:
        """Compute the BEV distance of each anchor center from the ego origin.

        Args:
            anchor: Anchor boxes.

        Returns:
            Euclidean norm of the ``(x, y)`` center of each anchor.
        """
        return torch.norm(anchor[..., :2], p=2, dim=-1)
