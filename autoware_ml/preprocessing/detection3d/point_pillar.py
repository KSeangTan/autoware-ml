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

"""PointPillars preprocessing for Detection3D models."""

from __future__ import annotations

from typing import Sequence

from jaxtyping import Float32
import torch
import torch.nn as nn

from autoware_ml.datamodule.multi_task.dataclasses.multi_task_features import (
    MultiTaskFeatures,
    Detection3DFeatures,
)
from autoware_ml.ops.voxelization.voxelization import hard_voxelize, VoxelsData


class PointPillarPreprocessor(nn.Module):
    """Convert batched point clouds into padded pillars for PointPillars models.

    The preprocessor voxelizes each point cloud using
    :func:`~autoware_ml.ops.voxelization.hard_voxelize`, pads variable-size
    pillars to ``max_num_points``, and packages the tensors expected by
    PointPillars-style detectors.

    Args:
        voxel_size: Voxel size along each axis ``[dx, dy, dz]`` in meters.
        point_cloud_range: Spatial range ``[x_min, y_min, z_min, x_max, y_max, z_max]``
            in meters.
        max_num_points: Maximum number of points kept per pillar.
        max_voxels: Maximum number of pillars retained per sample.
        voxelization_z_order_first: If ``True``, this preprocessor will transpose [x, y, z]
            coordinates to [z, y, x] in coords from voxelization.
            This is used for backward-compatible, and will be removed very soon.
        default_point_channels: Default number of point channels to be used when no points
            are provided in the batch. Default is 4, which corresponds to (x, y, z, intensity).
    """

    # Add class attributes for type checking
    voxel_size: Float32[torch.Tensor, " 3"]
    point_cloud_range: Float32[torch.Tensor, " 6"]

    def __init__(
        self,
        voxel_size: Sequence[float],
        point_cloud_range: Sequence[float],
        max_num_points: int,
        max_voxels: int,
        voxelization_z_order_first: bool = True,
        default_point_channels: int = 4,
    ) -> None:
        super().__init__()
        self.register_buffer("voxel_size", torch.tensor(voxel_size, dtype=torch.float32))
        self.register_buffer(
            "point_cloud_range", torch.tensor(point_cloud_range, dtype=torch.float32)
        )
        self.max_num_points = max_num_points
        self.max_voxels = max_voxels
        self.voxelization_z_order_first = voxelization_z_order_first
        self._default_point_channels = default_point_channels

    def forward(self, multi_task_features: MultiTaskFeatures) -> MultiTaskFeatures:
        """Voxelize batched point clouds and append pillar tensors.

        Args:
            multi_task_features: MultiTaskFeatures instance containing a ``"points"`` key
                with a list of ``(N_i, C)`` point tensors.

        Returns:
            Updated batch dictionary with the following additional keys:

            - ``"voxels"`` - padded pillar features ``(total_pillars, max_num_points, C)``.
            - ``"num_points"`` - per-pillar point counts ``(total_pillars,)``.
            - ``"voxel_coords"`` - pillar coordinates ``(total_pillars, 4)`` in
              ``[batch, z, y, x]`` order, ``dtype=torch.int32``.
        """
        if multi_task_features.multi_task_gt_batch.point_cloud_gt_batch is None:
            raise ValueError("MultiTaskFeatures must contain point cloud data for voxelization.")

        points_list = multi_task_features.multi_task_gt_batch.point_cloud_gt_batch.points
        if not len(points_list):
            voxels_data = VoxelsData(
                voxels=torch.zeros(
                    (0, self.max_num_points, self._default_point_channels),
                    device=self.voxel_size.device,
                ),
                num_points=torch.zeros((0,), device=self.voxel_size.device, dtype=torch.int32),
                coords=torch.zeros((0, 3), device=self.voxel_size.device, dtype=torch.int32),
                batch_indices=torch.zeros((0,), device=self.voxel_size.device, dtype=torch.int32),
            )
            return MultiTaskFeatures(
                multi_task_gt_batch=multi_task_features.multi_task_gt_batch,
                detection3d_features=Detection3DFeatures(voxels_data=voxels_data),
            )

        if len(points_list) != len(
            multi_task_features.multi_task_gt_batch.point_cloud_gt_batch.batch_indices
        ):
            raise ValueError(
                "Length of points list must match length of batch indices in MultiTaskGTBatch."
            )

        device = points_list[0].device
        voxel_size = self.voxel_size.to(device=device)
        point_cloud_range = self.point_cloud_range.to(device=device)
        points = multi_task_features.multi_task_gt_batch.point_cloud_gt_batch.points
        points_batch_indices = (
            multi_task_features.multi_task_gt_batch.point_cloud_gt_batch.batch_indices
        )

        voxels_data = hard_voxelize(
            points,
            points_batch_indices=points_batch_indices,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            max_num_points=self.max_num_points,
            max_voxels=self.max_voxels,
        )

        # Handle the case where no voxels are generated
        if not len(voxels_data.voxels):
            voxels_data = VoxelsData(
                voxels=torch.zeros(
                    (0, self.max_num_points, points.shape[1]),
                    device=device,
                ),
                num_points=torch.zeros((0,), device=device, dtype=torch.int32),
                coords=torch.zeros((0, 3), device=device, dtype=torch.int32),
                batch_indices=torch.zeros((0,), device=device, dtype=torch.int32),
            )
            return MultiTaskFeatures(
                multi_task_gt_batch=multi_task_features.multi_task_gt_batch,
                detection3d_features=Detection3DFeatures(
                    voxels_data=voxels_data,
                ),
            )

        # TODO (KokSeang): Remove this backward compatibility code in the future
        if self.voxelization_z_order_first:
            coords = voxels_data.coords[:, [2, 1, 0]].contiguous()
            # Re-create the VoxelsData with the updated coords
            voxels_data = VoxelsData(
                voxels=voxels_data.voxels,
                num_points=voxels_data.num_points,
                coords=coords,
                batch_indices=voxels_data.batch_indices,
            )

        return MultiTaskFeatures(
            multi_task_gt_batch=multi_task_features.multi_task_gt_batch,
            detection3d_features=Detection3DFeatures(
                voxels_data=voxels_data,
            ),
        )
