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

from torch.nn import nn


class BEVFusionLidarExportWrapper(nn.Module):
    """Wrap the lidar-only BEVFusion main body export.

    Used when the image branch is disabled. The wrapper exposes the same
    single-sample tensor interface as the camera-lidar main body without the
    image inputs.
    """

    def __init__(self, model: BEVFusionDetectionModel) -> None:
        """Initialize the lidar export wrapper.

        Args:
            model: BEVFusion model instance.
        """
        super().__init__()
        self.model = model

    def forward(
        self,
        voxels: Float32[torch.Tensor, "num_voxels max_num_points num_point_features"],
        coors: Int32[torch.Tensor, "num_voxels 3"],
        num_points_per_voxel: Int32[torch.Tensor, " num_voxels"],
    ) -> tuple[
        Float32[torch.Tensor, "num_box_code num_proposals"],
        Float32[torch.Tensor, " num_proposals"],
        Int64[torch.Tensor, " num_proposals"],
    ]:
        """Run export-time inference on lidar voxel inputs.

        Args:
            voxels: Voxel features.
            coors: Voxel coordinates in ``(z, y, x)`` order without batch column.
            num_points_per_voxel: Number of points in each voxel.

        Returns:
            Tuple of ``bbox_pred``, ``score``, and ``label_pred``.
        """
        outputs = self.model._forward_with_batch_size(
            voxels=voxels,
            num_points=num_points_per_voxel,
            voxel_coords=BEVFusionLidar.runtime_coors_to_voxel_coords(coors),
            batch_size=1,
        )
        return export_detection_outputs(self.model.bbox_head, outputs)
