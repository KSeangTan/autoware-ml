# Copyright 2023 OpenMMLab.
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

"""Native CenterPoint lidar detector wrapper.

This module provides the task-level training, inference, and export wrapper
around the reusable PointPillars and CenterPoint detection components.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType
from typing import Any

from jaxtyping import Float32
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from autoware_ml.datamodule.multi_task.dataclasses.multi_task_features import MultiTaskFeatures
from autoware_ml.metrics.base import MetricSuite
from autoware_ml.metrics.detection3d.eval_output import detection_eval_output
from autoware_ml.models.multi_task_base import LogDictConfigs, MultiTaskBaseModel
from autoware_ml.models.dataclasses.multi_task_outputs import MultiTaskOutputs
from autoware_ml.models.dataclasses.multi_task_predictions import MultiTaskPredictions
from autoware_ml.models.detection3d.dataclasses.outputs import Detection3DOutputs
from autoware_ml.models.detection3d.encoders.pillar.pillar_feature_net import PillarFeatureNet
from autoware_ml.models.detection3d.encoders.pillar.point_pillar_scatter import PointPillarsScatter
from autoware_ml.models.detection3d.heads.centerhead import CenterHead
from autoware_ml.preprocessing.multi_task_data_preprocessor import MultiTaskDataPreprocessor


class CenterPointDetectionModel(MultiTaskBaseModel):
    """Compose a CenterPoint detector from reusable lidar detection modules.

    The wrapper wires together pillar encoding, BEV feature extraction, and the
    CenterPoint dense head inside the shared :class:`BaseModel` interface.
    """

    def __init__(
        self,
        data_preprocessor: MultiTaskDataPreprocessor,
        # TODO(KokSeang): Encoder and middle_encoder should be standardized to a common interface for all voxel encoders
        pts_voxel_encoder: PillarFeatureNet,
        pts_middle_encoder: PointPillarsScatter,
        pts_backbone: torch.nn.Module,
        pts_neck: torch.nn.Module,
        # TODO(KokSeang): bbox_head should be standardized to a common interface for all detection heads
        bbox_head: CenterHead,
        log_dict_configs: LogDictConfigs,
        optimizer: Callable[..., Optimizer] | None = None,
        scheduler: Callable[[Optimizer], LRScheduler] | None = None,
        metrics: Sequence[MetricSuite] | None = None,
    ) -> None:
        """
        Initialize CenterPoint.

        Args:
            data_preprocessor: Multi-task data preprocessor.
            pts_voxel_encoder: Lidar voxel feature encoder.
            pts_middle_encoder: Sparse 3D or pillar-scatter middle encoder.
            pts_backbone: BEV backbone.
            pts_neck: BEV neck.
            bbox_head: CenterPoint dense detection head.
            log_dict_configs: Logging configuration for training and validation.
            optimizer: Optimizer factory.
            scheduler: Scheduler factory.
            metrics: Detection metrics accumulated during validation and test.
        """
        super().__init__(
            data_preprocessor=data_preprocessor,
            optimizer=optimizer,
            scheduler=scheduler,
            metrics=metrics,
            log_dict_configs=log_dict_configs,
        )
        self.pts_voxel_encoder = pts_voxel_encoder
        self.pts_middle_encoder = pts_middle_encoder
        self.pts_backbone = pts_backbone
        self.pts_neck = pts_neck
        self.bbox_head = bbox_head

    def build_eval_output(self, batch: Mapping[str, Any], outputs: Any) -> dict[str, Any]:
        """Decode detections and pair them with ground truth for metrics."""
        return detection_eval_output(self.bbox_head.predict(outputs), batch)

    def forward(self, multi_task_features: MultiTaskFeatures) -> MultiTaskOutputs:
        """Run the detector on voxelized lidar inputs.

        Args:
            voxels: Voxel features.
            num_points: Number of points in each voxel.
            voxel_coords: Batched voxel coordinates.

        Returns:
            Detection head outputs.
        """
        if multi_task_features.detection3d_features is None:
            raise ValueError(
                "MultiTaskFeatures must contain detection3d_features for CenterPoint forward pass."
            )

        if multi_task_features.detection3d_features.voxels_data is None:
            raise ValueError(
                "MultiTaskFeatures must contain voxels_data for CenterPoint forward pass."
            )

        batch_size = multi_task_features.multi_task_gt_batch.infer_batch_size()
        pillar_features = self.pts_voxel_encoder(multi_task_features)
        bev_features = self.pts_middle_encoder(
            pillar_features=pillar_features,
            coords=multi_task_features.detection3d_features.voxels_data.coords,
            batch_indices=multi_task_features.detection3d_features.voxels_data.batch_indices,
            batch_size=batch_size,
        )
        bev_features = self.pts_backbone(bev_features)
        bev_features = self.pts_neck(bev_features)
        head_outputs = self.bbox_head(bev_features)
        return MultiTaskOutputs(
            detection3d_outputs=Detection3DOutputs(
                center_head_outputs=head_outputs, transfusion_head_outputs=None
            )
        )

    def compute_metrics(
        self, multi_task_features: MultiTaskFeatures, multi_task_outputs: MultiTaskOutputs
    ) -> MappingProxyType[str, Float32[torch.Tensor, " 1"]]:
        """Compute CenterPoint training losses."""
        if multi_task_features.multi_task_gt_batch.detection3d_gt_batch is None:
            raise ValueError(
                "MultiTaskFeatures must contain detection3d_gt_batch for CenterPoint compute_metrics pass."
            )

        if multi_task_outputs.detection3d_outputs is None:
            raise ValueError(
                "MultiTaskOutputs must contain detection3d_outputs for CenterPoint compute_metrics pass."
            )

        gt_bboxes_3d = multi_task_features.multi_task_gt_batch.detection3d_gt_batch.gt_bboxes_3d
        gt_labels_3d = multi_task_features.multi_task_gt_batch.detection3d_gt_batch.gt_labels_3d
        gt_valid_bboxes = (
            multi_task_features.multi_task_gt_batch.detection3d_gt_batch.gt_valid_bboxes
        )

        return self.bbox_head.loss(
            outputs=multi_task_outputs.detection3d_outputs,
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_valid_bboxes=gt_valid_bboxes,
        )

    def decode_outputs(self, outputs: MultiTaskOutputs) -> MultiTaskPredictions:
        """Decode predictions for inference."""
        if outputs.detection3d_outputs is None:
            raise ValueError(
                "MultiTaskOutputs must contain detection3d_outputs for CenterPoint decode_outputs pass."
            )

        detection3d_predictions = self.bbox_head.decode_outputs(outputs=outputs.detection3d_outputs)
        return MultiTaskPredictions(detection3d_predictions=detection3d_predictions)
