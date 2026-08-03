from __future__ import annotations

from typing import NamedTuple

from autoware_ml.datamodule.multi_task.dataclasses.multi_task_samples import MultiTaskGTBatch
from autoware_ml.ops.voxelization.voxelization import VoxelsData


class Detection3DFeatures(NamedTuple):
    """Named tuple to represent the features for a 3D detection model."""

    voxels_data: VoxelsData | None


class MultiTaskFeatures(NamedTuple):
    """Named tuple to represent the features for a multi-task model."""

    # Keep the original MultiTaskGTSample for reference.
    multi_task_gt_batch: MultiTaskGTBatch

    # Input features for 3D detection model.
    detection3d_features: Detection3DFeatures | None

    # TODO(Kok Seang): Add input features for 3D segmentation model.

    # TODO(Kok Seang): Add input features for 2D detection/segmentation model.
