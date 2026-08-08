from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from autoware_ml.ops.voxelization.voxelization import VoxelsData


class MultiTaskBatchFeatures(BaseModel):
    """Named tuple to represent the data features for a multi-task model."""

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    voxels_data: VoxelsData | None

    # TODO(Kok Seang): Add input features for 3D segmentation model.

    # TODO(Kok Seang): Add input features for 2D detection/segmentation model.
