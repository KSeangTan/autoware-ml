"""
Modules to save raw outputs from multi-task models.
"""

from pydantic import BaseModel, ConfigDict
from autoware_ml.models.detection3d.dataclasses.outputs import Detection3DOutputs


class MultiTaskOutputs(BaseModel):
    """
    Dataclass to save predictions from multi-task models.

    Attributes:
        detection3d_outputs: Raw outputs from a 3D detection task.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    detection3d_outputs: Detection3DOutputs | None

    # TODO (Kok Seang): Add outputs for other tasks in the future.
