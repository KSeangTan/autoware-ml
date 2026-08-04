"""
Modules to save decoded predictions from multi-task models.
"""

from pydantic import BaseModel, ConfigDict
from autoware_ml.models.detection3d.dataclasses.predictions import Detection3DPredictions


class MultiTaskPredictions(BaseModel):
    """
    Dataclass to save decoded predictions from multi-task models.

    Attributes:
      detection_3d_predictions: Decoded predictions from a 3D detection task.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    detection_3d_predictions: Detection3DPredictions | None

    # TODO (Kok Seang): Add predictions for other tasks in the future.
