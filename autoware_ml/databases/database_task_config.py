from typing import Any, Mapping, Sequence
from types import MappingProxyType

from pydantic import BaseModel, ConfigDict, model_validator

from autoware_ml.types.tasks import TaskType


class DatabaseTaskConfig(BaseModel):
    """
    DatabaseTaskConfig is a Pydantic model that represents the configuration for a database task.

    Attributes:
        task_type (TaskType): The type of the task (e.g., detection3d, segmentation3d).
        label_names: Sequence of label names for the task.
        ignore_label_index: The index to use for ignored labels.
        label_remapper: Mapping to remap label names, if needed.
    """

    model_config = ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True)

    task_type: TaskType
    label_names: Sequence[str]
    ignore_label_index: int
    label_remapper: MappingProxyType[str, str] | None

    @model_validator(mode="before")
    @classmethod
    def _wrap_label_remapper(cls, data: Mapping[str, Any]) -> Any:
        """
        Wrap the label remapper in a MappingProxyType before validation.

        Hydra composes the label remapper as a plain dict, so it would not validate against
        the MappingProxyType field.

        Args:
            data: Raw input of the model, usually a mapping of field name to value.

        Returns:
            Any: The input with its label remapper wrapped in a MappingProxyType.
        """
        if not isinstance(data, Mapping):
            return data

        label_remapper = data.get("label_remapper", None)
        if label_remapper is None:
            return data

        if isinstance(label_remapper, Mapping) and not isinstance(label_remapper, MappingProxyType):
            label_remapper = MappingProxyType(label_remapper)
            data["label_remapper"] = label_remapper
            return data

        return data

    @property
    def hash_repr(self) -> str:
        """
        Generate a hash representation of the DatabaseTaskConfig.

        Returns:
            str: A string representation of the DatabaseTaskConfig for hashing.
        """
        return f"{self.task_type.value}_{self.label_names}_{self.ignore_label_index}_{self.label_remapper}"
