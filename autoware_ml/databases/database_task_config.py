from typing import Sequence
from types import MappingProxyType

from pydantic import BaseModel

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

    task_type: TaskType
    label_names: Sequence[str]
    ignore_label_index: int
    label_remapper: MappingProxyType[str, str] | None

    @property
    def hash_repr(self) -> str:
        """
        Generate a hash representation of the DatabaseTaskConfig.

        Returns:
            str: A string representation of the DatabaseTaskConfig for hashing.
        """
        return f"{self.task_type.value}_{self.label_names}_{self.ignore_label_index}_{self.label_remapper}"
