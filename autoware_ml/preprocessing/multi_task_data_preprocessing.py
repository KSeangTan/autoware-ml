from typing import Sequence

from torch import nn

from autoware_ml.datamodule.multi_task.dataclasses.multi_task_features import MultiTaskFeatures
from autoware_ml.datamodule.multi_task.dataclasses.multi_task_samples import (
    MultiTaskGTBatch,
)


class MultiTaskDataPreprocessing:
    """Class for runtime preprocessing of multi-task data.

    This class is responsible for applying runtime preprocessing to the input data before it is fed into the model. It can be used to perform any necessary transformations or augmentations on the input data.

    Args:
        preprocessing_modules: A sequence of nn.Module instances that perform preprocessing on the input batch.
    """

    def __init__(self, preprocessing_modules: Sequence[nn.Module]) -> None:
        self.preprocessing_modules = preprocessing_modules

    def __call__(self, multi_task_gt_batch: MultiTaskGTBatch) -> MultiTaskFeatures:
        """Apply runtime preprocessing to the input batch.

        Args:
            multi_task_gt_batch: The input batch of data to be preprocessed.

        Returns:
            The preprocessed batch of data.
        """
        # Build a MultiTaskFeatures instance from the input batch
        multi_task_features = MultiTaskFeatures(
            multi_task_gt_batch=multi_task_gt_batch,
            detection3d_features=None,  # Placeholder for 3D detection features
        )
        for module in self.preprocessing_modules:
            multi_task_features = module(multi_task_features)
        return multi_task_features
