from typing import Sequence, Callable

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from autoware_ml.dataclasses.models.detection3d.head_outputs import (
    Detection3DHeadOutputs,
)
from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.dataclasses.models.model_outputs import ModelOutputs
from autoware_ml.metrics.base import MetricSuite
from autoware_ml.preprocessing.data_preprocessor import DataPreprocessor
from autoware_ml.models.detection3d.main_modules.sparse4d_modules.sparse4d_camera import (
    Sparse4DCamera,
)
from autoware_ml.models.module_base_model import LogDictConfigs, ModuleBaseModel
from autoware_ml.models.detection3d.heads.sparse4d.sparse4d_head import Sparse4DHead


class Sparse4DDetectionModel(ModuleBaseModel):
    """ """

    def __init__(
        self,
        data_preprocessor: DataPreprocessor,
        sparse4d_camera_network: Sparse4DCamera | None,
        bbox_head: Sparse4DHead,
        log_dict_configs: LogDictConfigs,
        optimizer: Callable[..., Optimizer] | None = None,
        scheduler: Callable[[Optimizer], LRScheduler] | None = None,
        metrics: Sequence[MetricSuite] | None = None,
    ) -> None:
        """
        data_preprocessor: Preprocessor for the model inputs.
        sparse4d_camera_network: Camera-only Sparse4D main body. The outputs are camera-only
            pyramid features after image neck.
        image_backbone: Image backbone to extract image features.
        image_neck: Image neck features to extract pyramid-like image features.
        bbox_head: Detection head.
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
        self.sparse4d_camera_network = sparse4d_camera_network
        self.bbox_head = bbox_head

    def forward(self, multi_task_batch_inputs: ModelBatchInputs) -> ModelOutputs:
        """Run the detector on voxelized lidar inputs.

        Args:
            multi_task_batch_inputs: ModelBatchInputs containing the voxelized lidar inputs.

        Returns:
            Detection head outputs.
        """

        detection_head_outputs = self._forward_with_batch_size(multi_task_batch_inputs)
        return ModelOutputs(
            detection3d_head_outputs=Detection3DHeadOutputs(
                center_head_outputs=None, transfusion_head_outputs=detection_head_outputs
            )
        )
