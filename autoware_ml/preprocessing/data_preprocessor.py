from typing import Sequence

from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.dataclasses.batch.sample_batch import ModelGTBatch
from autoware_ml.preprocessing.data_preprocessor_modules import DataPreprocessorModule 


class DataPreprocessor:
    """Class for runtime preprocessing of multi-task data.

    This class is responsible for applying runtime preprocessing to the input data before it is fed into the model. It can be used to perform any necessary transformations or augmentations on the input data.

    Args:
        preprocessor_modules: A sequence of nn.Module instances that perform preprocessing
            on the input batch.
    """

    def __init__(self, preprocessor_modules: Sequence[DataPreprocessorModule]) -> None:
        self.preprocessor_modules = preprocessor_modules

    def __call__(self, batch: ModelGTBatch, *, is_training: bool) -> ModelBatchInputs:
        """Apply runtime preprocessing to the input batch.

        Args:
            batch (ModelGTBatch): The input batch of data to be preprocessed.
            is_training (bool): Set True if DataPreprocessor is run in the training mode.

        Returns:
            ModelBatchInputs: The batch of data after running the list of preprocessor_modules.
        """
        batch_inputs = ModelBatchInputs.from_gt_batch(batch)
        for layer in self.preprocessor_modules:
            batch_inputs = layer(batch_inputs, is_training=is_training)
        return batch_inputs
    
