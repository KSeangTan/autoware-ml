from abc import abstractmethod
import logging
from pathlib import Path
import time
from typing import Sequence

import polars as pl
from torch.utils.data import Dataset

from autoware_ml.databases.schemas.dataset_schemas import DatasetTableSchema
from autoware_ml.databases.schemas.image_frames import ImageFrameDatasetSchema
from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample, ModelGTBatch
from autoware_ml.transforms.base import PipelineContext, TransformsCompose
from autoware_ml.types.dataset import SplitType

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    """Multi-task dataset interface that can be shared by multiple databases."""

    def __init__(
        self,
        database_root_path: str,
        max_num_3d_gt_bboxes: int,
        split_type: SplitType,
        dataset_records_dataframe: pl.DataFrame | None,
        transforms: TransformsCompose | None,
        camera_order: Sequence[str] | None = None,
        filter_frames_with_camera_order: bool = False,
        filter_all_time_steps: bool = False,
    ) -> None:
        """
        Initialize the multi-task dataset interface.
        Args:
          database_root_path: Root directory of the dataset.
          max_num_3d_gt_bboxes: Maximum number of 3D ground truth bounding boxes in the dataset.
              This is allowed to be 0 if the dataset does not contain any 3D ground truth
              bounding boxes or it does not need to run 3D detection tasks.
          split_type: The split type of the dataset (train, val, test).
          dataset_records_dataframe: Polars DataFrame of dataset records to be used in the
              multi-task dataset. Accept None if the dataset records
              are not available at initialization.
          transforms: Global transforms to be applied to the dataset records.
          camera_order: Ordered camera names expected by the model. Required when
              ``filter_frames_with_camera_order`` is set.
          filter_frames_with_camera_order: Drop frames missing any camera in
              ``camera_order`` (absent or null image path), so downstream image loading always
              sees a complete multiview.
          filter_all_time_steps: When ``filter_frames_with_camera_order`` is set, check every
              time step of ``IMAGE_FRAMES`` (the current frame and all previous frames) instead
              of only the current frame at index 0.
        """
        super().__init__()
        self.database_root_path = Path(database_root_path)
        self.max_num_3d_gt_bboxes = max_num_3d_gt_bboxes
        self.transforms = transforms
        self.split_type = split_type
        self.camera_order = list(camera_order) if camera_order is not None else None
        self.filter_frames_with_camera_order = filter_frames_with_camera_order
        self.filter_all_time_steps = filter_all_time_steps
        if self.filter_frames_with_camera_order and not self.camera_order:
            raise ValueError(
                "camera_order must be provided when filter_frames_with_camera_order is set."
            )
        if self.filter_frames_with_camera_order and dataset_records_dataframe is not None:
            dataset_records_dataframe = self._filter_frames_with_camera_order(
                dataset_records_dataframe
            )
        self.dataset_records_dataframe = dataset_records_dataframe

    def __len__(self) -> int:
        """Return the number of dataset records.

        Returns:
          int: Number of dataset records.
        """
        if self.dataset_records_dataframe is None:
            raise ValueError("Dataset records dataframe is not available.")
        return len(self.dataset_records_dataframe)

    def __getitem__(self, index: int) -> ModelGTSample:
        """Load and transform one dataset sample.

        Args:
            index: Sample index.

        Returns:
            Transformed ModelGTSample instance.
        """
        start_time = time.perf_counter()
        multi_task_gt_sample = self.get_data_sample(index)
        context = PipelineContext(dataset=self, index=index)
        transformed_gt_sample = self.apply_transforms(
            multi_task_gt_sample, self.transforms, context
        )
        return transformed_gt_sample._replace(io_processing_time=time.perf_counter() - start_time)

    def scene_index_groups(self) -> Sequence[Sequence[int]]:
        """Group dataset indices by scene, preserving frame order, for streaming samplers.

        Returns:
            Sequence[Sequence[int]]: One sequence of scene-contiguous dataset indices per scene.
        """
        raise NotImplementedError("Dataset must implement scene_index_groups")

    def assign_dataset_records(self, dataset_records_dataframe: pl.DataFrame) -> None:
        """Assign the dataset records dataframe, filtered by ``camera_order`` when enabled.

        Args:
            dataset_records_dataframe: Polars DataFrame of dataset records.
        """
        if self.filter_frames_with_camera_order:
            dataset_records_dataframe = self._filter_frames_with_camera_order(
                dataset_records_dataframe
            )
        self.dataset_records_dataframe = dataset_records_dataframe

    @staticmethod
    def _build_time_step_has_camera_expression(camera_name: str) -> pl.Expr:
        """Build a boolean expression that is true when one time step holds a usable camera frame.

        The expression is meant to be evaluated inside a ``list.eval`` over ``IMAGE_FRAMES``, so
        ``pl.element()`` is one time step, which is a list of image frame structs. The time step
        holds the camera when any of its structs has the given channel name and a non-null image
        path.

        Args:
            camera_name: Camera channel name to look for in the time step.

        Returns:
            pl.Expr: Boolean expression over one time step.
        """
        channel_name_field = ImageFrameDatasetSchema.image_sensor_channel_name.name
        image_path_field = ImageFrameDatasetSchema.image_path.name
        return (
            pl.element()
            .list.eval(
                (pl.element().struct.field(channel_name_field) == camera_name)
                & pl.element().struct.field(image_path_field).is_not_null()
            )
            .list.any()
        )

    def _build_camera_order_complete_expression(self) -> pl.Expr:
        """Build a boolean expression that is true for rows whose time steps hold every camera.

        ``IMAGE_FRAMES`` is a list of time steps, the current frame first and the previous frames
        after it, and each time step is a list of per-camera image frame structs. A time step is
        complete when, for every camera in ``camera_order``, it holds an image frame of that
        channel with a non-null image path. A row is complete when it has at least one time step
        and the checked time steps are complete: every time step when ``filter_all_time_steps``
        is set, otherwise only the current frame at index 0. A null ``IMAGE_FRAMES`` value is
        incomplete.

        Returns:
            pl.Expr: Boolean expression over the ``IMAGE_FRAMES`` column.
        """
        assert self.camera_order, "camera_order must be set to build the camera order filter."
        time_step_complete = pl.all_horizontal(
            [
                self._build_time_step_has_camera_expression(camera_name)
                for camera_name in self.camera_order
            ]
        )
        image_frames = pl.col(DatasetTableSchema.IMAGE_FRAMES.name)
        checked_time_steps = (
            image_frames if self.filter_all_time_steps else image_frames.list.slice(0, 1)
        )
        return (
            (image_frames.list.len() > 0)
            & checked_time_steps.list.eval(time_step_complete).list.all()
        ).fill_null(False)

    def _filter_frames_with_camera_order(
        self, dataset_records_dataframe: pl.DataFrame
    ) -> pl.DataFrame:
        """Drop dataset records that are missing any camera required by ``camera_order``.

        A record is dropped when a checked ``IMAGE_FRAMES`` time step lacks a camera in
        ``camera_order`` or holds a null image path for it, so downstream image loading always
        sees a complete multiview. Only the current frame is checked unless
        ``filter_all_time_steps`` is set, in which case every previous frame is checked too.

        Args:
            dataset_records_dataframe: Polars DataFrame of dataset records.

        Returns:
            pl.DataFrame: The records whose image frames are complete, in their original order.
        """
        filtered_dataset_records_dataframe = dataset_records_dataframe.filter(
            self._build_camera_order_complete_expression()
        )
        dropped = len(dataset_records_dataframe) - len(filtered_dataset_records_dataframe)
        if dropped:
            logger.info(
                "Filtered %d/%d frames missing one or more cameras in camera_order %s.",
                dropped,
                len(dataset_records_dataframe),
                self.camera_order,
            )
        return filtered_dataset_records_dataframe

    @abstractmethod
    def get_data_sample(self, index: int) -> ModelGTSample:
        """Return raw metadata for a given dataset index.

        Args:
            index: Index of the sample.

        Returns:
            ModelGTSample instance consumed by the transform pipeline.
        """
        raise NotImplementedError("Dataset must implement get_data_sample")

    def apply_transforms(
        self,
        multi_task_gt_sample: ModelGTSample,
        transforms: TransformsCompose | None,
        context: PipelineContext,
    ) -> ModelGTSample:
        """Apply a specific transform pipeline to a sample.

        Also used by :meth:`PipelineContext.sample_secondary` to run a ``pre_transform``
        on a secondary sample with its own context.

        Args:
            multi_task_gt_sample: ModelGTSample instance.
            transforms: Transform pipeline applied to the sample, ``None`` to return it as is.
            context: Pipeline context associated with the sample.

        Returns:
            Transformed ModelGTSample instance.
        """
        if transforms is None:
            return multi_task_gt_sample
        return transforms(multi_task_gt_sample, context=context)

    def collate_fn(self, batch: Sequence[ModelGTSample]) -> ModelGTBatch:
        """
        Collate a batch of ModelGTSample into a ModelGTBatch.
        Args:
          batch: List of ModelGTSample instances to be collated.
        Returns:
          ModelGTBatch: Collated multi-task GT batch.
        """
        return ModelGTBatch.collate_gt_samples(
            gt_samples=batch, max_num_3d_gt_bboxes=self.max_num_3d_gt_bboxes
        )
