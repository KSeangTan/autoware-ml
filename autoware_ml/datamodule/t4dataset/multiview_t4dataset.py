# Copyright 2026 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import polars as pl
import torch

from autoware_ml.databases.schemas.dataset_schemas import DatasetTableSchema
from autoware_ml.databases.schemas.image_frames import ImageFrameDatasetSchema
from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample
from autoware_ml.dataclasses.geometry.images import ImageSample
from autoware_ml.datamodule.base_dataset_task import BaseDatasetTask
from autoware_ml.datamodule.t4dataset.t4dataset import T4Dataset
from autoware_ml.metrics.geometry.lanelet import LaneletMapProvider
from autoware_ml.transforms.base import TransformsCompose
from autoware_ml.types.dataset import SplitType
from autoware_ml.types.tasks import TaskType

logger = logging.getLogger(__name__)


class MultiViewT4Dataset(T4Dataset):
    """
    T4Dataset for models that consume multi-view cameras, optionally as a stream of frames.

    On top of the lidar and task annotations of :class:`T4Dataset`, every sample carries one
    :class:`ImageSample` per camera in ``camera_order`` for the current frame, which the image
    loading transforms turn into ``camera_image_data`` and the collation into an
    ``ImageGTBatch``. For streaming samplers, the dataset groups its indices by scenario and
    stamps the ``prev_exists`` flag derived from the ``previous_sample_id`` column onto the
    frame metadata of every sample.
    """

    def __init__(
        self,
        database_root_path: str,
        max_num_3d_gt_bboxes: int,
        split_type: SplitType,
        map_provider: LaneletMapProvider,
        dataset_records_dataframe: pl.DataFrame | None,
        transforms: TransformsCompose | None,
        dataset_tasks: MappingProxyType[TaskType | str, BaseDatasetTask],
        camera_order: Sequence[str],
        filter_frames_with_camera_order: bool = True,
        filter_all_time_steps: bool = False,
    ) -> None:
        """
        Initialize the MultiViewT4Dataset class.
        Args:
          database_root_path: Root directory of the dataset.
          max_num_3d_gt_bboxes: Maximum number of 3D ground truth bounding boxes in the dataset.
            This is allowed to be 0 if the dataset does not contain any 3D ground truth
            bounding boxes or it does not need to run 3D detection tasks.
          split_type: The split type of the dataset (train, val, test).
          map_provider: Lanelet map provider of the dataset.
          dataset_records_dataframe: Polars DataFrame of dataset records to be used in
            the multi-task dataset.
          transforms: Global transforms to be applied to the dataset records.
          dataset_tasks: Every task dataset that is part of the multi-task dataset, mapped by
            task type.
          camera_order: Ordered camera names expected by the model. Image samples are emitted
            in this order.
          filter_frames_with_camera_order: Drop frames missing any camera in
            ``camera_order`` (absent or null image path), so downstream image loading always
            sees a complete multiview.
          filter_all_time_steps: When ``filter_frames_with_camera_order`` is set, check every
            time step of ``IMAGE_FRAMES`` (the current frame and all previous frames) instead
            of only the current frame at index 0.
        """
        if len(camera_order) == 0:
            raise ValueError("camera_order must hold at least one camera.")
        self._scene_index_groups: Sequence[Sequence[int]] = []
        self.prev_exists: np.ndarray = np.zeros(0, dtype=np.float32)
        super().__init__(
            database_root_path=database_root_path,
            max_num_3d_gt_bboxes=max_num_3d_gt_bboxes,
            split_type=split_type,
            map_provider=map_provider,
            dataset_records_dataframe=dataset_records_dataframe,
            transforms=transforms,
            dataset_tasks=dataset_tasks,
            camera_order=camera_order,
            filter_frames_with_camera_order=filter_frames_with_camera_order,
            filter_all_time_steps=filter_all_time_steps,
        )
        if self.dataset_records_dataframe is not None:
            self._build_streaming_metadata(self.dataset_records_dataframe)

    def assign_dataset_records(self, dataset_records_dataframe: pl.DataFrame) -> None:
        """
        Assign the dataset records dataframe and rebuild the streaming metadata on top of it.

        Args:
            dataset_records_dataframe: Polars DataFrame of dataset records.
        """
        super().assign_dataset_records(dataset_records_dataframe)
        assert self.dataset_records_dataframe is not None
        self._build_streaming_metadata(self.dataset_records_dataframe)

    def _build_streaming_metadata(self, dataset_records_dataframe: pl.DataFrame) -> None:
        """
        Build the scene index groups and the ``prev_exists`` flags of the dataset records.

        Args:
            dataset_records_dataframe: Polars DataFrame of dataset records, already filtered.
        """
        self._scene_index_groups = self.build_scene_index_groups(dataset_records_dataframe)
        self.prev_exists = self.build_prev_exists(dataset_records_dataframe)

    @staticmethod
    def build_scene_index_groups(
        dataset_records_dataframe: pl.DataFrame,
    ) -> Sequence[Sequence[int]]:
        """
        Group dataset indices by scenario, preserving the row order inside every group.

        Args:
          dataset_records_dataframe: Polars DataFrame of dataset records.

        Returns:
          Sequence[Sequence[int]]: One sequence of dataset indices per scenario, in order of
            first appearance.
        """
        groups: dict[str, list[int]] = {}
        scenario_ids = dataset_records_dataframe[DatasetTableSchema.SCENARIO_ID.name].to_list()
        for index, scenario_id in enumerate(scenario_ids):
            groups.setdefault(scenario_id, []).append(index)
        return list(groups.values())

    @staticmethod
    def build_prev_exists(dataset_records_dataframe: pl.DataFrame) -> np.ndarray:
        """
        Build the stream continuity flag of every dataset record.

        A record continues the stream when its ``previous_sample_id`` is the ``sample_id`` of
        the record right before it. The first record of a scenario has a null
        ``previous_sample_id``, and a record whose predecessor was filtered out no longer finds
        it in the previous row, so both start a new stream.

        Args:
          dataset_records_dataframe: Polars DataFrame of dataset records.

        Returns:
          np.ndarray: Float32 flags, 1.0 when the previous row is the previous frame, else 0.0.
        """
        sample_ids = dataset_records_dataframe[DatasetTableSchema.SAMPLE_ID.name].to_list()
        if DatasetTableSchema.PREVIOUS_SAMPLE_ID.name in dataset_records_dataframe.columns:
            previous_sample_ids = dataset_records_dataframe[
                DatasetTableSchema.PREVIOUS_SAMPLE_ID.name
            ].to_list()
        else:
            # Older database caches predate this column, every frame starts a new stream
            previous_sample_ids = [None] * len(sample_ids)

        prev_exists = np.zeros(len(sample_ids), dtype=np.float32)
        for index in range(1, len(sample_ids)):
            previous_sample_id = previous_sample_ids[index]
            if previous_sample_id is not None and previous_sample_id == sample_ids[index - 1]:
                prev_exists[index] = 1.0
        return prev_exists

    def scene_index_groups(self) -> Sequence[Sequence[int]]:
        """
        Group dataset indices by scenario, preserving frame order.

        Returns:
          Sequence[Sequence[int]]: One list of scenario-contiguous dataset indices per scenario.
            A fresh copy is returned so callers may mutate it freely.
        """
        return [list(group) for group in self._scene_index_groups]

    def get_data_sample(self, index: int) -> ModelGTSample:
        """
        Process the dataset records dataframe for multiple tasks, attach the image samples and
        stamp the stream continuity flag onto the frame metadata.

        Args:
          index: Index of the specific record to be processed.

        Returns:
          ModelGTSample: Processed multi-task data row with ``image_samples`` populated and
            ``frame_meta.prev_exists`` set.
        """
        multi_task_gt_sample = super().get_data_sample(index)
        assert multi_task_gt_sample.frame_meta is not None
        frame_meta = multi_task_gt_sample.frame_meta._replace(
            prev_exists=bool(self.prev_exists[index])
        )
        return multi_task_gt_sample._replace(
            image_samples=self.get_image_samples(index), frame_meta=frame_meta
        )

    def get_image_samples(self, index: int) -> Sequence[ImageSample]:
        """
        Retrieve the image samples of the current frame for the given index.

        ``IMAGE_FRAMES`` is a list of time steps, the current frame first, and each time step
        is a list of per-camera image frames. Only the current frame is read here, and its
        cameras are emitted in ``camera_order``.

        Args:
          index: Index of the specific record to be processed.

        Returns:
          Sequence[ImageSample]: One image sample per camera in ``camera_order``.

        Raises:
          ValueError: If the record has no image frames or the current frame misses a camera.
        """
        assert self.dataset_records_dataframe is not None
        assert self.camera_order is not None
        image_frames = self.dataset_records_dataframe.item(
            index, DatasetTableSchema.IMAGE_FRAMES.name
        )
        if image_frames is None or len(image_frames) == 0:
            raise ValueError(f"Dataset record at index {index} has no image frames.")

        # index-0 means the current frame
        current_image_frames: Mapping[str, Mapping[str, Any]] = {
            image_frame[ImageFrameDatasetSchema.image_sensor_channel_name.name]: image_frame
            for image_frame in image_frames[0]
        }
        image_samples = []
        for camera_name in self.camera_order:
            if camera_name not in current_image_frames:
                raise ValueError(
                    f"Dataset record at index {index} misses camera {camera_name} in its "
                    f"current frame, found: {list(current_image_frames.keys())}."
                )
            image_samples.append(
                self.build_image_sample(camera_name, current_image_frames[camera_name])
            )
        return image_samples

    def build_image_sample(self, camera_name: str, image_frame: Mapping[str, Any]) -> ImageSample:
        """
        Build an image sample from one ``IMAGE_FRAMES`` struct.

        Args:
          camera_name: Camera channel name of the image frame.
          image_frame: One image frame struct of the dataset records dataframe.

        Returns:
          ImageSample: Image sample read by the image loading transforms.

        Raises:
          ValueError: If the image frame has no image path or no lidar to camera matrices.
        """
        image_path = image_frame[ImageFrameDatasetSchema.image_path.name]
        lidar2cam = image_frame[ImageFrameDatasetSchema.lidar2cam.name]
        lidar2img = image_frame[ImageFrameDatasetSchema.lidar2img.name]
        if image_path is None:
            raise ValueError(f"Image frame of camera {camera_name} has no image path.")
        if lidar2cam is None or lidar2img is None:
            raise ValueError(
                f"Image frame of camera {camera_name} has no lidar2cam or lidar2img matrix."
            )
        distortion_model = image_frame[ImageFrameDatasetSchema.image_distortion_model.name]
        distortion_coefficients = image_frame[
            ImageFrameDatasetSchema.image_distortion_coefficients.name
        ]
        return ImageSample(
            image_path=self._resolve_sensor_data_path(image_path),
            camera_name=camera_name,
            timestamp=image_frame[ImageFrameDatasetSchema.image_timestamp_seconds.name],
            camera_intrinsic=torch.tensor(
                image_frame[ImageFrameDatasetSchema.cam2img.name], dtype=torch.float32
            ),
            lidar2cam=torch.tensor(lidar2cam, dtype=torch.float32),
            lidar2image=torch.tensor(lidar2img, dtype=torch.float32),
            distortion_model=distortion_model if distortion_model is not None else "",
            distortion_coefficients=torch.tensor(
                distortion_coefficients if distortion_coefficients is not None else [],
                dtype=torch.float32,
            ),
        )
