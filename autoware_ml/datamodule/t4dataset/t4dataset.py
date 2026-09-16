import logging
from types import MappingProxyType
from typing import Sequence

from jaxtyping import Float32
import polars as pl
import numpy as np
import torch

from autoware_ml.databases.schemas.lidar_frames import LidarFrameDatasetSchema
from autoware_ml.databases.schemas.dataset_schemas import DatasetTableSchema
from autoware_ml.dataclasses.geometry.point_clouds import LiDARPointCloudSample
from autoware_ml.dataclasses.batch.frame_meta import FrameMetaSample
from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample
from autoware_ml.datamodule.base_dataset import (
    BaseDataset,
)
from autoware_ml.datamodule.base_dataset_task import BaseDatasetTask
from autoware_ml.datamodule.t4dataset.frame_meta import scene_dir_fragment
from autoware_ml.transforms.base import TransformsCompose
from autoware_ml.types.tasks import TaskType
from autoware_ml.types.dataset import SplitType
from autoware_ml.metrics.geometry.lanelet import LaneletMapProvider


logger = logging.getLogger(__name__)


class T4Dataset(BaseDataset):
    """
    A dataset class that supports T4Dataset with multiple tasks.
    It extends BaseDataset  to include implementation of data retrieval for multiple
    tasks in a single interface.
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
    ) -> None:
        """
        Initialize the T4Dataset class.
        Args:
          max_num_3d_gt_bboxes: Maximum number of 3D ground truth bounding boxes in the dataset.
            This is allowed to be 0 if the dataset does not contain any 3D ground truth
            bounding boxes or it does not need to run 3D detection tasks.
          dataset_records_dataframe: Polars DataFrame of dataset records to be used in
            the multi-task dataset.
          transforms: Global transforms to be applied to the dataset records.
          dataset_tasks: Every task dataset that is part of the multi-task dataset, mapped by
            task type.
        """
        super().__init__(
            database_root_path=database_root_path,
            max_num_3d_gt_bboxes=max_num_3d_gt_bboxes,
            dataset_records_dataframe=dataset_records_dataframe,
            transforms=transforms,
            split_type=split_type,
        )
        self.map_provider = map_provider

        # Convert the dataset_tasks to TaskType: BaseDatasetTask mapping if the keys are strings
        self.dataset_tasks: MappingProxyType[TaskType, BaseDatasetTask] = MappingProxyType(
            {
                TaskType(key) if isinstance(key, str) else key: value
                for key, value in dataset_tasks.items()
            }
        )
        logger.info(
            f"Initialized T4Dataset ({self.split_type}) with {len(self.dataset_tasks)} "
            f"task datasets: {list(self.dataset_tasks.keys())} "
            f"transforms: {self.transforms} and max_num_3d_gt_bboxes: {self.max_num_3d_gt_bboxes}"
        )

        # Log dataset information to console for each task dataset
        for dataset_task in self.dataset_tasks.values():
            dataset_task.log_dataset_info()

    def get_data_sample(self, index: int) -> ModelGTSample:
        """
        Process the dataset records dataframe for multiple tasks in the T4 dataset.

        Args:
          index: Index of the specific record to be processed.

        Returns:
          ModelGTSample: Processed multi-task data row, mapped by task type.
        """
        data_samples = {}
        for task_type, dataset_task in self.dataset_tasks.items():
            data_samples[task_type] = dataset_task.get_data_sample(index)

        # Retrieve general data row for the given index from the dataset records dataframe
        lidar_pointcloud_samples = self.get_lidar_pointcloud_data_samples(index)

        # Retrieve the detection3d_gt_bboxes_3d and segmentation3d_gt_sample from the data_samples dictionary
        detection3d_gt_sample: ModelGTSample | None = data_samples.get(TaskType.DETECTION3D, None)
        if detection3d_gt_sample is not None:
            detection3d_gt_bboxes_3d = detection3d_gt_sample.detection3d_gt_bboxes_3d
            detection3d_traffic_cone_barrier_bbox_status = (
                detection3d_gt_sample.detection3d_traffic_cone_barrier_bbox_status
            )
        else:
            detection3d_gt_bboxes_3d = None
            detection3d_traffic_cone_barrier_bbox_status = None

        segmentation3d_multi_task_gt_sample: ModelGTSample | None = data_samples.get(
            TaskType.SEGMENTATION3D, None
        )
        if segmentation3d_multi_task_gt_sample is not None:
            segmentation3d_gt_sample = segmentation3d_multi_task_gt_sample.segmentation3d_gt_sample
        else:
            segmentation3d_gt_sample = None

        # Merge the data samples from different tasks into a single multi-task data row
        return ModelGTSample(
            lidar_point_cloud_samples=lidar_pointcloud_samples,
            frame_meta=self.build_frame_meta_sample(
                lidar_pointcloud_samples, str(self.database_root_path)
            ),
            point_cloud_data=None,  # point cloud data will be populated in the transform pipeline
            detection3d_gt_bboxes_3d=detection3d_gt_bboxes_3d,
            segmentation3d_gt_sample=segmentation3d_gt_sample,
            image_samples=None,
            camera_image_data=None,
            detection3d_traffic_cone_barrier_bbox_status=detection3d_traffic_cone_barrier_bbox_status,
        )

    @staticmethod
    def build_frame_meta_sample(
        lidar_pointcloud_samples: Sequence[LiDARPointCloudSample], database_root_path: str
    ) -> FrameMetaSample:
        """
        Build the evaluation metadata of a frame from its main lidar record.

        The points and boxes are expressed in the main lidar frame, so the map transform the
        metrics read as ``ego2global`` composes the lidar mounting into the ego pose. The scene
        token is the ``<db>/<scene_uuid>/<version>`` fragment of the lidar path below the
        database root, which the lanelet map provider resolves to the scene's map.

        Args:
          lidar_pointcloud_samples: Lidar records of the frame, the main lidar first.
          database_root_path: Root directory of the database.

        Returns:
          FrameMetaSample: Map transform and scene token of the frame.

        Raises:
          ValueError: If the frame has no lidar record.
        """
        if len(lidar_pointcloud_samples) == 0:
            raise ValueError("At least one lidar point cloud sample is required for frame_meta.")
        main_lidar = lidar_pointcloud_samples[0]
        return FrameMetaSample(
            ego2global=main_lidar.lidar_to_ego_pose_to_global_matrix
            @ main_lidar.sensor_to_ego_pose_matrix,
            scene_token=scene_dir_fragment(main_lidar.point_cloud_path, database_root_path),
        )

    def _update_lidar_pointcloud_path(self, lidar_pointcloud_path: str) -> str:
        """
        Remove the absolute path prefix from the lidar pointcloud path and return the updated path with the dataset root.
        Note that this is dataset-specific.
        """
        # Return the relative path from the lidar pointcloud path
        relative_path = "/".join(lidar_pointcloud_path.split("/")[-6:])
        return str(self.database_root_path / relative_path)

    def get_lidar_pointcloud_data_samples(self, idx: int) -> Sequence[LiDARPointCloudSample]:
        """
        Retrieve the lidar point cloud data row for the given index.

        Args:
          idx: Index of the specific record to be processed.
        """
        # Retrieve the lidar point cloud data row for the given index from the dataset records dataframe
        lidar_pointcloud_metadata_samples = self.dataset_records_dataframe.item(
            idx, DatasetTableSchema.LIDAR_FRAMES.name
        )
        lidar_pointcloud_samples = []
        for lidar_pointcloud_metadata in lidar_pointcloud_metadata_samples:
            lidar_sensor_to_ego_pose_matrix: Float32[np.ndarray, "4 4"] = lidar_pointcloud_metadata[
                LidarFrameDatasetSchema.lidar_sensor_to_ego_pose_matrix.name
            ]
            lidar_to_ego_pose_to_global_matrix: Float32[np.ndarray, "4 4"] = (
                lidar_pointcloud_metadata[
                    LidarFrameDatasetSchema.lidar_frame_ego_pose_to_global_matrix.name
                ]
            )
            lidar_sensor_to_lidar_sweep_matrix: Float32[np.ndarray, "4 4"] = (
                lidar_pointcloud_metadata[
                    LidarFrameDatasetSchema.lidar_sensor_to_lidar_sweep_matrix.name
                ]
            )
            lidar_pointcloud_path = self._update_lidar_pointcloud_path(
                lidar_pointcloud_metadata[LidarFrameDatasetSchema.lidar_pointcloud_path.name]
            )

            lidar_pointcloud_samples.append(
                LiDARPointCloudSample(
                    point_cloud_path=lidar_pointcloud_path,
                    timestamp=lidar_pointcloud_metadata[
                        LidarFrameDatasetSchema.lidar_timestamp_seconds.name
                    ],
                    sensor_to_ego_pose_matrix=torch.tensor(
                        lidar_sensor_to_ego_pose_matrix,
                        dtype=torch.float32,
                    ),
                    lidar_to_ego_pose_to_global_matrix=torch.tensor(
                        lidar_to_ego_pose_to_global_matrix,
                        dtype=torch.float32,
                    ),
                    lidar_sensor_to_lidar_sweep_matrix=torch.tensor(
                        lidar_sensor_to_lidar_sweep_matrix,
                        dtype=torch.float32,
                    ),
                )
            )
        return lidar_pointcloud_samples

    def assign_dataset_records(self, dataset_records_dataframe: pl.DataFrame) -> None:
        """
        Recursively assign the dataset records dataframe to each task dataset as well and
        perform their pre_filtering .

        Args:
            dataset_records_dataframe: Polars DataFrame of dataset records.
        """
        self.dataset_records_dataframe = dataset_records_dataframe
        for dataset_task in self.dataset_tasks.values():
            filtered_dataset_records_dataframe = dataset_task.pre_filter_dataset_records(
                dataset_records_dataframe
            )
            dataset_task.dataset_records_dataframe = filtered_dataset_records_dataframe
            dataset_task.log_dataset_info()  # Log dataset information for each task dataset
