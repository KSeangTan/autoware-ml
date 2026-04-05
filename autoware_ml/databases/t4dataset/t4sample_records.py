import logging

from typing import Sequence

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict
from t4_devkit.dataclass.box import Box3D

from autoware_ml.databases.schemas import DatasetRecord

logger = logging.getLogger(__name__)


class T4SampleRecordBasicInfo(BaseModel):
    """Basic information of a T4 sample record."""

    model_config = ConfigDict(frozen=True, strict=True)

    scenario_id: str
    sample_id: str
    sample_index: int
    timestamp_seconds: float
    scenario_name: str
    location: str | None = None
    vehicle_type: str | None = None


class T4SampleRecordLidarInfo(BaseModel):
    """Lidar information of a T4 sample record."""

    model_config = ConfigDict(frozen=True, strict=True)

    lidar_calibrated_sensor_id: str
    lidar_calibrated_sensor_channel_name: str
    lidar_pointcloud_path: str
    lidar_pointcloud_num_features: int
    lidar_to_ego_pose_matrix: npt.NDArray[np.float64]  # (4, 4)
    boxes_3d: Sequence[Box3D]

    def parse_lidar_path(self) -> str:
        """Parse lidar path to {database_version}/{scene_id}/{dataset_version}/data/{lidar_token}/{frame}.bin from path."""
        return "/".join(self.lidar_pointcloud_path.split("/")[-6:])


class T4SampleRecord(BaseModel):
    """Temporary T4 sample record."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    basic_info: T4SampleRecordBasicInfo
    lidar_info: T4SampleRecordLidarInfo

    def to_dataset_record(self) -> DatasetRecord:
        """Convert T4 sample record to dataset record."""
        return DatasetRecord(
            # Basic Metadata
            scenario_id=self.basic_info.scenario_id,
            sample_id=self.basic_info.sample_id,
            sample_index=self.basic_info.sample_index,
            timestamp_seconds=self.basic_info.timestamp_seconds,
            scenario_name=self.basic_info.scenario_name,
            location=self.basic_info.location,
            vehicle_type=self.basic_info.vehicle_type,
            # LiDAR Metadata
            lidar_calibrated_sensor_id=self.lidar_info.lidar_calibrated_sensor_id,
            lidar_calibrated_sensor_channel_name=self.lidar_info.lidar_calibrated_sensor_channel_name,
            lidar_pointcloud_path=self.lidar_info.parse_lidar_path(),
            lidar_pointcloud_num_features=self.lidar_info.lidar_pointcloud_num_features,
            lidar_to_ego_pose_matrix=self.lidar_info.lidar_to_ego_pose_matrix,
        )
