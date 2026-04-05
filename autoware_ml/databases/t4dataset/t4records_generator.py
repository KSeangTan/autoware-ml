import logging

from pathlib import Path
from typing import Sequence

from t4_devkit import Tier4
from t4_devkit.schema import Sample, SampleData, CalibratedSensor, Scene
from t4_devkit.common.timestamp import microseconds2seconds

from autoware_ml.common.enums.enums import LidarChannel
from autoware_ml.databases.schemas import DatasetRecord
from autoware_ml.databases.scenarios import ScenarioData
from autoware_ml.databases.t4dataset.t4sample_records import (
    T4SampleRecord,
    T4SampleRecordBasicInfo,
    T4SampleRecordLidarInfo,
)
from autoware_ml.utils.dataset import convert_quaternion_to_matrix

logger = logging.getLogger(__name__)


class T4RecordsGenerator:
    """T4 dataset records generator."""

    def __init__(
        self,
        database_root_path: str,
        scenario_data: ScenarioData,
        max_sweeps: int,
        sample_steps: int = 1,
        lidar_pointcloud_num_features: int = 5,
    ) -> None:
        """
        Initialize T4 dataset generator.
        :param database_root_path: Root path of the T4 database.
        :param scenario_data: Scenario data.
        :param max_sweeps: Max number of lidar sweeps to include, only for 3D.
        :param sample_steps: Number of frames/samples to skip between each sample.
        """
        self.database_root_path = Path(database_root_path)
        self.scenario_data = scenario_data
        self.max_sweeps = max_sweeps
        self.sample_steps = sample_steps
        self.lidar_pointcloud_num_features = lidar_pointcloud_num_features
        self.t4_devkit_dataset = self._construct_t4_devkit_dataset()

    def _construct_t4_devkit_dataset(self) -> Tier4:
        """Construct T4 dataset."""
        scene_root_dir_path = (
            self.database_root_path
            / self.scenario_data.db_version
            / self.scenario_data.scenario_id
            / self.scenario_data.scenario_version
        )
        if not scene_root_dir_path.exists():
            raise ValueError(f"Scene root directory {scene_root_dir_path} does not exist.")
        return Tier4(data_root=scene_root_dir_path, verbose=False)

    def generate_dataset_records(self) -> Sequence[DatasetRecord]:
        """Generate dataset records."""
        records = []
        logger.info(
            f"Generating dataset records for scenario: {self.scenario_data.scenario_id} with sample steps: {self.sample_steps} and max sweeps: {self.max_sweeps}"
        )
        for sample_index in range(0, len(self.t4_devkit_dataset.sample), self.sample_steps):
            sample = self.t4_devkit_dataset.sample[sample_index]
            t4_sample_record = self.extract_t4_sample_record(sample, sample_index)
            records.append(t4_sample_record.to_dataset_record())

        return records

    def _extract_basic_info(self, sample: Sample, sample_index: int) -> T4SampleRecordBasicInfo:
        """Extract basic information from a T4 sample."""
        scene_record: Scene = self.t4_devkit_dataset.get("scene", sample.scene_token)
        return T4SampleRecordBasicInfo(
            scenario_id=self.scenario_data.scenario_id,
            sample_id=sample.token,
            sample_index=sample_index,
            location=self.scenario_data.location,
            vehicle_type=self.scenario_data.vehicle_type,
            timestamp_seconds=microseconds2seconds(sample.timestamp),
            scenario_name=scene_record.name,
        )

    def _extract_lidar_info(self, sample: Sample) -> T4SampleRecordLidarInfo:
        """Extract lidar information from a T4 sample."""
        # Read lidar channel name
        if LidarChannel.LIDAR_TOP in sample.data:
            lidar_channel_name = LidarChannel.LIDAR_TOP
        elif LidarChannel.LIDAR_CONCAT in sample.data:
            lidar_channel_name = LidarChannel.LIDAR_CONCAT
        else:
            raise ValueError(
                f"Lidar channel {LidarChannel.LIDAR_TOP} or {LidarChannel.LIDAR_CONCAT} not found in sample data."
            )

        calibrated_lidar_sample_data_token = sample.data[lidar_channel_name]
        sd_record: SampleData = self.t4_devkit_dataset.get(
            "sample_data", calibrated_lidar_sample_data_token
        )
        cs_record: CalibratedSensor = self.t4_devkit_dataset.get(
            "calibrated_sensor", sd_record.calibrated_sensor_token
        )
        lidar_to_ego_matrix = convert_quaternion_to_matrix(
            rotation_quaternion=cs_record.rotation, translation=cs_record.translation
        )
        lidar_path, boxes_3d, _ = self.t4_devkit_dataset.get_sample_data(
            sample_data_token=calibrated_lidar_sample_data_token,
            as_3d=True,
            as_sensor_coord=True,
        )
        return T4SampleRecordLidarInfo(
            lidar_calibrated_sensor_id=cs_record.sensor_token,
            lidar_calibrated_sensor_channel_name=lidar_channel_name,
            lidar_pointcloud_path=lidar_path,
            lidar_pointcloud_num_features=self.lidar_pointcloud_num_features,
            lidar_to_ego_pose_matrix=lidar_to_ego_matrix,
            boxes_3d=boxes_3d,
        )

    def extract_t4_sample_record(self, sample: Sample, sample_index: int) -> T4SampleRecord:
        """Extract T4 sample record from a T4Dataset."""

        # First, extract basic information from the T4Dataset
        basic_info = self._extract_basic_info(sample=sample, sample_index=sample_index)

        # Second, extract lidar information from the T4Dataset
        lidar_info = self._extract_lidar_info(sample=sample)
        # Third, extract multisweep lidar information from the T4Dataset
        # TODO (KokSeang): Extract more information, for example, boxes, from the T4Dataset.
        # Last, return the T4 sample record
        return T4SampleRecord(
            basic_info=basic_info,
            lidar_info=lidar_info,
        )
