from dataclasses import dataclass
from typing import NamedTuple

import polars as pl
from pydantic import BaseModel, ConfigDict


class DatasetTableColumn(NamedTuple):
    """
    Annotation table column.
    """

    name: str
    dtype: pl.DataType


@dataclass(frozen=True)
class DatasetTableSchema:
    """
    Annotation table schema.
    """

    SCENARIO_ID = DatasetTableColumn("scenario_id", pl.String)
    SAMPLE_ID = DatasetTableColumn("sample_id", pl.String)
    SAMPLE_INDEX = DatasetTableColumn("sample_index", pl.Int32)
    TIMESTAMP_SECONDS = DatasetTableColumn("timestamp_seconds", pl.Float64)
    LOCATION = DatasetTableColumn("location", pl.String)
    VEHICLE_TYPE = DatasetTableColumn("vehicle_type", pl.String)
    SCENARIO_NAME = DatasetTableColumn("scenario_name", pl.String)
    LIDAR_CHANNEL_NAME = DatasetTableColumn("lidar_channel_name", pl.String)
    LIDAR_SENSOR_TOKEN = DatasetTableColumn("lidar_sensor_token", pl.String)
    LIDAR_TRANSLATION = DatasetTableColumn("lidar_translation",
                                           pl.Array(pl.Float64, shape=(3, )))
    LIDAR_ROTATION = DatasetTableColumn("lidar_rotation",
                                        pl.Array(pl.Float64, shape=(3, 3)))
    LIDAR_POINTCLOUD_PATH = DatasetTableColumn("lidar_pointcloud_path",
                                               pl.String)
    LIDAR_POINTCLOUD_NUM_FEATURES = DatasetTableColumn(
        "lidar_pointcloud_num_features", pl.Int32)
    LIDAR_TO_EGO_POSE_MATRIX = DatasetTableColumn(
        "lidar_to_ego_pose_matrix", pl.Array(pl.Float64, shape=(4, 4)))
    # LIDAR_SWEEP_TIMESTAMPS = DatasetTableColumn("lidar_sweep_timestamps",
    #                                             pl.List(pl.Float64))
    # LIDAR_SWEEP_SAMPLE_DATA_TOKENS = DatasetTableColumn(
    #     "lidar_sweep_sample_data_tokens", pl.List(pl.String))
    # LIDAR_SWEEP_EGO_TO_GLOBAL_MATRICES = DatasetTableColumn(
    #     "lidar_sweep_ego_to_global_matrices", pl.List(pl.Array(pl.Float64, shape=(4, 4))))

    # Ego to global transformation matrix 4x4
    EGO_POSE_TO_GLOBAL_MATRIX = DatasetTableColumn(
        "ego_pose_to_global_matrix", pl.Array(pl.Float64, shape=(4, 4)))

    # List of 3D bounding boxes with center_x, center_y, center_z, length, width, height, yaw, velocity_x, velocity_y
    # BBOX_3D = DatasetTableColumn("bbox_3d",
    #                              pl.List(pl.Array(pl.Float64, shape=(9))))

    @classmethod
    def to_polars_schema(cls) -> pl.Schema:
        """
        Convert the dataset table schema to a Polars schema.
        """
        return pl.Schema({
            v.name: v.dtype
            for k, v in cls.__dict__.items()
            if not k.startswith("__") and isinstance(v, DatasetTableColumn)
        })


class DatasetRecord(BaseModel):
    """
    Data class to save a record for each column in the annotation table.
    :param scenario_id: Scenario id.
    :param sample_id: Sample id.
    :param location: Location of the vehicle.
    :param vehicle_type: Type of the vehicle.
    :param bbox_3d: List of 3D bounding boxes with center_x, center_y, center_z, length, width, height, yaw, velocity_x, velocity_y.
    :param bbox_2d: List of 2D bounding boxes with center_x, center_y, width, height.
    """

    # Set model config to frozen
    model_config = ConfigDict(frozen=True, strict=True)

    scenario_id: str
    sample_id: str
    sample_index: int
    timestamp_seconds: float
    location: str
    vehicle_type: str
    scenario_name: str

    # TODO (KokSeang): Add more annotation fields here
    # bbox_3d: Sequence[npt.NDArray[np.float64]]
