"""Tests for the per-frame evaluation metadata helpers."""

from __future__ import annotations

import pytest
import torch

from autoware_ml.dataclasses.geometry.point_clouds import LiDARPointCloudSample
from autoware_ml.datamodule.t4dataset.frame_meta import scene_dir_fragment
from autoware_ml.datamodule.t4dataset.t4dataset import T4Dataset

ROOT = "/data/t4"


def test_scene_dir_fragment() -> None:
    path = "db_j6gen2_v2/13cabeac-a81b/0/data/LIDAR_CONCAT/00000.pcd.bin"
    assert scene_dir_fragment(path, ROOT) == "db_j6gen2_v2/13cabeac-a81b/0"


def test_scene_dir_fragment_strips_data_root_prefix() -> None:
    path = "/data/t4/db_j6gen2_v2/13cabeac-a81b/0/data/LIDAR_CONCAT/00000.pcd.bin"
    assert scene_dir_fragment(path, ROOT) == "db_j6gen2_v2/13cabeac-a81b/0"
    relative_root = "data/t4"
    prefixed = "data/t4/db_j6gen2_v2/13cabeac-a81b/0/data/LIDAR_CONCAT/00000.pcd.bin"
    assert scene_dir_fragment(prefixed, relative_root) == "db_j6gen2_v2/13cabeac-a81b/0"


def test_scene_dir_fragment_rejects_absolute_path_outside_root() -> None:
    with pytest.raises(ValueError, match="absolute path must live under data_root"):
        scene_dir_fragment("/elsewhere/db/uuid/0/data/00000.pcd.bin", ROOT)


def test_scene_dir_fragment_rejects_short_paths() -> None:
    with pytest.raises(ValueError, match="scene directory"):
        scene_dir_fragment("no_scene.bin", ROOT)


def _lidar_sample(
    point_cloud_path: str, sensor_to_ego: torch.Tensor, ego_to_global: torch.Tensor
) -> LiDARPointCloudSample:
    return LiDARPointCloudSample(
        point_cloud_path=point_cloud_path,
        timestamp=0.0,
        sensor_to_ego_pose_matrix=sensor_to_ego,
        lidar_to_ego_pose_to_global_matrix=ego_to_global,
        lidar_sensor_to_lidar_sweep_matrix=torch.eye(4),
    )


def test_build_frame_meta_sample_composes_mounting_and_derives_scene_token() -> None:
    sensor_to_ego = torch.eye(4)
    sensor_to_ego[0, 3] = 1.0  # lidar mounted 1 m ahead of base_link
    ego_to_global = torch.eye(4)
    ego_to_global[1, 3] = 10.0  # ego 10 m along y in the map
    main_lidar = _lidar_sample(
        "/data/t4/db_j6gen2/scene-uuid/2/data/LIDAR_CONCAT/0.pcd.bin", sensor_to_ego, ego_to_global
    )
    # The sweeps carry other poses and must not influence the frame metadata.
    sweep = _lidar_sample(
        "/data/t4/db_other/other-uuid/0/data/LIDAR_CONCAT/1.pcd.bin", torch.eye(4), torch.eye(4)
    )

    frame_meta = T4Dataset.build_frame_meta_sample([main_lidar, sweep], "/data/t4")

    expected = ego_to_global @ sensor_to_ego
    assert torch.allclose(frame_meta.ego2global, expected)
    # The lidar origin lands at (1, 10) in the map frame.
    origin = frame_meta.ego2global @ torch.tensor([0.0, 0.0, 0.0, 1.0])
    assert torch.allclose(origin[:2], torch.tensor([1.0, 10.0]))
    assert frame_meta.scene_token == "db_j6gen2/scene-uuid/2"


def test_build_frame_meta_sample_requires_a_lidar_record() -> None:
    with pytest.raises(ValueError, match="lidar point cloud sample"):
        T4Dataset.build_frame_meta_sample([], "/data/t4")
