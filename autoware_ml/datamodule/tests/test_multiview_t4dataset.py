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

import unittest
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import polars as pl
import torch

from autoware_ml.databases.schemas.dataset_schemas import DatasetTableSchema
from autoware_ml.databases.schemas.image_frames import ImageFrameDatasetSchema
from autoware_ml.databases.schemas.lidar_frames import LidarFrameDatasetSchema
from autoware_ml.datamodule.t4dataset.multiview_t4dataset import MultiViewT4Dataset
from autoware_ml.metrics.geometry.lanelet import LaneletMapProvider
from autoware_ml.types.dataset import SplitType

DATABASE_ROOT_PATH = "/data/t4dataset"
CAMERA_ORDER = ["CAM_FRONT", "CAM_BACK"]
IDENTITY_3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
IDENTITY_4 = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]


class _NoMapResolver:
    """Resolver of a database without lanelet maps."""

    @property
    def cache_key(self) -> str:
        return "no-maps"

    def __call__(self, scene_token: object) -> str:
        raise FileNotFoundError(f"No lanelet map for {scene_token}")

    def available(self, scene_token: object) -> bool:
        return False


class MultiViewT4DatasetTest(unittest.TestCase):
    """Tests for the image samples and streaming metadata of MultiViewT4Dataset."""

    def build_lidar_frame(self, sample_id: str) -> Mapping[str, Any]:
        """Build one lidar frame struct with the fields the dataset reads."""
        return {
            LidarFrameDatasetSchema.lidar_timestamp_seconds.name: 1.0,
            LidarFrameDatasetSchema.lidar_pointcloud_path.name: (
                f"/src/db_test/scene-uuid/0/data/LIDAR_CONCAT/{sample_id}.pcd.bin"
            ),
            LidarFrameDatasetSchema.lidar_sensor_to_ego_pose_matrix.name: IDENTITY_4,
            LidarFrameDatasetSchema.lidar_frame_ego_pose_to_global_matrix.name: IDENTITY_4,
            LidarFrameDatasetSchema.lidar_sensor_to_lidar_sweep_matrix.name: IDENTITY_4,
        }

    def build_image_frame(
        self, sample_id: str, camera_name: str, focal_length: float = 100.0
    ) -> Mapping[str, Any]:
        """Build one image frame struct with the fields the dataset reads."""
        return {
            ImageFrameDatasetSchema.image_sensor_channel_name.name: camera_name,
            ImageFrameDatasetSchema.image_timestamp_seconds.name: 1.5,
            ImageFrameDatasetSchema.image_path.name: (
                f"/src/db_test/scene-uuid/0/data/{camera_name}/{sample_id}.jpg"
            ),
            ImageFrameDatasetSchema.cam2img.name: [
                [focal_length, 0.0, 3.0],
                [0.0, focal_length, 2.0],
                [0.0, 0.0, 1.0],
            ],
            ImageFrameDatasetSchema.image_distortion_coefficients.name: [0.1, 0.2, 0.0, 0.0],
            ImageFrameDatasetSchema.image_distortion_model.name: "plumb_bob",
            ImageFrameDatasetSchema.lidar2cam.name: IDENTITY_4,
            ImageFrameDatasetSchema.lidar2img.name: IDENTITY_4,
        }

    def build_time_step(
        self, sample_id: str, camera_names: Sequence[str]
    ) -> Sequence[Mapping[str, Any]]:
        """Build one time step holding an image frame per camera."""
        return [self.build_image_frame(sample_id, camera_name) for camera_name in camera_names]

    def build_record(
        self,
        scenario_id: str,
        sample_id: str,
        previous_sample_id: str | None,
        camera_names: Sequence[str] = CAMERA_ORDER,
    ) -> Mapping[str, Any]:
        """Build one dataset record row with a single, current time step."""
        return {
            DatasetTableSchema.SCENARIO_ID.name: scenario_id,
            DatasetTableSchema.SAMPLE_ID.name: sample_id,
            DatasetTableSchema.PREVIOUS_SAMPLE_ID.name: previous_sample_id,
            DatasetTableSchema.LIDAR_FRAMES.name: [self.build_lidar_frame(sample_id)],
            DatasetTableSchema.IMAGE_FRAMES.name: [self.build_time_step(sample_id, camera_names)],
        }

    def build_dataset_records_dataframe(self, records: Sequence[Mapping[str, Any]]) -> pl.DataFrame:
        """Build a dataset records dataframe typed with the dataset schema."""
        columns = [
            DatasetTableSchema.SCENARIO_ID,
            DatasetTableSchema.SAMPLE_ID,
            DatasetTableSchema.PREVIOUS_SAMPLE_ID,
            DatasetTableSchema.LIDAR_FRAMES,
            DatasetTableSchema.IMAGE_FRAMES,
        ]
        return pl.DataFrame(
            {column.name: [record[column.name] for record in records] for column in columns},
            schema={column.name: column.dtype for column in columns},
        )

    def build_dataset(
        self,
        dataset_records_dataframe: pl.DataFrame | None,
        filter_frames_with_camera_order: bool = True,
    ) -> MultiViewT4Dataset:
        """Build the dataset under test without any task dataset."""
        return MultiViewT4Dataset(
            database_root_path=DATABASE_ROOT_PATH,
            max_num_3d_gt_bboxes=0,
            split_type=SplitType.TRAIN,
            map_provider=LaneletMapProvider(resolve_osm=_NoMapResolver()),
            dataset_records_dataframe=dataset_records_dataframe,
            transforms=None,
            dataset_tasks=MappingProxyType({}),
            camera_order=CAMERA_ORDER,
            filter_frames_with_camera_order=filter_frames_with_camera_order,
        )

    def build_two_scenario_dataframe(self) -> pl.DataFrame:
        """Build two scenarios of two and three consecutive frames."""
        return self.build_dataset_records_dataframe(
            [
                self.build_record("scene-a", "a0", None),
                self.build_record("scene-a", "a1", "a0"),
                self.build_record("scene-b", "b0", None),
                self.build_record("scene-b", "b1", "b0"),
                self.build_record("scene-b", "b2", "b1"),
            ]
        )

    def test_get_data_sample_emits_image_samples_in_camera_order(self) -> None:
        """The current frame yields one image sample per camera, ordered by camera_order."""
        dataset = self.build_dataset(
            self.build_dataset_records_dataframe(
                [self.build_record("scene-a", "a0", None, camera_names=["CAM_BACK", "CAM_FRONT"])]
            )
        )

        multi_task_gt_sample = dataset.get_data_sample(0)

        self.assertIsNotNone(multi_task_gt_sample.image_samples)
        assert multi_task_gt_sample.image_samples is not None
        self.assertEqual(
            [image_sample.camera_name for image_sample in multi_task_gt_sample.image_samples],
            CAMERA_ORDER,
        )
        self.assertIsNotNone(multi_task_gt_sample.lidar_point_cloud_samples)
        self.assertIsNone(multi_task_gt_sample.camera_image_data)

    def test_image_sample_carries_the_image_frame_fields(self) -> None:
        """Paths are re-rooted under the database root and matrices become float32 tensors."""
        dataset = self.build_dataset(
            self.build_dataset_records_dataframe([self.build_record("scene-a", "a0", None)])
        )

        image_sample = dataset.get_image_samples(0)[0]

        self.assertEqual(
            image_sample.image_path,
            f"{DATABASE_ROOT_PATH}/db_test/scene-uuid/0/data/CAM_FRONT/a0.jpg",
        )
        self.assertEqual(image_sample.timestamp, 1.5)
        self.assertEqual(image_sample.distortion_model, "plumb_bob")
        self.assertEqual(image_sample.camera_intrinsic.dtype, torch.float32)
        self.assertEqual(image_sample.camera_intrinsic[0, 0].item(), 100.0)
        self.assertEqual(tuple(image_sample.lidar2cam.shape), (4, 4))
        self.assertEqual(tuple(image_sample.lidar2image.shape), (4, 4))
        np.testing.assert_allclose(
            image_sample.distortion_coefficients.numpy(),
            np.array([0.1, 0.2, 0.0, 0.0], dtype=np.float32),
        )

    def test_get_image_samples_reads_only_the_current_frame(self) -> None:
        """Previous time steps do not leak into the image samples."""
        record = dict(self.build_record("scene-a", "a1", "a0"))
        record[DatasetTableSchema.IMAGE_FRAMES.name] = [
            self.build_time_step("a1", CAMERA_ORDER),
            self.build_time_step("a0", CAMERA_ORDER),
        ]
        dataset = self.build_dataset(self.build_dataset_records_dataframe([record]))

        image_samples = dataset.get_image_samples(0)

        self.assertEqual(len(image_samples), 2)
        self.assertTrue(all(sample.image_path.endswith("a1.jpg") for sample in image_samples))

    def test_get_image_samples_rejects_a_missing_camera_when_unfiltered(self) -> None:
        """Without the camera order filter an incomplete frame fails loudly."""
        dataset = self.build_dataset(
            self.build_dataset_records_dataframe(
                [self.build_record("scene-a", "a0", None, camera_names=["CAM_FRONT"])]
            ),
            filter_frames_with_camera_order=False,
        )

        with self.assertRaises(ValueError):
            dataset.get_image_samples(0)

    def test_scene_index_groups_follow_the_scenario_ids(self) -> None:
        """Indices are grouped per scenario, in row order, as a fresh copy."""
        dataset = self.build_dataset(self.build_two_scenario_dataframe())

        groups = dataset.scene_index_groups()
        groups[0].append(99)

        self.assertEqual(dataset.scene_index_groups(), [[0, 1], [2, 3, 4]])

    def test_get_data_sample_stamps_prev_exists_onto_the_frame_meta(self) -> None:
        """The frame metadata of every sample carries its stream continuity flag."""
        dataset = self.build_dataset(self.build_two_scenario_dataframe())

        frame_metas = [dataset.get_data_sample(index).frame_meta for index in range(len(dataset))]

        self.assertTrue(all(frame_meta is not None for frame_meta in frame_metas))
        self.assertEqual(
            [frame_meta.prev_exists for frame_meta in frame_metas if frame_meta is not None],
            [False, True, False, True, True],
        )

    def test_prev_exists_follows_the_previous_sample_ids(self) -> None:
        """A frame continues its stream only when the previous row is its previous sample."""
        dataset = self.build_dataset(self.build_two_scenario_dataframe())

        np.testing.assert_array_equal(
            dataset.prev_exists, np.array([0.0, 1.0, 0.0, 1.0, 1.0], dtype=np.float32)
        )

    def test_prev_exists_breaks_the_stream_at_filtered_frames(self) -> None:
        """A frame whose predecessor was dropped by the camera filter starts a new stream."""
        dataset = self.build_dataset(
            self.build_dataset_records_dataframe(
                [
                    self.build_record("scene-a", "a0", None),
                    self.build_record("scene-a", "a1", "a0", camera_names=["CAM_FRONT"]),
                    self.build_record("scene-a", "a2", "a1"),
                ]
            )
        )

        self.assertEqual(len(dataset), 2)
        np.testing.assert_array_equal(dataset.prev_exists, np.array([0.0, 0.0], dtype=np.float32))
        self.assertEqual(dataset.scene_index_groups(), [[0, 1]])

    def test_assign_dataset_records_rebuilds_the_streaming_metadata(self) -> None:
        """Records assigned later are filtered and regrouped like records given at init."""
        dataset = self.build_dataset(dataset_records_dataframe=None)

        dataset.assign_dataset_records(self.build_two_scenario_dataframe())

        self.assertEqual(len(dataset), 5)
        self.assertEqual(dataset.scene_index_groups(), [[0, 1], [2, 3, 4]])
        np.testing.assert_array_equal(
            dataset.prev_exists, np.array([0.0, 1.0, 0.0, 1.0, 1.0], dtype=np.float32)
        )


if __name__ == "__main__":
    unittest.main()
