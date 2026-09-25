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
from typing import Any, Mapping, Sequence

import polars as pl

from autoware_ml.databases.schemas.dataset_schemas import DatasetTableSchema
from autoware_ml.databases.schemas.image_frames import ImageFrameDatasetSchema
from autoware_ml.dataclasses.batch.sample_batch import ModelGTSample
from autoware_ml.datamodule.base_dataset import BaseDataset
from autoware_ml.types.dataset import SplitType

CAMERA_ORDER = ["CAM_FRONT", "CAM_BACK"]


class _Dataset(BaseDataset):
    """Concrete dataset so the base class filtering can be exercised."""

    def get_data_sample(self, index: int) -> ModelGTSample:
        raise NotImplementedError


class BaseDatasetCameraOrderFilterTest(unittest.TestCase):
    """Tests for dropping records whose image frames do not cover the camera order."""

    def build_image_frame(self, channel_name: str, image_path: str | None) -> Mapping[str, Any]:
        """Build one image frame struct entry with only the fields the filter reads."""
        return {
            ImageFrameDatasetSchema.image_sensor_channel_name.name: channel_name,
            ImageFrameDatasetSchema.image_path.name: image_path,
        }

    def build_time_step(self, channel_names: Sequence[str]) -> Sequence[Mapping[str, Any]]:
        """Build one time step holding a valid image frame for each channel."""
        return [self.build_image_frame(channel_name, "image.jpg") for channel_name in channel_names]

    def build_dataset_records_dataframe(
        self, image_frames_per_sample: Mapping[str, Any]
    ) -> pl.DataFrame:
        """Build a dataset records dataframe with one row per sample id."""
        return pl.DataFrame(
            {
                DatasetTableSchema.SAMPLE_ID.name: list(image_frames_per_sample.keys()),
                DatasetTableSchema.IMAGE_FRAMES.name: list(image_frames_per_sample.values()),
            },
            schema={
                DatasetTableSchema.SAMPLE_ID.name: DatasetTableSchema.SAMPLE_ID.dtype,
                DatasetTableSchema.IMAGE_FRAMES.name: DatasetTableSchema.IMAGE_FRAMES.dtype,
            },
        )

    def build_dataset(
        self,
        dataset_records_dataframe: pl.DataFrame | None,
        filter_frames_with_camera_order: bool = True,
        filter_all_time_steps: bool = False,
    ) -> _Dataset:
        """Build the dataset under test."""
        return _Dataset(
            database_root_path="/tmp/unused",
            max_num_3d_gt_bboxes=0,
            split_type=SplitType.TRAIN,
            dataset_records_dataframe=dataset_records_dataframe,
            transforms=None,
            camera_order=CAMERA_ORDER,
            filter_frames_with_camera_order=filter_frames_with_camera_order,
            filter_all_time_steps=filter_all_time_steps,
        )

    def build_previous_frame_incomplete_dataframe(self) -> pl.DataFrame:
        """Build records where only a previous frame, never the current one, misses a camera."""
        return self.build_dataset_records_dataframe(
            {
                "previous-missing-front": [
                    self.build_time_step(CAMERA_ORDER),
                    self.build_time_step(["CAM_BACK"]),
                ],
                "complete": [
                    self.build_time_step(CAMERA_ORDER),
                    self.build_time_step(CAMERA_ORDER),
                ],
            }
        )

    def kept_sample_ids(self, dataset: _Dataset) -> Sequence[str]:
        """Return the sample ids that survived filtering."""
        self.assertIsNotNone(dataset.dataset_records_dataframe)
        assert dataset.dataset_records_dataframe is not None
        return dataset.dataset_records_dataframe[DatasetTableSchema.SAMPLE_ID.name].to_list()

    def test_keeps_records_with_every_camera_in_every_time_step(self) -> None:
        """Current and previous frames holding all cameras pass, extra cameras are allowed."""
        dataset = self.build_dataset(
            self.build_dataset_records_dataframe(
                {
                    "single-step": [self.build_time_step(CAMERA_ORDER)],
                    "two-steps": [
                        self.build_time_step(CAMERA_ORDER),
                        self.build_time_step(CAMERA_ORDER),
                    ],
                    "extra-camera": [self.build_time_step(["CAM_FRONT", "CAM_BACK", "CAM_LEFT"])],
                }
            )
        )

        self.assertEqual(
            self.kept_sample_ids(dataset), ["single-step", "two-steps", "extra-camera"]
        )

    def test_drops_records_missing_a_camera_in_the_current_frame(self) -> None:
        """A missing camera at time step 0 drops the record."""
        dataset = self.build_dataset(
            self.build_dataset_records_dataframe(
                {
                    "complete": [self.build_time_step(CAMERA_ORDER)],
                    "missing-back": [self.build_time_step(["CAM_FRONT"])],
                }
            )
        )

        self.assertEqual(self.kept_sample_ids(dataset), ["complete"])

    def test_keeps_records_missing_a_camera_only_in_a_previous_frame_by_default(self) -> None:
        """By default only the current frame at index 0 is checked."""
        dataset = self.build_dataset(self.build_previous_frame_incomplete_dataframe())

        self.assertEqual(self.kept_sample_ids(dataset), ["previous-missing-front", "complete"])

    def test_drops_records_missing_a_camera_in_a_previous_frame_with_all_time_steps(
        self,
    ) -> None:
        """With ``filter_all_time_steps`` every previous frame is checked too."""
        dataset = self.build_dataset(
            self.build_previous_frame_incomplete_dataframe(), filter_all_time_steps=True
        )

        self.assertEqual(self.kept_sample_ids(dataset), ["complete"])

    def test_drops_records_whose_camera_has_a_null_image_path(self) -> None:
        """A camera entry that is present but has no image path counts as missing."""
        dataset = self.build_dataset(
            self.build_dataset_records_dataframe(
                {
                    "null-path": [
                        [
                            self.build_image_frame("CAM_FRONT", "image.jpg"),
                            self.build_image_frame("CAM_BACK", None),
                        ]
                    ],
                    "complete": [self.build_time_step(CAMERA_ORDER)],
                }
            )
        )

        self.assertEqual(self.kept_sample_ids(dataset), ["complete"])

    def test_drops_records_without_image_frames(self) -> None:
        """Null and empty image frame lists cannot provide any camera."""
        dataset = self.build_dataset(
            self.build_dataset_records_dataframe(
                {
                    "null-frames": None,
                    "empty-frames": [],
                    "complete": [self.build_time_step(CAMERA_ORDER)],
                }
            )
        )

        self.assertEqual(self.kept_sample_ids(dataset), ["complete"])

    def test_keeps_every_record_when_filtering_is_disabled(self) -> None:
        """The flag turns the filter off entirely."""
        dataset = self.build_dataset(
            self.build_dataset_records_dataframe(
                {
                    "missing-back": [self.build_time_step(["CAM_FRONT"])],
                    "complete": [self.build_time_step(CAMERA_ORDER)],
                }
            ),
            filter_frames_with_camera_order=False,
        )

        self.assertEqual(self.kept_sample_ids(dataset), ["missing-back", "complete"])

    def test_rejects_filtering_without_a_camera_order(self) -> None:
        """The filter cannot run without knowing which cameras are required."""
        with self.assertRaises(ValueError):
            _Dataset(
                database_root_path="/tmp/unused",
                max_num_3d_gt_bboxes=0,
                split_type=SplitType.TRAIN,
                dataset_records_dataframe=None,
                transforms=None,
                camera_order=None,
                filter_frames_with_camera_order=True,
            )

    def test_filters_records_assigned_after_initialization(self) -> None:
        """Records assigned later go through the same filter."""
        dataset = self.build_dataset(dataset_records_dataframe=None)

        dataset.assign_dataset_records(
            self.build_dataset_records_dataframe(
                {
                    "missing-back": [self.build_time_step(["CAM_FRONT"])],
                    "complete": [self.build_time_step(CAMERA_ORDER)],
                }
            )
        )

        self.assertEqual(self.kept_sample_ids(dataset), ["complete"])


if __name__ == "__main__":
    unittest.main()
