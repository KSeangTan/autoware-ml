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

"""Unit tests for the T4 dataset 3D detection task."""

from typing import Any, Mapping, Sequence

import unittest

import polars as pl

from autoware_ml.databases.schemas.box3d_schemas import Box3DDatasetSchema
from autoware_ml.databases.schemas.dataset_schemas import DatasetTableSchema
from autoware_ml.datamodule.multi_task.dataclasses.multi_task_samples import MultiTaskGTSample
from autoware_ml.datamodule.multi_task.t4dataset.detection3d import T4Detection3DTask
from autoware_ml.geometry.bbox_3d.base_bbox3d import BaseBBoxes3D
from autoware_ml.types.geometry import Box3DFieldIndex

# The task only reads the dataset records, so the root path is never touched by these tests.
DATABASE_ROOT_PATH = "/nonexistent/database/root"

# Logger of the module under test, used to capture what log_dataset_info reports.
DETECTION3D_LOGGER_NAME = "autoware_ml.datamodule.multi_task.t4dataset.detection3d"


class BaseT4Detection3DTaskTestCase(unittest.TestCase):
    """Shared dataset record builders and optional field assertions for T4Detection3DTask."""

    def build_box3d_record(
        self,
        label_name: str,
        label_index: int,
        bbox_params: Sequence[float] | None = None,
        num_lidar_points: int = 10,
        valid: bool = True,
        attributes: Sequence[str] | None = None,
    ) -> Mapping[str, Any]:
        """Build a single 3D bounding box record following the dataset schema.

        Args:
            label_name: The label name the box carries.
            label_index: The label index the box carries.
            bbox_params: The ten box parameters ``[x, y, z, l, w, h, yaw, vx, vy, vz]``.
                Defaults to a physically sane, stationary box at the origin.
            num_lidar_points: The annotated lidar point count of the box.
            valid: The dataset validity flag of the box.
            attributes: The attributes the box carries.

        Returns:
            One ``boxes_3d`` struct entry.
        """
        if bbox_params is None:
            bbox_params = [0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0, 0.0, 0.0, 0.0]

        return {
            Box3DDatasetSchema.BOX3D_PARAMS.name: list(bbox_params),
            Box3DDatasetSchema.BOX3D_INSTANCE_ID.name: f"instance-{label_name}-{label_index}",
            Box3DDatasetSchema.BOX3D_DATASET_LABEL_NAME.name: label_name,
            Box3DDatasetSchema.BOX3D_LABEL_NAME.name: label_name,
            Box3DDatasetSchema.BOX3D_LABEL_INDEX.name: label_index,
            Box3DDatasetSchema.BOX3D_NUM_LIDAR_POINTS.name: num_lidar_points,
            Box3DDatasetSchema.BOX3D_NUM_RADAR_POINTS.name: 0,
            Box3DDatasetSchema.BOX3D_VALID.name: valid,
            Box3DDatasetSchema.BOX3D_ATTRIBUTES.name: list(attributes or []),
            Box3DDatasetSchema.BOX3D_COORDINATE.name: "lidar",
        }

    def build_dataset_records_dataframe(
        self,
        samples: Sequence[Sequence[Mapping[str, Any]]],
    ) -> pl.DataFrame:
        """Build a dataset records dataframe holding one row per sample.

        Args:
            samples: The 3D bounding box records of every sample, one inner sequence per row.

        Returns:
            Polars DataFrame typed with the dataset ``boxes_3d`` schema.
        """
        return pl.DataFrame(
            {DatasetTableSchema.BOXES_3D.name: list(samples)},
            schema={DatasetTableSchema.BOXES_3D.name: DatasetTableSchema.BOXES_3D.dtype},
        )

    def build_task(
        self,
        samples: Sequence[Sequence[Mapping[str, Any]]],
    ) -> T4Detection3DTask:
        """Build a task over the given samples.

        Args:
            samples: The 3D bounding box records of every sample, one inner sequence per row.

        Returns:
            The T4Detection3DTask under test.
        """
        return T4Detection3DTask(
            database_root_path=DATABASE_ROOT_PATH,
            dataset_records_dataframe=self.build_dataset_records_dataframe(samples),
        )

    def assert_bboxes_3d(self, multi_task_gt_sample: MultiTaskGTSample) -> BaseBBoxes3D:
        """Assert the sample carries 3D bounding boxes, and return them.

        ``detection3d_gt_bboxes_3d`` is optional on the sample, so every dereference below goes
        through this helper: a sample holding no annotation must still carry an empty container,
        never None, otherwise the transform pipeline fails its own required key validation.

        Args:
            multi_task_gt_sample: The sample the task built.

        Returns:
            The 3D bounding boxes the sample carries.
        """
        detection3d_gt_bboxes_3d = multi_task_gt_sample.detection3d_gt_bboxes_3d
        self.assertIsNotNone(
            detection3d_gt_bboxes_3d,
            "The task did not populate the detection3d_gt_bboxes_3d field of the sample.",
        )
        # Narrow the optional, so what follows is checked against the concrete type.
        assert detection3d_gt_bboxes_3d is not None
        return detection3d_gt_bboxes_3d

    def assert_bbox_attributes(self, bboxes_3d: BaseBBoxes3D) -> Sequence[Sequence[str]]:
        """Assert the bounding boxes carry attributes, and return them.

        ``bbox_attributes`` is optional because not every dataset provides attributes, but the
        T4 dataset always does, so the task must forward them.

        Args:
            bboxes_3d: The bounding boxes the task built.

        Returns:
            The attributes of every bounding box.
        """
        bbox_attributes = bboxes_3d.bbox_attributes
        self.assertIsNotNone(
            bbox_attributes, "The task did not forward the attributes of the bounding boxes."
        )
        # Narrow the optional, so what follows is checked against the concrete type.
        assert bbox_attributes is not None
        return bbox_attributes


class T4Detection3DTaskDataSampleTest(BaseT4Detection3DTaskTestCase):
    """Tests for how T4Detection3DTask turns dataset records into a sample."""

    def test_builds_detection_targets(self) -> None:
        """A record becomes bounding boxes carrying their labels, names and point counts."""
        task = self.build_task(
            samples=[
                [
                    self.build_box3d_record(
                        label_name="car",
                        label_index=0,
                        bbox_params=[1.0, 2.0, 3.0, 4.0, 1.5, 1.75, 0.125, 0.5, -0.25, 0.0],
                        num_lidar_points=12,
                    ),
                    self.build_box3d_record(
                        label_name="pedestrian",
                        label_index=1,
                        num_lidar_points=3,
                    ),
                ]
            ],
        )

        detection3d_gt_bboxes_3d = self.assert_bboxes_3d(task.get_data_sample(0))

        self.assertEqual(len(detection3d_gt_bboxes_3d), 2)
        self.assertEqual(detection3d_gt_bboxes_3d.bbox_labels.tolist(), [0, 1])
        self.assertEqual(list(detection3d_gt_bboxes_3d.bbox_label_names), ["car", "pedestrian"])
        self.assertEqual(detection3d_gt_bboxes_3d.bbox_num_lidar_points.tolist(), [12, 3])

    def test_forwards_every_box_parameter(self) -> None:
        """The ten box parameters, velocity included, reach the sample untouched."""
        # Every value is exactly representable in float32, so the round trip is a strict equality.
        bbox_params = [1.0, 2.0, 3.0, 4.0, 1.5, 1.75, 0.125, 0.5, -0.25, 0.25]
        task = self.build_task(
            samples=[[self.build_box3d_record("car", 0, bbox_params=bbox_params)]],
        )

        detection3d_gt_bboxes_3d = self.assert_bboxes_3d(task.get_data_sample(0))

        self.assertEqual(detection3d_gt_bboxes_3d.bbox_params.shape, (1, len(Box3DFieldIndex)))
        self.assertEqual(detection3d_gt_bboxes_3d.bbox_params[0].tolist(), bbox_params)
        # Velocity is forwarded verbatim, the physical sanity rules live in the filter pipeline.
        self.assertEqual(detection3d_gt_bboxes_3d.velocity[0].tolist(), [0.5, -0.25, 0.25])

    def test_forwards_the_box_attributes(self) -> None:
        """Attributes reach the sample so the attribute filter downstream can act on them."""
        task = self.build_task(
            samples=[
                [
                    self.build_box3d_record("bicycle", 0, attributes=["vehicle_state.parked"]),
                    self.build_box3d_record("bicycle", 0, attributes=[]),
                ]
            ],
        )

        detection3d_gt_bboxes_3d = self.assert_bboxes_3d(task.get_data_sample(0))

        self.assertEqual(
            self.assert_bbox_attributes(detection3d_gt_bboxes_3d),
            [["vehicle_state.parked"], []],
        )

    def test_reads_the_requested_sample_index(self) -> None:
        """Each row of the dataset records is addressed by its own index."""
        task = self.build_task(
            samples=[
                [self.build_box3d_record("car", 0)],
                [
                    self.build_box3d_record("pedestrian", 1),
                    self.build_box3d_record("pedestrian", 1),
                ],
            ],
        )

        first_bboxes_3d = self.assert_bboxes_3d(task.get_data_sample(0))
        second_bboxes_3d = self.assert_bboxes_3d(task.get_data_sample(1))

        self.assertEqual(list(first_bboxes_3d.bbox_label_names), ["car"])
        self.assertEqual(list(second_bboxes_3d.bbox_label_names), ["pedestrian", "pedestrian"])

    def test_leaves_the_other_sample_fields_unpopulated(self) -> None:
        """The task fills in the detection branch only, the pipeline loads the rest."""
        task = self.build_task(samples=[[self.build_box3d_record("car", 0)]])

        multi_task_gt_sample = task.get_data_sample(0)

        self.assertIsNone(multi_task_gt_sample.lidar_point_cloud_samples)
        self.assertIsNone(multi_task_gt_sample.point_cloud_data)
        self.assertIsNone(multi_task_gt_sample.image_samples)
        self.assertIsNone(multi_task_gt_sample.camera_image_data)
        self.assertIsNone(multi_task_gt_sample.segmentation3d_gt_sample)

    def test_builds_an_empty_container_for_a_sample_without_boxes(self) -> None:
        """A sample holding no annotation yields empty bounding boxes rather than None."""
        task = self.build_task(samples=[[]])

        detection3d_gt_bboxes_3d = self.assert_bboxes_3d(task.get_data_sample(0))

        self.assertEqual(len(detection3d_gt_bboxes_3d), 0)
        self.assertEqual(detection3d_gt_bboxes_3d.bbox_params.shape, (0, len(Box3DFieldIndex)))
        self.assertEqual(list(detection3d_gt_bboxes_3d.bbox_label_names), [])
        self.assertEqual(self.assert_bbox_attributes(detection3d_gt_bboxes_3d), [])

    def test_rejects_a_task_without_dataset_records(self) -> None:
        """Reading a sample from a task built without records is reported, not guessed at."""
        task = T4Detection3DTask(
            database_root_path=DATABASE_ROOT_PATH,
            dataset_records_dataframe=None,
        )

        with self.assertRaisesRegex(ValueError, "Dataset records dataframe is not available"):
            task.get_data_sample(0)


class T4Detection3DTaskValidMaskTest(BaseT4Detection3DTaskTestCase):
    """Tests that T4Detection3DTask hands every annotated box to the pipeline."""

    def build_partially_valid_samples(self) -> Sequence[Sequence[Mapping[str, Any]]]:
        """Build one sample holding a valid, an invalid, and a second valid bounding box.

        Returns:
            The dataset records of a single sample.
        """
        return [
            [
                self.build_box3d_record(
                    "car", 0, num_lidar_points=12, valid=True, attributes=["vehicle_state.moving"]
                ),
                self.build_box3d_record(
                    "truck", 1, num_lidar_points=5, valid=False, attributes=["vehicle_state.parked"]
                ),
                self.build_box3d_record(
                    "pedestrian", 2, num_lidar_points=8, valid=True, attributes=[]
                ),
            ]
        ]

    def test_keeps_the_boxes_whose_validity_flag_is_unset(self) -> None:
        """The validity flag of the dataset no longer drops boxes at the task level."""
        task = self.build_task(samples=self.build_partially_valid_samples())

        detection3d_gt_bboxes_3d = self.assert_bboxes_3d(task.get_data_sample(0))

        self.assertEqual(
            list(detection3d_gt_bboxes_3d.bbox_label_names), ["car", "truck", "pedestrian"]
        )

    def test_keeps_the_per_box_fields_aligned(self) -> None:
        """Labels, point counts and attributes stay aligned with the boxes they describe."""
        task = self.build_task(samples=self.build_partially_valid_samples())

        detection3d_gt_bboxes_3d = self.assert_bboxes_3d(task.get_data_sample(0))

        self.assertEqual(detection3d_gt_bboxes_3d.bbox_labels.tolist(), [0, 1, 2])
        self.assertEqual(detection3d_gt_bboxes_3d.bbox_num_lidar_points.tolist(), [12, 5, 8])
        self.assertEqual(
            self.assert_bbox_attributes(detection3d_gt_bboxes_3d),
            [["vehicle_state.moving"], ["vehicle_state.parked"], []],
        )


class T4Detection3DTaskDatasetRecordsTest(BaseT4Detection3DTaskTestCase):
    """Tests for how T4Detection3DTask prepares and reports its dataset records."""

    def test_pre_filter_keeps_only_the_boxes_3d_column(self) -> None:
        """Columns the 3D detection task does not read are dropped up front."""
        dataset_records_dataframe = self.build_dataset_records_dataframe(
            samples=[[self.build_box3d_record("car", 0)]]
        ).with_columns(pl.Series(DatasetTableSchema.SAMPLE_ID.name, ["sample-0"], dtype=pl.String))

        task = T4Detection3DTask(
            database_root_path=DATABASE_ROOT_PATH,
            dataset_records_dataframe=dataset_records_dataframe,
        )

        self.assertIsNotNone(task.dataset_records_dataframe)
        assert task.dataset_records_dataframe is not None
        self.assertEqual(task.dataset_records_dataframe.columns, [DatasetTableSchema.BOXES_3D.name])

    def test_pre_filter_passes_through_missing_dataset_records(self) -> None:
        """A task built without records keeps None rather than an empty dataframe."""
        task = T4Detection3DTask(
            database_root_path=DATABASE_ROOT_PATH,
            dataset_records_dataframe=None,
        )

        self.assertIsNone(task.dataset_records_dataframe)

    def test_str_names_the_task(self) -> None:
        """The task identifies itself by name, which the dataset logs rely on."""
        task = self.build_task(samples=[[self.build_box3d_record("car", 0)]])

        self.assertEqual(str(task), "T4Detection3DTask")

    def test_log_dataset_info_reports_total_and_lidar_point_counts(self) -> None:
        """The logged summary separates the annotated boxes from those holding lidar points."""
        task = self.build_task(
            samples=[
                [
                    self.build_box3d_record("car", 0, num_lidar_points=10),
                    self.build_box3d_record("car", 0, num_lidar_points=0),
                    self.build_box3d_record("pedestrian", 1, num_lidar_points=3),
                ]
            ],
        )

        with self.assertLogs(DETECTION3D_LOGGER_NAME, level="INFO") as captured_logs:
            task.log_dataset_info()

        logged_output = "\n".join(captured_logs.output)
        self.assertIn("Number of bboxes per class in the dataset: ", logged_output)
        self.assertIn(
            "Number of bboxes after filtering num_lidar_points > 0 per class: ", logged_output
        )
        self.assertIn("'car': 2", logged_output)
        self.assertIn("'car': 1", logged_output)
        self.assertIn("'pedestrian': 1", logged_output)

    def test_log_dataset_info_warns_without_dataset_records(self) -> None:
        """A task built without records says so instead of failing on the missing dataframe."""
        task = T4Detection3DTask(
            database_root_path=DATABASE_ROOT_PATH,
            dataset_records_dataframe=None,
        )

        with self.assertLogs(DETECTION3D_LOGGER_NAME, level="WARNING") as captured_logs:
            task.log_dataset_info()

        self.assertIn("Dataset records dataframe is not available.", captured_logs.output[0])


if __name__ == "__main__":
    unittest.main()
