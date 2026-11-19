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

"""Unit tests for the ignored low-point ground truth: the matching that leaves the predictions
on ignored boxes out of the curve, the suite option that builds the ignored boxes, and the
metric reporting them."""

import math
import unittest

import numpy as np
import torch

from autoware_ml.metrics.base import EvalStage, MetricRange
from autoware_ml.metrics.detection3d.box_counts import BoxCounts
from autoware_ml.metrics.detection3d.criticality import ignored_matches
from autoware_ml.metrics.detection3d.ignored_boxes import IgnoredBoxes
from autoware_ml.metrics.detection3d.matching import PreparedLabel, cost_center
from autoware_ml.metrics.detection3d.mean_ap import MeanAP
from autoware_ml.metrics.detection3d.structures import Detection3DSample
from autoware_ml.metrics.detection3d.suite import Detection3DMetricSuite

_CLASS_NAMES = ("car", "pedestrian")


def _box(x: float, y: float = 0.0) -> list[float]:
    """Box centered at ``(x, y)``, in the 9-value box layout."""
    return [x, y, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]


def _boxes(rows: list[list[float]]) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.float32).reshape(-1, 9)


def _sample(
    preds: list[list[float]],
    scores: list[float],
    gts: list[list[float]],
    ignores: list[list[float]] | None,
) -> Detection3DSample:
    """One frame of class 0 with the given predictions, counted and ignored ground truth."""
    return Detection3DSample(
        pred_boxes=_boxes(preds),
        pred_scores=torch.tensor(scores, dtype=torch.float32),
        pred_labels=torch.zeros(len(preds), dtype=torch.long),
        gt_boxes=_boxes(gts),
        gt_labels=torch.zeros(len(gts), dtype=torch.long),
        ignore_boxes=None if ignores is None else _boxes(ignores),
        ignore_labels=None if ignores is None else torch.zeros(len(ignores), dtype=torch.long),
    )


class TestIgnoredMatching(unittest.TestCase):
    """The shared greedy matching with ignored ground truth."""

    def test_prediction_on_an_ignored_box_leaves_the_curve(self) -> None:
        """One true positive, one prediction on an ignored box, one false positive."""
        sample = _sample(
            preds=[_box(10.0), _box(20.0), _box(40.0)],
            scores=[0.9, 0.8, 0.7],
            gts=[_box(10.0)],
            ignores=[_box(20.0)],
        )
        curve = PreparedLabel([sample], 0, cost_center).match(2.0)

        np.testing.assert_allclose(curve.scores, [0.9, 0.7])
        self.assertEqual(curve.true_positive.tolist(), [True, False])
        self.assertEqual(curve.false_positive.tolist(), [False, True])
        self.assertEqual(curve.total_gt, 1)
        self.assertEqual(curve.total_ignored_gt, 1)
        np.testing.assert_allclose(curve.ignored_scores, [0.8])

    def test_without_ignored_boxes_the_prediction_is_a_false_positive(self) -> None:
        """Dropping the box leaves its prediction in the curve as a false positive."""
        sample = _sample(
            preds=[_box(10.0), _box(20.0)], scores=[0.9, 0.8], gts=[_box(10.0)], ignores=None
        )
        curve = PreparedLabel([sample], 0, cost_center).match(2.0)

        self.assertEqual(curve.false_positive.tolist(), [False, True])
        self.assertEqual(curve.total_ignored_gt, 0)
        self.assertEqual(curve.ignored_scores.shape[0], 0)

    def test_counted_box_wins_over_a_nearer_ignored_box(self) -> None:
        """A prediction takes the counted box in reach even when an ignored box is nearer."""
        sample = _sample(preds=[_box(10.0)], scores=[0.9], gts=[_box(11.5)], ignores=[_box(10.0)])
        curve = PreparedLabel([sample], 0, cost_center).match(2.0)

        self.assertEqual(curve.true_positive.tolist(), [True])
        self.assertEqual(curve.ignored_scores.shape[0], 0)

    def test_an_ignored_box_takes_one_prediction(self) -> None:
        """The duplicate on an ignored box stays a false positive."""
        sample = _sample(
            preds=[_box(20.0), _box(20.5)], scores=[0.9, 0.8], gts=[], ignores=[_box(20.0)]
        )
        curve = PreparedLabel([sample], 0, cost_center).match(2.0)

        np.testing.assert_allclose(curve.ignored_scores, [0.9])
        np.testing.assert_allclose(curve.scores, [0.8])
        self.assertEqual(curve.false_positive.tolist(), [True])

    def test_ignored_box_beyond_the_threshold_absorbs_nothing(self) -> None:
        """An ignored box farther than the threshold leaves the false positive in place."""
        sample = _sample(preds=[_box(20.0)], scores=[0.9], gts=[], ignores=[_box(23.0)])
        curve = PreparedLabel([sample], 0, cost_center).match(2.0)

        self.assertEqual(curve.false_positive.tolist(), [True])
        self.assertEqual(curve.total_ignored_gt, 1)

    def test_ignored_boxes_of_frames_without_predictions_are_counted(self) -> None:
        """The ignored total covers the frames that hold no prediction of the class."""
        samples = [
            _sample(preds=[], scores=[], gts=[], ignores=[_box(5.0), _box(6.0)]),
            _sample(preds=[_box(20.0)], scores=[0.9], gts=[], ignores=[_box(20.0)]),
        ]
        curve = PreparedLabel(samples, 0, cost_center).match(2.0)

        self.assertEqual(curve.total_ignored_gt, 3)
        self.assertEqual(curve.ignored_scores.shape[0], 1)

    def test_claims_do_not_cross_frames(self) -> None:
        """A prediction never takes the ignored box of another frame."""
        samples = [
            _sample(preds=[_box(20.0)], scores=[0.9], gts=[], ignores=[]),
            _sample(preds=[_box(50.0)], scores=[0.8], gts=[], ignores=[_box(20.0)]),
        ]
        curve = PreparedLabel(samples, 0, cost_center).match(2.0)

        self.assertEqual(curve.false_positive.tolist(), [True, True])


class TestIgnoredMatches(unittest.TestCase):
    """The helper of the components that match on their own."""

    def test_only_unmatched_predictions_take_ignored_boxes(self) -> None:
        """A true positive next to an ignored box is not marked, the unmatched one is."""
        mask = ignored_matches(
            ignore_centers=np.array([[20.0, 0.0]]),
            pred_centers=np.array([[20.0, 0.0], [20.5, 0.0], [60.0, 0.0]]),
            pred_scores=np.array([0.9, 0.8, 0.7]),
            is_tp=np.array([True, False, False]),
            threshold=2.0,
        )
        self.assertEqual(mask.tolist(), [False, True, False])

    def test_no_ignored_boxes_marks_nothing(self) -> None:
        """An empty set of ignored boxes takes no prediction."""
        mask = ignored_matches(
            np.zeros((0, 2)), np.array([[1.0, 0.0]]), np.array([0.9]), np.array([False]), 2.0
        )
        self.assertEqual(mask.tolist(), [False])


class TestSuitePolicy(unittest.TestCase):
    """The suite option and the metric over the ignored boxes."""

    def _build_suite(self, low_point_policy: str) -> Detection3DMetricSuite:
        return Detection3DMetricSuite(
            components=[BoxCounts(), MeanAP(), IgnoredBoxes(score_thresholds=(0.5,))],
            class_names=_CLASS_NAMES,
            min_num_points=2,
            low_point_policy=low_point_policy,
            ranges=(MetricRange("0-50m", 0.0, 50.0), MetricRange("50-100m", 50.0, 100.0)),
        )

    def _update(self, suite: Detection3DMetricSuite) -> None:
        """One frame: two counted cars, two low-point cars, one low-point pedestrian far away.

        The predictions hit one counted car, both low-point cars (one of them with a low
        score) and one empty place.
        """
        suite.update(
            {
                "predictions": [
                    {
                        "bboxes_3d": _boxes([_box(10.0), _box(20.0), _box(30.0), _box(45.0)]),
                        "scores_3d": torch.tensor([0.6, 0.8, 0.2, 0.7]),
                        "labels_3d": torch.tensor([0, 0, 0, 0]),
                    }
                ],
                "gt_boxes": [_boxes([_box(10.0), _box(15.0), _box(20.0), _box(30.0), _box(70.0)])],
                "gt_labels": [torch.tensor([0, 0, 0, 0, 1])],
                "gt_num_points": [torch.tensor([10, 10, 1, 0, 1])],
            }
        )

    def _compute(self, low_point_policy: str, stage: EvalStage = EvalStage.TEST) -> dict:
        suite = self._build_suite(low_point_policy)
        suite.bind_stage(stage)
        self._update(suite)
        return suite.compute()

    def test_rejects_an_unknown_policy(self) -> None:
        with self.assertRaisesRegex(ValueError, "low_point_policy"):
            self._build_suite("keep")

    def test_ignore_needs_a_point_threshold(self) -> None:
        with self.assertRaisesRegex(ValueError, "min_num_points"):
            Detection3DMetricSuite(
                components=[MeanAP()], class_names=_CLASS_NAMES, low_point_policy="ignore"
            )

    def test_ignored_boxes_are_no_ground_truth(self) -> None:
        """The counted ground truth is the same under both policies."""
        for policy in ("drop", "ignore"):
            report = self._compute(policy)
            self.assertEqual(report["num_gts_car_0m_50m"], 2.0, policy)
            self.assertEqual(report["num_preds_car_0m_50m"], 4.0, policy)

    def test_ignore_raises_the_ap_of_drop(self) -> None:
        """Two of the three false positives sit on ignored boxes."""
        drop = self._compute("drop")["mAP_car_0m_50m"]
        ignore = self._compute("ignore")["mAP_car_0m_50m"]
        self.assertGreater(ignore, drop)

    def test_reports_the_ignored_boxes(self) -> None:
        """Counts and recall of the ignored boxes, per range and per class."""
        report = self._compute("ignore")

        self.assertEqual(report["num_ignored_gts_0m_50m"], 2.0)
        self.assertEqual(report["num_ignored_preds_0m_50m"], 2.0)
        self.assertEqual(report["ignored_recall_0m_50m"], 1.0)
        self.assertEqual(report["ignored_recall_conf0p5_0m_50m"], 0.5)
        self.assertEqual(report["num_ignored_gts_car_0m_50m"], 2.0)
        self.assertEqual(report["ignored_recall_car_0m_50m"], 1.0)
        self.assertEqual(report["num_ignored_gts_pedestrian_0m_50m"], 0.0)
        self.assertTrue(math.isnan(report["ignored_recall_pedestrian_0m_50m"]))
        # The far bucket holds the unmatched low-point pedestrian
        self.assertEqual(report["num_ignored_gts_50m_100m"], 1.0)
        self.assertEqual(report["ignored_recall_50m_100m"], 0.0)

    def test_validation_reports_no_per_class_keys(self) -> None:
        report = self._compute("ignore", EvalStage.VAL)
        self.assertIn("ignored_recall_0m_50m", report)
        self.assertNotIn("ignored_recall_car_0m_50m", report)

    def test_drop_reports_no_ignored_keys(self) -> None:
        report = self._compute("drop")
        self.assertFalse([key for key in report if "ignored" in key])


if __name__ == "__main__":
    unittest.main()
