"""Tests for the criticality metrics (B1 critical FP/FN, B2 weighted mAP) on the
reachability model: hand-built states carry a per-box TTC (as the suite's
collision provider would), plus the shared matching / weighted-AP helpers."""

from __future__ import annotations

from dataclasses import replace
from math import inf, isnan

import numpy as np
import pytest
import torch

from autoware_ml.metrics.base import EvalStage
from autoware_ml.metrics.detection3d.collision_weighted_map import CollisionWeightedMeanAP
from autoware_ml.metrics.detection3d.criticality import greedy_match, weighted_average_precision
from autoware_ml.metrics.detection3d.critical_fp_fn import CriticalFPFN
from autoware_ml.metrics.detection3d.matching import Detection3DState
from autoware_ml.metrics.detection3d.structures import Detection3DSample
from autoware_ml.metrics.tests.conftest import collision_box


def _sample(preds, pred_scores, pred_labels, gts, gt_labels, pred_ttc=None, gt_ttc=None):
    """A frame with per-box TTC; default TTC = 1.0 s (all boxes critical/weighted)."""
    n_pred, n_gt = len(preds), len(gts)
    pred_ttc = [1.0] * n_pred if pred_ttc is None else pred_ttc
    gt_ttc = [1.0] * n_gt if gt_ttc is None else gt_ttc
    return Detection3DSample(
        pred_boxes=torch.tensor(preds, dtype=torch.float32) if preds else torch.zeros((0, 9)),
        pred_scores=torch.tensor(pred_scores, dtype=torch.float32)
        if pred_scores
        else torch.zeros(0),
        pred_labels=torch.tensor(pred_labels, dtype=torch.long)
        if pred_labels
        else torch.zeros(0, dtype=torch.long),
        gt_boxes=torch.tensor(gts, dtype=torch.float32) if gts else torch.zeros((0, 9)),
        gt_labels=torch.tensor(gt_labels, dtype=torch.long)
        if gt_labels
        else torch.zeros(0, dtype=torch.long),
        pred_ttc=torch.tensor(pred_ttc, dtype=torch.float32) if n_pred else torch.zeros(0),
        gt_ttc=torch.tensor(gt_ttc, dtype=torch.float32) if n_gt else torch.zeros(0),
    )


def test_greedy_match_basic() -> None:
    tp, matched = greedy_match(np.array([[0.0, 0.0]]), np.array([[0.5, 0.0]]), np.array([0.9]), 2.0)
    assert tp.tolist() == [True]
    assert matched.tolist() == [0]


def test_weighted_ap_all_true_positive_is_one() -> None:
    ap = weighted_average_precision(
        np.array([1.0]), np.array([True]), np.array([0.9]), total_gt_weight=1.0
    )
    assert ap == pytest.approx(1.0)


def test_critical_fp_fn_counts_phantoms_and_misses() -> None:
    state = Detection3DState(
        samples=[
            _sample([collision_box(10.0)], [0.9], [0], [collision_box(10.0)], [0]),
            _sample(
                [collision_box(10.0), collision_box(30.0)],
                [0.9, 0.9],
                [0, 0],
                [collision_box(10.0)],
                [0],
            ),  # +1 FP
            _sample([], [], [], [collision_box(10.0)], [0]),  # +1 FN
        ],
        class_names=("car",),
    )
    out = CriticalFPFN(confidences=(0.5,), match_threshold=2.0).evaluate(state, EvalStage.TEST)
    assert out["critical_fp_conf0p5"] == pytest.approx(1 / 3)
    assert out["critical_fn_conf0p5"] == pytest.approx(1 / 3)
    assert out["critical_fp_car_conf0p5"] == pytest.approx(1 / 3)
    assert out["critical_fn_car_conf0p5"] == pytest.approx(1 / 3)


def test_critical_fp_fn_excludes_unreachable() -> None:
    # A phantom and a miss that are NOT reachable within the horizon (TTC = inf) are
    # not critical, neither is counted.
    state = Detection3DState(
        samples=[
            _sample([collision_box(30.0)], [0.9], [0], [], [], pred_ttc=[inf]),  # unreachable FP
            _sample([], [], [], [collision_box(30.0)], [0], gt_ttc=[inf]),  # unreachable FN
        ],
        class_names=("car",),
    )
    out = CriticalFPFN(confidences=(0.5,)).evaluate(state, EvalStage.TEST)
    assert out["critical_fp_conf0p5"] == pytest.approx(0.0)
    assert out["critical_fn_conf0p5"] == pytest.approx(0.0)


def test_critical_fp_fn_excludes_uncovered_frames() -> None:
    # A frame whose scene has no lanelet map (ttc_covered=False) neither counts
    # its boxes nor inflates the denominator, mirroring region-filter coverage.
    covered_fp = _sample(
        [collision_box(10.0), collision_box(30.0)], [0.9, 0.9], [0, 0], [collision_box(10.0)], [0]
    )
    uncovered = replace(
        _sample(
            [collision_box(30.0)],
            [0.9],
            [0],
            [collision_box(50.0)],
            [0],
            pred_ttc=[inf],
            gt_ttc=[inf],
        ),
        ttc_covered=False,
    )
    state = Detection3DState(samples=[covered_fp, uncovered], class_names=("car",))
    out = CriticalFPFN(confidences=(0.5,), match_threshold=2.0).evaluate(state, EvalStage.TEST)
    # Denominator is 1 (the covered frame), not 2.
    assert out["critical_fp_conf0p5"] == pytest.approx(1.0)

    weighted = CollisionWeightedMeanAP(thresholds=(2.0,), decay=0.5).evaluate(state, EvalStage.TEST)
    # The uncovered frame's phantom prediction contributes nothing: the covered
    # frame's perfect match still scores, so cw_mAP stays finite and positive.
    assert weighted["cw_mAP"] > 0.0


def test_critical_fp_fn_no_coverage_is_nan() -> None:
    frame = replace(
        _sample([collision_box(10.0)], [0.9], [0], [collision_box(10.0)], [0]), ttc_covered=False
    )
    state = Detection3DState(samples=[frame], class_names=("car",))
    out = CriticalFPFN(confidences=(0.5,)).evaluate(state, EvalStage.TEST)
    assert isnan(out["critical_fp_conf0p5"])  # no basis, never a fake zero


def test_critical_fp_fn_confidence_gate() -> None:
    state = Detection3DState(
        samples=[_sample([collision_box(30.0)], [0.4], [0], [], [])],
        class_names=("car",),
    )
    out = CriticalFPFN(confidences=(0.5,)).evaluate(state, EvalStage.TEST)
    assert out["critical_fp_conf0p5"] == pytest.approx(0.0)


def test_collision_weighted_map_perfect_detection3d() -> None:
    state = Detection3DState(
        samples=[_sample([collision_box(10.0)], [0.9], [0], [collision_box(10.0)], [0])],
        class_names=("car",),
    )
    out = CollisionWeightedMeanAP(thresholds=(2.0,), decay=0.5).evaluate(state, EvalStage.TEST)
    assert out["cw_mAP"] == pytest.approx(1.0)
    assert out["cw_mAP_car"] == pytest.approx(1.0)


def test_collision_weighted_map_unreachable_gt_has_no_weight() -> None:
    # All GT unreachable (TTC inf) -> total GT weight 0 -> AP is NaN (nothing to score).
    state = Detection3DState(
        samples=[
            _sample(
                [collision_box(10.0)],
                [0.9],
                [0],
                [collision_box(10.0)],
                [0],
                gt_ttc=[inf],
                pred_ttc=[inf],
            )
        ],
        class_names=("car",),
    )
    out = CollisionWeightedMeanAP(thresholds=(2.0,), decay=0.5).evaluate(state, EvalStage.TEST)
    assert isnan(out["cw_mAP_car"])


@pytest.mark.parametrize(
    "build",
    [
        lambda: CriticalFPFN(confidences=()),
        lambda: CriticalFPFN(confidences=(1.5,)),
        lambda: CriticalFPFN(match_threshold=0.0),
        lambda: CollisionWeightedMeanAP(thresholds=()),
        lambda: CollisionWeightedMeanAP(decay=float("inf")),
    ],
)
def test_criticality_metrics_reject_unusable_parameters(build) -> None:
    with pytest.raises(ValueError):
        build()


def _with_ignored(sample: Detection3DSample, boxes: list) -> Detection3DSample:
    """The frame with the given class 0 boxes as its ignored ground truth."""
    return replace(
        sample,
        ignore_boxes=torch.tensor(boxes, dtype=torch.float32).reshape(-1, 9),
        ignore_labels=torch.zeros(len(boxes), dtype=torch.long),
    )


def _phantom_frame() -> Detection3DSample:
    """A matched prediction and, scored above it, a prediction where no counted box is."""
    return _sample(
        [collision_box(10.0), collision_box(30.0)],
        [0.8, 0.9],
        [0, 0],
        [collision_box(10.0)],
        [0],
    )


def test_critical_fp_skips_predictions_on_ignored_boxes() -> None:
    dropped = Detection3DState(samples=[_phantom_frame()], class_names=("car",))
    ignored = Detection3DState(
        samples=[_with_ignored(_phantom_frame(), [collision_box(30.0)])], class_names=("car",)
    )
    metric = CriticalFPFN(confidences=(0.5,), match_threshold=2.0)
    assert metric.evaluate(dropped, EvalStage.TEST)["critical_fp_conf0p5"] == pytest.approx(1.0)
    out = metric.evaluate(ignored, EvalStage.TEST)
    assert out["critical_fp_conf0p5"] == pytest.approx(0.0)
    # An ignored box nobody matches is never a miss
    assert out["critical_fn_conf0p5"] == pytest.approx(0.0)


def test_collision_weighted_map_skips_predictions_on_ignored_boxes() -> None:
    dropped = Detection3DState(samples=[_phantom_frame()], class_names=("car",))
    ignored = Detection3DState(
        samples=[_with_ignored(_phantom_frame(), [collision_box(30.0)])], class_names=("car",)
    )
    metric = CollisionWeightedMeanAP(thresholds=(2.0,), decay=0.5)
    assert metric.evaluate(dropped, EvalStage.TEST)["cw_mAP"] < 1.0
    assert metric.evaluate(ignored, EvalStage.TEST)["cw_mAP"] == pytest.approx(1.0)
