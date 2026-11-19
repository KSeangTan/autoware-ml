"""Counts and recall of the ground truth a suite ignores for its point count."""

from __future__ import annotations

import numpy as np

from autoware_ml.metrics.base import EvalStage, Metric, number_token
from autoware_ml.metrics.detection3d.matching import Detection3DState
from autoware_ml.metrics.detection3d.naming import label_metric_name
from autoware_ml.metrics.detection3d.structures import DEFAULT_TP_THRESHOLD


class IgnoredBoxes(Metric[Detection3DState]):
    """How many ignored ground-truth boxes there are and how many the model still finds.

    A suite with ``low_point_policy: ignore`` keeps the ground truth below
    ``min_num_points`` as ignored boxes: a prediction matching one is neither a true
    nor a false positive. This metric reports what the AP family no longer shows
    about them. ``ignored_recall`` is the share of the ignored boxes a prediction
    matched, the measure of how well a model holds an object it cannot see in the
    current frame, as a temporal model does through an occlusion.
    ``num_ignored_preds`` is the number of predictions the ignored boxes took out of
    the false positives.

    The numbers come from the matching the AP family runs at ``match_threshold``:
    a prediction reaches an ignored box only when no counted box of its class is
    left within the threshold, and every ignored box takes one prediction. There is
    no precision over the ignored boxes, since a prediction that matches nothing
    belongs to neither set of boxes.

    A suite that drops the low-point boxes reports nothing here.
    """

    def __init__(
        self,
        match_threshold: float = DEFAULT_TP_THRESHOLD,
        score_thresholds: tuple[float, ...] | list[float] = (0.3, 0.5),
        stages: tuple[str, ...] | list[str] = ("val", "test"),
        filter=None,
    ) -> None:
        """Validate the operating points.

        Args:
            match_threshold: Match threshold in meters the ignored matches are read at.
            score_thresholds: Score floors the recall is also reported at, next to the recall
                over every prediction.
            stages: Stage names this metric reports for, as in :class:`Metric`.
            filter: Optional selection axis, as in :class:`Metric`.
        """
        super().__init__(stages, filter=filter)
        self.match_threshold = float(match_threshold)
        self.score_thresholds = tuple(float(score) for score in score_thresholds)
        if self.match_threshold <= 0.0:
            raise ValueError("match_threshold must be > 0.")
        if any(not 0.0 <= score <= 1.0 for score in self.score_thresholds):
            raise ValueError("score_thresholds must lie in [0, 1].")

    def evaluate(self, state: Detection3DState, stage: EvalStage) -> dict[str, float]:
        """Count the ignored boxes and the predictions that matched them.

        Args:
            state: Detection state with cached match curves.
            stage: Evaluation stage requesting the metrics.

        Returns:
            Mapping of metric names to scalar values, empty when the suite ignores no box.
        """
        if not state.ignores_low_points:
            return {}

        labels = state.labels(full=True)
        totals = {}
        matched_scores = {}
        for label in labels:
            curve = state.match_curve(label, self.match_threshold)
            totals[label] = curve.total_ignored_gt
            matched_scores[label] = curve.ignored_scores

        all_scores = (
            np.concatenate(list(matched_scores.values())) if labels else np.zeros(0, np.float64)
        )
        total = sum(totals.values())
        report = {
            "num_ignored_gts": float(total),
            "num_ignored_preds": float(all_scores.shape[0]),
            "ignored_recall": _recall(all_scores, 0.0, total),
        }
        for score in self.score_thresholds:
            report[f"ignored_recall_conf{number_token(score)}"] = _recall(all_scores, score, total)
        if stage is EvalStage.VAL:
            return report

        for label in labels:
            name = label_metric_name(label, state.class_names)
            report[f"num_ignored_gts_{name}"] = float(totals[label])
            report[f"ignored_recall_{name}"] = _recall(matched_scores[label], 0.0, totals[label])
        return report


def _recall(matched_scores: np.ndarray, score_threshold: float, total: int) -> float:
    """Share of ``total`` ignored boxes matched at or above ``score_threshold``, NaN for none."""
    if total == 0:
        return float("nan")
    return float(np.sum(matched_scores >= score_threshold)) / float(total)
