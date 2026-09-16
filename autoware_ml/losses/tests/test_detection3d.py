"""Tests for detection3d loss functions."""

from __future__ import annotations

import torch

from autoware_ml.losses.detection3d.focal import SigmoidFocalLoss
from autoware_ml.losses.detection3d.gaussian_focal import GaussianFocalLoss


def test_sigmoid_focal_loss_clamps_avg_factor() -> None:
    loss_fn = SigmoidFocalLoss()
    logits = torch.zeros((2, 2), dtype=torch.float32)
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)

    unclamped = loss_fn(logits, targets)
    clamped = loss_fn(logits, targets, avg_factor=0.0)

    assert torch.isclose(clamped, unclamped)


def test_sigmoid_focal_loss_broadcasts_query_weights() -> None:
    loss_fn = SigmoidFocalLoss()
    logits = torch.tensor([[2.0, -1.0], [0.5, 3.0]], dtype=torch.float32)
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    weights = torch.tensor([1.0, 0.0], dtype=torch.float32)

    weighted = loss_fn(logits, targets, weights=weights)
    expected = loss_fn(logits[:1], targets[:1])

    assert torch.isclose(weighted, expected)


def test_gaussian_focal_loss_handles_zero_positive_heatmap() -> None:
    loss_fn = GaussianFocalLoss()
    prediction = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
    target = torch.zeros_like(prediction)

    loss = loss_fn(prediction, target)

    assert torch.isfinite(loss)
    assert loss > 0


def test_sigmoid_focal_loss_applies_per_class_weights() -> None:
    """A zero per-class weight drops that class of the query, matching a loss without the column."""
    logits = torch.tensor([[2.0, -1.0], [0.5, 0.5], [-3.0, 1.5]])
    targets = torch.tensor([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
    weights = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    loss_fn = SigmoidFocalLoss()

    weighted = loss_fn(logits, targets, weights)
    first_class_only = loss_fn(logits[:, :1], targets[:, :1])

    assert torch.isclose(weighted, first_class_only)


def test_gaussian_focal_loss_weights_drop_cells_and_their_positives() -> None:
    """A zero class weight removes the class from the loss and from the normalizing peak count."""
    torch.manual_seed(0)
    prediction = torch.randn(2, 2, 4, 4)
    target = torch.zeros(2, 2, 4, 4)
    target[0, 0, 1, 1] = 1.0
    target[1, 1, 2, 2] = 1.0
    target[1, 1, 2, 3] = 0.5
    # (batch_size, num_classes, 1, 1) weights dropping the second class everywhere.
    weights = torch.tensor([[1.0, 0.0], [1.0, 0.0]])[:, :, None, None]
    loss_fn = GaussianFocalLoss()

    weighted = loss_fn(prediction, target, weights)
    first_class_only = loss_fn(prediction[:, :1], target[:, :1])

    assert torch.isclose(weighted, first_class_only)
    assert not torch.isclose(weighted, loss_fn(prediction, target))
