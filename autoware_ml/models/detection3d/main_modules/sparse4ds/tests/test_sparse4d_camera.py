"""Tests for the weights the camera branch of Sparse4D starts from."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from autoware_ml.models.detection3d.main_modules.sparse4ds.sparse4d_camera import Sparse4DCamera


def _camera_network(weights: Path | None = None) -> Sparse4DCamera:
    return Sparse4DCamera(
        nn.Sequential(nn.Conv2d(3, 4, kernel_size=1), nn.BatchNorm2d(4)),
        nn.Conv2d(4, 4, kernel_size=1),
        feature_strides=[1],
        weights=weights,
    )


def _state_dicts_equal(first: nn.Module, second: nn.Module) -> bool:
    first_state, second_state = first.state_dict(), second.state_dict()
    return first_state.keys() == second_state.keys() and all(
        torch.equal(first_state[key], second_state[key]) for key in first_state
    )


@pytest.mark.parametrize("prefix", ["", "camera_network."])
@pytest.mark.parametrize("wrapped", [False, True])
def test_the_backbone_and_the_neck_start_from_a_checkpoint(
    tmp_path: Path, prefix: str, wrapped: bool
) -> None:
    """
    Input: the state dict of a camera branch, bare or as a detector names it, with the keys of
    a head next to it.
    Expected: the branch starts from the backbone and the neck of the checkpoint, the head
    keys are left out.
    """
    torch.manual_seed(0)
    source = _camera_network()
    state_dict = {f"{prefix}{key}": value for key, value in source.state_dict().items()}
    state_dict["bbox_head.anchor"] = torch.zeros(2)
    path = tmp_path / "weights.pth"
    torch.save({"state_dict": state_dict} if wrapped else state_dict, path)

    torch.manual_seed(1)
    assert not _state_dicts_equal(_camera_network(), source)
    assert _state_dicts_equal(_camera_network(path), source)


def test_a_checkpoint_missing_a_part_of_the_branch_is_rejected(tmp_path: Path) -> None:
    state_dict = {
        key: value
        for key, value in _camera_network().state_dict().items()
        if key.startswith("img_backbone.")
    }
    backbone_only, head_only = tmp_path / "backbone.pth", tmp_path / "head.pth"
    torch.save(state_dict, backbone_only)
    torch.save({"bbox_head.anchor": torch.zeros(2)}, head_only)

    with pytest.raises(RuntimeError, match="img_neck"):
        _camera_network(backbone_only)
    with pytest.raises(ValueError, match="no img_backbone or img_neck"):
        _camera_network(head_only)
    with pytest.raises(FileNotFoundError):
        _camera_network(tmp_path / "missing.pth")
