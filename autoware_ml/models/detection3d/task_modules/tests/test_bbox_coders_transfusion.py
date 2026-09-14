"""Unit tests for the TransFusion box coder's height convention.

Ground-truth boxes in this framework are gravity center only (see ``Box3DCenterCoordinateType``),
so ``encode`` and ``decode_boxes`` must agree on what the height channel means: the value that
``encode`` writes for a gravity-center box has to come back out of ``decode_boxes`` as the same
gravity-center z, with no half-height offset in either direction.
"""

from __future__ import annotations

import torch

from autoware_ml.models.detection3d.task_modules.bbox_coders import TransFusionBBoxCoder
from autoware_ml.types.geometry import Box3DFieldIndex

_PC_RANGE = [-10.0, -20.0, -2.0, 10.0, 20.0, 4.0]
_VOXEL_SIZE = [0.5, 0.25, 0.2]
_OUT_SIZE_FACTOR = 2


def _coder(code_size: int = 10) -> TransFusionBBoxCoder:
    return TransFusionBBoxCoder(
        pc_range=_PC_RANGE,
        out_size_factor=_OUT_SIZE_FACTOR,
        voxel_size=_VOXEL_SIZE,
        score_threshold_groups=None,
        post_center_range=[-1.0, -1.0, -5.0, 10.0, 10.0, 5.0],
        code_size=code_size,
    )


def _gravity_center_boxes() -> torch.Tensor:
    """Two samples of gravity-center boxes: (cx, cy, cz, length, width, height, yaw, vx, vy).

    Heights differ per box so a half-height offset cannot cancel out across the batch.
    """
    return torch.tensor(
        [
            [
                [2.0, 4.0, 1.0, 4.0, 2.0, 1.5, 0.25, 0.1, -0.2],
                [-3.0, 0.5, -0.75, 0.8, 0.8, 1.7, -2.9, 0.0, 0.0],
            ],
            [
                [6.0, -8.0, 0.2, 10.0, 2.5, 3.6, 1.2, 3.0, 0.5],
                [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            ],
        ],
        dtype=torch.float32,
    )


def _split_encoded(
    encoded: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Slice ``(batch, boxes, code)`` targets into the ``(batch, channels, boxes)`` head layout."""
    channels_first = encoded.permute(0, 2, 1)
    return {
        "centers": channels_first[:, 0:2, :],
        "heights": channels_first[:, 2:3, :],
        "dims": channels_first[:, 3:6, :],
        "rots": channels_first[:, 6:8, :],
        "vels": channels_first[:, 8:10, :] if encoded.shape[-1] == 10 else None,
    }


def test_encode_keeps_gravity_center_height() -> None:
    """The height target of a gravity-center box is its z, not z shifted by half its height."""
    boxes = _gravity_center_boxes()

    encoded = _coder().encode(boxes)

    torch.testing.assert_close(encoded[..., 2], boxes[..., Box3DFieldIndex.Z])


def test_decode_boxes_emits_height_channel_as_gravity_center_z() -> None:
    """``decode_boxes`` passes the regressed height straight through as the box z."""
    boxes = _gravity_center_boxes()
    parts = _split_encoded(_coder().encode(boxes))

    decoded = _coder().decode_boxes(**parts)

    torch.testing.assert_close(decoded[..., Box3DFieldIndex.Z], parts["heights"][:, 0, :])


def test_encode_decode_boxes_round_trip_preserves_height() -> None:
    """A gravity-center box survives ``encode`` followed by ``decode_boxes`` unchanged in z."""
    boxes = _gravity_center_boxes()
    coder = _coder()

    decoded = coder.decode_boxes(**_split_encoded(coder.encode(boxes)))

    torch.testing.assert_close(
        decoded[..., Box3DFieldIndex.Z], boxes[..., Box3DFieldIndex.Z], rtol=1e-5, atol=1e-5
    )


def test_encode_decode_boxes_round_trip_preserves_full_geometry() -> None:
    """Every channel, not only z, comes back after ``encode`` then ``decode_boxes``."""
    boxes = _gravity_center_boxes()
    coder = _coder()

    decoded = coder.decode_boxes(**_split_encoded(coder.encode(boxes)))

    assert decoded.shape == boxes.shape
    # Yaw is stored as (sin, cos) so it round-trips through atan2 with float noise only.
    torch.testing.assert_close(decoded, boxes, rtol=1e-5, atol=1e-5)


def test_round_trip_height_is_independent_of_velocity_channels() -> None:
    """The height convention does not change when the coder has no velocity channels."""
    boxes = _gravity_center_boxes()
    coder = _coder(code_size=8)
    parts = _split_encoded(coder.encode(boxes))
    assert parts["vels"] is None

    decoded = coder.decode_boxes(**parts)

    assert decoded.shape == (*boxes.shape[:2], 7)
    torch.testing.assert_close(
        decoded[..., Box3DFieldIndex.Z], boxes[..., Box3DFieldIndex.Z], rtol=1e-5, atol=1e-5
    )
