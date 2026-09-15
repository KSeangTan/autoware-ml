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

"""Shared geometric operations for 3D scene augmentations.

This module holds the *math* behind the rotation / scale / translation / flip
augmentations as plain functions. It owns no transform classes and declares no
required keys, so the transforms in ``transforms.point_cloud.geometry`` and
``transforms.camera_lidar.geometry`` can reuse exactly the same computations.

Two groups of helpers:

* **pure array math** - matrices and per-array transforms that take and return
  numpy arrays (``rotation_matrix``, ``rot_scale_trans_matrix``,
  ``flip_matrix``);
* **sampling + dict application** - draw augmentation parameters and apply them
  in place to whichever of ``coord`` / ``points`` / ``normal`` / ``gt_boxes`` /
  camera matrices a transform decides to touch. The transform classes choose
  *which* helpers to call (and require the matching keys); these helpers never
  silently skip work based on key presence beyond the documented point/normal
  optionality.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

POINT_KEYS = ("coord", "points")


def rotation_matrix(axis: str, angle: float) -> npt.NDArray[np.float32]:
    """Build a 3x3 rotation matrix for a single axis."""
    cos, sin = np.cos(angle), np.sin(angle)
    if axis == "x":
        return np.array([[1, 0, 0], [0, cos, -sin], [0, sin, cos]], dtype=np.float32)
    if axis == "y":
        return np.array([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]], dtype=np.float32)
    if axis == "z":
        return np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]], dtype=np.float32)
    raise NotImplementedError(f"Unsupported rotation axis: {axis}")


def resolve_rotation_center(
    coord: npt.NDArray[np.float32], configured_center: npt.NDArray[np.float32] | None
) -> npt.NDArray[np.float32]:
    """Resolve the rotation center for a point cloud (config value or bbox center)."""
    if configured_center is not None:
        return configured_center
    return (coord.min(axis=0) + coord.max(axis=0)) / 2.0


def has_point_cloud(input_dict: dict[str, Any]) -> bool:
    """Return whether any point representation (``coord`` / ``points``) is present."""
    return any(input_dict.get(key) is not None for key in POINT_KEYS)


def require_point_cloud(input_dict: dict[str, Any]) -> None:
    """Raise if no point representation is present (no silent skip)."""
    if not has_point_cloud(input_dict):
        raise KeyError(
            f"a point representation ({' or '.join(POINT_KEYS)}) is required but none was found"
        )


def apply_to_point_xyz(
    input_dict: dict[str, Any], fn: Callable[[npt.NDArray[np.float32]], npt.NDArray[np.float32]]
) -> None:
    """Apply ``fn`` to the XYZ columns of every present point representation."""
    for key in POINT_KEYS:
        array = input_dict.get(key)
        if array is None:
            continue
        array = np.asarray(array).copy()
        array[:, :3] = fn(array[:, :3]).astype(array.dtype)
        input_dict[key] = array


def rotate_points_about_center(
    input_dict: dict[str, Any], rotation: npt.NDArray[np.float32], center: npt.NDArray[np.float32]
) -> None:
    """Rotate every present point representation about ``center``."""
    apply_to_point_xyz(input_dict, lambda xyz: (xyz - center) @ rotation.T + center)


def transform_normal(input_dict: dict[str, Any], rotation: npt.NDArray[np.float32]) -> None:
    """Rotate per-point ``normal`` vectors when present."""
    if "normal" in input_dict:
        input_dict["normal"] = np.asarray(input_dict["normal"]) @ rotation.T


def rotate_boxes_about_center(
    input_dict: dict[str, Any],
    rotation: npt.NDArray[np.float32],
    rotation_angle: float,
    center: npt.NDArray[np.float32],
) -> None:
    """Rotate ``gt_boxes`` about ``center`` consistently with the point rotation."""
    if "gt_boxes" not in input_dict:
        return
    boxes = np.asarray(input_dict["gt_boxes"]).copy()
    boxes[:, :3] = (boxes[:, :3] - center) @ rotation.T + center
    if boxes.shape[1] > 6:
        boxes[:, 6] += rotation_angle
    if boxes.shape[1] >= 9:
        boxes[:, 7:9] = boxes[:, 7:9] @ rotation[:2, :2].T
    input_dict["gt_boxes"] = boxes
