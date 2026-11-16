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

"""Cameras and image pyramids over the 8 m test scene of the Sparse4D fixtures.

The front camera sits 4 m behind the scene at its middle and looks along the x axis of the lidar
frame, so it sees the whole ``[0, 8] x [0, 8]`` m scene. The rear camera sits at the same place
and looks the other way, so it sees none of it.
"""

from __future__ import annotations

import torch

from autoware_ml.ops.deformable_aggregation.image_deformable_aggregation import ImageFeatures

IMAGE_SIZE = (16, 32)
IMAGE_STRIDES = (4, 8)
IMAGE_CHANNELS = 32
CAMERA_POSITION = (-4.0, 4.0, 0.0)
FOCAL_LENGTH = 16.0


def lidar2image(forward: float = 1.0) -> torch.Tensor:
    """Lidar frame to pixel matrix of a camera at ``CAMERA_POSITION`` looking along ``x``.

    Args:
        forward: 1 for the front camera looking along ``+x``, -1 for the rear one.

    Returns:
        The matrix ``(4, 4)``: the optical axis is ``forward * x``, the image ``x`` axis points
        to the right of it and the image ``y`` axis down.
    """
    height, width = IMAGE_SIZE
    rotation = torch.tensor(
        [[0.0, -forward, 0.0], [0.0, 0.0, -1.0], [forward, 0.0, 0.0]], dtype=torch.float32
    )
    lidar2cam = torch.eye(4)
    lidar2cam[:3, :3] = rotation
    lidar2cam[:3, 3] = -rotation @ torch.tensor(CAMERA_POSITION)
    intrinsics = torch.eye(4)
    intrinsics[0, 0] = intrinsics[1, 1] = FOCAL_LENGTH
    # The optical axis meets the image at its middle, half a pixel before the pixel center
    intrinsics[0, 2] = width / 2.0 - 0.5
    intrinsics[1, 2] = height / 2.0 - 0.5
    return intrinsics @ lidar2cam


def image_features(
    batch_size: int = 1, views: tuple[float, ...] = (1.0, -1.0), seed: int = 0
) -> ImageFeatures:
    """Random pyramid of the cameras over the test scene, with features that track gradients."""
    generator = torch.Generator().manual_seed(seed)
    height, width = IMAGE_SIZE
    levels = tuple(
        torch.randn(
            (batch_size, len(views), IMAGE_CHANNELS, height // stride, width // stride),
            generator=generator,
        ).requires_grad_(True)
        for stride in IMAGE_STRIDES
    )
    projection = torch.stack([lidar2image(forward) for forward in views])
    return ImageFeatures(
        levels=levels,
        strides=IMAGE_STRIDES,
        lidar2image=projection[None].repeat(batch_size, 1, 1, 1),
        image_size=IMAGE_SIZE,
    )
