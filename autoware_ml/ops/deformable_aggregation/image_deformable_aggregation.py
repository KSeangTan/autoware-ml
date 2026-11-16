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

"""Multi-view image pyramids as levels of the deformable aggregation.

The image blocks aggregate with :func:`bev_deformable_aggregation`, the function of the lidar
BEV blocks. This module gives it the image side: the map of one camera at one stride is a level
that samples metric keypoints at their projection into the image.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

# A keypoint closer to the image plane than this, or behind it, is not seen by the camera
MIN_DEPTH = 0.1
# Where the keypoints a camera does not see are sent to, outside the ``[-1, 1]`` of the image
OUTSIDE = -2.0


@dataclass(frozen=True)
class ImageLevel:
    """Features of one camera at one stride with the projection that places the keypoints.

    The image counterpart of :class:`BEVLevel`, a level of
    :func:`bev_deformable_aggregation`.

    Attributes:
        feat: Image features of the camera ``(B, C, H, W)``.
        lidar2image: Lidar frame to pixel matrices of the camera ``(B, 4, 4)``.
        image_size: ``(height, width)`` of the images the pixels are of.
    """

    feat: torch.Tensor
    lidar2image: torch.Tensor
    image_size: tuple[int, int]

    def sample(self, key_points: torch.Tensor) -> torch.Tensor:
        """Sample the map bilinearly at the projected keypoints, unseen ones read zeros.

        Args:
            key_points: Metric keypoints in the lidar frame ``(B, N, P, 3)``.

        Returns:
            Sampled features ``(B, N, P, C)``.
        """
        grid, _ = project_key_points(key_points, self.lidar2image[:, None], self.image_size)
        sampled = F.grid_sample(
            self.feat,
            grid[:, 0].to(self.feat.dtype),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        return sampled.permute(0, 2, 3, 1)


@dataclass(frozen=True)
class ImageFeatures:
    """Feature pyramid of every camera of a frame with the projection that places it.

    The image features of a head, kept apart from its lidar features: the pyramid of the camera
    branch, from the finest stride to the coarsest, and the matrices that take a point of the
    lidar frame to a pixel of the images the pyramid was computed on.

    Attributes:
        levels: One map ``(B, V, C, H / stride, W / stride)`` per stride, finest first.
        strides: Stride of every level against the images.
        lidar2image: Lidar frame to pixel matrices ``(B, V, 4, 4)`` of those images, with the
            lidar and the image augmentations folded in.
        image_size: ``(height, width)`` of the images.
    """

    levels: tuple[torch.Tensor, ...]
    strides: tuple[int, ...]
    lidar2image: torch.Tensor
    image_size: tuple[int, int]

    def __post_init__(self) -> None:
        """Check the pyramid against its strides.

        Raises:
            ValueError: If the levels do not match the strides, the cameras or the image size.
        """
        if len(self.levels) == 0 or len(self.levels) != len(self.strides):
            raise ValueError(
                f"The image pyramid needs one level per stride, got {len(self.levels)} levels "
                f"and strides {self.strides}."
            )
        if list(self.strides) != sorted(set(self.strides)):
            raise ValueError(f"The image strides must rise from the finest, got {self.strides}.")
        height, width = self.image_size
        views = self.lidar2image.shape[:2]
        for level, stride in zip(self.levels, self.strides, strict=True):
            expected = (math.ceil(height / stride), math.ceil(width / stride))
            if level.dim() != 5 or tuple(level.shape[:2]) != tuple(views):
                raise ValueError(
                    f"The level of stride {stride} must be (B, V, C, H, W) over the "
                    f"{tuple(views)} cameras of lidar2image, got {tuple(level.shape)}."
                )
            if tuple(level.shape[-2:]) != expected:
                raise ValueError(
                    f"The level of stride {stride} over {height}x{width} images must be "
                    f"{expected[0]}x{expected[1]}, got {tuple(level.shape[-2:])}."
                )

    @property
    def batch_size(self) -> int:
        """Number of frames."""
        return self.levels[0].shape[0]

    def camera_levels(self) -> list[ImageLevel]:
        """Level of every camera and stride, camera major and the strides from the finest."""
        return [
            ImageLevel(level[:, view], self.lidar2image[:, view], self.image_size)
            for view in range(self.lidar2image.shape[1])
            for level in self.levels
        ]

    def normalized_projection(self) -> torch.Tensor:
        """Projection of every camera onto the unit image ``(B, V, 3, 4)``, in float32.

        The pixel rows of ``lidar2image`` divided by the image size, so the matrix does not
        change with the resolution the images were resized to.
        """
        height, width = self.image_size
        projection = self.lidar2image[..., :3, :].float()
        scale = projection.new_tensor([1.0 / width, 1.0 / height, 1.0])
        return projection * scale[:, None]


def project_key_points(
    key_points: torch.Tensor, lidar2image: torch.Tensor, image_size: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project metric keypoints into every camera as a ``grid_sample`` grid.

    The projection computes in float32 whatever the autocast of the caller, half precision
    holds a pixel of a megapixel image only to about a pixel. A pixel center sits on the
    integer pixel coordinate, so the grid takes it half a pixel in.

    Args:
        key_points: Keypoints in the lidar frame ``(B, N, P, 3)``.
        lidar2image: Lidar frame to pixel matrices ``(B, V, 4, 4)``.
        image_size: ``(height, width)`` of the images.

    Returns:
        The grid ``(B, V, N, P, 2)`` as ``(x, y)`` in ``[-1, 1]`` over the image, and whether
        the camera sees every keypoint ``(B, V, N, P)``: in front of it and inside the image.
        The keypoints a camera does not see sit outside the image, where a sample reads zeros.
    """
    height, width = image_size
    with torch.autocast(device_type=key_points.device.type, enabled=False):
        points = key_points.float()
        homogeneous = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)
        projected = torch.einsum("bvij,bnpj->bvnpi", lidar2image.float()[..., :3, :], homogeneous)
        depth = projected[..., 2]
        in_front = depth > MIN_DEPTH
        pixels = projected[..., :2] / depth.clamp(min=MIN_DEPTH)[..., None]
        size = pixels.new_tensor([width, height])
        grid = (pixels + 0.5) / size * 2.0 - 1.0
        visible = in_front & (grid.abs() < 1.0).all(dim=-1)
        grid = torch.where(visible[..., None], grid, torch.full_like(grid, OUTSIDE))
    return grid, visible
