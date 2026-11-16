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

"""Tests for the image deformable aggregation operator."""

from __future__ import annotations

import pytest
import torch

from autoware_ml.ops.deformable_aggregation.bev_deformable_aggregation import (
    bev_deformable_aggregation,
)
from autoware_ml.ops.deformable_aggregation.image_deformable_aggregation import (
    OUTSIDE,
    ImageFeatures,
    ImageLevel,
    project_key_points,
)

IMAGE_SIZE = (2, 4)
# Turns the lidar frame half around its z axis
_FLIP = torch.tensor([-1.0, -1.0, 1.0, 1.0])


def _camera_along_x() -> torch.Tensor:
    """Camera at the origin looking along ``x``, one metre wide per pixel at one metre depth."""
    lidar2cam = torch.tensor(
        [[0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
    intrinsics = torch.eye(4)
    intrinsics[0, 2] = IMAGE_SIZE[1] / 2.0 - 0.5
    intrinsics[1, 2] = IMAGE_SIZE[0] / 2.0 - 0.5
    return (intrinsics @ lidar2cam)[None, None]


def _row_major(height: int, width: int) -> torch.Tensor:
    """One camera map of one channel whose cells count up row by row from 1."""
    values = torch.arange(1.0, height * width + 1.0)
    return values.reshape(1, 1, 1, height, width)


def test_projection_places_the_optical_axis_at_the_middle_and_hides_what_is_behind() -> None:
    # On the axis, one pixel to the right and half a pixel down of it, behind the camera and
    # in front of it past the left border of the image
    key_points = torch.tensor(
        [[[[2.0, 0.0, 0.0], [2.0, -2.0, -1.0], [-2.0, 0.0, 0.0], [1.0, 5.0, 0.0]]]]
    )

    grid, visible = project_key_points(key_points, _camera_along_x(), IMAGE_SIZE)

    assert grid.shape == (1, 1, 1, 4, 2)
    assert visible[0, 0, 0].tolist() == [True, True, False, False]
    assert torch.allclose(grid[0, 0, 0, 0], torch.tensor([0.0, 0.0]))
    assert torch.allclose(grid[0, 0, 0, 1], torch.tensor([0.5, 0.5]))
    assert torch.all(grid[0, 0, 0, 2:] == OUTSIDE)


def test_image_level_reads_a_pixel_at_its_center_and_zeros_where_the_camera_does_not_see() -> None:
    level = ImageLevel(_row_major(*IMAGE_SIZE)[:, 0], _camera_along_x()[:, 0], IMAGE_SIZE)
    # The pixel centers of the top row sit 1.5 and 0.5 pixels left of the axis and half up,
    # the last keypoint is behind the camera
    key_points = torch.tensor(
        [[[[1.0, 1.5, 0.5], [1.0, 0.5, 0.5], [1.0, -1.5, -0.5], [-1.0, 0.0, 0.0]]]]
    )

    sampled = level.sample(key_points)

    assert sampled.shape == (1, 1, 4, 1)
    assert torch.allclose(sampled.flatten(), torch.tensor([1.0, 2.0, 8.0, 0.0]))


def test_camera_levels_feed_the_aggregation_of_the_bev_maps_camera_major() -> None:
    generator = torch.Generator().manual_seed(0)
    fine = torch.randn((1, 2, 4, 2, 4), generator=generator)
    coarse = torch.randn((1, 2, 4, 1, 2), generator=generator)
    # The second camera looks the other way, so the keypoints in front of the first are behind it
    lidar2image = torch.cat([_camera_along_x(), _camera_along_x() @ torch.diag(_FLIP)], dim=1)
    features = ImageFeatures(
        levels=(fine, coarse), strides=(1, 2), lidar2image=lidar2image, image_size=IMAGE_SIZE
    )
    key_points = torch.rand((1, 3, 5, 3), generator=generator) - 0.5
    key_points[..., 0] = torch.tensor([2.0, 3.0, -2.0])[:, None]
    weights = torch.rand((1, 3, 2, 2, 5, 2), generator=generator)

    levels = features.camera_levels()
    aggregated = bev_deformable_aggregation(key_points, weights.flatten(2, 4), levels)

    assert [tuple(level.feat.shape[-2:]) for level in levels] == [(2, 4), (1, 2), (2, 4), (1, 2)]
    expected = torch.zeros(1, 3, 4)
    for view in range(2):
        for level_index, level in enumerate((fine, coarse)):
            sampled = ImageLevel(level[:, view], lidar2image[:, view], IMAGE_SIZE).sample(
                key_points
            )
            for group in range(2):
                channels = slice(2 * group, 2 * group + 2)
                weight = weights[:, :, view, level_index, :, group]
                expected[..., channels] += (sampled[..., channels] * weight[..., None]).sum(dim=2)
    assert torch.allclose(aggregated, expected, atol=1e-6)
    # Each camera sees the anchors in front of it alone
    seen = [level.sample(key_points).abs().sum(dim=(2, 3))[0] > 0 for level in levels[::2]]
    assert seen[0].tolist() == [True, True, False]
    assert seen[1].tolist() == [False, False, True]


def test_normalized_projection_does_not_change_with_the_image_resolution() -> None:
    small = ImageFeatures(
        levels=(torch.zeros(1, 1, 1, 2, 4),),
        strides=(1,),
        lidar2image=_camera_along_x(),
        image_size=IMAGE_SIZE,
    )
    doubling = torch.diag(torch.tensor([2.0, 2.0, 1.0, 1.0]))
    large = ImageFeatures(
        levels=(torch.zeros(1, 1, 1, 4, 8),),
        strides=(1,),
        lidar2image=doubling @ _camera_along_x(),
        image_size=(4, 8),
    )

    assert small.normalized_projection().shape == (1, 1, 3, 4)
    assert torch.allclose(small.normalized_projection(), large.normalized_projection())


@pytest.mark.parametrize(
    ("levels", "strides", "match"),
    [
        ((), (), "one level per stride"),
        ((torch.zeros(1, 1, 1, 2, 4),), (1, 2), "one level per stride"),
        ((torch.zeros(1, 1, 1, 1, 2), torch.zeros(1, 1, 1, 2, 4)), (2, 1), "rise from the finest"),
        ((torch.zeros(1, 1, 1, 2, 4),), (2,), "must be 1x2"),
        ((torch.zeros(1, 3, 1, 2, 4),), (1,), "cameras of lidar2image"),
    ],
)
def test_image_features_check_the_pyramid_against_its_strides(
    levels: tuple[torch.Tensor, ...], strides: tuple[int, ...], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        ImageFeatures(
            levels=levels, strides=strides, lidar2image=_camera_along_x(), image_size=IMAGE_SIZE
        )
