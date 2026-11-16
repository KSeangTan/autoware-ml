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

"""Tests for the image blocks of the Sparse4D head, next to and without its lidar blocks."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from autoware_ml.models.detection3d.heads.sparse4d.image_aggregation import (
    ImageDeformableAggregation,
)
from autoware_ml.models.detection3d.heads.sparse4d.sparse4d_head import (
    ANCHOR_DIM,
    Sparse4DHead,
    SparseBox3DKeyPointsGenerator,
)
from autoware_ml.models.detection3d.tests.sparse4d_fixtures import (
    build_sparse4d_head,
    make_anchor_file,
)
from autoware_ml.models.detection3d.tests.sparse4d_image_fixtures import (
    IMAGE_CHANNELS,
    IMAGE_STRIDES,
    image_features,
)

EMBED_DIMS = 32
BEV_CHANNELS = 32
LIDAR_SETTINGS = {
    "bev": {"lidar_sampling": "bev", "bev_cell_size": 1.0, "bev_channels": BEV_CHANNELS},
    "bev_attention": {
        "lidar_sampling": "bev_attention",
        "bev_cell_size": 1.0,
        "bev_channels": BEV_CHANNELS,
    },
    "none": {"lidar_sampling": "none"},
}


def _head(tmp_path: Path, lidar: str, images: bool = True, **overrides: object) -> Sparse4DHead:
    torch.manual_seed(0)
    settings = dict(LIDAR_SETTINGS[lidar])
    if images:
        settings.update(image_strides=IMAGE_STRIDES, image_channels=IMAGE_CHANNELS)
    settings.update(overrides)
    return build_sparse4d_head(make_anchor_file(tmp_path), **settings)


def _bev_map() -> torch.Tensor:
    generator = torch.Generator().manual_seed(1)
    return torch.randn((1, BEV_CHANNELS, 8, 8), generator=generator).requires_grad_(True)


def _block(camera_embed: bool = True) -> ImageDeformableAggregation:
    torch.manual_seed(0)
    return ImageDeformableAggregation(
        embed_dims=EMBED_DIMS,
        image_channels=IMAGE_CHANNELS,
        num_groups=4,
        num_levels=len(IMAGE_STRIDES),
        kps_generator=SparseBox3DKeyPointsGenerator(
            embed_dims=EMBED_DIMS, fix_scale=((0.0, 0.0, 0.0),), num_learnable_pts=2
        ),
        camera_embed=camera_embed,
    )


def _anchor(x: float, y: float) -> torch.Tensor:
    """One unit box at ``(x, y)`` on the ground, heading along ``x``."""
    anchor = torch.zeros(1, 1, ANCHOR_DIM)
    anchor[..., 0], anchor[..., 1], anchor[..., 7] = x, y, 1.0
    return anchor


@pytest.mark.parametrize("lidar", ["bev", "bev_attention", "none"])
def test_head_trains_on_the_images_next_to_every_lidar_block(tmp_path: Path, lidar: str) -> None:
    head = _head(tmp_path, lidar)
    bev_map = None if lidar == "none" else _bev_map()
    images = image_features()

    outputs = head(bev_map, image_features=images)
    loss = outputs.classification.sum() + outputs.prediction.sum() + outputs.quality.sum()
    loss.backward()

    assert outputs.prediction.shape == (1, 2, 8, ANCHOR_DIM)
    assert torch.isfinite(outputs.prediction).all()
    assert all(level.grad is not None and level.grad.abs().sum() > 0 for level in images.levels)
    assert bev_map is None or bev_map.grad.abs().sum() > 0
    missing = [
        name
        for name, parameter in head.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    assert missing == []


@pytest.mark.parametrize("lidar", ["bev", "bev_attention"])
def test_image_blocks_leave_the_lidar_blocks_as_they_are(tmp_path: Path, lidar: str) -> None:
    lidar_only = _head(tmp_path, lidar, images=False).state_dict()
    fusion = _head(tmp_path, lidar).state_dict()

    added = set(fusion) - set(lidar_only)

    assert set(lidar_only) <= set(fusion)
    assert all(fusion[name].shape == value.shape for name, value in lidar_only.items())
    assert added and all(name.startswith(("image_aggregation.", "image_norm.")) for name in added)


def test_camera_only_head_builds_no_lidar_block(tmp_path: Path) -> None:
    head = _head(tmp_path, "none")

    assert len(head.deformable) == 0 and len(head.deformable_norm) == 0
    assert len(head.image_aggregation) == head.num_decoder
    assert not any(name.startswith("deformable") for name in head.state_dict())


def test_lidar_and_image_blocks_place_keypoints_of_their_own(tmp_path: Path) -> None:
    head = _head(tmp_path, "bev")

    lidar_points = {id(block.kps_generator) for block in head.deformable}
    image_points = {id(block.kps_generator) for block in head.image_aggregation}

    assert len(lidar_points) == len(image_points) == head.num_decoder
    assert lidar_points.isdisjoint(image_points)


def test_head_reads_the_modalities_it_has_blocks_for(tmp_path: Path) -> None:
    fusion = _head(tmp_path, "bev")
    lidar_only = _head(tmp_path, "bev", images=False)
    camera_only = _head(tmp_path, "none")

    with pytest.raises(ValueError, match="got no image features"):
        fusion(_bev_map())
    with pytest.raises(ValueError, match="got no lidar features"):
        fusion(None, image_features=image_features())
    with pytest.raises(ValueError, match="no image blocks"):
        lidar_only(_bev_map(), image_features=image_features())
    with pytest.raises(ValueError, match="no lidar blocks"):
        camera_only(_bev_map(), image_features=image_features())


def test_head_checks_the_strides_of_the_pyramid(tmp_path: Path) -> None:
    head = _head(tmp_path, "none")
    images = image_features()
    one_stride = replace(images, levels=images.levels[:1], strides=images.strides[:1])

    with pytest.raises(ValueError, match="read the strides"):
        head(None, image_features=one_stride)


@pytest.mark.parametrize(
    ("settings", "match"),
    [
        ({"lidar_sampling": "none"}, "set image_strides"),
        ({"lidar_sampling": "sparse", "image_channels": 16}, "without the strides"),
        ({"lidar_sampling": "sparse", "image_camera_embed": False}, "without the strides"),
        ({"lidar_sampling": "none", "image_strides": (4,), "bev_cell_size": 1.0}, "none without"),
    ],
)
def test_head_rejects_image_settings_that_do_not_fit(
    tmp_path: Path, settings: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        build_sparse4d_head(make_anchor_file(tmp_path), **settings)


def test_block_reads_the_cameras_that_see_the_anchor_alone() -> None:
    block = _block().eval()
    images = image_features()
    instance_feature = torch.randn(1, 1, EMBED_DIMS)
    anchor_embed = torch.zeros(1, 1, EMBED_DIMS)

    updated = block(instance_feature, _anchor(4.0, 4.0), anchor_embed, images)
    updated.sum().backward()

    # The front camera looks at the scene, the rear camera looks away from it
    assert not torch.allclose(updated, instance_feature)
    assert all(level.grad[:, 0].abs().sum() > 0 for level in images.levels)
    assert all(level.grad[:, 1].abs().sum() == 0 for level in images.levels)


def test_block_gives_an_anchor_no_camera_sees_no_image_features() -> None:
    block = _block().eval()
    rear_only = image_features(views=(-1.0,))
    instance_feature = torch.randn(1, 1, EMBED_DIMS)

    updated = block(instance_feature, _anchor(4.0, 4.0), torch.zeros(1, 1, EMBED_DIMS), rear_only)

    assert torch.isfinite(updated).all()
    assert torch.allclose(updated, instance_feature + block.output_proj.bias)


def test_block_weights_sum_to_one_over_the_samples_the_cameras_see() -> None:
    block = _block().eval()
    images = image_features()
    query = torch.randn(1, 3, EMBED_DIMS)
    # Seen by the front camera alone: the first anchor at every keypoint, the second at one
    visible = torch.zeros(1, 2, 3, block.num_pts, dtype=torch.bool)
    visible[0, 0, 0] = True
    visible[0, 0, 1, 0] = True

    weights = block._weights(query, images, visible)

    assert weights.shape == (1, 3, 2, len(IMAGE_STRIDES), block.num_pts, 4)
    totals = weights.sum(dim=(2, 3, 4))
    assert torch.allclose(totals[0, :2], torch.ones(2, 4))
    assert torch.all(totals[0, 2] == 0.0)
    assert torch.all(weights[:, :, 1] == 0.0)
    assert torch.all(weights[0, 1, 0, :, 1:] == 0.0)


def test_block_tells_the_cameras_apart_by_their_projection() -> None:
    images = image_features(views=(1.0, 1.0))
    moved = images.lidar2image.clone()
    moved[:, 1, 0, 3] += 4.0
    images = replace(images, lidar2image=moved)
    query = torch.randn(1, 1, EMBED_DIMS)
    visible = torch.ones(1, 2, 1, 3, dtype=torch.bool)

    def camera_weights(camera_embed: bool) -> torch.Tensor:
        block = _block(camera_embed).eval()
        torch.nn.init.normal_(block.weights_fc.weight)
        return block._weights(query, images, visible)

    with_embedding = camera_weights(True)
    without_embedding = camera_weights(False)

    assert not torch.allclose(with_embedding[:, :, 0], with_embedding[:, :, 1])
    assert torch.allclose(without_embedding[:, :, 0], without_embedding[:, :, 1])


def test_block_checks_the_number_of_strides() -> None:
    block = _block()
    images = image_features()
    one_stride = replace(images, levels=images.levels[:1], strides=images.strides[:1])

    with pytest.raises(ValueError, match="samples 2 strides"):
        block(
            torch.zeros(1, 1, EMBED_DIMS),
            _anchor(4.0, 4.0),
            torch.zeros(1, 1, EMBED_DIMS),
            one_stride,
        )


def test_temporal_head_carries_its_instances_over_the_images(tmp_path: Path) -> None:
    head = _head(tmp_path, "bev", num_temp_instances=4)
    pose = torch.eye(4)[None]

    for frame_index in range(2):
        outputs = head(
            _bev_map(),
            timestamps_seconds=torch.tensor([100.0 + 0.1 * frame_index], dtype=torch.float64),
            lidar2global=pose,
            scene_tokens=["scene"],
            image_features=image_features(seed=frame_index),
        )

    assert torch.isfinite(outputs.prediction).all()
    assert head.instance_bank._slot_valid.tolist() == [True]
