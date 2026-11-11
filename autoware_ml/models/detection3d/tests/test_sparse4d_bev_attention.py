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

"""Tests for the CMT positional embedding and the BEV cross attention of the Sparse4D head."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from autoware_ml.models.detection3d.heads.sparse4d.cmt_position_embedding import (
    CMTPositionEmbedding,
    sincos_position_embedding,
)
from autoware_ml.models.detection3d.heads.sparse4d.sparse4d_head import (
    BEVCrossAttention,
    Sparse4DHead,
)
from autoware_ml.models.detection3d.tests.ptv3_detection3d_fixtures import POINT_CLOUD_RANGE
from autoware_ml.models.detection3d.tests.sparse4d_fixtures import (
    build_sparse4d_head,
    make_anchor_file,
)
from autoware_ml.ops.deformable_aggregation.bev_deformable_aggregation import BEVLevel

EMBED_DIMS = 8
BEV_CHANNELS = 4


def _embedding() -> CMTPositionEmbedding:
    return CMTPositionEmbedding(EMBED_DIMS, POINT_CLOUD_RANGE)


def _block(embedding: CMTPositionEmbedding) -> BEVCrossAttention:
    return BEVCrossAttention(
        embed_dims=EMBED_DIMS,
        bev_channels=BEV_CHANNELS,
        num_heads=2,
        position_embedding=embedding,
    )


def _level(cells: int, cell_size: float, seed: int) -> BEVLevel:
    generator = torch.Generator().manual_seed(seed)
    feat = torch.randn((1, BEV_CHANNELS, cells, cells), generator=generator).requires_grad_(True)
    return BEVLevel(feat, (POINT_CLOUD_RANGE[0], POINT_CLOUD_RANGE[1]), cell_size)


def _attention_head(tmp_path: Path) -> Sparse4DHead:
    return build_sparse4d_head(
        make_anchor_file(tmp_path),
        lidar_sampling="bev_attention",
        bev_cell_size=1.0,
        bev_channels=BEV_CHANNELS,
    )


def test_sincos_embedding_alternates_the_sine_and_the_cosine_with_y_first() -> None:
    at_origin = sincos_position_embedding(torch.zeros(2), 4)
    quarter_x = sincos_position_embedding(torch.tensor([0.25, 0.0]), 2)

    assert at_origin.shape == (8,)
    assert torch.equal(at_origin[0::2], torch.zeros(4))
    assert torch.equal(at_origin[1::2], torch.ones(4))
    assert torch.allclose(quarter_x, torch.tensor([0.0, 1.0, 1.0, 0.0]), atol=1e-6)


def test_the_embedding_needs_even_dims() -> None:
    with pytest.raises(ValueError, match="even"):
        CMTPositionEmbedding(7, POINT_CLOUD_RANGE)


def test_a_query_and_the_cell_under_it_share_one_embedding() -> None:
    embedding = _embedding()
    level = BEVLevel(torch.zeros(1, BEV_CHANNELS, 4, 4), (0.0, 0.0), 2.0)

    cells = embedding.cells(level)
    query = embedding(torch.tensor([[3.0, 5.0]]))

    assert cells.shape == (16, EMBED_DIMS)
    # The query sits in the cell of column 1 and row 2, the map flattens row major
    assert torch.allclose(query[0], cells[2 * 4 + 1])
    assert not torch.allclose(query[0], cells[0])


def test_the_scale_takes_the_constant_part_out_of_the_sincos_features() -> None:
    # The grid of the 122 m lidar range, where the embedding of CMT collapsed to one vector
    point_cloud_range = (-122.4, -122.4, -3.0, 122.4, 122.4, 5.0)
    level = BEVLevel(torch.zeros(1, BEV_CHANNELS, 180, 180), (-122.4, -122.4), 1.36)

    def sincos_features(temperature: float, scale: float) -> torch.Tensor:
        embedding = CMTPositionEmbedding(256, point_cloud_range, temperature, scale)
        embedding.mlp = torch.nn.Identity()
        return embedding.cells(level).reshape(180, 180, -1)

    def constant_and_varying(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cells = features.flatten(end_dim=1)
        return cells.mean(dim=0).norm(), cells.var(dim=0).sum().sqrt()

    def similarity_4_cells_apart(features: torch.Tensor) -> torch.Tensor:
        return torch.cosine_similarity(features[:, :-4], features[:, 4:], dim=-1).mean()

    cmt = sincos_features(10000.0, 1.0)
    scaled = sincos_features(64.0, 32.0)

    # Most features of CMT hold one value over the map, cells 5 m apart get one embedding
    constant, varying = constant_and_varying(cmt)
    assert constant > 2.0 * varying
    assert similarity_4_cells_apart(cmt) > 0.999
    constant, varying = constant_and_varying(scaled)
    assert varying > 4.0 * constant
    assert similarity_4_cells_apart(scaled) < 0.8


def test_the_scale_keeps_the_period_of_the_fastest_wave_in_meters() -> None:
    # The range is 8 m wide, with 4 turns the fastest wave repeats every 2 m
    embedding = CMTPositionEmbedding(EMBED_DIMS, POINT_CLOUD_RANGE, temperature=16.0, scale=4.0)
    embedding.mlp = torch.nn.Identity()

    features = embedding(torch.tensor([[1.0, 3.0], [3.0, 3.0], [1.5, 3.0]]))

    # The x features follow the y features, the first pair of each axis is the fastest wave
    fastest_x = features[:, EMBED_DIMS : EMBED_DIMS + 2]
    assert torch.allclose(fastest_x[0], fastest_x[1], atol=1e-5)
    assert not torch.allclose(fastest_x[0], fastest_x[2], atol=1e-2)
    # The slower waves still tell the two positions apart
    assert not torch.allclose(features[0], features[1], atol=1e-2)


def test_the_slowest_wave_may_not_repeat_within_the_range() -> None:
    with pytest.raises(ValueError, match="slowest wave"):
        CMTPositionEmbedding(256, POINT_CLOUD_RANGE, temperature=16.0, scale=32.0)


def test_the_head_passes_the_scale_and_the_temperature_to_the_embedding(tmp_path: Path) -> None:
    head = build_sparse4d_head(
        make_anchor_file(tmp_path),
        lidar_sampling="bev_attention",
        bev_cell_size=1.0,
        bev_channels=BEV_CHANNELS,
        position_temperature=64.0,
        position_scale=32.0,
    )

    embedding = head.deformable[0].position_embedding
    assert (embedding.temperature, embedding.scale) == (64.0, 32.0)


def test_the_embedding_norm_does_not_follow_the_mlp_weights() -> None:
    embedding = _embedding()
    position = torch.tensor([[3.0, 5.0], [-20.0, 11.0]])
    before = embedding(position).norm(dim=-1)

    # Training grew the spectral norm of the MLP layers, the logits with its output squared
    with torch.no_grad():
        embedding.mlp[0].weight.mul_(10.0)
        embedding.mlp[2].weight.mul_(10.0)

    assert torch.allclose(embedding(position).norm(dim=-1), before, rtol=1e-3)
    assert torch.allclose(before, torch.full_like(before, EMBED_DIMS**0.5), rtol=1e-3)


def test_the_block_attends_to_every_cell_of_every_level() -> None:
    embedding = _embedding()
    block = _block(embedding)
    levels = [_level(4, 2.0, 0), _level(2, 4.0, 1)]
    instance_feature = torch.randn(1, 3, EMBED_DIMS)
    anchor = torch.zeros(1, 3, 10)
    anchor[..., :2] = torch.tensor([[1.0, 1.0], [7.0, 7.0], [4.0, 4.0]])

    output = block(instance_feature, anchor, torch.zeros_like(instance_feature), [], levels)
    output.sum().backward()

    assert output.shape == (1, 3, EMBED_DIMS)
    assert all(level.feat.grad is not None and level.feat.grad.abs().sum() > 0 for level in levels)
    assert all(parameter.grad is not None for parameter in embedding.parameters())


def test_one_outlier_cell_does_not_take_the_attention() -> None:
    torch.manual_seed(0)
    block = _block(_embedding())
    level = _level(4, 2.0, 0)
    # The neck output of the diverged runs: one cell a thousand times the norm of the others
    with torch.no_grad():
        level.feat[0, :, 3, 3] *= 1000.0
    instance_feature = torch.zeros(1, 3, EMBED_DIMS)
    anchor = torch.zeros(1, 3, 10)
    anchor[..., :2] = torch.tensor([[1.0, 1.0], [7.0, 7.0], [4.0, 4.0]])

    tokens = block.bev_norm(level.feat.flatten(2).transpose(1, 2))
    output = block(instance_feature, anchor, instance_feature, [], [level])
    output.sum().backward()

    # Every token keeps the unit scale of the layer norm, the outlier included
    norms = tokens.norm(dim=-1)
    assert torch.allclose(norms, torch.full_like(norms, BEV_CHANNELS**0.5), rtol=1e-2)
    grad = level.feat.grad.norm(dim=0)[0]
    assert torch.isfinite(output).all()
    # The input gradient of the outlier cell shrinks by its scale instead of growing with it
    assert grad[3, 3] < grad.median()


def _local_block(local_radius: float) -> BEVCrossAttention:
    return BEVCrossAttention(
        embed_dims=EMBED_DIMS,
        bev_channels=BEV_CHANNELS,
        num_heads=2,
        position_embedding=_embedding(),
        local_radius=local_radius,
    )


def test_a_local_block_reads_only_the_cells_within_the_radius_of_the_anchor() -> None:
    torch.manual_seed(0)
    block = _local_block(2.0).eval()
    # 8 by 8 cells of 1 m, the query sits on the corner shared by the cells of rows and columns
    # 1 and 2, whose centers are 0.71 m away
    level = _level(8, 1.0, 0)
    instance_feature = torch.randn(1, 1, EMBED_DIMS)
    anchor = torch.zeros(1, 1, 10)
    anchor[..., :2] = torch.tensor([2.0, 2.0])

    output = block(instance_feature, anchor, instance_feature, [], [level])
    output.sum().backward()

    read = level.feat.grad.abs().sum(dim=(0, 1)) > 0
    ys, xs = torch.meshgrid(torch.arange(8) + 0.5, torch.arange(8) + 0.5, indexing="ij")
    within = (xs - 2.0) ** 2 + (ys - 2.0) ** 2 <= 2.0**2
    assert torch.equal(read, within)
    assert 0 < within.sum() < 64


def test_a_local_block_changes_nothing_with_the_whole_map_in_the_radius() -> None:
    torch.manual_seed(0)
    embedding = _embedding()
    block = _block(embedding).eval()
    local = _local_block(100.0).eval()
    # The weights of a block of every cell carry no radius, the local block keeps its own
    local.load_state_dict(block.state_dict(), strict=False)
    level = _level(8, 1.0, 0)
    instance_feature = torch.randn(1, 3, EMBED_DIMS)
    anchor = torch.zeros(1, 3, 10)
    anchor[..., :2] = torch.tensor([[1.0, 1.0], [7.0, 7.0], [4.0, 4.0]])

    with torch.no_grad():
        expected = block(instance_feature, anchor, instance_feature, [], [level])
        output = local(instance_feature, anchor, instance_feature, [], [level])

    assert torch.allclose(output, expected, atol=1e-6)


def test_a_query_out_of_the_map_keeps_its_nearest_cell() -> None:
    block = _local_block(2.0).eval()
    level = _level(8, 1.0, 0)
    instance_feature = torch.zeros(1, 1, EMBED_DIMS)
    anchor = torch.zeros(1, 1, 10)
    anchor[..., :2] = torch.tensor([-50.0, 3.2])

    output = block(instance_feature, anchor, instance_feature, [], [level])
    output.sum().backward()

    # No cell lies within 2 m, the query reads the cell of row 3 at the edge it left through
    read = level.feat.grad.abs().sum(dim=(0, 1)) > 0
    assert torch.isfinite(output).all()
    assert read.nonzero().tolist() == [[3, 0]]


def test_the_local_radius_must_be_positive() -> None:
    with pytest.raises(ValueError, match="local_radius"):
        _local_block(0.0)
    with pytest.raises(ValueError, match="local_radius"):
        _local_block(2.0).set_local_radius(-1.0)


def test_a_block_of_every_cell_takes_no_radius() -> None:
    block = _block(_embedding())

    assert "local_radius" not in block.state_dict()
    with pytest.raises(ValueError, match="every cell"):
        block.set_local_radius(2.0)


def test_a_local_block_follows_its_radius_and_keeps_it_in_the_checkpoint() -> None:
    torch.manual_seed(0)
    block = _local_block(2.0).eval()
    level = _level(8, 1.0, 0)
    instance_feature = torch.randn(1, 1, EMBED_DIMS)
    anchor = torch.zeros(1, 1, 10)
    anchor[..., :2] = torch.tensor([2.0, 2.0])

    def cells_read() -> int:
        level.feat.grad = None
        block(instance_feature, anchor, instance_feature, [], [level]).sum().backward()
        return int((level.feat.grad.abs().sum(dim=(0, 1)) > 0).sum())

    within_2m = cells_read()
    block.set_local_radius(4.0)
    within_4m = cells_read()
    block.set_local_radius(None)
    every_cell = cells_read()
    assert within_2m < within_4m < every_cell == 64

    # A block of another epoch reads the map as the checkpoint did
    block.set_local_radius(4.0)
    restored = _local_block(2.0)
    restored.load_state_dict(block.state_dict())
    assert float(restored.local_radius) == 4.0


def _local_head(tmp_path: Path, **overrides: object) -> Sparse4DHead:
    return build_sparse4d_head(
        make_anchor_file(tmp_path),
        lidar_sampling="bev_attention",
        bev_cell_size=1.0,
        bev_channels=BEV_CHANNELS,
        **overrides,
    )


def _radii(head: Sparse4DHead) -> set[float | None]:
    return {
        None if block.local_radius is None else float(block.local_radius)
        for block in head.deformable
    }


def test_the_head_attends_to_every_cell_unless_local_attention_is_set(tmp_path: Path) -> None:
    head = _local_head(tmp_path, local_attention_radii=[[0, 3.0]])
    assert _radii(head) == {None}
    head.set_epoch(5)
    assert _radii(head) == {None}

    assert _radii(_local_head(tmp_path, local_attention=True)) == {30.0}


def test_the_local_radius_of_the_head_follows_the_epoch(tmp_path: Path) -> None:
    head = _local_head(
        tmp_path,
        local_attention=True,
        local_attention_radii=[[0, 3.0], [2, 5.0], [5, None]],
    )

    radii = []
    for epoch in range(7):
        head.set_epoch(epoch)
        radii.append(_radii(head))

    infinity = float("inf")
    assert radii == [{3.0}, {3.0}, {5.0}, {5.0}, {5.0}, {infinity}, {infinity}]
    # The whole map again: an epoch of every cell builds no mask
    bev_map = torch.randn(1, BEV_CHANNELS, 8, 8)
    every_cell = _local_head(tmp_path)
    every_cell.load_state_dict(head.state_dict(), strict=False)
    with torch.no_grad():
        outputs = head.eval()(bev_map)
        expected = every_cell.eval()(bev_map)
    assert torch.allclose(outputs.prediction, expected.prediction, atol=1e-5)


@pytest.mark.parametrize(
    "radii", [[], [[1, 3.0]], [[0, 3.0], [0, 5.0]], [[0, 3.0], [4, 5.0], [2, 6.0]], [[0, 0.0]]]
)
def test_the_local_radii_rise_from_epoch_0_with_positive_radii(
    tmp_path: Path, radii: list[list[float]]
) -> None:
    with pytest.raises(ValueError, match="local_attention_radii"):
        _local_head(tmp_path, local_attention=True, local_attention_radii=radii)


def test_local_attention_needs_the_attention_head(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="local_attention"):
        build_sparse4d_head(make_anchor_file(tmp_path), local_attention=True)


def test_the_block_needs_bev_maps_and_reads_no_context() -> None:
    block = _block(_embedding())
    instance_feature = torch.zeros(1, 2, EMBED_DIMS)
    anchor = torch.zeros(1, 2, 10)

    with pytest.raises(ValueError, match="BEV maps"):
        block(instance_feature, anchor, instance_feature, [], [])
    with pytest.raises(ValueError, match="frame context"):
        block(instance_feature, anchor, instance_feature, [], [_level(2, 4.0, 0)], context=1)


def test_an_attention_head_needs_one_bev_source(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        build_sparse4d_head(
            make_anchor_file(tmp_path), lidar_sampling="bev_attention", bev_channels=BEV_CHANNELS
        )


def test_an_attention_head_trains_every_parameter_and_samples_no_voxels(tmp_path: Path) -> None:
    head = _attention_head(tmp_path)
    head.cls_threshold_to_reg = 1.0
    bev_map = torch.randn(1, BEV_CHANNELS, 8, 8)
    gt_boxes = [torch.tensor([[4.0, 4.0, 0.0, 4.5, 1.9, 1.7, 0.4, 0.0, 0.0]])]

    outputs = head(bev_map)
    losses = head.loss(outputs, gt_boxes, [torch.tensor([1])], [torch.tensor([0])], [0])
    losses["loss"].backward()

    missing = [
        name
        for name, parameter in head.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    assert missing == []
    assert head.levels(bev_map) == []
    assert all(isinstance(block, BEVCrossAttention) for block in head.deformable)
    shared = head.deformable[0].position_embedding
    assert all(block.position_embedding is shared for block in head.deformable)


def test_an_attention_head_exports_the_training_forward(tmp_path: Path) -> None:
    head = _attention_head(tmp_path).eval()
    export_head = head.prepare_for_export()
    bev_map = torch.randn(1, BEV_CHANNELS, 8, 8)

    with torch.no_grad():
        train_outputs = head(bev_map)
        export_outputs = export_head(bev_map)

    classification, prediction = export_outputs.export_tensors(["classification", "prediction"])
    assert torch.allclose(classification, train_outputs.classification[:, -1], atol=1e-5)
    assert torch.allclose(prediction, train_outputs.prediction[:, -1], atol=1e-5)
