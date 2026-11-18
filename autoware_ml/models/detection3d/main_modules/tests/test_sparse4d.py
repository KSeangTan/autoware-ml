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

"""Tests for the Sparse4D detector on the BEVFusion lidar network."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from autoware_ml.models.detection3d.heads.sparse4d.sparse4d_head import (
    Sparse4DHead,
)
from autoware_ml.models.detection3d.heads.sparse4d.sparse_query_head import (
    TEMPORAL_STATE_INPUT_NAMES,
)
from autoware_ml.dataclasses.geometry.images import ImageGTBatch
from autoware_ml.dataclasses.models.model_batch_inputs import ModelBatchInputs
from autoware_ml.models.detection3d.main_modules.sparse4d import Sparse4DDetectionModel
from autoware_ml.models.detection3d.main_modules.sparse4ds.sparse4d_camera import Sparse4DCamera
from autoware_ml.models.detection3d.main_modules.sparse4ds.tests.lidar_fixtures import (
    NECK_CHANNELS,
    build_lidar_network,
)
from autoware_ml.models.detection3d.tests.ptv3_detection3d_fixtures import build_inputs
from autoware_ml.models.detection3d.tests.sparse4d_fixtures import (
    build_sparse4d_head,
    make_anchor_file,
)
from autoware_ml.models.detection3d.tests.sparse4d_image_fixtures import (
    IMAGE_CHANNELS,
    IMAGE_SIZE,
    IMAGE_STRIDES,
    lidar2image,
)
from autoware_ml.models.module_base_model import LogDictConfigs
from autoware_ml.models.multi.tests.test_ptv3_det3d_seg3d_sparse4d import frame_inputs
from autoware_ml.preprocessing.data_preprocessor import DataPreprocessor


def _build_head(
    tmp_path: Path, num_temp_instances: int = 0, dn_query_budget: int = 0
) -> Sparse4DHead:
    """Build a small head sampling the one metre cells of the neck map at the keypoints."""
    return build_sparse4d_head(
        make_anchor_file(tmp_path),
        num_temp_instances=num_temp_instances,
        dn_query_budget=dn_query_budget,
        lidar_sampling="bev",
        bev_cell_size=1.0,
        bev_channels=NECK_CHANNELS,
    )


def _build_model(
    tmp_path: Path, num_temp_instances: int = 0, dn_query_budget: int = 0
) -> Sparse4DDetectionModel:
    """Build a small detector over the stub lidar branch."""
    torch.manual_seed(0)
    return Sparse4DDetectionModel(
        data_preprocessor=DataPreprocessor(preprocessor_modules=[]),
        lidar_network=build_lidar_network(),
        camera_network=None,
        bbox_head=_build_head(tmp_path, num_temp_instances, dn_query_budget),
        log_dict_configs=LogDictConfigs(prog_bar=False),
    )


def test_a_branch_is_required(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="lidar_network, a camera_network or both"):
        Sparse4DDetectionModel(
            data_preprocessor=DataPreprocessor(preprocessor_modules=[]),
            lidar_network=None,
            camera_network=None,
            bbox_head=_build_head(tmp_path),
            log_dict_configs=LogDictConfigs(prog_bar=False),
        )


class _StubImageBackbone(nn.Module):
    """One stride 4 convolution over the images."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, IMAGE_CHANNELS, kernel_size=4, stride=4)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.conv(images)


class _StubImageNeck(nn.Module):
    """The backbone map and a stride 2 convolution of it, the strides 4 and 8."""

    def __init__(self) -> None:
        super().__init__()
        self.down = nn.Conv2d(IMAGE_CHANNELS, IMAGE_CHANNELS, kernel_size=2, stride=2)

    def forward(self, features: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        return features[0], self.down(features[0])


def _camera_network() -> Sparse4DCamera:
    return Sparse4DCamera(_StubImageBackbone(), _StubImageNeck(), IMAGE_STRIDES)


def _with_images(batch: ModelBatchInputs) -> ModelBatchInputs:
    """Add the front and the rear camera of the image fixtures to a batch of one frame."""
    projection = torch.stack([lidar2image(1.0), lidar2image(-1.0)])[None]
    identity = torch.eye(4).expand(1, 2, 4, 4)
    return batch.replace(
        image_data=ImageGTBatch(
            images=torch.randn(1, 2, 3, *IMAGE_SIZE),
            depth_maps=None,
            camera_intrinsics=torch.eye(3).expand(1, 2, 3, 3),
            image_augmentation_matrices=identity,
            lidar2images=projection,
            lidar2cams=identity,
            calibration_statuses=None,
        )
    )


def test_a_lidar_only_detector_keeps_one_optimizer_group(tmp_path: Path) -> None:
    model = _build_model(tmp_path)

    groups = model.build_optimizer_groups()

    assert list(groups) == ["default"]
    assert len(groups["default"]) == sum(p.requires_grad for p in model.parameters())


def test_the_image_backbone_and_neck_train_at_a_fraction_of_the_learning_rate(
    tmp_path: Path,
) -> None:
    model = Sparse4DDetectionModel(
        data_preprocessor=DataPreprocessor(preprocessor_modules=[]),
        lidar_network=None,
        camera_network=_camera_network(),
        bbox_head=build_sparse4d_head(
            make_anchor_file(tmp_path),
            lidar_sampling="none",
            image_strides=IMAGE_STRIDES,
            image_channels=IMAGE_CHANNELS,
        ),
        log_dict_configs=LogDictConfigs(prog_bar=False),
        optimizer=partial(torch.optim.AdamW, lr=1e-4),
        optimizer_group_overrides={
            "img_backbone": {"lr_scale": 0.1},
            "img_neck": {"lr_scale": 0.5},
        },
    )

    optimizer = model.configure_optimizers()
    learning_rates = {
        id(parameter): group["lr"]
        for group in optimizer.param_groups
        for parameter in group["params"]
    }

    # Every trainable parameter is optimized once, in the group of its module
    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    assert len(learning_rates) == len(trainable)
    assert sum(len(group["params"]) for group in optimizer.param_groups) == len(learning_rates)
    for name, parameter in trainable.items():
        expected = (
            1e-5
            if name.startswith("camera_network.img_backbone.")
            else 5e-5
            if name.startswith("camera_network.img_neck.")
            else 1e-4
        )
        assert learning_rates[id(parameter)] == pytest.approx(expected), name


@pytest.mark.parametrize("lidar_sampling", ["bev", "bev_attention", "none"])
def test_detector_trains_on_the_images_and_reaches_every_parameter(
    tmp_path: Path, lidar_sampling: str
) -> None:
    torch.manual_seed(0)
    with_lidar = lidar_sampling != "none"
    lidar_settings = {"bev_cell_size": 1.0, "bev_channels": NECK_CHANNELS} if with_lidar else {}
    model = Sparse4DDetectionModel(
        data_preprocessor=DataPreprocessor(preprocessor_modules=[]),
        lidar_network=build_lidar_network() if with_lidar else None,
        camera_network=_camera_network(),
        bbox_head=build_sparse4d_head(
            make_anchor_file(tmp_path),
            lidar_sampling=lidar_sampling,
            image_strides=IMAGE_STRIDES,
            image_channels=IMAGE_CHANNELS,
            num_temp_instances=4,
            dn_query_budget=8,
            **lidar_settings,
        ),
        log_dict_configs=LogDictConfigs(prog_bar=False),
    )

    for frame_index in range(2):
        model.zero_grad()
        batch = _with_images(frame_inputs(torch.device("cpu"), 100.0 + 0.1 * frame_index))
        outputs = model(batch)
        metrics = model.compute_metrics(batch, outputs)
        metrics["loss"].backward()

        assert torch.isfinite(metrics["loss"])
        missing = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and parameter.grad is None
        ]
        assert missing == []

    assert model.bbox_head.instance_bank._slot_valid.tolist() == [True]


def test_camera_branch_needs_the_images(tmp_path: Path) -> None:
    model = Sparse4DDetectionModel(
        data_preprocessor=DataPreprocessor(preprocessor_modules=[]),
        lidar_network=None,
        camera_network=_camera_network(),
        bbox_head=build_sparse4d_head(
            make_anchor_file(tmp_path),
            lidar_sampling="none",
            image_strides=IMAGE_STRIDES,
            image_channels=IMAGE_CHANNELS,
        ),
        log_dict_configs=LogDictConfigs(prog_bar=False),
    )

    with pytest.raises(ValueError, match="needs the images"):
        model(frame_inputs(torch.device("cpu"), 100.0))


def test_camera_branch_checks_the_maps_of_its_neck() -> None:
    camera_network = Sparse4DCamera(_StubImageBackbone(), _StubImageNeck(), (4,))

    with pytest.raises(ValueError, match="returned 2 maps"):
        camera_network(torch.zeros(1, 2, 3, *IMAGE_SIZE), torch.eye(4).expand(1, 2, 4, 4))
    with pytest.raises(ValueError, match="needs the strides"):
        Sparse4DCamera(_StubImageBackbone(), _StubImageNeck(), ())


def test_export_covers_the_lidar_only_detector(tmp_path: Path) -> None:
    model = Sparse4DDetectionModel(
        data_preprocessor=DataPreprocessor(preprocessor_modules=[]),
        lidar_network=build_lidar_network(),
        camera_network=_camera_network(),
        bbox_head=_build_head(tmp_path),
        log_dict_configs=LogDictConfigs(prog_bar=False),
    )

    with pytest.raises(NotImplementedError, match="camera"):
        model.build_export_specs(frame_inputs(torch.device("cpu"), 100.0))


def test_the_local_attention_of_the_head_follows_the_training_epoch(tmp_path: Path) -> None:
    torch.manual_seed(0)
    head = build_sparse4d_head(
        make_anchor_file(tmp_path),
        lidar_sampling="bev_attention",
        bev_cell_size=1.0,
        bev_channels=NECK_CHANNELS,
        local_attention=True,
        local_attention_radii=[[0, 3.0], [2, None]],
    )
    model = Sparse4DDetectionModel(
        data_preprocessor=DataPreprocessor(preprocessor_modules=[]),
        lidar_network=build_lidar_network(),
        camera_network=None,
        bbox_head=head,
        log_dict_configs=LogDictConfigs(prog_bar=False),
    )

    def radius_at(epoch: int) -> float:
        model._trainer = SimpleNamespace(current_epoch=epoch)
        model.on_train_epoch_start()
        return float(head.deformable[0].local_radius)

    assert [radius_at(epoch) for epoch in range(4)] == [3.0, 3.0, float("inf"), float("inf")]


def test_temporal_detector_trains_on_a_stream_and_reaches_every_parameter(
    tmp_path: Path,
) -> None:
    device = torch.device("cpu")
    model = _build_model(tmp_path, num_temp_instances=4, dn_query_budget=8)

    for frame_index in range(2):
        model.zero_grad()
        batch = frame_inputs(device, 100.0 + 0.1 * frame_index)
        outputs = model(batch)
        metrics = model.compute_metrics(batch, outputs)
        metrics["loss"].backward()

        assert torch.isfinite(metrics["loss"])
        assert outputs.segmentation3d_head_outputs is None
        missing = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and parameter.grad is None
        ]
        assert missing == []

    # The second frame of the scene reads the instances the first one cached
    assert model.bbox_head.instance_bank._slot_valid.tolist() == [True]


def test_detector_eval_output_and_predictions_carry_boxes_only(tmp_path: Path) -> None:
    device = torch.device("cpu")
    model = _build_model(tmp_path, num_temp_instances=4).eval()

    eval_outputs = []
    with torch.no_grad():
        for frame_index in range(2):
            batch = frame_inputs(device, 100.0 + 0.1 * frame_index)
            outputs = model(batch)
            eval_outputs.append(model.build_eval_output(batch, outputs))
        predictions = model.decode_outputs(batch, outputs)

    first, second = eval_outputs
    assert first["scene_token"] == ["db/scene/0"]
    assert first["gt_observed"][0].all()
    assert first["stream_ids"][0] == second["stream_ids"][0]
    assert "pred_labels" not in first
    assert predictions.detection3d_predictions[0].bboxes_3d.shape[1] == 9
    assert predictions.segmentation3d_predictions is None


def test_forward_needs_frame_metadata(tmp_path: Path) -> None:
    model = _build_model(tmp_path)
    with pytest.raises(ValueError, match="frame metadata"):
        model(build_inputs())


def test_forward_needs_voxels(tmp_path: Path) -> None:
    model = _build_model(tmp_path)
    batch = frame_inputs(torch.device("cpu"), 100.0)
    with pytest.raises(ValueError, match="voxels"):
        model(batch.replace(voxels_data=None))


def test_export_spec_takes_the_voxels_and_the_bank_state_of_the_last_frame(
    tmp_path: Path,
) -> None:
    device = torch.device("cpu")
    model = _build_model(tmp_path, num_temp_instances=4).eval()
    batch = frame_inputs(device, 100.0)

    specs = model.build_export_specs(batch)
    spec = specs["sparse4d_lidar"]
    with torch.no_grad():
        first = dict(zip(spec.output_names, spec.module(*spec.args), strict=True))

    assert list(specs) == ["sparse4d_lidar"]
    assert spec.input_names[:3] == ["voxels", "coors", "num_points_per_voxel"]
    assert spec.input_names[3:] == list(TEMPORAL_STATE_INPUT_NAMES)
    assert len(spec.args) == len(spec.input_names)
    # The runtime layout: single sample voxels with (z, y, x) coordinates
    assert spec.args[1].shape[1] == 3
    assert spec.output_names[:3] == ["classification", "prediction", "quality"]
    assert spec.output_names[-1] == "temp_query_index_out"
    assert first["temp_anchor_out"].shape == (1, 4, 10)
    assert first["temp_query_index_out"].shape == (1, 4)
    # The export copy is prepared apart from the trained model
    assert spec.module.lidar_network.pts_middle_encoder.exportable
    assert not model.lidar_network.pts_middle_encoder.exportable


def test_exported_graph_reads_the_state_of_the_last_frame(tmp_path: Path) -> None:
    device = torch.device("cpu")
    model = _build_model(tmp_path, num_temp_instances=4).eval()
    spec = model.build_export_specs(frame_inputs(device, 100.0))["sparse4d_lidar"]
    voxel_args = spec.args[:3]

    with torch.no_grad():
        first = dict(zip(spec.output_names, spec.module(*spec.args), strict=True))
        second = dict(
            zip(
                spec.output_names,
                spec.module(
                    voxel_args[0],
                    voxel_args[1],
                    voxel_args[2],
                    first["temp_instance_feature_out"],
                    first["temp_anchor_out"],
                    first["temp_confidence_out"],
                    torch.ones(1, device=device),
                    torch.full((1,), 0.1, device=device),
                    torch.eye(4, device=device)[None],
                ),
                strict=True,
            )
        )

    assert second["classification"].shape == first["classification"].shape
    assert not torch.equal(second["classification"], first["classification"])
