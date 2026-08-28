"""Unit tests for the native TransFusion detector.

TransFusionHead itself is covered by
autoware_ml/models/detection3d/heads/tests/test_transfusion_head.py.
"""

from __future__ import annotations

import pytest
import torch

from autoware_ml.models.detection3d.backbones.second import SECONDBackbone
from autoware_ml.models.detection3d.encoders.sparse import SparseConv3d as NativeSparseConv3d
from autoware_ml.models.detection3d.encoders.sparse import SubMConv3d as NativeSubMConv3d
from autoware_ml.models.detection3d.encoders.voxel import HardSimpleVoxelSinCosEncoder
from autoware_ml.models.detection3d.heads.transfusion import (
    ExportableMultiheadAttention,
    TransFusionHead,
)
from autoware_ml.models.detection3d.necks.second_fpn import SECONDFPN
from autoware_ml.models.detection3d.task_modules.assigners import HungarianAssigner3D
from autoware_ml.models.detection3d.task_modules.bbox_coders import TransFusionBBoxCoder
from autoware_ml.models.detection3d.task_modules.match_costs import (
    BBoxBEVL1Cost,
    ClassificationCost,
    IoU3DCost,
)
from autoware_ml.models.detection3d.transfusion import TransFusionDetectionModel
from autoware_ml.ops.spconv.availability import IS_SPCONV_AVAILABLE
from autoware_ml.ops.spconv.sparse_conv import SubMConv3d as ExportableSubMConv3d

# Scaled-down mirror of tasks/detection3d/transfusion/base.yaml: an 8 m range
# with 0.25 m voxels gives a 32x32x40 grid, and the SparseEncoder's three
# stride-2 stages reduce it to the 4x4 BEV expected by the head at
# out_size_factor 8. The channel wiring is the config's (voxel encoder 32 ->
# sparse 128*Z2=256 -> SECOND [128, 256] -> FPN concat 512 -> head).
_POINT_CLOUD_RANGE = [0.0, 0.0, -5.0, 8.0, 8.0, 3.0]
_VOXEL_SIZE = [0.25, 0.25, 0.2]
_SPARSE_SHAPE = [32, 32, 41]
_OUT_SIZE_FACTOR = 8


def _build_model() -> TransFusionDetectionModel:
    from autoware_ml.models.detection3d.encoders.sparse import SparseEncoder

    return TransFusionDetectionModel(
        pts_voxel_encoder=HardSimpleVoxelSinCosEncoder(
            in_channels=4,
            min_norm_values=[0.0, 0.0, -5.0, 0.0],
            max_norm_values=[8.0, 8.0, 3.0, 255.0],
        ),
        pts_middle_encoder=SparseEncoder(
            in_channels=32,
            sparse_shape=_SPARSE_SHAPE,
            dense_output_shapes=[4, 4, 2],
        ),
        pts_backbone=SECONDBackbone(
            in_channels=256,
            out_channels=[128, 256],
            layer_nums=[1, 1],
            layer_strides=[1, 2],
        ),
        pts_neck=SECONDFPN(
            in_channels=[128, 256],
            out_channels=[256, 256],
            upsample_strides=[1, 2],
        ),
        bbox_head=TransFusionHead(
            num_proposals=8,
            auxiliary=True,
            in_channels=512,
            hidden_channel=128,
            num_classes=2,
            num_decoder_layers=1,
            num_heads=8,
            feedforward_channels=256,
            common_heads={
                "center": (2, 2),
                "height": (1, 2),
                "dim": (3, 2),
                "rot": (2, 2),
                "vel": (2, 2),
            },
            bbox_coder=TransFusionBBoxCoder(
                pc_range=[0.0, 0.0],
                out_size_factor=_OUT_SIZE_FACTOR,
                voxel_size=_VOXEL_SIZE[:2],
                post_center_range=[-1.0, -1.0, -10.0, 10.0, 10.0, 10.0],
                code_size=10,
            ),
            assigner=HungarianAssigner3D(
                cls_cost=ClassificationCost(weight=0.15),
                reg_cost=BBoxBEVL1Cost(weight=0.25),
                iou_cost=IoU3DCost(weight=0.25),
            ),
            point_cloud_range=_POINT_CLOUD_RANGE,
            voxel_size=_VOXEL_SIZE[:2],
            out_size_factor=_OUT_SIZE_FACTOR,
            code_weights=[1.0] * 8 + [0.2, 0.2],
            min_radius=1,
            gaussian_overlap=0.1,
            score_threshold=0.1,
            post_max_size=8,
            nms_min_radius=1.0,
        ),
    )


def _build_voxel_inputs(device: torch.device) -> dict[str, torch.Tensor]:
    """Voxelized inputs with unique in-grid coordinates in [batch, z, y, x]."""
    torch.manual_seed(0)
    height, width, depth = _SPARSE_SHAPE
    cells = torch.randperm(height * width * (depth - 1))[:12]
    coords = torch.stack(
        [
            torch.zeros_like(cells),
            cells // (height * width),
            cells % (height * width) // width,
            cells % width,
        ],
        dim=1,
    )
    return {
        "voxels": torch.randn(12, 5, 4, device=device),
        "num_points": torch.randint(1, 5, (12,), dtype=torch.int32, device=device),
        "voxel_coords": coords.to(dtype=torch.int32, device=device),
    }


@pytest.mark.skipif(
    not IS_SPCONV_AVAILABLE or not torch.cuda.is_available(),
    reason="TransFusion sparse middle encoder requires CUDA spconv",
)
def test_transfusion_forward_returns_query_predictions() -> None:
    model = _build_model().cuda()

    outputs = model(**_build_voxel_inputs(torch.device("cuda")))

    assert "dense_heatmap" in outputs
    assert "query_heatmap_score" in outputs
    assert "query_labels" in outputs
    assert outputs["heatmap"].shape[-1] == 8
    assert outputs["center"].shape[-1] == 8


@pytest.mark.skipif(
    not IS_SPCONV_AVAILABLE or not torch.cuda.is_available(),
    reason="TransFusion sparse middle encoder requires CUDA spconv",
)
def test_transfusion_build_export_spec_uses_deployment_io_contract() -> None:
    model = _build_model().cuda().eval()

    spec = model.build_export_spec(_build_voxel_inputs(torch.device("cuda")))
    with torch.no_grad():
        cls_score0, bbox_pred0, dir_cls_pred0 = spec.module(*spec.args)

    assert spec.input_param_names == ["voxels", "num_points", "coors"]
    assert spec.output_names == ["cls_score0", "bbox_pred0", "dir_cls_pred0"]
    assert cls_score0.shape == (1, 2, 8)
    assert bbox_pred0.shape == (1, 8, 8)
    assert dir_cls_pred0.shape == (1, 2, 8)


@pytest.mark.skipif(not IS_SPCONV_AVAILABLE, reason="TransFusion export prep requires spconv")
def test_transfusion_build_export_spec_prepares_modules_without_mutating_model() -> None:
    model = _build_model().eval()

    spec = model.build_export_spec(_build_voxel_inputs(torch.device("cpu")))

    assert isinstance(model.bbox_head.decoder[0].self_attn, torch.nn.MultiheadAttention)
    assert isinstance(spec.module.bbox_head.decoder[0].self_attn, ExportableMultiheadAttention)
    assert isinstance(model.pts_middle_encoder.conv_input[0], NativeSubMConv3d)
    assert isinstance(spec.module.pts_middle_encoder.conv_input[0], ExportableSubMConv3d)
    assert not any(
        isinstance(module, (NativeSubMConv3d, NativeSparseConv3d))
        for module in spec.module.pts_middle_encoder.modules()
    )


def test_transfusion_coder_supports_per_class_score_thresholds() -> None:
    coder = TransFusionBBoxCoder(
        pc_range=[0.0, 0.0],
        out_size_factor=1,
        voxel_size=[1.0, 1.0],
        post_center_range=[-100.0, -100.0, -100.0, 100.0, 100.0, 100.0],
        score_threshold=[0.3, 0.6],
        code_size=10,
    )
    heatmap = torch.tensor([[[0.5, 0.2, 0.1], [0.1, 0.1, 0.7]]])
    rot = torch.stack([torch.zeros(1, 1, 3), torch.ones(1, 1, 3)], dim=1).squeeze(2)
    zeros = torch.zeros(1, 3, 3)

    predictions = coder.decode(
        heatmap,
        rot,
        zeros,
        torch.zeros(1, 2, 3),
        torch.zeros(1, 1, 3),
        torch.zeros(1, 2, 3),
        filter_predictions=True,
    )

    # Class 0 keeps only 0.5 (> 0.3); class 1 keeps only 0.7 (> 0.6).
    assert predictions[0]["labels"].tolist() == [0, 1]
    assert torch.allclose(predictions[0]["scores"], torch.tensor([0.5, 0.7]))


def test_transfusion_coder_rejects_mismatched_threshold_length() -> None:
    coder = TransFusionBBoxCoder(
        pc_range=[0.0, 0.0],
        out_size_factor=1,
        voxel_size=[1.0, 1.0],
        score_threshold=[0.3],
        code_size=10,
    )
    heatmap = torch.rand(1, 2, 3)
    rot = torch.rand(1, 2, 3)

    with pytest.raises(ValueError, match="one value per class"):
        coder.decode(
            heatmap,
            rot,
            torch.rand(1, 3, 3),
            torch.rand(1, 2, 3),
            torch.rand(1, 1, 3),
            torch.rand(1, 2, 3),
            filter_predictions=True,
        )
