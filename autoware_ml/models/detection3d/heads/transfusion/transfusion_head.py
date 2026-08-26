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

"""TransFusion detection head components.

This module contains the query decoder, target generation, and loss logic used
by the native TransFusion lidar detector.
"""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from typing import Any, NamedTuple, Mapping
from types import MappingProxyType

from jaxtyping import Float32, Int32
import torch
import torch.nn as nn
import torch.nn.functional as F

from autoware_ml.dataclasses.multi_task_outputs import MultiTaskOutputs
from autoware_ml.dataclasses.detection3d.head_targets import TransFusionHeadTargets
from autoware_ml.dataclasses.detection3d.head_outputs import (
    Detection3DHeadOutputs,
    TransFusionHeadOutputs,
    TransFusionSeparateHeadOutputs,
)
from autoware_ml.losses.detection3d.focal import SigmoidFocalLoss
from autoware_ml.losses.detection3d.gaussian_focal import GaussianFocalLoss
from autoware_ml.models.common.layers.conv import ConvModule
from autoware_ml.models.detection3d.task_modules.assigners import HungarianAssigner3D
from autoware_ml.models.detection3d.task_modules.bbox_coders import TransFusionBBoxCoder
from autoware_ml.models.detection3d.task_modules.heatmap import (
    circle_nms,
    create_oriented_gaussian_heatmaps,
    vectorize_gaussian_radii,
    create_gaussian_heatmaps,
)
from autoware_ml.models.detection3d.heads.transfusion.transfusion_decoder_layer import (
    TransFusionDecoderLayer,
)
from autoware_ml.models.detection3d.heads.transfusion.exportable_multi_head_attention import (
    ExportableMultiheadAttention,
)
from autoware_ml.types.geometry import Box3DFieldIndex


class NMSGroupConfig(NamedTuple):
    """
    Config for each NMS group.
    Args:
        class_names: Names of classes for this group.
        class_ids: Optional class indices for this group.
            If not provided, class_names will be used to resolve indices.
        nms_radius: Optional NMS radius threshold for this group.
            If not provided, the default nms_radius will be used.
        max_size: Optional maximum number of predictions to keep for this group.
            If not provided, the default max_size will be used.
    """

    class_names: Sequence[str]
    nms_radius: float | None
    max_size: int | None


class NMSGroup(NamedTuple):
    """
    Resolved config for each NMS group.
    Args:
        class_ids: Class indices for this group.
        nms_radius: NMS radius threshold for this group.
        max_size: Maximum number of predictions to keep for this group.
    """

    class_ids: Sequence[int]
    nms_radius: float
    max_size: int


class SeparateHead1D(nn.Module):
    """Apply per-query prediction branches for TransFusion outputs.

    Each branch is implemented as a lightweight 1D convolutional stack over the
    query dimension.
    """

    def __init__(self, in_channels: int, heads: MappingProxyType[str, tuple[int, int]]) -> None:
        """Initialize the per-query prediction heads.

        Args:
            in_channels: Input feature dimension.
            heads: Mapping from head name to ``(out_channels, num_convs)``.
        """
        super().__init__()
        self.heads = nn.ModuleDict()
        for name, (out_channels, num_convs) in heads.items():
            layers: list[nn.Module] = []
            current_channels = in_channels
            for _ in range(max(num_convs - 1, 0)):
                layers.append(nn.Conv1d(current_channels, in_channels, kernel_size=1, bias=False))
                layers.append(nn.BatchNorm1d(in_channels, eps=1e-3, momentum=0.01))
                layers.append(nn.ReLU(inplace=True))
                current_channels = in_channels
            layers.append(nn.Conv1d(current_channels, out_channels, kernel_size=1))
            self.heads[name] = nn.Sequential(*layers)

        heatmaps = self.heads.get("heatmap", None)  # type: ignore
        if heatmaps is not None:
            nn.init.constant(heatmaps[-1].bias, -2.19)

    def forward(
        self, query_feats: Float32[torch.Tensor, "batch_size num_channels num_queries"]
    ) -> TransFusionSeparateHeadOutputs:
        """Apply all prediction branches to the query features.

        Args:
            query_feats: Per-query feature tensor.

        Returns:
            TransFusionSeparateHeadOutputs: Dataclass to save for each prediction branch.
        """
        outputs = MappingProxyType({name: head(query_feats) for name, head in self.heads.items()})
        return TransFusionSeparateHeadOutputs.from_dict(outputs)


class TransFusionHead(nn.Module):
    """Implement the native TransFusion lidar detection head.

    The head predicts a dense BEV heatmap, selects top proposals, refines them
    with decoder layers, and computes assignment-based training targets.
    """

    def __init__(
        self,
        num_proposals: int,
        auxiliary: bool,
        in_channels: int,
        hidden_channel: int,
        class_names: Sequence[str],
        num_decoder_layers: int,
        num_heads: int,
        feedforward_channels: int,
        common_heads: MappingProxyType[str, tuple[int, int]],
        bbox_coder: TransFusionBBoxCoder,
        assigner: HungarianAssigner3D,
        point_cloud_range: list[float],
        voxel_size: list[float],
        out_size_factor: int,
        code_weights: list[float],
        min_radius: int,
        gaussian_overlap: float,
        score_threshold: float,
        post_max_size: int,
        nms_min_radius: float,
        dense_heatmap_pooling_class_names: Sequence[str],
        heatmap_target: str = "round",
        nms_type: str | None = None,
        nms_group_configs: Sequence[NMSGroupConfig] | None = None,
        loss_cls_weight: float = 1.0,
        loss_bbox_weight: float = 0.25,
        loss_heatmap_weight: float = 1.0,
        heatmap_init_bias: float = -2.19,
        nms_kernel_size: int = 3,
        use_velocity: bool = True,
    ) -> None:
        """Initialize the TransFusion detection head.

        Args:
            num_proposals: Number of proposals selected from the dense heatmap.
            auxiliary: Whether to keep predictions from all decoder layers.
            class_names: Sequence of class names.
            in_channels: Input BEV feature dimension.
            hidden_channel: Internal decoder feature dimension.
            num_classes: Number of foreground classes.
            num_decoder_layers: Number of decoder refinement layers.
            num_heads: Number of attention heads per decoder layer.
            feedforward_channels: Hidden dimension of decoder feed-forward blocks.
            common_heads: Prediction-branch specification shared by decoder layers.
            bbox_coder: Box coder used for encoding and decoding predictions.
            assigner: Proposal assigner used during training.
            point_cloud_range: Detector point-cloud range.
            voxel_size: Voxel size used by the detector.
            out_size_factor: BEV downsampling factor between point and feature space.
            code_weights: Weights applied to each regression channel.
            min_radius: Minimum heatmap Gaussian radius.
            gaussian_overlap: Required overlap for Gaussian radius computation.
                Only used by the ``"round"`` heatmap target.
            score_threshold: Prediction score threshold used during decoding.
            post_max_size: Maximum number of predictions kept after NMS.
            nms_min_radius: Minimum center distance used by circle NMS.
            dense_heatmap_pooling_class_names: Optional class names that should use local max
                pooling before proposal selection.
            class_names: Optional ordered class names used to resolve config-friendly
                class lists.
            heatmap_target: Shape of the dense heatmap supervision. ``"round"``
                (default) draws a circular Gaussian sized by ``gaussian_radius``.
                ``"oriented"`` draws an elliptical Gaussian stretched along the box
                length and rotated by the box yaw, so elongated objects such as a
                tractor and trailer rig get one connected positive region instead of
                a small blob in the low-density gap at the coupling.
            nms_type: Optional NMS type applied during prediction. Supported values
                are ``None`` and ``"circle"``.
            nms_group_configs: Optional grouped NMS configuration. Each entry must provide
                ``class_names`` or ``class_ids`` and may override ``nms_radius``
                and ``post_max_size``. A group with ``nms_radius`` of ``0`` keeps
                its highest-scoring predictions up to ``post_max_size``.
            loss_cls_weight: Weight applied to the classification loss.
            loss_bbox_weight: Weight applied to the box regression loss.
            loss_heatmap_weight: Weight applied to the dense heatmap loss.
            heatmap_init_bias: Initial bias used by the dense heatmap branch.
            nms_kernel_size: Kernel size used for local-maximum suppression.
            use_velocity: Whether the head predicts object velocity.
        """
        super().__init__()
        self.num_proposals = num_proposals
        self.auxiliary = auxiliary
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.out_size_factor = out_size_factor
        self.code_weights = code_weights
        self.min_radius = min_radius
        self.gaussian_overlap = gaussian_overlap
        if heatmap_target not in {"round", "oriented"}:
            raise ValueError(f"Unsupported TransFusion heatmap_target: {heatmap_target!r}")
        self.heatmap_target = heatmap_target
        self.score_threshold = score_threshold
        self.post_max_size = post_max_size
        self.nms_min_radius = nms_min_radius
        self.nms_kernel_size = nms_kernel_size
        self.bbox_coder = bbox_coder
        self.assigner = assigner
        self.loss_cls_weight = loss_cls_weight
        self.loss_bbox_weight = loss_bbox_weight
        self.loss_heatmap_weight = loss_heatmap_weight
        self.heatmap_init_bias = heatmap_init_bias
        self.use_velocity = use_velocity
        if nms_type not in {None, "circle"}:
            raise ValueError(f"Unsupported TransFusion NMS type: {nms_type!r}")
        self.nms_type = nms_type
        self.dense_heatmap_pooling_class_ids = self._resolve_class_ids(
            dense_heatmap_pooling_class_names
        )
        self.nms_groups = self._resolve_nms_groups(nms_group_configs)

        self.shared_block = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channel, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.dense_heatmap_head = self._build_dense_heatmaps(hidden_channel)
        self.class_encoding = nn.Conv1d(self.num_classes, hidden_channel, kernel_size=1)
        self.decoder = nn.ModuleList(
            [
                TransFusionDecoderLayer(hidden_channel, num_heads, feedforward_channels)
                for _ in range(num_decoder_layers)
            ]
        )
        # Build prediction heads for each decoder layer.
        self.prediction_heads = self._build_prediction_heads(
            hidden_channel, num_decoder_layers, common_heads
        )

        self.loss_heatmap = GaussianFocalLoss()
        self.loss_cls = SigmoidFocalLoss()
        self.loss_bbox = nn.L1Loss(reduction="none")

    def _build_prediction_heads(
        self,
        in_channels: int,
        num_decoder_layers: int,
        prediction_head_configs: Mapping[str, tuple[int, int]],
    ) -> nn.ModuleList:
        """
        Build a list of per-query prediction heads for each decoder layer.

        Args:
            in_channels: Input feature dimension.
            num_decoder_layers: Number of decoder layers.
            prediction_head_configs: Mapping from head name to ``(out_channels, num_convs)``.
        """
        prediction_head_configs = dict(prediction_head_configs)
        if (
            "heatmap" in prediction_head_configs
            and prediction_head_configs["heatmap"][0] != self.num_classes
        ):
            raise ValueError(
                "TransFusionHead prediction heatmap head is not compatible "
                "with the number of classes."
            )
        else:
            prediction_head_configs["heatmap"] = (self.num_classes, 2)

        if not self.use_velocity and "vels" in prediction_head_configs:
            prediction_head_configs.pop("vels")

        prediction_heads = nn.ModuleList(
            [
                SeparateHead1D(in_channels, MappingProxyType(prediction_head_configs))
                for _ in range(num_decoder_layers)
            ]
        )
        return prediction_heads

    def _build_dense_heatmaps(self, in_channels: int) -> nn.Module:
        """
        Build the dense heatmap head used to predict BEV heatmaps for proposal selection.

        Args:
            in_channels: Input feature dimension.

        Returns:
            nn.Module: Dense heatmap head.
        """
        dense_heatmap_head = nn.Sequential(
            ConvModule(in_channels, in_channels),
            nn.Conv2d(in_channels, self.num_classes, kernel_size=3, padding=1),
        )
        nn.init.constant_(dense_heatmap_head[-1].bias, self.heatmap_init_bias)  # type: ignore
        return dense_heatmap_head

    def _resolve_class_ids(self, selected_class_names: Sequence[str]) -> Sequence[int]:
        """Resolve class names or indices into validated class ids."""
        resolved: list[int] = []
        for class_name in selected_class_names:
            class_id = self.class_names.index(class_name)
            if not 0 <= class_id < self.num_classes:
                raise ValueError(f"TransFusionHead class index: {class_id} out of range")
            resolved.append(class_id)

        return sorted(set(resolved))

    def _resolve_nms_groups(
        self, nms_config_groups: Sequence[NMSGroupConfig] | None
    ) -> Sequence[NMSGroup] | None:
        """Resolve grouped NMS configuration into class-id form."""
        if nms_config_groups is None:
            return None

        resolved_groups: list[NMSGroup] = []
        for nms_config_group in nms_config_groups:
            class_ids = self._resolve_class_ids(nms_config_group.class_names)
            if class_ids is None:
                raise ValueError("TransFusionHead NMS group must define class_names.")

            resolved_groups.append(
                NMSGroup(
                    class_ids=class_ids,
                    nms_radius=nms_config_group.nms_radius or self.nms_min_radius,
                    max_size=nms_config_group.max_size or self.post_max_size,
                )
            )
        return resolved_groups

    def _suppress_dense_heatmaps(
        self, heatmaps: Float32[torch.Tensor, "batch_size num_classes height width"]
    ) -> Float32[torch.Tensor, "batch_size num_classes height width"]:
        """
        Suppress non-maximal dense heatmap activations before proposal sampling. It produces a
        local-maximum mask for selected classes, where the value is 0 when it gets suppressed, and
        1 when it is kept.
        """
        # If the ids are empty, then no classes are selected for local max pooling,
        # so return the heatmaps as-is
        if not self.dense_heatmap_pooling_class_ids:
            return heatmaps

        local_max = heatmaps.clone()
        padding = self.nms_kernel_size // 2
        selected_heatmap = heatmaps[:, self.dense_heatmap_pooling_class_ids, :, :]
        pooled = F.max_pool2d(
            selected_heatmap,
            kernel_size=self.nms_kernel_size,
            stride=1,
            padding=0,
        )

        if padding == 0:
            local_max[:, self.dense_heatmap_pooling_class_ids, :, :] = pooled
        else:
            local_max[
                :,
                self.dense_heatmap_pooling_class_ids,
                padding:-padding,
                padding:-padding,
            ] = pooled
        return heatmaps * (local_max == heatmaps)

    def _circle_nms_groups(self) -> list[dict[str, Any]]:
        """Build grouped circle-NMS rules for prediction."""
        if self.nms_groups is not None:
            return list(self.nms_groups)
        return [
            {
                "class_ids": [class_id],
                "nms_radius": self.nms_min_radius,
                "post_max_size": self.post_max_size,
            }
            for class_id in range(self.num_classes)
        ]

    def _apply_circle_nms(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Apply grouped circle NMS and return kept indices."""

        keep_mask = torch.zeros(scores.shape[0], dtype=torch.bool, device=scores.device)
        covered_mask = torch.zeros(scores.shape[0], dtype=torch.bool, device=scores.device)

        for group in self._circle_nms_groups():
            group_mask = torch.zeros(scores.shape[0], dtype=torch.bool, device=scores.device)
            for class_id in group["class_ids"]:
                group_mask |= labels == class_id
            if not group_mask.any():
                continue
            covered_mask |= group_mask
            group_indices = group_mask.nonzero(as_tuple=False).squeeze(1)
            group_post_max_size = group["post_max_size"]
            if group["nms_radius"] <= 0:
                # No suppression: keep the group's highest-scoring predictions.
                keep = scores[group_mask].argsort(descending=True)[:group_post_max_size]
            else:
                keep = circle_nms(
                    boxes[group_mask],
                    scores[group_mask],
                    group["nms_radius"],
                    group_post_max_size,
                )
            keep_mask[group_indices[keep]] = True

        keep_mask |= ~covered_mask
        return keep_mask.nonzero(as_tuple=False).squeeze(1)

    def _create_2d_grid(
        self, width: int, height: int, device: torch.device
    ) -> Float32[torch.Tensor, "1 height*width 2"]:
        """Create BEV cell centers for positional encoding.

        Args:
            width: Feature-map width.
            height: Feature-map height.
            device: Device used for the created tensor.

        Returns:
            Flattened BEV (height*width) cell-center coordinates (x, y).
        """
        grid_y, grid_x = torch.meshgrid(
            torch.arange(height, device=device, dtype=torch.float32),
            torch.arange(width, device=device, dtype=torch.float32),
            indexing="ij",
        )
        grid = torch.stack([grid_x + 0.5, grid_y + 0.5], dim=-1)
        return grid.view(1, -1, 2)

    def _select_top_proposals(
        self,
        flatten_dense_heatmaps: Float32[torch.Tensor, "batch_size num_classes height*width"],
        flatten_bev_features: Float32[torch.Tensor, "batch_size channels height*width"],
        batch_size: int,
    ) -> tuple[
        Float32[torch.Tensor, "batch_size num_classes height*width"],
        Int32[torch.Tensor, "batch_size num_proposals"],
        Int32[torch.Tensor, "batch_size num_proposals"],
    ]:
        """Select top proposals from the dense heatmap.

        Args:
            flatten_dense_heatmaps: Flattened dense BEV heatmap tensor.
            flatten_bev_features: Flattened BEV feature tensor.
            batch_size: Batch size.

        Returns:
            query_feat: Selected query features.
        """
        proposal_count = min(
            self.num_proposals, flatten_dense_heatmaps.shape[1] * flatten_dense_heatmaps.shape[2]
        )
        _, top_indices = torch.topk(
            flatten_dense_heatmaps.view(batch_size, -1),
            k=proposal_count,
            dim=-1,
            largest=True,
            sorted=False,  # The order of proposals does not matter.
        )
        top_classes = top_indices // flatten_dense_heatmaps.shape[-1]
        top_positions = top_indices % flatten_dense_heatmaps.shape[-1]

        # (batch_size, channels, num_proposals)
        query_feat = flatten_bev_features.gather(
            2, top_positions[:, None, :].expand(-1, flatten_bev_features.shape[1], -1)
        )
        class_one_hot = (
            F.one_hot(top_classes, num_classes=self.num_classes).permute(0, 2, 1).float()
        )
        query_feat = query_feat + self.class_encoding(class_one_hot)
        return query_feat, top_classes, top_positions

    def _decoder_layer_forward(
        self,
        query_feats: Float32[torch.Tensor, "batch_size channels num_proposals"],
        flatten_bev_features: Float32[torch.Tensor, "batch_size channels height*width"],
        bev_pos: Float32[torch.Tensor, "batch_size height*width 2"],
        query_pos: Float32[torch.Tensor, "batch_size num_proposals 2"],
    ) -> Sequence[TransFusionSeparateHeadOutputs]:
        """Run the decoder layers and produce per-layer predictions.

        Args:
            query_feats: Per-query feature tensor.

        Returns:
            List of per-layer prediction outputs.
        """
        predictions: list[TransFusionSeparateHeadOutputs] = []
        for decoder_layer, prediction_head in zip(self.decoder, self.prediction_heads):
            query_feat = decoder_layer(
                query=query_feats, key=flatten_bev_features, query_pos=query_pos, bev_pos=bev_pos
            )
            prediction: TransFusionSeparateHeadOutputs = prediction_head(query_feat)
            centers = prediction.centers + query_pos.permute(0, 2, 1)
            prediction = prediction.model_copy(update={"centers": centers})
            predictions.append(prediction)
            query_pos = centers.detach().permute(0, 2, 1)

        return predictions

    def _generate_outputs(
        self,
        predictions: Sequence[TransFusionSeparateHeadOutputs],
        dense_heatmaps: Float32[torch.Tensor, "batch_size num_classes height width"],
        flatten_dense_heatmaps: Float32[torch.Tensor, "batch_size num_classes height*width"],
        top_classes: Int32[torch.Tensor, "batch_size num_proposals"],
        top_positions: Int32[torch.Tensor, "batch_size num_proposals"],
    ) -> TransFusionHeadOutputs:
        """Combine decoder predictions with dense heatmap and proposal info.

        Args:
            predictions: List of per-layer prediction outputs.
            dense_heatmap: Dense BEV heatmap tensor.
            flatten_dense_heatmap: Flattened dense BEV heatmap tensor.
            top_classes: Class indices of the selected proposals.
            top_positions: Spatial positions of the selected proposals.

        Returns:
            TransFusionHeadOutputs: Dataclass containing all outputs.
        """
        if self.auxiliary:
            outputs = {}
            ordered_keys = predictions[0].ordered_keys
            outputs = {key: [] for key in ordered_keys}
            for prediction in predictions:
                model_dump = prediction.model_dump()
                for key in ordered_keys:
                    outputs[key].append(model_dump[key])

            concat_outputs = {}
            for key in outputs.keys():
                concat_outputs[key] = torch.cat(outputs[key], dim=-1)

            separate_head_outputs = TransFusionSeparateHeadOutputs.from_dict(
                MappingProxyType(concat_outputs)
            )
        else:
            separate_head_outputs = predictions[-1]

        query_heatmap_scores = flatten_dense_heatmaps.gather(
            2, top_positions[:, None, :].expand(-1, flatten_dense_heatmaps.shape[1], -1)
        )
        return TransFusionHeadOutputs(
            dense_heatmaps=dense_heatmaps,
            query_heatmap_scores=query_heatmap_scores,
            query_labels=top_classes,
            separate_head_outputs=separate_head_outputs,
        )

    def forward(
        self, bev_features: Float32[torch.Tensor, "batch_size channels height width"]
    ) -> TransFusionHeadOutputs:
        """Predict TransFusion heatmap, queries, and box parameters.

        Args:
            bev_feats: BEV feature tensor.

        Returns:
            TransFusionHeadOutputs: Dataclass containing all outputs.
        """
        batch_size, _, height, width = bev_features.shape
        bev_features = self.shared_block(bev_features)
        bev_pos = self._create_2d_grid(width, height, bev_features.device).repeat(batch_size, 1, 1)

        # Do not use autocast for the dense heatmap head since the precision can affect
        # the selected proposals.
        with torch.autocast(device_type="cuda", enabled=False):
            dense_heatmaps = self.dense_heatmap_head(bev_features)

        suppressed_dense_heatmaps = self._suppress_dense_heatmaps(dense_heatmaps.detach().sigmoid())
        # (batch_size, num_classes, height, width) -> (batch_size, num_classes, height * width)
        flatten_dense_heatmaps = suppressed_dense_heatmaps.view(
            batch_size, suppressed_dense_heatmaps.shape[1], -1
        )

        # (batch_size, channels, height, width) -> (batch_size, channels, height * width)
        flatten_bev_features = bev_features.flatten(2)
        query_feat, top_classes, top_positions = self._select_top_proposals(
            flatten_dense_heatmaps, flatten_bev_features, batch_size
        )
        query_pos = bev_pos.gather(1, top_positions[..., None].expand(-1, -1, bev_pos.shape[-1]))
        decoder_layer_predictions = self._decoder_layer_forward(
            query_feats=query_feat,
            flatten_bev_features=flatten_bev_features,
            bev_pos=bev_pos,
            query_pos=query_pos,
        )
        return self._generate_outputs(
            predictions=decoder_layer_predictions,
            dense_heatmaps=dense_heatmaps,
            flatten_dense_heatmaps=flatten_dense_heatmaps,
            top_classes=top_classes,
            top_positions=top_positions,
        )

    def decode_outputs(self, outputs: Detection3DHeadOutputs) -> MultiTaskOutputs:
        """
        Decode TransFusion outputs into Detection3DSamplePredictions
        for each batch element.
        """
        return
        # if outputs.transfusion_head_outputs is None:
        #     raise ValueError(
        #         "TransFusionHead decode_outputs requires TransFusionHeadOutputs from forward()."
        #     )

        # transfusion_head_outputs = outputs.transfusion_head_outputs
        # batch_scores = transfusion_head_outputs.separate_head_outputs.heatmaps[
        #     ..., -self.num_proposals :
        # ].sigmoid()
        # query_labels = transfusion_head_outputs.query_labels

        # one_hot = (
        #     F.one_hot(query_labels, num_classes=self.num_classes)
        #     .permute(0, 2, 1)
        #     .to(batch_scores.dtype)
        # )
        # batch_scores = batch_scores * transfusion_head_outputs.query_heatmap_scores * one_hot
        # batch_centers = transfusion_head_outputs.separate_head_outputs.centers[
        #     ..., -self.num_proposals :
        # ]
        # batch_heights = transfusion_head_outputs.separate_head_outputs.heights[
        #     ..., -self.num_proposals :
        # ]
        # batch_dims = transfusion_head_outputs.separate_head_outputs.dims[..., -self.num_proposals :]
        # batch_rots = transfusion_head_outputs.separate_head_outputs.rots[..., -self.num_proposals :]
        # batch_vels = transfusion_head_outputs.separate_head_outputs.vels
        # if batch_vels is not None:
        #     batch_vels = batch_vels[..., -self.num_proposals :]

        # _ = self.bbox_coder.decode(
        #     batch_scores,
        #     batch_rots,
        #     batch_dims,
        #     batch_centers,
        #     batch_heights,
        #     batch_vels,
        #     filter_predictions=True,
        # )

    def predict(self, outputs: TransFusionHeadOutputs) -> list[dict[str, torch.Tensor]]:
        """Decode predictions into metric-space boxes.

        Args:
            outputs: Raw prediction tensors produced by the head.

        Returns:
            List of decoded prediction dictionaries, one per batch element.
        """
        batch_score = outputs["heatmap"][..., -self.num_proposals :].sigmoid()
        query_labels = outputs.get("query_labels")
        if query_labels is None:
            raise ValueError("TransFusion prediction requires query_labels from forward().")
        one_hot = (
            F.one_hot(query_labels, num_classes=self.num_classes)
            .permute(0, 2, 1)
            .to(batch_score.dtype)
        )
        batch_score = batch_score * outputs["query_heatmap_score"] * one_hot
        batch_center = outputs["center"][..., -self.num_proposals :]
        batch_height = outputs["height"][..., -self.num_proposals :]
        batch_dim = outputs["dim"][..., -self.num_proposals :]
        batch_rot = outputs["rot"][..., -self.num_proposals :]
        batch_vel = outputs.get("vel")
        if batch_vel is not None:
            batch_vel = batch_vel[..., -self.num_proposals :]

        decoded = self.bbox_coder.decode(
            batch_score,
            batch_rot,
            batch_dim,
            batch_center,
            batch_height,
            batch_vel,
            filter_predictions=True,
        )

        results = []
        for prediction in decoded:
            boxes = prediction["bboxes"]
            scores = prediction["scores"]
            labels = prediction["labels"]
            if boxes.numel() == 0:
                results.append({"bboxes_3d": boxes, "scores_3d": scores, "labels_3d": labels})
                continue
            if self.nms_type is None:
                kept_indices = torch.arange(scores.shape[0], device=scores.device)
            elif self.nms_type == "circle":
                kept_indices = self._apply_circle_nms(boxes, scores, labels)
            else:
                raise RuntimeError(
                    f"Unsupported TransFusion NMS type at runtime: {self.nms_type!r}"
                )
            results.append(
                {
                    "bboxes_3d": boxes[kept_indices],
                    "scores_3d": scores[kept_indices],
                    "labels_3d": labels[kept_indices],
                }
            )
        return results

    def _build_dense_heatmap_targets(
        self,
        gt_bboxes_3d: Float32[torch.Tensor, "batch_size max_num_3d_gt_bboxes num_Box3DFieldIndex"],
        gt_labels_3d: Float32[torch.Tensor, "batch_size max_num_3d_gt_bboxes"],
        gt_valid_bboxes: Int32[torch.Tensor, " batch_size"],
        feature_map_size: tuple[int, int],
        device: torch.device,
    ) -> Float32[torch.Tensor, "batch_size num_classes height width"]:
        """Build dense heatmap targets for query initialization.

        Args:
            gt_bboxes_3d: Ground-truth 3D boxes for each batch element.
            gt_labels: Ground-truth labels for each batch element.
            feature_shape: Heatmap height and width.
            device: Device used for the generated target tensor.

        Returns:
            Dense training heatmap targets.
        """
        batch_size = len(gt_bboxes_3d)
        height, width = feature_map_size

        """Build heatmap and regression targets for one batch."""
        batch_size = len(gt_bboxes_3d)
        max_num_bboxes = gt_bboxes_3d.shape[1]
        feature_height, feature_width = feature_map_size

        # Movement of tensors to the correct device and type
        # Get only the first K params for ground truths
        gt_bboxes_3d[:, :, : self.box_code_size] = gt_bboxes_3d[:, :, : self.box_code_size].to(
            device=device
        )
        gt_labels_3d = gt_labels_3d.to(device=device, dtype=torch.long)
        gt_valid_bboxes = gt_valid_bboxes.to(device=device)

        # Vectorization implementation instead of for-loops
        center_x = (
            (gt_bboxes_3d[:, :, Box3DFieldIndex.X] - self.point_cloud_range[0])
            / self.voxel_size[0]
            / self.out_size_factor
        )
        center_y = (
            (gt_bboxes_3d[:, :, Box3DFieldIndex.Y] - self.point_cloud_range[1])
            / self.voxel_size[1]
            / self.out_size_factor
        )

        # (batch_size, max_num_bboxes) boolean mask for valid boxes based on the distance
        valid_distance_masks = (
            (center_x >= 0)
            & (center_x < feature_width)
            & (center_y >= 0)
            & (center_y < feature_height)
        )

        # (batch_size, max_num_bboxes) boolean mask for valid boxes based on the number of valid boxes per sample
        # (max_num_bboxes) -> (1, max_num_bboxes) -> (batch_size, max_num_bboxes) < gt_valid_bboxes.unsqueeze(1) (batch_size, 1)
        # -> (batch_size, max_num_bboxes)
        valid_num_bboxes_masks = torch.arange(max_num_bboxes, device=device).unsqueeze(0).expand(
            batch_size, -1
        ) < gt_valid_bboxes.unsqueeze(1)
        # (batch_size, max_num_bboxes) boolean mask for valid boxes based on both distance and number of valid boxes
        valid_bbox_masks = valid_distance_masks & valid_num_bboxes_masks

        lengths = (
            gt_bboxes_3d[:, :, Box3DFieldIndex.LENGTH] / self.voxel_size[0] / self.out_size_factor
        )
        widths = (
            gt_bboxes_3d[:, :, Box3DFieldIndex.WIDTH] / self.voxel_size[1] / self.out_size_factor
        )

        center = torch.stack((center_x, center_y), dim=-1)
        # (batch_size, max_num_bboxes, 2)
        center_int = center.floor().to(torch.long)

        if self.heatmap_target == "oriented":
            # (batch_size, max_num_bboxes)
            dense_heatmaps = create_oriented_gaussian_heatmaps(
                heatmap_width=feature_width,
                heatmap_height=feature_height,
                num_classes=self.num_classes,
                centers=center_int,
                lengths_cells=lengths,
                widths_cells=widths,
                yaws=gt_bboxes_3d[:, :, Box3DFieldIndex.YAW],
                gt_bboxes_labels=gt_labels_3d,
                valid_masks=valid_bbox_masks,
                device=device,
                min_sigma=self.min_radius / 3.0,
            )
        else:
            # (batch_size, max_num_bboxes)
            gaussian_radii = vectorize_gaussian_radii(
                widths=lengths,
                heights=widths,
                min_overlap=self.gaussian_overlap,
            ).to(device)
            # Clamp the Gaussian radii to ensure they are at least the minimum radius
            gaussian_radii = torch.clamp(gaussian_radii, min=self.min_radius)

            dense_heatmaps = create_gaussian_heatmaps(
                heatmap_width=feature_width,
                heatmap_height=feature_height,
                num_classes=self.num_classes,
                centers=center_int,
                gaussian_radii=gaussian_radii.long(),
                gt_bboxes_labels=gt_labels_3d,
                valid_masks=valid_bbox_masks,
                device=device,
            )

        return dense_heatmaps

    def get_targets(
        self,
        gt_bboxes_3d: Float32[torch.Tensor, "batch_size max_num_3d_gt_bboxes num_Box3DFieldIndex"],
        gt_labels_3d: Float32[torch.Tensor, "batch_size max_num_3d_gt_bboxes"],
        gt_valid_bboxes: Int32[torch.Tensor, " batch_size"],
        feature_map_size: tuple[int, int],
        device: torch.device,
    ) -> TransFusionHeadTargets:
        """Build TransFusion training targets.

        Args:
            gt_boxes: Ground-truth boxes for each batch element.
            gt_labels: Ground-truth labels for each batch element.
            outputs: Raw prediction tensors produced by the head.

        Returns:
            Structured training targets for classification, boxes, and heatmaps.
        """
        # batch_size = len(gt_valid_bboxes)
        _ = self._build_dense_heatmap_targets(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_valid_bboxes=gt_valid_bboxes,
            feature_map_size=feature_map_size,
            device=device,
        )

        # batch_size = len(gt_boxes)
        # num_layers = outputs["center"].shape[-1] // self.num_proposals
        # all_labels = []
        # all_label_weights = []
        # all_bbox_targets = []
        # all_bbox_weights = []
        # num_pos = 0
        # matched_ious = 0.0

        # score = outputs["heatmap"].detach()
        # center = outputs["center"].detach()
        # height = outputs["height"].detach()
        # dim = outputs["dim"].detach()
        # rot = outputs["rot"].detach()
        # vel = outputs.get("vel")
        # if vel is not None:
        #     vel = vel.detach()

        # for batch_index in range(batch_size):
        #     batch_labels = []
        #     batch_label_weights = []
        #     batch_bbox_targets = []
        #     batch_bbox_weights = []
        #     gt_boxes_tensor = gt_boxes[batch_index].to(score.device, dtype=torch.float32)
        #     gt_labels_tensor = gt_labels[batch_index].to(score.device, dtype=torch.long)
        #     for layer_index in range(num_layers):
        #         start = layer_index * self.num_proposals
        #         end = (layer_index + 1) * self.num_proposals
        #         decoded = self.bbox_coder.decode(
        #             score[batch_index : batch_index + 1, :, start:end],
        #             rot[batch_index : batch_index + 1, :, start:end],
        #             dim[batch_index : batch_index + 1, :, start:end],
        #             center[batch_index : batch_index + 1, :, start:end],
        #             height[batch_index : batch_index + 1, :, start:end],
        #             vel[batch_index : batch_index + 1, :, start:end] if vel is not None else None,
        #             filter_predictions=False,
        #         )[0]["bboxes"]
        #         assign_result = self.assigner.assign(
        #             bboxes=decoded,
        #             gt_bboxes=gt_boxes_tensor[:, :7],
        #             gt_labels=gt_labels_tensor,
        #             cls_pred=score[batch_index, :, start:end],
        #             point_cloud_range=self.point_cloud_range,
        #         )
        #         labels = decoded.new_full((self.num_proposals,), self.num_classes, dtype=torch.long)
        #         label_weights = decoded.new_ones((self.num_proposals,))
        #         bbox_targets = decoded.new_zeros((self.num_proposals, self.bbox_coder.code_size))
        #         bbox_weights = decoded.new_zeros((self.num_proposals, self.bbox_coder.code_size))

        #         pos_mask = assign_result.gt_inds > 0
        #         if pos_mask.any():
        #             pos_gt_inds = assign_result.gt_inds[pos_mask] - 1
        #             labels[pos_mask] = gt_labels_tensor[pos_gt_inds]
        #             bbox_targets[pos_mask] = self.bbox_coder.encode(gt_boxes_tensor[pos_gt_inds])
        #             bbox_weights[pos_mask] = 1.0
        #             num_pos += int(pos_mask.sum().item())
        #             if assign_result.max_overlaps is not None:
        #                 matched_ious += float(assign_result.max_overlaps[pos_mask].sum().item())

        #         batch_labels.append(labels)
        #         batch_label_weights.append(label_weights)
        #         batch_bbox_targets.append(bbox_targets)
        #         batch_bbox_weights.append(bbox_weights)

        #     all_labels.append(torch.cat(batch_labels, dim=0))
        #     all_label_weights.append(torch.cat(batch_label_weights, dim=0))
        #     all_bbox_targets.append(torch.cat(batch_bbox_targets, dim=0))
        #     all_bbox_weights.append(torch.cat(batch_bbox_weights, dim=0))

        # dense_heatmap = self._build_heatmap_targets(
        #     gt_boxes,
        #     gt_labels,
        #     outputs["dense_heatmap"].shape[-2:],
        #     outputs["dense_heatmap"].device,
        # )
        # return TransFusionTargets(
        #     labels=torch.stack(all_labels, dim=0),
        #     label_weights=torch.stack(all_label_weights, dim=0),
        #     bbox_targets=torch.stack(all_bbox_targets, dim=0),
        #     bbox_weights=torch.stack(all_bbox_weights, dim=0),
        #     num_pos=num_pos,
        #     matched_iou=matched_ious / max(num_pos, 1),
        #     heatmap=dense_heatmap,
        # )
        return

    def loss(
        self,
        outputs: Detection3DHeadOutputs,
        gt_bboxes_3d: Float32[torch.Tensor, "batch_size max_num_3d_gt_bboxes num_Box3DFieldIndex"],
        gt_labels_3d: Float32[torch.Tensor, "batch_size max_num_3d_gt_bboxes"],
        gt_valid_bboxes: Int32[torch.Tensor, " batch_size"],
    ) -> MappingProxyType[str, torch.Tensor]:
        """
        Compute TransFusionHead losses. It will decode raw outputs and assign predictions to
        ground-truth boxes (1-to-1) for each decoder layer, and obtain positive and negative pairs.
        It computes the following losses:
          1) dense_heatmap loss: GaussianFocalLoss between the predicted dense heatmap
          and the target dense heatmap.
          2) cls_loss: SigmoidFocalLoss between the predicted class logits and the target labels
          from queries. For negative queries, the target label is set to the background class.
          3) bbox_loss: L1Loss between the predicted box parameters and the target box parameters.
        """
        _ = self.get_targets(
            gt_bboxes_3d,
            gt_labels_3d,
            gt_valid_bboxes,
            outputs["dense_heatmap"].shape[-2:],
            outputs["dense_heatmap"].device,
        )
        return {}
        # loss_dict: dict[str, torch.Tensor] = {}
        # loss_heatmap = self.loss_heatmap(outputs["dense_heatmap"], targets.heatmap)
        # loss_dict["loss_heatmap"] = self.loss_heatmap_weight * loss_heatmap

        # num_layers = outputs["center"].shape[-1] // self.num_proposals
        # for layer_index in range(num_layers):
        #     start = layer_index * self.num_proposals
        #     end = (layer_index + 1) * self.num_proposals
        #     prefix = "layer_-1" if layer_index == num_layers - 1 else f"layer_{layer_index}"

        #     layer_logits = (
        #         outputs["heatmap"][..., start:end].permute(0, 2, 1).reshape(-1, self.num_classes)
        #     )
        #     layer_labels = targets.labels[:, start:end].reshape(-1)
        #     cls_targets = layer_logits.new_zeros((layer_labels.shape[0], self.num_classes))
        #     valid_mask = layer_labels < self.num_classes
        #     cls_targets[valid_mask, layer_labels[valid_mask]] = 1.0
        #     layer_label_weights = targets.label_weights[:, start:end].reshape(-1)
        #     loss_cls = self.loss_cls(
        #         layer_logits, cls_targets, layer_label_weights, avg_factor=max(targets.num_pos, 1)
        #     )

        #     preds = torch.cat(
        #         [
        #             outputs["center"][..., start:end],
        #             outputs["height"][..., start:end],
        #             outputs["dim"][..., start:end],
        #             outputs["rot"][..., start:end],
        #             outputs["vel"][..., start:end]
        #             if "vel" in outputs
        #             else outputs["center"].new_zeros(
        #                 outputs["center"].shape[0], 0, self.num_proposals
        #             ),
        #         ],
        #         dim=1,
        #     ).permute(0, 2, 1)
        #     layer_bbox_targets = targets.bbox_targets[:, start:end, :]
        #     layer_bbox_weights = targets.bbox_weights[:, start:end, :] * preds.new_tensor(
        #         self.code_weights
        #     )
        #     loss_bbox = self.loss_bbox(preds, layer_bbox_targets)
        #     loss_bbox = (loss_bbox * layer_bbox_weights).

    # def loss(
    #     self,
    #     outputs: dict[str, torch.Tensor],
    #     gt_boxes: list[torch.Tensor],
    #     gt_labels: list[torch.Tensor],
    # ) -> dict[str, torch.Tensor]:
    #     """Compute TransFusion losses.

    #     Args:
    #         outputs: Raw prediction tensors produced by the head.
    #         gt_boxes: Ground-truth boxes for each batch element.
    #         gt_labels: Ground-truth labels for each batch element.

    #     Returns:
    #         Loss dictionary consumed by the training loop.
    #     """
    #     targets = self.get_targets(gt_boxes, gt_labels, outputs)
    #     loss_dict: dict[str, torch.Tensor] = {}
    #     loss_heatmap = self.loss_heatmap(outputs["dense_heatmap"], targets.heatmap)
    #     loss_dict["loss_heatmap"] = self.loss_heatmap_weight * loss_heatmap

    #     num_layers = outputs["center"].shape[-1] // self.num_proposals
    #     for layer_index in range(num_layers):
    #         start = layer_index * self.num_proposals
    #         end = (layer_index + 1) * self.num_proposals
    #         prefix = "layer_-1" if layer_index == num_layers - 1 else f"layer_{layer_index}"

    #         layer_logits = (
    #             outputs["heatmap"][..., start:end].permute(0, 2, 1).reshape(-1, self.num_classes)
    #         )
    #         layer_labels = targets.labels[:, start:end].reshape(-1)
    #         cls_targets = layer_logits.new_zeros((layer_labels.shape[0], self.num_classes))
    #         valid_mask = layer_labels < self.num_classes
    #         cls_targets[valid_mask, layer_labels[valid_mask]] = 1.0
    #         layer_label_weights = targets.label_weights[:, start:end].reshape(-1)
    #         loss_cls = self.loss_cls(
    #             layer_logits, cls_targets, layer_label_weights, avg_factor=max(targets.num_pos, 1)
    #         )

    #         preds = torch.cat(
    #             [
    #                 outputs["center"][..., start:end],
    #                 outputs["height"][..., start:end],
    #                 outputs["dim"][..., start:end],
    #                 outputs["rot"][..., start:end],
    #                 outputs["vel"][..., start:end]
    #                 if "vel" in outputs
    #                 else outputs["center"].new_zeros(
    #                     outputs["center"].shape[0], 0, self.num_proposals
    #                 ),
    #             ],
    #             dim=1,
    #         ).permute(0, 2, 1)
    #         layer_bbox_targets = targets.bbox_targets[:, start:end, :]
    #         layer_bbox_weights = targets.bbox_weights[:, start:end, :] * preds.new_tensor(
    #             self.code_weights
    #         )
    #         loss_bbox = self.loss_bbox(preds, layer_bbox_targets)
    #         loss_bbox = (loss_bbox * layer_bbox_weights).sum() / max(targets.num_pos, 1)

    #         loss_dict[f"{prefix}_loss_cls"] = self.loss_cls_weight * loss_cls
    #         loss_dict[f"{prefix}_loss_bbox"] = self.loss_bbox_weight * loss_bbox

    #     loss_dict["matched_ious"] = outputs["dense_heatmap"].new_tensor(targets.matched_iou)
    #     loss_dict["loss"] = sum(value for key, value in loss_dict.items() if "loss" in key)
    #     return loss_dict

    def prepare_for_export(self) -> "TransFusionHead":
        """Return an export-ready copy with attention replaced by exportable equivalents.

        Returns:
            Deep copy of the head with MultiheadAttention layers replaced by
            ExportableMultiheadAttention in all decoder layers.
        """
        head = deepcopy(self).eval()
        if not hasattr(head, "decoder"):
            return head
        for decoder_layer in head.decoder:
            if isinstance(decoder_layer.self_attn, nn.MultiheadAttention):
                decoder_layer.self_attn = ExportableMultiheadAttention(decoder_layer.self_attn)
            if isinstance(decoder_layer.cross_attn, nn.MultiheadAttention):
                decoder_layer.cross_attn = ExportableMultiheadAttention(decoder_layer.cross_attn)
        return head
