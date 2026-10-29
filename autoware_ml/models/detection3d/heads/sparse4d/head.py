# Copyright (c) 2024 Horizon Robotics. MIT License.
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
#
# Adapted from projects/mmdet3d_plugin/models/sparse4d_head.py of
# https://github.com/HorizonRobotics/Sparse4D (MIT License). The operation order of the
# reference is one ``Sparse4DDecoderLayer`` per layer built from prototype modules instead
# of registry configs, the decoupled attention lives in ``GraphAttention``, the frame
# metadata comes in as tensors, and the losses are normalized by the positive count averaged
# over the DDP ranks like the StreamPETR head.

"""The Sparse4D head.

The head refines a set of anchors through ``num_decoder`` decoder layers, see
:class:`Sparse4DDecoderLayer`. The instance bank carries the most confident instances to the
next frame, merged in after the single-frame layers, and in training noised copies of the
ground truth boxes are refined alongside.
"""

from __future__ import annotations

from collections.abc import Sequence
import copy
from typing import NamedTuple

from jaxtyping import Bool, Float, Float64, Int64
import torch
import torch.nn as nn
import torch.nn.functional as F

from autoware_ml.dataclasses.models.detection2d.feature_maps import FlattenedFeatureMaps
from autoware_ml.dataclasses.models.detection3d.head_outputs import (
    Sparse4DDenoisingOutputs,
    Sparse4DHeadOutputs,
)
from autoware_ml.dataclasses.models.detection3d.predictions import Detection3DSamplePredictions
from autoware_ml.losses.detection3d.focal import SigmoidFocalLoss
from autoware_ml.models.detection3d.heads.sparse4d.blocks import (
    AsymmetricFFN,
    DeformableFeatureAggregation,
)
from autoware_ml.models.detection3d.heads.sparse4d.box3d import (
    SparseBox3DEncoder,
    SparseBox3DRefinementModule,
)
from autoware_ml.models.detection3d.heads.sparse4d.decoder import SparseBox3DDecoder
from autoware_ml.models.detection3d.heads.sparse4d.decoder_layer import Sparse4DDecoderLayer
from autoware_ml.models.detection3d.heads.sparse4d.instance_bank import (
    InstanceBank,
    InstanceBankState,
)
from autoware_ml.models.detection3d.heads.sparse4d.loss import SparseBox3DLoss
from autoware_ml.models.detection3d.heads.sparse4d.target import SparseBox3DTarget
from autoware_ml.models.detection3d.task_modules.streaming import (
    StreamContinuity,
    reduce_mean_count,
)


class DenoisingTargets(NamedTuple):
    """Targets of the noised anchors appended to the learnable instances.

    Attributes:
        box_target: Encoded box every noised anchor is matched to, in the anchor width.
        cls_target: Class every noised anchor is matched to.
        valid_mask: True for the noised anchors that come from a real box.
        id_target: Instance id every noised anchor is matched to, ``None`` without ids.
    """

    box_target: Float[torch.Tensor, "batch_size num_noised anchor_dims"]
    cls_target: Int64[torch.Tensor, "batch_size num_noised"]
    valid_mask: Bool[torch.Tensor, "batch_size num_noised"]
    id_target: Int64[torch.Tensor, "batch_size num_noised"] | None


class NoisedInstances(NamedTuple):
    """The learnable instances with the noised anchors appended.

    Attributes:
        instance_feature: Features of the learnable instances followed by zeros for the
            noised anchors.
        anchor: The learnable anchors followed by the noised ones.
        attn_mask: True where an instance must not attend to another, ``None`` without
            noised anchors.
        dn_targets: Targets of the noised anchors, ``None`` without them.
    """

    instance_feature: Float[torch.Tensor, "batch_size num_instances embed_dims"]
    anchor: Float[torch.Tensor, "batch_size num_instances anchor_dims"]
    attn_mask: Bool[torch.Tensor, "num_instances num_instances"] | None
    dn_targets: DenoisingTargets | None


class RefinedLayers(NamedTuple):
    """Outputs of every decoder layer and the state after the last one.

    Attributes:
        prediction: Anchors after every layer.
        classification: Class logits after every layer.
        quality: Quality logits after every layer, ``None`` without quality estimation.
        instance_feature: Instance features after the last layer.
        anchor: Anchors after the last layer.
        cls: Class logits after the last layer.
        temp_dn_targets: Targets of the noised anchors after the cached groups came back,
            ``None`` when none did.
    """

    prediction: Float[torch.Tensor, "num_layers batch_size num_instances anchor_dims"]
    classification: Float[torch.Tensor, "num_layers batch_size num_instances num_classes"]
    quality: Float[torch.Tensor, "num_layers batch_size num_instances 2"] | None
    instance_feature: Float[torch.Tensor, "batch_size num_instances embed_dims"]
    anchor: Float[torch.Tensor, "batch_size num_instances anchor_dims"]
    cls: Float[torch.Tensor, "batch_size num_instances num_classes"]
    temp_dn_targets: DenoisingTargets | None


def clone_module(module: nn.Module) -> nn.Module:
    """Copy a prototype module with freshly drawn parameters."""
    clone = copy.deepcopy(module)
    for submodule in clone.modules():
        if hasattr(submodule, "reset_parameters"):
            submodule.reset_parameters()
    if hasattr(clone, "init_weights"):
        clone.init_weights()
    return clone


class Sparse4DHead(nn.Module):
    """Sparse4D decoder head with the instance bank, the targets, the losses and the decoder.

    The blocks of every decoder layer are clones of the given prototypes, so a config
    describes each block once; ``init_weights`` draws their parameters as the reference does.
    """

    # Scene and time of the previous frame of every batch lane.
    stream_continuity: StreamContinuity

    def __init__(
        self,
        instance_bank: InstanceBank,
        anchor_encoder: SparseBox3DEncoder,
        ffn: AsymmetricFFN,
        deformable_model: DeformableFeatureAggregation,
        refine_layer: SparseBox3DRefinementModule,
        target: SparseBox3DTarget,
        loss_reg: SparseBox3DLoss,
        decoder: SparseBox3DDecoder,
        num_classes: int,
        num_heads: int = 8,
        num_decoder: int = 6,
        num_single_frame_decoder: int = 1,
        temporal: bool = True,
        decouple_attn: bool = True,
        attn_drop: float = 0.1,
        reg_weights: Sequence[float] | None = None,
        cls_threshold_to_reg: float = -1.0,
        loss_cls_weight: float = 2.0,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        max_stream_gap: float | None = None,
    ) -> None:
        """Initialize the head.

        Args:
            instance_bank: The learnable and the temporal instances.
            anchor_encoder: The anchor embedding.
            ffn: Prototype of the feed-forward network of every layer.
            deformable_model: Prototype of the deformable aggregation of every layer.
            refine_layer: Prototype of the refinement of every layer.
            target: The matching and the denoising.
            loss_reg: The box regression loss.
            decoder: The decoding of the final anchors.
            num_classes: Number of classes, which the refinement predicts.
            num_heads: Attention heads of the graph attentions.
            num_decoder: Number of decoder layers.
            num_single_frame_decoder: Number of leading layers without temporal attention;
                the instance bank merges the temporal instances after them.
            temporal: Whether the temporal attentions exist.
            decouple_attn: Whether the graph attentions attend over the instance feature and
                its anchor embedding concatenated, with one pair of value projections shared
                by every layer as in the reference.
            attn_drop: Dropout of the attention weights.
            reg_weights: Weight of every regressed box field in the losses; their count is
                the number of leading box fields the losses read. ``None`` for one on ten
                fields.
            cls_threshold_to_reg: Confidence a matched anchor needs to join the box loss,
                negative for all.
            loss_cls_weight: Weight of the classification loss.
            focal_gamma: Focal exponent of the classification loss.
            focal_alpha: Focal balance of the classification loss.
            max_stream_gap: Longest gap in seconds between two frames of a lane that keeps
                the temporal instances, ``None`` for any gap (see :class:`StreamContinuity`).

        Raises:
            ValueError: If there is no decoder layer or more single-frame layers than layers.
        """
        super().__init__()
        if num_decoder < 1:
            raise ValueError(f"num_decoder must be positive, got {num_decoder}.")
        if not 0 <= num_single_frame_decoder <= num_decoder:
            raise ValueError(
                f"num_single_frame_decoder={num_single_frame_decoder} must lie within the "
                f"{num_decoder} decoder layers."
            )

        self.instance_bank = instance_bank
        self.anchor_encoder = anchor_encoder
        self.target = target
        self.loss_reg = loss_reg
        self.decoder = decoder
        self.num_classes = num_classes
        self.num_decoder = num_decoder
        self.num_single_frame_decoder = num_single_frame_decoder
        self.decouple_attn = decouple_attn
        self.reg_weights = list(reg_weights) if reg_weights is not None else [1.0] * 10
        self.cls_threshold_to_reg = cls_threshold_to_reg
        self.loss_cls_weight = loss_cls_weight
        self.loss_cls = SigmoidFocalLoss(gamma=focal_gamma, alpha=focal_alpha)

        self.embed_dims = instance_bank.embed_dims
        # The decoupled attentions of every layer share one pair of value projections, as in
        # the reference; a shared module appears under every layer in the state dict.
        value_proj = output_proj = None
        if decouple_attn:
            value_proj = nn.Linear(self.embed_dims, self.embed_dims * 2, bias=False)
            output_proj = nn.Linear(self.embed_dims * 2, self.embed_dims, bias=False)
        # The first layer has no attention: every instance starts from the same feature.
        self.layers = nn.ModuleList(
            Sparse4DDecoderLayer(
                self.embed_dims,
                num_heads,
                clone_module(ffn),
                clone_module(deformable_model),
                clone_module(refine_layer),
                with_self_attention=index > 0,
                with_temporal_attention=temporal and index >= num_single_frame_decoder,
                attn_drop=attn_drop,
                decouple_attn=decouple_attn,
                value_proj=value_proj,
                output_proj=output_proj,
            )
            for index in range(num_decoder)
        )
        self.stream_continuity = StreamContinuity(max_stream_gap=max_stream_gap)
        self.init_weights()

    def init_weights(self) -> None:
        """Draw the parameters of the layers, the instance bank and the anchor encoder."""
        for layer in self.layers:
            layer.init_weights()
        for module in (self.instance_bank, self.anchor_encoder):
            if hasattr(module, "init_weights"):
                module.init_weights()

    def reset_memory(self) -> None:
        """Forget the temporal instances, the cached denoising groups and the previous frames."""
        self.instance_bank.reset()
        self.target.dn_cache = None
        self.stream_continuity.reset()

    def forward(
        self,
        feature_maps: FlattenedFeatureMaps
        | Sequence[Float[torch.Tensor, "batch_size num_cams embed_dims height width"]],
        projection_mat: Float[torch.Tensor, "batch_size num_cams 4 4"],
        image_wh: Float[torch.Tensor, "batch_size num_cams 2"],
        timestamp: Float64[torch.Tensor, " batch_size"],
        ego2global: Float[torch.Tensor, "batch_size 4 4"],
        prev_exists: Float[torch.Tensor, " batch_size"] | Bool[torch.Tensor, " batch_size"],
        padding_mask: Bool[torch.Tensor, " batch_size"] | None = None,
        gt_boxes: Sequence[Float[torch.Tensor, "num_boxes num_fields"]] | None = None,
        gt_labels: Sequence[Int64[torch.Tensor, " num_boxes"]] | None = None,
        gt_instance_ids: Sequence[Int64[torch.Tensor, " num_boxes"]] | None = None,
    ) -> Sparse4DHeadOutputs:
        """Refine the anchors of a frame.

        Args:
            feature_maps: Image feature pyramid of every camera, flattened for the CUDA
                kernel or one tensor per level for the ``grid_sample`` path.
            projection_mat: Transform from the ego frame to the pixels of every camera.
            image_wh: Width and height of every camera image.
            timestamp: Sample time of every lane in seconds.
            ego2global: Transform from the ego frame of every lane to the map frame.
            prev_exists: Nonzero for every lane whose stream continues from the previous
                frame, from :attr:`stream_continuity`.
            padding_mask: True for the padding samples of a streaming batch, which start a
                stream too and carry no boxes.
            gt_boxes: Metric boxes of every sample, see ``Box3DFieldIndex``, which seed the
                noised anchors in training.
            gt_labels: Their classes.
            gt_instance_ids: Their instance ids, for the temporal denoising groups.

        Returns:
            The predictions of every decoder layer.
        """
        state = self._start_frame(timestamp, ego2global, prev_exists, padding_mask)
        num_free_instance = state.anchor.shape[1]
        noised = self._append_noised_anchors(
            state.instance_feature, state.anchor, gt_boxes, gt_labels, gt_instance_ids
        )
        refined = self._refine(
            noised.instance_feature,
            noised.anchor,
            state.temp_instance_feature,
            state.temp_anchor,
            feature_maps,
            projection_mat,
            image_wh,
            state.time_interval,
            noised.attn_mask,
            noised.dn_targets,
        )
        refined, denoising_outputs = self._split_denoising(
            refined, noised.dn_targets, num_free_instance
        )

        self.instance_bank.cache(
            refined.instance_feature, refined.anchor, refined.cls, timestamp, ego2global
        )
        instance_id = None
        if not self.training:
            instance_id = self.instance_bank.get_instance_id(
                refined.cls, self.decoder.score_threshold
            )
        return Sparse4DHeadOutputs(
            prediction=refined.prediction,
            classification=refined.classification,
            quality=refined.quality,
            instance_id=instance_id,
            denoising=denoising_outputs,
        )

    def _start_frame(
        self,
        timestamp: Float64[torch.Tensor, " batch_size"],
        ego2global: Float[torch.Tensor, "batch_size 4 4"],
        prev_exists: Float[torch.Tensor, " batch_size"] | Bool[torch.Tensor, " batch_size"],
        padding_mask: Bool[torch.Tensor, " batch_size"] | None,
    ) -> InstanceBankState:
        """Read the instances a frame starts from and move the caches into its ego frame.

        A lane continues when its stream does and it is not padding. The cached denoising
        groups follow the temporal instances through the bank's projection, and are dropped
        when the batch size changed.

        Args:
            timestamp: Sample time of every lane in seconds.
            ego2global: Transform from the ego frame of every lane to the map frame.
            prev_exists: Nonzero for every lane whose stream continues.
            padding_mask: True for the padding samples, ``None`` for none.

        Returns:
            The instances of the frame.
        """
        batch_size = timestamp.shape[0]
        continues = prev_exists.to(dtype=torch.bool)
        if padding_mask is not None:
            continues = continues & ~padding_mask

        dn_cache = self.target.dn_cache
        if dn_cache is not None and dn_cache.anchor.shape[0] != batch_size:
            self.target.dn_cache = dn_cache = None
        state = self.instance_bank.get(
            timestamp,
            ego2global,
            continues,
            temp_dn_anchor=None if dn_cache is None else dn_cache.anchor,
        )
        if dn_cache is not None and state.temp_dn_anchor is not None:
            self.target.dn_cache = dn_cache._replace(anchor=state.temp_dn_anchor)
        return state

    def _append_noised_anchors(
        self,
        instance_feature: Float[torch.Tensor, "batch_size num_anchors embed_dims"],
        anchor: Float[torch.Tensor, "batch_size num_anchors anchor_dims"],
        gt_boxes: Sequence[Float[torch.Tensor, "num_boxes num_fields"]] | None,
        gt_labels: Sequence[Int64[torch.Tensor, " num_boxes"]] | None,
        gt_instance_ids: Sequence[Int64[torch.Tensor, " num_boxes"]] | None,
    ) -> NoisedInstances:
        """Append the noised copies of the ground truth boxes in training.

        The noised anchors hold the regressed fields and start at zero on the rest, with
        zero features; the attention mask keeps them apart from the learnable instances
        and from the other groups. Outside training, or without boxes, the instances come
        back unchanged.

        Args:
            instance_feature: Features of the learnable instances.
            anchor: Their anchors.
            gt_boxes: Metric boxes of every sample, see ``Box3DFieldIndex``.
            gt_labels: Their classes.
            gt_instance_ids: Their instance ids, for the temporal denoising groups.

        Returns:
            The instances with the noised anchors, their attention mask and their targets.
        """
        denoising = None
        if self.training and gt_boxes is not None and gt_labels is not None:
            denoising = self.target.get_dn_anchors(
                gt_labels, gt_boxes, gt_instance_ids, box_dims=len(self.reg_weights)
            )
        if denoising is None:
            return NoisedInstances(instance_feature, anchor, None, None)

        batch_size, num_free_instance = anchor.shape[:2]
        num_dn_anchor = denoising.anchors.shape[1]
        missing = anchor.shape[-1] - denoising.anchors.shape[-1]
        dn_anchor = F.pad(denoising.anchors.to(anchor.dtype), (0, missing))
        anchor = torch.cat([anchor, dn_anchor], dim=1)
        instance_feature = torch.cat(
            [
                instance_feature,
                instance_feature.new_zeros(batch_size, num_dn_anchor, self.embed_dims),
            ],
            dim=1,
        )
        num_instance = anchor.shape[1]
        attn_mask = anchor.new_ones((num_instance, num_instance), dtype=torch.bool)
        attn_mask[:num_free_instance, :num_free_instance] = False
        attn_mask[num_free_instance:, num_free_instance:] = denoising.attn_mask
        dn_targets = DenoisingTargets(
            box_target=F.pad(denoising.box_target.to(anchor.dtype), (0, missing)),
            cls_target=denoising.cls_target,
            valid_mask=denoising.valid_mask,
            id_target=denoising.id_target,
        )
        return NoisedInstances(instance_feature, anchor, attn_mask, dn_targets)

    def _split_denoising(
        self,
        refined: RefinedLayers,
        dn_targets: DenoisingTargets | None,
        num_free_instance: int,
    ) -> tuple[RefinedLayers, Sparse4DDenoisingOutputs | None]:
        """Split the noised anchors off the refined instances and cache their groups.

        Args:
            refined: The outputs of the layers over the learnable and the noised instances.
            dn_targets: Targets of the noised anchors, ``None`` without them.
            num_free_instance: Number of learnable instances leading the set.

        Returns:
            The outputs of the learnable instances alone, and the outputs of the noised
            anchors (``None`` without them).
        """
        if dn_targets is None:
            return refined, None
        # The cached groups carry the targets they were last trained on.
        temporal = refined.temp_dn_targets
        latest = dn_targets if temporal is None else temporal
        denoising_outputs = Sparse4DDenoisingOutputs(
            prediction=refined.prediction[:, :, num_free_instance:],
            classification=refined.classification[:, :, num_free_instance:],
            box_target=dn_targets.box_target,
            cls_target=dn_targets.cls_target,
            valid_mask=dn_targets.valid_mask,
            temp_box_target=None if temporal is None else temporal.box_target,
            temp_cls_target=None if temporal is None else temporal.cls_target,
            temp_valid_mask=None if temporal is None else temporal.valid_mask,
            id_target=latest.id_target,
        )
        self.target.cache_dn(
            refined.instance_feature[:, num_free_instance:],
            refined.anchor[:, num_free_instance:],
            latest.cls_target,
            latest.valid_mask,
            latest.id_target,
        )
        learnable = RefinedLayers(
            prediction=refined.prediction[:, :, :num_free_instance],
            classification=refined.classification[:, :, :num_free_instance],
            quality=None if refined.quality is None else refined.quality[:, :, :num_free_instance],
            instance_feature=refined.instance_feature[:, :num_free_instance],
            anchor=refined.anchor[:, :num_free_instance],
            cls=refined.cls[:, :num_free_instance],
            temp_dn_targets=None,
        )
        return learnable, denoising_outputs

    def _refine(
        self,
        instance_feature: Float[torch.Tensor, "batch_size num_instances embed_dims"],
        anchor: Float[torch.Tensor, "batch_size num_instances anchor_dims"],
        temp_instance_feature: Float[torch.Tensor, "batch_size num_temp embed_dims"] | None,
        temp_anchor: Float[torch.Tensor, "batch_size num_temp anchor_dims"] | None,
        feature_maps: FlattenedFeatureMaps
        | Sequence[Float[torch.Tensor, "batch_size num_cams embed_dims height width"]],
        projection_mat: Float[torch.Tensor, "batch_size num_cams 4 4"],
        image_wh: Float[torch.Tensor, "batch_size num_cams 2"],
        time_interval: Float[torch.Tensor, " batch_size"],
        attn_mask: Bool[torch.Tensor, "num_instances num_instances"] | None,
        dn_targets: DenoisingTargets | None,
    ) -> RefinedLayers:
        """Run the decoder layers over the instances of a frame.

        After the single-frame layers the instance bank merges the temporal instances in,
        and the cached denoising groups, when any, replace the leading noised groups.

        Args:
            instance_feature: Features of the instances, the noised ones last.
            anchor: Their anchors.
            temp_instance_feature: Features of the temporal instances, ``None`` on the first
                frame of a batch.
            temp_anchor: Their anchors in the current ego frame.
            feature_maps: Image feature pyramid of every camera.
            projection_mat: Transform from the ego frame to the pixels of every camera.
            image_wh: Width and height of every camera image.
            time_interval: Time since the previous frame of every lane.
            attn_mask: True where an instance must not attend to another.
            dn_targets: Targets of the noised anchors, ``None`` without them.

        Returns:
            The outputs of every layer and the state after the last one.
        """
        anchor_embed = self.anchor_encoder(anchor)
        temp_anchor_embed = None if temp_anchor is None else self.anchor_encoder(temp_anchor)
        temp_dn_targets = None
        predictions: list[torch.Tensor] = []
        classifications: list[torch.Tensor] = []
        qualities: list[torch.Tensor | None] = []
        for layer_index, layer in enumerate(self.layers):
            instance_feature, anchor, cls, quality = layer(
                instance_feature,
                anchor,
                anchor_embed,
                feature_maps,
                projection_mat,
                image_wh,
                time_interval,
                temp_instance_feature=temp_instance_feature,
                temp_anchor_embed=temp_anchor_embed,
                attn_mask=attn_mask,
            )
            predictions.append(anchor)
            classifications.append(cls)
            qualities.append(quality)
            if layer_index + 1 == self.num_single_frame_decoder:
                instance_feature, anchor = self.instance_bank.update(instance_feature, anchor, cls)
                if (
                    dn_targets is not None
                    and self.target.num_temp_dn_groups > 0
                    and dn_targets.id_target is not None
                ):
                    if self.instance_bank.mask is None:
                        # The bank restarted, so the cached groups have no frame to follow.
                        self.target.dn_cache = None
                    elif self.target.dn_cache is not None:
                        updated = self.target.update_dn(
                            instance_feature,
                            anchor,
                            dn_targets.box_target,
                            dn_targets.cls_target,
                            dn_targets.valid_mask,
                            dn_targets.id_target,
                            self.instance_bank.num_anchor,
                            self.instance_bank.mask,
                        )
                        instance_feature = updated.instance_feature
                        anchor = updated.anchor
                        temp_dn_targets = DenoisingTargets(
                            box_target=updated.box_target,
                            cls_target=updated.cls_target,
                            valid_mask=updated.valid_mask,
                            id_target=updated.id_target,
                        )
            if layer_index + 1 < self.num_decoder:
                anchor_embed = self.anchor_encoder(anchor)
                if (
                    layer_index + 1 >= self.num_single_frame_decoder
                    and temp_anchor_embed is not None
                ):
                    temp_anchor_embed = anchor_embed[:, : self.instance_bank.num_temp_instances]
        return RefinedLayers(
            prediction=torch.stack(predictions),
            classification=torch.stack(classifications),
            quality=None if qualities[0] is None else torch.stack(qualities),
            instance_feature=instance_feature,
            anchor=anchor,
            cls=cls,
            temp_dn_targets=temp_dn_targets,
        )

    def _one_hot(self, cls_target: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        """One-hot class targets with a zero row for every background anchor."""
        foreground = (cls_target >= 0) & (cls_target < self.num_classes)
        one_hot = like.new_zeros((cls_target.shape[0], self.num_classes))
        one_hot[foreground, cls_target[foreground]] = 1.0
        return one_hot

    def _layer_losses(
        self,
        layer_index: int,
        cls: torch.Tensor,
        reg: torch.Tensor,
        quality: torch.Tensor | None,
        gt_boxes: Sequence[torch.Tensor],
        gt_labels: Sequence[torch.Tensor],
        label_weights: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Losses of the learnable instances of one decoder layer."""
        num_fields = len(self.reg_weights)
        reg = reg[..., :num_fields]
        matched = self.target.sample(cls, reg, gt_labels, gt_boxes)
        reg_target = matched.box_target[..., :num_fields]
        mask = matched.cls_target < self.num_classes
        # One collective per decoder layer, which every rank runs, so the ranks stay in step.
        num_pos = reduce_mean_count(mask.sum()).item()
        if self.cls_threshold_to_reg > 0:
            mask = mask & (cls.max(dim=-1).values.sigmoid() > self.cls_threshold_to_reg)

        cls_flat = cls.flatten(end_dim=1)
        cls_target = matched.cls_target.flatten()
        weights = label_weights.expand(cls.shape[0], cls.shape[1], 1).reshape(-1, 1)
        loss_cls = (
            self.loss_cls_weight
            * self.loss_cls(cls_flat, self._one_hot(cls_target, cls_flat), weights=weights)
            / num_pos
        )

        mask = mask.flatten()
        reg_weights = matched.reg_weights[..., :num_fields] * reg.new_tensor(self.reg_weights)
        losses = self.loss_reg(
            reg.flatten(end_dim=1)[mask],
            reg_target.flatten(end_dim=1)[mask],
            weight=reg_weights.flatten(end_dim=1)[mask],
            avg_factor=num_pos,
            quality=None if quality is None else quality.flatten(end_dim=1)[mask],
            cls_target=cls_target[mask],
        )
        prefix = "" if layer_index == self.num_decoder - 1 else f"d{layer_index}."
        output = {f"{prefix}loss_cls": loss_cls, f"{prefix}loss_box": losses.loss_box}
        if losses.loss_centerness is not None:
            output[f"{prefix}loss_cns"] = losses.loss_centerness
        if losses.loss_yawness is not None:
            output[f"{prefix}loss_yns"] = losses.loss_yawness
        return output

    def _denoising_layer_losses(
        self,
        layer_index: int,
        cls: torch.Tensor,
        reg: torch.Tensor,
        box_target: torch.Tensor,
        cls_target: torch.Tensor,
        valid_mask: torch.Tensor,
        avg_factor: float,
    ) -> dict[str, torch.Tensor]:
        """Losses of the noised anchors of one decoder layer."""
        num_fields = len(self.reg_weights)
        valid = valid_mask.flatten()
        cls_target = cls_target.flatten()[valid]
        reg_target = box_target.flatten(end_dim=1)[valid][..., :num_fields]
        positive = cls_target >= 0
        reg_target = reg_target[positive]

        cls_flat = cls.flatten(end_dim=1)[valid]
        loss_cls = (
            self.loss_cls_weight
            * self.loss_cls(cls_flat, self._one_hot(cls_target, cls_flat))
            / avg_factor
        )
        reg_flat = reg.flatten(end_dim=1)[valid][positive][..., :num_fields]
        losses = self.loss_reg(
            reg_flat,
            reg_target,
            weight=reg.new_tensor(self.reg_weights).expand_as(reg_flat),
            avg_factor=avg_factor,
        )
        prefix = "" if layer_index == self.num_decoder - 1 else f"d{layer_index}."
        return {f"{prefix}dn_loss_cls": loss_cls, f"{prefix}dn_loss_box": losses.loss_box}

    def loss(
        self,
        outputs: Sparse4DHeadOutputs,
        gt_boxes: Sequence[Float[torch.Tensor, "num_boxes num_fields"]],
        gt_labels: Sequence[Int64[torch.Tensor, " num_boxes"]],
        padding_mask: Bool[torch.Tensor, " batch_size"],
    ) -> dict[str, torch.Tensor]:
        """Compute the losses of every decoder layer.

        The classification and the box losses of a layer are normalized by its positive
        count averaged over the DDP ranks; the denoising losses by the count of valid noised
        anchors, averaged likewise once per step whenever denoising is on, so a rank without
        boxes joins the collective. ``padding_mask`` flags the padding samples of a streaming
        batch: their anchors get a zero classification weight, and they come without boxes,
        so they give no positive and no valid noised anchor either.

        Args:
            outputs: The head outputs of the frame.
            gt_boxes: Metric boxes of every sample, see ``Box3DFieldIndex``.
            gt_labels: Their classes.
            padding_mask: True for the padding samples.

        Returns:
            The losses by name, the final layer's without a prefix and the others under
            ``d{layer}.``, with their sum under ``loss``.
        """
        label_weights = (~padding_mask).to(outputs.classification.dtype).view(-1, 1, 1)
        loss_dict: dict[str, torch.Tensor] = {}
        for layer_index in range(self.num_decoder):
            loss_dict.update(
                self._layer_losses(
                    layer_index,
                    outputs.classification[layer_index],
                    outputs.prediction[layer_index],
                    None if outputs.quality is None else outputs.quality[layer_index],
                    gt_boxes,
                    gt_labels,
                    label_weights,
                )
            )

        denoising = outputs.denoising
        counts = [0.0, 0.0]
        if denoising is not None:
            counts[0] = float(denoising.valid_mask.sum().item())
            if denoising.temp_valid_mask is not None:
                counts[1] = float(denoising.temp_valid_mask.sum().item())
        dn_counts = outputs.classification.new_tensor(counts)
        if self.training and self.target.num_dn_groups > 0:
            # Gated on flags every rank shares, not on this rank's ground truth: a rank
            # without boxes joins the collective with counts of 0.
            dn_counts = reduce_mean_count(dn_counts)
        else:
            dn_counts = dn_counts.clamp(min=1.0)
        dn_avg_factor, temp_dn_avg_factor = dn_counts.tolist()
        if self.training and self.target.num_dn_groups > 0:
            if denoising is None:
                # Keep the loss dictionary uniform across ranks: every logged key is a
                # distributed collective.
                for layer_index in range(self.num_decoder):
                    prefix = "" if layer_index == self.num_decoder - 1 else f"d{layer_index}."
                    loss_dict[f"{prefix}dn_loss_cls"] = outputs.classification.new_tensor(0.0)
                    loss_dict[f"{prefix}dn_loss_box"] = outputs.classification.new_tensor(0.0)
        if denoising is not None:
            for layer_index in range(self.num_decoder):
                temporal = (
                    denoising.temp_valid_mask is not None
                    and layer_index >= self.num_single_frame_decoder
                )
                loss_dict.update(
                    self._denoising_layer_losses(
                        layer_index,
                        denoising.classification[layer_index],
                        denoising.prediction[layer_index],
                        denoising.temp_box_target if temporal else denoising.box_target,
                        denoising.temp_cls_target if temporal else denoising.cls_target,
                        denoising.temp_valid_mask if temporal else denoising.valid_mask,
                        temp_dn_avg_factor if temporal else dn_avg_factor,
                    )
                )

        loss_dict["loss"] = sum(loss_dict.values())
        return loss_dict

    def predict(self, outputs: Sparse4DHeadOutputs) -> list[Detection3DSamplePredictions]:
        """Decode the boxes of the final decoder layer.

        Args:
            outputs: The head outputs of the frame.

        Returns:
            One prediction per sample, with tracking ids when the head ran in eval mode.
        """
        return self.decoder.decode(
            outputs.classification[-1],
            outputs.prediction[-1],
            quality=None if outputs.quality is None else outputs.quality[-1],
            instance_ids=outputs.instance_id,
        )
