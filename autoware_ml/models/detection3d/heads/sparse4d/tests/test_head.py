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

"""Unit tests for the Sparse4D head."""

from __future__ import annotations

import math
from typing import NamedTuple
import unittest

import torch
import torch.nn as nn

from autoware_ml.dataclasses.models.detection3d.head_outputs import Sparse4DHeadOutputs
from autoware_ml.dataclasses.models.detection3d.predictions import Detection3DSamplePredictions
from autoware_ml.models.detection3d.heads.sparse4d.blocks import (
    AsymmetricFFN,
    DeformableFeatureAggregation,
)
from autoware_ml.models.detection3d.heads.sparse4d.box3d import (
    ANCHOR_DIMS,
    AnchorFieldIndex,
    SparseBox3DEncoder,
    SparseBox3DKeyPointsGenerator,
    SparseBox3DRefinementModule,
)
from autoware_ml.models.detection3d.heads.sparse4d.decoder import SparseBox3DDecoder
from autoware_ml.models.detection3d.heads.sparse4d.head import Sparse4DHead, clone_module
from autoware_ml.models.detection3d.heads.sparse4d.instance_bank import InstanceBank
from autoware_ml.models.detection3d.heads.sparse4d.loss import SparseBox3DLoss
from autoware_ml.models.detection3d.heads.sparse4d.target import SparseBox3DTarget
from autoware_ml.types.geometry import Box3DFieldIndex

EMBED_DIMS = 16
# Width of the decoupled attention and of the aggregation output with the "cat" residual.
ATTN_DIMS = 32
NUM_CLASSES = 2
NUM_ANCHOR = 4
NUM_TEMP = 2
NUM_CAMS = 2
NUM_DECODER = 3
REG_WEIGHTS = [2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
IMAGE_WH = (24.0, 16.0)
LEVEL_SHAPES = ((4, 6), (2, 3))


class Frame(NamedTuple):
    """Inputs of one frame."""

    feature_maps: list[torch.Tensor]
    projection_mat: torch.Tensor
    image_wh: torch.Tensor
    timestamp: torch.Tensor
    ego2global: torch.Tensor


def _anchors() -> torch.Tensor:
    """Four unit anchors 5 m in front of the cameras, spread along x."""
    anchors = torch.zeros(NUM_ANCHOR, ANCHOR_DIMS)
    anchors[:, AnchorFieldIndex.X] = torch.tensor([-1.0, -0.3, 0.3, 1.0])
    anchors[:, AnchorFieldIndex.Z] = 5.0
    anchors[:, AnchorFieldIndex.COS_YAW] = 1.0
    return anchors


def _metric_box(x: float) -> torch.Tensor:
    """A unit box at (x, 0, 5) in the Box3DFieldIndex layout."""
    box = torch.zeros(len(Box3DFieldIndex))
    box[Box3DFieldIndex.X] = x
    box[Box3DFieldIndex.Z] = 5.0
    box[Box3DFieldIndex.LENGTH] = 1.0
    box[Box3DFieldIndex.WIDTH] = 1.0
    box[Box3DFieldIndex.HEIGHT] = 1.0
    return box


def _build_head(temporal: bool = True, num_single_frame_decoder: int = 1) -> Sparse4DHead:
    """A tiny head of three decoder layers on two cameras with two feature levels."""
    torch.manual_seed(0)
    kps_generator = SparseBox3DKeyPointsGenerator(
        embed_dims=EMBED_DIMS, num_learnable_pts=1, fix_scale=[[0.0, 0.0, 0.0]]
    )
    return Sparse4DHead(
        instance_bank=InstanceBank(
            num_anchor=NUM_ANCHOR,
            embed_dims=EMBED_DIMS,
            anchor=_anchors(),
            num_temp_instances=NUM_TEMP if temporal else 0,
            feat_grad=False,
        ),
        anchor_encoder=SparseBox3DEncoder(
            embed_dims=[8, 2, 2, 4], mode="cat", output_fc=False, in_loops=1, out_loops=1
        ),
        ffn=AsymmetricFFN(
            in_channels=ATTN_DIMS,
            pre_norm=True,
            embed_dims=EMBED_DIMS,
            feedforward_channels=ATTN_DIMS,
        ),
        deformable_model=DeformableFeatureAggregation(
            kps_generator,
            embed_dims=EMBED_DIMS,
            num_groups=2,
            num_levels=len(LEVEL_SHAPES),
            num_cams=NUM_CAMS,
            use_camera_embed=True,
            residual_mode="cat",
        ),
        refine_layer=SparseBox3DRefinementModule(
            embed_dims=EMBED_DIMS,
            num_cls=NUM_CLASSES,
            refine_yaw=True,
            with_quality_estimation=True,
        ),
        target=SparseBox3DTarget(
            num_dn_groups=2,
            num_temp_dn_groups=1 if temporal else 0,
            dn_noise_scale=0.1,
            max_dn_gt=4,
            add_neg_dn=True,
        ),
        loss_reg=SparseBox3DLoss(cls_allow_reverse=[1]),
        decoder=SparseBox3DDecoder(num_output=4),
        num_classes=NUM_CLASSES,
        num_heads=2,
        num_decoder=NUM_DECODER,
        num_single_frame_decoder=num_single_frame_decoder,
        temporal=temporal,
        attn_drop=0.0,
        reg_weights=REG_WEIGHTS,
    )


def _frame(batch_size: int, timestamp: float, seed: int, x_shift: float = 0.0) -> Frame:
    """Feature maps and camera geometry of one frame."""
    generator = torch.Generator().manual_seed(seed)
    feature_maps = [
        torch.randn(batch_size, NUM_CAMS, EMBED_DIMS, height, width, generator=generator)
        for height, width in LEVEL_SHAPES
    ]
    projection = torch.tensor(
        [
            [10.0, 0.0, 12.0, 0.0],
            [0.0, 10.0, 8.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    ego2global = torch.eye(4)
    ego2global[0, 3] = x_shift
    return Frame(
        feature_maps=feature_maps,
        projection_mat=projection.expand(batch_size, NUM_CAMS, 4, 4).clone(),
        image_wh=torch.tensor(IMAGE_WH).expand(batch_size, NUM_CAMS, 2).clone(),
        timestamp=torch.full((batch_size,), timestamp, dtype=torch.float64),
        ego2global=ego2global.expand(batch_size, 4, 4).clone(),
    )


def _run(
    head: Sparse4DHead,
    frame: Frame,
    prev_exists: torch.Tensor,
    padding_mask: torch.Tensor | None = None,
    boxes: list[torch.Tensor] | None = None,
    labels: list[torch.Tensor] | None = None,
    ids: list[torch.Tensor] | None = None,
) -> Sparse4DHeadOutputs:
    """Run the head on a frame."""
    return head(
        frame.feature_maps,
        frame.projection_mat,
        frame.image_wh,
        frame.timestamp,
        frame.ego2global,
        prev_exists,
        padding_mask,
        boxes,
        labels,
        ids,
    )


def _ground_truth() -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """Two boxes in the first sample and one in the second, with instance ids."""
    boxes = [
        torch.stack([_metric_box(-1.0), _metric_box(1.0)]),
        torch.stack([_metric_box(0.3)]),
    ]
    labels = [torch.tensor([0, 1]), torch.tensor([1])]
    ids = [torch.tensor([7, 8]), torch.tensor([9])]
    return boxes, labels, ids


def _no_ground_truth() -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """No boxes in either sample."""
    boxes = [torch.zeros(0, len(Box3DFieldIndex)), torch.zeros(0, len(Box3DFieldIndex))]
    labels = [torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)]
    return boxes, labels


class TestSparse4DHeadForward(unittest.TestCase):
    """The forward of the head."""

    def setUp(self) -> None:
        """Set up the tiny head and a two-sample frame."""
        self.head = _build_head()
        self.frame = _frame(batch_size=2, timestamp=10.0, seed=1)
        self.next_frame = _frame(batch_size=2, timestamp=10.5, seed=2)
        self.starts = torch.zeros(2)
        self.continues = torch.ones(2)

    def test_training_forward_refines_and_appends_the_noised_anchors(self) -> None:
        """
        Input: the frame with two and one boxes in training.
        Expected: three layers of four learnable anchors with quality, and the noised
            anchors: two groups of two max boxes with negatives make eight, separated from
            the learnable ones; no ids in training.
        Check: the output shapes, the None ids and the valid mask of the padding.
        """
        boxes, labels, ids = _ground_truth()
        self.head.train()

        outputs = _run(self.head, self.frame, self.starts, boxes=boxes, labels=labels, ids=ids)

        self.assertEqual(outputs.prediction.shape, (NUM_DECODER, 2, NUM_ANCHOR, ANCHOR_DIMS))
        self.assertEqual(outputs.classification.shape, (NUM_DECODER, 2, NUM_ANCHOR, NUM_CLASSES))
        self.assertEqual(outputs.quality.shape, (NUM_DECODER, 2, NUM_ANCHOR, 2))
        self.assertIsNone(outputs.instance_id)
        self.assertEqual(outputs.denoising.prediction.shape, (NUM_DECODER, 2, 8, ANCHOR_DIMS))
        self.assertEqual(outputs.denoising.box_target.shape, (2, 8, ANCHOR_DIMS))
        self.assertTrue(bool(outputs.denoising.valid_mask[0].all()))
        self.assertEqual(int(outputs.denoising.valid_mask[1].sum()), 4)
        self.assertIsNone(outputs.denoising.temp_valid_mask)
        self.assertIsNotNone(self.head.target.dn_cache)

    def test_eval_forward_gives_ids_and_no_noised_anchors(self) -> None:
        """
        Input: the frame in eval mode with boxes given.
        Expected: no noised anchors even with boxes, and an id for every anchor.
        Check: None denoising, the id shape and ids from 0 to 7.
        """
        boxes, labels, _ = _ground_truth()
        self.head.eval()

        outputs = _run(self.head, self.frame, self.starts, boxes=boxes, labels=labels)

        self.assertIsNone(outputs.denoising)
        self.assertEqual(outputs.instance_id.shape, (2, NUM_ANCHOR))
        torch.testing.assert_close(outputs.instance_id.flatten(), torch.arange(8))

    def test_second_frame_uses_the_temporal_instances(self) -> None:
        """
        Input: a frame, then a frame 0.5 s later whose lanes continue.
        Expected: the bank keeps both lanes, and the two temporal slots of every lane carry
            ids of the first frame while the other two slots get the new ids 8 to 11.
        Check: the bank mask and the ids of the second frame.
        """
        self.head.eval()
        first = _run(self.head, self.frame, self.starts)

        second = _run(self.head, self.next_frame, self.continues)

        torch.testing.assert_close(self.head.instance_bank.mask, torch.tensor([True, True]))
        for lane in range(2):
            for slot in range(NUM_TEMP):
                self.assertIn(
                    second.instance_id[lane, slot].item(), first.instance_id[lane].tolist()
                )
        self.assertEqual(int(second.instance_id.max()), 11)

    def test_padding_and_ended_streams_reset_their_lane(self) -> None:
        """
        Input: a frame, then a frame whose first lane continues and whose second lane is
            padding, then one whose first lane ended.
        Expected: the padding lane and the ended lane are reset in the bank.
        Check: the bank mask of both frames.
        """
        self.head.eval()
        _run(self.head, self.frame, self.starts)

        _run(self.head, self.next_frame, self.continues, padding_mask=torch.tensor([False, True]))
        torch.testing.assert_close(self.head.instance_bank.mask, torch.tensor([True, False]))

        third = _frame(batch_size=2, timestamp=11.0, seed=3)
        _run(self.head, third, torch.tensor([0.0, 1.0]))
        torch.testing.assert_close(self.head.instance_bank.mask, torch.tensor([False, True]))

    def test_temporal_denoising_groups_come_back(self) -> None:
        """
        Input: two training frames with instance ids whose lanes continue.
        Expected: the second frame has temporal denoising targets.
        Check: the temporal targets are set and keep the noised anchor count.
        """
        boxes, labels, ids = _ground_truth()
        self.head.train()
        _run(self.head, self.frame, self.starts, boxes=boxes, labels=labels, ids=ids)

        outputs = _run(
            self.head, self.next_frame, self.continues, boxes=boxes, labels=labels, ids=ids
        )

        self.assertIsNotNone(outputs.denoising.temp_valid_mask)
        self.assertEqual(outputs.denoising.temp_cls_target.shape, (2, 8))
        self.assertEqual(outputs.denoising.id_target.shape, (2, 8))

    def test_batch_size_change_drops_the_caches(self) -> None:
        """
        Input: a training frame of two samples, then one of one sample.
        Expected: the bank and the denoising cache restart for the new batch size.
        Check: the forward succeeds and the caches hold one sample.
        """
        boxes, labels, ids = _ground_truth()
        self.head.train()
        _run(self.head, self.frame, self.starts, boxes=boxes, labels=labels, ids=ids)

        single = _frame(batch_size=1, timestamp=10.5, seed=2)
        _run(self.head, single, torch.ones(1), boxes=boxes[:1], labels=labels[:1], ids=ids[:1])

        self.assertEqual(self.head.instance_bank.cached_anchor.shape[0], 1)
        self.assertEqual(self.head.target.dn_cache.anchor.shape[0], 1)

    def test_reset_memory_forgets_everything(self) -> None:
        """
        Input: a training frame, then reset_memory.
        Expected: the bank, the denoising cache and the continuity are empty.
        Check: the three states.
        """
        boxes, labels, ids = _ground_truth()
        self.head.train()
        _run(self.head, self.frame, self.starts, boxes=boxes, labels=labels, ids=ids)
        self.head.stream_continuity(["a", "b"], self.frame.timestamp)

        self.head.reset_memory()

        self.assertIsNone(self.head.instance_bank.cached_anchor)
        self.assertIsNone(self.head.target.dn_cache)
        self.assertIsNone(self.head.stream_continuity.scene_tokens)

    def test_single_frame_head_without_temporal_layers(self) -> None:
        """
        Input: a head without temporal attention or temporal instances, run on two frames.
        Expected: the temporal layers are absent and the frames run independently.
        Check: None temporal layers, no cache and the output shape.
        """
        head = _build_head(temporal=False).eval()

        _run(head, self.frame, self.starts)
        outputs = _run(head, self.next_frame, self.continues)

        self.assertTrue(all(layer.temp_gnn is None for layer in head.layers))
        self.assertIsNone(head.instance_bank.cached_anchor)
        self.assertEqual(outputs.prediction.shape, (NUM_DECODER, 2, NUM_ANCHOR, ANCHOR_DIMS))

    def test_init_weights_zeroes_the_aggregation_weights(self) -> None:
        """
        Input: the head after construction.
        Expected: every deformable aggregation starts with zero sampling weights and every
            refinement with the focal class bias, as the modules' own init_weights set.
        Check: the weights_fc weights and the class bias of every layer.
        """
        for layer in self.head.layers:
            weights = layer.deformable.weights_fc.weight
            torch.testing.assert_close(weights, torch.zeros_like(weights))
            bias = layer.refine.cls_layers[-1].bias
            torch.testing.assert_close(bias, torch.full_like(bias, -math.log(99.0)))

    def test_clone_module_draws_fresh_parameters(self) -> None:
        """
        Input: a linear prototype.
        Expected: the clone is another module with other parameters, and the prototype is
            left alone.
        Check: distinct objects, distinct weights and the unchanged prototype weight.
        """
        torch.manual_seed(0)
        prototype = nn.Linear(4, 4)
        weight = prototype.weight.detach().clone()

        clone = clone_module(prototype)

        self.assertIsNot(clone, prototype)
        self.assertFalse(torch.equal(clone.weight, prototype.weight))
        torch.testing.assert_close(prototype.weight, weight)

    def test_layers_own_clones_of_the_prototypes(self) -> None:
        """
        Input: the head built from one prototype per block.
        Expected: every layer holds its own blocks, distinct from the other layers'.
        Check: the ffn, aggregation and refinement objects differ between layers.
        """
        first, second = self.head.layers[0], self.head.layers[1]

        self.assertIsNot(first.ffn, second.ffn)
        self.assertIsNot(first.deformable, second.deformable)
        self.assertIsNot(first.refine, second.refine)

    def test_decoupled_attentions_share_the_value_projections(self) -> None:
        """
        Input: the head with decoupled attention.
        Expected: every graph attention attends at twice the width and shares one pair of
            value projections.
        Check: the attention width and the identity of the projections across layers.
        """
        attentions = [layer.gnn for layer in self.head.layers if layer.gnn is not None] + [
            layer.temp_gnn for layer in self.head.layers if layer.temp_gnn is not None
        ]

        self.assertEqual(len(attentions), 4)
        for attention in attentions:
            self.assertEqual(attention.attn.embed_dim, ATTN_DIMS)
            self.assertIs(attention.value_proj, attentions[0].value_proj)
            self.assertIs(attention.output_proj, attentions[0].output_proj)

    def test_layers_follow_the_recipe(self) -> None:
        """
        Input: the head with three layers, one single-frame.
        Expected: the first layer has no attention, the second and third have both the
            self and the temporal attention.
        Check: the attention blocks of every layer.
        """
        layers = list(self.head.layers)

        self.assertEqual(len(layers), NUM_DECODER)
        self.assertIsNone(layers[0].gnn)
        self.assertIsNone(layers[0].temp_gnn)
        self.assertIsNone(layers[0].norm_attn)
        for layer in layers[1:]:
            self.assertIsNotNone(layer.gnn)
            self.assertIsNotNone(layer.temp_gnn)

    def test_rejects_an_empty_or_inconsistent_stack(self) -> None:
        """
        Input: no decoder layer, and more single-frame layers than layers.
        Expected: both are refused.
        Check: ValueError in both cases.
        """
        with self.assertRaisesRegex(ValueError, "num_decoder"):
            _build_head_with_layers(num_decoder=0)
        with self.assertRaisesRegex(ValueError, "num_single_frame_decoder"):
            _build_head(num_single_frame_decoder=4)


def _build_head_with_layers(num_decoder: int) -> Sparse4DHead:
    """A tiny head with the given number of layers."""
    head = _build_head()
    return Sparse4DHead(
        instance_bank=head.instance_bank,
        anchor_encoder=head.anchor_encoder,
        ffn=head.layers[0].ffn,
        deformable_model=head.layers[0].deformable,
        refine_layer=head.layers[0].refine,
        target=head.target,
        loss_reg=head.loss_reg,
        decoder=head.decoder,
        num_classes=NUM_CLASSES,
        num_heads=2,
        num_decoder=num_decoder,
        num_single_frame_decoder=0,
    )


class TestSparse4DHeadLoss(unittest.TestCase):
    """The losses and the predictions of the head."""

    def setUp(self) -> None:
        """Set up the tiny head in training on a two-sample frame."""
        self.head = _build_head().train()
        self.frame = _frame(batch_size=2, timestamp=10.0, seed=1)
        self.boxes, self.labels, self.ids = _ground_truth()
        self.padding_mask = torch.zeros(2, dtype=torch.bool)

    def _losses(self) -> dict[str, torch.Tensor]:
        outputs = _run(
            self.head,
            self.frame,
            torch.zeros(2),
            padding_mask=self.padding_mask,
            boxes=self.boxes,
            labels=self.labels,
            ids=self.ids,
        )
        return self.head.loss(outputs, self.boxes, self.labels, self.padding_mask)

    def test_loss_names_every_term_and_backpropagates(self) -> None:
        """
        Input: the training frame with boxes.
        Expected: the final layer's terms without prefix, the earlier ones under d0 and d1,
            the denoising terms, and their sum under loss; all finite and differentiable.
        Check: the key set, finite values and a gradient on the anchors.
        """
        losses = self._losses()

        expected = {"loss"}
        for prefix in ("", "d0.", "d1."):
            expected |= {
                f"{prefix}loss_cls",
                f"{prefix}loss_box",
                f"{prefix}loss_cns",
                f"{prefix}loss_yns",
                f"{prefix}dn_loss_cls",
                f"{prefix}dn_loss_box",
            }
        self.assertEqual(set(losses), expected)
        for name, value in losses.items():
            self.assertTrue(torch.isfinite(value).item(), name)
        losses["loss"].backward()
        self.assertIsNotNone(self.head.instance_bank.anchor.grad)

    def test_without_boxes_the_denoising_terms_are_zero(self) -> None:
        """
        Input: a training frame without boxes.
        Expected: no noised anchors, so the denoising terms are reported as zero while the
            classification loss stays and the box loss has no positive.
        Check: the denoising keys at zero, a positive classification loss and a zero box loss.
        """
        self.boxes, self.labels = _no_ground_truth()
        self.ids = None

        losses = self._losses()

        self.assertEqual(losses["dn_loss_cls"].item(), 0.0)
        self.assertEqual(losses["d0.dn_loss_box"].item(), 0.0)
        self.assertGreater(losses["loss_cls"].item(), 0.0)
        self.assertEqual(losses["loss_box"].item(), 0.0)

    def test_padding_samples_take_no_part(self) -> None:
        """
        Input: both samples flagged as padding, without boxes as the model gives them.
        Expected: zero classification weight everywhere, so every loss is zero.
        Check: the total loss equals zero.
        """
        self.boxes, self.labels = _no_ground_truth()
        self.ids = None
        self.padding_mask = torch.ones(2, dtype=torch.bool)

        losses = self._losses()

        self.assertEqual(losses["loss"].item(), 0.0)

    def test_cls_threshold_keeps_weak_anchors_out_of_the_box_loss(self) -> None:
        """
        Input: a classification threshold above one, which no anchor passes.
        Expected: the box, centerness and yawness losses vanish while the classification
            loss and the denoising box loss stay.
        Check: zero box losses, a positive classification loss and a positive denoising
            box loss.
        """
        self.head.cls_threshold_to_reg = 1.5

        losses = self._losses()

        self.assertEqual(losses["loss_box"].item(), 0.0)
        self.assertEqual(losses["loss_cns"].item(), 0.0)
        self.assertGreater(losses["loss_cls"].item(), 0.0)
        self.assertGreater(losses["dn_loss_box"].item(), 0.0)

    def test_predict_decodes_every_sample_with_ids(self) -> None:
        """
        Input: an eval forward of the frame.
        Expected: one prediction per sample with four boxes of nine parameters and ids.
        Check: the prediction type, count, box shape and ids.
        """
        self.head.eval()
        outputs = _run(self.head, self.frame, torch.zeros(2))

        predictions = self.head.predict(outputs)

        self.assertEqual(len(predictions), 2)
        for prediction in predictions:
            self.assertIsInstance(prediction, Detection3DSamplePredictions)
            self.assertEqual(prediction.bboxes_3d.shape, (NUM_ANCHOR, 9))
            self.assertEqual(prediction.instance_ids_3d.shape, (NUM_ANCHOR,))
