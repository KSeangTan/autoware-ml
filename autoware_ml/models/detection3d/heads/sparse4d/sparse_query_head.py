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

"""Decoder of instance queries shared by the Sparse4D heads.

The decoder owns the order of a block, the instance attention, the lidar block, the image block,
the FFN and the refinement, and the temporal export. The blocks themselves live with their
modality: :mod:`lidar_aggregation` and :class:`BEVCrossAttention` read the lidar features,
:mod:`image_aggregation` reads the image features.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from autoware_ml.models.detection3d.heads.sparse4d.image_aggregation import (
    ImageDeformableAggregation,
)
from autoware_ml.models.detection3d.heads.sparse4d.lidar_aggregation import (
    BEVDeformableAggregation,
    KeyPointsGenerator,
    LidarFeatures,
    SparseDeformableAggregation,
    SparseLevelGrid,
)
from autoware_ml.models.detection3d.heads.transfusions.exportable_multi_head_attention import (
    ExportableMultiheadAttention,
)
from autoware_ml.ops.deformable_aggregation.bev_deformable_aggregation import BEVLevel
from autoware_ml.ops.deformable_aggregation.image_deformable_aggregation import ImageFeatures
from autoware_ml.ops.deformable_aggregation.sparse_deformable_aggregation import SparseLevel

if TYPE_CHECKING:
    from autoware_ml.models.detection3d.heads.sparse4d.sparse4d_head import (
        BEVCrossAttention,
        DenoisingQueries,
    )

# Instance bank state at the temporal export graph boundary
TEMPORAL_STATE_INPUT_NAMES = (
    "temp_instance_feature",
    "temp_anchor",
    "temp_confidence",
    "temp_valid",
    "time_interval",
    "prev2current",
)
TEMPORAL_STATE_OUTPUT_NAMES = (
    "temp_instance_feature_out",
    "temp_anchor_out",
    "temp_confidence_out",
)
# Query every state slot was selected from, the runtime carries its track ids into the slots
TEMPORAL_QUERY_INDEX_OUTPUT_NAME = "temp_query_index_out"


class DecoupledInstanceAttention(nn.Module):
    """Instance attention with decoupled (concatenated) anchor embeddings.

    With key inputs it reads the temporal memory.
    """

    def __init__(self, embed_dims: int, num_heads: int, dropout: float) -> None:
        """Initialize the decoupled attention block.

        Args:
            embed_dims: Instance feature dimension.
            num_heads: Number of attention heads.
            dropout: Attention and residual dropout probability.
        """
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dims * 2, num_heads, dropout=dropout, batch_first=True
        )
        self.fc_value = nn.Linear(embed_dims, embed_dims * 2, bias=False)
        self.fc_out = nn.Linear(embed_dims * 2, embed_dims, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor_embed: torch.Tensor,
        key_feature: torch.Tensor | None = None,
        key_embed: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Attend with anchor position aware queries and keys.

        Args:
            instance_feature: Query instance features ``(B, N, C)``.
            anchor_embed: Query anchor embeddings ``(B, N, C)``.
            key_feature: Optional key and value instance features ``(B, K, C)``.
            key_embed: Optional key anchor embeddings ``(B, K, C)``.
            attn_mask: Optional boolean mask ``(N, K)`` where ``True`` blocks
                attention, the denoising isolation mask of training steps.
            key_padding_mask: Optional boolean mask ``(B, K)`` where ``True`` hides a key from
                every query of its row.

        Returns:
            Updated instance features ``(B, N, C)``.
        """
        tokens = torch.cat([instance_feature, anchor_embed], dim=-1)
        if key_feature is None:
            key_tokens = tokens
            value = self.fc_value(instance_feature)
        else:
            key_tokens = torch.cat([key_feature, key_embed], dim=-1)
            value = self.fc_value(key_feature)
        # Without the averaged weights the attention runs on the fused scaled dot product
        # kernels, flash attention for half precision inputs without a mask.
        if attn_mask is None and key_padding_mask is None:
            attended = self.attn(tokens, key_tokens, value, need_weights=False)[0]
        else:
            attended = self.attn(
                tokens,
                key_tokens,
                value,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )[0]
        return self.fc_out(tokens + self.dropout(attended))


@dataclass
class _DecoderOutputs:
    """Per layer outputs of the decoder and its final state.

    Attributes:
        classification: Class logits of every layer ``(B, N + D, num_classes)``.
        prediction: Anchor states of every layer ``(B, N + D, 10)``.
        quality: Quality logits of every layer ``(B, N + D, 2)``.
        instance_feature: Final instance features of the regular queries ``(B, N, C)``.
        anchor: Final anchors of the regular queries ``(B, N, 10)``.
        cls_logits: Final class logits of the regular queries ``(B, N, num_classes)``.
    """

    classification: list[torch.Tensor]
    prediction: list[torch.Tensor]
    quality: list[torch.Tensor]
    instance_feature: torch.Tensor
    anchor: torch.Tensor
    cls_logits: torch.Tensor


class SparseQueryHead(nn.Module):
    """Decoder of instance queries over sparse encoder stages, as in Sparse4D.

    Every block attends to the temporal memory and to the other queries, samples the stages at
    keypoints of its anchors, runs an FFN and refines the anchors. A subclass sets the anchor
    encoding, the keypoints and the refinement of its anchor kind.

    A block reads the two modalities with two blocks of its own, the lidar block
    (``deformable``) over the lidar features and the image block (``image_aggregation``) over
    the image features, in that order. They share the instance features and nothing else: no
    weights, no keypoints and no inputs, and a head builds either one without the other.
    """

    def _build_decoder(
        self,
        *,
        embed_dims: int,
        num_decoder: int,
        num_heads: int,
        num_groups: int,
        feedforward_channels: int,
        dropout: float,
        attn_drop: float,
        temporal: bool,
        point_cloud_range: Sequence[float],
        grid_size: float,
        feature_strides: Sequence[int],
        anchor_encoder: nn.Module,
        kps_generator: Callable[[], KeyPointsGenerator],
        refine: Callable[[], nn.Module],
        bev: nn.Module | None = None,
        bev_cell_size: float | None = None,
        bev_aggregation: Callable[[], BEVDeformableAggregation | None] = lambda: None,
        sample_sparse: bool = True,
        aggregation: Callable[..., SparseDeformableAggregation] = SparseDeformableAggregation,
        bev_attention: Callable[[], BEVCrossAttention] | None = None,
        bev_shared_block: nn.Module | None = None,
        read_lidar: bool = True,
        image_aggregation: Callable[[], ImageDeformableAggregation] | None = None,
        image_strides: Sequence[int] = (),
    ) -> None:
        """Create the decoder blocks.

        Args:
            embed_dims: Instance feature dimension.
            num_decoder: Number of decoder blocks.
            num_heads: Attention heads of the instance attention.
            num_groups: Feature groups of the deformable aggregation.
            feedforward_channels: Hidden width of the FFN blocks.
            dropout: Dropout probability in the attention and FFN blocks.
            attn_drop: Keypoint dropout of the aggregation, or the attention dropout of the BEV
                attention, in training.
            temporal: Whether the blocks attend to a temporal memory.
            point_cloud_range: Metric range of the voxel grid.
            grid_size: Metric size of the input voxel grid.
            feature_strides: Total pooling strides of the sampled stages.
            anchor_encoder: Embedding of the anchors.
            kps_generator: Builds the keypoint generator of one block.
            refine: Builds the refinement of one block.
            bev: Builds the BEV levels of a frame from the stages, None without one.
            bev_cell_size: Metric cell size of the BEV map the head gets as its lidar features,
                which it places on the grid of ``point_cloud_range`` as its one BEV level. None
                without one.
            bev_aggregation: Builds the BEV sampling of one block, None without BEV maps.
            sample_sparse: Whether the blocks sample the sparse voxels.
            aggregation: Builds the deformable aggregation of one block from its keyword
                arguments.
            bev_attention: Builds the BEV cross attention of one block, which then takes the
                place of its deformable aggregation. None for the aggregation.
            bev_shared_block: Refines the BEV map the head gets as its lidar features once per
                frame, before the blocks read it. None reads the map as it is.
            read_lidar: Whether the blocks have a lidar block, a camera only head has none.
            image_aggregation: Builds the image block of one decoder block, None for a head
                without image features.
            image_strides: Strides of the image pyramid the image blocks read, finest first.

        Raises:
            ValueError: If the blocks read neither the lidar nor the image features.
        """
        if not read_lidar and image_aggregation is None:
            raise ValueError("The decoder reads the lidar features, the image features or both.")
        self.read_lidar = read_lidar
        self.image_strides = tuple(int(stride) for stride in image_strides)
        self.num_decoder = num_decoder
        self.bev = bev
        self.shared_block = bev_shared_block
        self.bev_cell_size = None if bev_cell_size is None else float(bev_cell_size)
        self.bev_grid_origin = (float(point_cloud_range[0]), float(point_cloud_range[1]))
        self.sample_sparse = sample_sparse
        self.anchor_encoder = anchor_encoder
        self.gnn = nn.ModuleList(
            [
                DecoupledInstanceAttention(embed_dims, num_heads, dropout)
                for _ in range(num_decoder - 1)
            ]
        )
        self.gnn_norm = nn.ModuleList([nn.LayerNorm(embed_dims) for _ in range(num_decoder - 1)])
        if temporal:
            self.temp_gnn = nn.ModuleList(
                [
                    DecoupledInstanceAttention(embed_dims, num_heads, dropout)
                    for _ in range(num_decoder - 1)
                ]
            )
            self.temp_gnn_norm = nn.ModuleList(
                [nn.LayerNorm(embed_dims) for _ in range(num_decoder - 1)]
            )
        self.level_grids = nn.ModuleList(
            [SparseLevelGrid(point_cloud_range, grid_size, stride) for stride in feature_strides]
        )
        # The lidar block of every decoder, the deformable aggregation or the BEV attention. It
        # keeps the names it had before the image block, so the lidar checkpoints still load.
        self.deformable = nn.ModuleList(
            [
                aggregation(
                    embed_dims=embed_dims,
                    num_groups=num_groups,
                    num_levels=len(feature_strides),
                    kps_generator=kps_generator(),
                    proj_drop=dropout,
                    attn_drop=attn_drop,
                    bev_aggregation=bev_aggregation(),
                    sample_sparse=sample_sparse,
                )
                if bev_attention is None
                else bev_attention()
                for _ in range(num_decoder if read_lidar else 0)
            ]
        )
        self.deformable_norm = nn.ModuleList(
            [nn.LayerNorm(embed_dims) for _ in range(len(self.deformable))]
        )
        # The image block of every decoder, after its lidar block
        self.image_aggregation = nn.ModuleList(
            [image_aggregation() for _ in range(0 if image_aggregation is None else num_decoder)]
        )
        self.image_norm = nn.ModuleList(
            [nn.LayerNorm(embed_dims) for _ in range(len(self.image_aggregation))]
        )
        self.ffn = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(embed_dims, feedforward_channels),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(feedforward_channels, embed_dims),
                    nn.Dropout(dropout),
                )
                for _ in range(num_decoder)
            ]
        )
        self.ffn_norm = nn.ModuleList([nn.LayerNorm(embed_dims) for _ in range(num_decoder)])
        self.refine = nn.ModuleList([refine() for _ in range(num_decoder)])

    def _refine(
        self,
        block_index: int,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
        time_interval: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Refine the anchors of one block and classify the queries.

        Args:
            block_index: Index of the decoder block.
            instance_feature: Instance features ``(B, N, C)``.
            anchor: Anchors ``(B, N, D)``.
            anchor_embed: Anchor embeddings ``(B, N, C)``.
            time_interval: Frame gaps ``(B,)`` in seconds.

        Returns:
            Refined anchors, class logits and quality logits, None for a head without quality.
        """
        raise NotImplementedError

    def _block_context(self, anchor: torch.Tensor, context: Any) -> Any:
        """Frame context the blocks read for the current anchors, the head context as given."""
        return context

    def levels(self, lidar_features: LidarFeatures | None) -> list[SparseLevel]:
        """Look up the voxels of every sampled stage once per frame, none for a BEV only head.

        Raises:
            ValueError: If a head sampling the sparse voxels gets a BEV map.
        """
        if not self.sample_sparse:
            return []
        if isinstance(lidar_features, torch.Tensor):
            raise ValueError("A head sampling the sparse voxels needs the sparse stages.")
        return [grid(*stage) for grid, stage in zip(self.level_grids, lidar_features, strict=True)]

    def bev_levels(self, lidar_features: LidarFeatures | None) -> list[BEVLevel]:
        """Build the BEV levels of the frame once, none for a head without them.

        The levels come from the ``bev`` module over the stages, or the lidar features are one
        BEV map placed on the voxel grid with ``bev_cell_size``. The shared block of the head
        refines that map first, so every decoder block reads the same refined map.

        Raises:
            ValueError: If a head placing a BEV map gets the sparse stages.
        """
        if not self.read_lidar:
            return []
        if self.bev is not None:
            return self.bev(lidar_features)
        if self.bev_cell_size is None:
            return []
        if not isinstance(lidar_features, torch.Tensor):
            raise ValueError("A head sampling a BEV map needs the map as its lidar features.")
        if self.shared_block is not None:
            lidar_features = self.shared_block(lidar_features)
        return [BEVLevel(lidar_features, self.bev_grid_origin, self.bev_cell_size)]

    def _check_modalities(
        self, lidar_features: LidarFeatures | None, image_features: ImageFeatures | None
    ) -> None:
        """Check that the head gets the features of the modalities its blocks read, and no other.

        Raises:
            ValueError: If the features of a modality are missing or have no block to read them.
        """
        for name, features, read in (
            ("lidar", lidar_features, self.read_lidar),
            ("image", image_features, len(self.image_aggregation) > 0),
        ):
            if read and features is None:
                raise ValueError(f"The head has {name} blocks and got no {name} features.")
            if not read and features is not None:
                raise ValueError(f"The head has no {name} blocks to read its {name} features.")
        if image_features is not None and image_features.strides != self.image_strides:
            raise ValueError(
                f"The image blocks read the strides {self.image_strides}, the pyramid has "
                f"{image_features.strides}."
            )

    @staticmethod
    def _batch_size(
        lidar_features: LidarFeatures | None, image_features: ImageFeatures | None = None
    ) -> int:
        """Batch size of the frame, from the lidar features or the images of a camera only head.

        The lidar features give it from the BEV map or the offsets of the first stage.
        """
        if lidar_features is None:
            return image_features.batch_size
        if isinstance(lidar_features, torch.Tensor):
            return lidar_features.shape[0]
        return lidar_features[0][2].shape[0]

    def _decode(
        self,
        levels: Sequence[SparseLevel],
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        time_interval: torch.Tensor,
        temp_feature: torch.Tensor | None,
        temp_anchor: torch.Tensor | None,
        merge: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
        ]
        | None,
        denoising: DenoisingQueries | None = None,
        context: Any = None,
        bev_levels: Sequence[BEVLevel] = (),
        image_features: ImageFeatures | None = None,
    ) -> _DecoderOutputs:
        """Run the decoder blocks over the lidar and the image features of a frame.

        Denoising queries ride at the tail and never reach the merge or the final state.

        Args:
            levels: Sampled stages with their voxel lookup.
            instance_feature: Fresh instance features ``(B, N, C)``.
            anchor: Fresh anchors ``(B, N, D)``.
            time_interval: Frame gaps of the slots ``(B,)`` in seconds.
            temp_feature: Gated temporal instance features ``(B, K, C)``, or
                ``None`` in single frame mode.
            temp_anchor: Warped temporal anchors ``(B, K, D)``.
            merge: Temporal merge applied after block 0, or ``None``.
            denoising: Optional denoising queries appended at the tail with
                zero features and the isolation mask on the instance attention.
            context: Frame context of a subclass head, None for the lidar head.
            bev_levels: BEV maps of a head with a BEV branch.
            image_features: Pyramid and projection of the cameras, read by the image blocks
                alone. None for a head without them.

        Returns:
            The per layer outputs over the regular and the denoising queries
            and the final state of the regular queries.
        """
        num_query = instance_feature.shape[1]
        attn_mask = None
        if denoising is not None:
            dn_feature = instance_feature.new_zeros(
                (instance_feature.shape[0], denoising.anchor.shape[1], instance_feature.shape[-1])
            )
            instance_feature = torch.cat([instance_feature, dn_feature], dim=1)
            anchor = torch.cat([anchor, denoising.anchor], dim=1)
            attn_mask = denoising.attn_mask
        anchor_embed = self.anchor_encoder(anchor)
        temp_anchor_embed = None if temp_anchor is None else self.anchor_encoder(temp_anchor)
        block_context = self._block_context(anchor, context)
        classification: list[torch.Tensor] = []
        prediction: list[torch.Tensor] = []
        quality: list[torch.Tensor] = []
        for block_index in range(self.num_decoder):
            if block_index > 0:
                if temp_feature is not None:
                    instance_feature = self.temp_gnn[block_index - 1](
                        instance_feature, anchor_embed, temp_feature, temp_anchor_embed
                    )
                    instance_feature = self.temp_gnn_norm[block_index - 1](instance_feature)
                instance_feature = self.gnn[block_index - 1](
                    instance_feature, anchor_embed, attn_mask=attn_mask
                )
                instance_feature = self.gnn_norm[block_index - 1](instance_feature)
            if self.read_lidar:
                instance_feature = self.deformable[block_index](
                    instance_feature, anchor, anchor_embed, levels, bev_levels, block_context
                )
                instance_feature = self.deformable_norm[block_index](instance_feature)
            if image_features is not None:
                instance_feature = self.image_aggregation[block_index](
                    instance_feature, anchor, anchor_embed, image_features
                )
                instance_feature = self.image_norm[block_index](instance_feature)
            instance_feature = self.ffn_norm[block_index](
                instance_feature + self.ffn[block_index](instance_feature)
            )
            anchor, cls_logits, quality_logits = self._refine(
                block_index, instance_feature, anchor, anchor_embed, time_interval
            )
            classification.append(cls_logits)
            prediction.append(anchor)
            if quality_logits is not None:
                quality.append(quality_logits)
            if block_index == 0 and merge is not None:
                merged_feature, merged_anchor = merge(
                    instance_feature[:, :num_query],
                    anchor[:, :num_query],
                    cls_logits[:, :num_query],
                )
                instance_feature = torch.cat(
                    [merged_feature, instance_feature[:, num_query:]], dim=1
                )
                anchor = torch.cat([merged_anchor, anchor[:, num_query:]], dim=1)
                # The merge brings in the temporal instances, the context follows the new set
                block_context = self._block_context(anchor, context)
            if block_index != self.num_decoder - 1:
                anchor_embed = self.anchor_encoder(anchor)
                # Read the temporal memory at the refined anchors
                if block_index > 0 and temp_anchor_embed is not None:
                    temp_anchor_embed = anchor_embed[:, : temp_anchor_embed.shape[1]]
        return _DecoderOutputs(
            classification=classification,
            prediction=prediction,
            quality=quality,
            instance_feature=instance_feature[:, :num_query],
            anchor=anchor[:, :num_query],
            cls_logits=cls_logits[:, :num_query],
        )

    def forward_temporal_export(
        self,
        lidar_features: LidarFeatures | None,
        temp_instance_feature: torch.Tensor,
        temp_anchor: torch.Tensor,
        temp_confidence: torch.Tensor,
        temp_valid: torch.Tensor,
        time_interval: torch.Tensor,
        prev2current: torch.Tensor,
        context: Any = None,
        image_features: ImageFeatures | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run the temporal decoder with the bank state at the graph boundary.

        A frame's ``*_out`` state feeds the next one, a reset feeds zeros and ``temp_valid = 0``.
        The query index of every state slot lets the runtime carry its track ids into the slots.

        Args:
            lidar_features: Features, voxel coordinates and batch offsets of the sampled stages,
                from the finest to the coarsest, or the dense BEV map of a BEV only head.
            temp_instance_feature: Cached instance features ``(B, K, C)``.
            temp_anchor: Cached anchors ``(B, K, D)`` in the previous frame.
            temp_confidence: Cached instance confidences ``(B, K)``.
            temp_valid: Continuity gates of the slots ``(B,)`` in ``{0, 1}``.
            time_interval: Frame gaps ``(B,)`` in seconds.
            prev2current: Transforms from the previous into the current frame ``(B, 4, 4)``.
            context: Frame context of a subclass head, None for the lidar head.
            image_features: Pyramid and projection of the cameras, None for a head without
                image blocks.

        Returns:
            Last layer ``classification``, ``prediction``, ``instance_feature`` and, for a head
            with quality, ``quality``, the state of the next frame, ``temp_instance_feature_out``, ``temp_anchor_out`` and
            ``temp_confidence_out``, and the query of every state slot, ``temp_query_index_out``.

        Raises:
            ValueError: If the head has no temporal memory.
        """
        bank = self.instance_bank
        if bank.num_temp_instances == 0:
            raise ValueError(
                "forward_temporal_export needs a temporal head (num_temp_instances > 0)."
            )
        self._check_modalities(lidar_features, image_features)
        batch_size = self._batch_size(lidar_features, image_features)
        instance_feature = bank.instance_feature[None].expand(batch_size, -1, -1)
        anchor = bank.anchor[None].expand(batch_size, -1, -1)

        gate = temp_valid[:, None, None]
        warped_anchor = bank._project_anchor(temp_anchor, time_interval, prev2current)
        temp_feature = temp_instance_feature * gate
        temp_anchor = warped_anchor * gate
        prev_confidence = temp_confidence * temp_valid[:, None]

        def merge(
            feature: torch.Tensor, anchor_state: torch.Tensor, cls_logits: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return bank.merge_temporal(
                feature,
                anchor_state,
                cls_logits,
                temp_feature,
                temp_anchor,
                gate,
                bank.num_temp_instances,
            )

        decoded = self._decode(
            self.levels(lidar_features),
            instance_feature,
            anchor,
            time_interval,
            temp_feature,
            temp_anchor,
            merge,
            context=context,
            bev_levels=self.bev_levels(lidar_features),
            image_features=image_features,
        )
        state_feature, state_anchor, state_confidence, query_index = bank.select_temporal_state(
            decoded.instance_feature,
            decoded.anchor,
            decoded.cls_logits,
            prev_confidence,
            bank.confidence_decay,
            time_interval,
            bank.num_temp_instances,
            bank.anchor_dim,
        )
        outputs = {
            "classification": decoded.classification[-1],
            "prediction": decoded.prediction[-1],
            "instance_feature": decoded.instance_feature,
            "temp_instance_feature_out": state_feature,
            "temp_anchor_out": state_anchor,
            "temp_confidence_out": state_confidence,
            TEMPORAL_QUERY_INDEX_OUTPUT_NAME: query_index,
        }
        if decoded.quality:
            outputs["quality"] = decoded.quality[-1]
        return outputs

    @property
    def is_temporal(self) -> bool:
        """Whether the head propagates instances across frames."""
        return self.instance_bank.num_temp_instances > 0

    def temporal_state_export_inputs(self, device: torch.device) -> tuple[torch.Tensor, ...]:
        """Return reset state sample inputs for tracing the temporal export.

        Raises:
            ValueError: If the head has no temporal memory.
        """
        bank = self.instance_bank
        if bank.num_temp_instances == 0:
            raise ValueError(
                "Temporal export inputs need a temporal head (num_temp_instances > 0)."
            )
        num_temp = bank.num_temp_instances
        return (
            torch.zeros(1, num_temp, self.embed_dims, device=device),
            torch.zeros(1, num_temp, bank.anchor_dim, device=device),
            torch.zeros(1, num_temp, device=device),
            torch.zeros(1, device=device),
            torch.full((1,), bank.default_time_interval, device=device),
            torch.eye(4, device=device).unsqueeze(0),
        )

    def prepare_for_export(self) -> "SparseQueryHead":
        """Return an export ready copy that emits the last layer only.

        Temporal heads export through :meth:`forward_temporal_export`.
        """
        head = deepcopy(self)
        head._export_mode = True
        for attention_block in head.gnn:
            attention_block.attn = ExportableMultiheadAttention(attention_block.attn)
        if head.is_temporal:
            for attention_block in head.temp_gnn:
                attention_block.attn = ExportableMultiheadAttention(attention_block.attn)
        return head.eval()
