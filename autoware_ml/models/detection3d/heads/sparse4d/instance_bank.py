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

"""Temporal instance bank of the Sparse4D head."""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

from jaxtyping import Float32
import numpy as np
import torch
import torch.nn as nn

from autoware_ml.models.detection3d.heads.sparse4d.key_points import (
    SparseBox3DKeyPointsGenerator,
)


class Sparse4DFrameMetas(NamedTuple):
    """Per-frame metadata the instance bank needs to carry instances across frames.

    Attributes:
        timestamps: Per-sample capture time in seconds.
        ego2globals: Per-sample 4x4 transform from the ego frame the anchors are expressed in to
            the map frame.
    """

    timestamps: Float32[torch.Tensor, " batch_size"]
    ego2globals: Float32[torch.Tensor, "batch_size 4 4"]


class InstanceBankOutputs(NamedTuple):
    """Instances handed to the head for the current frame.

    Attributes:
        instance_feature: Learned initial instance features tiled over the batch.
        anchor: Learned initial anchors tiled over the batch.
        cached_feature: Features of the temporal instances kept from the previous frame, or
            None when there is no usable history.
        cached_anchor: Anchors of the temporal instances projected into the current ego frame,
            or None when there is no usable history.
        time_interval: Per-sample time gap to the previous frame, clamped to the default when
            the history is missing or stale.
    """

    instance_feature: Float32[torch.Tensor, "batch_size num_anchor embed_dims"]
    anchor: Float32[torch.Tensor, "batch_size num_anchor anchor_dims"]
    cached_feature: Float32[torch.Tensor, "batch_size num_temp_instances embed_dims"] | None
    cached_anchor: Float32[torch.Tensor, "batch_size num_temp_instances anchor_dims"] | None
    time_interval: Float32[torch.Tensor, " batch_size"]


def topk_instances(
    confidence: Float32[torch.Tensor, "batch_size num_instances"],
    k: int,
    inputs: Sequence[Float32[torch.Tensor, "batch_size num_instances channels"]],
) -> tuple[
    Float32[torch.Tensor, "batch_size k"],
    list[Float32[torch.Tensor, "batch_size k channels"]],
]:
    """Select the ``k`` most confident instances of every sample.

    Args:
        confidence: Per-instance confidence.
        k: Number of instances to keep per sample.
        inputs: Per-instance tensors to gather with the same selection.

    Returns:
        The selected confidences and the gathered inputs, both ordered by descending confidence.
    """
    batch_size, num_instances = confidence.shape[:2]
    confidence, indices = torch.topk(confidence, k, dim=1)
    # Offset each sample's indices into the flattened (batch_size * num_instances) axis so one
    # gather serves the whole batch.
    flat_indices = (
        indices + torch.arange(batch_size, device=indices.device)[:, None] * num_instances
    ).reshape(-1)
    outputs = [
        tensor.flatten(end_dim=1)[flat_indices].reshape(batch_size, k, -1) for tensor in inputs
    ]
    return confidence, outputs


class InstanceBank(nn.Module):
    """Keep the most confident instances of a frame and hand them to the next one.

    The bank owns the learned initial anchors and instance features. After every frame it caches
    the ``num_temp_instances`` most confident instances. On the next frame it projects the cached
    anchors into the current ego frame, drops samples whose history is too old, and lets the head
    replace the least confident current instances with the cached ones.
    """

    def __init__(
        self,
        num_anchor: int,
        embed_dims: int,
        anchor: str | Sequence[Sequence[float]] | np.ndarray,
        num_temp_instances: int = 0,
        default_time_interval: float = 0.5,
        confidence_decay: float = 0.6,
        anchor_grad: bool = True,
        feat_grad: bool = True,
        max_time_interval: float = 2.0,
    ) -> None:
        """Initialize the instance bank.

        Args:
            num_anchor: Number of anchors handed to the head per sample. Capped by the number of
                anchors provided.
            embed_dims: Dimension of the instance features.
            anchor: Initial anchors, either as an array of shape ``(num_anchor, anchor_dims)``
                or as the path of a ``.npy`` file holding one.
            num_temp_instances: Number of instances carried over from the previous frame.
            default_time_interval: Time gap in seconds assumed when no usable history exists.
            confidence_decay: Factor applied to cached confidences before they compete with the
                confidences of the current frame.
            anchor_grad: Whether the initial anchors are trainable.
            feat_grad: Whether the initial instance features are trainable.
            max_time_interval: Largest time gap in seconds for which the history is still used.
        """
        super().__init__()
        self.embed_dims = embed_dims
        self.num_temp_instances = num_temp_instances
        self.default_time_interval = default_time_interval
        self.confidence_decay = confidence_decay
        self.max_time_interval = max_time_interval

        if isinstance(anchor, str):
            anchor = np.load(anchor)
        anchor = np.asarray(anchor, dtype=np.float32)
        self.num_anchor = min(len(anchor), num_anchor)
        anchor = anchor[: self.num_anchor]
        self.anchor = nn.Parameter(torch.tensor(anchor), requires_grad=anchor_grad)
        self.register_buffer("anchor_init", torch.tensor(anchor), persistent=False)
        self.instance_feature = nn.Parameter(
            torch.zeros([self.num_anchor, self.embed_dims]), requires_grad=feat_grad
        )
        self._init_weights()

        self.cached_feature: (
            Float32[torch.Tensor, "batch_size num_temp_instances embed_dims"] | None
        )
        self.cached_anchor: (
            Float32[torch.Tensor, "batch_size num_temp_instances anchor_dims"] | None
        )
        self.confidence: Float32[torch.Tensor, "batch_size num_temp_instances"] | None
        self.mask: torch.Tensor | None
        self.previous_metas: Sparse4DFrameMetas | None
        self.reset()

    def _init_weights(self) -> None:
        """Reset the anchors to their initial values and Xavier-initialize trainable features."""
        with torch.no_grad():
            self.anchor.copy_(self.anchor_init)
            if self.instance_feature.requires_grad:
                nn.init.xavier_uniform_(self.instance_feature, gain=1)

    def reset(self) -> None:
        """Drop the cached instances so the next frame starts a new stream."""
        self.cached_feature = None
        self.cached_anchor = None
        self.confidence = None
        self.mask = None
        self.previous_metas = None

    def get(self, batch_size: int, metas: Sparse4DFrameMetas | None = None) -> InstanceBankOutputs:
        """Fetch the initial instances and the projected history for the current frame.

        Args:
            batch_size: Number of samples in the current frame.
            metas: Metadata of the current frame. Required whenever a history is cached.

        Returns:
            The initial instances, the projected temporal instances and the time gap to the
            previous frame.
        """
        # Step 1: Tile the learned initial instances over the batch.
        # instance_feature: (batch_size, num_anchor, embed_dims)
        # anchor: (batch_size, num_anchor, anchor_dims)
        instance_feature = torch.tile(self.instance_feature[None], (batch_size, 1, 1))
        anchor = torch.tile(self.anchor[None], (batch_size, 1, 1))

        # Step 2: Project the cached anchors into the current ego frame. The previous anchors
        # live in the previous ego frame, so the transform composes previous ego to map with map
        # to current ego. Samples whose history is older than max_time_interval are masked out.
        # A batch size change means a new stream, so the history is dropped instead.
        # T_temp2cur: (batch_size, 4, 4)
        # mask: (batch_size,)
        if (
            self.cached_anchor is not None
            and self.previous_metas is not None
            and metas is not None
            and batch_size == self.cached_anchor.shape[0]
        ):
            T_temp2cur = torch.linalg.inv(metas.ego2globals) @ self.previous_metas.ego2globals
            self.cached_anchor = SparseBox3DKeyPointsGenerator.anchor_projection(
                self.cached_anchor,
                [T_temp2cur.to(self.cached_anchor)],
                self.previous_metas.timestamps,
                [metas.timestamps],
            )[0]
            self.mask = (
                torch.abs(metas.timestamps - self.previous_metas.timestamps)
                <= self.max_time_interval
            )
        else:
            self.cached_feature = None
            self.cached_anchor = None
            self.confidence = None

        # Step 3: Compute the time gap to the previous frame. Without a matching history the
        # default gap is used. A zero or overly large gap also falls back to the default so the
        # velocity-based motion terms in the head stay well conditioned.
        # time_interval: (batch_size,)
        if (
            metas is None
            or self.previous_metas is None
            or self.previous_metas.timestamps.shape[0] != batch_size
        ):
            time_interval = instance_feature.new_full((batch_size,), self.default_time_interval)
        else:
            time_interval = (metas.timestamps - self.previous_metas.timestamps).to(
                dtype=instance_feature.dtype
            )
            time_interval = torch.where(
                torch.logical_or(
                    time_interval == 0, torch.abs(time_interval) > self.max_time_interval
                ),
                time_interval.new_tensor(self.default_time_interval),
                time_interval,
            )
        return InstanceBankOutputs(
            instance_feature=instance_feature,
            anchor=anchor,
            cached_feature=self.cached_feature,
            cached_anchor=self.cached_anchor,
            time_interval=time_interval,
        )

    def update(
        self,
        instance_feature: Float32[torch.Tensor, "batch_size num_anchor embed_dims"],
        anchor: Float32[torch.Tensor, "batch_size num_anchor anchor_dims"],
        confidence: Float32[torch.Tensor, "batch_size num_anchor num_classes"],
    ) -> tuple[
        Float32[torch.Tensor, "batch_size num_anchor embed_dims"],
        Float32[torch.Tensor, "batch_size num_anchor anchor_dims"],
    ]:
        """Replace the least confident current instances with the cached temporal instances.

        Args:
            instance_feature: Instance features refined by the current frame so far.
            anchor: Anchors refined by the current frame so far.
            confidence: Per-class classification logits of every instance.

        Returns:
            Instance features and anchors where, for samples with a usable history, the cached
            instances lead and the most confident current instances fill the remaining slots.
        """
        if self.cached_feature is None or self.cached_anchor is None or self.mask is None:
            return instance_feature, anchor

        # Step 1: Keep the most confident current instances that fit next to the cached ones.
        # confidence: (batch_size, num_anchor)
        # selected_feature: (batch_size, num_anchor - num_temp_instances, embed_dims)
        # selected_anchor: (batch_size, num_anchor - num_temp_instances, anchor_dims)
        num_current = self.num_anchor - self.num_temp_instances
        confidence = confidence.max(dim=-1).values
        _, (selected_feature, selected_anchor) = topk_instances(
            confidence, num_current, [instance_feature, anchor]
        )

        # Step 2: Prepend the cached instances so temporal instances occupy the leading slots.
        # selected_feature: (batch_size, num_anchor, embed_dims)
        # selected_anchor: (batch_size, num_anchor, anchor_dims)
        selected_feature = torch.cat([self.cached_feature, selected_feature], dim=1)
        selected_anchor = torch.cat([self.cached_anchor, selected_anchor], dim=1)

        # Step 3: Only samples with a fresh enough history take the merged instances. Others keep
        # the current-frame instances untouched.
        instance_feature = torch.where(self.mask[:, None, None], selected_feature, instance_feature)
        anchor = torch.where(self.mask[:, None, None], selected_anchor, anchor)
        return instance_feature, anchor

    def cache(
        self,
        instance_feature: Float32[torch.Tensor, "batch_size num_anchor embed_dims"],
        anchor: Float32[torch.Tensor, "batch_size num_anchor anchor_dims"],
        confidence: Float32[torch.Tensor, "batch_size num_anchor num_classes"],
        metas: Sparse4DFrameMetas | None = None,
    ) -> None:
        """Store the most confident instances of the current frame for the next one.

        Args:
            instance_feature: Final instance features of the current frame.
            anchor: Final anchors of the current frame.
            confidence: Per-class classification logits of every instance.
            metas: Metadata of the current frame, stored so the next frame can project the
                cached anchors.
        """
        if self.num_temp_instances <= 0:
            return

        # Step 1: Detach so no gradient flows across frames.
        instance_feature = instance_feature.detach()
        anchor = anchor.detach()
        confidence = confidence.detach()
        self.previous_metas = metas

        # Step 2: Score every instance by its best class. Instances that were themselves carried
        # over keep a decayed version of their previous score if that is higher, so a stable
        # track is not dropped after a single weak frame.
        # confidence: (batch_size, num_anchor)
        confidence = confidence.max(dim=-1).values.sigmoid()
        if self.confidence is not None:
            confidence[:, : self.num_temp_instances] = torch.maximum(
                self.confidence * self.confidence_decay,
                confidence[:, : self.num_temp_instances],
            )

        # Step 3: Keep the most confident instances as the history of the next frame.
        # self.confidence: (batch_size, num_temp_instances)
        # self.cached_feature: (batch_size, num_temp_instances, embed_dims)
        # self.cached_anchor: (batch_size, num_temp_instances, anchor_dims)
        self.confidence, (self.cached_feature, self.cached_anchor) = topk_instances(
            confidence, self.num_temp_instances, [instance_feature, anchor]
        )
