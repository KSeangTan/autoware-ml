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
# Adapted from projects/mmdet3d_plugin/models/instance_bank.py of
# https://github.com/HorizonRobotics/Sparse4D (MIT License). The bank reads the frame
# metadata as tensors instead of mmcv ``img_metas`` dicts, takes the stream continuity of
# every lane from the caller and keeps the anchor projection of the key points generator.

"""The instance bank of Sparse4D.

The bank owns the learnable anchors and instance features every frame starts from, and
carries the most confident instances of a frame over to the next one as temporal instances:
their anchors are moved into the new ego frame along their velocity, their features are kept,
and their confidence decays so a stale instance is eventually replaced.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

from jaxtyping import Bool, Float, Float64, Int64
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from autoware_ml.models.detection3d.heads.sparse4d.box3d import (
    ANCHOR_DIMS,
    SparseBox3DKeyPointsGenerator,
)
from autoware_ml.models.detection3d.task_modules.streaming import topk_gather


class InstanceBankState(NamedTuple):
    """Instances a frame starts from.

    Attributes:
        instance_feature: Features of the learnable instances.
        anchor: Anchors of the learnable instances.
        temp_instance_feature: Features of the temporal instances, ``None`` for the first
            frame of a batch.
        temp_anchor: Anchors of the temporal instances in the current ego frame, ``None``
            likewise.
        time_interval: Time since the previous frame of every lane in seconds, the default
            interval where the lane starts a stream.
        temp_dn_anchor: The cached denoising anchors moved into the current ego frame, ``None``
            when none were given.
    """

    instance_feature: Float[torch.Tensor, "batch_size num_anchors embed_dims"]
    anchor: Float[torch.Tensor, "batch_size num_anchors anchor_dims"]
    temp_instance_feature: Float[torch.Tensor, "batch_size num_temp embed_dims"] | None
    temp_anchor: Float[torch.Tensor, "batch_size num_temp anchor_dims"] | None
    time_interval: Float[torch.Tensor, " batch_size"]
    temp_dn_anchor: Float[torch.Tensor, "batch_size num_groups num_noised anchor_dims"] | None


def _load_anchors(anchor: torch.Tensor | np.ndarray | str | Path) -> torch.Tensor:
    """Read the initial anchors from a tensor, an array or a ``.npy`` file."""
    if isinstance(anchor, (str, Path)):
        anchor = np.load(anchor)
    anchor = torch.as_tensor(anchor)
    if anchor.ndim != 2 or anchor.shape[1] != ANCHOR_DIMS:
        raise ValueError(
            f"anchor must have shape (num_anchors, {ANCHOR_DIMS}), got {tuple(anchor.shape)}."
        )
    return anchor.to(dtype=torch.float32)


class InstanceBank(nn.Module):
    """Learnable instances of every frame and the temporal instances carried between frames."""

    def __init__(
        self,
        num_anchor: int,
        embed_dims: int,
        anchor: torch.Tensor | np.ndarray | str | Path,
        num_temp_instances: int = 0,
        default_time_interval: float = 0.5,
        confidence_decay: float = 0.6,
        anchor_grad: bool = True,
        feat_grad: bool = True,
        max_time_interval: float = 2.0,
    ) -> None:
        """Initialize the bank.

        Args:
            num_anchor: Number of instances of a frame; the initial anchors are cut to it.
            embed_dims: Width of the instance features.
            anchor: Initial anchors ``(num_anchors, 11)``, see ``AnchorFieldIndex``, or the
                path of a ``.npy`` file holding them.
            num_temp_instances: Number of instances carried over to the next frame, ``0`` for
                a single-frame bank.
            default_time_interval: Time interval in seconds assumed where a lane starts a
                stream.
            confidence_decay: Factor the confidence of a temporal instance decays by per frame.
            anchor_grad: Whether the anchors are learned.
            feat_grad: Whether the instance features are learned.
            max_time_interval: Longest gap in seconds the temporal instances survive.

        Raises:
            ValueError: If the anchors do not have 11 fields or fewer than ``num_anchor`` are
                given, or ``num_temp_instances`` does not fit into ``num_anchor``.
        """
        super().__init__()
        anchors = _load_anchors(anchor)
        if anchors.shape[0] < num_anchor:
            raise ValueError(f"num_anchor={num_anchor} exceeds the {anchors.shape[0]} anchors.")
        if num_temp_instances >= num_anchor:
            raise ValueError(
                f"num_temp_instances={num_temp_instances} must leave room for new instances "
                f"among the {num_anchor} anchors."
            )
        self.embed_dims = embed_dims
        self.num_anchor = num_anchor
        self.num_temp_instances = max(num_temp_instances, 0)
        self.default_time_interval = default_time_interval
        self.confidence_decay = confidence_decay
        self.max_time_interval = max_time_interval

        self.register_buffer("anchor_init", anchors[:num_anchor].clone(), persistent=False)
        self.anchor = nn.Parameter(anchors[:num_anchor].clone(), requires_grad=anchor_grad)
        self.instance_feature = nn.Parameter(
            torch.zeros(num_anchor, embed_dims), requires_grad=feat_grad
        )
        self.reset()

    def init_weights(self) -> None:
        """Restore the initial anchors and draw the learnable instance features."""
        with torch.no_grad():
            self.anchor.copy_(self.anchor_init)
            if self.instance_feature.requires_grad:
                nn.init.xavier_uniform_(self.instance_feature, gain=1)

    def reset(self) -> None:
        """Forget the temporal instances of every lane."""
        self.cached_feature: torch.Tensor | None = None
        self.cached_anchor: torch.Tensor | None = None
        self.cached_timestamp: torch.Tensor | None = None
        self.cached_ego2global: torch.Tensor | None = None
        self.mask: torch.Tensor | None = None
        self.confidence: torch.Tensor | None = None
        self.temp_confidence: torch.Tensor | None = None
        self.instance_id: torch.Tensor | None = None
        self.prev_id = 0

    def get(
        self,
        timestamp: Float64[torch.Tensor, " batch_size"],
        ego2global: Float[torch.Tensor, "batch_size 4 4"],
        prev_exists: Bool[torch.Tensor, " batch_size"] | None = None,
        temp_dn_anchor: Float[torch.Tensor, "batch_size num_groups num_noised anchor_dims"]
        | None = None,
    ) -> InstanceBankState:
        """Start a frame.

        The learnable instances are replicated for the batch. When the previous frame was
        cached for the same batch size, the temporal instances are moved from the previous
        ego frame into the current one, along their velocity for the elapsed time, and the
        lanes whose gap exceeds ``max_time_interval`` or whose stream ended are marked for a
        reset in :meth:`update`. Otherwise the bank resets.

        Args:
            timestamp: Sample time of every lane in seconds.
            ego2global: Transform from the current ego frame of every lane to the map frame.
            prev_exists: True for every lane whose stream continues from the previous frame,
                ``None`` to decide on the time gap alone.
            temp_dn_anchor: Cached denoising anchors of the previous frame to move along with
                the temporal instances.

        Returns:
            The instances of the frame.
        """
        batch_size = timestamp.shape[0]
        instance_feature = self.instance_feature.unsqueeze(0).repeat(batch_size, 1, 1)
        anchor = self.anchor.unsqueeze(0).repeat(batch_size, 1, 1)

        if self.cached_anchor is None or batch_size != self.cached_anchor.shape[0]:
            self.reset()
            time_interval = anchor.new_full((batch_size,), self.default_time_interval)
            return InstanceBankState(instance_feature, anchor, None, None, time_interval, None)

        time_interval = (timestamp - self.cached_timestamp).to(dtype=anchor.dtype)
        mask = time_interval.abs() <= self.max_time_interval
        if prev_exists is not None:
            mask = mask & prev_exists
        self.mask = mask

        # Kilometer-scale poses are composed in double, see StreamPETR's memory bank.
        temp2cur = (torch.linalg.inv(ego2global.double()) @ self.cached_ego2global.double()).to(
            dtype=anchor.dtype
        )
        self.cached_anchor = SparseBox3DKeyPointsGenerator.anchor_projection(
            self.cached_anchor, temp2cur, -time_interval
        )
        if temp_dn_anchor is not None and batch_size == temp_dn_anchor.shape[0]:
            num_groups, num_noised = temp_dn_anchor.shape[1:3]
            temp_dn_anchor = SparseBox3DKeyPointsGenerator.anchor_projection(
                temp_dn_anchor.flatten(1, 2), temp2cur, -time_interval
            ).reshape(batch_size, num_groups, num_noised, -1)
        else:
            temp_dn_anchor = None

        time_interval = torch.where(
            (time_interval != 0) & mask,
            time_interval,
            time_interval.new_tensor(self.default_time_interval),
        )
        return InstanceBankState(
            instance_feature,
            anchor,
            self.cached_feature,
            self.cached_anchor,
            time_interval,
            temp_dn_anchor,
        )

    def update(
        self,
        instance_feature: Float[torch.Tensor, "batch_size num_instances embed_dims"],
        anchor: Float[torch.Tensor, "batch_size num_instances anchor_dims"],
        confidence: Float[torch.Tensor, "batch_size num_instances num_classes"],
    ) -> tuple[
        Float[torch.Tensor, "batch_size num_instances embed_dims"],
        Float[torch.Tensor, "batch_size num_instances anchor_dims"],
    ]:
        """Merge the temporal instances into the instances after the single-frame decoder.

        For every continuing lane the temporal instances lead the set and the most confident
        of the current instances fill the remaining slots; a lane that starts a stream keeps
        its instances. Denoising instances appended after the ``num_anchor`` learnable ones
        are passed through untouched.

        Args:
            instance_feature: Features of the current instances, the denoising ones last.
            anchor: Anchors of the current instances.
            confidence: Class logits of the current instances.

        Returns:
            The merged features and anchors.
        """
        if self.cached_feature is None or self.mask is None:
            return instance_feature, anchor

        num_dn = instance_feature.shape[1] - self.num_anchor
        dn_instance_feature = instance_feature[:, self.num_anchor :]
        dn_anchor = anchor[:, self.num_anchor :]
        instance_feature = instance_feature[:, : self.num_anchor]
        anchor = anchor[:, : self.num_anchor]
        confidence = confidence[:, : self.num_anchor]

        num_new = self.num_anchor - self.num_temp_instances
        confidence = confidence.max(dim=-1).values
        _, indices = torch.topk(confidence, num_new, dim=1)
        selected_feature = torch.cat(
            [self.cached_feature, topk_gather(instance_feature, indices)], dim=1
        )
        selected_anchor = torch.cat([self.cached_anchor, topk_gather(anchor, indices)], dim=1)
        lane_mask = self.mask[:, None, None]
        instance_feature = torch.where(lane_mask, selected_feature, instance_feature)
        anchor = torch.where(lane_mask, selected_anchor, anchor)
        if self.instance_id is not None:
            self.instance_id = torch.where(
                self.mask[:, None], self.instance_id, self.instance_id.new_tensor(-1)
            )

        if num_dn > 0:
            instance_feature = torch.cat([instance_feature, dn_instance_feature], dim=1)
            anchor = torch.cat([anchor, dn_anchor], dim=1)
        return instance_feature, anchor

    def cache(
        self,
        instance_feature: Float[torch.Tensor, "batch_size num_anchors embed_dims"],
        anchor: Float[torch.Tensor, "batch_size num_anchors anchor_dims"],
        confidence: Float[torch.Tensor, "batch_size num_anchors num_classes"],
        timestamp: Float64[torch.Tensor, " batch_size"],
        ego2global: Float[torch.Tensor, "batch_size 4 4"],
    ) -> None:
        """Keep the most confident instances of the frame for the next one.

        The confidence of an instance that was already temporal is the larger of its decayed
        previous confidence and its current one, so a well-tracked instance is not dropped
        after one weak frame. Nothing is cached by a single-frame bank.

        Args:
            instance_feature: Features of the learnable instances after the last decoder.
            anchor: Their anchors.
            confidence: Their class logits.
            timestamp: Sample time of every lane in seconds.
            ego2global: Transform from the ego frame of every lane to the map frame.
        """
        if self.num_temp_instances <= 0:
            return
        instance_feature = instance_feature.detach()
        anchor = anchor.detach()
        confidence = confidence.detach().max(dim=-1).values.sigmoid()

        self.cached_timestamp = timestamp.detach().clone()
        self.cached_ego2global = ego2global.detach().clone()
        if self.confidence is not None:
            decayed = torch.maximum(
                self.confidence * self.confidence_decay,
                confidence[:, : self.num_temp_instances],
            )
            confidence = torch.cat([decayed, confidence[:, self.num_temp_instances :]], dim=1)
        self.temp_confidence = confidence

        self.confidence, indices = torch.topk(confidence, self.num_temp_instances, dim=1)
        self.cached_feature = topk_gather(instance_feature, indices)
        self.cached_anchor = topk_gather(anchor, indices)

    def get_instance_id(
        self,
        confidence: Float[torch.Tensor, "batch_size num_anchors num_classes"],
        threshold: float | None = None,
    ) -> Int64[torch.Tensor, "batch_size num_anchors"]:
        """Assign a tracking id to every instance of the frame.

        A temporal instance keeps its id; every other instance, or only those at least
        ``threshold`` confident, gets a new one. Call it after :meth:`cache` so the ids of
        the cached instances follow them to the next frame.

        Args:
            confidence: Class logits of the learnable instances after the last decoder.
            threshold: Confidence a new instance needs to get an id, ``None`` for all.

        Returns:
            The id of every instance, ``-1`` for an instance without one.
        """
        confidence = confidence.max(dim=-1).values.sigmoid()
        instance_id = confidence.new_full(confidence.shape, -1).long()
        if self.instance_id is not None and self.instance_id.shape[0] == instance_id.shape[0]:
            instance_id[:, : self.instance_id.shape[1]] = self.instance_id

        mask = instance_id < 0
        if threshold is not None:
            mask = mask & (confidence >= threshold)
        num_new_instance = int(mask.sum().item())
        new_ids = torch.arange(num_new_instance, device=instance_id.device) + self.prev_id
        instance_id[mask] = new_ids
        self.prev_id += num_new_instance
        if self.num_temp_instances > 0:
            self.update_instance_id(instance_id, confidence)
        return instance_id

    def update_instance_id(
        self,
        instance_id: Int64[torch.Tensor, "batch_size num_anchors"],
        confidence: Float[torch.Tensor, "batch_size num_anchors"],
    ) -> None:
        """Keep the ids of the instances :meth:`cache` kept, in the order it kept them.

        Args:
            instance_id: Id of every instance of the frame.
            confidence: Confidence of every instance, read when nothing was cached.
        """
        temp_conf = self.temp_confidence if self.temp_confidence is not None else confidence
        _, indices = torch.topk(temp_conf, self.num_temp_instances, dim=1)
        instance_id = torch.gather(instance_id, 1, indices)
        self.instance_id = F.pad(
            instance_id, (0, self.num_anchor - self.num_temp_instances), value=-1
        )
