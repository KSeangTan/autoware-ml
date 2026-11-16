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

"""Camera branch of the Sparse4D detector."""

from __future__ import annotations

from collections.abc import Sequence

from jaxtyping import Float32
import torch
import torch.nn as nn

from autoware_ml.models.common.multiview_image_encoder import encode_multiview_pyramid
from autoware_ml.ops.deformable_aggregation.image_deformable_aggregation import ImageFeatures


class Sparse4DCamera(nn.Module):
    """Encode multiview images into the feature pyramid the Sparse4D head samples.

    The branch owns the whole camera path, the image backbone and the neck, and keeps every
    stride of the neck. It lifts nothing into BEV: the head projects its keypoints into the
    cameras and reads the pyramid there, so the image features stay apart from the BEV map of
    the lidar branch up to the instance features.
    """

    def __init__(
        self, img_backbone: nn.Module, img_neck: nn.Module, feature_strides: Sequence[int]
    ) -> None:
        """Initialize the camera branch.

        Args:
            img_backbone: Image backbone run on every camera.
            img_neck: Image neck returning one map per stride.
            feature_strides: Stride of every map of the neck against the images, finest first.

        Raises:
            ValueError: If no stride is given.
        """
        super().__init__()
        if len(feature_strides) == 0:
            raise ValueError("The camera branch needs the strides of its neck maps.")
        self.img_backbone = img_backbone
        self.img_neck = img_neck
        self.feature_strides = tuple(int(stride) for stride in feature_strides)

    def forward(
        self,
        image_batch: Float32[torch.Tensor, "batch_size num_cams 3 height width"],
        lidar2images: Float32[torch.Tensor, "batch_size num_cams 4 4"],
    ) -> ImageFeatures:
        """Encode multiview images into the pyramid of every camera.

        Args:
            image_batch: Multiview images ``(B, V, 3, H, W)``.
            lidar2images: Lidar frame to pixel matrices of those images, with the lidar and the
                image augmentations folded in.

        Returns:
            The pyramid with the projection that places it.

        Raises:
            ValueError: If the neck does not return one map per stride.
        """
        levels = encode_multiview_pyramid(self.img_backbone, self.img_neck, image_batch)
        if len(levels) != len(self.feature_strides):
            raise ValueError(
                f"The image neck returned {len(levels)} maps for the strides "
                f"{self.feature_strides}."
            )
        return ImageFeatures(
            levels=levels,
            strides=self.feature_strides,
            lidar2image=lidar2images,
            image_size=(image_batch.shape[-2], image_batch.shape[-1]),
        )
