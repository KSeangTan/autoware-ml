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

"""Camera branch of the native BEVFusion detector."""

from __future__ import annotations


from jaxtyping import Float32
import torch
import torch.nn as nn


class Sparse4DCamera(nn.Module):
    """
    Encode multiview images into tuple of image features to be used in the downstream
    Sparse4D head
    .

    The branch owns the whole camera path: the image backbone and neck that encode the multiview
    images
    """

    def __init__(
        self,
        img_backbone: nn.Module,
        img_neck: nn.Module,
    ) -> None:
        """Initialize the camera branch.

        Args:
            img_backbone: Image backbone.
            img_neck: Image neck applied after the backbone.
        """
        super().__init__()
        self.img_backbone = img_backbone
        self.img_neck = img_neck

    def extract_image_features(
        self, image_batch: Float32[torch.Tensor, "batch_size num_cams 3 height width"]
    ) -> tuple[Float32[torch.Tensor, "batch_size num_cams channels feature_height feature_width"]]:
        """Encode multiview images into the neck features expected by the view transform.

        Args:
            image_batch: Image batch with shape ``(B, N, C, H, W)``.

        Returns:
            Tuple of neck feature tensor consumed by the downstream sparse4D head.
        """
        batch_size, num_cams, channels, image_height, image_width = image_batch.shape
        flat_images = image_batch.view(batch_size * num_cams, channels, image_height, image_width)
        image_features = self.img_backbone(flat_images)
        if isinstance(image_features, torch.Tensor):
            image_features = (image_features,)
        image_features = self.img_neck(image_features)
        return image_features

    def forward(
        self,
        image_batch: Float32[torch.Tensor, "batch_size num_cams 3 height width"],
    ) -> tuple[Float32[torch.Tensor, "batch_size num_cams channels feature_height feature_width"]]:
        """Encode multiview images into a BEV feature map.

        Args:
            image_batch: Multiview image tensors.

        Returns:
            Tuple of neck feature tensor consumed by the downstream sparse4D head.
        """
        return self.extract_image_features(image_batch=image_batch)
