# Copyright 2025 TIER IV, Inc.
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

"""Unit tests for camera-lidar transforms."""

import numpy as np

from autoware_ml.transforms.camera_lidar.camera_lidar import ImageAug3D


def test_image_aug3d_creates_img_aug_matrix() -> None:
    sample = {"img": [np.ones((8, 8, 3), dtype=np.float32), np.ones((8, 8, 3), dtype=np.float32)]}

    output = ImageAug3D(
        final_dim=[6, 6], resize_lim=[1.0, 1.0], bot_pct_lim=[0.0, 0.0], training=False
    )(sample)

    assert len(output["img"]) == 2
    assert output["img"][0].shape[:2] == (6, 6)
    assert output["img_aug_matrix"].shape == (2, 4, 4)
