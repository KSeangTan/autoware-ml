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
from autoware_ml.transforms.camera_lidar.geometry import GlobalRotScaleTrans, RandomFlip3D


def test_global_rot_scale_trans_image_updates_lidar2img() -> None:
    sample = {
        "points": np.array([[1.0, 0.0, 0.0, 1.0]], dtype=np.float32),
        "gt_boxes": np.array([[1.0, 0.0, 0.0, 4.0, 2.0, 1.0, 0.0]], dtype=np.float32),
        "lidar2cam": np.tile(np.eye(4, dtype=np.float32), (2, 1, 1)),
        "camera_intrinsics": np.tile(np.eye(4, dtype=np.float32), (2, 1, 1)),
    }

    np.random.seed(0)
    output = GlobalRotScaleTrans(rot_range=[0.1, 0.1], scale_ratio_range=[1.5, 1.5])(sample)

    assert output["points"].shape == (1, 4)
    assert output["lidar2img"].shape == (2, 4, 4)
    assert np.allclose(output["gt_boxes"][0, 3:6], np.array([6.0, 3.0, 1.5], dtype=np.float32))


def test_bevfusion_random_flip3d_records_flip_matrix() -> None:
    sample = {
        "points": np.array([[1.0, 2.0, 0.0, 1.0]], dtype=np.float32),
        "gt_boxes": np.array([[1.0, 2.0, 0.0, 4.0, 2.0, 1.0, 0.2]], dtype=np.float32),
        "lidar2cam": np.tile(np.eye(4, dtype=np.float32), (2, 1, 1)),
        "camera_intrinsics": np.tile(np.eye(4, dtype=np.float32), (2, 1, 1)),
    }

    output = RandomFlip3D(flip_ratio_bev_horizontal=1.0, flip_ratio_bev_vertical=0.0)(sample)

    assert np.allclose(output["points"][0, 1], -2.0)
    assert output["bev_flip_matrix"][1, 1] == -1.0


def test_image_aug3d_creates_img_aug_matrix() -> None:
    sample = {"img": [np.ones((8, 8, 3), dtype=np.float32), np.ones((8, 8, 3), dtype=np.float32)]}

    output = ImageAug3D(
        final_dim=[6, 6], resize_lim=[1.0, 1.0], bot_pct_lim=[0.0, 0.0], training=False
    )(sample)

    assert len(output["img"]) == 2
    assert output["img"][0].shape[:2] == (6, 6)
    assert output["img_aug_matrix"].shape == (2, 4, 4)
