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

"""Field layout of Sparse4D 3D box anchors."""

from enum import IntEnum


class AnchorIndex(IntEnum):
    """
    Sparse4D anchor field index.

    Anchors are stored as ``(x, y, z, log_l, log_w, log_h, sin_yaw, cos_yaw, vx, vy, vz)``.
    Sizes are stored in log space and the yaw is stored as its sine and cosine.

    Attributes:
      X: X coordinate of the box center.
      Y: Y coordinate of the box center.
      Z: Z coordinate of the box center.
      LOG_L: Log length of the box along its X axis.
      LOG_W: Log width of the box along its Y axis.
      LOG_H: Log height of the box.
      SIN_YAW: Sine of the box yaw.
      COS_YAW: Cosine of the box yaw.
      VX: Velocity in the X direction.
      VY: Velocity in the Y direction.
      VZ: Velocity in the Z direction.
    """

    X = 0
    Y = 1
    Z = 2
    LOG_L = 3
    LOG_W = 4
    LOG_H = 5
    SIN_YAW = 6
    COS_YAW = 7
    VX = 8
    VY = 9
    VZ = 10
