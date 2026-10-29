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

"""Building blocks, instance bank, targets, losses and decoder of the Sparse4D head."""

from autoware_ml.models.detection3d.heads.sparse4d.blocks import (
    AsymmetricFFN,
    DeformableFeatureAggregation,
    DenseDepthNet,
)
from autoware_ml.models.detection3d.heads.sparse4d.box3d import (
    ANCHOR_DIMS,
    AnchorFieldIndex,
    QualityIndex,
    SparseBox3DEncoder,
    SparseBox3DKeyPointsGenerator,
    SparseBox3DRefinementModule,
)
from autoware_ml.models.detection3d.heads.sparse4d.decoder import SparseBox3DDecoder, decode_box
from autoware_ml.models.detection3d.heads.sparse4d.instance_bank import (
    InstanceBank,
    InstanceBankState,
)
from autoware_ml.models.detection3d.heads.sparse4d.layers import Scale, linear_relu_ln
from autoware_ml.models.detection3d.heads.sparse4d.loss import SparseBox3DLoss, SparseBox3DLosses
from autoware_ml.models.detection3d.heads.sparse4d.target import (
    NEGATIVE_DN_CLASS,
    SparseBox3DTarget,
    TemporalDenoisingCache,
    TemporalDenoisingUpdate,
    encode_reg_target,
)

__all__ = [
    "ANCHOR_DIMS",
    "AnchorFieldIndex",
    "AsymmetricFFN",
    "DeformableFeatureAggregation",
    "DenseDepthNet",
    "InstanceBank",
    "InstanceBankState",
    "NEGATIVE_DN_CLASS",
    "QualityIndex",
    "Scale",
    "SparseBox3DDecoder",
    "SparseBox3DEncoder",
    "SparseBox3DKeyPointsGenerator",
    "SparseBox3DLoss",
    "SparseBox3DLosses",
    "SparseBox3DRefinementModule",
    "SparseBox3DTarget",
    "TemporalDenoisingCache",
    "TemporalDenoisingUpdate",
    "decode_box",
    "encode_reg_target",
    "linear_relu_ln",
]
