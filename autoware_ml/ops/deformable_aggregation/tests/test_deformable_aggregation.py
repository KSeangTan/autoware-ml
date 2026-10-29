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

"""Unit tests for the deformable aggregation extension."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from autoware_ml.dataclasses.models.detection2d.feature_maps import FlattenedFeatureMaps
from autoware_ml.ops.deformable_aggregation import deformable_aggregation, flatten_feature_maps

# Outputs of the original Sparse4D v3 extension
SPARSE4D_V3_REFERENCE = Path(__file__).parent / "test_data" / "sparse4d_v3_reference.npz"


def _grid_sample_reference(
    feature_maps: Sequence[torch.Tensor],
    sampling_locations: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Aggregate with one ``grid_sample`` per camera and level, the reference formulation.

    Every (point, camera, level) sample is weighted by the weight of its channel group and the
    samples are summed. A location on or outside the image border is left out, as the kernel
    does.

    Args:
        feature_maps: One ``(batch, cameras, channels, height, width)`` tensor per level.
        sampling_locations: ``(batch, anchors, points, cameras, 2)`` locations in ``[0, 1]``.
        weights: ``(batch, anchors, points, cameras, levels, groups)`` aggregation weights.

    Returns:
        The aggregated features ``(batch, anchors, channels)``.
    """
    batch_size, num_anchors, num_points, num_cams, _ = sampling_locations.shape
    num_channels = feature_maps[0].shape[2]
    group_channels = num_channels // weights.shape[-1]
    inside = torch.logical_and(
        (sampling_locations > 0).all(dim=-1), (sampling_locations < 1).all(dim=-1)
    )
    grid = sampling_locations.mul(2.0).sub(1.0)

    output = torch.zeros(
        batch_size, num_anchors, num_channels, dtype=weights.dtype, device=weights.device
    )
    for cam in range(num_cams):
        cam_grid = grid[:, :, :, cam].reshape(batch_size, -1, 1, 2)
        cam_inside = inside[:, :, :, cam].unsqueeze(-1).to(weights.dtype)
        for level, feature_map in enumerate(feature_maps):
            sampled = F.grid_sample(
                feature_map[:, cam],
                cam_grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            sampled = sampled.squeeze(-1).reshape(batch_size, num_channels, num_anchors, num_points)
            sampled = sampled.permute(0, 2, 3, 1)
            level_weights = weights[:, :, :, cam, level].repeat_interleave(group_channels, dim=-1)
            output = output + sampled.mul(level_weights).mul(cam_inside).sum(dim=2)
    return output


def _pyramid(
    batch_size: int,
    num_cams: int,
    num_channels: int,
    level_shapes: Sequence[tuple[int, int]],
    device: torch.device,
) -> list[torch.Tensor]:
    """Draw a random feature pyramid with one tensor per level."""
    return [
        torch.randn(batch_size, num_cams, num_channels, height, width, device=device)
        for height, width in level_shapes
    ]


def _locations(
    batch_size: int, num_anchors: int, num_points: int, num_cams: int, device: torch.device
) -> torch.Tensor:
    """Draw locations inside the image, with every fourth point pushed outside it."""
    locations = torch.rand(batch_size, num_anchors, num_points, num_cams, 2, device=device)
    locations = locations.mul(0.9).add(0.05)
    outside = torch.zeros_like(locations, dtype=torch.bool)
    outside[:, :, 3::4] = True
    return torch.where(outside, locations.add(1.0), locations)


def _weights(
    batch_size: int,
    num_anchors: int,
    num_points: int,
    num_cams: int,
    num_levels: int,
    num_groups: int,
    device: torch.device,
) -> torch.Tensor:
    """Draw aggregation weights that sum to one over the points, cameras and levels."""
    logits = torch.randn(
        batch_size, num_anchors, num_points, num_cams, num_levels, num_groups, device=device
    )
    return logits.flatten(2, 4).softmax(dim=2).reshape(logits.shape)


class TestFlattenFeatureMaps(unittest.TestCase):
    """The flattening lays the tokens out camera by camera, level by level, row-major."""

    def test_tokens_follow_the_camera_level_pixel_order(self) -> None:
        """
        Input: a two-camera pyramid with a 2x3 level holding 0..11 and a 1x2 level holding
            100..103, one channel.
        Expected: 16 tokens per batch element, the (height, width) of both levels repeated per
            camera, the level offsets 0, 6 for camera 0 and 8, 14 for camera 1, and every pixel
            at its level offset plus its row-major index.
        Check: the shapes, the two index tensors, and the tokens of camera 1 level 0 pixel
            (1, 2) and camera 0 level 1 pixel (0, 1).
        """
        level0 = torch.arange(12, dtype=torch.float32).reshape(1, 2, 1, 2, 3)
        level1 = torch.arange(4, dtype=torch.float32).add(100).reshape(1, 2, 1, 1, 2)

        flattened = flatten_feature_maps([level0, level1])

        self.assertEqual(flattened.features.shape, (1, 16, 1))
        torch.testing.assert_close(
            flattened.spatial_shapes, torch.tensor([[[2, 3], [1, 2]], [[2, 3], [1, 2]]])
        )
        torch.testing.assert_close(flattened.level_start_index, torch.tensor([[0, 6], [8, 14]]))
        # Camera 1, level 0, pixel (row 1, column 2): offset 8, index 1 x 3 + 2 = 5, value 11.
        self.assertEqual(flattened.features[0, 13, 0].item(), 11.0)
        # Camera 0, level 1, pixel (row 0, column 1): offset 6, index 1, value 101.
        self.assertEqual(flattened.features[0, 7, 0].item(), 101.0)

    def test_keeps_the_channels_of_every_token(self) -> None:
        """
        Input: a single-camera 2x2 level with two channels holding 0..7 in (channels, rows,
            columns) order.
        Expected: the channel axis moves last, so the token of pixel (1, 0) carries the values
            of both channels at that pixel.
        Check: token 2 equals [2, 6].
        """
        level0 = torch.arange(8, dtype=torch.float32).reshape(1, 1, 2, 2, 2)

        flattened = flatten_feature_maps([level0])

        torch.testing.assert_close(flattened.features[0, 2], torch.tensor([2.0, 6.0]))

    def test_rejects_an_empty_pyramid(self) -> None:
        """
        Input: no level at all.
        Expected: the flattening refuses, there is nothing to lay out.
        Check: a ValueError is raised.
        """
        with self.assertRaises(ValueError):
            flatten_feature_maps([])

    def test_rejects_levels_with_different_cameras(self) -> None:
        """
        Input: a level with two cameras and a level with three.
        Expected: the flattening refuses, every level must cover the same batch, cameras and
            channels.
        Check: a ValueError is raised.
        """
        level0 = torch.zeros(1, 2, 4, 2, 2)
        level1 = torch.zeros(1, 3, 4, 1, 1)

        with self.assertRaises(ValueError):
            flatten_feature_maps([level0, level1])


@unittest.skipUnless(torch.cuda.is_available(), "deformable_aggregation requires CUDA")
class TestDeformableAggregationSampling(unittest.TestCase):
    """Hand-checked sampling and weighting of the CUDA kernel."""

    def setUp(self) -> None:
        """Pin the device and build a 2x2 single-camera, single-level feature map."""
        self.device = torch.device("cuda:0")
        # Pixel (row, column) holds [row x 2 + column + 1, 10 x that]: (0,0) -> [1, 10], (0,1)
        # -> [2, 20], (1,0) -> [3, 30], (1,1) -> [4, 40].
        feature_map = torch.tensor(
            [[[[1.0, 2.0], [3.0, 4.0]], [[10.0, 20.0], [30.0, 40.0]]]], device=self.device
        )
        self.flattened = flatten_feature_maps([feature_map.unsqueeze(0)])

    def _aggregate(
        self, locations: Sequence[Sequence[float]], weights: Sequence[float]
    ) -> torch.Tensor:
        """Aggregate one anchor from ``(x, y)`` locations and one weight per location."""
        sampling_locations = torch.tensor(locations, device=self.device).reshape(1, 1, -1, 1, 2)
        point_weights = torch.tensor(weights, device=self.device).reshape(1, 1, -1, 1, 1, 1)
        return deformable_aggregation(
            self.flattened.features,
            self.flattened.spatial_shapes,
            self.flattened.level_start_index,
            sampling_locations,
            point_weights,
        )

    def test_samples_the_pixel_center_exactly(self) -> None:
        """
        Input: one point at (0.25, 0.75), the center of pixel (row 1, column 0), weight 1.
        Expected: the channels of that pixel, [3, 30], without interpolation.
        Check: the output equals [3, 30].
        """
        output = self._aggregate([[0.25, 0.75]], [1.0])

        torch.testing.assert_close(output, torch.tensor([[[3.0, 30.0]]], device=self.device))

    def test_interpolates_between_neighboring_pixel_centers(self) -> None:
        """
        Input: one point at (0.5, 0.25), halfway between the centers of pixels (0, 0) and
            (0, 1), weight 1.
        Expected: the mean of the two pixels, [1.5, 15].
        Check: the output equals [1.5, 15].
        """
        output = self._aggregate([[0.5, 0.25]], [1.0])

        torch.testing.assert_close(output, torch.tensor([[[1.5, 15.0]]], device=self.device))

    def test_zero_pads_past_the_last_pixel_center(self) -> None:
        """
        Input: one point at (0.125, 0.25), a quarter pixel left of the center of pixel (0, 0),
            weight 1.
        Expected: the missing left neighbor counts as zero, so pixel (0, 0) is weighted by
            0.75: [0.75, 7.5].
        Check: the output equals [0.75, 7.5].
        """
        output = self._aggregate([[0.125, 0.25]], [1.0])

        torch.testing.assert_close(output, torch.tensor([[[0.75, 7.5]]], device=self.device))

    def test_locations_on_or_outside_the_border_contribute_nothing(self) -> None:
        """
        Input: six points at x or y equal to 0 or 1, or beyond the image, weight 1, with the
            features, locations and weights requiring gradients.
        Expected: every point is skipped, so the output is zero and nothing receives a gradient.
        Check: the output and the three gradients are all zero.
        """
        locations = [[0.0, 0.5], [1.0, 0.5], [0.5, 0.0], [0.5, 1.0], [-0.2, 0.5], [1.3, 0.5]]
        sampling_locations = torch.tensor(locations, device=self.device).reshape(1, 1, -1, 1, 2)
        sampling_locations.requires_grad_(True)
        weights = torch.ones(1, 1, 6, 1, 1, 1, device=self.device, requires_grad=True)
        features = self.flattened.features.clone().requires_grad_(True)

        output = deformable_aggregation(
            features,
            self.flattened.spatial_shapes,
            self.flattened.level_start_index,
            sampling_locations,
            weights,
        )
        output.sum().backward()

        torch.testing.assert_close(output, torch.zeros_like(output))
        torch.testing.assert_close(features.grad, torch.zeros_like(features))
        torch.testing.assert_close(sampling_locations.grad, torch.zeros_like(sampling_locations))
        torch.testing.assert_close(weights.grad, torch.zeros_like(weights))

    def test_weights_scale_and_sum_the_samples(self) -> None:
        """
        Input: two points at the centers of pixels (0, 0) and (1, 1) with weights 2 and 0.5.
        Expected: 2 x [1, 10] + 0.5 x [4, 40] = [4, 40].
        Check: the output equals [4, 40].
        """
        output = self._aggregate([[0.25, 0.25], [0.75, 0.75]], [2.0, 0.5])

        torch.testing.assert_close(output, torch.tensor([[[4.0, 40.0]]], device=self.device))

    def test_channel_groups_take_their_own_weight(self) -> None:
        """
        Input: a 1x1 map with four channels [1, 2, 3, 4], one point at its center, and two
            group weights [2, 3].
        Expected: the first two channels take the first weight and the last two the second:
            [2, 4, 9, 12].
        Check: the output equals [2, 4, 9, 12].
        """
        feature_map = torch.tensor([1.0, 2.0, 3.0, 4.0], device=self.device).reshape(1, 1, 4, 1, 1)
        flattened = flatten_feature_maps([feature_map])
        sampling_locations = torch.tensor([[0.5, 0.5]], device=self.device).reshape(1, 1, 1, 1, 2)
        weights = torch.tensor([2.0, 3.0], device=self.device).reshape(1, 1, 1, 1, 1, 2)

        output = deformable_aggregation(
            flattened.features,
            flattened.spatial_shapes,
            flattened.level_start_index,
            sampling_locations,
            weights,
        )

        torch.testing.assert_close(
            output, torch.tensor([[[2.0, 4.0, 9.0, 12.0]]], device=self.device)
        )

    def test_cameras_and_levels_are_summed(self) -> None:
        """
        Input: two cameras with a 2x2 level and a 1x1 level. Camera 0 holds pixels 1..4 (mean
            2.5) and 10, camera 1 holds 100s and 1000. One point at the image center of every
            camera, weights 1 for camera 0 on both levels and for camera 1 on level 1 only.
        Expected: 2.5 + 10 + 1000 = 1012.5.
        Check: the output equals 1012.5.
        """
        level0 = torch.tensor(
            [[[[[1.0, 2.0], [3.0, 4.0]]], [[[100.0, 100.0], [100.0, 100.0]]]]], device=self.device
        )
        level1 = torch.tensor([[[[[10.0]]], [[[1000.0]]]]], device=self.device)
        flattened = flatten_feature_maps([level0, level1])
        sampling_locations = torch.full((1, 1, 1, 2, 2), 0.5, device=self.device)
        weights = torch.tensor([[1.0, 1.0], [0.0, 1.0]], device=self.device).reshape(
            1, 1, 1, 2, 2, 1
        )

        output = deformable_aggregation(
            flattened.features,
            flattened.spatial_shapes,
            flattened.level_start_index,
            sampling_locations,
            weights,
        )

        torch.testing.assert_close(output, torch.tensor([[[1012.5]]], device=self.device))

    def test_cameras_with_different_pyramid_shapes(self) -> None:
        """
        Input: a hand-built layout where camera 0 has a 1x2 level holding [1, 2] and camera 1
            a 2x1 level holding [3, 4], a point on column 1 of camera 0 and row 1 of camera 1,
            weights 1.
        Expected: the kernel reads each camera's own shape, so it sums pixel 2 and pixel 4.
        Check: the output equals 6.
        """
        flattened = FlattenedFeatureMaps(
            features=torch.tensor([[[1.0], [2.0], [3.0], [4.0]]], device=self.device),
            spatial_shapes=torch.tensor([[[1, 2]], [[2, 1]]], device=self.device),
            level_start_index=torch.tensor([[0], [2]], device=self.device),
        )
        sampling_locations = torch.tensor([[0.75, 0.5], [0.5, 0.75]], device=self.device).reshape(
            1, 1, 1, 2, 2
        )
        weights = torch.ones(1, 1, 1, 2, 1, 1, device=self.device)

        output = deformable_aggregation(
            flattened.features,
            flattened.spatial_shapes,
            flattened.level_start_index,
            sampling_locations,
            weights,
        )

        torch.testing.assert_close(output, torch.tensor([[[6.0]]], device=self.device))

    def test_empty_anchors_give_an_empty_output(self) -> None:
        """
        Input: locations and weights for zero anchors.
        Expected: no kernel work and an empty output of the right shape.
        Check: the output shape is (1, 0, 2).
        """
        sampling_locations = torch.zeros(1, 0, 3, 1, 2, device=self.device)
        weights = torch.zeros(1, 0, 3, 1, 1, 1, device=self.device)

        output = deformable_aggregation(
            self.flattened.features,
            self.flattened.spatial_shapes,
            self.flattened.level_start_index,
            sampling_locations,
            weights,
        )

        self.assertEqual(output.shape, (1, 0, 2))


@unittest.skipUnless(torch.cuda.is_available(), "deformable_aggregation requires CUDA")
class TestDeformableAggregationAgainstSparse4DV3(unittest.TestCase):
    """The port reproduces the outputs recorded from the original Sparse4D v3 extension.

    The golden data holds a two-sample, three-camera, four-level pyramid with 16 channels in 8
    groups, 5 anchors with 7 points each, some points outside the image and some pinned to a
    pixel center and to the borders, the flattening of the reference ``feature_maps_format``,
    and the forward output and the three gradients of the reference kernels.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """Load the golden data onto the GPU."""
        cls.device = torch.device("cuda:0")
        with np.load(SPARSE4D_V3_REFERENCE) as reference:
            cls.reference = {
                name: torch.from_numpy(reference[name]).to(cls.device) for name in reference.files
            }
        cls.levels = [cls.reference[f"level_{level}"] for level in range(4)]

    def test_flattening_matches_the_reference_layout(self) -> None:
        """
        Input: the four pyramid levels of the golden data.
        Expected: the same tokens, spatial shapes and level offsets as the reference
            ``feature_maps_format``.
        Check: the three tensors of the flattening equal the recorded ones.
        """
        flattened = flatten_feature_maps(self.levels)

        torch.testing.assert_close(flattened.features, self.reference["flattened_features"])
        torch.testing.assert_close(flattened.spatial_shapes, self.reference["spatial_shapes"])
        torch.testing.assert_close(flattened.level_start_index, self.reference["level_start_index"])

    def test_forward_matches_the_reference_output(self) -> None:
        """
        Input: the recorded flattened features, index tensors, locations and weights.
        Expected: the aggregated features the reference forward kernel produced.
        Check: the output equals the recorded output, up to the float32 summation order of
            the atomics.
        """
        output = deformable_aggregation(
            self.reference["flattened_features"],
            self.reference["spatial_shapes"],
            self.reference["level_start_index"],
            self.reference["sampling_locations"],
            self.reference["weights"],
        )

        torch.testing.assert_close(output, self.reference["output"], rtol=1e-5, atol=1e-5)

    def test_backward_matches_the_reference_gradients(self) -> None:
        """
        Input: the recorded inputs requiring gradients and the recorded output gradient.
        Expected: the gradients of the features, the locations and the weights the reference
            backward kernel produced.
        Check: the three gradients equal the recorded ones, up to the float32 summation order
            of the atomics.
        """
        features = self.reference["flattened_features"].clone().requires_grad_(True)
        sampling_locations = self.reference["sampling_locations"].clone().requires_grad_(True)
        weights = self.reference["weights"].clone().requires_grad_(True)

        output = deformable_aggregation(
            features,
            self.reference["spatial_shapes"],
            self.reference["level_start_index"],
            sampling_locations,
            weights,
        )
        output.backward(self.reference["grad_output"])

        torch.testing.assert_close(
            features.grad, self.reference["grad_features"], rtol=1e-5, atol=1e-5
        )
        torch.testing.assert_close(
            sampling_locations.grad,
            self.reference["grad_sampling_locations"],
            rtol=1e-5,
            atol=1e-5,
        )
        torch.testing.assert_close(
            weights.grad, self.reference["grad_weights"], rtol=1e-5, atol=1e-5
        )


@unittest.skipUnless(torch.cuda.is_available(), "deformable_aggregation requires CUDA")
class TestDeformableAggregationAgainstGridSample(unittest.TestCase):
    """The kernel and its gradients match the grid_sample formulation on random inputs."""

    def setUp(self) -> None:
        """Draw a multi-camera, multi-level problem with points inside and outside the images."""
        torch.manual_seed(0)
        self.device = torch.device("cuda:0")
        self.batch_size = 2
        self.num_cams = 3
        self.num_channels = 8
        self.num_groups = 4
        self.level_shapes = [(5, 7), (3, 4), (2, 2)]
        self.num_anchors = 6
        self.num_points = 5
        self.feature_maps = _pyramid(
            self.batch_size, self.num_cams, self.num_channels, self.level_shapes, self.device
        )
        self.sampling_locations = _locations(
            self.batch_size, self.num_anchors, self.num_points, self.num_cams, self.device
        )
        self.weights = _weights(
            self.batch_size,
            self.num_anchors,
            self.num_points,
            self.num_cams,
            len(self.level_shapes),
            self.num_groups,
            self.device,
        )

    def _aggregate(
        self,
        feature_maps: Sequence[torch.Tensor],
        sampling_locations: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Run the kernel on a pyramid."""
        flattened = flatten_feature_maps(feature_maps)
        return deformable_aggregation(
            flattened.features,
            flattened.spatial_shapes,
            flattened.level_start_index,
            sampling_locations,
            weights,
        )

    def test_forward_matches_grid_sample(self) -> None:
        """
        Input: a random three-camera, three-level pyramid with 8 channels in 4 groups, 6
            anchors with 5 points each, every fourth point outside the image.
        Expected: the fused kernel equals one grid_sample per camera and level followed by the
            weighted sum over the points, cameras and levels.
        Check: the output shape and its values against the grid_sample formulation.
        """
        output = self._aggregate(self.feature_maps, self.sampling_locations, self.weights)

        expected = _grid_sample_reference(self.feature_maps, self.sampling_locations, self.weights)
        self.assertEqual(output.shape, (self.batch_size, self.num_anchors, self.num_channels))
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)

    def test_backward_matches_grid_sample_autograd(self) -> None:
        """
        Input: the same random problem with the levels, the locations and the weights
            requiring gradients, and a random output gradient.
        Expected: the analytic gradients of the kernel equal autograd through the grid_sample
            formulation.
        Check: the gradient of every level, of the locations and of the weights.
        """
        grad_output = torch.randn(
            self.batch_size, self.num_anchors, self.num_channels, device=self.device
        )

        kernel_maps = [
            feature_map.clone().requires_grad_(True) for feature_map in self.feature_maps
        ]
        kernel_locations = self.sampling_locations.clone().requires_grad_(True)
        kernel_weights = self.weights.clone().requires_grad_(True)
        output = self._aggregate(kernel_maps, kernel_locations, kernel_weights)
        kernel_grads = torch.autograd.grad(
            output.mul(grad_output).sum(), kernel_maps + [kernel_locations, kernel_weights]
        )

        reference_maps = [
            feature_map.clone().requires_grad_(True) for feature_map in self.feature_maps
        ]
        reference_locations = self.sampling_locations.clone().requires_grad_(True)
        reference_weights = self.weights.clone().requires_grad_(True)
        expected = _grid_sample_reference(reference_maps, reference_locations, reference_weights)
        reference_grads = torch.autograd.grad(
            expected.mul(grad_output).sum(),
            reference_maps + [reference_locations, reference_weights],
        )

        for kernel_grad, reference_grad in zip(kernel_grads, reference_grads):
            torch.testing.assert_close(kernel_grad, reference_grad, rtol=1e-4, atol=1e-4)

    def test_half_inputs_are_computed_in_float32(self) -> None:
        """
        Input: the random problem with the features, locations and weights in float16.
        Expected: the inputs are cast to float32 inside, so the output is the float32 result
            of the rounded inputs.
        Check: the output dtype and its values against the kernel run on the rounded inputs.
        """
        flattened = flatten_feature_maps(self.feature_maps)
        half_output = deformable_aggregation(
            flattened.features.half(),
            flattened.spatial_shapes,
            flattened.level_start_index,
            self.sampling_locations.half(),
            self.weights.half(),
        )

        expected = deformable_aggregation(
            flattened.features.half().float(),
            flattened.spatial_shapes,
            flattened.level_start_index,
            self.sampling_locations.half().float(),
            self.weights.half().float(),
        )
        self.assertEqual(half_output.dtype, torch.float32)
        torch.testing.assert_close(half_output, expected, rtol=1e-6, atol=1e-6)

    def test_non_contiguous_inputs_are_accepted(self) -> None:
        """
        Input: the random problem with the locations stored transposed in memory.
        Expected: the wrapper makes them contiguous, so the result is unchanged.
        Check: the output against the grid_sample formulation.
        """
        flattened = flatten_feature_maps(self.feature_maps)
        transposed_locations = self.sampling_locations.transpose(1, 2).contiguous().transpose(1, 2)
        self.assertFalse(transposed_locations.is_contiguous())

        output = deformable_aggregation(
            flattened.features,
            flattened.spatial_shapes,
            flattened.level_start_index,
            transposed_locations,
            self.weights,
        )

        expected = _grid_sample_reference(self.feature_maps, self.sampling_locations, self.weights)
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)


@unittest.skipUnless(torch.cuda.is_available(), "deformable_aggregation requires CUDA")
class TestDeformableAggregationValidation(unittest.TestCase):
    """The binding rejects inputs that do not fit the kernel's layout."""

    def setUp(self) -> None:
        """Build a small valid problem to break one input at a time."""
        self.device = torch.device("cuda:0")
        self.flattened = flatten_feature_maps(_pyramid(1, 2, 4, [(2, 2)], self.device))
        self.sampling_locations = torch.full((1, 3, 2, 2, 2), 0.5, device=self.device)
        self.weights = torch.ones(1, 3, 2, 2, 1, 2, device=self.device)

    def _aggregate(
        self,
        flattened: FlattenedFeatureMaps,
        sampling_locations: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Run the kernel on the given inputs."""
        return deformable_aggregation(
            flattened.features,
            flattened.spatial_shapes,
            flattened.level_start_index,
            sampling_locations,
            weights,
        )

    def test_accepts_the_valid_problem(self) -> None:
        """
        Input: a two-camera, single-level pyramid with 4 channels, 3 anchors with 2 points and
            2 weight groups.
        Expected: the checks pass and the kernel runs.
        Check: the output shape is (1, 3, 4).
        """
        output = self._aggregate(self.flattened, self.sampling_locations, self.weights)

        self.assertEqual(output.shape, (1, 3, 4))

    def test_rejects_channels_not_divisible_by_the_groups(self) -> None:
        """
        Input: weights with 3 groups for 4 channels.
        Expected: the binding refuses, the channels cannot be split into the groups.
        Check: a RuntimeError mentioning divisibility is raised.
        """
        weights = torch.ones(1, 3, 2, 2, 1, 3, device=self.device)

        with self.assertRaisesRegex(RuntimeError, "divisible"):
            self._aggregate(self.flattened, self.sampling_locations, weights)

    def test_rejects_locations_with_another_camera_count(self) -> None:
        """
        Input: locations for 3 cameras on a 2-camera pyramid.
        Expected: the binding refuses, the locations must cover the cameras of the pyramid.
        Check: a RuntimeError mentioning the cameras is raised.
        """
        sampling_locations = torch.full((1, 3, 2, 3, 2), 0.5, device=self.device)

        with self.assertRaisesRegex(RuntimeError, "cameras"):
            self._aggregate(self.flattened, sampling_locations, self.weights)

    def test_rejects_weights_with_another_level_count(self) -> None:
        """
        Input: weights for 2 levels on a single-level pyramid.
        Expected: the binding refuses, the weights must cover the levels of the pyramid.
        Check: a RuntimeError mentioning the levels is raised.
        """
        weights = torch.ones(1, 3, 2, 2, 2, 2, device=self.device)

        with self.assertRaisesRegex(RuntimeError, "levels"):
            self._aggregate(self.flattened, self.sampling_locations, weights)

    def test_rejects_cpu_tensors(self) -> None:
        """
        Input: the valid problem on the CPU.
        Expected: the binding refuses, the kernel is CUDA only.
        Check: a RuntimeError mentioning CUDA is raised.
        """
        flattened = flatten_feature_maps(_pyramid(1, 2, 4, [(2, 2)], torch.device("cpu")))

        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            self._aggregate(flattened, self.sampling_locations.cpu(), self.weights.cpu())


if __name__ == "__main__":
    unittest.main()
