// Copyright (c) 2024 Horizon Robotics. MIT License.
// Copyright 2026 TIER IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Adapted from projects/mmdet3d_plugin/ops/src/deformable_aggregation.cpp of
// https://github.com/HorizonRobotics/Sparse4D (MIT License). The reference trusted its
// inputs; this binding checks their shapes, dtypes, layout and device, and the backward
// allocates and returns the gradients instead of writing into caller-provided tensors.

#include <torch/torch.h>

#include <vector>

#ifdef WITH_CUDA
// Implemented in deformable_aggregation_cuda.cu
void deformable_aggregation_forward_cuda(
  const at::Tensor & features, const at::Tensor & spatial_shapes,
  const at::Tensor & level_start_index, const at::Tensor & sampling_locations,
  const at::Tensor & weights, at::Tensor & output);

void deformable_aggregation_backward_cuda(
  const at::Tensor & features, const at::Tensor & spatial_shapes,
  const at::Tensor & level_start_index, const at::Tensor & sampling_locations,
  const at::Tensor & weights, const at::Tensor & grad_output, at::Tensor & grad_features,
  at::Tensor & grad_sampling_locations, at::Tensor & grad_weights);
#endif

namespace
{
/**
 * @brief Check the shapes, dtypes, layout and device shared by the forward and the backward.
 *
 * See deformable_aggregation_cuda.cu for the tensor layouts.
 */
void check_inputs(
  const at::Tensor & features, const at::Tensor & spatial_shapes,
  const at::Tensor & level_start_index, const at::Tensor & sampling_locations,
  const at::Tensor & weights)
{
  TORCH_CHECK(features.dim() == 3, "features must have shape (batch, num_tokens, channels)");
  TORCH_CHECK(
    spatial_shapes.dim() == 3 && spatial_shapes.size(2) == 2,
    "spatial_shapes must have shape (cameras, levels, 2)");
  TORCH_CHECK(level_start_index.dim() == 2, "level_start_index must have shape (cameras, levels)");
  TORCH_CHECK(
    sampling_locations.dim() == 5 && sampling_locations.size(4) == 2,
    "sampling_locations must have shape (batch, anchors, points, cameras, 2)");
  TORCH_CHECK(
    weights.dim() == 6,
    "weights must have shape (batch, anchors, points, cameras, levels, groups)");

  const int64_t batch_size = features.size(0);
  const int64_t num_channels = features.size(2);
  const int64_t num_cams = spatial_shapes.size(0);
  const int64_t num_levels = spatial_shapes.size(1);
  const int64_t num_anchors = sampling_locations.size(1);
  const int64_t num_points = sampling_locations.size(2);
  const int64_t num_groups = weights.size(5);
  TORCH_CHECK(
    level_start_index.size(0) == num_cams && level_start_index.size(1) == num_levels,
    "level_start_index must match the (cameras, levels) of spatial_shapes");
  TORCH_CHECK(
    sampling_locations.size(0) == batch_size,
    "sampling_locations must have the batch size of features");
  TORCH_CHECK(
    sampling_locations.size(3) == num_cams,
    "sampling_locations must have the number of cameras of spatial_shapes");
  TORCH_CHECK(
    weights.size(0) == batch_size && weights.size(1) == num_anchors &&
      weights.size(2) == num_points && weights.size(3) == num_cams && weights.size(4) == num_levels,
    "weights must match the (batch, anchors, points, cameras) of sampling_locations and the "
    "levels of spatial_shapes");
  TORCH_CHECK(num_groups > 0, "weights must hold at least one group");
  TORCH_CHECK(
    num_channels % num_groups == 0, "the channels of features (", num_channels,
    ") must be divisible by the groups of weights (", num_groups, ")");

  TORCH_CHECK(features.scalar_type() == at::ScalarType::Float, "features must be float32");
  TORCH_CHECK(spatial_shapes.scalar_type() == at::ScalarType::Int, "spatial_shapes must be int32");
  TORCH_CHECK(
    level_start_index.scalar_type() == at::ScalarType::Int, "level_start_index must be int32");
  TORCH_CHECK(
    sampling_locations.scalar_type() == at::ScalarType::Float,
    "sampling_locations must be float32");
  TORCH_CHECK(weights.scalar_type() == at::ScalarType::Float, "weights must be float32");

  TORCH_CHECK(features.is_contiguous(), "features must be contiguous");
  TORCH_CHECK(spatial_shapes.is_contiguous(), "spatial_shapes must be contiguous");
  TORCH_CHECK(level_start_index.is_contiguous(), "level_start_index must be contiguous");
  TORCH_CHECK(sampling_locations.is_contiguous(), "sampling_locations must be contiguous");
  TORCH_CHECK(weights.is_contiguous(), "weights must be contiguous");

  TORCH_CHECK(features.is_cuda(), "deformable_aggregation is only implemented for CUDA tensors");
  TORCH_CHECK(
    spatial_shapes.device() == features.device() &&
      level_start_index.device() == features.device() &&
      sampling_locations.device() == features.device() && weights.device() == features.device(),
    "all inputs must be on the device of features");
}
}  // namespace

/**
 * @brief Aggregate the multi-camera, multi-level features sampled at the key points of anchors.
 *
 * @param features Flattened feature pyramids ``(batch, num_tokens, channels)``, float32.
 * @param spatial_shapes ``(cameras, levels, 2)`` heights and widths, int32.
 * @param level_start_index ``(cameras, levels)`` token offsets, int32.
 * @param sampling_locations ``(batch, anchors, points, cameras, 2)`` in [0, 1], float32.
 * @param weights ``(batch, anchors, points, cameras, levels, groups)``, float32.
 * @return Aggregated features ``(batch, anchors, channels)``, float32.
 */
at::Tensor deformable_aggregation_forward(
  const at::Tensor & features, const at::Tensor & spatial_shapes,
  const at::Tensor & level_start_index, const at::Tensor & sampling_locations,
  const at::Tensor & weights)
{
  check_inputs(features, spatial_shapes, level_start_index, sampling_locations, weights);
  at::Tensor output =
    at::zeros({features.size(0), sampling_locations.size(1), features.size(2)}, features.options());
#ifdef WITH_CUDA
  deformable_aggregation_forward_cuda(
    features, spatial_shapes, level_start_index, sampling_locations, weights, output);
  return output;
#else
  TORCH_CHECK(false, "deformable_aggregation_forward was compiled without CUDA");
#endif
}

/**
 * @brief Gradients of the aggregation with respect to the features, the locations and the weights.
 *
 * @param features Flattened feature pyramids ``(batch, num_tokens, channels)``, float32.
 * @param spatial_shapes ``(cameras, levels, 2)`` heights and widths, int32.
 * @param level_start_index ``(cameras, levels)`` token offsets, int32.
 * @param sampling_locations ``(batch, anchors, points, cameras, 2)`` in [0, 1], float32.
 * @param weights ``(batch, anchors, points, cameras, levels, groups)``, float32.
 * @param grad_output Gradient of the output ``(batch, anchors, channels)``, float32.
 * @return The gradients of ``features``, ``sampling_locations`` and ``weights``, in that order.
 */
std::vector<at::Tensor> deformable_aggregation_backward(
  const at::Tensor & features, const at::Tensor & spatial_shapes,
  const at::Tensor & level_start_index, const at::Tensor & sampling_locations,
  const at::Tensor & weights, const at::Tensor & grad_output)
{
  check_inputs(features, spatial_shapes, level_start_index, sampling_locations, weights);
  TORCH_CHECK(
    grad_output.sizes() ==
      at::IntArrayRef({features.size(0), sampling_locations.size(1), features.size(2)}),
    "grad_output must have shape (batch, anchors, channels)");
  TORCH_CHECK(grad_output.scalar_type() == at::ScalarType::Float, "grad_output must be float32");
  TORCH_CHECK(grad_output.is_contiguous(), "grad_output must be contiguous");
  TORCH_CHECK(
    grad_output.device() == features.device(), "grad_output must be on the features device");

  at::Tensor grad_features = at::zeros_like(features);
  at::Tensor grad_sampling_locations = at::zeros_like(sampling_locations);
  at::Tensor grad_weights = at::zeros_like(weights);
#ifdef WITH_CUDA
  deformable_aggregation_backward_cuda(
    features, spatial_shapes, level_start_index, sampling_locations, weights, grad_output,
    grad_features, grad_sampling_locations, grad_weights);
  return {grad_features, grad_sampling_locations, grad_weights};
#else
  TORCH_CHECK(false, "deformable_aggregation_backward was compiled without CUDA");
#endif
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.doc() = "Efficient deformable aggregation of Sparse4D";
  m.def(
    "deformable_aggregation_forward", &deformable_aggregation_forward,
    "Weighted sum of the multi-camera, multi-level features sampled at anchor key points (CUDA)",
    py::arg("features"), py::arg("spatial_shapes"), py::arg("level_start_index"),
    py::arg("sampling_locations"), py::arg("weights"));
  m.def(
    "deformable_aggregation_backward", &deformable_aggregation_backward,
    "Gradients of deformable_aggregation_forward (CUDA)", py::arg("features"),
    py::arg("spatial_shapes"), py::arg("level_start_index"), py::arg("sampling_locations"),
    py::arg("weights"), py::arg("grad_output"));
}
