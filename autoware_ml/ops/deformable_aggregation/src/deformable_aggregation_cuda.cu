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
// Adapted from projects/mmdet3d_plugin/ops/src/deformable_aggregation_cuda.cu of
// https://github.com/HorizonRobotics/Sparse4D (MIT License).
//
// Differences from the reference: the kernels run on the current CUDA stream under a device
// guard, the flat work index is 64-bit and walked with a grid-stride loop, the removed
// THC atomics header is gone (float atomicAdd is a CUDA builtin), and the launchers take
// tensors instead of raw pointers.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdint>

/**
 * @file deformable_aggregation_cuda.cu
 * @brief CUDA kernels of the efficient deformable aggregation of Sparse4D.
 *
 * Every anchor owns a few 3D key points. Each key point is projected into every camera and
 * sampled bilinearly from every level of that camera's feature pyramid; the samples are then
 * summed with learned weights over the points, the cameras and the levels. The reference
 * implementation is one grid_sample per level plus a weighted sum over a
 * (points, cameras, levels, channels) tensor per anchor; this kernel fuses the two and never
 * materializes the sampled tensor.
 *
 * Tensor layouts (all contiguous):
 *   features           (batch, num_tokens, channels): the feature pyramids of every camera,
 *                      flattened camera by camera, level by level, pixels row-major.
 *   spatial_shapes     (cameras, levels, 2): (height, width) of every level of every camera.
 *   level_start_index  (cameras, levels): token offset of every level inside ``features``.
 *   sampling_locations (batch, anchors, points, cameras, 2): (x, y) in [0, 1] of the image.
 *   weights            (batch, anchors, points, cameras, levels, groups): aggregation weights,
 *                      shared by the channels of a group.
 *   output             (batch, anchors, channels).
 *
 * A location on or outside the image border (x or y not strictly inside (0, 1)) contributes
 * nothing and receives no gradient. Inside the image the sample is the bilinear interpolation
 * of the four surrounding pixel centers, where pixel (i, j) is centered at
 * ((j + 0.5) / width, (i + 0.5) / height), with zero padding past the last centers. That is
 * the convention of ``torch.nn.functional.grid_sample`` with ``align_corners=False`` and
 * ``padding_mode="zeros"``.
 *
 * One thread handles one (batch, anchor, point, camera, level, channel) term and adds it to
 * the output with an atomic, so the summation order of the float32 terms is not fixed from
 * run to run.
 */

namespace
{
/// Number of threads per CUDA block.
constexpr int kThreadsPerBlock = 256;

/// Upper bound on the number of blocks; the grid-stride loop covers the remaining work.
constexpr int64_t kMaxBlocks = 65536;

/**
 * @brief Bilinearly sample one channel of a feature level at a pixel-space position.
 *
 * @param features Features of the sampled batch element, ``(num_tokens, channels)``.
 * @param height Height of the sampled level.
 * @param width Width of the sampled level.
 * @param num_channels Channels of every token.
 * @param y_pixel Row position, where row ``i`` is centered at ``i``.
 * @param x_pixel Column position, where column ``j`` is centered at ``j``.
 * @param base_offset Offset of the sampled channel of the first token of the level.
 * @return The interpolated value, zero padded past the last pixel centers.
 */
__device__ __forceinline__ float bilinear_sample(
  const float * __restrict__ features, const int height, const int width, const int num_channels,
  const float y_pixel, const float x_pixel, const int64_t base_offset)
{
  const int y_low = floorf(y_pixel);
  const int x_low = floorf(x_pixel);
  const int y_high = y_low + 1;
  const int x_high = x_low + 1;

  const float ly = y_pixel - y_low;
  const float lx = x_pixel - x_low;
  const float hy = 1.0f - ly;
  const float hx = 1.0f - lx;

  const int64_t x_stride = num_channels;
  const int64_t y_stride = static_cast<int64_t>(width) * x_stride;
  const int64_t y_low_offset = y_low * y_stride;
  const int64_t y_high_offset = y_low_offset + y_stride;
  const int64_t x_low_offset = x_low * x_stride;
  const int64_t x_high_offset = x_low_offset + x_stride;

  float v1 = 0.0f;
  if (y_low >= 0 && x_low >= 0) {
    v1 = features[y_low_offset + x_low_offset + base_offset];
  }
  float v2 = 0.0f;
  if (y_low >= 0 && x_high <= width - 1) {
    v2 = features[y_low_offset + x_high_offset + base_offset];
  }
  float v3 = 0.0f;
  if (y_high <= height - 1 && x_low >= 0) {
    v3 = features[y_high_offset + x_low_offset + base_offset];
  }
  float v4 = 0.0f;
  if (y_high <= height - 1 && x_high <= width - 1) {
    v4 = features[y_high_offset + x_high_offset + base_offset];
  }

  const float w1 = hy * hx;
  const float w2 = hy * lx;
  const float w3 = ly * hx;
  const float w4 = ly * lx;
  return w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
}

/**
 * @brief Back-propagate one sampled term to the features, the location and the weight.
 *
 * Mirrors :func:`bilinear_sample`. The gradient of the location is expressed per unit of
 * normalized image coordinate, hence the ``width`` and ``height`` factors.
 *
 * @param features Features of the sampled batch element, ``(num_tokens, channels)``.
 * @param weight Aggregation weight of the term.
 * @param height Height of the sampled level.
 * @param width Width of the sampled level.
 * @param num_channels Channels of every token.
 * @param y_pixel Row position of the sample.
 * @param x_pixel Column position of the sample.
 * @param base_offset Offset of the sampled channel of the first token of the level.
 * @param grad_output Gradient of the output channel the term adds to.
 * @param grad_features Gradient of ``features``, accumulated atomically.
 * @param grad_location Gradient of the ``(x, y)`` location of the term, accumulated atomically.
 * @param grad_weight Gradient of the weight of the term, accumulated atomically.
 */
__device__ __forceinline__ void bilinear_sample_grad(
  const float * __restrict__ features, const float weight, const int height, const int width,
  const int num_channels, const float y_pixel, const float x_pixel, const int64_t base_offset,
  const float grad_output, float * __restrict__ grad_features, float * __restrict__ grad_location,
  float * __restrict__ grad_weight)
{
  const int y_low = floorf(y_pixel);
  const int x_low = floorf(x_pixel);
  const int y_high = y_low + 1;
  const int x_high = x_low + 1;

  const float ly = y_pixel - y_low;
  const float lx = x_pixel - x_low;
  const float hy = 1.0f - ly;
  const float hx = 1.0f - lx;

  const int64_t x_stride = num_channels;
  const int64_t y_stride = static_cast<int64_t>(width) * x_stride;
  const int64_t y_low_offset = y_low * y_stride;
  const int64_t y_high_offset = y_low_offset + y_stride;
  const int64_t x_low_offset = x_low * x_stride;
  const int64_t x_high_offset = x_low_offset + x_stride;

  const float w1 = hy * hx;
  const float w2 = hy * lx;
  const float w3 = ly * hx;
  const float w4 = ly * lx;
  const float grad_sample = grad_output * weight;
  float grad_y = 0.0f;
  float grad_x = 0.0f;

  float v1 = 0.0f;
  if (y_low >= 0 && x_low >= 0) {
    const int64_t offset = y_low_offset + x_low_offset + base_offset;
    v1 = features[offset];
    grad_y -= hx * v1;
    grad_x -= hy * v1;
    atomicAdd(grad_features + offset, w1 * grad_sample);
  }
  float v2 = 0.0f;
  if (y_low >= 0 && x_high <= width - 1) {
    const int64_t offset = y_low_offset + x_high_offset + base_offset;
    v2 = features[offset];
    grad_y -= lx * v2;
    grad_x += hy * v2;
    atomicAdd(grad_features + offset, w2 * grad_sample);
  }
  float v3 = 0.0f;
  if (y_high <= height - 1 && x_low >= 0) {
    const int64_t offset = y_high_offset + x_low_offset + base_offset;
    v3 = features[offset];
    grad_y += hx * v3;
    grad_x -= ly * v3;
    atomicAdd(grad_features + offset, w3 * grad_sample);
  }
  float v4 = 0.0f;
  if (y_high <= height - 1 && x_high <= width - 1) {
    const int64_t offset = y_high_offset + x_high_offset + base_offset;
    v4 = features[offset];
    grad_y += lx * v4;
    grad_x += ly * v4;
    atomicAdd(grad_features + offset, w4 * grad_sample);
  }

  const float sample = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
  atomicAdd(grad_weight, grad_output * sample);
  atomicAdd(grad_location, width * grad_x * grad_sample);
  atomicAdd(grad_location + 1, height * grad_y * grad_sample);
}

/**
 * @brief Decomposed indices of one (batch, anchor, point, camera, level, channel) term.
 */
struct TermIndex
{
  int batch;
  int anchor;
  int point;
  int camera;
  int level;
  int channel;
  int64_t weight_offset;  ///< Offset of the weight of the term, shared by its channel group.
};

/**
 * @brief Decompose a flat work index, channel fastest, into the indices of its term.
 */
__device__ __forceinline__ TermIndex decompose(
  int64_t index, const int num_anchors, const int num_points, const int num_cams,
  const int num_levels, const int num_channels, const int group_channels)
{
  TermIndex term;
  term.weight_offset = index / group_channels;
  term.channel = index % num_channels;
  index /= num_channels;
  term.level = index % num_levels;
  index /= num_levels;
  term.camera = index % num_cams;
  index /= num_cams;
  term.point = index % num_points;
  index /= num_points;
  term.anchor = index % num_anchors;
  index /= num_anchors;
  term.batch = static_cast<int>(index);
  return term;
}

/**
 * @brief Forward kernel: every thread samples one term and adds it to its output channel.
 */
__global__ void deformable_aggregation_forward_kernel(
  const int64_t num_kernels, const float * __restrict__ features,
  const int * __restrict__ spatial_shapes, const int * __restrict__ level_start_index,
  const float * __restrict__ sampling_locations, const float * __restrict__ weights,
  const int num_tokens, const int num_channels, const int num_cams, const int num_levels,
  const int num_anchors, const int num_points, const int num_groups, float * __restrict__ output)
{
  const int group_channels = num_channels / num_groups;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < num_kernels; index += stride) {
    const TermIndex term =
      decompose(index, num_anchors, num_points, num_cams, num_levels, num_channels, group_channels);
    const int64_t batch_anchor = static_cast<int64_t>(term.batch) * num_anchors + term.anchor;
    const int64_t location_offset =
      ((batch_anchor * num_points + term.point) * num_cams + term.camera) * 2;

    const float x = sampling_locations[location_offset];
    if (x <= 0.0f || x >= 1.0f) {
      continue;
    }
    const float y = sampling_locations[location_offset + 1];
    if (y <= 0.0f || y >= 1.0f) {
      continue;
    }

    const int cam_level = term.camera * num_levels + term.level;
    const int height = spatial_shapes[cam_level * 2];
    const int width = spatial_shapes[cam_level * 2 + 1];
    const int64_t base_offset =
      (static_cast<int64_t>(term.batch) * num_tokens + level_start_index[cam_level]) *
        num_channels +
      term.channel;

    const float sample = bilinear_sample(
      features, height, width, num_channels, y * height - 0.5f, x * width - 0.5f, base_offset);
    atomicAdd(
      output + batch_anchor * num_channels + term.channel, weights[term.weight_offset] * sample);
  }
}

/**
 * @brief Backward kernel: every thread back-propagates one term.
 */
__global__ void deformable_aggregation_backward_kernel(
  const int64_t num_kernels, const float * __restrict__ features,
  const int * __restrict__ spatial_shapes, const int * __restrict__ level_start_index,
  const float * __restrict__ sampling_locations, const float * __restrict__ weights,
  const float * __restrict__ grad_output, const int num_tokens, const int num_channels,
  const int num_cams, const int num_levels, const int num_anchors, const int num_points,
  const int num_groups, float * __restrict__ grad_features,
  float * __restrict__ grad_sampling_locations, float * __restrict__ grad_weights)
{
  const int group_channels = num_channels / num_groups;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < num_kernels; index += stride) {
    const TermIndex term =
      decompose(index, num_anchors, num_points, num_cams, num_levels, num_channels, group_channels);
    const int64_t batch_anchor = static_cast<int64_t>(term.batch) * num_anchors + term.anchor;
    const int64_t location_offset =
      ((batch_anchor * num_points + term.point) * num_cams + term.camera) * 2;

    const float x = sampling_locations[location_offset];
    if (x <= 0.0f || x >= 1.0f) {
      continue;
    }
    const float y = sampling_locations[location_offset + 1];
    if (y <= 0.0f || y >= 1.0f) {
      continue;
    }

    const int cam_level = term.camera * num_levels + term.level;
    const int height = spatial_shapes[cam_level * 2];
    const int width = spatial_shapes[cam_level * 2 + 1];
    const int64_t base_offset =
      (static_cast<int64_t>(term.batch) * num_tokens + level_start_index[cam_level]) *
        num_channels +
      term.channel;

    bilinear_sample_grad(
      features, weights[term.weight_offset], height, width, num_channels, y * height - 0.5f,
      x * width - 0.5f, base_offset, grad_output[batch_anchor * num_channels + term.channel],
      grad_features, grad_sampling_locations + location_offset, grad_weights + term.weight_offset);
  }
}

/// Number of blocks covering ``num_kernels`` threads, capped so the grid-stride loop takes over.
int64_t num_blocks(const int64_t num_kernels)
{
  return std::min<int64_t>((num_kernels + kThreadsPerBlock - 1) / kThreadsPerBlock, kMaxBlocks);
}
}  // namespace

/**
 * @brief Launch the forward kernel. The inputs are validated by the caller.
 *
 * @param features Flattened feature pyramids ``(batch, num_tokens, channels)``, float32.
 * @param spatial_shapes ``(cameras, levels, 2)``, int32.
 * @param level_start_index ``(cameras, levels)``, int32.
 * @param sampling_locations ``(batch, anchors, points, cameras, 2)``, float32.
 * @param weights ``(batch, anchors, points, cameras, levels, groups)``, float32.
 * @param output Zero-initialized ``(batch, anchors, channels)``, float32, accumulated into.
 */
void deformable_aggregation_forward_cuda(
  const at::Tensor & features, const at::Tensor & spatial_shapes,
  const at::Tensor & level_start_index, const at::Tensor & sampling_locations,
  const at::Tensor & weights, at::Tensor & output)
{
  const at::cuda::CUDAGuard device_guard(features.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int num_tokens = features.size(1);
  const int num_channels = features.size(2);
  const int num_cams = spatial_shapes.size(0);
  const int num_levels = spatial_shapes.size(1);
  const int num_anchors = sampling_locations.size(1);
  const int num_points = sampling_locations.size(2);
  const int num_groups = weights.size(5);
  const int64_t num_kernels = weights.numel() / num_groups * num_channels;
  if (num_kernels == 0) {
    return;
  }

  deformable_aggregation_forward_kernel<<<num_blocks(num_kernels), kThreadsPerBlock, 0, stream>>>(
    num_kernels, features.data_ptr<float>(), spatial_shapes.data_ptr<int>(),
    level_start_index.data_ptr<int>(), sampling_locations.data_ptr<float>(),
    weights.data_ptr<float>(), num_tokens, num_channels, num_cams, num_levels, num_anchors,
    num_points, num_groups, output.data_ptr<float>());
  AT_CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Launch the backward kernel. The inputs are validated by the caller.
 *
 * @param features Flattened feature pyramids ``(batch, num_tokens, channels)``, float32.
 * @param spatial_shapes ``(cameras, levels, 2)``, int32.
 * @param level_start_index ``(cameras, levels)``, int32.
 * @param sampling_locations ``(batch, anchors, points, cameras, 2)``, float32.
 * @param weights ``(batch, anchors, points, cameras, levels, groups)``, float32.
 * @param grad_output Gradient of the output ``(batch, anchors, channels)``, float32.
 * @param grad_features Zero-initialized gradient of ``features``, accumulated into.
 * @param grad_sampling_locations Zero-initialized gradient of ``sampling_locations``.
 * @param grad_weights Zero-initialized gradient of ``weights``.
 */
void deformable_aggregation_backward_cuda(
  const at::Tensor & features, const at::Tensor & spatial_shapes,
  const at::Tensor & level_start_index, const at::Tensor & sampling_locations,
  const at::Tensor & weights, const at::Tensor & grad_output, at::Tensor & grad_features,
  at::Tensor & grad_sampling_locations, at::Tensor & grad_weights)
{
  const at::cuda::CUDAGuard device_guard(features.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int num_tokens = features.size(1);
  const int num_channels = features.size(2);
  const int num_cams = spatial_shapes.size(0);
  const int num_levels = spatial_shapes.size(1);
  const int num_anchors = sampling_locations.size(1);
  const int num_points = sampling_locations.size(2);
  const int num_groups = weights.size(5);
  const int64_t num_kernels = weights.numel() / num_groups * num_channels;
  if (num_kernels == 0) {
    return;
  }

  deformable_aggregation_backward_kernel<<<num_blocks(num_kernels), kThreadsPerBlock, 0, stream>>>(
    num_kernels, features.data_ptr<float>(), spatial_shapes.data_ptr<int>(),
    level_start_index.data_ptr<int>(), sampling_locations.data_ptr<float>(),
    weights.data_ptr<float>(), grad_output.data_ptr<float>(), num_tokens, num_channels, num_cams,
    num_levels, num_anchors, num_points, num_groups, grad_features.data_ptr<float>(),
    grad_sampling_locations.data_ptr<float>(), grad_weights.data_ptr<float>());
  AT_CUDA_CHECK(cudaGetLastError());
}
