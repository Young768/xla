/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_SERVICE_GPU_KERNELS_CUTLASS_GEMM_H_
#define XLA_SERVICE_GPU_KERNELS_CUTLASS_GEMM_H_

#include <cstdint>
#include <optional>

namespace xla::gpu::kernel::gemm_universal {

// A mapping to get a Gemm kernel argument from a custom fusion parameters.
//
// Example:
//   se::KernelArgsDeviceMemoryArray args = ...
//   void* lhs = args->device_memory_ptr(indices.lhs);
//
// Custom fusion instruction can have parameters in arbitrary order, and we need
// a mapping from a custom kernel argument to the fusion instruction parameter.
struct ArgsIndices {
  int64_t lhs;
  int64_t rhs;
  int64_t out;
};

// TODO(ezhulenev): Support dynamic slices along all dimensions, today we assume
// that we can slice only along the leading dimension (batch).

// A mapping to get Gemm kernel dynamic slice arguments from a custom fusion
// parameters. Dynamic slices are optional, and by default Gemm kernel uses
// pointers defined by `ArgsIndices`.
struct DynamicSliceIndices {
  std::optional<int64_t> out;
};

// A structure to pass pointers to buffers with dynamic slice parameters to a
// device kernel, so that we can do address computation on device.
struct DynamicSliceParams {
  std::optional<int32_t*> out;
};

}  // namespace xla::gpu::kernel::gemm_universal

#endif  // XLA_SERVICE_GPU_KERNELS_CUTLASS_GEMM_H_