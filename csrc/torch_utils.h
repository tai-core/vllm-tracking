#pragma once

#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/util/shim_utils.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <deque>
#include <mutex>
#include <string>
#include <vector>

#ifndef TORCH_UTILS_CHECK
#define TORCH_UTILS_CHECK STD_TORCH_CHECK
#endif

// Stable ABI equivalent of TORCH_CHECK_NOT_IMPLEMENTED.
#define STD_TORCH_CHECK_NOT_IMPLEMENTED(cond, ...) \
  STD_TORCH_CHECK(cond, "NotImplementedError: ", __VA_ARGS__)

// Device properties cache for stable ABI compatibility.
// Uses raw CUDA/HIP APIs instead of ATen functions.
// Using inline ensures a single instance across all translation units.
inline std::deque<std::once_flag> device_flags;
inline std::vector<musaDeviceProp> device_properties;
inline std::once_flag vectors_init_flag;

inline void do_init_device_vectors() {
  int device_count;
  musaError_t err = musaGetDeviceCount(&device_count);
  if (err != musaSuccess) {
    STD_TORCH_CHECK(false, "musaGetDeviceCount failed: " +
                               std::string(musaGetErrorString(err)));
  }
  device_flags.resize(device_count);
  device_properties.resize(device_count);
}

inline void initDeviceVectors() {
  std::call_once(vectors_init_flag, do_init_device_vectors);
}

inline void initDeviceProperty(int device_index) {
  musaDeviceProp device_prop{};
  musaError_t err = musaGetDeviceProperties(&device_prop, device_index);
  if (err != musaSuccess) {
    STD_TORCH_CHECK(false, "musaGetDeviceProperties failed: " +
                               std::string(musaGetErrorString(err)));
  }
  device_properties[device_index] = device_prop;
}

// Get device properties using raw CUDA/HIP APIs (stable ABI compatible).
// Caches results per device so musaGetDeviceProperties is called at most once
// per device.
inline musaDeviceProp* get_device_prop() {
  initDeviceVectors();
  int device_index;
  musaError_t err = musaGetDevice(&device_index);
  if (err != musaSuccess) {
    STD_TORCH_CHECK(
        false, "musaGetDevice failed: " + std::string(musaGetErrorString(err)));
  }
  STD_TORCH_CHECK(device_index >= 0 && static_cast<size_t>(device_index) <
                                           device_properties.size(),
                  "CUDA device index " + std::to_string(device_index) +
                      " out of range [0, " +
                      std::to_string(device_properties.size()) + ")");

  std::call_once(device_flags[device_index], initDeviceProperty, device_index);
  return &device_properties[device_index];
}

// Utility to get the current CUDA stream for a given device using stable APIs.
// Returns a musaStream_t for use in kernel launches.
inline musaStream_t get_current_cuda_stream(int32_t device_index = -1) {
  void* stream_ptr = nullptr;
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_get_current_musa_stream(device_index, &stream_ptr));
  return reinterpret_cast<musaStream_t>(stream_ptr);
}

// Utility to get the current cuBLAS handle using stable APIs.
inline mublasHandle_t get_current_cuda_blas_handle() {
  void* blas_handle_ptr = nullptr;
  TORCH_ERROR_CODE_CHECK(torch_get_current_cuda_blas_handle(&blas_handle_ptr));
  return reinterpret_cast<mublasHandle_t>(blas_handle_ptr);
}
