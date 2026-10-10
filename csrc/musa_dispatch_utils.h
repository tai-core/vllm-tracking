#pragma once

// MUSA: ATen dispatch macros for the regular-ABI kernels.
#include <torch/all.h>

// Nested FP8 dispatch names the dispatched type 'fp8_t' instead of 'scalar_t'.
#define AT_DISPATCH_FP8_CASE(enum_type, ...) \
  AT_PRIVATE_CASE_TYPE_USING_HINT(enum_type, fp8_t, __VA_ARGS__)

#define VLLM_DISPATCH_CASE_FP8_TYPES(...) \
  AT_DISPATCH_FP8_CASE(at::ScalarType::Float8_e4m3fn, __VA_ARGS__)

#define VLLM_DISPATCH_FP8_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, VLLM_DISPATCH_CASE_FP8_TYPES(__VA_ARGS__))

// Dispatch int32/int64 index tensors inside an outer value-type dispatch
// without shadowing the outer scalar_t alias.
#define VLLM_DISPATCH_IDX_CASE(enum_type, ...) \
  AT_PRIVATE_CASE_TYPE_USING_HINT(enum_type, idx_t, __VA_ARGS__)

#define VLLM_DISPATCH_CASE_IDX_TYPES(...)                  \
  VLLM_DISPATCH_IDX_CASE(at::ScalarType::Int, __VA_ARGS__) \
  VLLM_DISPATCH_IDX_CASE(at::ScalarType::Long, __VA_ARGS__)

#define VLLM_DISPATCH_IDX_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, VLLM_DISPATCH_CASE_IDX_TYPES(__VA_ARGS__))

#define VLLM_DISPATCH_CASE_FLOATING_TYPES(...)         \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#define VLLM_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, VLLM_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))

#define VLLM_DISPATCH_CASE_INTEGRAL_TYPES(...)         \
  AT_DISPATCH_CASE(at::ScalarType::Byte, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::Char, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::Short, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Int, __VA_ARGS__)   \
  AT_DISPATCH_CASE(at::ScalarType::Long, __VA_ARGS__)

#define VLLM_DISPATCH_CASE_INTEGRAL_AND_UNSIGNED_TYPES(...) \
  AT_DISPATCH_CASE(at::ScalarType::Byte, __VA_ARGS__)       \
  AT_DISPATCH_CASE(at::ScalarType::Char, __VA_ARGS__)       \
  AT_DISPATCH_CASE(at::ScalarType::Short, __VA_ARGS__)      \
  AT_DISPATCH_CASE(at::ScalarType::Int, __VA_ARGS__)        \
  AT_DISPATCH_CASE(at::ScalarType::Long, __VA_ARGS__)       \
  AT_DISPATCH_CASE(at::ScalarType::UInt16, __VA_ARGS__)     \
  AT_DISPATCH_CASE(at::ScalarType::UInt32, __VA_ARGS__)     \
  AT_DISPATCH_CASE(at::ScalarType::UInt64, __VA_ARGS__)

#define VLLM_DISPATCH_INTEGRAL_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, VLLM_DISPATCH_CASE_INTEGRAL_TYPES(__VA_ARGS__))

#define VLLM_DISPATCH_INTEGRAL_AND_UNSIGNED_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME,                                   \
                     VLLM_DISPATCH_CASE_INTEGRAL_AND_UNSIGNED_TYPES(__VA_ARGS__))
