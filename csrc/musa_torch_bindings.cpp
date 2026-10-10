#include "core/registration.h"

#include <torch/all.h>
#include <torch/library.h>

// MUSA: vllm._C carries only the regular-ABI CPU-to-device view op; the other
// _C ops are registered by _C_stable_libtorch.
torch::Tensor get_cuda_view_from_cpu_tensor(torch::Tensor& cpu_tensor);

TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, mcv) {
  mcv.impl("get_cuda_view_from_cpu_tensor", torch::kCPU,
           &get_cuda_view_from_cpu_tensor);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
