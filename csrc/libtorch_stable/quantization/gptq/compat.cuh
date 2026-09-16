/*
Copied from https://github.com/turboderp/exllamav2
*/

#ifndef _compat_cuh
#define _compat_cuh

namespace vllm {
namespace gptq {
// atomicAdd for half types, to support CC < 7.x

__device__ __forceinline__ void atomicAdd_half(half* address, half val) {
  unsigned int* address_as_ui =
      (unsigned int*)((char*)address - ((size_t)address & 2));
  unsigned int old = *address_as_ui;
  unsigned int assumed;

  do {
    assumed = old;
    __half_raw hsum;
    hsum.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
    half tmpres = __hadd(hsum, val);
    hsum = __half_raw(tmpres);
    old = (size_t)address & 2 ? (old & 0xffff) | (hsum.x << 16)
                              : (old & 0xffff0000) | hsum.x;
    old = atomicCAS(address_as_ui, assumed, old);
  } while (assumed != old);
}

// atomicAdd for half2 types

__device__ __forceinline__ void atomicAdd_half2(half2* address, half2 val) {
  unsigned int* address_as_ui = (unsigned int*)address;
  unsigned int old = *address_as_ui;
  unsigned int assumed;
  do {
    assumed = old;
    half2 old_val = *((half2*)&old);
    half2 new_val = __hadd2(old_val, val);
    old = atomicCAS(address_as_ui, assumed, *((unsigned int*)&new_val));
  } while (assumed != old);
}

//
// GPTQ atomicAdd dispatch: q_gemm.cu calls gptq_atomicAdd BY NAME so the
// lookup is unambiguous. A HIP SDK may backport native half/half2
// atomicAdd before the version threshold below; with an UNQUALIFIED
// atomicAdd call, ordinary lookup + ADL could see both the SDK native
// overloads and the namespace fallbacks above. The named dispatchers
// below preserve the ORIGINAL guard semantics exactly:
//   - old-HIP (USE_ROCM, pre-713) and old-CUDA (CC < 700/600): CAS
//     fallback bodies (atomicAdd_half / atomicAdd_half2) — this is the
//     intended existing choice EVEN IF the SDK also provides native
//     overloads;
//   - modern HIP (>= 713) and modern CUDA (CC >= 700/600): the native
//     ::atomicAdd.
__device__ __forceinline__ void gptq_atomicAdd(half* address, half val) {
#if defined(__CUDA_ARCH__) || \
    (defined(USE_ROCM) && (HIP_VERSION_MAJOR * 100 + HIP_VERSION_MINOR) < 713)
  #if __CUDA_ARCH__ < 700 || defined(USE_ROCM)
  atomicAdd_half(address, val);
  return;
  #endif
#endif
  ::atomicAdd(address, val);
}

__device__ __forceinline__ void gptq_atomicAdd(half2* address, half2 val) {
#if defined(__CUDA_ARCH__) || \
    (defined(USE_ROCM) && (HIP_VERSION_MAJOR * 100 + HIP_VERSION_MINOR) < 713)
  #if __CUDA_ARCH__ < 600 || defined(USE_ROCM)
  atomicAdd_half2(address, val);
  return;
  #endif
#endif
  ::atomicAdd(address, val);
}

}  // namespace gptq
}  // namespace vllm
#endif
