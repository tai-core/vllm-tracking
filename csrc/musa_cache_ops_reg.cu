// MUSA: regular-ABI registration of the _C_cache_ops library.
//
// The cache/MLA kernels in cache_kernels{,_fused}.cu take
// std::string kv_cache_dtype, which torch_musa's stable ABI cannot marshal
// (from_ivalue throws on StringType at the dispatch boundary). Register them
// through a regular at::Tensor TORCH_LIBRARY instead, where std::string passes
// natively. Each op is a thin wrapper that borrows the incoming at::Tensor as a
// torch::stable::Tensor (sharing storage, so in-place writes land on the
// caller's tensor) and calls the real kernel. This TU is compiled into the
// stable extension so the kernel symbols resolve.
#include <torch/all.h>
#include <torch/library.h>
#include <optional>
#include <string>

#include <torch/csrc/inductor/aoti_torch/utils.h>  // new_tensor_handle
#include <torch/csrc/stable/tensor.h>               // torch::stable::Tensor

namespace stable = torch::stable;

// Borrow an at::Tensor as a stable::Tensor. new_tensor_handle heap-allocates a
// new at::Tensor that shares the same TensorImpl/storage, and the stable::Tensor
// takes ownership of that heap copy (freed on scope exit). The underlying
// storage is shared, so writes through the stable handle modify the caller's
// tensor in place.
static inline stable::Tensor S(const at::Tensor& t) {
  return stable::Tensor(torch::aot_inductor::new_tensor_handle(at::Tensor(t)));
}
static inline std::optional<stable::Tensor> S(
    const std::optional<at::Tensor>& t) {
  if (!t.has_value()) return std::nullopt;
  return S(*t);
}

// Real stable-ABI kernels (defined in cache_kernels{,_fused}.cu, same extension).
void swap_blocks(stable::Tensor& src, stable::Tensor& dst,
                 int64_t block_size_in_bytes,
                 const stable::Tensor& block_mapping);
void swap_blocks_batch(const stable::Tensor& src_ptrs,
                       const stable::Tensor& dst_ptrs,
                       const stable::Tensor& sizes, bool is_src_access_order_any);
void reshape_and_cache(stable::Tensor& key, stable::Tensor& value,
                       stable::Tensor& key_cache, stable::Tensor& value_cache,
                       stable::Tensor& slot_mapping,
                       const std::string& kv_cache_dtype,
                       stable::Tensor& k_scale, stable::Tensor& v_scale);
void reshape_and_cache_flash(stable::Tensor& key, stable::Tensor& value,
                             stable::Tensor& key_cache,
                             stable::Tensor& value_cache,
                             stable::Tensor& slot_mapping,
                             const std::string& kv_cache_dtype,
                             stable::Tensor& k_scale, stable::Tensor& v_scale);
void concat_and_cache_mla(stable::Tensor& kv_c, stable::Tensor& k_pe,
                          stable::Tensor& kv_cache, stable::Tensor& slot_mapping,
                          const std::string& kv_cache_dtype,
                          stable::Tensor& scale);
void concat_and_cache_mla_grouped(stable::Tensor& kv_c, stable::Tensor& k_pe,
                                  stable::Tensor& kv_cache_ptrs,
                                  stable::Tensor& slot_mapping,
                                  int64_t block_size, int64_t block_stride,
                                  int64_t entry_stride,
                                  std::optional<stable::Tensor> kv_scales,
                                  const std::string& kv_cache_dtype);
void concat_and_cache_mla_rope_fused(
    stable::Tensor& positions, stable::Tensor& q_pe, stable::Tensor& k_pe,
    stable::Tensor& kv_c, stable::Tensor& rope_cos_sin_cache, bool rope_is_neox,
    stable::Tensor& slot_mapping, stable::Tensor& kv_cache,
    const std::string& kv_cache_dtype, stable::Tensor& kv_cache_quant_scale);
void convert_fp8(stable::Tensor& dst_cache, stable::Tensor& src_cache,
                 const double scale, const std::string& kv_cache_dtype);
void gather_and_maybe_dequant_cache(
    stable::Tensor const& src_cache, stable::Tensor const& dst,
    stable::Tensor const& block_table, stable::Tensor const& cu_seq_lens,
    stable::Tensor const& token_to_seq, int64_t num_tokens,
    const std::string& kv_cache_dtype, stable::Tensor const& scale,
    std::optional<stable::Tensor> seq_starts);
void cp_gather_cache(stable::Tensor const& src_cache, stable::Tensor const& dst,
                     stable::Tensor const& block_table,
                     stable::Tensor const& cu_seq_lens, int64_t batch_size,
                     std::optional<stable::Tensor> seq_starts);
void cp_gather_and_upconvert_fp8_kv_cache(
    stable::Tensor const& src_cache, stable::Tensor const& dst,
    stable::Tensor const& block_table, stable::Tensor const& workspace_starts,
    int64_t batch_size, std::optional<stable::Tensor> seq_starts,
    std::optional<stable::Tensor> host_cache,
    std::optional<stable::Tensor> host_row_ids,
    std::optional<stable::Tensor> device_row_ids);
void indexer_k_quant_and_cache(stable::Tensor& k, stable::Tensor& kv_cache,
                               stable::Tensor& slot_mapping,
                               int64_t quant_block_size,
                               const std::string& scale_fmt);
void concat_mla_q(stable::Tensor& ql_nope, stable::Tensor& q_pe,
                  stable::Tensor& q_out);
void cp_gather_indexer_k_quant_cache(stable::Tensor const& kv_cache,
                                     stable::Tensor& dst_k,
                                     stable::Tensor& dst_scale,
                                     stable::Tensor const& block_table,
                                     stable::Tensor const& cu_seq_lens);

// ---- at::Tensor bridge wrappers (regular ABI) ------------------------------
namespace musa_cache_bridge {

void swap_blocks(at::Tensor& src, at::Tensor& dst, int64_t n,
                 at::Tensor& block_mapping) {
  auto a = S(src), b = S(dst), m = S(block_mapping);
  ::swap_blocks(a, b, n, m);
}
void swap_blocks_batch(at::Tensor& src_ptrs, at::Tensor& dst_ptrs,
                       at::Tensor& sizes, bool any) {
  auto a = S(src_ptrs), b = S(dst_ptrs), c = S(sizes);
  ::swap_blocks_batch(a, b, c, any);
}
void reshape_and_cache(at::Tensor& key, at::Tensor& value, at::Tensor& kc,
                       at::Tensor& vc, at::Tensor& slot,
                       const std::string& dtype, at::Tensor& ks,
                       at::Tensor& vs) {
  auto a = S(key), b = S(value), c = S(kc), d = S(vc), e = S(slot), f = S(ks),
       g = S(vs);
  ::reshape_and_cache(a, b, c, d, e, dtype, f, g);
}
void reshape_and_cache_flash(at::Tensor& key, at::Tensor& value, at::Tensor& kc,
                             at::Tensor& vc, at::Tensor& slot,
                             const std::string& dtype, at::Tensor& ks,
                             at::Tensor& vs) {
  auto a = S(key), b = S(value), c = S(kc), d = S(vc), e = S(slot), f = S(ks),
       g = S(vs);
  ::reshape_and_cache_flash(a, b, c, d, e, dtype, f, g);
}
void concat_and_cache_mla(at::Tensor& kv_c, at::Tensor& k_pe, at::Tensor& cache,
                          at::Tensor& slot, const std::string& dtype,
                          at::Tensor& scale) {
  auto a = S(kv_c), b = S(k_pe), c = S(cache), d = S(slot), e = S(scale);
  ::concat_and_cache_mla(a, b, c, d, dtype, e);
}
void concat_and_cache_mla_grouped(at::Tensor& kv_c, at::Tensor& k_pe,
                                  at::Tensor& cache_ptrs, at::Tensor& slot,
                                  int64_t block_size, int64_t block_stride,
                                  int64_t entry_stride,
                                  const std::optional<at::Tensor>& kv_scales,
                                  const std::string& kv_cache_dtype) {
  auto a = S(kv_c), b = S(k_pe), c = S(cache_ptrs), d = S(slot);
  ::concat_and_cache_mla_grouped(a, b, c, d, block_size, block_stride,
                                 entry_stride, S(kv_scales), kv_cache_dtype);
}
void concat_and_cache_mla_rope_fused(at::Tensor& positions, at::Tensor& q_pe,
                                     at::Tensor& k_pe, at::Tensor& kv_c,
                                     at::Tensor& cos_sin, bool neox,
                                     at::Tensor& slot, at::Tensor& cache,
                                     const std::string& dtype,
                                     at::Tensor& scale) {
  auto a = S(positions), b = S(q_pe), c = S(k_pe), d = S(kv_c), e = S(cos_sin),
       f = S(slot), g = S(cache), h = S(scale);
  ::concat_and_cache_mla_rope_fused(a, b, c, d, e, neox, f, g, dtype, h);
}
void convert_fp8(at::Tensor& dst, at::Tensor& src, double scale,
                 const std::string& dtype) {
  auto a = S(dst), b = S(src);
  ::convert_fp8(a, b, scale, dtype);
}
void gather_and_maybe_dequant_cache(at::Tensor& src, at::Tensor& dst,
                                    at::Tensor& bt, at::Tensor& csl,
                                    at::Tensor& tts, int64_t num_tokens,
                                    const std::string& dtype, at::Tensor& scale,
                                    std::optional<at::Tensor> seq_starts) {
  ::gather_and_maybe_dequant_cache(S(src), S(dst), S(bt), S(csl), S(tts),
                                   num_tokens, dtype, S(scale), S(seq_starts));
}
void cp_gather_cache(at::Tensor& src, at::Tensor& dst, at::Tensor& bt,
                     at::Tensor& csl, int64_t bs,
                     std::optional<at::Tensor> seq_starts) {
  ::cp_gather_cache(S(src), S(dst), S(bt), S(csl), bs, S(seq_starts));
}
void cp_gather_and_upconvert_fp8_kv_cache(at::Tensor& src, at::Tensor& dst,
                                          at::Tensor& bt, at::Tensor& ws,
                                          int64_t bs,
                                          std::optional<at::Tensor> seq_starts,
                                          std::optional<at::Tensor> host_cache,
                                          std::optional<at::Tensor> host_row_ids,
                                          std::optional<at::Tensor> device_row_ids) {
  ::cp_gather_and_upconvert_fp8_kv_cache(
      S(src), S(dst), S(bt), S(ws), bs, S(seq_starts), S(host_cache),
      S(host_row_ids), S(device_row_ids));
}
void indexer_k_quant_and_cache(at::Tensor& k, at::Tensor& cache,
                               at::Tensor& slot, int64_t qbs,
                               const std::string& fmt) {
  auto a = S(k), b = S(cache), c = S(slot);
  ::indexer_k_quant_and_cache(a, b, c, qbs, fmt);
}
void concat_mla_q(at::Tensor& ql_nope, at::Tensor& q_pe, at::Tensor& q_out) {
  auto a = S(ql_nope), b = S(q_pe), c = S(q_out);
  ::concat_mla_q(a, b, c);
}
void cp_gather_indexer_k_quant_cache(at::Tensor& kv_cache, at::Tensor& dst_k,
                                     at::Tensor& dst_scale, at::Tensor& bt,
                                     at::Tensor& csl) {
  auto sdk = S(dst_k), sds = S(dst_scale);  // mutable refs need lvalues
  ::cp_gather_indexer_k_quant_cache(S(kv_cache), sdk, sds, S(bt), S(csl));
}

}  // namespace musa_cache_bridge

// On MUSA the stable STABLE_TORCH_LIBRARY_FRAGMENT(_C_cache_ops) def block and
// its CUDA/CPU TORCH_BOX impls are compiled out (#if !defined(USE_MUSA)) because
// the std::string args cannot cross the stable ABI. Own the whole _C_cache_ops
// library here with a regular TORCH_LIBRARY (schema + kernels), dispatching the
// device kernels under PrivateUse1 (MUSA's key).
namespace mcb = musa_cache_bridge;
TORCH_LIBRARY(_C_cache_ops, m) {
  m.def(
      "swap_blocks(Tensor src, Tensor! dst,"
      "            int block_size_in_bytes, Tensor block_mapping) -> ()");
  m.impl("swap_blocks", torch::kPrivateUse1, &mcb::swap_blocks);

  m.def(
      "swap_blocks_batch(Tensor src_ptrs, Tensor dst_ptrs,"
      "                  Tensor sizes,"
      "                  bool is_src_access_order_any=False) -> ()");
  m.impl("swap_blocks_batch", torch::kCPU, &mcb::swap_blocks_batch);

  m.def(
      "reshape_and_cache(Tensor key, Tensor value,"
      "                  Tensor! key_cache, Tensor! value_cache,"
      "                  Tensor slot_mapping,"
      "                  str kv_cache_dtype,"
      "                  Tensor k_scale, Tensor v_scale) -> ()");
  m.impl("reshape_and_cache", torch::kPrivateUse1, &mcb::reshape_and_cache);

  m.def(
      "reshape_and_cache_flash(Tensor key, Tensor value,"
      "                        Tensor! key_cache,"
      "                        Tensor! value_cache,"
      "                        Tensor slot_mapping,"
      "                        str kv_cache_dtype,"
      "                        Tensor k_scale, Tensor v_scale) -> ()");
  m.impl("reshape_and_cache_flash", torch::kPrivateUse1,
         &mcb::reshape_and_cache_flash);

  m.def(
      "concat_and_cache_mla(Tensor kv_c, Tensor k_pe,"
      "                     Tensor! kv_cache,"
      "                     Tensor slot_mapping,"
      "                     str kv_cache_dtype,"
      "                     Tensor scale) -> ()");
  m.impl("concat_and_cache_mla", torch::kPrivateUse1,
         &mcb::concat_and_cache_mla);

  m.def(
      "concat_and_cache_mla_grouped(Tensor kv_c, Tensor k_pe,"
      "                             Tensor kv_cache_ptrs,"
      "                             Tensor slot_mapping,"
      "                             int block_size, int block_stride,"
      "                             int entry_stride, Tensor? kv_scales=None,"
      "                             str kv_cache_dtype=\"auto\") -> ()");
  m.impl("concat_and_cache_mla_grouped", torch::kPrivateUse1,
         &mcb::concat_and_cache_mla_grouped);

  m.def(
      "concat_and_cache_mla_rope_fused("
      "                     Tensor positions,"
      "                     Tensor! q_pe,"
      "                     Tensor! k_pe,"
      "                     Tensor kv_c,"
      "                     Tensor cos_sin_cache,"
      "                     bool is_neox,"
      "                     Tensor slot_mapping,"
      "                     Tensor! kv_cache,"
      "                     str kv_cache_dtype,"
      "                     Tensor kv_cache_scale) -> ()");
  m.impl("concat_and_cache_mla_rope_fused", torch::kPrivateUse1,
         &mcb::concat_and_cache_mla_rope_fused);

  m.def(
      "convert_fp8(Tensor! dst_cache, Tensor src_cache, float scale, "
      "str kv_cache_dtype) -> ()");
  m.impl("convert_fp8", torch::kPrivateUse1, &mcb::convert_fp8);

  m.def(
      "gather_and_maybe_dequant_cache(Tensor src_cache, Tensor! dst, "
      "                               Tensor block_table, Tensor cu_seq_lens, "
      "                               Tensor token_to_seq, "
      "                               int num_tokens, "
      "                               str kv_cache_dtype, "
      "                               Tensor scale, Tensor? seq_starts) -> ()");
  m.impl("gather_and_maybe_dequant_cache", torch::kPrivateUse1,
         &mcb::gather_and_maybe_dequant_cache);

  m.def(
      "cp_gather_cache(Tensor src_cache, Tensor! dst, Tensor block_table, "
      "Tensor cu_seq_lens, int batch_size, Tensor? seq_starts) -> ()");
  m.impl("cp_gather_cache", torch::kPrivateUse1, &mcb::cp_gather_cache);

  m.def(
      "cp_gather_and_upconvert_fp8_kv_cache(Tensor src_cache, Tensor! dst, "
      "Tensor block_table, Tensor workspace_starts, int batch_size, Tensor? "
      "seq_starts, Tensor? host_cache=None, Tensor? host_row_ids=None, "
      "Tensor? device_row_ids=None) -> ()");
  m.impl("cp_gather_and_upconvert_fp8_kv_cache", torch::kPrivateUse1,
         &mcb::cp_gather_and_upconvert_fp8_kv_cache);

  m.def(
      "indexer_k_quant_and_cache(Tensor k, Tensor! kv_cache, Tensor "
      "slot_mapping, "
      "int quant_block_size, str kv_cache_dtype) -> ()");
  m.impl("indexer_k_quant_and_cache", torch::kPrivateUse1,
         &mcb::indexer_k_quant_and_cache);

  m.def("concat_mla_q(Tensor ql_nope, Tensor q_pe, Tensor! q_out) -> ()");
  m.impl("concat_mla_q", torch::kPrivateUse1, &mcb::concat_mla_q);

  m.def(
      "cp_gather_indexer_k_quant_cache(Tensor kv_cache, Tensor! dst_k, Tensor! "
      "dst_scale, Tensor block_table, Tensor cu_seq_lens) -> ()");
  m.impl("cp_gather_indexer_k_quant_cache", torch::kPrivateUse1,
         &mcb::cp_gather_indexer_k_quant_cache);
}

// merge_attn_states needs a MUSA registration: its stable CUDA impl is
// !USE_MUSA-guarded and the regular _C one is not registered. Its kernel lives
// in attention/merge_attn_states.cu (same extension), so register it here.
void merge_attn_states(at::Tensor& output, std::optional<at::Tensor> output_lse,
                       const at::Tensor& prefix_output,
                       const at::Tensor& prefix_lse,
                       const at::Tensor& suffix_output,
                       const at::Tensor& suffix_lse,
                       std::optional<int64_t> prefill_tokens_with_context,
                       const std::optional<at::Tensor>& output_scale);
TORCH_LIBRARY_IMPL(_C, PrivateUse1, m) {
  m.impl("merge_attn_states", &merge_attn_states);
}
