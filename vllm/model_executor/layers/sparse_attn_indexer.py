# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom Sparse Attention Indexer layers."""

import dataclasses
import os

import torch
import torch.distributed as dist

import vllm.envs as envs
from vllm_musa import _custom_ops as _musa_custom_ops
from vllm import _custom_ops as ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import (
    CUDAGraphMode,
    get_current_vllm_config,
    get_current_vllm_config_or_none,
)
from vllm.distributed import (
    get_dcp_group,
    get_pcp_group,
    get_tensor_model_parallel_rank,
    get_tp_group,
)
from vllm_musa.optimization_contract import (
    deepseek_v4_long_prefill_logits_budget_mb,
    deepseek_v4_long_prefill_tp_partition_min_seq_len,
)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.kernels.attention.dsa.candidate_blocks import (
    apply_candidate_mask as _apply_candidate_mask,
)
from vllm.model_executor.kernels.attention.dsa.candidate_blocks import (
    select_candidate_blocks as _select_candidate_blocks,
)
from vllm.model_executor.layers.indexer_topk import (
    RADIX_TOPK_WORKSPACE_SIZE,
    get_indexer_topk,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.deep_gemm import (
    fp8_fp4_mqa_logits,
    fp8_fp4_paged_mqa_logits,
    has_deep_gemm,
)
from vllm.utils.import_utils import has_cutedsl
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadata,
    tp_prefill_row_sharding_supported,
)
from vllm.v1.attention.ops.common import pack_seq_triton, unpack_seq_triton
from vllm.v1.attention.ops.pcp import maybe_gather_indexer_k
from vllm.v1.worker.workspace import current_workspace_manager

logger = init_logger(__name__)


def _musa_sparse_indexer_is_current_stream_capturing() -> bool:
    for module_name in ("musa", "cuda"):
        module = getattr(torch, module_name, None)
        if module is None:
            continue
        is_capturing = getattr(module, "is_current_stream_capturing", None)
        if is_capturing is None:
            continue
        try:
            return bool(is_capturing())
        except Exception:
            continue
    return False


def _musa_decode_seq_len_fits_native_contract(
    decode_metadata,
    kv_cache: torch.Tensor,
    limit: int,
) -> bool:
    """Eager uses request length. Capture never host-syncs.

    The native kernel clamps to its compile-time max, so capture always
    selects it once the dtype and shape gates have passed.
    """
    if _musa_sparse_indexer_is_current_stream_capturing():
        return True
    seq_lens = getattr(decode_metadata, "seq_lens", None)
    if seq_lens is None:
        capacity = int(decode_metadata.block_table.shape[1]) * int(kv_cache.shape[1])
        return capacity <= limit
    try:
        return int(seq_lens.reshape(-1).max().item()) <= limit
    except (RuntimeError, ValueError, TypeError):
        capacity = int(decode_metadata.block_table.shape[1]) * int(kv_cache.shape[1])
        return capacity <= limit


def _musa_sparse_indexer_native_decode_enabled() -> bool:
    # The native DeepSeek-V4 indexer implementation is the validated MUSA
    # path. Keep it shape-gated at the call sites, but do not make production
    # dispatch depend on a process-wide environment variable.
    return True


def _musa_sparse_indexer_glm52_native_enabled() -> bool:
    return os.getenv("VLLM_MUSA_GLM52_INDEXER_TOPK_NATIVE", "1") == "1"


def _musa_sparse_indexer_glm52_fused_enabled() -> bool:
    # The single-block fused scorer is useful for correctness experiments but
    # slower than the materialized MQA scorer on S5000 for rows above top-k.
    return os.getenv("VLLM_MUSA_GLM52_INDEXER_TOPK_FUSED", "0") == "1"


def _musa_sparse_indexer_glm52_materialized_enabled() -> bool:
    return os.getenv("VLLM_MUSA_GLM52_INDEXER_TOPK_MATERIALIZED", "1") == "1"


def _musa_sparse_indexer_materialized_prefill_enabled() -> bool:
    return True


def _musa_sparse_indexer_materialized_prefill_overselect(
    topk: int,
    total_seq_lens: int,
) -> int:
    width = 640
    width = max(width, int(topk))
    return max(0, min(width, int(total_seq_lens)))


def _musa_sparse_indexer_materialized_prefill_chunk_rows(
    rows: int,
    total_seq_lens: int,
    is_deepseek_v4: bool = False,
) -> int:
    """Bound each FP32 logits tile by the configured indexer budget."""
    chunk_rows = 512
    logits_budget_mb = (
        deepseek_v4_long_prefill_logits_budget_mb()
        if is_deepseek_v4
        else int(envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB)
    )
    max_logits_elems = max(1, logits_budget_mb) * 1024 * 1024 // 4
    rows_for_budget = max(1, max_logits_elems // max(1, int(total_seq_lens)))
    chunk_rows = min(chunk_rows, rows_for_budget)
    return max(1, min(chunk_rows, int(rows)))


def _musa_sparse_indexer_materialized_prefill_topk_sorted() -> bool:
    return False


def _musa_sparse_indexer_materialized_prefill_direct_topk_enabled() -> bool:
    return True


def _musa_decode_block_table_for_token_rows(
    block_table: torch.Tensor,
    rows: int,
) -> torch.Tensor | None:
    block_rows = block_table.shape[0]
    if rows <= 0:
        return block_table[:0]
    if block_rows == rows:
        return block_table[:rows]
    if block_rows <= 0 or rows % block_rows != 0:
        return None
    return block_table.repeat_interleave(rows // block_rows, dim=0)


@triton.jit
def _musa_paged_mqa_logits_kernel(
    q_ptr,
    q_stride0,
    q_stride1,
    kv_ptr,
    kv_block_stride,
    num_blocks,
    weights_ptr,
    weights_stride0,
    weights_stride1,
    seq_lens_ptr,
    seq_lens_stride,
    block_table_ptr,
    block_table_stride0,
    logits_ptr,
    logits_stride0,
    width,
    BLOCK_SIZE: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # One program scores BLOCK_N cached keys of one decode row. Each cache
    # block holds BLOCK_SIZE x HEAD_DIM fp8 keys followed by BLOCK_SIZE fp32
    # scales. Rows read their length on device, so captured graphs replay it.
    row = tl.program_id(0)
    start = tl.program_id(1) * BLOCK_N
    seq_len = tl.minimum(tl.load(seq_lens_ptr + row * seq_lens_stride), width)
    if start >= seq_len:
        return
    offs_n = start + tl.arange(0, BLOCK_N)
    in_row = offs_n < seq_len
    pos = offs_n % BLOCK_SIZE
    block = tl.load(
        block_table_ptr + row * block_table_stride0 + offs_n // BLOCK_SIZE,
        mask=in_row,
        other=-1,
    )
    valid = in_row & (block >= 0) & (block < num_blocks)
    block_base = kv_ptr + tl.where(valid, block, 0).to(tl.int64) * kv_block_stride
    offs_d = tl.arange(0, HEAD_DIM)
    k = tl.load(
        block_base[:, None] + pos[:, None] * HEAD_DIM + offs_d[None, :],
        mask=valid[:, None],
        other=0,
    )
    k = k.to(tl.float8e4nv, bitcast=True)
    k_scale = tl.load(
        (block_base + BLOCK_SIZE * HEAD_DIM + pos * 4).to(tl.pointer_type(tl.float32)),
        mask=valid,
        other=0.0,
    )
    if NUM_HEADS >= 16:
        offs_h = tl.arange(0, NUM_HEADS)
        q = tl.load(q_ptr + row * q_stride0 + offs_h[None, :] * q_stride1 + offs_d[:, None])
        # MUSA: dot on the FP8 operands; converting them to BF16 first scores wrongly.
        scores = tl.dot(k, q)
        w = tl.load(weights_ptr + row * weights_stride0 + offs_h * weights_stride1)
        logits = tl.sum(tl.maximum(scores, 0.0) * w[None, :], axis=1)
    else:
        k = k.to(tl.float32)
        logits = tl.zeros((BLOCK_N,), dtype=tl.float32)
        q_row = q_ptr + row * q_stride0
        for h in tl.static_range(NUM_HEADS):
            q_h = tl.load(q_row + h * q_stride1 + offs_d).to(tl.float32)
            w_h = tl.load(weights_ptr + row * weights_stride0 + h * weights_stride1)
            logits += tl.maximum(tl.sum(k * q_h[None, :], axis=1), 0.0) * w_h
    logits = tl.where(valid, logits * k_scale, float("-inf"))
    tl.store(logits_ptr + row * logits_stride0 + offs_n, logits, mask=in_row)


def _musa_select_indexer_topk(
    logits: torch.Tensor,
    row_lens: torch.Tensor,
    topk_indices: torch.Tensor,
    topk: int,
) -> None:
    """Fill each row's top-k index set over its first row_lens logits."""
    if topk <= 512:
        # MUSA: the staged radix select returns the same set and order.
        _musa_custom_ops.deepseek_v4_sparse_indexer_topk_decode(
            logits, row_lens, topk_indices, topk
        )
        return
    _musa_custom_ops.sparse_indexer_topk_decode(logits, row_lens, topk_indices, topk)


def _musa_indexer_windows_start_at_zero(row_starts: torch.Tensor, topk: int) -> bool:
    """Whether eager windowed top-k rows can use the row-length select."""
    return (
        topk <= 512
        and not _musa_sparse_indexer_is_current_stream_capturing()
        and not bool(row_starts.any())
    )


def _musa_select_indexer_topk_windows(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    topk_indices: torch.Tensor,
    topk: int,
    starts_at_zero: bool,
) -> None:
    """Fill window-local top-k index sets over [row_starts, row_ends)."""
    if starts_at_zero:
        _musa_select_indexer_topk(logits, row_ends, topk_indices, topk)
        return
    _musa_custom_ops.sparse_indexer_topk(
        logits, row_starts, row_ends, topk_indices, topk
    )


def _musa_try_fill_decode_topk_from_paged_logits(
    q_quant: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    decode_metadata,
    topk_indices_buffer: torch.Tensor,
    topk_tokens: int,
    head_dim: int,
) -> bool:
    """Graph-safe FP8 decode top-k for any indexer cache block size."""
    if (
        q_quant.ndim != 3
        or q_quant.dtype != torch.float8_e4m3fn
        or q_quant.shape[2] != head_dim
        or q_quant.stride(2) != 1
        or head_dim & (head_dim - 1)
        or kv_cache.dtype != torch.uint8
        or kv_cache.ndim != 3
        or kv_cache.shape[2] != head_dim + 4
        or weights.dtype != torch.float32
        or weights.ndim != 2
        or topk_tokens > 2048
    ):
        return False

    seq_lens = decode_metadata.seq_lens.reshape(-1)
    rows = min(q_quant.shape[0], seq_lens.numel(), topk_indices_buffer.shape[0])
    topk = min(int(topk_tokens), topk_indices_buffer.shape[1])
    if rows <= 0 or topk <= 0:
        return True
    block_table = _musa_decode_block_table_for_token_rows(
        decode_metadata.block_table,
        rows,
    )
    if block_table is None:
        return False

    block_size = kv_cache.shape[1]
    width = block_table.shape[1] * block_size
    logits = torch.empty((rows, width), dtype=torch.float32, device=q_quant.device)
    block_n = 64
    _musa_paged_mqa_logits_kernel[(rows, triton.cdiv(width, block_n))](
        q_quant,
        q_quant.stride(0),
        q_quant.stride(1),
        kv_cache,
        kv_cache.stride(0),
        kv_cache.shape[0],
        weights,
        weights.stride(0),
        weights.stride(1),
        seq_lens,
        seq_lens.stride(0),
        block_table,
        block_table.stride(0),
        logits,
        logits.stride(0),
        width,
        BLOCK_SIZE=block_size,
        NUM_HEADS=q_quant.shape[1],
        HEAD_DIM=head_dim,
        BLOCK_N=block_n,
    )
    _musa_select_indexer_topk(
        logits,
        seq_lens[:rows],
        topk_indices_buffer[:rows, :topk],
        topk,
    )
    return True


def _musa_try_fill_all_sparse_indexer_indices(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    topk_tokens: int,
    topk_indices_buffer: torch.Tensor,
) -> bool:
    """Skip indexer scoring when every causal window fits in top-k.

    This is the production form of the GLM-5.2 dashboard SOTA's full-row
    shortcut: if all valid keys are selected, their scores cannot affect the
    selected set.  Fill the contiguous local indices directly and avoid KV
    gather, dequantization, einsum, and top-k entirely.
    """
    attn_metadata = get_forward_context().attn_metadata
    if not isinstance(attn_metadata, dict):
        return False

    metadata = attn_metadata.get(_resolve_layer_name(k_cache_prefix))
    if not isinstance(metadata, DeepseekV32IndexerMetadata):
        return False
    if int(metadata.max_seq_len) > int(topk_tokens):
        return False

    topk = min(int(topk_tokens), topk_indices_buffer.shape[1])
    topk_indices_buffer[: hidden_states.shape[0], :topk] = -1
    if topk <= 0:
        return True

    if metadata.num_decodes > 0 and metadata.decode is not None:
        lengths = metadata.decode.seq_lens.reshape(-1)
        rows = min(
            int(metadata.num_decode_tokens),
            lengths.numel(),
            topk_indices_buffer.shape[0],
        )
        if rows > 0:
            _musa_custom_ops.sparse_indexer_fill_all(
                lengths[:rows],
                topk_indices_buffer[:rows, :topk],
                topk,
            )

    if metadata.num_prefills > 0 and metadata.prefill is not None:
        for chunk in metadata.prefill.chunks:
            token_start = int(chunk.token_start)
            token_end = min(
                int(chunk.token_end),
                topk_indices_buffer.shape[0],
            )
            rows = max(0, token_end - token_start)
            if rows <= 0:
                continue
            lengths = (
                chunk.cu_seqlen_ke[:rows] - chunk.cu_seqlen_ks[:rows]
            ).contiguous()
            _musa_custom_ops.sparse_indexer_fill_all(
                lengths,
                topk_indices_buffer[token_start:token_end, :topk],
                topk,
            )

    return True


def _musa_try_fill_decode_topk_from_materialized_logits(
    q_quant: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    decode_metadata,
    topk_indices: torch.Tensor,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
) -> bool:
    is_glm52 = (
        q_quant.ndim == 3
        and q_quant.shape[1] == 32
        and topk_tokens <= 2048
    )
    is_deepseek_v4 = (
        q_quant.ndim == 3
        and q_quant.shape[1] == 64
        and topk_tokens <= 512
    )
    if (
        not (
            (is_glm52 and _musa_sparse_indexer_glm52_materialized_enabled())
            or is_deepseek_v4
        )
        or head_dim != 128
        or q_quant.dtype != torch.float8_e4m3fn
        or kv_cache.dtype != torch.uint8
        or weights.dtype != torch.float32
        or kv_cache.ndim != 3
        or decode_metadata.schedule_metadata is None
    ):
        return False

    context_lens = decode_metadata.seq_lens
    if context_lens.ndim == 1:
        context_lens = context_lens.reshape(-1, 1)
    if context_lens.ndim != 2:
        return False
    batch_size, next_n = context_lens.shape
    rows = min(q_quant.shape[0], context_lens.numel(), topk_indices.shape[0])
    topk = min(int(topk_tokens), topk_indices.shape[1])
    page_capacity = int(decode_metadata.block_table.shape[1]) * int(kv_cache.shape[1])
    materialized_width = max(int(topk), min(int(max_model_len), page_capacity))
    if rows <= 0 or topk <= 0:
        return True
    if materialized_width <= topk:
        return False
    if rows != batch_size * next_n:
        return False

    context_lens = context_lens.contiguous()
    if context_lens.dtype != torch.int32:
        context_lens = context_lens.to(torch.int32)
    seq_lens = context_lens.reshape(-1)
    block_table = decode_metadata.block_table[:batch_size].contiguous()
    q_native = q_quant[:rows].reshape(
        batch_size,
        next_n,
        q_quant.shape[1],
        q_quant.shape[2],
    )
    kv_paged = kv_cache.unsqueeze(-2)

    def _select(logits: torch.Tensor) -> bool:
        if logits.shape[0] < rows or logits.shape[1] < materialized_width:
            return False
        _musa_select_indexer_topk(
            logits[:rows, :materialized_width],
            seq_lens,
            topk_indices[:rows, :topk],
            topk,
        )
        return True

    for provider_name in ("mate.deep_gemm", "deep_gemm"):
        try:
            if provider_name == "mate.deep_gemm":
                from mate import deep_gemm as _musa_deep_gemm
            else:
                import deep_gemm as _musa_deep_gemm

            logits = _musa_deep_gemm.fp8_paged_mqa_logits(
                q_native,
                kv_paged,
                weights[:rows].contiguous(),
                context_lens,
                block_table,
                decode_metadata.schedule_metadata,
                materialized_width,
                False,
            )
            if _select(logits):
                return True
        except Exception:
            pass

    try:
        from vllm.utils import deep_gemm as _vllm_deep_gemm

        logits = _vllm_deep_gemm.fp8_fp4_paged_mqa_logits(
            (q_native, None),
            kv_paged,
            weights[:rows].contiguous(),
            context_lens,
            block_table,
            schedule_metadata=decode_metadata.schedule_metadata,
            max_model_len=materialized_width,
            clean_logits=False,
        )
        return _select(logits)
    except Exception:
        return False


def _musa_try_fill_decode_topk_from_indexer_cache_native(
    q_quant: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    decode_metadata,
    topk_indices_buffer: torch.Tensor,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    allow_materialized: bool = True,
) -> bool:
    if allow_materialized and _musa_try_fill_decode_topk_from_materialized_logits(
        q_quant,
        kv_cache,
        weights,
        decode_metadata,
        topk_indices_buffer,
        topk_tokens,
        head_dim,
        max_model_len,
    ):
        return True

    if (
        _musa_sparse_indexer_glm52_fused_enabled()
        and head_dim == 128
        and q_quant.ndim == 3
        and q_quant.shape[1] == 32
        and topk_tokens <= 2048
        and q_quant.dtype == torch.float8_e4m3fn
        and kv_cache.dtype == torch.uint8
        and weights.dtype == torch.float32
        and _musa_decode_seq_len_fits_native_contract(
            decode_metadata, kv_cache, 8192
        )
    ):
        seq_lens = decode_metadata.seq_lens.reshape(-1)
        rows = min(
            q_quant.shape[0],
            seq_lens.numel(),
            topk_indices_buffer.shape[0],
        )
        topk = min(int(topk_tokens), topk_indices_buffer.shape[1])
        if rows <= 0 or topk <= 0:
            return True
        block_table = _musa_decode_block_table_for_token_rows(
            decode_metadata.block_table,
            rows,
        )
        if block_table is None:
            return False
        _musa_custom_ops.glm52_indexer_topk_decode(
            q_quant[:rows],
            kv_cache,
            weights[:rows],
            seq_lens[:rows],
            block_table,
            topk_indices_buffer[:rows, :topk],
            topk,
        )
        return True

    if (
        not _musa_sparse_indexer_native_decode_enabled()
        or head_dim != 128
        or q_quant.ndim != 3
        or q_quant.shape[1] != 64
        or topk_tokens > 512
        or q_quant.dtype != torch.float8_e4m3fn
        or kv_cache.dtype != torch.uint8
        or weights.dtype != torch.float32
        or not _musa_decode_seq_len_fits_native_contract(
            decode_metadata, kv_cache, 4096
        )
    ):
        return _musa_try_fill_decode_topk_from_paged_logits(
            q_quant,
            kv_cache,
            weights,
            decode_metadata,
            topk_indices_buffer,
            topk_tokens,
            head_dim,
        )

    seq_lens = decode_metadata.seq_lens.reshape(-1)
    rows = min(q_quant.shape[0], seq_lens.numel(), topk_indices_buffer.shape[0])
    topk = min(int(topk_tokens), topk_indices_buffer.shape[1])
    if rows <= 0 or topk <= 0:
        return True

    block_table = _musa_decode_block_table_for_token_rows(
        decode_metadata.block_table,
        rows,
    )
    if block_table is None:
        return False

    _musa_custom_ops.deepseek_v4_indexer_topk_decode(
        q_quant[:rows],
        kv_cache,
        weights[:rows],
        seq_lens[:rows],
        block_table,
        topk_indices_buffer[:rows, :topk],
        topk,
    )
    return True


def _musa_try_fill_prefill_topk_from_materialized_logits(
    q_quant: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    chunk,
    topk_indices: torch.Tensor,
    topk_tokens: int,
    head_dim: int,
    allow_deepseek_v4: bool = False,
) -> bool:
    is_glm52 = (
        q_quant.ndim == 3
        and q_quant.shape[1] == 32
        and topk_tokens <= 2048
    )
    is_deepseek_v4 = (
        q_quant.ndim == 3
        and q_quant.shape[1] == 64
        and topk_tokens <= 512
    )
    implementation_enabled = (
        is_glm52 and _musa_sparse_indexer_glm52_materialized_enabled()
    ) or (
        is_deepseek_v4
        and allow_deepseek_v4
        and _musa_sparse_indexer_materialized_prefill_enabled()
    )
    if (
        not implementation_enabled
        or head_dim != 128
        or q_quant.dtype != torch.float8_e4m3fn
        or kv_cache.dtype != torch.uint8
        or weights.dtype != torch.float32
        or kv_cache.ndim != 3
        or int(chunk.num_reqs) != 1
    ):
        return False

    rows = min(q_quant.shape[0], chunk.cu_seqlen_ks.numel(), topk_indices.shape[0])
    topk = min(int(topk_tokens), topk_indices.shape[1])
    total_seq_lens = int(chunk.total_seq_lens)
    overselect = _musa_sparse_indexer_materialized_prefill_overselect(
        topk,
        total_seq_lens,
    )
    if rows <= 0 or topk <= 0:
        return True
    if total_seq_lens <= 0 or overselect <= 0:
        return True

    block_size = int(kv_cache.shape[1])
    if block_size <= 0:
        return False

    use_bounded_long_prefill = is_deepseek_v4 and total_seq_lens > 4096
    paged_block_size = block_size
    paged_kv_cache = kv_cache
    paged_block_table = chunk.block_table[:1]
    if use_bounded_long_prefill:
        # MUSA: paged-MQA must see live compressed pages, not max_model_len.
        needed_src_pages = (int(total_seq_lens) + block_size - 1) // block_size
        paged_block_table = paged_block_table[:, : max(1, needed_src_pages)].contiguous()
    if use_bounded_long_prefill and block_size != 64:
        if block_size % 64 != 0:
            return False
        pages_per_block = block_size // 64
        paged_block_size = 64
        paged_kv_cache = kv_cache.view(-1, 64, kv_cache.shape[-1])
        page_offsets = torch.arange(
            pages_per_block,
            device=paged_block_table.device,
            dtype=paged_block_table.dtype,
        )
        original_pages = paged_block_table.unsqueeze(-1)
        paged_block_table = torch.where(
            original_pages >= 0,
            original_pages * pages_per_block + page_offsets,
            torch.full_like(original_pages + page_offsets, -1),
        ).flatten(1)

    row_starts = chunk.cu_seqlen_ks[:rows].clamp(min=0, max=total_seq_lens)
    row_ends = chunk.cu_seqlen_ke[:rows].clamp(min=0, max=total_seq_lens)
    # Paged-MQA logits use absolute prefix coordinates; row_starts is applied
    # below as a mask over the materialized logits.
    context_lens = row_ends.reshape(rows, 1).contiguous()
    if context_lens.dtype != torch.int32:
        context_lens = context_lens.to(torch.int32)

    kv_paged = paged_kv_cache.unsqueeze(-2)
    row_starts = row_starts.to(torch.long)
    row_ends = row_ends.to(torch.long)
    positions = (
        None
        if use_bounded_long_prefill
        else torch.arange(total_seq_lens, device=q_quant.device, dtype=torch.long)
    )

    work_q = q_quant[:rows]
    work_weights = weights[:rows]
    work_starts = row_starts
    work_ends = row_ends
    work_context_lens = context_lens
    work_output = topk_indices[:rows, :topk]
    work_rows = rows
    tp_group = get_tp_group()
    tp_world_size = int(tp_group.world_size)
    # MUSA: Gloo D2H+CPU gather only pays off past ~64k compressed keys.
    tp_partition_rows = (
        use_bounded_long_prefill
        and tp_world_size > 1
        and int(total_seq_lens)
        > deepseek_v4_long_prefill_tp_partition_min_seq_len()
    )
    tp_rows_per_rank = 0
    if tp_partition_rows:
        tp_rows_per_rank = (rows + tp_world_size - 1) // tp_world_size
        local_row_start = min(
            int(tp_group.rank_in_group) * tp_rows_per_rank,
            rows,
        )
        local_row_end = min(local_row_start + tp_rows_per_rank, rows)
        local_rows = local_row_end - local_row_start

        def _musa_slice_and_pad_rows(tensor: torch.Tensor) -> torch.Tensor:
            local = tensor[local_row_start:local_row_end]
            if local_rows == tp_rows_per_rank:
                return local
            padded = torch.empty(
                (tp_rows_per_rank, *tensor.shape[1:]),
                device=tensor.device,
                dtype=tensor.dtype,
            )
            if local_rows > 0:
                padded[:local_rows].copy_(local)
            padded[local_rows:].copy_(
                tensor[rows - 1 : rows].expand(
                    tp_rows_per_rank - local_rows,
                    *tensor.shape[1:],
                )
            )
            return padded

        work_q = _musa_slice_and_pad_rows(q_quant)
        work_weights = _musa_slice_and_pad_rows(weights)
        work_starts = _musa_slice_and_pad_rows(row_starts)
        work_ends = _musa_slice_and_pad_rows(row_ends)
        work_context_lens = _musa_slice_and_pad_rows(context_lens)
        work_output = torch.empty(
            tp_rows_per_rank,
            topk,
            device=topk_indices.device,
            dtype=topk_indices.dtype,
        )
        work_rows = tp_rows_per_rank

    def _musa_finish_materialized_prefill() -> bool:
        if not tp_partition_rows:
            return True
        # MUSA: gather compact top-k indices over Gloo. Device all_gather
        # hangs on this long-prefill path even when every rank participates.
        cpu_work = work_output.detach().to("cpu").contiguous()
        gathered_parts = [
            torch.empty_like(cpu_work) for _ in range(tp_world_size)
        ]
        torch.distributed.all_gather(
            gathered_parts, cpu_work, group=tp_group.cpu_group
        )
        gathered = torch.cat(gathered_parts, dim=0)
        topk_indices[:rows, :topk].copy_(
            gathered[:rows].to(
                device=topk_indices.device, dtype=topk_indices.dtype
            )
        )
        return True

    materialized_chunk_rows = _musa_sparse_indexer_materialized_prefill_chunk_rows(
        work_rows,
        total_seq_lens,
        is_deepseek_v4=is_deepseek_v4,
    )
    if is_deepseek_v4 and total_seq_lens > 4096:
        logger.info_once(
            "Using bounded MUSA DeepSeek-V4 materialized indexer prefill "
            "for %d compressed keys (%d query rows per logits tile, "
            "tp_partition=%s, local_rows=%d).",
            total_seq_lens,
            materialized_chunk_rows,
            tp_partition_rows,
            work_rows,
        )
    materialized_topk_sorted = _musa_sparse_indexer_materialized_prefill_topk_sorted()
    materialized_direct_topk = (
        _musa_sparse_indexer_materialized_prefill_direct_topk_enabled()
    )
    windows_start_at_zero = (
        is_glm52 or use_bounded_long_prefill
    ) and _musa_indexer_windows_start_at_zero(work_starts, topk)

    def _musa_fill_chunk_from_logits(
        logits: torch.Tensor,
        row_start: int,
        row_end: int,
    ) -> bool:
        chunk_rows = row_end - row_start
        if logits.shape[0] < chunk_rows or logits.shape[1] < total_seq_lens:
            return False

        logits = logits[:chunk_rows, :total_seq_lens]
        starts = work_starts[row_start:row_end]
        ends = work_ends[row_start:row_end]
        if is_glm52:
            _musa_select_indexer_topk_windows(
                logits,
                starts,
                ends,
                work_output[row_start:row_end, :topk],
                topk,
                windows_start_at_zero,
            )
            return True

        if use_bounded_long_prefill:
            _musa_select_indexer_topk_windows(
                logits,
                starts.to(torch.int32).contiguous(),
                ends.to(torch.int32).contiguous(),
                work_output[row_start:row_end, :topk],
                topk,
                windows_start_at_zero,
            )
            return True

        assert positions is not None
        valid_positions = (
            (positions.unsqueeze(0) >= starts.unsqueeze(1))
            & (positions.unsqueeze(0) < ends.unsqueeze(1))
        )
        logits.masked_fill_(~valid_positions, float("-inf"))

        if materialized_direct_topk:
            direct_width = min(topk, total_seq_lens)
            direct_abs = torch.topk(
                logits,
                direct_width,
                dim=-1,
                sorted=True,
            ).indices
            row_lens = (ends - starts).clamp(min=0, max=total_seq_lens)
            direct_local = torch.full(
                (chunk_rows, topk),
                -1,
                device=q_quant.device,
                dtype=torch.long,
            )
            local_prefix = direct_abs - starts.unsqueeze(1)
            prefix_valid = (
                (local_prefix >= 0) & (local_prefix < row_lens.unsqueeze(1))
            )
            direct_local[:, :direct_width].copy_(
                torch.where(
                    prefix_valid,
                    local_prefix,
                    torch.full_like(local_prefix, -1),
                )
            )
            topk_offsets = torch.arange(topk, device=q_quant.device, dtype=torch.long)
            full_valid = topk_offsets.unsqueeze(0) < row_lens.unsqueeze(1)
            full_local = torch.where(
                full_valid,
                topk_offsets.unsqueeze(0).expand(chunk_rows, -1),
                torch.full(
                    (chunk_rows, topk),
                    -1,
                    device=q_quant.device,
                    dtype=torch.long,
                ),
            )
            direct_local = torch.where(
                (row_lens <= topk).unsqueeze(1),
                full_local,
                direct_local,
            )
            work_output[row_start:row_end, :topk].copy_(
                direct_local.to(topk_indices.dtype)
            )
            return True

        approx_abs = torch.topk(
            logits,
            overselect,
            dim=-1,
            sorted=materialized_topk_sorted,
        ).indices.contiguous()
        _musa_custom_ops.deepseek_v4_indexer_rerank_prefill(
            work_q[row_start:row_end],
            kv_cache,
            work_weights[row_start:row_end],
            chunk.block_table,
            chunk.cu_seq_lens,
            chunk.token_to_seq,
            chunk.cu_seqlen_ks[row_start:row_end],
            chunk.cu_seqlen_ke[row_start:row_end],
            approx_abs,
            work_output[row_start:row_end, :topk],
            topk,
        )
        return True

    def _musa_fill_with_deep_gemm_provider(_musa_deep_gemm) -> bool:
        get_num_mps = getattr(
            _musa_deep_gemm,
            "get_num_mps",
            getattr(_musa_deep_gemm, "get_num_sms", None),
        )
        try:
            num_mps = int(get_num_mps()) if get_num_mps is not None else 0
        except Exception:
            num_mps = 0

        for row_start in range(0, work_rows, materialized_chunk_rows):
            row_end = min(row_start + materialized_chunk_rows, work_rows)
            chunk_rows = row_end - row_start
            context_chunk = work_context_lens[row_start:row_end].contiguous()
            schedule_meta = _musa_deep_gemm.get_paged_mqa_logits_metadata(
                context_chunk, paged_block_size, num_mps
            )
            logits = _musa_deep_gemm.fp8_paged_mqa_logits(
                work_q[row_start:row_end].unsqueeze(1).contiguous(),
                kv_paged,
                work_weights[row_start:row_end].contiguous(),
                context_chunk,
                paged_block_table.expand(chunk_rows, -1).contiguous(),
                schedule_meta,
                total_seq_lens,
                False,
            )
            if not _musa_fill_chunk_from_logits(logits, row_start, row_end):
                return False
        return True

    def _musa_fill_with_vllm_deep_gemm(_vllm_deep_gemm) -> bool:
        for row_start in range(0, work_rows, materialized_chunk_rows):
            row_end = min(row_start + materialized_chunk_rows, work_rows)
            chunk_rows = row_end - row_start
            context_chunk = work_context_lens[row_start:row_end].contiguous()
            schedule_meta = _vllm_deep_gemm.get_paged_mqa_logits_metadata(
                context_chunk,
                paged_block_size,
                _vllm_deep_gemm.get_num_sms(),
            )
            logits = _vllm_deep_gemm.fp8_fp4_paged_mqa_logits(
                (work_q[row_start:row_end].unsqueeze(1).contiguous(), None),
                kv_paged,
                work_weights[row_start:row_end].contiguous(),
                context_chunk,
                paged_block_table.expand(chunk_rows, -1).contiguous(),
                schedule_metadata=schedule_meta,
                max_model_len=total_seq_lens,
                clean_logits=False,
            )
            if not _musa_fill_chunk_from_logits(logits, row_start, row_end):
                return False
        return True

    provider_errors: list[str] = []
    for provider_name in ("mate.deep_gemm", "deep_gemm"):
        try:
            if provider_name == "mate.deep_gemm":
                from mate import deep_gemm as _musa_deep_gemm
            else:
                import deep_gemm as _musa_deep_gemm

            if _musa_fill_with_deep_gemm_provider(_musa_deep_gemm):
                return _musa_finish_materialized_prefill()
        except Exception as exc:
            provider_errors.append(
                f"{provider_name}: {type(exc).__name__}: {str(exc)[:160]}"
            )

    try:
        from vllm.utils import deep_gemm as _vllm_deep_gemm

        if _musa_fill_with_vllm_deep_gemm(_vllm_deep_gemm):
            return _musa_finish_materialized_prefill()
        provider_errors.append("vllm.utils.deep_gemm: returned no result")
    except Exception as exc:
        provider_errors.append(
            f"vllm.utils.deep_gemm: {type(exc).__name__}: {str(exc)[:160]}"
        )
    if use_bounded_long_prefill:
        logger.warning_once(
            "MUSA DeepSeek-V4 bounded long-prefill indexer providers "
            "failed; falling back to the Torch exact path: %s",
            "; ".join(provider_errors),
        )
    return False


def _musa_try_fill_prefill_topk_from_indexer_cache_native(
    q_quant: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    chunk,
    topk_indices: torch.Tensor,
    topk_tokens: int,
    head_dim: int,
    allow_deepseek_v4_materialized: bool = False,
) -> bool:
    if (
        q_quant.ndim == 3
        and q_quant.shape[1] == 32
        and _musa_try_fill_prefill_topk_from_materialized_logits(
            q_quant,
            kv_cache,
            weights,
            chunk,
            topk_indices,
            topk_tokens,
            head_dim,
        )
    ):
        return True

    if (
        _musa_sparse_indexer_glm52_fused_enabled()
        and head_dim == 128
        and q_quant.ndim == 3
        and q_quant.shape[1] == 32
        and topk_tokens <= 2048
        and q_quant.dtype == torch.float8_e4m3fn
        and kv_cache.dtype == torch.uint8
        and weights.dtype == torch.float32
        and int(chunk.total_seq_lens) <= 8192
    ):
        rows = min(
            q_quant.shape[0],
            chunk.cu_seqlen_ks.numel(),
            topk_indices.shape[0],
        )
        topk = min(int(topk_tokens), topk_indices.shape[1])
        if rows <= 0 or topk <= 0:
            return True
        _musa_custom_ops.glm52_indexer_topk_prefill(
            q_quant[:rows],
            kv_cache,
            weights[:rows],
            chunk.block_table,
            chunk.cu_seq_lens,
            chunk.token_to_seq,
            chunk.cu_seqlen_ks[:rows],
            chunk.cu_seqlen_ke[:rows],
            topk_indices[:rows, :topk],
            topk,
        )
        return True

    is_deepseek_v4 = (
        head_dim == 128
        and q_quant.ndim == 3
        and q_quant.shape[1] == 64
        and topk_tokens <= 512
        and q_quant.dtype == torch.float8_e4m3fn
        and kv_cache.dtype == torch.uint8
        and weights.dtype == torch.float32
    )
    if (
        is_deepseek_v4
        and allow_deepseek_v4_materialized
        and int(chunk.total_seq_lens) > 4096
        and _musa_try_fill_prefill_topk_from_materialized_logits(
            q_quant,
            kv_cache,
            weights,
            chunk,
            topk_indices,
            topk_tokens,
            head_dim,
            allow_deepseek_v4=True,
        )
    ):
        return True

    if (
        not _musa_sparse_indexer_native_decode_enabled()
        or head_dim != 128
        or q_quant.ndim != 3
        or q_quant.shape[1] != 64
        or topk_tokens > 512
        or q_quant.dtype != torch.float8_e4m3fn
        or kv_cache.dtype != torch.uint8
        or weights.dtype != torch.float32
        or int(chunk.total_seq_lens) > 4096
    ):
        return False

    rows = min(q_quant.shape[0], chunk.cu_seqlen_ks.numel(), topk_indices.shape[0])
    topk = min(int(topk_tokens), topk_indices.shape[1])
    if rows <= 0 or topk <= 0:
        return True

    if _musa_try_fill_prefill_topk_from_materialized_logits(
        q_quant[:rows],
        kv_cache,
        weights[:rows],
        chunk,
        topk_indices[:rows, :topk],
        topk,
        head_dim,
        allow_deepseek_v4_materialized,
    ):
        return True

    _musa_custom_ops.deepseek_v4_indexer_topk_prefill(
        q_quant[:rows],
        kv_cache,
        weights[:rows],
        chunk.block_table,
        chunk.cu_seq_lens,
        chunk.token_to_seq,
        chunk.cu_seqlen_ks[:rows],
        chunk.cu_seqlen_ke[:rows],
        topk_indices[:rows, :topk],
        topk,
    )
    return True


def _musa_indexer_cache_block(kv_cache: torch.Tensor, block_id: int) -> torch.Tensor:
    return kv_cache[block_id].view(torch.uint8).flatten()


def _musa_indexer_cache_rows(kv_cache: torch.Tensor) -> torch.Tensor:
    # One row per block over that block's own bytes: the block stride can also
    # span other caches that share the page.
    return kv_cache.as_strided(
        (kv_cache.shape[0], kv_cache[0].numel()),
        (kv_cache.stride(0), 1),
    )


def _musa_dequant_indexer_fp8_cache_row(
    kv_cache: torch.Tensor,
    block_id: int,
    pos_in_block: int,
    head_dim: int,
) -> torch.Tensor:
    block_size = kv_cache.shape[1]
    scale_dim = 4
    cache_block = _musa_indexer_cache_block(kv_cache, block_id)
    token_base = pos_in_block * head_dim
    scale_base = block_size * head_dim + pos_in_block * scale_dim
    values = (
        cache_block[token_base : token_base + head_dim]
        .contiguous()
        .view(torch.float8_e4m3fn)
        .to(torch.float32)
    )
    scale = (
        cache_block[scale_base : scale_base + scale_dim]
        .contiguous()
        .view(torch.float32)
    )
    return values * scale


def _musa_dequant_indexer_fp8_cache_rows(
    kv_cache: torch.Tensor,
    block_ids: torch.Tensor,
    pos_in_block: torch.Tensor,
    head_dim: int,
) -> torch.Tensor:
    if block_ids.numel() == 0:
        return torch.empty((0, head_dim), dtype=torch.float32, device=kv_cache.device)
    block_size = kv_cache.shape[1]
    block_ids = block_ids.to(torch.long)
    pos_in_block = pos_in_block.to(torch.long)
    valid = (
        (block_ids >= 0)
        & (block_ids < kv_cache.shape[0])
        & (pos_in_block >= 0)
        & (pos_in_block < block_size)
    )
    safe_blocks = block_ids.clamp(0, kv_cache.shape[0] - 1)
    safe_pos = pos_in_block.clamp(0, block_size - 1)
    selected_blocks = _musa_indexer_cache_rows(kv_cache).index_select(0, safe_blocks)
    value_offsets = safe_pos.unsqueeze(-1) * head_dim + torch.arange(
        head_dim, device=kv_cache.device, dtype=torch.long
    )
    values = (
        torch.gather(selected_blocks, 1, value_offsets)
        .contiguous()
        .view(torch.float8_e4m3fn)
        .to(torch.float32)
    )
    scale_offsets = (
        block_size * head_dim
        + safe_pos.unsqueeze(-1) * 4
        + torch.arange(4, device=kv_cache.device, dtype=torch.long)
    )
    scales = (
        torch.gather(selected_blocks, 1, scale_offsets)
        .contiguous()
        .view(torch.float32)
        .reshape(-1, 1)
    )
    dequant = values * scales
    return torch.where(valid.unsqueeze(-1), dequant, torch.zeros_like(dequant))


def _musa_gather_indexer_fp8_cache(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    head_dim: int,
) -> torch.Tensor:
    total_seq_lens = int(cu_seq_lens[-1].item())
    if total_seq_lens <= 0:
        return torch.empty((0, head_dim), dtype=torch.float32, device=kv_cache.device)
    block_size = kv_cache.shape[1]
    pieces = []
    for req_idx in range(block_table.shape[0]):
        start = int(cu_seq_lens[req_idx].item())
        end = int(cu_seq_lens[req_idx + 1].item())
        length = end - start
        if length <= 0:
            continue
        local_pos = torch.arange(length, device=kv_cache.device, dtype=torch.long)
        physical_blocks = block_table[
            req_idx,
            torch.div(local_pos, block_size, rounding_mode="floor").clamp(
                0, block_table.shape[1] - 1
            ),
        ]
        pieces.append(
            _musa_dequant_indexer_fp8_cache_rows(
                kv_cache,
                physical_blocks,
                local_pos.remainder(block_size),
                head_dim,
            )
        )
    if not pieces:
        return torch.empty((0, head_dim), dtype=torch.float32, device=kv_cache.device)
    return torch.cat(pieces, dim=0)


def _musa_gather_indexer_fp8_cache_quantized(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FP8 keys and FP32 key scales of each request's cached rows, concatenated."""
    device = kv_cache.device
    num_blocks, block_size = kv_cache.shape[0], kv_cache.shape[1]
    value_bytes = block_size * head_dim
    rows = _musa_indexer_cache_rows(kv_cache)
    values, scales = [], []
    for req_idx in range(block_table.shape[0]):
        start = int(cu_seq_lens[req_idx].item())
        length = int(cu_seq_lens[req_idx + 1].item()) - start
        if length <= 0:
            continue
        req_blocks = block_table[req_idx, : (length + block_size - 1) // block_size].to(
            torch.long
        )
        data = rows.index_select(0, req_blocks.clamp(0, num_blocks - 1))
        valid = ((req_blocks >= 0) & (req_blocks < num_blocks)).repeat_interleave(
            block_size
        )[:length]
        values.append(data[:, :value_bytes].reshape(-1, head_dim)[:length])
        req_scales = (
            data[:, value_bytes : value_bytes + block_size * 4]
            .contiguous()
            .view(torch.float32)
            .reshape(-1)[:length]
        )
        scales.append(torch.where(valid, req_scales, torch.zeros_like(req_scales)))
    if not values:
        return (
            torch.empty((0, head_dim), dtype=torch.float8_e4m3fn, device=device),
            torch.empty((0,), dtype=torch.float32, device=device),
        )
    return (
        torch.cat(values).contiguous().view(torch.float8_e4m3fn),
        torch.cat(scales).contiguous(),
    )


# Bounds the [rows, keys] FP32 logits of one prefill tile.
_MUSA_INDEXER_MQA_LOGITS_TILE_BYTES = 256 << 20


def _musa_try_fill_prefill_topk_from_fp8_mqa_logits(
    q_quant: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    chunk,
    topk_indices: torch.Tensor,
    topk_tokens: int,
    head_dim: int,
) -> bool:
    """Prefill top-k from FP8 MQA logits over the gathered FP8 keys.

    The logits kernel and the windowed top-k take the FP8 cache values and
    scales directly; indices are window-local with -1 in unused slots.
    """
    if (
        head_dim != 128
        or q_quant.ndim != 3
        or q_quant.dtype != torch.float8_e4m3fn
        or kv_cache.dtype != torch.uint8
        or kv_cache.ndim != 3
        or weights.dtype != torch.float32
        or topk_tokens > 2048
    ):
        return False
    try:
        from mate.deep_gemm import fp8_mqa_logits
    except ImportError:
        return False
    rows = min(q_quant.shape[0], chunk.cu_seqlen_ks.numel(), topk_indices.shape[0])
    topk = min(int(topk_tokens), topk_indices.shape[1])
    if rows <= 0 or topk <= 0:
        return True
    k_fp8, k_scale = _musa_gather_indexer_fp8_cache_quantized(
        kv_cache, chunk.block_table, chunk.cu_seq_lens, head_dim
    )
    num_keys = k_fp8.shape[0]
    if num_keys == 0:
        topk_indices[:rows, :topk] = -1
        return True
    starts = chunk.cu_seqlen_ks[:rows].to(torch.int32).clamp(0, num_keys)
    ends = chunk.cu_seqlen_ke[:rows].to(torch.int32).clamp(0, num_keys)
    tile_rows = max(1, _MUSA_INDEXER_MQA_LOGITS_TILE_BYTES // (num_keys * 4))
    starts_at_zero = _musa_indexer_windows_start_at_zero(starts, topk)
    for row_start in range(0, rows, tile_rows):
        row_end = min(rows, row_start + tile_rows)
        tile_starts = starts[row_start:row_end].contiguous()
        tile_ends = ends[row_start:row_end].contiguous()
        logits = fp8_mqa_logits(
            q_quant[row_start:row_end].contiguous(),
            (k_fp8, k_scale),
            weights[row_start:row_end].contiguous(),
            tile_starts,
            tile_ends,
        )
        _musa_select_indexer_topk_windows(
            logits,
            tile_starts,
            tile_ends,
            topk_indices[row_start:row_end, :topk],
            topk,
            starts_at_zero,
        )
    return True


def _musa_sparse_indexer_logits(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    # q is FP8-dequantized without an explicit q scale; the FP8 q scale is
    # already folded into weights by fused_indexer_q_rope_quant.
    per_head = torch.einsum("h d, n d -> h n", q.to(torch.float32), k)
    per_head = per_head.clamp_min(0.0)
    return (per_head * weights.to(torch.float32).unsqueeze(-1)).sum(dim=0)


# Bounds the [rows, heads, keys] FP32 scores of one batched prefill chunk.
_MUSA_INDEXER_PREFILL_CHUNK_BYTES = 128 << 20


def _musa_fill_topk_rows_from_indexer_logits(
    q_deq: torch.Tensor,
    k_deq: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_tokens: int,
) -> None:
    """Top-k of each row's key window ``[ks, ke)``, as window-local indices.

    Rows are scored in chunks against all keys with the keys outside each
    window masked; unused top-k slots keep the caller's -1 fill.
    """
    rows = min(q_deq.shape[0], topk_indices.shape[0], cu_seqlen_ks.numel())
    num_keys = k_deq.shape[0]
    width = min(int(topk_tokens), topk_indices.shape[1], num_keys)
    if rows <= 0 or width <= 0:
        return
    heads = q_deq.shape[1]
    starts = cu_seqlen_ks[:rows].to(torch.long).unsqueeze(1)
    ends = cu_seqlen_ke[:rows].to(torch.long).unsqueeze(1)
    positions = torch.arange(num_keys, device=k_deq.device, dtype=torch.long)
    slots = torch.arange(width, device=k_deq.device, dtype=torch.long)
    keys_t = k_deq.t()
    chunk_rows = max(1, _MUSA_INDEXER_PREFILL_CHUNK_BYTES // (heads * num_keys * 4))
    for row_start in range(0, rows, chunk_rows):
        row_end = min(rows, row_start + chunk_rows)
        per_head = torch.matmul(q_deq[row_start:row_end], keys_t).clamp_min_(0.0)
        logits = (per_head * weights[row_start:row_end].unsqueeze(-1)).sum(dim=1)
        lo = starts[row_start:row_end]
        hi = ends[row_start:row_end]
        logits.masked_fill_((positions < lo) | (positions >= hi), float("-inf"))
        top = torch.topk(logits, width, dim=-1).indices
        in_window = slots < (hi - lo).clamp_min(0)
        topk_indices[row_start:row_end, :width] = torch.where(
            in_window, top - lo, torch.full_like(top, -1)
        ).to(topk_indices.dtype)


def _musa_fill_decode_topk_from_indexer_cache(
    q_deq: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    decode_metadata,
    topk_indices_buffer: torch.Tensor,
    topk_tokens: int,
    head_dim: int,
) -> None:
    seq_lens = decode_metadata.seq_lens.reshape(-1)
    rows = min(q_deq.shape[0], seq_lens.numel(), topk_indices_buffer.shape[0])
    block_size = kv_cache.shape[1]
    for row in range(rows):
        seq_len = int(seq_lens[row].item())
        if seq_len <= 0:
            continue
        block_row = min(row, decode_metadata.block_table.shape[0] - 1)
        local_pos = torch.arange(seq_len, device=kv_cache.device, dtype=torch.long)
        physical_blocks = decode_metadata.block_table[
            block_row,
            torch.div(local_pos, block_size, rounding_mode="floor").clamp(
                0, decode_metadata.block_table.shape[1] - 1
            ),
        ]
        gathered = _musa_dequant_indexer_fp8_cache_rows(
                kv_cache,
                physical_blocks,
                local_pos.remainder(block_size),
                head_dim,
        )
        k_i = min(int(topk_tokens), seq_len, topk_indices_buffer.shape[1])
        logits = _musa_sparse_indexer_logits(q_deq[row], gathered, weights[row])
        topk_indices_buffer[row, :k_i] = torch.topk(logits, k_i, dim=-1).indices.to(
            topk_indices_buffer.dtype
        )


def _musa_fill_decode_topk_from_indexer_cache_capture(
    q_deq: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    decode_metadata,
    topk_indices_buffer: torch.Tensor,
    topk_tokens: int,
    head_dim: int,
) -> None:
    seq_lens = decode_metadata.seq_lens.reshape(-1).to(torch.long)
    rows = min(q_deq.shape[0], seq_lens.numel(), topk_indices_buffer.shape[0])
    topk = min(int(topk_tokens), topk_indices_buffer.shape[1])
    if rows <= 0 or topk <= 0:
        return

    block_table = _musa_decode_block_table_for_token_rows(
        decode_metadata.block_table,
        rows,
    )
    if block_table is None:
        return
    block_size = kv_cache.shape[1]
    max_positions = block_table.shape[1] * block_size
    topk = min(topk, max_positions)
    topk_indices_buffer[:rows, :topk] = -1
    if max_positions <= 0:
        return

    local_pos = torch.arange(max_positions, device=kv_cache.device, dtype=torch.long)
    block_cols = torch.div(local_pos, block_size, rounding_mode="floor").clamp(
        0, block_table.shape[1] - 1
    )
    physical_blocks = block_table[:, block_cols]
    pos_in_block = local_pos.remainder(block_size).expand(rows, -1)

    gathered = _musa_dequant_indexer_fp8_cache_rows(
        kv_cache,
        physical_blocks.reshape(-1),
        pos_in_block.reshape(-1),
        head_dim,
    ).view(rows, max_positions, head_dim)

    q_rows = q_deq[:rows].to(torch.float32)
    weight_rows = weights[:rows].to(torch.float32)
    per_head = torch.einsum("r h d, r n d -> r h n", q_rows, gathered)
    per_head = per_head.clamp_min(0.0)
    logits = (per_head * weight_rows.unsqueeze(-1)).sum(dim=1)

    valid = local_pos.unsqueeze(0) < seq_lens[:rows].unsqueeze(-1)
    logits = torch.where(
        valid,
        logits,
        torch.full((), float("-inf"), dtype=logits.dtype, device=logits.device),
    )
    indices = torch.topk(logits, topk, dim=-1).indices.to(topk_indices_buffer.dtype)
    valid_counts = seq_lens[:rows].clamp(min=0, max=topk).unsqueeze(-1)
    rank_offsets = torch.arange(topk, device=kv_cache.device, dtype=torch.long)
    indices = torch.where(
        rank_offsets.unsqueeze(0) < valid_counts,
        indices,
        torch.full_like(indices, -1),
    )
    topk_indices_buffer[:rows, :topk] = indices


def _musa_fill_exact_sparse_indexer_indices_capture(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    topk_indices_buffer: torch.Tensor,
    use_fp4_cache: bool,
) -> torch.Tensor:
    # Avoid logger calls in fallback forward paths; these functions may be
    # reached while TorchDynamo/CUDA Graph capture is active.
    if use_fp4_cache or isinstance(q_quant, tuple):
        raise NotImplementedError(
            "MUSA learned-indexer capture supports the FP8 path only"
        )

    attn_metadata = get_forward_context().attn_metadata
    if not isinstance(attn_metadata, dict):
        return topk_indices_buffer

    metadata = attn_metadata.get(_resolve_layer_name(k_cache_prefix))
    if not isinstance(metadata, DeepseekV32IndexerMetadata):
        return topk_indices_buffer

    topk_indices_buffer[: hidden_states.shape[0]] = -1

    if metadata.num_decodes > 0 and metadata.decode is not None:
        if not _musa_try_fill_decode_topk_from_indexer_cache_native(
            q_quant[: metadata.num_decode_tokens],
            kv_cache,
            weights[: metadata.num_decode_tokens],
            metadata.decode,
            topk_indices_buffer[: metadata.num_decode_tokens, :topk_tokens],
            topk_tokens,
            head_dim,
            max_model_len,
        ):
            raise RuntimeError(
                "No graph-safe MUSA decode indexer top-k path for this indexer "
                "cache; use cudagraph_mode=PIECEWISE."
            )

    if metadata.num_prefills > 0:
        # Prefill is handled by the eager materialized/native path. Graph
        # capture is decode-only for the DSV4 serving contract.
        pass

    return topk_indices_buffer


def _musa_split_prefill_chunk_per_request(chunk) -> list | None:
    """Split a multi-request prefill chunk into one chunk per request.

    The MUSA prefill top-k paths past 4096 compressed keys take a single
    request. Returns None for a single-request chunk, DCP-local bounds, or a
    request without compressed keys.
    """
    num_reqs = int(chunk.num_reqs)
    rows = int(chunk.token_end) - int(chunk.token_start)
    if num_reqs <= 1 or rows <= 0 or chunk.skip_kv_gather:
        return None
    if chunk.local_cu_seq_lens is not None and (
        chunk.local_cu_seq_lens is not chunk.cu_seq_lens
    ):
        return None

    cu_seq_lens = chunk.cu_seq_lens[: num_reqs + 1].to(torch.long)
    row_starts = chunk.cu_seqlen_ks[:rows].to(torch.long)
    # Rows are grouped by request; each row starts at its request's key offset.
    first_rows = (row_starts.unsqueeze(0) < cu_seq_lens[:-1].unsqueeze(1)).sum(dim=1)
    bounds = torch.cat([first_rows, cu_seq_lens]).tolist()
    row_bounds = bounds[:num_reqs] + [rows]
    seq_bounds = bounds[num_reqs:]

    sub_chunks = []
    for req in range(num_reqs):
        row_start, row_end = row_bounds[req], row_bounds[req + 1]
        seq_start, seq_end = seq_bounds[req], seq_bounds[req + 1]
        if row_end <= row_start or seq_end <= seq_start:
            return None
        cu_seq_lens_req = chunk.cu_seq_lens[req : req + 2] - seq_start
        sub_chunks.append(
            dataclasses.replace(
                chunk,
                block_table=chunk.block_table[req : req + 1],
                cu_seqlen_ks=chunk.cu_seqlen_ks[row_start:row_end] - seq_start,
                cu_seqlen_ke=chunk.cu_seqlen_ke[row_start:row_end] - seq_start,
                cu_seq_lens=cu_seq_lens_req,
                token_to_seq=chunk.token_to_seq[seq_start:seq_end] - req,
                total_seq_lens=seq_end - seq_start,
                token_start=int(chunk.token_start) + row_start,
                token_end=int(chunk.token_start) + row_end,
                num_reqs=1,
                local_cu_seq_lens=cu_seq_lens_req,
                local_total_seq_lens=seq_end - seq_start,
                max_local_total_seq_lens=seq_end - seq_start,
            )
        )
    return sub_chunks


def _musa_fill_exact_sparse_indexer_indices(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    topk_indices_buffer: torch.Tensor,
    use_fp4_cache: bool,
    use_musa_materialized_prefill: bool = False,
) -> torch.Tensor:
    if _musa_try_fill_all_sparse_indexer_indices(
        hidden_states,
        k_cache_prefix,
        topk_tokens,
        topk_indices_buffer,
    ):
        return topk_indices_buffer

    if _musa_sparse_indexer_is_current_stream_capturing():
        return _musa_fill_exact_sparse_indexer_indices_capture(
            hidden_states,
            k_cache_prefix,
            kv_cache,
            q_quant,
            weights,
            topk_tokens,
            head_dim,
            max_model_len,
            topk_indices_buffer,
            use_fp4_cache,
        )

    if use_fp4_cache or isinstance(q_quant, tuple):
        raise NotImplementedError(
            "MUSA exact sparse-attention indexer fallback currently supports "
            "the FP8 indexer-cache path only."
        )

    attn_metadata = get_forward_context().attn_metadata
    if not isinstance(attn_metadata, dict):
        return topk_indices_buffer

    metadata = attn_metadata.get(_resolve_layer_name(k_cache_prefix))
    if not isinstance(metadata, DeepseekV32IndexerMetadata):
        return topk_indices_buffer

    topk_indices_buffer[: hidden_states.shape[0]] = -1
    q_deq = None
    weights_fp32 = None

    if metadata.num_prefills > 0 and metadata.prefill is not None:
        chunks = list(metadata.prefill.chunks)
        while chunks:
            chunk = chunks.pop(0)
            token_start = int(chunk.token_start)
            token_end = int(chunk.token_end)
            if _musa_try_fill_prefill_topk_from_indexer_cache_native(
                q_quant[token_start:token_end],
                kv_cache,
                weights[token_start:token_end],
                chunk,
                topk_indices_buffer[token_start:token_end, :topk_tokens],
                topk_tokens,
                head_dim,
                use_musa_materialized_prefill,
            ):
                continue
            sub_chunks = _musa_split_prefill_chunk_per_request(chunk)
            if sub_chunks is not None:
                chunks[:0] = sub_chunks
                continue
            if _musa_try_fill_prefill_topk_from_fp8_mqa_logits(
                q_quant[token_start:token_end],
                kv_cache,
                weights[token_start:token_end],
                chunk,
                topk_indices_buffer[token_start:token_end, :topk_tokens],
                topk_tokens,
                head_dim,
            ):
                continue
            if q_deq is None:
                q_deq = q_quant.to(torch.float32)
            if weights_fp32 is None:
                weights_fp32 = weights.to(torch.float32)
            k_deq = _musa_gather_indexer_fp8_cache(
                kv_cache,
                chunk.block_table,
                chunk.cu_seq_lens,
                head_dim,
            )
            _musa_fill_topk_rows_from_indexer_logits(
                q_deq[token_start:token_end],
                k_deq,
                weights_fp32[token_start:token_end],
                chunk.cu_seqlen_ks,
                chunk.cu_seqlen_ke,
                topk_indices_buffer[token_start:token_end, :topk_tokens],
                topk_tokens,
            )

    if metadata.num_decodes > 0 and metadata.decode is not None:
        if not _musa_try_fill_decode_topk_from_indexer_cache_native(
            q_quant[: metadata.num_decode_tokens],
            kv_cache,
            weights[: metadata.num_decode_tokens],
            metadata.decode,
            topk_indices_buffer[: metadata.num_decode_tokens, :topk_tokens],
            topk_tokens,
            head_dim,
            max_model_len,
        ):
            if q_deq is None:
                q_deq = q_quant.to(torch.float32)
            if weights_fp32 is None:
                weights_fp32 = weights.to(torch.float32)
            _musa_fill_decode_topk_from_indexer_cache_capture(
                q_deq[: metadata.num_decode_tokens],
                kv_cache,
                weights_fp32[: metadata.num_decode_tokens],
                metadata.decode,
                topk_indices_buffer[: metadata.num_decode_tokens, :topk_tokens],
                topk_tokens,
                head_dim,
            )

    return topk_indices_buffer


# MXFP4 layout: 2 values packed per byte, ue8m0 (1-byte) scale per block of 32.
MXFP4_BLOCK_SIZE = 32


def _assert_cutedsl_dcp_merge_supported(
    logits: torch.Tensor,
    topk_indices: torch.Tensor,
    k: int,
) -> None:
    # The DCP merge only supports the CuteDSL path (Triton pack kernel + CuteDSL
    # stable-topk selector); there is no PyTorch fallback. The first cut targets
    # Blackwell/Hopper with index_topk in (512, 1024, 2048) (the selector's radix
    # sizing); the Triton pack itself has no shape/topk constraints.
    if not has_cutedsl():
        raise RuntimeError(
            "DCP sparse-indexer merge requires CuteDSL; install it or disable DCP."
        )
    if logits.device.type != "cuda":
        raise RuntimeError("DCP sparse-indexer merge requires CUDA tensors.")
    if logits.dtype != torch.float32 or topk_indices.dtype != torch.int32:
        raise RuntimeError(
            "DCP sparse-indexer merge requires fp32 logits and int32 indices."
        )
    if k not in (512, 1024, 2048):
        raise RuntimeError(
            f"DCP sparse-indexer merge requires index_topk in (512, 1024, 2048); "
            f"got {k}."
        )


def _merge_dcp_topk_global(
    logits: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_tokens: int,
    dcp_rank: int,
    dcp_world_size: int,
    cp_interleave: int,
    row_starts: torch.Tensor | None = None,
) -> None:
    """Merge each DCP rank's local top-K into the global top-K.

    ``topk_indices`` are this rank's local top-K positions into its 1/N KV
    shard. A token in the global top-K must also be in its owning rank's local
    top-K (at most ``topk_tokens - 1`` tokens rank globally above it, hence at
    most that many on its own rank), so exchanging only the per-rank local
    candidates is exact -- equivalent to all-gathering the full logit matrix,
    but it ships ``dcp_world_size * topk_tokens`` candidates instead of the whole
    score row. Overwrites ``topk_indices`` with global token ids (``-1`` for
    padding); the attention backend localizes them back to physical slots per
    rank.
    """
    if dcp_world_size <= 1:
        return

    # CuteDSL-only path (no PyTorch fallback): Triton-pack each rank's
    # (score, global_id) candidates on-device, all-gather, then the CuteDSL
    # stable-topk selector.
    _assert_cutedsl_dcp_merge_supported(logits, topk_indices, topk_tokens)
    from vllm.model_executor.kernels.attention.dsa.dcp_indexer_cutedsl import (
        pack_dcp_topk_candidates_cutedsl,
        stable_topk_from_gathered_candidates_cutedsl,
    )

    packed = torch.empty(
        (*topk_indices.shape, 2),
        dtype=torch.float32,
        device=topk_indices.device,
    )
    pack_dcp_topk_candidates_cutedsl(
        logits,
        topk_indices,
        packed,
        dcp_rank,
        dcp_world_size,
        cp_interleave,
        row_starts,
    )
    gathered = get_dcp_group().all_gather(packed, dim=1)
    stable_topk_from_gathered_candidates_cutedsl(
        gathered, topk_tokens, out=topk_indices
    )


def dcp_gather_kv_rows(
    local_padded: torch.Tensor,
    deinterleave_idx: torch.Tensor,
    gathered_buf: torch.Tensor,
    out_buf: torch.Tensor,
) -> torch.Tensor:
    """All-gather this rank's KV shard and return it in global token order."""
    gathered = gathered_buf[: get_dcp_group().world_size * local_padded.shape[0]]
    dist.all_gather_into_tensor(
        gathered, local_padded.contiguous(), group=get_dcp_group().device_group
    )
    return torch.index_select(
        gathered, 0, deinterleave_idx, out=out_buf[: deinterleave_idx.shape[0]]
    )


@triton.jit
def _fused_indexer_q_rope_quant_kernel(
    positions,
    q,
    q_s0,
    q_s1,
    cos_sin_cache,
    cos_sin_s0,
    q_fp8,
    q_fp8_s0,
    q_fp8_s1,
    weights,
    weights_s0,
    weights_s1,
    weights_out,
    weights_out_s0,
    weights_out_s1,
    softmax_scale,
    head_scale,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    is_neox: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    offs32 = tl.arange(0, 32)
    offs64 = tl.arange(0, 64)

    pos = tl.load(positions + token)
    cos = tl.load(cos_sin_cache + pos * cos_sin_s0 + offs32).to(tl.float32)
    sin = tl.load(cos_sin_cache + pos * cos_sin_s0 + 32 + offs32).to(tl.float32)
    q_base = q + token * q_s0 + head * q_s1
    out_base = q_fp8 + token * q_fp8_s0 + head * q_fp8_s1

    if is_neox:
        # NeoX layout, x0 = q[0:32], x1 = q[32:64]
        x0 = tl.load(q_base + offs32).to(tl.float32)
        x1 = tl.load(q_base + 32 + offs32).to(tl.float32)
    else:
        # interleaved layout
        # x0 = q[0, 2, 4, ...], x1 = q[1, 3, 5, ...]
        x0 = tl.load(q_base + offs32 * 2).to(tl.float32)
        x1 = tl.load(q_base + offs32 * 2 + 1).to(tl.float32)
    r0 = (x0 * cos - x1 * sin).to(tl.bfloat16).to(tl.float32)
    r1 = (x1 * cos + x0 * sin).to(tl.bfloat16).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(r0)), tl.max(tl.abs(r1)))

    q_nope = tl.load(q_base + 64 + offs64).to(tl.float32)
    amax = tl.maximum(amax, tl.max(tl.abs(q_nope)))
    scale_raw = tl.maximum(amax, 1e-10) * (1.0 / fp8_max)
    # e8m0 format
    q_scale = tl.math.exp2(tl.ceil(tl.log2(scale_raw)))

    if is_neox:
        tl.store(
            out_base + offs32,
            tl.clamp(r0 / q_scale, fp8_min, fp8_max).to(q_fp8.dtype.element_ty),
        )
        tl.store(
            out_base + 32 + offs32,
            tl.clamp(r1 / q_scale, fp8_min, fp8_max).to(q_fp8.dtype.element_ty),
        )
    else:
        tl.store(
            out_base + offs32 * 2,
            tl.clamp(r0 / q_scale, fp8_min, fp8_max).to(q_fp8.dtype.element_ty),
        )
        tl.store(
            out_base + offs32 * 2 + 1,
            tl.clamp(r1 / q_scale, fp8_min, fp8_max).to(q_fp8.dtype.element_ty),
        )
    tl.store(
        out_base + 64 + offs64,
        tl.clamp(q_nope / q_scale, fp8_min, fp8_max).to(q_fp8.dtype.element_ty),
    )

    weight = tl.load(weights + token * weights_s0 + head * weights_s1).to(tl.float32)
    tl.store(
        weights_out + token * weights_out_s0 + head * weights_out_s1,
        weight * q_scale * softmax_scale * head_scale,
    )


def fused_indexer_q_rope_quant(
    positions: torch.Tensor,
    q: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    weights: torch.Tensor,
    softmax_scale: float,
    head_scale: float,
    is_neox: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert current_platform.is_cuda()
    assert q.dtype == torch.bfloat16
    assert q.shape[-1] == 128
    assert cos_sin_cache.shape[-1] == 64
    assert weights.shape == q.shape[:2]

    q_fp8 = torch.empty_like(q, dtype=current_platform.fp8_dtype())
    weights_out = torch.empty_like(weights, dtype=torch.float32)
    fp8_min, fp8_max = get_fp8_min_max()
    _fused_indexer_q_rope_quant_kernel[(q.shape[0], q.shape[1])](
        positions,
        q,
        q.stride(0),
        q.stride(1),
        cos_sin_cache,
        cos_sin_cache.stride(0),
        q_fp8,
        q_fp8.stride(0),
        q_fp8.stride(1),
        weights,
        weights.stride(0),
        weights.stride(1),
        weights_out,
        weights_out.stride(0),
        weights_out.stride(1),
        softmax_scale,
        head_scale,
        fp8_min=fp8_min,
        fp8_max=fp8_max,
        is_neox=is_neox,
        num_warps=1,
    )
    return q_fp8, weights_out


def _gather_workspace_shapes(
    total_seq_lens: int,
    head_dim: int,
    fp8_dtype: torch.dtype,
    use_fp4_cache: bool,
) -> tuple[tuple[tuple[int, int], torch.dtype], tuple[tuple[int, int], torch.dtype]]:
    """Return ((values_shape, values_dtype), (scales_shape, scales_dtype)) for
    the K-gather workspace. FP8 path: (T, head_dim) fp8 + (T, 4) uint8 fp32
    scales. MXFP4 path: (T, head_dim // 2) uint8 packed mxfp4 +
    (T, head_dim // MXFP4_BLOCK_SIZE) uint8 ue8m0 scales."""
    if use_fp4_cache:
        return (
            ((total_seq_lens, head_dim // 2), torch.uint8),
            ((total_seq_lens, head_dim // MXFP4_BLOCK_SIZE), torch.uint8),
        )
    return (
        ((total_seq_lens, head_dim), fp8_dtype),
        ((total_seq_lens, 4), torch.uint8),
    )


def kv_cache_as_quant_view(
    kv_cache: torch.Tensor,
    head_dim: int,
    use_fp4_cache: bool,
) -> torch.Tensor:
    """4D ``[num_blocks, block_size, 1, head_width]`` view expected by
    DeepGEMM, from the 3D indexer kv-cache allocation."""
    if use_fp4_cache:
        assert kv_cache.ndim == 3 and kv_cache.dtype == torch.uint8
        num_blocks, block_size, _ = kv_cache.shape
        page_bytes = int(kv_cache.stride(0))
        fp4_bytes = head_dim // 2 + head_dim // MXFP4_BLOCK_SIZE
        return torch.as_strided(
            kv_cache,
            size=(num_blocks, block_size, 1, fp4_bytes),
            stride=(page_bytes, fp4_bytes, fp4_bytes, 1),
        )
    return kv_cache.unsqueeze(-2)


class PrefillRowShard:
    """One TP rank's contiguous slice of the replicated prefill rows.

    Rows are independent, so each rank scores only its slice and the group
    exchanges final int32 top-k indices instead of the logits."""

    def __init__(self, num_decode_tokens: int, sizes: list[int], rank: int):
        self.num_decode_tokens = num_decode_tokens
        self.sizes = sizes
        self.start = num_decode_tokens + sum(sizes[:rank])
        self.stop = self.start + sizes[rank]

    @classmethod
    def from_metadata(
        cls, prefill_metadata, num_decode_tokens: int
    ) -> "PrefillRowShard | None":
        sizes = getattr(prefill_metadata, "row_shard_sizes", None)
        if sizes is None:
            return None
        return cls(num_decode_tokens, sizes, get_tensor_model_parallel_rank())

    def narrow(self, chunk) -> "tuple[int, int, torch.Tensor, torch.Tensor] | None":
        """Row window and narrowed causal bounds of ``chunk`` for this rank."""
        start = max(chunk.token_start, self.start)
        end = min(chunk.token_end, self.stop)
        if start >= end:
            return None
        lo, hi = start - chunk.token_start, end - chunk.token_start
        return start, end, chunk.cu_seqlen_ks[lo:hi], chunk.cu_seqlen_ke[lo:hi]

    def exchange_topk(self, topk_buffer: torch.Tensor, width: int) -> None:
        """Gather per-rank final indices back into global row order."""
        # The local input must outlive the async PyNCCL gather's enqueued copy.
        local = topk_buffer[self.start : self.stop, :width].contiguous()
        end = self.num_decode_tokens + sum(self.sizes)
        topk_buffer[self.num_decode_tokens : end, :width] = get_tp_group().all_gatherv(
            local, dim=0, sizes=self.sizes
        )


def _reserve_row_shard_exchange_transient(
    topk_indices_buffer: torch.Tensor, device: torch.device
) -> None:
    """Profiling/dummy runs return before ``exchange_topk`` executes; reserve
    its worst-case transient so the memory profile accounts for it.

    The exchange stages this rank's contiguous slice and the all_gatherv
    wrapper allocates the full gathered tensor; each is bounded by the
    buffer's row capacity times its (tile-padded) width.
    """
    cfg = get_current_vllm_config_or_none()
    if cfg is None:
        return
    parallel = cfg.parallel_config
    if not tp_prefill_row_sharding_supported(
        cfg,
        parallel.decode_context_parallel_size,
        parallel.prefill_context_parallel_size > 1,
        get_tp_group().world_size,
    ):
        return
    rows, width = topk_indices_buffer.shape[:2]
    _ = torch.empty(
        2 * rows * width * topk_indices_buffer.element_size(),
        dtype=torch.uint8,
        device=device,
    )


@eager_break_during_capture
def sparse_attn_indexer(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor | None,
    k: torch.Tensor | None,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor,
    skip_k_cache_insert: bool,
    use_pcp: bool,
    pcp_shard_decode_requests: bool,
    dense_mha_metadata_layer_name: LayerNameType,
    use_fp4_cache: bool = False,
    dcp_rank: int = 0,
    dcp_world_size: int = 1,
    cp_kv_cache_interleave_size: int = 1,
    skip_topk_buffer_clear: bool = False,
    candidate_blocks: torch.Tensor | None = None,
    candidate_block_size: int = 0,
    candidate_write: bool = False,
    topk_backend: str = "auto",
) -> torch.Tensor:
    # careful! this will be None in dummy run
    forward_context = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    fp8_dtype = current_platform.fp8_dtype()
    k_cache_prefix = _resolve_layer_name(k_cache_prefix)

    if candidate_blocks is not None:
        # Candidate blocks are request-local; the DCP-sharded logits layout
        # would need per-rank translation that is not implemented.
        assert dcp_world_size == 1, (
            "v4.1 two-level candidate filtering is not supported with DCP."
        )
        assert candidate_block_size > 0

    # assert isinstance(attn_metadata, dict)
    if not isinstance(attn_metadata, dict):
        # Reserve workspace for indexer during profiling run
        values_spec, scales_spec = _gather_workspace_shapes(
            total_seq_lens, head_dim, fp8_dtype, use_fp4_cache
        )
        profile_specs: list[tuple[tuple[int, ...], torch.dtype]] = [
            values_spec,
            scales_spec,
            ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
        ]
        if use_pcp and dcp_world_size > 1:
            # The PCP+DCP path takes an all-gather destination and a
            # de-interleaved result.
            gather_spec = _gather_workspace_shapes(
                total_seq_lens, head_dim, fp8_dtype, use_fp4_cache
            )
            profile_specs.extend(gather_spec * 2)
        current_workspace_manager().get_simultaneous(*profile_specs)

        # Dummy allocation to simulate for peak logits tensor memory during inference.
        # FP8 elements so elements == bytes
        max_logits_elems = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
        _ = torch.empty(
            max_logits_elems, dtype=torch.uint8, device=hidden_states.device
        )

        # The dummy run returns before exchange_topk executes, so reserve its
        # worst-case transient for the memory profile.
        _reserve_row_shard_exchange_transient(topk_indices_buffer, hidden_states.device)

        return sparse_attn_indexer_fake(
            hidden_states,
            k_cache_prefix,
            kv_cache,
            q_quant,
            q_scale,
            k,
            weights,
            quant_block_size,
            scale_fmt,
            topk_tokens,
            head_dim,
            max_model_len,
            total_seq_lens,
            topk_indices_buffer,
            skip_k_cache_insert,
            use_pcp,
            pcp_shard_decode_requests,
            dense_mha_metadata_layer_name,
            use_fp4_cache,
            candidate_blocks=candidate_blocks,
            candidate_block_size=candidate_block_size,
            candidate_write=candidate_write,
        )
    attn_metadata_narrowed = attn_metadata[k_cache_prefix]
    assert isinstance(attn_metadata_narrowed, DeepseekV32IndexerMetadata)
    slot_mapping = attn_metadata_narrowed.slot_mapping
    has_decode = attn_metadata_narrowed.num_decodes > 0
    has_prefill = attn_metadata_narrowed.num_prefills > 0
    num_decode_tokens = attn_metadata_narrowed.num_decode_tokens

    # q_scale is required iff the FP4 cache path is enabled; the FP8 path
    # folds the Q scale into `weights` inside fused_indexer_q_rope_quant.
    if use_fp4_cache:
        assert q_scale is not None, "use_fp4_cache=True requires q_scale"
    else:
        assert q_scale is None, "q_scale must be None when use_fp4_cache=False"

    # During speculative decoding, k may be padded to the CUDA graph batch
    # size while slot_mapping only covers actual tokens. Truncate k to avoid
    # out-of-bounds reads in the kernel.
    # Keep PCP padding so every rank contributes the same all-gather shape.
    num_tokens = slot_mapping.shape[0]
    if k is not None:
        # MTP draft decodes are replicated and use an unexpanded slot mapping.
        if use_pcp and num_tokens > k.shape[0]:
            num_tokens //= get_pcp_group().world_size
        k = k[:num_tokens]

    if not skip_k_cache_insert:
        assert k is not None
        k, slot_mapping_for_cache = maybe_gather_indexer_k(
            k,
            slot_mapping,
            num_decode_tokens,
            use_pcp,
            pcp_shard_decode_requests=pcp_shard_decode_requests,
        )
        # scale_fmt can be None, but the function expects str
        assert scale_fmt is not None
        assert not use_fp4_cache, "Unfused FP4 Insert is not supported yet"
        ops.indexer_k_quant_and_cache(
            k,
            kv_cache,
            slot_mapping_for_cache,
            quant_block_size,
            scale_fmt,
        )
    # The indexer and main MLA may classify the same short extend differently
    # because they use independent decode thresholds. Only the main MLA route
    # can determine whether the top-k indices will be consumed.
    if forward_context.cudagraph_runtime_mode != CUDAGraphMode.FULL:
        dense_mha_layer = _resolve_layer_name(dense_mha_metadata_layer_name)
        if dense_mha_layer:
            mla_metadata = attn_metadata.get(dense_mha_layer)
            prefill_metadata = getattr(mla_metadata, "prefill", None)
            if (
                getattr(prefill_metadata, "use_dense_mha", False)
                and getattr(mla_metadata, "num_decode_tokens", -1) == 0
                and not torch.cuda.is_current_stream_capturing()
            ):
                # Deliberately leave the buffer untouched. Dense MHA does not
                # consume top-k indices for this batch; clearing it would be
                # unnecessary work.
                return topk_indices_buffer

    # The buffer must be pre-filled with -1 (the "no token" sentinel) before the
    # top-k kernels scatter valid indices into it. On the fused deepseek_v32
    # nvidia path, _fused_norm_rope_kernel already cleared the same
    # [:num_tokens, :topk] region earlier in this forward, so skip the redundant
    # fill.
    if not skip_topk_buffer_clear:
        topk_indices_buffer[: hidden_states.shape[0]] = -1
    if has_prefill:
        prefill_metadata = attn_metadata_narrowed.prefill
        assert prefill_metadata is not None

        # Get the full shared workspace buffers once (will allocate on first use).
        # Layout switches between FP8 (head_dim bytes + 4-byte fp32 scale) and
        # MXFP4 (head_dim/2 bytes packed + head_dim/MXFP4_BLOCK_SIZE ue8m0
        # scales) based on use_fp4_cache.
        workspace_manager = current_workspace_manager()
        values_spec, scales_spec = _gather_workspace_shapes(
            total_seq_lens, head_dim, fp8_dtype, use_fp4_cache
        )
        # PCP + DCP needs two more pairs: the rank-major all-gather destination
        # and the de-interleaved result.
        pcp_chunks = [
            (c, c.pcp_deinterleave_idx)
            for c in prefill_metadata.chunks
            if c.pcp_deinterleave_idx is not None
        ]
        gather_specs: list[tuple[tuple[int, int], torch.dtype]] = []
        if pcp_chunks:
            gathered_rows = max(
                dcp_world_size * c.local_total_seq_lens for c, _ in pcp_chunks
            )
            deinterleaved_rows = max(idx.shape[0] for _, idx in pcp_chunks)
            for rows in (gathered_rows, deinterleaved_rows):
                gather_specs.extend(
                    _gather_workspace_shapes(rows, head_dim, fp8_dtype, use_fp4_cache)
                )
        k_quant_full, k_scale_full, *gather_bufs = workspace_manager.get_simultaneous(
            values_spec,
            scales_spec,
            *gather_specs,
        )
        # One row window per rank; DSV4.1 candidate rows stay replicated.
        row_shard = (
            None
            if candidate_blocks is not None
            else PrefillRowShard.from_metadata(prefill_metadata, num_decode_tokens)
        )
        for chunk in prefill_metadata.chunks:
            cu_seqlen_ks = chunk.cu_seqlen_ks
            cu_seqlen_ke = chunk.cu_seqlen_ke
            assert chunk.local_cu_seq_lens is not None
            k_quant = k_quant_full[: chunk.max_local_total_seq_lens]
            k_scale = k_scale_full[: chunk.max_local_total_seq_lens]
            if not chunk.skip_kv_gather and chunk.local_total_seq_lens > 0:
                ops.cp_gather_indexer_k_quant_cache(
                    kv_cache,
                    k_quant,
                    k_scale,
                    chunk.block_table,
                    chunk.local_cu_seq_lens,
                )

            # Narrow scoring to this rank; KV gathers remain unconditional for
            # continuation chunks that reuse the workspace.
            row_start, row_end = chunk.token_start, chunk.token_end
            if row_shard is not None:
                narrowed = row_shard.narrow(chunk)
                if narrowed is None:
                    continue
                row_start, row_end, cu_seqlen_ks, cu_seqlen_ke = narrowed

            # PCP + DCP KV all-gather.
            deinterleave_idx = chunk.pcp_deinterleave_idx
            if deinterleave_idx is not None:
                # local_total_seq_lens is the PCP-padded extent here, so every
                # rank contributes an identically shaped shard.
                gathered_values, gathered_scales, out_values, out_scales = gather_bufs
                shard = k_quant[: chunk.local_total_seq_lens]
                k_quant = dcp_gather_kv_rows(
                    shard, deinterleave_idx, gathered_values, out_values
                )
                k_scale = dcp_gather_kv_rows(
                    k_scale[: chunk.local_total_seq_lens],
                    deinterleave_idx,
                    gathered_scales,
                    out_scales,
                )

            q_slice = q_quant[row_start:row_end]
            q_scale_slice = q_scale[row_start:row_end] if q_scale is not None else None
            topk_indices = topk_indices_buffer[row_start:row_end, :topk_tokens]

            if chunk.local_total_seq_lens == 0 or q_slice.shape[0] == 0:
                logits = q_slice.new_empty((q_slice.shape[0], 0), dtype=torch.float32)
                topk_indices.fill_(-1)
            else:
                # DeepGEMM scalar-type tags (zero-copy): MXFP4 values → int8
                # (kPackedFP4), scales → int32 squeezed to 1-D kv_sf / 2-D q_sf.
                if use_fp4_cache:
                    q_slice_cast = q_slice.view(torch.int8)
                    k_quant_cast = k_quant.view(torch.int8)
                    k_scale_cast = k_scale.view(torch.int32).squeeze(-1)
                else:
                    q_slice_cast = q_slice
                    k_quant_cast = k_quant
                    k_scale_cast = k_scale.view(torch.float32).squeeze(-1)
                if current_platform.is_xpu():
                    if q_scale_slice is not None:
                        raise RuntimeError("XPU fp8_mqa_logits does not support FP4 Q")
                    logits = torch.ops.vllm.xpu_fp8_mqa_logits(
                        q_slice_cast,
                        k_quant_cast,
                        k_scale_cast,
                        weights[row_start:row_end],
                        cu_seqlen_ks,
                        cu_seqlen_ke,
                    )
                else:
                    logits = fp8_fp4_mqa_logits(
                        (q_slice_cast, q_scale_slice),
                        (k_quant_cast, k_scale_cast),
                        weights[row_start:row_end],
                        cu_seqlen_ks,
                        cu_seqlen_ke,
                        clean_logits=False,
                    )
                num_rows = logits.shape[0]
                if candidate_blocks is not None:
                    # Two-level selection (v4.1): the candidate source
                    # publishes its top blocks; later indexers mask their
                    # scores to them. Both before the row top-k.
                    chunk_candidates = candidate_blocks[
                        chunk.token_start : chunk.token_end
                    ]
                    if candidate_write:
                        _select_candidate_blocks(
                            logits,
                            cu_seqlen_ks,
                            cu_seqlen_ke,
                            chunk_candidates.shape[1],
                            candidate_block_size,
                            chunk_candidates,
                        )
                    else:
                        _apply_candidate_mask(
                            logits,
                            cu_seqlen_ks,
                            cu_seqlen_ke,
                            chunk_candidates,
                            candidate_block_size,
                        )
                ops.top_k_per_row_prefill(
                    logits,
                    cu_seqlen_ks,
                    cu_seqlen_ke,
                    topk_indices,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    topk_tokens,
                )

            if deinterleave_idx is None:
                # Under the PCP path the top-k already ran over the whole
                # context.
                _merge_dcp_topk_global(
                    logits,
                    topk_indices,
                    topk_tokens,
                    dcp_rank,
                    dcp_world_size,
                    cp_kv_cache_interleave_size,
                    row_starts=cu_seqlen_ks,
                )

        if row_shard is not None:
            # Rows are already fully ranked, so this is a layout gather.
            row_shard.exchange_topk(topk_indices_buffer, topk_tokens)

    if has_decode:
        decode_metadata = attn_metadata_narrowed.decode
        assert decode_metadata is not None
        kv_cache = kv_cache_as_quant_view(kv_cache, head_dim, use_fp4_cache)
        decode_lens = decode_metadata.decode_lens
        # requires_padding can be computed False for ragged warmup/mixed
        # batches (seen on SM120 TP=2: 8 tokens over 6 seqs -> uniform
        # reshape below would crash). The token-count/batch-size divisibility
        # check is a necessary-but-not-sufficient proxy for "uniform batch"
        # (e.g. decode_lens=[1, 2, 3] divides evenly by 3 but isn't uniform);
        # it catches the specific case seen in practice without requiring a
        # device sync to inspect decode_lens directly, which would break
        # CUDA-graph-capture safety on this path. Shared by both the pack
        # branch below and the matching unpack branch further down — they
        # must agree on which path was taken.
        needs_padded_path = decode_metadata.requires_padding or (
            num_decode_tokens % decode_lens.shape[0] != 0
        )
        if num_decode_tokens == 0:
            padded_q_quant_decode_tokens = q_quant[:1].reshape(1, 1, *q_quant.shape[1:])
            padded_q_scale = (
                q_scale[:1].reshape(1, 1, *q_scale.shape[1:])
                if q_scale is not None
                else None
            )
        elif needs_padded_path:
            # pad in edge case where we have short chunked prefill length <
            # decode_threshold since we unstrictly split
            # prefill and decode by decode_threshold
            # (currently set to 1 + speculative tokens).
            # FP8 Q is float8_e4m3fn (pack_seq_triton's fp32 pad path is OK —
            # downstream context_lens masks stale slots). MXFP4 Q is two
            # uint8 tensors (values + ue8m0 scales) — use the dedicated uint8
            # packer with pad_byte=0 so padded slots dequantize to 0 and
            # can't produce NaN/Inf in the logits kernel.
            if q_scale is not None:
                padded_q_quant_decode_tokens = pack_seq_triton(
                    q_quant[:num_decode_tokens], decode_lens, pad_value=0
                )
                padded_q_scale = pack_seq_triton(
                    q_scale[:num_decode_tokens], decode_lens, pad_value=0
                )
            else:
                padded_q_quant_decode_tokens = pack_seq_triton(
                    q_quant[:num_decode_tokens], decode_lens
                )
                padded_q_scale = None
        else:
            padded_q_quant_decode_tokens = q_quant[:num_decode_tokens].reshape(
                decode_lens.shape[0], -1, *q_quant.shape[1:]
            )
            if q_scale is not None:
                padded_q_scale = q_scale[:num_decode_tokens].reshape(
                    decode_lens.shape[0], -1, *q_scale.shape[1:]
                )
            else:
                padded_q_scale = None
        # TODO: move and optimize below logic with triton kernels
        batch_size = padded_q_quant_decode_tokens.shape[0]
        next_n = padded_q_quant_decode_tokens.shape[1]
        num_padded_tokens = batch_size * next_n
        seq_lens = decode_metadata.seq_lens[:batch_size]
        # seq_lens is always 2D: (B, next_n) for native spec decode, (B, 1)
        # otherwise. deep_gemm fp8_fp4_paged_mqa_logits requires 2D context_lens;
        # the downstream topk kernels accept both 1D and 2D.
        padded_q_quant_cast = (
            padded_q_quant_decode_tokens.view(torch.int8)
            if use_fp4_cache
            else padded_q_quant_decode_tokens
        )
        if current_platform.is_xpu():
            if padded_q_scale is not None:
                raise RuntimeError("XPU fp8_paged_mqa_logits does not support FP4 Q")
            seq_lens_xpu = (
                seq_lens[:, -1].contiguous() if seq_lens.ndim == 2 else seq_lens
            )
            logits = torch.ops.vllm.xpu_fp8_paged_mqa_logits(
                padded_q_quant_cast,
                kv_cache,
                weights[:num_padded_tokens],
                seq_lens_xpu,
                decode_metadata.block_table,
                decode_metadata.schedule_metadata,
                max_model_len,
            )
        else:
            logits = fp8_fp4_paged_mqa_logits(
                (padded_q_quant_cast, padded_q_scale),
                kv_cache,
                weights[:num_padded_tokens],
                seq_lens,
                decode_metadata.block_table,
                decode_metadata.schedule_metadata,
                max_model_len=max_model_len,
                clean_logits=False,
                indices=decode_metadata.indices,
            )
        num_rows = logits.shape[0]
        if candidate_blocks is not None:
            # Two-level selection (v4.1) on the decode logits; columns are
            # request-local compressed positions. seq_lens is (B, next_n)
            # for native spec decode (per-row effective lens) and (B, 1)
            # otherwise.
            vis = seq_lens.reshape(-1)
            row_repeat = next_n if vis.numel() != num_rows else 1
            vis = vis[:num_rows]
            decode_candidates = candidate_blocks[:num_rows]
            if candidate_write:
                _select_candidate_blocks(
                    logits,
                    None,
                    vis,
                    decode_candidates.shape[1],
                    candidate_block_size,
                    decode_candidates,
                    row_repeat,
                )
            else:
                _apply_candidate_mask(
                    logits,
                    None,
                    vis,
                    decode_candidates,
                    candidate_block_size,
                    row_repeat,
                )
        topk_indices = topk_indices_buffer[:num_padded_tokens, :topk_tokens]

        # The backend comes from the layer (config is only readable at model
        # construction); dispatchers are cached per backend.
        get_indexer_topk(topk_backend)(
            logits,
            seq_lens,
            next_n,
            topk_indices,
            topk_tokens,
            attn_metadata_narrowed.max_seq_len,
        )

        if decode_metadata.global_seq_lens is not None:
            _merge_dcp_topk_global(
                logits,
                topk_indices,
                topk_tokens,
                dcp_rank,
                dcp_world_size,
                cp_kv_cache_interleave_size,
            )

        if needs_padded_path:
            # if padded, we need to unpack
            # the topk indices removing padded tokens
            topk_indices = unpack_seq_triton(
                topk_indices.reshape(batch_size, -1, topk_indices.shape[-1]),
                decode_lens,
            )
            topk_indices_buffer[: topk_indices.shape[0], : topk_indices.shape[-1]] = (
                topk_indices
            )

    return topk_indices_buffer


def sparse_attn_indexer_fake(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor | None,
    k: torch.Tensor | None,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor | None,
    skip_k_cache_insert: bool,
    use_pcp: bool,
    pcp_shard_decode_requests: bool,
    dense_mha_metadata_layer_name: LayerNameType,
    use_fp4_cache: bool = False,
    dcp_rank: int = 0,
    dcp_world_size: int = 1,
    cp_kv_cache_interleave_size: int = 1,
    skip_topk_buffer_clear: bool = False,
    candidate_blocks: torch.Tensor | None = None,
    candidate_block_size: int = 0,
    candidate_write: bool = False,
    topk_backend: str = "auto",
) -> torch.Tensor:
    return topk_indices_buffer


direct_register_custom_op(
    op_name="sparse_attn_indexer",
    op_func=sparse_attn_indexer,
    mutates_args=["topk_indices_buffer", "candidate_blocks"],
    fake_impl=sparse_attn_indexer_fake,
    dispatch_key=current_platform.dispatch_key,
)


@CustomOp.register("sparse_attn_indexer")
class SparseAttnIndexer(CustomOp):
    """Sparse Attention Indexer Custom Op Layer. This layer is extracted as a
    separate custom op since it involves heavy custom kernels like `mqa_logits`,
    `paged_mqa_logits` and `top_k_per_row`, etc. Those kernels maybe requires
    specific memory layout or implementation for different hardware backends to
    achieve optimal performance.

    For now, the default native path will use CUDA backend path. Other platform
    may requires add the corresponding Custom Op name `sparse_attn_indexer` to
    `custom_ops` in `CompilationConfig` to enable the platform specific path.
    """

    def __init__(
        self,
        k_cache,
        quant_block_size: int,
        scale_fmt: str,
        topk_tokens: int,
        head_dim: int,
        max_model_len: int,
        max_total_seq_len: int,
        topk_indices_buffer: torch.Tensor,
        skip_k_cache_insert: bool = False,
        use_fp4_cache: bool = False,
        compress_ratio: int = 1,
        candidate_blocks: torch.Tensor | None = None,
        candidate_block_size: int = 0,
        candidate_write: bool = False,
        use_musa_native_indexer: bool = False,
        use_musa_materialized_prefill: bool = False,
    ):
        super().__init__()
        self.k_cache = k_cache
        self.quant_block_size = quant_block_size
        self.scale_fmt = scale_fmt
        self.topk_tokens = topk_tokens
        self.head_dim = head_dim
        self.max_model_len = max_model_len
        self.max_total_seq_len = max_total_seq_len
        self.topk_indices_buffer = topk_indices_buffer
        self.skip_k_cache_insert = skip_k_cache_insert
        self.use_fp4_cache = use_fp4_cache
        self.compress_ratio = compress_ratio
        # v4.1 two-level selection: the candidate source indexer writes the
        # top candidate blocks here; later indexers mask their scores with it.
        self.candidate_blocks = candidate_blocks
        self.candidate_block_size = candidate_block_size
        self.candidate_write = candidate_write
        self.dense_mha_metadata_layer_name = ""
        # DCP scalars are constant for the run; resolve them here (config is set
        # during model construction) and pass them into the custom op, rather
        # than threading them through per-step metadata.
        vllm_config = get_current_vllm_config()
        parallel_config = vllm_config.parallel_config
        self.topk_backend = vllm_config.kernel_config.sparse_indexer_topk_backend
        self._parallel_config = parallel_config
        self.dcp_world_size = parallel_config.decode_context_parallel_size
        self.dcp_rank = get_dcp_group().rank_in_group if self.dcp_world_size > 1 else 0
        self.pcp_shard_decode_requests = parallel_config.pcp_shard_decode_requests
        self._cp_kv_cache_interleave_size: int | None = None
        self.use_pcp = parallel_config.prefill_context_parallel_size > 1
        self.use_musa_native_indexer = use_musa_native_indexer
        self.use_musa_materialized_prefill = use_musa_materialized_prefill
        if current_platform.is_cuda() and not has_deep_gemm():
            raise RuntimeError(
                "Sparse Attention Indexer CUDA op requires DeepGEMM support in "
                "the current vLLM environment."
            )

        if vllm_config.kernel_config.enable_jit_warmup:
            from vllm.v1.attention.ops.common import (
                _PACK_SEQ_TRITON_KERNEL,
                _UNPACK_SEQ_TRITON_KERNEL,
            )

            pack_dtype = torch.uint8 if use_fp4_cache else current_platform.fp8_dtype()
            _PACK_SEQ_TRITON_KERNEL.register_warmup(
                dtype=pack_dtype,
                pad_value=0 if use_fp4_cache else -float("inf"),
            )
            _UNPACK_SEQ_TRITON_KERNEL.register_warmup()

            if self.dcp_world_size > 1 and current_platform.is_cuda() and has_cutedsl():
                from vllm.model_executor.kernels.attention.dsa.dcp_indexer_cutedsl import (  # noqa: E501
                    _PACK_DCP_TOPK_CANDIDATES_KERNEL,
                    _STABLE_TOPK_FROM_GATHERED_CANDIDATES_KERNEL,
                )

                _PACK_DCP_TOPK_CANDIDATES_KERNEL.register_warmup()
                _STABLE_TOPK_FROM_GATHERED_CANDIDATES_KERNEL.register_warmup()

    @property
    def cp_kv_cache_interleave_size(self) -> int:
        """With PD+DCP, the real value isn't known until block_size is finalized,
        which happens after this layer is built. Safe to cache after the first access,
        as long as the adjustment always runs before any forward pass
        (it's set up in Worker.initialize_from_config, ahead of warmup/serving).
        """
        if self._cp_kv_cache_interleave_size is None:
            value = self._parallel_config.cp_kv_cache_interleave_size
            if isinstance(get_forward_context().attn_metadata, dict):
                self._cp_kv_cache_interleave_size = value
            return value
        return self._cp_kv_cache_interleave_size

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor | None,
        weights: torch.Tensor,
    ):
        if current_platform.is_cuda() or current_platform.is_xpu():
            return self.forward_cuda(hidden_states, q_quant, k, weights)
        elif current_platform.is_rocm():
            if self.use_pcp and self.dcp_world_size > 1:
                raise NotImplementedError(
                    "The ROCm sparse-indexer path does not support PCP+DCP."
                )
            return self.forward_hip(hidden_states, q_quant, k, weights)
        elif current_platform.is_cpu():
            return self.forward_cpu(hidden_states, q_quant, k, weights)
        elif (
            current_platform.is_musa()
            and (
                self.use_musa_native_indexer
                or (
                    not isinstance(q_quant, tuple)
                    and q_quant.ndim == 3
                    and q_quant.shape[1] == 32
                    and os.getenv(
                        "VLLM_MUSA_ENABLE_GLM52_SPARSE_INDEXER_MUSA_IMPL",
                        "1",
                    )
                    == "1"
                )
                or os.getenv(
                    "VLLM_MUSA_ENABLE_TORCH_SPARSE_ATTN_INDEXER_FALLBACK",
                    "0",
                )
                == "1"
            )
        ):
            try:
                return _musa_fill_exact_sparse_indexer_indices(
                    hidden_states,
                    self.k_cache.prefix,
                    self.k_cache.kv_cache,
                    q_quant,
                    weights,
                    self.topk_tokens,
                    self.head_dim,
                    self.max_model_len,
                    self.topk_indices_buffer,
                    self.use_fp4_cache,
                    self.use_musa_materialized_prefill,
                )
            except NotImplementedError:
                raise
        else:
            raise NotImplementedError(
                "SparseAttnIndexer native forward is only implemented for "
                "CUDA, ROCm, XPU and CPU platforms."
            )

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor | None,
        weights: torch.Tensor,
    ):
        # FP8 path: single tensor (per-token scale is folded into `weights`).
        # FP4 path: (values, scales) tuple with scales required by the kernel.
        if isinstance(q_quant, tuple):
            q_values, q_scale = q_quant
        else:
            q_values, q_scale = q_quant, None
        return torch.ops.vllm.sparse_attn_indexer(
            hidden_states,
            _encode_layer_name(self.k_cache.prefix),
            self.k_cache.kv_cache,
            q_values,
            q_scale,
            k,
            weights,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
            self.skip_k_cache_insert,
            self.use_pcp,
            self.pcp_shard_decode_requests,
            _encode_layer_name(self.dense_mha_metadata_layer_name),
            self.use_fp4_cache,
            self.dcp_rank,
            self.dcp_world_size,
            self.cp_kv_cache_interleave_size,
            False,
            candidate_blocks=self.candidate_blocks,
            candidate_block_size=self.candidate_block_size,
            candidate_write=self.candidate_write,
            topk_backend=self.topk_backend,
        )

    def forward_xpu(
        self,
        hidden_states: torch.Tensor,
        q_fp8: torch.Tensor,
        k: torch.Tensor | None,
        weights: torch.Tensor,
    ):
        return self.forward_cuda(hidden_states, q_fp8, k, weights)

    def forward_hip(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor | None,
        weights: torch.Tensor,
    ):
        assert not self.use_fp4_cache, "AMD platform doesn't support fp4 cache yet"
        assert isinstance(q_quant, torch.Tensor), (
            "AMD sparse_attn_indexer expects a single FP8 q_quant tensor"
        )
        from vllm.platforms.rocm import on_gfx11, on_gfx950

        if (
            rocm_aiter_ops.is_enabled()
            or rocm_aiter_ops.is_rdna_aiter_enabled()
            or on_gfx11()
            # The so-called AITER sparse indexer op has a native gfx950 path:
            # its cache insert, MQA logits, and top-k fallbacks are implemented
            # by local Triton/C++ kernels and do not require the aiter package.
            or on_gfx950()
        ):
            return torch.ops.vllm.rocm_aiter_sparse_attn_indexer(
                hidden_states,
                _encode_layer_name(self.k_cache.prefix),
                self.k_cache.kv_cache,
                q_quant,
                k,
                weights,
                self.quant_block_size,
                self.scale_fmt,
                self.topk_tokens,
                self.head_dim,
                self.max_model_len,
                self.max_total_seq_len,
                self.topk_indices_buffer,
                skip_k_cache_insert=self.skip_k_cache_insert,
                compress_ratio=self.compress_ratio,
                candidate_blocks=self.candidate_blocks,
                candidate_block_size=self.candidate_block_size,
                candidate_write=self.candidate_write,
            )
        raise RuntimeError(
            "Sparse attention indexer ROCm path requires AITER or a supported "
            "native architecture (gfx950/gfx11)."
        )

    def forward_cpu(
        self,
        hidden_states: torch.Tensor,
        q_quant: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        k: torch.Tensor | None,
        weights: torch.Tensor,
    ):
        """CPU sparse attention indexer: cache write stays eager Python glue
        (own K-cache layout, not shared with the main attention cache
        write). PREFILL and DECODE both call the ported
        ``fp8_paged_mqa_logits_cpu``/``topk_transform_512_cpu`` kernels,
        which read the paged K-cache directly via ``page_table`` -- no
        eager gather step, no per-request Python loop.

        ``prefill_metadata.chunks`` always has exactly one entry here:
        ``DeepseekV4CPUIndexerMetadataBuilder`` overrides the base chunk
        split to always return the whole step's prefill batch as one
        chunk, since the base chunking only bounds CUDA/XPU's dense M*N
        logits tensor and flat K-gather workspace, neither of which this
        paged kernel allocates.
        """
        assert not self.use_fp4_cache, (
            "CPU sparse indexer doesn't support fp4 cache yet"
        )
        assert isinstance(q_quant, torch.Tensor), (
            "CPU sparse_attn_indexer expects a single FP8 q_quant tensor"
        )
        assert self.dcp_world_size <= 1 and not self.use_pcp, (
            "CPU sparse indexer doesn't support decode/prefill context parallelism yet."
        )

        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        attn_metadata_narrowed: DeepseekV32IndexerMetadata | None = None
        if isinstance(attn_metadata, dict):
            metadata = attn_metadata[self.k_cache.prefix]
            assert isinstance(metadata, DeepseekV32IndexerMetadata)
            attn_metadata_narrowed = metadata
        if attn_metadata_narrowed is None:
            # Profiling/dummy run: no real metadata to act on.
            return self.topk_indices_buffer

        kv_cache = self.k_cache.kv_cache
        topk_tokens = self.topk_tokens
        topk_indices_buffer = self.topk_indices_buffer
        slot_mapping = attn_metadata_narrowed.slot_mapping
        has_decode = attn_metadata_narrowed.num_decodes > 0
        has_prefill = attn_metadata_narrowed.num_prefills > 0
        num_decode_tokens = attn_metadata_narrowed.num_decode_tokens

        num_tokens = slot_mapping.shape[0]
        if k is not None:
            k = k[:num_tokens]

        if not self.skip_k_cache_insert:
            # Only reachable via DeepseekV32Attention with
            # prefill_context_parallel_size > 1 on CPU -- set_k_cpu/set_s_cpu
            # (the kernels this used to call) have been removed as unused/
            # untested (csrc/cpu/sgl-kernels/store_cache.cpp).
            raise NotImplementedError(
                "SparseAttnIndexer.forward_cpu: skip_k_cache_insert=False "
                "(prefill context parallel on CPU) is not supported."
            )

        topk_indices_buffer[: hidden_states.shape[0]] = -1

        if has_prefill:
            assert topk_tokens == 512, (
                "topk_transform_512_cpu only supports index_topk == 512."
            )
            prefill_metadata = attn_metadata_narrowed.prefill
            assert prefill_metadata is not None
            assert len(prefill_metadata.chunks) == 1, (
                "forward_cpu expects the prefill metadata builder to always "
                "produce a single chunk -- see "
                "DeepseekV4CPUIndexerMetadataBuilder._split_indexer_prefill_chunks."
            )
            chunk = prefill_metadata.chunks[0]
            # kv_cache is a per-layer view into vLLM's shared multi-layer
            # cache allocation, so its block stride generally exceeds
            # block_size * page_width; fp8_paged_mqa_logits_cpu reads
            # kv_view.stride(0) explicitly, so no copy is needed here.
            kv_view = kv_cache.view(kv_cache.shape[0], -1)
            block_size = kv_cache.shape[1]
            q_slice = q_quant[chunk.token_start : chunk.token_end]
            topk_indices = topk_indices_buffer[
                chunk.token_start : chunk.token_end, :topk_tokens
            ]
            if chunk.local_total_seq_lens == 0:
                topk_indices.fill_(-1)
            else:
                assert chunk.local_cu_seq_lens is not None
                # Each token's own (per-request, DCP-local) causal length.
                local_seq_lens = chunk.cu_seqlen_ke - chunk.cu_seqlen_ks
                # Recover each token's owning request from its row-start
                # tag. Ties (from zero-length requests) are harmless: those
                # tokens have local length 0 and never dereference
                # page_table.
                req_idx = (
                    torch.searchsorted(
                        chunk.local_cu_seq_lens, chunk.cu_seqlen_ks, right=True
                    )
                    - 1
                )
                page_table = chunk.block_table[req_idx]
                # The true max over this chunk's own rows, NOT
                # chunk.max_local_total_seq_lens (that field sums every
                # request's length in the chunk, bounding the old
                # flat-gather buffer this paged path no longer allocates).
                max_seq_len = int(local_seq_lens.max().item())

                logits = ops.fp8_paged_mqa_logits_cpu(
                    q_slice,
                    kv_view,
                    weights[chunk.token_start : chunk.token_end],
                    local_seq_lens,
                    page_table,
                    block_size,
                    max_seq_len,
                )
                out_page_scratch = torch.empty(
                    (q_slice.shape[0], topk_tokens),
                    dtype=torch.int32,
                    device=kv_cache.device,
                )
                ops.topk_transform_512_cpu(
                    logits,
                    local_seq_lens,
                    page_table,
                    out_page_scratch,
                    block_size,
                    topk_indices,
                )

        if has_decode:
            decode_metadata = attn_metadata_narrowed.decode
            assert decode_metadata is not None
            assert not decode_metadata.requires_padding, (
                "CPU sparse indexer decode path does not support speculative "
                "decoding (native MTP) yet."
            )
            batch_size = decode_metadata.decode_lens.shape[0]
            if batch_size > 0:
                # No native MTP on CPU (asserted above) => exactly one
                # query token per decode request, so the flat slice below
                # is already the batch-major layout
                # fp8_paged_mqa_logits_cpu wants.
                assert num_decode_tokens == batch_size, (
                    "CPU sparse indexer decode path expects exactly one query "
                    "token per decode request."
                )
                assert topk_tokens == 512, (
                    "topk_transform_512_cpu only supports index_topk == 512."
                )
                seq_lens = decode_metadata.seq_lens
                seq_lens = (
                    seq_lens[:, -1].contiguous() if seq_lens.ndim == 2 else seq_lens
                )
                block_table = decode_metadata.block_table[:batch_size]
                block_size = kv_cache.shape[1]
                kv_view = kv_cache.view(kv_cache.shape[0], -1)

                logits = ops.fp8_paged_mqa_logits_cpu(
                    q_quant[:num_decode_tokens],
                    kv_view,
                    weights[:num_decode_tokens],
                    seq_lens,
                    block_table,
                    block_size,
                    attn_metadata_narrowed.max_seq_len,
                )

                # out_page_indices is a required kernel output but unused:
                # the indexer's topk output must stay local/compressed-
                # context positions (resolved later by
                # DeepseekV4CPUAttention.forward_mqa via
                # map_local_to_global_slots_cpu).
                out_page_scratch = torch.empty(
                    (batch_size, topk_tokens),
                    dtype=torch.int32,
                    device=kv_cache.device,
                )
                ops.topk_transform_512_cpu(
                    logits,
                    seq_lens,
                    block_table,
                    out_page_scratch,
                    block_size,
                    topk_indices_buffer[:num_decode_tokens, :topk_tokens],
                )

        return topk_indices_buffer
