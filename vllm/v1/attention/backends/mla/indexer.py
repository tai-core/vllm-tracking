# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import torch

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed import get_dcp_group, get_pcp_group
from vllm.logger import init_logger
from vllm.model_executor.warmup.jit_warmup import (
    VllmJitKernel,
    WarmupIntRange,
)
from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    TritonPointerInputVariant,
    TritonWarmupTensor,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.deep_gemm import (
    get_paged_mqa_logits_metadata,
    has_deep_gemm,
    native_next_n_supported,
)
from vllm.utils.platform_utils import num_compute_units
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.mla.compressor_utils import get_compressed_slot_mapping
from vllm.v1.attention.backends.utils import (
    get_dcp_local_seq_lens,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheLayout,
    KVCacheSpec,
    MLAAttentionSpec,
)

logger = init_logger(__name__)

# The DSA indexer K cache is always quantized; "auto" means fp8 (V3.2 layout)
# and mxfp4 is the opt-in Blackwell path.
DSA_INDEXER_KV_DTYPES = ("fp8", "mxfp4")


def dsa_indexer_uses_fp4(vllm_config: VllmConfig) -> bool:
    """Whether the DeepSeek sparse indexer should use the MXFP4 K cache."""
    kv_dtype = vllm_config.attention_config.resolve_indexer_kv_dtype("fp8")
    if kv_dtype not in DSA_INDEXER_KV_DTYPES:
        raise ValueError(
            f"indexer_kv_dtype={kv_dtype!r} is not supported by the DeepSeek "
            f"sparse indexer (expected one of {DSA_INDEXER_KV_DTYPES})."
        )
    use_fp4 = kv_dtype == "mxfp4"
    if use_fp4 and not current_platform.is_device_capability_family(100):
        raise ValueError(
            "indexer_kv_dtype='mxfp4' requires Blackwell datacenter GPUs "
            "(sm_10x, e.g. B200/GB200); sm_120 (consumer Blackwell) and "
            "earlier architectures are not supported."
        )
    return use_fp4


@triton.jit
def _prepare_uniform_decode_kernel(
    seq_lens_ptr,
    decode_seq_lens_ptr,
    block_table_ptr,
    block_table_stride,
    expanded_block_table_ptr,
    expanded_bt_stride,
    decode_lens_ptr,
    max_decode_len,
    BLOCK_SIZE: tl.constexpr,
):
    idx = tl.program_id(0)
    req_id = idx // max_decode_len
    local_idx = idx % max_decode_len

    # Compute number of KVs attended to by this token. Padding requests have
    # seq_len == 0, which would otherwise make the first token of each padded
    # request negative (e.g. next_n=2 gives 0-2+0+1 = -1). Downstream kernels
    # read these as uint32, turning -1 into ~4e9.
    seq_len = tl.load(seq_lens_ptr + req_id)
    per_token_seq_len = tl.maximum(seq_len - max_decode_len + local_idx + 1, 0)
    tl.store(decode_seq_lens_ptr + idx, per_token_seq_len)

    # Copy block table row.
    src = block_table_ptr + req_id * block_table_stride
    dst = expanded_block_table_ptr + idx * expanded_bt_stride
    for i in tl.range(0, expanded_bt_stride, BLOCK_SIZE):
        off = i + tl.arange(0, BLOCK_SIZE)
        mask = off < expanded_bt_stride
        src_block = tl.load(src + off, mask=mask)
        tl.store(dst + off, src_block, mask=mask)

    # All reqs now have decode_len = 1.
    tl.store(decode_lens_ptr + idx, 1)


def split_indexer_prefill_chunks(
    seq_lens_cpu: torch.Tensor,
    query_lens_cpu: torch.Tensor,
    workspace_size: int,
    max_logits_bytes: int,
    request_offset: int = 0,
) -> list[tuple[slice, slice]]:
    """
    Split prefill requests into chunks for the sparse indexer, respecting:
    - N constraint: total_seq_lens <= workspace_size (existing O(N) workspace)
    - Logits constraint: M * N * 4 <= max_logits_bytes

    When a single request-level chunk still exceeds the logits budget,
    sub-chunks on the query dimension (M) to bound peak memory.

    Returns list of (req_slice, query_slice) tuples.
    """
    chunks: list[tuple[slice, slice]] = []
    n = len(seq_lens_cpu)
    max_logits_elems = max_logits_bytes // 4
    end = 0

    while end < n:
        start, chunk_m, chunk_n = end, 0, 0

        while end < n:
            q, s = query_lens_cpu[end].item(), seq_lens_cpu[end].item()
            new_m, new_n = chunk_m + q, chunk_n + s
            if new_n <= workspace_size and new_m * new_n <= max_logits_elems:
                chunk_m, chunk_n = new_m, new_n
                end += 1
            else:
                break

        # A single request can exceed the budget, requiring sub-chunking
        # on the query dimension.
        if end == start:
            chunk_m, chunk_n = query_lens_cpu[end].item(), seq_lens_cpu[end].item()
            end += 1

        req_slice = slice(start + request_offset, end + request_offset)
        max_q = max(1, max_logits_elems // chunk_n) if chunk_n > 0 else max(1, chunk_m)
        for q_off in range(0, chunk_m, max_q):
            sub_m = min(max_q, chunk_m - q_off)
            chunks.append((req_slice, slice(q_off, q_off + sub_m)))

    return chunks


class DeepseekV32IndexerBackend(AttentionBackend):
    @classmethod
    def supports_pcp(cls) -> bool:
        return True

    @classmethod
    def supports_device_cpu_query_lens_mismatch(cls) -> bool:
        # Only the varlen paged MQA logits kernel takes per-request query
        # lengths from device tensors natively. Hopper can instead flatten each
        # query into a single-token row using device-built metadata.
        return _supports_varlen_paged_mqa_logits() or (
            _supports_flattened_device_query_lens()
        )

    @staticmethod
    def get_name() -> str:
        return "DEEPSEEK_V32_INDEXER"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [1, MultipleOf(16)] if current_platform.is_rocm() else [64]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [32, 64, 128]

    @staticmethod
    def get_builder_cls() -> type["DeepseekV32IndexerMetadataBuilder"]:
        return DeepseekV32IndexerMetadataBuilder


class KpoolTailBackend(DeepseekV32IndexerBackend):
    """Storage-only backend for the GLM-5.3-Flash kpool tail cache."""

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        return (KVCacheLayout.LBHNC,)

    @staticmethod
    def get_name() -> str:
        return "KPOOL_TAIL"

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return []

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(1)]

    @staticmethod
    def get_builder_cls() -> type["KpoolTailMetadataBuilder"]:  # type: ignore[override]
        return KpoolTailMetadataBuilder


class DeepseekV4IndexerBackend(DeepseekV32IndexerBackend):
    @staticmethod
    def get_name() -> str:
        return "DEEPSEEK_V4_INDEXER"

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        # DeepSeek-V4 packs the indexer pages beside the MLA latent pages inside
        # each block, so the layer dim must sit inside the block dim.
        return (KVCacheLayout.BLHNC, KVCacheLayout.BLNHC)

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [256]


@dataclass
class DeepseekV32IndexerPrefillChunkMetadata:
    block_table: torch.Tensor
    # Under DCP (dcp_world_size > 1) these hold this rank's local row bounds;
    # otherwise they hold the global bounds.
    cu_seqlen_ks: torch.Tensor
    cu_seqlen_ke: torch.Tensor
    cu_seq_lens: torch.Tensor
    token_to_seq: torch.Tensor
    total_seq_lens: int
    token_start: int
    token_end: int
    num_reqs: int
    skip_kv_gather: bool = False
    local_cu_seq_lens: torch.Tensor | None = None
    local_total_seq_lens: int = 0
    max_local_total_seq_lens: int = 0
    # Per-LOCAL-pool request index [local_total_seq_lens] int32 under
    # DCP (each rank-local pool -> its request's index within the
    # chunk); ALIASES token_to_seq when DCP is off. Legacy manual
    # metadata may leave this None — DCP>1 consumers MUST reject None
    # instead of silently using the global map.
    local_token_to_seq: torch.Tensor | None = None
    # The builder's ACTUAL DCP world size for this chunk (explicit
    # topology; default 1 keeps legacy manual metadata on the W1
    # path). Consumers must never infer parallelism from tensor
    # identity/aliasing — a W1 clone of cu_seq_lens is still W1, and
    # a DCP>1 chunk missing the local fields is a hard error.
    dcp_world_size: int = 1
    # The builder's ACTUAL DCP rank and POOL-space interleave J for
    # this chunk (trusted fields, same provenance as
    # dcp_world_size; defaults 0/1 keep legacy W1 metadata valid).
    # The merge consumes these instead of re-deriving them.
    dcp_rank: int = 0
    pool_interleave: int = 1


_BUILD_PREFILL_CHUNK_METADATA_INPUT_VARIANTS = (
    TritonPointerInputVariant.from_alignment(uncompressed_seq_lens=True),
    TritonPointerInputVariant.from_alignment(uncompressed_seq_lens=False),
)


class BuildPrefillChunkMetadataKernel(
    VllmJitKernel["BuildPrefillChunkMetadataKernel.CompileKey"]
):
    BLOCK_SIZE = 1024

    @dataclass(frozen=True)
    class CompileKey:
        query_slice_start: int
        query_slice_stop: int
        DCP_RANK: int
        DCP_WORLD: int
        DCP_INTERLEAVE: int
        BLOCK_SIZE: int
        COMPRESS_RATIO: int
        input_variant: TritonPointerInputVariant

    @staticmethod
    @triton.jit
    def kernel(
        # Inputs
        query_start_loc_ptr,
        uncompressed_seq_lens_ptr,
        cu_compressed_seq_lens_ptr,
        # Row-start base for cu_seq_len_ks/ke: local cumulative lens under DCP,
        # aliases cu_compressed_seq_lens_ptr otherwise.
        row_start_cu_compressed_seq_lens_ptr,
        # Outputs
        token_to_seq_ptr,
        cu_compressed_seq_len_ks_ptr,
        cu_compressed_seq_len_ke_ptr,
        query_slice_start,
        query_slice_stop,
        DCP_RANK,
        DCP_WORLD,
        DCP_INTERLEAVE,
        BLOCK_SIZE: tl.constexpr,
        COMPRESS_RATIO: tl.constexpr,
    ):
        batch_idx = tl.program_id(0)

        query_start = tl.load(query_start_loc_ptr + batch_idx)
        query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
        query_len = query_end - query_start

        seq_start = tl.load(cu_compressed_seq_lens_ptr + batch_idx)
        seq_end = tl.load(cu_compressed_seq_lens_ptr + batch_idx + 1)
        compressed_seq_len = seq_end - seq_start

        # Row start for the (possibly localized) cu_seq_len_ks/ke. Equals seq_start
        # when DCP is disabled (the pointer aliases cu_compressed_seq_lens_ptr).
        row_start = tl.load(row_start_cu_compressed_seq_lens_ptr + batch_idx)

        uncompressed_seq_len = tl.load(uncompressed_seq_lens_ptr + batch_idx)
        start_pos = uncompressed_seq_len - query_len

        for i in range(0, query_len, BLOCK_SIZE):
            offset = i + tl.arange(0, BLOCK_SIZE)
            abs_pos = query_start + offset
            mask = (
                (offset < query_len)
                & (abs_pos >= query_slice_start)
                & (abs_pos < query_slice_stop)
            )
            out_pos = abs_pos - query_slice_start

            # cu_seq_len_ks: row start in the gathered K buffer.
            tl.store(cu_compressed_seq_len_ks_ptr + out_pos, row_start, mask=mask)

            # cu_seq_len_ke: row start + per-token context length. Under DCP the
            # global per-token length is sharded across ranks.
            global_ctx = start_pos + 1 + offset
            len_per_token = global_ctx // COMPRESS_RATIO
            if DCP_WORLD > 1:
                # Per-rank local context length under interleave-aware DCP, matching
                # get_dcp_local_seq_lens. K == 1 reduces to (len + world-1-rank)//world.
                base = (len_per_token // DCP_INTERLEAVE // DCP_WORLD) * DCP_INTERLEAVE
                remainder = len_per_token - base * DCP_WORLD
                remainder = tl.minimum(
                    tl.maximum(remainder - DCP_RANK * DCP_INTERLEAVE, 0), DCP_INTERLEAVE
                )
                len_per_token = base + remainder
            tl.store(
                cu_compressed_seq_len_ke_ptr + out_pos,
                row_start + len_per_token,
                mask=mask,
            )

        # Compute token_to_seq
        for i in range(0, compressed_seq_len, BLOCK_SIZE):
            offset = i + tl.arange(0, BLOCK_SIZE)
            mask = offset < compressed_seq_len
            tl.store(token_to_seq_ptr + seq_start + offset, batch_idx, mask=mask)

    def dispatch(  # type: ignore[override]
        self,
        *,
        query_slice_start: int,
        query_slice_stop: int,
        DCP_RANK: int,
        DCP_WORLD: int,
        DCP_INTERLEAVE: int,
        BLOCK_SIZE: int,
        COMPRESS_RATIO: int,
        input_variant: TritonPointerInputVariant,
    ) -> CompileKey:
        return self.CompileKey(
            query_slice_start=query_slice_start,
            query_slice_stop=query_slice_stop,
            DCP_RANK=DCP_RANK,
            DCP_WORLD=DCP_WORLD,
            DCP_INTERLEAVE=DCP_INTERLEAVE,
            BLOCK_SIZE=BLOCK_SIZE,
            COMPRESS_RATIO=COMPRESS_RATIO,
            input_variant=input_variant,
        )

    def get_warmup_keys(self, vllm_config: VllmConfig) -> list[CompileKey]:
        max_tokens = max(1, min(vllm_config.scheduler_config.max_num_batched_tokens, 8))
        hf_config = vllm_config.model_config.hf_config
        parallel_config = vllm_config.parallel_config
        dcp_world = parallel_config.decode_context_parallel_size
        dcp_interleave = parallel_config.cp_kv_cache_interleave_size
        dcp_rank = get_dcp_group().rank_in_group if dcp_world > 1 else 0
        compress_ratios = tuple(
            max(1, int(ratio))
            for ratio in (getattr(hf_config, "compress_ratios", None) or (1,))
        )
        index_kpool = getattr(hf_config, "index_kpool", None)
        if index_kpool and index_kpool > 1 and index_kpool not in compress_ratios:
            compress_ratios = compress_ratios + (index_kpool,)
        # Each ratio pairs with its OWN pool-space interleave
        # J = I // R so the warmup compile keys match the runtime
        # derivation exactly (no wrong-unit key is cached). An
        # unsupported (I, R, W) combination raises here — fail-closed
        # — rather than warming a key the runtime would reject.
        warmups = []
        for ratio in compress_ratios:
            pool_interleave = derive_pool_interleave(dcp_interleave, ratio, dcp_world)
            warmups.extend(
                self._trace_dispatch(self.dispatch)(
                    query_slice_start=WarmupIntRange(0, 2),
                    query_slice_stop=(1, 2 * max_tokens - 1, 2 * max_tokens),
                    DCP_RANK=dcp_rank,
                    DCP_WORLD=dcp_world,
                    DCP_INTERLEAVE=pool_interleave,
                    BLOCK_SIZE=self.BLOCK_SIZE,
                    COMPRESS_RATIO=[ratio],
                    input_variant=_BUILD_PREFILL_CHUNK_METADATA_INPUT_VARIANTS,
                )
            )
        return warmups

    def compile(self, compile_key: CompileKey) -> None:
        warmup = getattr(self.kernel, "warmup", None)
        assert warmup is not None
        int32_ptr = TritonWarmupTensor(torch.int32)
        warmup(
            int32_ptr,
            compile_key.input_variant.pointer("uncompressed_seq_lens", torch.int32),
            int32_ptr,
            int32_ptr,
            int32_ptr,
            int32_ptr,
            int32_ptr,
            compile_key.query_slice_start,
            compile_key.query_slice_stop,
            compile_key.DCP_RANK,
            compile_key.DCP_WORLD,
            compile_key.DCP_INTERLEAVE,
            BLOCK_SIZE=compile_key.BLOCK_SIZE,
            COMPRESS_RATIO=compile_key.COMPRESS_RATIO,
            grid=(1,),
        )

    def __call__(
        self,
        query_start_loc: torch.Tensor,
        uncompressed_seq_lens: torch.Tensor,
        cu_compressed_seq_lens: torch.Tensor,
        row_start_cu_compressed_seq_lens: torch.Tensor,
        token_to_seq: torch.Tensor,
        cu_compressed_seq_len_ks: torch.Tensor,
        cu_compressed_seq_len_ke: torch.Tensor,
        query_slice_start: int,
        query_slice_stop: int,
        DCP_RANK: int,
        DCP_WORLD: int,
        DCP_INTERLEAVE: int,
        *,
        num_reqs: int,
        COMPRESS_RATIO: int,
    ) -> None:
        self.kernel[(num_reqs,)](
            query_start_loc,
            uncompressed_seq_lens,
            cu_compressed_seq_lens,
            row_start_cu_compressed_seq_lens,
            token_to_seq,
            cu_compressed_seq_len_ks,
            cu_compressed_seq_len_ke,
            query_slice_start,
            query_slice_stop,
            DCP_RANK,
            DCP_WORLD,
            DCP_INTERLEAVE,
            BLOCK_SIZE=self.BLOCK_SIZE,
            COMPRESS_RATIO=COMPRESS_RATIO,
        )


_BUILD_PREFILL_CHUNK_METADATA_KERNEL = BuildPrefillChunkMetadataKernel()


@dataclass
class DeepseekV32IndexerPrefillMetadata:
    chunks: list[DeepseekV32IndexerPrefillChunkMetadata]
    max_prefill_seq_len: int = -1


@dataclass
class DeepSeekV32IndexerDecodeMetadata:
    block_table: torch.Tensor
    # seq_lens: per-token effective context lengths.
    #   - flatten path / plain decode: 1D (batch_size,)
    #   - native MTP path: 2D (B, next_n) where [b,j] = L_b - next_n + j + 1
    # Both fp8_fp4_paged_mqa_logits and the topk kernels accept both shapes.
    seq_lens: torch.Tensor
    decode_lens: torch.Tensor
    requires_padding: bool
    schedule_metadata: torch.Tensor
    global_seq_lens: torch.Tensor | None = None
    per_req_decode_lens: torch.Tensor | None = None
    decode_is_uniform: bool = True
    write_max_decode_len: int = 0
    indices: torch.Tensor | None = None
    # Trusted DCP topology for the decode merge (same provenance as
    # the prefill chunk fields; defaults 1/0/1 keep legacy W1
    # metadata valid). Consumers must never infer parallelism from
    # tensor shapes or object identity.
    dcp_world_size: int = 1
    dcp_rank: int = 0
    pool_interleave: int = 1


@dataclass
class DeepseekV32IndexerMetadata:
    # FIXME (zyongye)
    # hacky way to access the data now, need to be in chunked meta
    seq_lens: torch.Tensor
    max_seq_len: int
    slot_mapping: torch.Tensor

    # New for MLA (compared to FlashAttention)
    # For handling prefill decode split
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int

    decode: DeepSeekV32IndexerDecodeMetadata | None = None
    prefill: DeepseekV32IndexerPrefillMetadata | None = None


def compute_kpool_tail_slot_mapping(
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
    num_actual_tokens: int,
    num_reqs: int,
    tail_capacity: int | None = None,
    out: torch.Tensor | None = None,
    *,
    kpool: int | None = None,
) -> torch.Tensor:
    """Map every token to its request's one circular tail block.

    ``tail_capacity`` is the PHYSICAL ring capacity C (the tail
    spec's block_size — NOT the logical pool size): the slot encodes
    ``physical_block * C + pos % C``, matching the seed/decode
    kernels' tslot decode. Each request uses its own block, so
    tokens never cross requests.

    Validity is RECOMPUTED per token: a token maps to a real slot
    only if its position is non-negative AND its request resolves to
    a valid row with a non-negative block id; everything else
    (padded positions, invalid requests, negative block ids) maps to
    -1 — never a bogus address formed from a negative remainder.
    The incoming ``slot_mapping``'s sign is NOT trusted: the generic
    value may be -1 for unrelated reasons (other shards, compressed
    slots) while the tail needs a REAL slot for every real token,
    and conversely a stale positive generic value must not leak
    through for a padded token.
    """
    # Legacy keyword alias: `kpool` was the parameter's old name
    # (it always meant the tail allocation size — the physical
    # capacity). Both may be given only if they agree.
    if kpool is not None:
        if tail_capacity is not None and kpool != tail_capacity:
            raise ValueError(
                f"conflicting tail capacities: tail_capacity="
                f"{tail_capacity} vs kpool={kpool}"
            )
        tail_capacity = kpool
    if tail_capacity is None:
        raise ValueError("tail_capacity (or the legacy kpool) is required")
    if type(tail_capacity) is not int or tail_capacity < 1:
        raise ValueError(
            f"tail_capacity must be a positive exact int, got {tail_capacity!r}"
        )
    if out is None:
        out = slot_mapping.clone()
    else:
        assert out.shape == slot_mapping.shape
        out.copy_(slot_mapping)
    if num_actual_tokens == 0:
        return out
    if num_reqs <= 0 or block_table.shape[0] == 0:
        # No valid requests: every token maps to -1 (never an
        # index_select on an empty table).
        out[:num_actual_tokens] = -1
        return out
    tokens = torch.arange(num_actual_tokens, device=slot_mapping.device)
    raw_req = torch.searchsorted(query_start_loc, tokens, right=True) - 1
    # Validity from the RAW request index (BEFORE the clamp): a
    # token beyond the last request's range belongs to no request
    # and must not inherit the clamped row's block.
    req_valid = (raw_req >= 0) & (raw_req < num_reqs)
    safe_req = raw_req.clamp_(min=0, max=num_reqs - 1)
    own_block = block_table[:num_reqs, 0].index_select(0, safe_req).to(torch.int64)
    pos = positions[:num_actual_tokens].to(torch.int64)
    valid = (pos >= 0) & req_valid & (own_block >= 0)
    computed = own_block * tail_capacity + torch.remainder(pos, tail_capacity)
    out[:num_actual_tokens] = torch.where(
        valid, computed, torch.full_like(computed, -1)
    )
    return out


class KpoolTailMetadataBuilder(AttentionMetadataBuilder):
    """Build only the circular slot mapping needed by the storage-only tail."""

    _cudagraph_support = AttentionCGSupport.ALWAYS
    supports_update_block_table = False
    reorder_batch_threshold = None

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.slot_mapping_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            dtype=torch.int64,
            device=device,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> DeepseekV32IndexerMetadata:
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(common_attn_metadata)
        )
        slot_mapping = common_attn_metadata.slot_mapping
        positions = common_attn_metadata.positions
        if positions is not None:
            slot_mapping_buffer = self.slot_mapping_buffer[
                : slot_mapping.numel()
            ].view_as(slot_mapping)
            # kv_cache_spec.block_size IS the physical tail capacity
            # C (the spec's block_size follows the ring allocation);
            # the logical pool size never enters the slot mapping.
            slot_mapping = compute_kpool_tail_slot_mapping(
                slot_mapping,
                common_attn_metadata.block_table_tensor,
                common_attn_metadata.query_start_loc,
                positions,
                common_attn_metadata.num_actual_tokens,
                common_attn_metadata.num_reqs,
                self.kv_cache_spec.block_size,
                out=slot_mapping_buffer,
            )
        return DeepseekV32IndexerMetadata(
            seq_lens=common_attn_metadata.seq_lens,
            max_seq_len=common_attn_metadata.max_seq_len,
            slot_mapping=slot_mapping,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
        )


def get_max_prefill_buffer_size(vllm_config: VllmConfig):
    max_model_len = vllm_config.model_config.max_model_len
    # NOTE(Chen): 40 is a magic number for controlling the prefill buffer size.
    # Each entry is 128 fp8 bytes and 4 scale bytes for a total of 132 bytes.
    # The flashmla_sparse backend uses a workspace size of 5 * max_model_len.
    # The memory usage of the workspace there is 576 * 2 bytes; so we size this as
    # (576 * 2 // 132) * 5 = 40 to maximize this workspace size while still fitting
    # within the flashmla_sparse workspace.
    # For DeepSeek-V3.2, the max_model_len is 163840.
    #   40 * 163840 * 132 = 865075200 bytes = 825 MB
    return max_model_len * 40


def _supports_varlen_paged_mqa_logits() -> bool:
    return (
        current_platform.is_cuda()
        and current_platform.is_device_capability_family(100)
        and has_deep_gemm()
    )


def _supports_flattened_device_query_lens() -> bool:
    return (
        current_platform.is_cuda()
        and current_platform.is_device_capability_family(90)
        and has_deep_gemm()
    )


def _supports_native_decode(next_n: int) -> bool:
    """Whether decode can pass `next_n` Q rows per request to the kernel
    instead of flattening to one single-token row per query, which re-reads
    the KV tile once per row.
    """
    if not (current_platform.is_cuda() and has_deep_gemm()):
        return next_n in (1, 2)
    if current_platform.is_device_capability_family(100):
        return True
    if current_platform.is_device_capability_family(90):
        return native_next_n_supported(next_n)
    return next_n in (1, 2)


# Backend capability flag: the wired HCU plugin's indexer patch may set
# this on the builder CLASS (never an instance field, never a user
# parameter) to attest that THIS deployment's compressed-indexer DCP
# implementation (merge primitives, kernels, collectives) is present.
# Default False everywhere in core: the DCP+compression combination
# below stays rejected unless the flag is explicitly granted.
SUPPORTS_GLM_KPOOL_BF16_DCP = False
# The FP8 MLA + compressed-indexer opt-in: DEFAULT FALSE in core. Only a
# matching HCU plugin indexer patch may set this True on the class (it
# attests the wired FP8 DS-656 MLA storage with the compressed indexer:
# the rank-local writer/reader, the collectives, and the runtime gate
# this worktree validated). When True, the validator below admits the
# global MLA cache_dtype 'fp8_ds_mla' IN ADDITION to 'bfloat16' for the
# restricted candidate; the INDEXER's own spec format is checked
# separately (the actual registered cache's spec — never assumed from
# the global dtype). NOT a user parameter.
SUPPORTS_GLM_KPOOL_FP8_MLA_DCP = False


def validate_dcp_compressed_indexer_support(
    vllm_config: VllmConfig,
    compress_ratio: int,
    cp_kv_cache_interleave_size: int,
    *,
    capability_enabled: bool,
    actual_spec: MLAAttentionSpec | None = None,
    fp8_mla_enabled: bool = False,
) -> None:
    """The capability predicate replacing the two blanket DCP guards.

    Only the fully-validated combination passes — DCP>1 together with
    compressed (R>1) or interleaved (I>1) indexer KV requires ALL of:
    the backend capability flag (set only by the wired HCU plugin
    patch), a REAL ``Glm5NextTextConfig`` text config (an
    ``isinstance`` check against the imported real class — resolved
    through ``model_config.hf_text_config``; a same-named unrelated
    class is NOT accepted), ``index_kpool_compress`` enabled with
    ``index_kpool == R > 1``, PCP==1, pool-coherent interleave
    (I >= R, I % R == 0), the ACTUAL spec layout proof (when
    ``actual_spec`` is given — the constructor always passes it —
    ``actual_spec.block_size % I == 0`` and
    ``actual_spec.tokens_per_state == R``; the cache_config
    ``block_size`` alone is the CONFIG value, not a layout proof), an
    explicitly authorized MLA KV ``cache_dtype == "bfloat16"``
    ("auto" is rejected: resolving it requires the model dtype and
    is NOT proven here), no ``kv_transfer_config`` (single-node
    candidate), and a RESOLVED non-negative MTP ``S`` (a present
    speculative_config with ``num_speculative_tokens is None`` is
    rejected with a named reason — it would fail later in the tail
    helper after buffers are allocated; no speculative_config means
    S = 0 and is legal) with the tail capacity derived by the
    existing ``Glm5NextTailCache`` contract (``C = R + S`` when
    DCP>1). Everything else — V4, plain NVIDIA, FP8 MLA, PCP>1,
    transfer, misaligned I, capability flag False — fails CLOSED
    with a named reason. DCP1 keeps the legacy behavior for ANY
    (I, R) and returns immediately (no rejection path).
    """
    parallel_config = vllm_config.parallel_config
    dcp_world = parallel_config.decode_context_parallel_size
    if dcp_world <= 1:
        # DCP1: legacy behavior for any (I, R); nothing to validate.
        return
    interleave = cp_kv_cache_interleave_size
    if interleave <= 1 and compress_ratio <= 1:
        # DCP without compression/interleave: the long-supported path.
        return
    # --- DCP>1 with compression (R>1) and/or interleave (I>1): the
    # restricted capability zone. Fail closed with named reasons.

    def _reject(reason: str) -> None:
        raise NotImplementedError(
            "DCP sparse indexer with compressed/interleaved KV requires "
            f"the GLM kpool BF16 DCP capability (all conditions must hold); "
            f"rejected: {reason}."
        )

    if not capability_enabled:
        _reject(
            "backend capability flag supports_glm_kpool_bf16_dcp is not "
            "granted (default False; only the wired HCU plugin indexer "
            "patch may set it)"
        )
    model_config = vllm_config.model_config
    text_config = getattr(model_config, "hf_text_config", None)
    from vllm.transformers_utils.configs.glm5_next import Glm5NextTextConfig

    if (
        not isinstance(text_config, Glm5NextTextConfig)
        or getattr(text_config, "model_type", None) != "glm5_next_text"
    ):
        # The nested/real text config must BE the real GLM text config
        # CLASS (isinstance — a same-named fabricated type is not
        # accepted; hf_text_config resolves the outer glm5_next
        # wrapper). Real HF subclasses pass per HF semantics.
        _reject(
            "model text config is not the real Glm5NextTextConfig "
            f"(got {type(text_config).__name__}, model_type="
            f"{getattr(text_config, 'model_type', None)!r})"
        )
    if getattr(text_config, "index_kpool_compress", False) is not True:
        _reject("index_kpool_compress is not enabled")
    model_kpool = getattr(text_config, "index_kpool", None)
    if (
        type(model_kpool) is not int
        or model_kpool != compress_ratio
        or model_kpool <= 1
    ):
        _reject(
            f"index_kpool ({model_kpool!r}) does not match the cache "
            f"compress_ratio ({compress_ratio!r}) or is not > 1"
        )
    if parallel_config.prefill_context_parallel_size != 1:
        _reject(
            "prefill_context_parallel_size must be 1 (got "
            f"{parallel_config.prefill_context_parallel_size})"
        )
    # Pool-coherent interleave: derive_pool_interleave is the single
    # source of truth for I>=R and I%R==0 (it raises ValueError with
    # the exact reason).
    derive_pool_interleave(interleave, compress_ratio, dcp_world)
    block_size = vllm_config.cache_config.block_size
    if type(block_size) is not int or block_size % interleave != 0:
        _reject(
            f"cache block_size ({block_size!r}) is not divisible by "
            f"cp_kv_cache_interleave_size ({interleave!r}) — the storage "
            "layout is incompatible with interleaved DCP sharding"
        )
    if actual_spec is not None:
        # The ACTUAL spec layout proof (the constructor always passes
        # the spec it will allocate against): the spec's own block
        # size must tile the interleave, and its tokens_per_state
        # must agree with the compress_ratio the caller derived. The
        # cache_config block_size above is the CONFIG value; this is
        # the LAYOUT the buffers will actually use.
        spec_block = actual_spec.block_size
        if type(spec_block) is not int or spec_block % interleave != 0:
            _reject(
                f"actual kv_cache_spec block_size ({spec_block!r}) is not "
                f"divisible by cp_kv_cache_interleave_size "
                f"({interleave!r}) — the allocated layout is incompatible "
                "with interleaved DCP sharding"
            )
        spec_tps = actual_spec.tokens_per_state
        if type(spec_tps) is not int or spec_tps != compress_ratio:
            _reject(
                f"actual kv_cache_spec tokens_per_state ({spec_tps!r}) "
                f"does not match the compress_ratio ({compress_ratio!r})"
            )
    cache_dtype = vllm_config.cache_config.cache_dtype
    # The GLOBAL MLA cache dtype: 'bfloat16' (the BF16 candidate) or,
    # ONLY under the separate FP8 MLA opt-in, 'fp8_ds_mla' (the FP8
    # DS-656 MLA storage with the compressed indexer — the global MLA
    # dtype, DISTINCT from the indexer's own spec format checked
    # separately below). "auto" resolution is NOT proven here (it
    # depends on the model dtype path) — fail closed, named.
    if cache_dtype == "fp8_ds_mla":
        if not fp8_mla_enabled:
            _reject(
                "cache_dtype 'fp8_ds_mla' requires the separate FP8 MLA "
                "opt-in (SUPPORTS_GLM_KPOOL_FP8_MLA_DCP, default False; "
                "only the wired HCU plugin indexer patch may set it)"
            )
        # The FP8 grant's OWN restrictions: the EXACT single-module
        # MTP4 profile (method 'mtp', resolved int S == 4,
        # parallel_drafting False, draft_model_config present,
        # use_multi_module_mtp() False — the OBSERVED production
        # MTP4 class is the single-module MTPSpeculator; the class
        # is NEVER guessed from S=4 alone and MultiModule stays
        # rejected for FP8; the draft KV override must be None or
        # fp8_ds_mla), or NO speculative config at all (the verified
        # MTP-off FP8 candidate — unchanged), the ag_rs comm
        # backend, eager compilation, and no KV transfer (the last
        # re-checked below for BOTH dtypes).
        from vllm.v1.attention.backends.mla.dcp_mtp_capability import (
            _fp8_single_module_mtp4_profile,
        )

        if vllm_config.speculative_config is not None and not (
            _fp8_single_module_mtp4_profile(vllm_config)
        ):
            _reject(
                "cache_dtype 'fp8_ds_mla' with a speculative config "
                "requires the EXACT single-module MTP4 profile "
                "(method 'mtp', resolved int S == 4, "
                "parallel_drafting False, draft_model_config present, "
                "use_multi_module_mtp() False, draft KV override None "
                "or fp8_ds_mla); every other MTP shape (MultiModule, "
                "other S, parallel drafting, wrong method/override) "
                "stays rejected"
            )
        if parallel_config.dcp_comm_backend != "ag_rs":
            _reject(
                "cache_dtype 'fp8_ds_mla' requires the default 'ag_rs' DCP "
                f"comm backend (got {parallel_config.dcp_comm_backend!r})"
            )
        from vllm.config import CompilationMode

        if vllm_config.compilation_config.mode != CompilationMode.NONE:
            _reject(
                "cache_dtype 'fp8_ds_mla' requires eager compilation "
                "(CompilationMode.NONE)"
            )
        if vllm_config.kv_transfer_config is not None:
            _reject(
                "cache_dtype 'fp8_ds_mla' requires no KV transfer config "
                "(the candidate is single-node)"
            )
    elif cache_dtype != "bfloat16":
        _reject(
            f"cache_dtype must be explicitly 'bfloat16' for this candidate "
            f"(got {cache_dtype!r}; 'auto' is not resolved to BF16 here; "
            f"'fp8_ds_mla' requires the separate FP8 opt-in)"
        )
    if vllm_config.kv_transfer_config is not None:
        _reject("kv_transfer_config is set (the candidate is single-node)")
    speculative_config = vllm_config.speculative_config
    if speculative_config is None:
        num_spec = 0  # no spec config: S = 0 is legal
    else:
        num_spec = speculative_config.num_speculative_tokens
        if num_spec is None:
            # A present-but-unresolved speculative config would fail
            # later in the tail helper (after buffers are allocated);
            # reject HERE with the named reason instead.
            _reject(
                "speculative_config is present but "
                "num_speculative_tokens is not resolved (None)"
            )
        if type(num_spec) is not int or num_spec < 0:
            _reject(f"num_speculative_tokens is not a non-negative int ({num_spec!r})")
    # The tail capacity C = R + S is derived by Glm5NextTailCache from
    # the SAME vllm_config (its own DCP>1 branch); the 1-block copy
    # strategy and C>R-with-transfer rejection live THERE. This
    # validator does not re-derive or fake-check it.


def _use_flattening(vllm_config: VllmConfig) -> bool:
    speculative_config = vllm_config.speculative_config
    next_n = 1 + vllm_config.num_speculative_tokens
    return not _supports_native_decode(next_n) or (
        speculative_config is not None
        and speculative_config.enable_adaptive_verification
        and _supports_flattened_device_query_lens()
    )


class DeepseekV32IndexerMetadataBuilder(AttentionMetadataBuilder):
    # The indexer opts out of the shared reorder-threshold vote (see __init__),
    # so this is None; its own split uses self.decode_threshold.
    reorder_batch_threshold: int | None = None
    requires_block_table_width = True
    # Backend capability for the GLM kpool BF16 DCP candidate: DEFAULT
    # FALSE in core. Only a matching HCU plugin indexer patch may set
    # this True on the class (it attests the wired implementation —
    # the merge primitives, kernels, and collectives this worktree
    # validated). This is NOT a user parameter; absence of the plugin
    # keeps DCP+compression rejected exactly as before.
    supports_glm_kpool_bf16_dcp = SUPPORTS_GLM_KPOOL_BF16_DCP
    # Same restricted opt-in pattern for the FP8 MLA storage variant.
    supports_glm_kpool_fp8_mla_dcp = SUPPORTS_GLM_KPOOL_FP8_MLA_DCP

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec: KVCacheSpec,
    ) -> AttentionCGSupport:
        if _supports_varlen_paged_mqa_logits() or _use_flattening(vllm_config):
            return AttentionCGSupport.ALWAYS
        return AttentionCGSupport.UNIFORM_BATCH

    def __init__(self, *args, block_table_width: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        scheduler_config = self.vllm_config.scheduler_config
        parallel_config = self.vllm_config.parallel_config
        self.dcp_world_size = parallel_config.decode_context_parallel_size
        self.dcp_rank = get_dcp_group().rank_in_group if self.dcp_world_size > 1 else 0
        self.pcp_world_size = parallel_config.prefill_context_parallel_size
        self.use_pcp = self.pcp_world_size > 1
        self.cp_kv_cache_interleave_size = parallel_config.cp_kv_cache_interleave_size
        # KV compression. Default to 1 for no compression. Determined
        # UP FRONT (before any device buffer allocation) so the DCP
        # capability predicate below can reject unsupported
        # combinations without first allocating large GPU buffers.
        self.compress_ratio = 1
        # Get compress_ratio for DeepseekV4 support
        if isinstance(self.kv_cache_spec, MLAAttentionSpec):
            # MLA compression is a whole number of tokens per state (fractions
            # are whisper block pooling and never reach MLA).
            assert isinstance(self.kv_cache_spec.tokens_per_state, int)
            self.compress_ratio = self.kv_cache_spec.tokens_per_state
        # The DCP+compression/interleave capability predicate (replaces
        # the former blanket rejections): DCP>1 with R>1 or I>1 passes
        # ONLY through the validated GLM kpool BF16 candidate; every
        # other combination fails closed with a named reason. DCP1 and
        # the R1/I1 DCP path keep the legacy behavior untouched.
        validate_dcp_compressed_indexer_support(
            self.vllm_config,
            self.compress_ratio,
            self.cp_kv_cache_interleave_size,
            capability_enabled=type(self).supports_glm_kpool_bf16_dcp,
            actual_spec=(
                self.kv_cache_spec
                if isinstance(self.kv_cache_spec, MLAAttentionSpec)
                else None
            ),
            fp8_mla_enabled=type(self).supports_glm_kpool_fp8_mla_dcp,
        )
        # NOTE(Chen):an estimated max size of flattened_kv. Need to double check.
        self.max_prefill_buffer_size = get_max_prefill_buffer_size(self.vllm_config)
        self.num_speculative_tokens = (
            self.vllm_config.speculative_config.num_speculative_tokens
            if self.vllm_config.speculative_config
            else 0
        )
        self.use_fp4_indexer_cache = dsa_indexer_uses_fp4(self.vllm_config)

        next_n = self.num_speculative_tokens + 1
        self.decode_threshold = next_n
        self.reorder_batch_threshold = None
        self.use_flattening = _use_flattening(self.vllm_config)
        self.supports_varlen = _supports_varlen_paged_mqa_logits()
        logger.info_once(
            "DSA indexer decode path: use_flattening=%s supports_varlen=%s "
            "(next_n=%d, use_fp4_cache=%s)",
            self.use_flattening,
            self.supports_varlen,
            next_n,
            self.use_fp4_indexer_cache,
        )

        sm_count = num_compute_units(self.device.index)
        self.num_sms = sm_count

        self.offsets_buffer = torch.arange(
            next_n, device=self.device, dtype=torch.int32
        )
        self.decode_lens_buffer = torch.zeros(
            (scheduler_config.max_num_batched_tokens,),
            dtype=torch.int32,
            device=self.device,
        )
        self.per_req_decode_lens_buffer = torch.zeros(
            (scheduler_config.max_num_batched_tokens,),
            dtype=torch.int32,
            device=self.device,
        )
        # Shared workspace for decode seq_lens. Native MTP views this as
        # (B, max_decode_len) at runtime, keeping context_lens contiguous even
        # when max_decode_len is smaller than next_n.
        self.decode_seq_lens_buffer = torch.zeros(
            (scheduler_config.max_num_batched_tokens,),
            dtype=torch.int32,
            device=self.device,
        )
        self.global_decode_seq_lens_buffer = torch.zeros(
            (scheduler_config.max_num_batched_tokens,),
            dtype=torch.int32,
            device=self.device,
        )
        self.decode_indices_buffer = torch.zeros(
            (scheduler_config.max_num_batched_tokens,),
            dtype=torch.int32,
            device=self.device,
        )
        self.arange_buffer = torch.arange(
            max(
                scheduler_config.max_num_seqs * next_n,
                scheduler_config.max_num_batched_tokens,
            ),
            dtype=torch.int32,
            device=self.device,
        )
        self.expanded_block_table_buffer = torch.zeros(
            (scheduler_config.max_num_batched_tokens, block_table_width),
            dtype=torch.int32,
            device=self.device,
        )

        # See: DeepGMM/csrc/apis/attention.hpp. Sized for one slot per SM;
        # build() narrows it to whatever the kernel actually schedules.
        self.scheduler_metadata_buffer = torch.empty(
            (self.num_sms + 1, 2), dtype=torch.int32, device=self.device
        )

        # Pre-allocate buffers for CUDA graph compatibility when
        if self.compress_ratio > 1:
            # compress_ratio > 1 (DeepseekV4)
            # Compressed slot mapping output buffer
            self.compressed_slot_mapping_buffer = torch.zeros(
                (scheduler_config.max_num_batched_tokens,),
                dtype=torch.int64,
                device=self.device,
            )
            # Buffer for compressed seq_lens in decode path
            self.expanded_seq_lens_buffer = torch.zeros(
                (scheduler_config.max_num_batched_tokens,),
                dtype=torch.int32,
                device=self.device,
            )
        self.indexer_decode_block_table_buffer: torch.Tensor | None = None
        self._max_num_batched_tokens = scheduler_config.max_num_batched_tokens

    def _dcp_localize_decode_seq_lens(
        self,
        seq_lens: torch.Tensor,
        num_decodes: int,
        seq_lens_is_buffer_view: bool,
    ) -> torch.Tensor:
        local_seq_lens = get_dcp_local_seq_lens(
            seq_lens,
            self.dcp_world_size,
            self.dcp_rank,
            self.cp_kv_cache_interleave_size,
        )
        if seq_lens_is_buffer_view:
            seq_lens.copy_(local_seq_lens)
            return seq_lens

        out = self.decode_seq_lens_buffer[:num_decodes]
        out.copy_(local_seq_lens)
        return out

    def _prepare_decode_tensors(
        self,
        seq_lens: torch.Tensor,
        block_table: torch.Tensor,
        decode_lens: torch.Tensor,
        decode_lens_cpu: torch.Tensor,
        query_start_loc: torch.Tensor,
        num_decodes: int,
        num_decode_tokens: int,
        use_native: bool,
        next_n: int,
        max_decode_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, bool]:
        """Prepare native or per-token flattened decode tensors."""
        spec_config = self.vllm_config.speculative_config
        adaptive = bool(spec_config and spec_config.enable_adaptive_verification)
        min_decode_len = int(decode_lens_cpu.min().item())
        if not use_native:
            assert self.decode_seq_lens_buffer.dim() == 1
            if (
                not self.supports_varlen
                and (num_decodes == 1 or not adaptive)
                and min_decode_len == max_decode_len
                and num_decodes * max_decode_len == num_decode_tokens
            ):
                # Uniform decode lengths with no cudagraph token padding.
                _prepare_uniform_decode_kernel[(num_decode_tokens,)](
                    seq_lens,
                    self.decode_seq_lens_buffer,
                    block_table,
                    block_table.stride(0),
                    self.expanded_block_table_buffer,
                    self.expanded_block_table_buffer.stride(0),
                    self.decode_lens_buffer,
                    max_decode_len,
                    BLOCK_SIZE=1024,
                )
                self.decode_seq_lens_buffer[num_decode_tokens:] = 0
                seq_lens = self.decode_seq_lens_buffer[:num_decode_tokens]
                block_table = self.expanded_block_table_buffer[:num_decode_tokens]
                decode_lens = self.decode_lens_buffer[:num_decode_tokens]
                return seq_lens, block_table, decode_lens, num_decode_tokens, False
            else:
                # Variable decode lengths.
                # Assume 4 requests with seq_lens [10, 7, 12, 0] (the final req is
                # padding) and decode_lens [3, 1, 4, 0] in the below example comments.
                # The context lengths are therefore
                # [10-3, 7-1, 12-4, 0-0] = [7, 6, 8, 0].

                # 3 + 1 + 4 + 0 = 8
                actual_expanded = int(decode_lens_cpu.sum().item())

                # Fuse expanded_base and expanded_starts into a single
                # repeat_interleave:
                # seq_len_i = (context_start[b] - query_start_loc[b]) + arange[i] + 1
                # where context_start[b] = seq_lens[b] - decode_lens[b].
                # Example: offsets = [7-0, 6-3, 8-4, 0-8] = [7, 3, 4, -8]
                # expanded_offsets  = [7, 7, 7, 3, 4, 4, 4, 4]
                # result            = [8, 9, 10, 7, 9, 10, 11, 12]
                expanded_offsets = torch.repeat_interleave(
                    seq_lens - decode_lens - query_start_loc,
                    decode_lens,
                    output_size=actual_expanded,
                )

                # [8, 9, 10, 7, 9, 10, 11, 12, ...] where ... is unused buffer space
                self.decode_seq_lens_buffer[:actual_expanded] = (
                    expanded_offsets + self.arange_buffer[:actual_expanded] + 1
                )
                self.decode_seq_lens_buffer[actual_expanded:] = 0
                seq_lens = self.decode_seq_lens_buffer[:num_decode_tokens]

                # Give each of the flattened entries the same block table row as the
                # original request.
                self.expanded_block_table_buffer[:actual_expanded] = (
                    torch.repeat_interleave(
                        block_table, decode_lens, dim=0, output_size=actual_expanded
                    )
                )
                if actual_expanded < num_decode_tokens:
                    self.expanded_block_table_buffer[
                        actual_expanded:num_decode_tokens, 0
                    ] = 0
                block_table = self.expanded_block_table_buffer[:num_decode_tokens]

                # All reqs now have decode_len=1
                self.decode_lens_buffer[:num_decode_tokens] = 1
                decode_lens = self.decode_lens_buffer[:num_decode_tokens]
                return seq_lens, block_table, decode_lens, num_decode_tokens, False
        else:
            # Native path: plain decode (next_n==1) or spec decode
            # with 2D per-token context lengths (next_n > 1).
            #
            # When decode_lens are not truly uniform (e.g. some requests have
            # decode_len < next_n due to padding or short prefills), the simple
            # reshape in sparse_attn_indexer won't work. Use pack_seq_triton
            # (requires_padding) instead.
            requires_padding = min_decode_len != max_decode_len
            if use_native and next_n > 1:
                assert self.decode_seq_lens_buffer.dim() == 1
                # (B, max_decode_len): token j attends to
                # L - max_decode_len + j + 1 KV tokens.
                seq_lens_buffer = self.decode_seq_lens_buffer[
                    : num_decodes * max_decode_len
                ].view(num_decodes, max_decode_len)
                if self.dcp_world_size > 1:
                    # DCP: build the per-token GLOBAL bounds from each
                    # request's ACTUAL decode_len — token j's bound is
                    # L_b - decode_len_b + 1 + j for j < decode_len_b
                    # and 0 (masked) beyond — NOT from the batch-wide
                    # max_decode_len. With the legacy construction a
                    # short request's REAL query row gets the wrong
                    # bound (e.g. L=[10,7], decode_lens=[2,1],
                    # next_n=2: legacy [[9,10],[6,7]] — row 6 for the
                    # actual token whose true bound is 7) and its pad
                    # row gets a spurious positive bound; the DCP
                    # merge would then read both rows as valid against
                    # localized bounds. The mask zeroes pad rows
                    # BEFORE localization and //R; W1 keeps the legacy
                    # construction byte-for-byte.
                    per_req = (
                        seq_lens.unsqueeze(1)
                        - decode_lens.unsqueeze(1)
                        + 1
                        + self.offsets_buffer[:max_decode_len]
                    )
                    valid_j = self.offsets_buffer[:max_decode_len] < (
                        decode_lens.unsqueeze(1)
                    )
                    seq_lens_buffer[:] = torch.where(
                        valid_j, per_req.clamp_min(0), torch.zeros_like(per_req)
                    )
                else:
                    # Clamp at 0: padding requests have seq_len == 0, which
                    # would otherwise make token 0 negative (next_n=2 gives
                    # 0-2+1+0 = -1). Downstream kernels read these as uint32,
                    # turning -1 into ~4e9.
                    seq_lens_buffer[:] = (
                        seq_lens.unsqueeze(1)
                        - max_decode_len
                        + 1
                        + self.offsets_buffer[:max_decode_len]
                    ).clamp_(min=0)
                seq_lens = seq_lens_buffer
            return seq_lens, block_table, decode_lens, num_decodes, requires_padding

    def _prepare_global_decode_seq_lens(
        self,
        global_seq_lens: torch.Tensor | None,
        decode_lens: torch.Tensor,
        decode_lens_cpu: torch.Tensor,
        query_start_loc: torch.Tensor,
        num_decode_tokens: int,
        use_native: bool,
        max_decode_len: int,
    ) -> torch.Tensor | None:
        if global_seq_lens is None:
            return None
        if use_native or max_decode_len <= 1:
            return global_seq_lens

        actual_expanded = int(decode_lens_cpu.sum().item())
        if actual_expanded > 0:
            expanded_offsets = torch.repeat_interleave(
                global_seq_lens - decode_lens - query_start_loc,
                decode_lens,
                output_size=actual_expanded,
            )
            self.global_decode_seq_lens_buffer[:actual_expanded] = (
                expanded_offsets + self.arange_buffer[:actual_expanded] + 1
            )
        self.global_decode_seq_lens_buffer[actual_expanded:num_decode_tokens] = 0
        return self.global_decode_seq_lens_buffer[:num_decode_tokens]

    def _build_varlen_decode_indices(
        self,
        decode_lens: torch.Tensor,
        decode_lens_cpu: torch.Tensor,
        num_decode_tokens: int,
    ) -> torch.Tensor:
        """Build request ids for flattened SM100 varlen rows."""
        indices = self.decode_indices_buffer[:num_decode_tokens]
        actual_expanded = int(decode_lens_cpu.sum().item())
        num_decodes = decode_lens.shape[0]
        indices[:actual_expanded] = torch.repeat_interleave(
            self.arange_buffer[:num_decodes],
            decode_lens,
            output_size=actual_expanded,
        )
        if actual_expanded < num_decode_tokens:
            pad = num_decode_tokens - actual_expanded
            indices[actual_expanded:num_decode_tokens] = (
                num_decodes + self.arange_buffer[:pad]
            )
        return indices

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> DeepseekV32IndexerMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_tokens = common_attn_metadata.num_actual_tokens
        query_start_loc = common_attn_metadata.query_start_loc
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        seq_lens = common_attn_metadata.seq_lens
        slot_mapping = common_attn_metadata.slot_mapping
        block_table = common_attn_metadata.block_table_tensor
        dcp_local_seq_lens = common_attn_metadata.dcp_local_seq_lens
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.decode_threshold,
                require_uniform=not (self.use_flattening or self.supports_varlen),
                treat_short_extends_as_decodes=not self.use_pcp,
            )
        )

        assert num_decodes + num_prefills == num_reqs
        assert num_decode_tokens + num_prefill_tokens == num_tokens

        compressed_slot_mapping = slot_mapping
        compressed_seq_lens = seq_lens
        indexer_block_table = block_table
        if self.compress_ratio > 1:
            kernel_block_size = self.kernel_block_size
            if (
                kernel_block_size is not None
                and self.kv_cache_spec.block_size != kernel_block_size
                and self.kv_cache_spec.block_size % kernel_block_size == 0
            ):
                factor = self.kv_cache_spec.block_size // kernel_block_size
                indexer_block_table = (block_table[:, ::factor] // factor).contiguous()
            padded_num_tokens = num_tokens
            if self.pcp_world_size > 1:
                padded_num_tokens = slot_mapping.shape[0] // self.pcp_world_size
            compressed_slot_mapping = get_compressed_slot_mapping(
                num_tokens,
                query_start_loc,
                seq_lens,
                indexer_block_table,
                self.kv_cache_spec.num_states,
                self.compress_ratio,
                out=self.compressed_slot_mapping_buffer,
                dcp_world_size=self.dcp_world_size,
                dcp_rank=self.dcp_rank,
                pool_interleave=derive_pool_interleave(
                    self.cp_kv_cache_interleave_size,
                    self.compress_ratio,
                    self.dcp_world_size,
                ),
            )
            if self.pcp_world_size > 1:
                compressed_slot_mapping = get_pcp_group().all_gather(
                    self.compressed_slot_mapping_buffer[:padded_num_tokens],
                    dim=0,
                )
            compressed_seq_lens = seq_lens // self.compress_ratio

        prefill_metadata = None
        if num_prefills > 0:
            # This CPU value is an upper bound for async-spec extend rows.  It
            # is safe for chunking/allocation because CUDA metadata below is
            # built from exact device seq_lens and gather ignores the tail.
            assert common_attn_metadata.seq_lens_cpu_upper_bound is not None
            seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
            compressed_seq_lens_cpu = (
                seq_lens_cpu // self.compress_ratio
                if self.compress_ratio > 1
                else seq_lens_cpu
            )
            prefill_query_lens_cpu = torch.diff(
                query_start_loc_cpu[num_decodes : num_decodes + num_prefills + 1]
            )
            max_logits_bytes = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
            # Upper bound is exact for prefill rows (the `[num_decodes:]`
            # slice below).
            assert common_attn_metadata.seq_lens_cpu_upper_bound is not None
            seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
            chunk_specs = split_indexer_prefill_chunks(
                compressed_seq_lens_cpu[num_decodes:],
                prefill_query_lens_cpu,
                self.max_prefill_buffer_size,
                max_logits_bytes,
                request_offset=num_decodes,
            )

            chunks = []
            for req_slice, query_slice in chunk_specs:
                metadata = build_prefill_chunk_metadata(
                    req_slice.start,
                    req_slice.stop,
                    query_start_loc,
                    query_start_loc_cpu,
                    seq_lens,
                    compressed_seq_lens,
                    compressed_seq_lens_cpu,
                    indexer_block_table,
                    self.compress_ratio,
                    query_slice=query_slice,
                    skip_kv_gather=query_slice.start > 0,
                    dcp_rank=self.dcp_rank,
                    dcp_world_size=self.dcp_world_size,
                    cp_kv_cache_interleave_size=self.cp_kv_cache_interleave_size,
                )
                # Skip when total_seq_lens is 0 (i.e., no compressed token).
                if metadata is not None:
                    chunks.append(metadata)
            prefill_metadata = DeepseekV32IndexerPrefillMetadata(
                chunks,
                max_prefill_seq_len=(
                    int(seq_lens_cpu[num_decodes:].max().item())
                    if num_prefills > 0
                    else 0
                ),
            )

        decode_metadata = None
        if num_decodes > 0:
            torch.diff(
                common_attn_metadata.query_start_loc[: num_decodes + 1],
                out=self.decode_lens_buffer[:num_decodes],
            )
            decode_lens = self.decode_lens_buffer[:num_decodes]
            self.per_req_decode_lens_buffer[:num_decodes].copy_(decode_lens)
            decode_lens_cpu = torch.diff(
                common_attn_metadata.query_start_loc_cpu[: num_decodes + 1]
            )

            # Under DCP the per-token decode bounds must be localized AFTER the
            # per-token expansion below, not before. Expanding from a
            # request-level localized length subtracts decode offsets in local
            # space and yields too-short bounds (e.g. world=2, rank=1, global
            # per-token bounds [8, 9, 10] -> [3, 4, 5] instead of [4, 4, 5]), so
            # the first decode token would run top-k against too short a local KV
            # range and miss valid tokens. Keep the global seq_lens here and
            # localize the expanded bounds further down.
            global_seq_lens_for_decode: torch.Tensor | None = None
            if dcp_local_seq_lens is not None:
                global_seq_lens_for_decode = common_attn_metadata.seq_lens[:num_decodes]
            seq_lens = common_attn_metadata.seq_lens[:num_decodes]
            block_table = common_attn_metadata.block_table_tensor[:num_decodes, ...]

            max_decode_len = int(decode_lens_cpu.max().item())
            min_decode_len = int(decode_lens_cpu.min().item())
            write_is_uniform = min_decode_len == max_decode_len
            next_n = 1 + self.num_speculative_tokens
            # The kernel sees max_decode_len Q rows, not the configured next_n,
            # so legality is per-step: on SM90 a uniformly 3-deep batch has no
            # native kernel. max_decode_len <= 1 always has one.
            step_next_n_ok = max_decode_len <= 1 or _supports_native_decode(
                max_decode_len
            )
            use_native = (
                not (self.use_flattening or self.supports_varlen)
                and max_decode_len <= next_n
                and step_next_n_ok
            )

            global_seq_lens_for_decode = self._prepare_global_decode_seq_lens(
                global_seq_lens=global_seq_lens_for_decode,
                decode_lens=decode_lens,
                decode_lens_cpu=decode_lens_cpu,
                query_start_loc=common_attn_metadata.query_start_loc[:num_decodes],
                num_decode_tokens=num_decode_tokens,
                use_native=use_native,
                max_decode_len=max_decode_len,
            )

            decode_indices = None
            if self.supports_varlen:
                decode_indices = self._build_varlen_decode_indices(
                    decode_lens=decode_lens,
                    decode_lens_cpu=decode_lens_cpu,
                    num_decode_tokens=num_decode_tokens,
                )

            seq_lens, block_table, decode_lens, batch_size, requires_padding = (
                self._prepare_decode_tensors(
                    seq_lens=seq_lens,
                    block_table=block_table,
                    decode_lens=decode_lens,
                    decode_lens_cpu=decode_lens_cpu,
                    query_start_loc=common_attn_metadata.query_start_loc[:num_decodes],
                    num_decodes=num_decodes,
                    num_decode_tokens=num_decode_tokens,
                    use_native=use_native,
                    next_n=next_n,
                    max_decode_len=max_decode_len,
                )
            )

            if self.compress_ratio > 1:
                kernel_block_size = self.kernel_block_size
                if (
                    kernel_block_size is not None
                    and self.kv_cache_spec.block_size != kernel_block_size
                    and self.kv_cache_spec.block_size % kernel_block_size == 0
                ):
                    factor = self.kv_cache_spec.block_size // kernel_block_size
                    compressed = block_table[:, ::factor] // factor
                    rows, cols = compressed.shape
                    if self.indexer_decode_block_table_buffer is None:
                        self.indexer_decode_block_table_buffer = torch.zeros(
                            (self._max_num_batched_tokens, cols),
                            dtype=torch.int32,
                            device=self.device,
                        )
                    self.indexer_decode_block_table_buffer[:rows, :cols].copy_(
                        compressed
                    )
                    block_table = self.indexer_decode_block_table_buffer[:rows, :cols]

            seq_lens_is_buffer_view = (use_native and next_n > 1) or (
                not use_native and max_decode_len > 1
            )

            # DCP: localize the now-expanded per-token global bounds to this
            # rank's owned KV. Done here (after expansion) so each token's global
            # causal length is localized individually; see the comment above.
            if dcp_local_seq_lens is not None:
                seq_lens = self._dcp_localize_decode_seq_lens(
                    seq_lens, num_decodes, seq_lens_is_buffer_view
                )

            # For DeepseekV4 (compress_ratio > 1), the indexer KV cache stores
            # compressed tokens. Convert uncompressed seq_lens to compressed.
            if self.compress_ratio > 1:
                if seq_lens_is_buffer_view:
                    seq_lens //= self.compress_ratio
                else:
                    # Copy to avoid mutating shared state; keeps CG address stable.
                    self.expanded_seq_lens_buffer[:num_decodes] = (
                        seq_lens // self.compress_ratio
                    )
                    self.expanded_seq_lens_buffer[num_decodes:num_decode_tokens] = 0
                    seq_lens = self.expanded_seq_lens_buffer[:num_decode_tokens]

            # Non-MTP: deep_gemm paged MQA logits requires 2D context_lens
            # (csrc/apis/attention.hpp). Unsqueeze to (B, 1) so downstream
            # kernels see the same (B, next_n) layout as the MTP path.
            if seq_lens.dim() == 1:
                seq_lens = seq_lens.unsqueeze(-1)

            # DeepGEMM is required for the paged MQA logits on CUDA devices
            schedule_metadata = self.scheduler_metadata_buffer
            if current_platform.is_cuda() and has_deep_gemm():
                metadata = get_paged_mqa_logits_metadata(
                    seq_lens,
                    self.kv_cache_spec.num_states,
                    self.num_sms,
                    indices=decode_indices,
                )
                schedule_metadata = self.scheduler_metadata_buffer[: metadata.shape[0]]
                schedule_metadata[:] = metadata

            decode_metadata = DeepSeekV32IndexerDecodeMetadata(
                block_table=block_table,
                seq_lens=seq_lens,
                decode_lens=decode_lens,
                requires_padding=requires_padding,
                schedule_metadata=schedule_metadata,
                indices=decode_indices,
                global_seq_lens=global_seq_lens_for_decode,
                per_req_decode_lens=self.per_req_decode_lens_buffer[:num_decodes],
                decode_is_uniform=write_is_uniform,
                write_max_decode_len=max_decode_len,
                dcp_world_size=self.dcp_world_size,
                dcp_rank=self.dcp_rank,
                pool_interleave=derive_pool_interleave(
                    self.cp_kv_cache_interleave_size,
                    self.compress_ratio,
                    self.dcp_world_size,
                ),
            )

        attn_metadata = DeepseekV32IndexerMetadata(
            seq_lens=common_attn_metadata.seq_lens,
            max_seq_len=common_attn_metadata.max_seq_len,
            slot_mapping=compressed_slot_mapping,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            prefill=prefill_metadata,
            decode=decode_metadata,
        )

        return attn_metadata


def derive_pool_interleave(
    cp_kv_cache_interleave_size: int,
    compress_ratio: int,
    dcp_world_size: int,
) -> int:
    """Derive the POOL-space interleave J from the TOKEN-space I.

    The public ``cp_kv_cache_interleave_size`` stays in TOKEN units
    (I). Pool-granular DCP localization (pool-granular lengths, the
    prefill chunk kernel's DCP_INTERLEAVE, and the compressed slot
    mapping) must shard in POOL units: J = I // R. Requirements for
    DCP with compression (W > 1 and R > 1): I >= R and I % R == 0
    (pool-coherent sharding; every pool's R tokens sit on one rank).
    DCP1 (W == 1) keeps the legacy behavior for ANY I and R — the
    value is unused there and 1 is returned for I == 1 exactly as
    before. R == 1 keeps J == I (no compression, token == pool).
    """
    if type(cp_kv_cache_interleave_size) is not int or (
        cp_kv_cache_interleave_size < 1
    ):
        raise ValueError(
            "cp_kv_cache_interleave_size must be a positive exact "
            f"int, got {cp_kv_cache_interleave_size!r}"
        )
    if type(compress_ratio) is not int or compress_ratio < 1:
        raise ValueError(
            f"compress_ratio must be a positive exact int, got {compress_ratio!r}"
        )
    if type(dcp_world_size) is not int or dcp_world_size < 1:
        raise ValueError(
            f"dcp_world_size must be a positive exact int, got {dcp_world_size!r}"
        )
    if dcp_world_size == 1 or compress_ratio == 1:
        # Legacy/unused path: DCP1 never localizes; R==1 means token
        # IS pool. Preserve the historical value (1 for I==1).
        return cp_kv_cache_interleave_size
    if (
        cp_kv_cache_interleave_size < compress_ratio
        or cp_kv_cache_interleave_size % compress_ratio != 0
    ):
        raise ValueError(
            "DCP with compressed indexer KV requires "
            "cp_kv_cache_interleave_size >= compress_ratio and divisible "
            f"by it (pool-coherent sharding); got interleave="
            f"{cp_kv_cache_interleave_size}, compress_ratio={compress_ratio}"
        )
    return cp_kv_cache_interleave_size // compress_ratio


def build_prefill_chunk_metadata(
    start_idx: int,
    end_idx: int,
    query_start_loc: torch.Tensor,
    query_start_loc_cpu: torch.Tensor,
    uncompressed_seq_lens: torch.Tensor,
    compressed_seq_lens: torch.Tensor,
    compressed_seq_lens_cpu: torch.Tensor,
    block_table: torch.Tensor,
    compress_ratio: int,
    query_slice: slice | None = None,
    skip_kv_gather: bool = False,
    dcp_rank: int = 0,
    dcp_world_size: int = 1,
    cp_kv_cache_interleave_size: int = 1,
) -> DeepseekV32IndexerPrefillChunkMetadata | None:
    total_seq_lens = compressed_seq_lens_cpu[start_idx:end_idx].sum().item()
    if total_seq_lens == 0:
        return None

    num_reqs = end_idx - start_idx
    device = block_table.device
    token_to_seq = torch.empty(total_seq_lens, dtype=torch.int32, device=device)

    cu_seq_lens = torch.empty(num_reqs + 1, dtype=torch.int32, device=device)
    # Assigning to slice avoids cpu sync.
    cu_seq_lens[:1] = 0
    torch.cumsum(compressed_seq_lens[start_idx:end_idx], dim=0, out=cu_seq_lens[1:])

    local_cu_seq_lens = cu_seq_lens
    local_total_seq_lens = total_seq_lens
    max_local_total_seq_lens = total_seq_lens
    # Aliases the global map when DCP is off (same tensor, same
    # semantics: one entry per pool in request order).
    local_token_to_seq = token_to_seq
    # The POOL-space interleave for every DCP localization below. The
    # public parameter stays TOKEN units (I); pool-granular lengths
    # shard in pool units (J = I // R). DCP1 and R==1 keep the legacy
    # value; DCP+compression validates pool coherence here.
    pool_interleave = derive_pool_interleave(
        cp_kv_cache_interleave_size, compress_ratio, dcp_world_size
    )
    if dcp_world_size > 1:
        # Per-rank local KV length under interleave-aware DCP sharding, shape
        # [num_reqs, dcp_world_size]. Reuse the canonical CP helper so the
        # sharding matches the rest of the DCP pipeline (decode/prefill).
        local_seq_lens = get_dcp_local_seq_lens(
            compressed_seq_lens[start_idx:end_idx],
            dcp_world_size,
            None,
            pool_interleave,
        )
        this_rank_counts = local_seq_lens[:, dcp_rank].to(torch.int32)
        local_cu_seq_lens = torch.zeros(num_reqs + 1, dtype=torch.int32, device=device)
        torch.cumsum(this_rank_counts, dim=0, out=local_cu_seq_lens[1:])
        local_total_seq_lens = int(local_cu_seq_lens[-1].item())
        max_local_total_seq_lens = int(local_seq_lens.sum(dim=0).max().item())
        # Rank-local pool -> request index, mirroring the global
        # token_to_seq the kernel writes (one entry per LOCAL pool in
        # request order). repeat_interleave over the local counts is
        # the same construction the kernel performs per request; the
        # host total above bounds output_size, so no extra sync.
        local_token_to_seq = torch.repeat_interleave(
            torch.arange(num_reqs, dtype=torch.int32, device=device),
            this_rank_counts,
            output_size=local_total_seq_lens,
        )

    query_start_loc = (
        query_start_loc[start_idx : end_idx + 1] - query_start_loc[start_idx]
    )

    total_query_len = int(
        (query_start_loc_cpu[end_idx] - query_start_loc_cpu[start_idx]).item()
    )
    if query_slice is not None:
        qs_start = query_slice.start
        qs_stop = query_slice.stop
    else:
        qs_start = 0
        qs_stop = total_query_len
    output_query_len = qs_stop - qs_start

    cu_seq_len_ks = torch.empty(output_query_len, dtype=torch.int32, device=device)
    cu_seq_len_ke = torch.empty(output_query_len, dtype=torch.int32, device=device)

    # Under DCP the kernel writes this rank's local row bounds into
    # cu_seq_len_ks/ke; otherwise local_cu_seq_lens aliases cu_seq_lens.
    _BUILD_PREFILL_CHUNK_METADATA_KERNEL(
        query_start_loc,
        uncompressed_seq_lens[start_idx:end_idx],
        cu_seq_lens,
        local_cu_seq_lens,
        token_to_seq,
        cu_seq_len_ks,
        cu_seq_len_ke,
        qs_start,
        qs_stop,
        dcp_rank,
        dcp_world_size,
        pool_interleave,
        num_reqs=num_reqs,
        COMPRESS_RATIO=compress_ratio,
    )

    token_start = query_start_loc_cpu[start_idx].item()
    if query_slice is not None:
        token_end = token_start + qs_stop
        token_start = token_start + qs_start
        skip_kv_gather = skip_kv_gather or qs_start > 0
    else:
        token_end = query_start_loc_cpu[end_idx].item()

    return DeepseekV32IndexerPrefillChunkMetadata(
        cu_seqlen_ks=cu_seq_len_ks,
        cu_seqlen_ke=cu_seq_len_ke,
        cu_seq_lens=cu_seq_lens,
        token_to_seq=token_to_seq,
        total_seq_lens=total_seq_lens,
        block_table=block_table[start_idx:end_idx],
        token_start=token_start,
        token_end=token_end,
        num_reqs=num_reqs,
        skip_kv_gather=skip_kv_gather,
        local_cu_seq_lens=local_cu_seq_lens,
        local_total_seq_lens=local_total_seq_lens,
        max_local_total_seq_lens=max_local_total_seq_lens,
        local_token_to_seq=local_token_to_seq,
        dcp_world_size=dcp_world_size,
        dcp_rank=dcp_rank,
        pool_interleave=pool_interleave,
    )
