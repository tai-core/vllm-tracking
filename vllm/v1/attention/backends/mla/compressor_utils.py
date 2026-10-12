# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv

_DSPARK_SWA_INDEX_ALIGNMENT = 64


def get_dspark_swa_index_width(
    window_size: int,
    num_speculative_tokens: int,
) -> int:
    """Return the padded width of non-causal DSpark SWA indices."""
    width = max(int(window_size), 0) + max(int(num_speculative_tokens), 0)
    return cdiv(width, _DSPARK_SWA_INDEX_ALIGNMENT) * _DSPARK_SWA_INDEX_ALIGNMENT


@triton.jit
def _compressed_slot_mapping_kernel(
    # [num_tokens]
    slot_mapping_ptr,
    # [num_reqs + 1]
    query_start_loc_ptr,
    # [num_reqs]
    seq_lens_ptr,
    # [num_reqs, max_num_blocks]
    block_table_ptr,
    block_table_stride,
    block_size,
    COMPRESS_RATIO: tl.constexpr,
    DCP_WORLD_SIZE: tl.constexpr,
    DCP_RANK: tl.constexpr,
    POOL_INTERLEAVE: tl.constexpr,
    PAD_ID: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)

    query_start = tl.load(query_start_loc_ptr + batch_idx)
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
    query_len = query_end - query_start

    seq_len = tl.load(seq_lens_ptr + batch_idx)
    start_pos = seq_len - query_len

    for i in range(0, query_len, TRITON_BLOCK_SIZE):
        offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
        mask = offset < query_len

        pos = start_pos + i + tl.arange(0, TRITON_BLOCK_SIZE)
        is_valid = (pos >= 0) & ((pos + 1) % COMPRESS_RATIO == 0)
        pos_after_compress = pos // COMPRESS_RATIO

        if DCP_WORLD_SIZE == 1:
            # Non-DCP path: the global compressed position IS local.
            local_pool_id = pos_after_compress
        else:
            # DCP path: pools are interleaved across ranks in groups of
            # POOL_INTERLEAVE (pool-space units; block_size here is the
            # per-state pool granularity on the spec, so both are pool units
            # -- never mix the token-space interleave into this API).
            # The owner check runs BEFORE any block_table load so a remote
            # pool's global id can never index another rank's (shorter)
            # block table.
            owning_rank = (pos_after_compress // POOL_INTERLEAVE) % DCP_WORLD_SIZE
            is_valid = is_valid & (owning_rank == DCP_RANK)
            local_pool_id = (
                pos_after_compress // (DCP_WORLD_SIZE * POOL_INTERLEAVE)
            ) * POOL_INTERLEAVE + pos_after_compress % POOL_INTERLEAVE

        block_ids = local_pool_id // block_size
        # Masked-off lanes (remote pools, incomplete pools, padded rows)
        # may carry negative block ids; clamp only the LOWER bound so the
        # tl.load address stays non-negative. The masked load then never
        # reads memory for those lanes -- clamping is not needed for
        # safety, only to keep the address arithmetic well-defined. The
        # UPPER bound is deliberately NOT clamped: an out-of-range block
        # id on an OWNED lane means the caller's block table is too
        # short for its own sequence, which must surface as an error
        # rather than be silently redirected to a wrong block.
        safe_block_ids = tl.maximum(block_ids, 0)
        block_numbers = tl.load(
            block_table_ptr + batch_idx * block_table_stride + safe_block_ids,
            mask=mask & is_valid,
            other=0,
        )
        slot_ids = block_numbers * block_size + local_pool_id % block_size

        # NOTE
        slot_ids = tl.where(is_valid, slot_ids, PAD_ID)
        tl.store(slot_mapping_ptr + query_start + offset, slot_ids, mask=mask)


def get_compressed_slot_mapping(
    num_tokens: int,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    compress_ratio: int,
    out: torch.Tensor | None = None,
    *,
    dcp_world_size: int = 1,
    dcp_rank: int = 0,
    pool_interleave: int = 1,
) -> torch.Tensor:
    """Compute the pool-granular slot mapping for compressed indexer writes.

    Args:
        num_tokens: Number of query tokens in the batch.
        query_start_loc: [num_reqs + 1] token offsets of each request.
        seq_lens: [num_reqs] sequence lengths.
        block_table: [num_reqs, max_num_blocks] this rank's LOCAL pool block
            table (already DCP-divided when dcp_world_size > 1).
        block_size: Pool-state units per block (the spec's num_states; both
            this and ``pool_interleave`` are POOL units -- do not pass the
            token-space interleave here).
        compress_ratio: Tokens per stored state (R).
        out: Optional preallocated output; filled with -1 first.
        dcp_world_size: DCP world size (default 1 -- the legacy behavior,
            byte-for-byte).
        dcp_rank: This rank's DCP rank (default 0).
        pool_interleave: Pool-space interleave J (default 1 -- plain
            round-robin). Only meaningful when dcp_world_size > 1.

    Returns:
        [num_tokens] int64 slot ids: for each query token whose global
        position completes a pool owned by this rank, the physical slot
        ``block_number * block_size + local_pool_offset`` resolved through
        THIS rank's block table; every other token (incomplete pool, remote
        owner, padded row) gets -1.
    """
    if type(dcp_world_size) is not int or dcp_world_size < 1:
        raise ValueError(f"dcp_world_size must be an int >= 1, got {dcp_world_size!r}")
    if type(dcp_rank) is not int or not 0 <= dcp_rank < dcp_world_size:
        raise ValueError(
            f"dcp_rank must be an int in [0, {dcp_world_size}), got {dcp_rank!r}"
        )
    if type(pool_interleave) is not int or pool_interleave < 1:
        raise ValueError(
            f"pool_interleave must be an int >= 1, got {pool_interleave!r}"
        )
    if out is not None:
        # Guard: for padded / invalid sequences.
        # Negative positions produce bogus block indices that lead to illegal memory
        # accesses inside the block_table load.
        # NOTE: Fill -1 to the whole tensor, not just the first `num_tokens`.
        out.fill_(-1)
        slot_mapping = out[:num_tokens]
    else:
        slot_mapping = torch.full(
            (num_tokens,), -1, dtype=torch.int64, device=query_start_loc.device
        )

    num_reqs = block_table.shape[0]
    _compressed_slot_mapping_kernel[(num_reqs,)](
        slot_mapping,
        query_start_loc,
        seq_lens,
        block_table,
        block_table.stride(0),
        block_size,
        compress_ratio,
        dcp_world_size,
        dcp_rank,
        pool_interleave,
        PAD_ID=-1,
        TRITON_BLOCK_SIZE=1024,
    )
    return slot_mapping
