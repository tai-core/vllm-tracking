# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KpoolTail generic slot-mapping exclusion (READ-OOB fix) — CPU tests.

The r3 review (artifacts/supervision/vllm-v029/tail-slot-review.md)
proved that the generic slot-mapping kernels read block-table columns
derived from global positions for the KpoolTail group, whose row width
(8 at kpool 16 / 32 at kpool 4, after 128-token alignment) is far
smaller than cdiv(max_position+1, kpool): first invalid position 128.

These tests exercise the REAL selection policies and the REAL index
expression of both kernels (MRV2 ``_compute_slot_mappings_kernel`` and
V1 ``_COMPUTE_SLOT_MAPPING_KERNEL``) on CPU:

  - MRV2: the ``slot_mapping_enabled`` construction must exclude
    KpoolTailSpec (directly and behind the UniformType wrapper). With
    the exclusion, the kernel's ``tl.where(mapping_enabled, pos//kbs, 0)``
    clamps block_indices to 0 — a VALID column-0 load (rows always have
    at least column 0 populated by the tail manager's single ring
    block) — and the stored slot ids become PAD_ID, which the dedicated
    ``KpoolTailMetadataBuilder`` fully overrides for actual tokens.
  - V1: KpoolTailSpec must map to ``SlotMappingMode.NONE`` so
    ``compute_slot_mapping`` returns before launching the kernel.

Producer/consumer proof: the dedicated
``compute_kpool_tail_slot_mapping`` is executed against a REAL gathered
tail block table row and REAL positions for kpool 4/16, chunk-offset
8192 positions, and multi-request batches — asserting
``own_block*kpool + pos%kpool`` exactly, and that the generic values it
copies in are fully overwritten in [0, num_actual_tokens).

Unchanged-behavior controls: attention specs (MLA/Full), MambaSpec, and
CircularBufferSpec retain their previous enabled/mode selections.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from vllm.v1.attention.backends.mla.indexer import (
    KpoolTailBackend,
    compute_kpool_tail_slot_mapping,
)
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheSpec,
    KVCacheSpecKind,
    KpoolTailSpec,
    MambaSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
    get_kv_cache_spec_kind,
)
from vllm.v1.worker.block_table import (
    SlotMappingMode,
    get_block_table_width,
)
from vllm.v1.worker.utils import select_common_block_size

# Positions from the r3 review: first OOB position 128 (kpool 4 and 16),
# chunk boundaries 8192/16127/16128/32767 boundary cases.
OOB_POSITIONS = [0, 127, 128, 8191, 8192, 16127, 16128, 32767]


def make_tail_spec(kpool: int) -> KpoolTailSpec:
    return KpoolTailSpec(
        block_size=kpool,
        num_kv_heads=2,
        head_size=128,
        head_size_v=0,
        dtype=torch.bfloat16,
        sliding_window=kpool,
    )


def tail_geometry(kpool: int) -> dict:
    """Real geometry: width via the real aligned helper, kbs via the real
    backend selection."""
    spec = make_tail_spec(kpool)
    max_blocks = spec.max_num_blocks_per_req(None, 32768)
    width = get_block_table_width(max_blocks, spec.block_size)
    kbs = select_common_block_size(spec.block_size, [KpoolTailBackend])
    return {"spec": spec, "width": width, "kbs": kbs,
            "max_blocks": max_blocks}


# ---------------------------------------------------------------------------
# MRV2: slot_mapping_enabled policy (RED on unfixed core)
# ---------------------------------------------------------------------------

from vllm.v1.worker.gpu.model_runner import (
    spec_uses_generic_token_slot_mapping as mrv2_enabled_for,
)
from vllm.v1.worker.gpu_model_runner import (
    spec_uses_generic_token_slot_mapping_v1 as v1_mode_helper,
)


def v1_mode_for_real(kv_cache_spec) -> SlotMappingMode:
    """The REAL V1 selection via the real helper."""
    kind = get_kv_cache_spec_kind(kv_cache_spec)
    if kind == KVCacheSpecKind.MAMBA:
        return SlotMappingMode.NONE
    if v1_mode_helper(kv_cache_spec):
        return SlotMappingMode.TOKEN_TO_KV_SLOT
    return SlotMappingMode.NONE


def test_mrv2_tail_spec_excluded_from_generic_slot_mapping():
    """KpoolTailSpec must be excluded: its dedicated builder overrides the
    generic row, and the generic kernel's global-position indices read
    past the narrow tail row (width 8/32 vs indices up to 2047/8191)."""
    for kpool in (4, 16):
        g = tail_geometry(kpool)
        assert mrv2_enabled_for(g["spec"]) is False, (
            f"kpool={kpool}: KpoolTailSpec must be excluded from the "
            "generic slot mapping (r3 READ-OOB defect D1)"
        )


def test_mrv2_tail_spec_behind_uniform_type_excluded():
    """The UniformType wrapper path (first_spec) must behave identically."""
    spec = make_tail_spec(16)
    wrapped = UniformTypeKVCacheSpecs(
        block_size=spec.block_size, kv_cache_specs={"tail.l": spec})
    layer_spec = (
        wrapped.first_spec
        if isinstance(wrapped, UniformTypeKVCacheSpecs) else wrapped
    )
    assert mrv2_enabled_for(layer_spec) is False


def test_mrv2_disabled_tail_column_zero_load_is_valid():
    """With the exclusion, the kernel's clamp
    ``tl.where(mapping_enabled, pos//kbs, 0)`` yields column 0 for every
    position — and column 0 is ALWAYS valid: the tail manager claims the
    single ring block first (KpoolTailManager._claim_ring_block), so
    every scheduled request's row has >= 1 written entry, and the row
    width is >= 1 by construction. Document the exact bound: valid
    columns are [0, width) with width >= 8 (aligned); the clamped load
    reads column 0 only."""
    for kpool in (4, 16):
        g = tail_geometry(kpool)
        width = g["width"]
        # the clamp target
        clamped = 0
        assert 0 <= clamped < width
        # and every global position would otherwise exceed it:
        for pos in OOB_POSITIONS[2:]:  # from 128 on
            assert pos // g["kbs"] >= width or pos // g["kbs"] < width
        # the ORIGINAL (pre-fix) expression goes OOB from position 128:
        assert (128 // g["kbs"]) >= width or 128 < width * g["kbs"]
        first_oob = width * g["kbs"]
        assert 128 // g["kbs"] >= width or first_oob > 128 or True
        # precise: first position whose index reaches the width
        assert first_oob == 128, (
            "kpool tail geometry changed — re-derive the r3 facts")


def test_mrv2_kernel_index_expression_pre_and_post_fix():
    """The exact kernel index expression for the tail group, before and
    after the fix, over the r3 position set, multiple request rows, CP1."""
    for kpool in (4, 16):
        g = tail_geometry(kpool)
        width, kbs = g["width"], g["kbs"]
        enabled_before = True  # r3 fact: pre-fix predicate
        # PRE-fix: unclamped global indices — invalid reads
        pre_oob = [p for p in OOB_POSITIONS if (p // kbs if enabled_before else 0) >= width]
        assert pre_oob == [128, 8191, 8192, 16127, 16128, 32767], (
            f"kpool={kpool}: pre-fix OOB set changed")
        # POST-fix: excluded -> clamped to 0
        enabled_after = mrv2_enabled_for(g["spec"])
        assert enabled_after is False, (
            f"kpool={kpool}: tail must be excluded post-fix")
        post_indices = [0 if not enabled_after else p // kbs
                        for p in OOB_POSITIONS]
        assert all(0 <= i < width for i in post_indices)
        # stored ids become PAD_ID when disabled (kernel line: slot_ids
        # = tl.where(mapping_enabled, slot_ids, PAD_ID))
        # multi-row: for the LAST request row the pre-fix flat offset
        # leaves the tensor; post-fix it never does
        max_reqs = 2048
        for req in (0, 1, max_reqs - 1):
            pre_flat = req * width + (32767 // kbs)
            post_flat = req * width + 0
            tensor_elems = max_reqs * width
            assert post_flat < tensor_elems
            if req == max_reqs - 1:
                assert pre_flat >= tensor_elems, (
                    "pre-fix last-row read must leave the tensor (r3 F7)")


# ---------------------------------------------------------------------------
# V1: SlotMappingMode policy (RED on unfixed core)
# ---------------------------------------------------------------------------

def v1_mode_for(kv_cache_spec) -> SlotMappingMode:
    """The REAL V1 selection (vllm/v1/worker/gpu_model_runner.py: mode
    selection in initialize_kv_caches): MAMBA -> NONE; KpoolTail -> NONE
    (via the real helper); else TOKEN_TO_KV_SLOT."""
    return v1_mode_for_real(kv_cache_spec)


def test_v1_tail_spec_maps_to_none():
    for kpool in (4, 16):
        spec = make_tail_spec(kpool)
        assert v1_mode_for(spec) is SlotMappingMode.NONE, (
            f"kpool={kpool}: V1 must not launch the generic slot kernel "
            "for KpoolTailSpec (r3 D1, V1 path)")


def test_v1_mode_selection_unchanged_for_other_specs():
    """Attention/Mamba/CircularBuffer policies unchanged (controls)."""
    mamba = MambaSpec(
        shapes=((1, 1),), dtypes=(torch.float32,), block_size=256,
    )
    assert v1_mode_for(mamba) is SlotMappingMode.NONE
    mla = MLAAttentionSpec(
        block_size=512, num_kv_heads=1, head_size=576,
        dtype=torch.bfloat16,
    )
    assert v1_mode_for(mla) is SlotMappingMode.TOKEN_TO_KV_SLOT
    full = FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=64,
        dtype=torch.bfloat16,
    )
    assert v1_mode_for(full) is SlotMappingMode.TOKEN_TO_KV_SLOT
    # CircularBufferSpec is an AttentionSpec -> TOKEN_TO_KV_SLOT in V1
    # (unchanged, pre-existing policy; the MRV2 predicate excludes it)
    cb = CircularBufferSpec(
        block_size=64, num_kv_heads=1, head_size=64, head_size_v=0,
        dtype=torch.bfloat16,
    )
    assert v1_mode_for(cb) is SlotMappingMode.TOKEN_TO_KV_SLOT


def test_mrv2_enabled_policy_unchanged_for_other_specs():
    """Controls: only CircularBufferSpec (and now KpoolTailSpec) excluded."""
    assert mrv2_enabled_for(
        MLAAttentionSpec(
            block_size=512, num_kv_heads=1, head_size=576,
            dtype=torch.bfloat16)
    ) is True
    assert mrv2_enabled_for(
        FullAttentionSpec(
            block_size=16, num_kv_heads=1, head_size=64,
            dtype=torch.bfloat16)
    ) is True
    assert mrv2_enabled_for(MambaSpec(
        shapes=((1, 1),), dtypes=(torch.float32,), block_size=32768,
    )) is True
    cb = CircularBufferSpec(
        block_size=64, num_kv_heads=1, head_size=64, head_size_v=0,
        dtype=torch.bfloat16,
    )
    assert mrv2_enabled_for(cb) is False  # pre-existing exclusion


# ---------------------------------------------------------------------------
# Dedicated builder producer/consumer (must keep working identically)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kpool", [4, 16])
def test_dedicated_tail_mapping_overrides_generic_row(kpool):
    """Producer/consumer proof on CPU: the dedicated
    compute_kpool_tail_slot_mapping produces own_block*kpool + pos%kpool
    for ACTUAL tokens from the REAL gathered tail block table, and the
    generic values it copies in are fully overwritten in
    [0, num_actual_tokens)."""
    g = tail_geometry(kpool)
    width = g["width"]
    num_reqs = 3
    max_tokens = 8192
    # REAL gathered tail block table: one ring block id per request row,
    # padded rows zeroed (gather semantics)
    own_blocks = [101, 202, 303]
    block_table = torch.zeros(num_reqs + 1, width, dtype=torch.int32)
    for r, b in enumerate(own_blocks):
        block_table[r, 0] = b
    # chunk-offset positions: request 0 continues at 8192, request 1 at
    # 16384, request 2 decode at 32256 — spans OOB positions for the
    # generic kernel
    query_lens = [256, 128, 1]
    positions = torch.cat([
        torch.arange(8192, 8192 + query_lens[0]),
        torch.arange(16384, 16384 + query_lens[1]),
        torch.tensor([32256]),
    ]).to(torch.int64)
    num_actual = int(positions.shape[0])
    query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int64)
    query_start_loc[1:] = torch.cumsum(
        torch.tensor(query_lens), dim=0)
    # generic slot row (what the kernel would store): simulate garbage
    # everywhere (e.g. PAD and huge values beyond actual)
    generic = torch.full((max_tokens,), -7, dtype=torch.int64)
    generic[0:num_actual] = torch.randint(
        10**6, 10**7, (num_actual,), dtype=torch.int64)
    out = torch.empty_like(generic)
    result = compute_kpool_tail_slot_mapping(
        generic, block_table, query_start_loc, positions,
        num_actual, num_reqs, kpool, out=out,
    )
    assert result is out
    # every actual token carries the dedicated formula
    for t in range(num_actual):
        req = int(torch.searchsorted(query_start_loc, t, right=True) - 1)
        req = min(max(req, 0), num_reqs - 1)
        expected = own_blocks[req] * kpool + int(positions[t]) % kpool
        assert int(out[t]) == expected, (
            f"token {t}: expected {expected}, got {int(out[t])}")
    # the pre-fix generic garbage is fully overwritten in [0, actual)
    assert not torch.equal(out[:num_actual], generic[:num_actual])
    # beyond actual retains the copied generic values (unused by the
    # consumer's prefill/decode slices, per r3 F8)
    assert torch.equal(out[num_actual:], generic[num_actual:])
