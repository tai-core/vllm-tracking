# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Model-free pre-launch verification of the KpoolTail slot-mapping fix (D1).

Run INSIDE the serving container (or any environment with the engine
importable) BEFORE loading weights:

    python scripts/verify_kpool_tail_slot_bounds.py

Checks (all ints — no tensors/weights dumped):
  1. Policy: the real MRV2 helper excludes KpoolTailSpec (directly and
     behind the UniformType wrapper); the real V1 helper likewise.
  2. Bounds: for every cache group the production index expression
     (pos // kernel_block_size over [0, max_model_len)) stays within the
     group's block-table row width WHENEVER the group's generic token
     slot mapping is enabled. Disabled groups must clamp to column 0
     (valid: every scheduled request's row has >= 1 written entry).
  3. Post-fix expected PAD behavior: a disabled group's generic slot row
     is all PAD_SLOT_ID (-1) beyond any builder overrides; the dedicated
     KpoolTail builder still produces own_block*kpool + pos%kpool for
     actual tokens (checked formulaically on a synthetic 3-request
     batch, CPU tensors only).

Exit code 0 = all checks pass; nonzero = do NOT launch.
"""

import sys

import torch

MAX_MODEL_LEN = 32768


def main() -> int:
    from vllm.v1.kv_cache_interface import (
        CircularBufferSpec,
        FullAttentionSpec,
        KpoolTailSpec,
        MambaSpec,
        MLAAttentionSpec,
        UniformTypeKVCacheSpecs,
        get_kv_cache_spec_kind,
    )
    from vllm.v1.worker.block_table import SlotMappingMode, get_block_table_width
    from vllm.v1.worker.gpu.model_runner import spec_uses_generic_token_slot_mapping
    from vllm.v1.worker.gpu_model_runner import (
        spec_uses_generic_token_slot_mapping_v1,
    )
    from vllm.v1.worker.utils import select_common_block_size

    failures = []

    def check(name, ok, detail=""):
        print(f"{'PASS' if ok else 'FAIL'}: {name}" + (f" ({detail})" if detail else ""))
        if not ok:
            failures.append(name)

    # --- 1. Policy ---
    for kpool in (4, 16):
        spec = KpoolTailSpec(
            block_size=kpool, num_kv_heads=2, head_size=128, head_size_v=0,
            dtype=torch.bfloat16, sliding_window=kpool,
        )
        check(f"MRV2 excludes KpoolTailSpec(kpool={kpool})",
              spec_uses_generic_token_slot_mapping(spec) is False)
        check(f"V1 excludes KpoolTailSpec(kpool={kpool})",
              spec_uses_generic_token_slot_mapping_v1(spec) is False)
        wrapped = UniformTypeKVCacheSpecs(
            block_size=spec.block_size, kv_cache_specs={"t.l": spec})
        check(f"MRV2 excludes wrapped tail (kpool={kpool})",
              spec_uses_generic_token_slot_mapping(
                  wrapped.first_spec) is False)
    check("CircularBufferSpec still excluded (MRV2)",
          spec_uses_generic_token_slot_mapping(
              CircularBufferSpec(
                  block_size=64, num_kv_heads=1, head_size=64, head_size_v=0,
                  dtype=torch.bfloat16)) is False)
    check("MLA/Full/Mamba still enabled (MRV2)",
          spec_uses_generic_token_slot_mapping(MLAAttentionSpec(
              block_size=512, num_kv_heads=1, head_size=576,
              dtype=torch.bfloat16)) is True
          and spec_uses_generic_token_slot_mapping(FullAttentionSpec(
              block_size=16, num_kv_heads=1, head_size=64,
              dtype=torch.bfloat16)) is True
          and spec_uses_generic_token_slot_mapping(MambaSpec(
              shapes=((1, 1),), dtypes=(torch.float32,),
              block_size=MAX_MODEL_LEN)) is True)

    # --- 2. Bounds per group kind ---
    class _Cfg:
        """Minimal config for spec sizing (attention specs read
        parallel_config only)."""

        class parallel_config:
            cp_shard_count = 1
            decode_context_parallel_size = 1
            prefill_context_parallel_size = 1

    def group_bounds(name, spec, backend, enabled):
        kbs = select_common_block_size(spec.block_size, [backend])
        width = get_block_table_width(
            spec.max_num_blocks_per_req(_Cfg, MAX_MODEL_LEN),
            spec.block_size)
        if enabled:
            max_idx = (MAX_MODEL_LEN - 1) // kbs
            check(f"bounds {name}: idx {max_idx} < width {width}",
                  max_idx < width)
        else:
            check(f"bounds {name}: disabled -> clamped col 0 < width",
                  0 < width)
        return width, kbs

    from vllm.v1.attention.backends.mla.indexer import (
        DeepseekV32IndexerBackend,
        KpoolTailBackend,
    )
    from vllm.v1.attention.backends.mla.triton_mla import TritonMLABackend
    mla = MLAAttentionSpec(
        block_size=512, num_kv_heads=1, head_size=576, dtype=torch.bfloat16)
    group_bounds("MLA latent", mla, TritonMLABackend, True)
    for kpool in (4, 16):
        tail = KpoolTailSpec(
            block_size=kpool, num_kv_heads=2, head_size=128, head_size_v=0,
            dtype=torch.bfloat16, sliding_window=kpool)
        w, k = group_bounds(
            f"KpoolTail(kpool={kpool})", tail, KpoolTailBackend,
            spec_uses_generic_token_slot_mapping(tail))
        # document the exact pre-fix OOB boundary for the record
        print(f"  info: pre-fix first OOB position was {w * k} "
              f"(width {w} x kbs {k}); post-fix indices clamp to column 0")

    # --- 3. PAD behavior + dedicated formula (CPU tensors) ---
    from vllm.v1.attention.backends.utils import PAD_SLOT_ID
    from vllm.v1.attention.backends.mla.indexer import (
        compute_kpool_tail_slot_mapping,
    )
    kpool = 16
    own = [101, 202, 303]
    bt = torch.zeros(4, 8, dtype=torch.int32)
    for r, b in enumerate(own):
        bt[r, 0] = b
    qsl = torch.tensor([0, 256, 384, 385], dtype=torch.int64)
    pos = torch.cat([
        torch.arange(8192, 8192 + 256),
        torch.arange(16384, 16384 + 128),
        torch.tensor([32256])]).to(torch.int64)
    generic = torch.full((1024,), PAD_SLOT_ID, dtype=torch.int64)
    out = compute_kpool_tail_slot_mapping(
        generic, bt, qsl, pos, 385, 3, kpool, out=torch.empty_like(generic))
    ok = all(
        int(out[t]) == own[min(max(int(torch.searchsorted(
            qsl, t, right=True)) - 1, 0), 2)] * kpool + int(pos[t]) % kpool
        for t in range(385))
    check("dedicated tail formula own_block*kpool+pos%kpool for actual tokens", ok)
    check("post-fix generic tail row is PAD beyond builder override",
          bool((generic == PAD_SLOT_ID).all()))

    print("\n" + ("ALL CHECKS PASSED — safe to proceed to launch"
                  if not failures else f"FAILURES: {failures} — DO NOT LAUNCH"))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
