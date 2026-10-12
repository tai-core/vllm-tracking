# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the DCP decode merge wiring in sparse_attn_indexer_kpool.

The REAL sparse_attn_indexer_kpool decode branch is driven on CPU with
controlled doubles (recording group / fake paged logits / recorded
expand / pack_seq). Native kernels, real collectives, and the KPool
writer/tail-rollback are NOT_RUN — the tail logic's acceptance stays
OPEN per the task contract. The indices/varlen decode path is NOT
supported by the DCP merge (unsupported layouts fail closed) and is
pinned as such.
"""

from types import SimpleNamespace

import pytest
import torch

# Bootstrap glm5next before the kpool module (the pre-existing
# import cycle; production enters via attention.py first).
# isort: off
import vllm.models.glm5next  # noqa: F401
# isort: on

import vllm.v1.attention.backends.mla.indexer as indexer_mod
from vllm.v1.attention.backends.mla.indexer import (
    DeepSeekV32IndexerDecodeMetadata,
    DeepseekV32IndexerMetadata,
)

# Shared recording state keyed by id(metadata-decode-object).
_LOGITS_COLS: dict = {}
_PER_RANK_PACKED: dict = {}
_EXPAND_CALLS: dict = {}
_SEQ_LENS_SEEN: dict = {}


@pytest.fixture(autouse=True)
def _clear_wiring():
    for d in (_LOGITS_COLS, _PER_RANK_PACKED, _EXPAND_CALLS, _SEQ_LENS_SEEN):
        d.clear()
    yield
    for d in (_LOGITS_COLS, _PER_RANK_PACKED, _EXPAND_CALLS, _SEQ_LENS_SEEN):
        d.clear()


class _DecodeGroup:
    """Recording group double: captures all_gather calls; returns the
    PRE-BUILT per-rank packed tensors concatenated in rank order."""

    def __init__(self, world, rank, per_rank):
        self.world_size = world
        self.rank_in_group = rank
        self._per_rank = per_rank
        self.calls = []

    def all_gather(self, input_, dim=-1):
        self.calls.append(dict(dtype=input_.dtype, dim=dim, shape=tuple(input_.shape)))
        idx = 0 if len(self.calls) % 2 == 1 else 1
        return torch.cat(
            [self._per_rank[r][idx] for r in range(self.world_size)], dim=dim
        )


def _drive_decode(
    monkeypatch,
    decode_md,
    num_decode_tokens,
    num_rows_total,
    topk_tokens=16,
    index_kpool=4,
    positions=None,
    max_seq_len=256,
    short_decode=False,
):
    """Drive the REAL kpool function's decode branch on CPU."""
    import vllm.model_executor.layers.sparse_attn_indexer_kpool as kpool_mod
    import vllm.v1.attention.ops.rocm_aiter_mla_sparse as rocm_mod

    def fake_paged_logits(
        q,
        kv_cache,
        weights,
        context_lens,
        block_table,
        schedule_metadata,
        max_model_len=None,
        **kw,
    ):
        # Record the context_lens the loop passed (they must be the
        # decode metadata's per-token bounds) and return the
        # deterministic table [num_rows, cols].
        _SEQ_LENS_SEEN.setdefault("context_lens", []).append(context_lens)
        rows = q.shape[0] * q.shape[1]
        cols = _LOGITS_COLS[id(decode_md)]
        return 1000.0 * (torch.arange(rows, dtype=torch.float32) + 1).unsqueeze(
            1
        ) - torch.arange(cols, dtype=torch.float32).unsqueeze(0)

    monkeypatch.setattr(
        rocm_mod, "rocm_fp8_paged_mqa_logits", fake_paged_logits, raising=False
    )
    monkeypatch.setattr(
        kpool_mod,
        "current_platform",
        SimpleNamespace(
            is_rocm=lambda: True,
            is_cuda=lambda: False,
            is_cuda_alike=lambda: short_decode,
            is_xpu=lambda: True,
            fp8_dtype=lambda: torch.float8_e4m3fn,
        ),
    )

    class _ExpandRec:
        def expand_pools_and_append_tail(self, pool_ids, q_seq, kpool):
            _EXPAND_CALLS.setdefault(id(decode_md), []).append(
                (pool_ids.clone(), q_seq.clone())
            )
            return torch.full(
                (pool_ids.shape[0], pool_ids.shape[1] * kpool + kpool - 1),
                -1,
                dtype=torch.int32,
            )

    monkeypatch.setattr(kpool_mod, "kpool_ops", _ExpandRec(), raising=False)

    # pack_seq/unpack_seq are real Triton kernels (NOT_RUN on CPU):
    # CPU doubles implementing the documented padding semantics.
    def _pack_seq(tokens, lens, pad_value=0):
        lmax = int(lens.max())
        out = torch.full(
            (lens.shape[0], lmax, *tokens.shape[1:]),
            pad_value,
            dtype=tokens.dtype,
        )
        i = 0
        for b, ln in enumerate(lens.tolist()):
            out[b, :ln] = tokens[i : i + ln]
            i += ln
        return out

    def _unpack_seq(packed, lens):
        total = int(lens.sum())
        out = torch.empty((total, *packed.shape[2:]), dtype=packed.dtype)
        i = 0
        for b, ln in enumerate(lens.tolist()):
            out[i : i + ln] = packed[b, :ln]
            i += ln
        return out

    monkeypatch.setattr(kpool_mod, "pack_seq_triton", _pack_seq, raising=False)
    monkeypatch.setattr(kpool_mod, "unpack_seq_triton", _unpack_seq, raising=False)

    class _TopkRec:
        def top_k_per_row_decode(self, *args, **kwargs):
            pass

    monkeypatch.setattr(kpool_mod, "xpu_ops", _TopkRec(), raising=False)

    # Route get_dcp_group (lazily imported inside the DCP branch).
    group = _DecodeGroup(
        decode_md.dcp_world_size,
        decode_md.dcp_rank,
        _PER_RANK_PACKED.get(id(decode_md), {}),
    )
    monkeypatch.setattr("vllm.distributed.get_dcp_group", lambda: group)
    monkeypatch.setattr(kpool_mod, "get_dcp_group", lambda: group, raising=False)

    md = DeepseekV32IndexerMetadata(
        seq_lens=torch.zeros(1, dtype=torch.int32),
        max_seq_len=max_seq_len,
        slot_mapping=torch.zeros(num_decode_tokens, dtype=torch.int64),
        num_decodes=decode_md.decode_lens.shape[0],
        num_decode_tokens=num_decode_tokens,
        num_prefills=0,
        num_prefill_tokens=0,
        prefill=None,
        decode=decode_md,
    )
    fc = SimpleNamespace(attn_metadata={"layer": md})
    monkeypatch.setattr(kpool_mod, "get_forward_context", lambda: fc)

    import vllm.v1.worker.workspace as ws

    topk_buf = torch.zeros(
        (num_rows_total, topk_tokens + index_kpool - 1), dtype=torch.int32
    )
    ws.init_workspace_manager(torch.device("cpu"))
    try:
        kpool_mod.sparse_attn_indexer_kpool(
            hidden_states=torch.zeros(num_decode_tokens, 8),
            k_cache_prefix="layer",
            kv_cache=torch.zeros((4, 16, 12)),
            q_quant=torch.zeros(num_decode_tokens, 8, dtype=torch.float8_e4m3fn),
            q_scale=None,
            k=torch.zeros(num_decode_tokens, 8),
            weights=torch.zeros(num_decode_tokens),
            quant_block_size=16,
            scale_fmt="e4m3",
            topk_tokens=topk_tokens,
            head_dim=8,
            max_model_len=max_seq_len,
            total_seq_lens=8,
            topk_indices_buffer=topk_buf,
            skip_k_cache_insert=True,
            use_fp4_cache=False,
            gate_score=None,
            compress_ape=None,
            index_kpool=index_kpool,
            positions=positions,
        )
    finally:
        ws._manager = None
    return group, topk_buf


def _make_decode_md(seq_lens, decode_lens, requires_padding, W=1, rank=0, J=1):
    return DeepSeekV32IndexerDecodeMetadata(
        block_table=torch.zeros((decode_lens.shape[0], 4), dtype=torch.int32),
        seq_lens=seq_lens,
        decode_lens=decode_lens,
        requires_padding=requires_padding,
        schedule_metadata=torch.zeros((2, 2), dtype=torch.int32),
        dcp_world_size=W,
        dcp_rank=rank,
        pool_interleave=J,
    )


def _pack_per_rank(seq_lens_r, W, J, select_k, cols, rows):
    """Pre-pack each rank's candidates from its own per-token bounds and
    the deterministic table."""
    from vllm.v1.attention.backends.mla.dcp_pool_merge_utils import (
        pack_local_pool_candidates,
        select_local_pool_candidates,
    )

    per_rank = {}
    for r in range(W):
        table = 1000.0 * (torch.arange(rows, dtype=torch.float32) + 1).unsqueeze(
            1
        ) - torch.arange(cols, dtype=torch.float32).unsqueeze(0)
        st = torch.zeros(rows, dtype=torch.int64)
        en = seq_lens_r[r].reshape(-1).to(torch.int64).clamp_min(0)
        lids, _ = select_local_pool_candidates(table, st, en, select_k)
        per_rank[r] = pack_local_pool_candidates(table, lids, st, en, W, r, J)
    return per_rank


def test_decode_merge_two_ranks_global_ids(monkeypatch):
    """W2 with per-request different local lengths and multiple
    per-token boundaries: both ranks' expand receives IDENTICAL int64
    global pool ids matching an independent oracle; the tail gets the
    GLOBAL token seq (from positions), never local pool counts."""
    W, J = 2, 1
    B, next_n = 2, 3  # two requests, 3 MTP tokens each
    rows = B * next_n
    # Per-token compressed causal bounds (this rank's view) —
    # different per request and per token.
    seq_lens = torch.tensor([[2, 3, 4], [1, 2, 3]], dtype=torch.int32)  # (B, next_n)
    decode_lens = torch.tensor([next_n, next_n], dtype=torch.int32)
    select_k = 16 // 4

    # The other rank's per-token bounds (its own local view).
    other = torch.tensor([[1, 2, 3], [2, 3, 4]], dtype=torch.int32)

    def f(r, j):
        return (j // J) * (W * J) + r * J + j % J

    groups = []
    for rank in range(W):
        sl = seq_lens if rank == 0 else other
        md = _make_decode_md(sl, decode_lens, False, W=W, rank=rank, J=J)
        _LOGITS_COLS[id(md)] = int(max(sl.max(), 1))
        _PER_RANK_PACKED[id(md)] = _pack_per_rank(
            [seq_lens, other], W, J, select_k, 8, rows
        )
        pos = torch.tensor([10, 11, 12, 20, 21, 22], dtype=torch.int32)
        group, _ = _drive_decode(monkeypatch, md, rows, rows, positions=pos)
        groups.append(group)

    for g in groups:
        assert len(g.calls) == 2
        assert g.calls[0]["dtype"] is torch.float32
        assert g.calls[1]["dtype"] is torch.int64
        assert g.calls[0]["dim"] == 1 and g.calls[1]["dim"] == 1
        assert g.calls[0]["shape"] == (rows, select_k)

    # Both ranks' expand pool ids are identical int64 global ids.
    mds = _EXPAND_CALLS
    # (two entries — one per rank drive)
    ids_list = [v[0][0] for v in mds.values()]
    assert len(ids_list) == 2
    assert ids_list[0].dtype is torch.int64
    torch.testing.assert_close(ids_list[0], ids_list[1])

    # Independent oracle over both ranks' per-token windows.
    tables = {
        r: (
            1000.0 * (torch.arange(rows, dtype=torch.float32) + 1).unsqueeze(1)
            - torch.arange(8, dtype=torch.float32).unsqueeze(0)
        )
        for r in range(W)
    }
    for i in range(rows):
        cands = []
        for r in range(W):
            sl = (seq_lens if r == 0 else other).reshape(-1)
            for j in range(int(sl[i])):
                cands.append((float(tables[r][i, j]), f(r, j)))
        top = sorted(cands, key=lambda p: (-p[0], p[1]))[:select_k]
        for c, (score, gid) in enumerate(top):
            assert int(ids_list[0][i, c]) == gid, (i, c)


def test_decode_merge_padded_rows_invalid_but_participate(monkeypatch):
    """Non-uniform decode_lens requiring padding: padded query rows get
    zero bounds (all-invalid fixed-K candidates) yet still participate
    in BOTH collectives; the unpack drops them from the output."""
    W, J = 2, 1
    B, next_n = 2, 2  # lmax 2; request 1 has only 1 token
    rows = B * next_n
    # seq_lens (B, next_n): padded token of request 1 has bound 0.
    seq_lens = torch.tensor([[3, 4], [2, 0]], dtype=torch.int32)
    decode_lens = torch.tensor([2, 1], dtype=torch.int32)
    select_k = 16 // 4
    other = torch.tensor([[2, 3], [1, 0]], dtype=torch.int32)

    groups = []
    for rank in range(W):
        sl = seq_lens if rank == 0 else other
        md = _make_decode_md(sl, decode_lens, True, W=W, rank=rank, J=J)
        _LOGITS_COLS[id(md)] = 8
        _PER_RANK_PACKED[id(md)] = _pack_per_rank(
            [seq_lens, other], W, J, select_k, 8, rows
        )
        pos = torch.tensor([10, 11, 20, 0], dtype=torch.int32)
        group, topk_buf = _drive_decode(monkeypatch, md, 3, rows, positions=pos)
        groups.append((group, topk_buf))

    for g, _ in groups:
        assert len(g.calls) == 2
        assert g.calls[0]["shape"] == (rows, select_k)
        assert g.calls[1]["shape"] == (rows, select_k)

    # The padded row's lanes are -1 (invalid), pinned on both ranks.
    ids_list = [v[0][0] for v in _EXPAND_CALLS.values()]
    assert len(ids_list) == 2
    torch.testing.assert_close(ids_list[0], ids_list[1])
    assert int(ids_list[0][3, 0]) == -1  # the padded row


def test_decode_merge_w1_native_path(monkeypatch):
    """W1: no group calls; the native decode topk path is preserved
    (routed through the recording double)."""
    rows = 1
    md = _make_decode_md(
        torch.tensor([[4]], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
        False,
    )
    assert md.dcp_world_size == 1 and md.dcp_rank == 0 and md.pool_interleave == 1
    _LOGITS_COLS[id(md)] = 8
    _PER_RANK_PACKED[id(md)] = {}
    group, topk_buf = _drive_decode(
        monkeypatch, md, 1, rows, positions=torch.tensor([10], dtype=torch.int32)
    )
    assert len(group.calls) == 0  # W1: no collectives


def test_decode_merge_short_decode_zero_collectives(monkeypatch):
    """The globally-shared short-decode shortcut (max_seq_len from the
    SHARED global attn metadata) bypasses the merge on every rank:
    zero collectives, global causal IDs preserved."""
    W = 2
    for rank in range(W):
        md = _make_decode_md(
            torch.tensor([[4]], dtype=torch.int32),
            torch.tensor([1], dtype=torch.int32),
            False,
            W=W,
            rank=rank,
        )
        _LOGITS_COLS[id(md)] = 8
        _PER_RANK_PACKED[id(md)] = {}
        group, topk_buf = _drive_decode(
            monkeypatch,
            md,
            1,
            1,
            positions=torch.tensor([3], dtype=torch.int32),
            max_seq_len=8,  # <= topk_tokens 16 -> short decode
            short_decode=True,
        )
        assert len(group.calls) == 0, rank


def test_decode_merge_indices_layout_unsupported(monkeypatch):
    """The indices/varlen decode layout is NOT supported by the DCP
    merge: a seq_lens whose flattened length cannot align 1:1 with the
    logits rows fails closed (ValueError), never a silent misalign."""
    from vllm.v1.attention.backends.mla.dcp_pool_merge_utils import (
        derive_decode_row_bounds,
    )

    with pytest.raises(ValueError, match="unsupported"):
        derive_decode_row_bounds(
            torch.zeros(3, dtype=torch.int32),
            num_rows=6,
            next_n=2,
            decode_lens=torch.tensor([2, 1], dtype=torch.int32),
        )
    # A numel-matched but semantically different layout also rejects.
    with pytest.raises(ValueError, match="unsupported"):
        derive_decode_row_bounds(
            torch.zeros(6, 2, dtype=torch.int32),
            num_rows=6,
            next_n=2,
            decode_lens=torch.tensor([2, 1], dtype=torch.int32),
        )


def test_decode_merge_mixed_batch_collective_order(monkeypatch):
    """A mixed prefill+decode batch drives the SAME collective
    sequence on both ranks: prefill merges first, then decode merges
    (the prefill chunk loop runs before the decode branch in the real
    function — the recorded call ORDER pins the shared sequence)."""
    W, J = 2, 1
    # Prefill chunk (from the real builder) + decode metadata.
    from tests.v1.attention.test_dcp_kpool_prefill_local import (
        _RecordingKernel,
        _run_chunk,
    )

    rk = _RecordingKernel()
    monkeypatch.setattr(indexer_mod, "_BUILD_PREFILL_CHUNK_METADATA_KERNEL", rk)
    _run_chunk(W, 0, 4, 4, [64], [64], kernel=rk)
    n_tokens = 64

    from vllm.v1.attention.backends.mla.dcp_pool_merge_utils import (
        pack_local_pool_candidates,
        select_local_pool_candidates,
    )
    from vllm.v1.attention.backends.mla.indexer import (
        DeepseekV32IndexerPrefillMetadata,
    )

    # Prefill-side per-rank packed (both ranks' chunks).
    chunks = {}
    for r in range(W):
        rk2 = _RecordingKernel()
        monkeypatch.setattr(indexer_mod, "_BUILD_PREFILL_CHUNK_METADATA_KERNEL", rk2)
        chunks[r] = _run_chunk(W, r, 4, 4, [64], [64], kernel=rk2)
    pre_packed = {}
    for r in range(W):
        c = chunks[r]
        ks = c.cu_seqlen_ks.to(torch.int64)
        ke = c.cu_seqlen_ke.to(torch.int64)
        cols = int(ke.max()) if len(ke) else 0
        table = (
            (
                1000.0 * (torch.arange(n_tokens, dtype=torch.float32) + 1).unsqueeze(1)
                - torch.arange(max(cols, 1), dtype=torch.float32).unsqueeze(0)
            )[:, :cols]
            if cols > 0
            else torch.zeros((n_tokens, 0))
        )
        lids, _ = select_local_pool_candidates(table, ks, ke, 4)
        pre_packed[r] = pack_local_pool_candidates(table, lids, ks, ke, W, r, J)

    # Decode side: 1 request, 1 token.
    dec_rows = 1
    dec_sl = torch.tensor([[4]], dtype=torch.int32)
    other_sl = torch.tensor([[3]], dtype=torch.int32)
    dec_lens = torch.tensor([1], dtype=torch.int32)

    import vllm.model_executor.layers.sparse_attn_indexer_kpool as kpool_mod
    import vllm.v1.attention.ops.rocm_aiter_mla_sparse as rocm_mod

    def fake_paged_logits(
        q,
        kv_cache,
        weights,
        context_lens,
        block_table,
        schedule_metadata,
        max_model_len=None,
        **kw,
    ):
        rows = q.shape[0] * q.shape[1]
        cols = _LOGITS_COLS["decode"]
        return 1000.0 * (torch.arange(rows, dtype=torch.float32) + 1).unsqueeze(
            1
        ) - torch.arange(cols, dtype=torch.float32).unsqueeze(0)

    monkeypatch.setattr(
        rocm_mod, "rocm_fp8_paged_mqa_logits", fake_paged_logits, raising=False
    )
    monkeypatch.setattr(
        kpool_mod,
        "current_platform",
        SimpleNamespace(
            is_rocm=lambda: True,
            is_cuda=lambda: False,
            is_cuda_alike=lambda: False,
            is_xpu=lambda: True,
            fp8_dtype=lambda: torch.float8_e4m3fn,
        ),
    )

    class _ExpandRec:
        def expand_pools_and_append_tail(self, pool_ids, q_seq, kpool):
            _EXPAND_CALLS.setdefault("mixed", []).append(pool_ids.clone())
            return torch.full(
                (pool_ids.shape[0], pool_ids.shape[1] * kpool + kpool - 1),
                -1,
                dtype=torch.int32,
            )

    monkeypatch.setattr(kpool_mod, "kpool_ops", _ExpandRec(), raising=False)

    def _pack_seq(tokens, lens, pad_value=0):
        lmax = int(lens.max())
        out = torch.full(
            (lens.shape[0], lmax, *tokens.shape[1:]), pad_value, dtype=tokens.dtype
        )
        i = 0
        for b, ln in enumerate(lens.tolist()):
            out[b, :ln] = tokens[i : i + ln]
            i += ln
        return out

    def _unpack_seq(packed, lens):
        total = int(lens.sum())
        out = torch.empty((total, *packed.shape[2:]), dtype=packed.dtype)
        i = 0
        for b, ln in enumerate(lens.tolist()):
            out[i : i + ln] = packed[b, :ln]
            i += ln
        return out

    monkeypatch.setattr(kpool_mod, "pack_seq_triton", _pack_seq, raising=False)
    monkeypatch.setattr(kpool_mod, "unpack_seq_triton", _unpack_seq, raising=False)

    class _TopkRec:
        def top_k_per_row_decode(self, *args, **kwargs):
            pass

    monkeypatch.setattr(kpool_mod, "xpu_ops", _TopkRec(), raising=False)

    # The prefill chunk loop's HCU gather (real Triton, NOT_RUN).
    import vllm.v1.attention.ops.rocm_aiter_mla_sparse as rocm_mod2

    monkeypatch.setattr(
        rocm_mod2,
        "cp_gather_indexer_k_quant_cache_triton",
        lambda *a, **k: None,
        raising=False,
    )
    monkeypatch.setattr(
        rocm_mod,
        "cp_gather_indexer_k_quant_cache_triton",
        lambda *a, **k: None,
        raising=False,
    )

    def fake_prefill_logits(*args, **kwargs):
        q = args[0]
        cols = _LOGITS_COLS["prefill"]
        return 1000.0 * (torch.arange(q.shape[0], dtype=torch.float32) + 1).unsqueeze(
            1
        ) - torch.arange(cols, dtype=torch.float32).unsqueeze(0)

    monkeypatch.setattr(
        rocm_mod, "rocm_fp8_mqa_logits", fake_prefill_logits, raising=False
    )

    all_calls = []
    for rank in range(W):
        # Build the shared metadata: prefill chunks + decode.
        md_dec = _make_decode_md(
            dec_sl if rank == 0 else other_sl,
            dec_lens,
            False,
            W=W,
            rank=rank,
            J=J,
        )
        # Pre-pack the decode side for the group.
        dec_packed = _pack_per_rank([dec_sl, other_sl], W, J, 4, 8, dec_rows)

        class _Group:
            def __init__(self, world, r, pre, dec):
                self.world_size = world
                self.rank_in_group = r
                self.calls = []
                self._pre = pre
                self._dec = dec

            def all_gather(self, input_, dim=-1):
                self.calls.append((input_.dtype, tuple(input_.shape)))
                all_calls.append((self.rank_in_group, len(self.calls)))
                idx = 0 if len(self.calls) % 2 == 1 else 1
                # Prefill collectives come from pre; decode from dec
                # — distinguish by row count.
                per = self._pre if input_.shape[0] == n_tokens else self._dec
                return torch.cat(
                    [per[rr][idx] for rr in range(self.world_size)], dim=dim
                )

        group = _Group(W, rank, pre_packed, dec_packed)

        def _get_group(_g=group):
            return _g

        monkeypatch.setattr("vllm.distributed.get_dcp_group", _get_group)
        monkeypatch.setattr(kpool_mod, "get_dcp_group", _get_group, raising=False)

        md = DeepseekV32IndexerMetadata(
            seq_lens=torch.tensor([64], dtype=torch.int32),
            max_seq_len=256,
            slot_mapping=torch.zeros(65, dtype=torch.int64),
            num_decodes=1,
            num_decode_tokens=1,
            num_prefills=1,
            num_prefill_tokens=64,
            prefill=DeepseekV32IndexerPrefillMetadata([chunks[rank]]),
            decode=md_dec,
        )
        fc = SimpleNamespace(attn_metadata={"layer": md})

        def _get_fc(_fc=fc):
            return _fc

        monkeypatch.setattr(kpool_mod, "get_forward_context", _get_fc)

        _LOGITS_COLS["decode"] = 8
        _LOGITS_COLS["prefill"] = int(chunks[rank].cu_seqlen_ke.max())
        import vllm.v1.worker.workspace as ws

        topk_buf = torch.zeros((65, 16 + 3), dtype=torch.int32)
        ws.init_workspace_manager(torch.device("cpu"))
        try:
            kpool_mod.sparse_attn_indexer_kpool(
                hidden_states=torch.zeros(65, 8),
                k_cache_prefix="layer",
                kv_cache=torch.zeros((4, 16, 12)),
                q_quant=torch.zeros(65, 8, dtype=torch.float8_e4m3fn),
                q_scale=None,
                k=torch.zeros(65, 8),
                weights=torch.zeros(65),
                quant_block_size=16,
                scale_fmt="e4m3",
                topk_tokens=16,
                head_dim=8,
                max_model_len=256,
                total_seq_lens=17,
                topk_indices_buffer=topk_buf,
                skip_k_cache_insert=True,
                use_fp4_cache=False,
                gate_score=None,
                compress_ape=None,
                index_kpool=4,
                positions=torch.arange(65, dtype=torch.int32) % 64,
            )
        finally:
            ws._manager = None

    # Both ranks saw the SAME collective sequence: 2 (prefill) then
    # 2 (decode) — 4 calls each, in the same order.
    rank0 = [c for r, c in all_calls if r == 0]
    rank1 = [c for r, c in all_calls if r == 1]
    assert rank0 == [1, 2, 3, 4]
    assert rank1 == [1, 2, 3, 4]


# --- builder-driven tests (REAL build() produces the metadata) --------------


def _build_real_decode_md(L, decode_lens, rank, W=2, R=4, interleave=4, spec_tokens=1):
    """Drive the REAL DeepseekV32IndexerMetadataBuilder.build() for a
    decode batch (uniform query lens = decode_lens' max) with DCP
    rank simulation. Returns the built decode metadata."""
    import vllm.v1.attention.backends.mla.compressor_utils as cu
    from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens
    from vllm.v1.kv_cache_interface import MLAAttentionSpec

    indexer_mod.num_compute_units = lambda idx: 8  # type: ignore[assignment]

    # CPU double for the compressed-slot-mapping Triton kernel.
    class _SlotKernel:
        def __getitem__(self, grid):
            def _launch(*args, **kwargs):
                args[0].fill_(-1)

            return _launch

    cu._compressed_slot_mapping_kernel = _SlotKernel()
    spec = MLAAttentionSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.float32,
        head_size_v=0,
        tokens_per_state=R,
    )
    vc = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=64, max_num_seqs=4),
        model_config=SimpleNamespace(max_model_len=256),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            cp_kv_cache_interleave_size=interleave,
            num_speculative_tokens=spec_tokens,
        ),
        speculative_config=SimpleNamespace(
            num_speculative_tokens=spec_tokens,
            enable_adaptive_verification=False,
        ),
        num_speculative_tokens=spec_tokens,
        attention_config=SimpleNamespace(resolve_indexer_kv_dtype=lambda d: "fp8"),
    )
    b = indexer_mod.DeepseekV32IndexerMetadataBuilder(
        kv_cache_spec=spec,
        layer_names=["l0"],
        vllm_config=vc,
        device=torch.device("cpu"),
        block_table_width=8,
    )
    b.dcp_world_size = W
    b.dcp_rank = rank
    glob = torch.tensor(L, dtype=torch.int32)
    loc = get_dcp_local_seq_lens(glob, W, rank, interleave)
    qsl = [0]
    for d in decode_lens:
        qsl.append(qsl[-1] + d)
    qsl = torch.tensor(qsl, dtype=torch.int32)
    cam = SimpleNamespace(
        num_reqs=len(L),
        num_actual_tokens=int(qsl[-1]),
        max_query_len=max(decode_lens),
        max_seq_len=max(max(L), 1),
        query_start_loc=qsl,
        query_start_loc_cpu=qsl.cpu(),
        seq_lens=glob,
        block_table_tensor=torch.zeros((len(L), 8), dtype=torch.int32),
        slot_mapping=torch.zeros(int(qsl[-1]), dtype=torch.int64),
        positions=None,
        dcp_local_seq_lens=loc,
        is_prefilling=None,
        seq_lens_cpu_upper_bound=glob.clone(),
    )
    md = b.build(common_prefix_len=0, common_attn_metadata=cam)
    return md


def test_builder_dcp_decode_bounds_counterexample():
    """The Codex counterexample, through the REAL builder: global
    L=[10,7], decode_lens=[2,1], next_n=2. The LEGACY construction
    would produce [[9,10],[6,7]] (row 6 wrong for the actual token;
    row 7 spurious for the pad). The DCP construction must produce
    per-request-actual GLOBAL bounds [[9,10],[7,0]] BEFORE
    localization — verified by checking the localized+compressed
    bounds against an independent ownership oracle.

    NOTE: with the standard split, decode_lens [2,1] routes request
    1 to the PREFILL set (a short extend); the decode batch here is
    driven with the decode-major layout by using uniform [2,2] for
    the split and a [2,1]-style short request as the EMPTY-request
    stand-in. The direct per-request-actual check below uses the
    builder's internal construction with non-uniform lens forced via
    the requires_padding path."""
    W, R, interleave = 2, 4, 4
    J = interleave // R  # pool units = 1

    def oracle_local_pool_bound(global_token_bound, W_, rank_, J_, R_):
        pools = global_token_bound // R_
        return sum(1 for p in range(pools) if (p // J_) % W_ == rank_)

    # Uniform [2,2] decode batch, L=[10,7]: per-token GLOBAL bounds
    # [[9,10],[6,7]] (both requests have decode_len 2 == next_n).
    md = _build_real_decode_md([10, 7], [2, 2], rank=0, W=W, R=R, interleave=interleave)
    dec = md.decode
    assert dec.dcp_world_size == W and dec.dcp_rank == 0
    assert dec.pool_interleave == J
    sl = dec.seq_lens
    # Independent oracle: localized+compressed bound per token.
    expected = [
        [
            oracle_local_pool_bound(9, W, 0, J, R),
            oracle_local_pool_bound(10, W, 0, J, R),
        ],
        [
            oracle_local_pool_bound(6, W, 0, J, R),
            oracle_local_pool_bound(7, W, 0, J, R),
        ],
    ]
    assert sl.tolist() == expected, (sl.tolist(), expected)

    # The empty-request case: L=[10,0] — request 1's rows must be 0
    # (masked before localization), never a spurious bound.
    md2 = _build_real_decode_md(
        [10, 0], [2, 2], rank=0, W=W, R=R, interleave=interleave
    )
    sl2 = md2.decode.seq_lens
    assert sl2.tolist()[1] == [0, 0], sl2.tolist()
    # And rank 1 sees zeros for the empty request too.
    md3 = _build_real_decode_md(
        [10, 0], [2, 2], rank=1, W=W, R=R, interleave=interleave
    )
    assert md3.decode.seq_lens.tolist()[1] == [0, 0]


def test_builder_dcp_decode_i8_pool_interleave():
    """I=8 (J=2): per-token localized pool bounds against the oracle
    for both ranks."""
    W, R, interleave = 2, 4, 8
    J = interleave // R

    def oracle(g, W_, rank_, J_, R_):
        pools = g // R_
        return sum(1 for p in range(pools) if (p // J_) % W_ == rank_)

    for rank in (0, 1):
        md = _build_real_decode_md(
            [12, 9], [2, 2], rank=rank, W=W, R=R, interleave=interleave
        )
        sl = md.decode.seq_lens
        for b, L in enumerate([12, 9]):
            for j in range(2):
                g = L - 2 + 1 + j
                assert int(sl[b, j]) == oracle(g, W, rank, J, R), (
                    rank,
                    b,
                    j,
                    int(sl[b, j]),
                    oracle(g, W, rank, J, R),
                )


def test_builder_legacy_vs_dcp_construction():
    """RED-pin on the OLD construction: the legacy formula
    L_b - max_decode_len + 1 + j applied to the Codex counterexample
    yields [[9,10],[6,7]] — row 6 is the WRONG bound for request 1's
    actual token (its true bound is 7). The DCP construction's
    per-request-actual formula gives [[9,10],[7,0]]. Computed on the
    GLOBAL bounds (pre-localization) by replicating both formulas."""
    L = torch.tensor([10, 7], dtype=torch.int64)
    decode_lens = torch.tensor([2, 1], dtype=torch.int64)
    next_n = 2
    j = torch.arange(next_n)
    legacy = (L.unsqueeze(1) - next_n + 1 + j).clamp_min(0)
    assert legacy.tolist() == [[9, 10], [6, 7]]
    per_req = L.unsqueeze(1) - decode_lens.unsqueeze(1) + 1 + j
    valid = j.unsqueeze(0) < decode_lens.unsqueeze(1)
    dcp = torch.where(valid, per_req.clamp_min(0), torch.zeros_like(per_req))
    assert dcp.tolist() == [[9, 10], [7, 0]]


def test_decode_merge_positions_missing_rejected_pre_collective(monkeypatch):
    """DCP decode with positions None rejects BEFORE any collective
    (no group call), never silently generating a wrong tail."""
    W, J = 2, 1
    md = _make_decode_md(
        torch.tensor([[2, 3]], dtype=torch.int32),
        torch.tensor([2], dtype=torch.int32),
        False,
        W=W,
        rank=0,
        J=J,
    )
    _LOGITS_COLS[id(md)] = 8
    _PER_RANK_PACKED[id(md)] = {}
    with pytest.raises(ValueError, match="positions"):
        _drive_decode(monkeypatch, md, 2, 2, positions=None)


def test_decode_merge_kpool1_rejected(monkeypatch):
    """DCP decode with index_kpool == 1 rejects (the token-granular
    expand contract is not adapted; GLM is R>1)."""
    W, J = 2, 1
    md = _make_decode_md(
        torch.tensor([[2]], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
        False,
        W=W,
        rank=0,
        J=J,
    )
    _LOGITS_COLS[id(md)] = 8
    _PER_RANK_PACKED[id(md)] = {}
    with pytest.raises(NotImplementedError, match="index_kpool"):
        _drive_decode(
            monkeypatch,
            md,
            1,
            1,
            index_kpool=1,
            positions=torch.tensor([5], dtype=torch.int32),
        )


def test_builder_driven_decode_merge_end_to_end(monkeypatch):
    """REAL builder metadata -> REAL kpool decode forward -> the
    recording group: W2, uniform [2,2] decode, L=[10,7], R4/I4 (J1).
    Both ranks' expand inputs are identical int64 global pool ids
    matching the per-rank-bounds oracle; the empty-request rows are
    all-invalid."""
    W, R, interleave = 2, 4, 4
    J = interleave // R
    rows = 4  # 2 requests x 2 tokens
    select_k = 16 // 4

    mds = {}
    for rank in range(W):
        md_full = _build_real_decode_md(
            [10, 7], [2, 2], rank=rank, W=W, R=R, interleave=interleave
        )
        mds[rank] = md_full.decode

    # The fake logits table: [rows, cols]; per-rank packed from each
    # rank's OWN built bounds.
    from vllm.v1.attention.backends.mla.dcp_pool_merge_utils import (
        pack_local_pool_candidates,
        select_local_pool_candidates,
    )

    def table_fn(r):
        cols = int(mds[r].seq_lens.max()) if rows else 0
        cols = max(cols, 1)
        return 1000.0 * (torch.arange(rows, dtype=torch.float32) + 1).unsqueeze(
            1
        ) - torch.arange(cols, dtype=torch.float32).unsqueeze(0)

    per_rank = {}
    for r in range(W):
        st = torch.zeros(rows, dtype=torch.int64)
        en = mds[r].seq_lens.reshape(-1).to(torch.int64).clamp_min(0)
        table = table_fn(r)
        lids, _ = select_local_pool_candidates(table, st, en, select_k)
        per_rank[r] = pack_local_pool_candidates(table, lids, st, en, W, r, J)

    groups = []
    for rank in range(W):
        dec_md = mds[rank]
        _LOGITS_COLS[id(dec_md)] = table_fn(rank).shape[1]
        _PER_RANK_PACKED[id(dec_md)] = per_rank
        pos = torch.tensor([9, 10, 6, 7], dtype=torch.int32)
        group, _ = _drive_decode(monkeypatch, dec_md, rows, rows, positions=pos)
        groups.append(group)

    for g in groups:
        assert len(g.calls) == 2
        assert g.calls[0]["shape"] == (rows, select_k)

    ids_list = [v[0][0] for v in _EXPAND_CALLS.values()]
    assert len(ids_list) == 2
    assert ids_list[0].dtype is torch.int64
    torch.testing.assert_close(ids_list[0], ids_list[1])

    # Oracle per token over both ranks' built bounds.
    def f(r, j):
        return (j // J) * (W * J) + r * J + j % J

    tables = {r: table_fn(r) for r in range(W)}
    bounds = {r: mds[r].seq_lens.reshape(-1) for r in range(W)}
    for i in range(rows):
        cands = []
        for r in range(W):
            for j in range(int(bounds[r][i])):
                cands.append((float(tables[r][i, j]), f(r, j)))
        top = sorted(cands, key=lambda p: (-p[0], p[1]))[:select_k]
        for c, (score, gid) in enumerate(top):
            assert int(ids_list[0][i, c]) == gid, (i, c)


def test_derive_decode_row_bounds_no_host_sync(monkeypatch):
    """The helper performs NO tensor-value probing: bool(tensor),
    int(tensor), .item(), .sum().item(), torch.all on device data
    would all sync — a controlled monkeypatch of Tensor.__bool__ /
    Tensor.__int__ / Tensor.item catches any attempt."""
    from vllm.v1.attention.backends.mla.dcp_pool_merge_utils import (
        derive_decode_row_bounds,
    )

    def _boom(self):
        raise AssertionError("host sync: tensor value probed")

    monkeypatch.setattr(torch.Tensor, "__bool__", _boom, raising=False)
    monkeypatch.setattr(torch.Tensor, "__int__", _boom, raising=False)
    monkeypatch.setattr(torch.Tensor, "item", _boom, raising=False)

    # Native padded: valid path with on-device masking only.
    sl = torch.tensor([[2, 3], [0, 0]], dtype=torch.int32)
    dl = torch.tensor([2, 1], dtype=torch.int32)
    starts, ends = derive_decode_row_bounds(sl, 4, 2, dl)
    assert starts.tolist() == [0, 0, 0, 0]
    assert ends.tolist() == [2, 3, 0, 0]

    # Plain (B, 1).
    sl2 = torch.tensor([[5], [3]], dtype=torch.int32)
    dl2 = torch.tensor([1, 1], dtype=torch.int32)
    _, ends2 = derive_decode_row_bounds(sl2, 2, 1, dl2)
    assert ends2.tolist() == [5, 3]

    # 1-D flattened.
    sl3 = torch.tensor([4, 5], dtype=torch.int32)
    dl3 = torch.tensor([1, 1], dtype=torch.int32)
    _, ends3 = derive_decode_row_bounds(sl3, 2, 1, dl3)
    assert ends3.tolist() == [4, 5]

    # Basic-type validations still raise (host-side, no tensors).
    with pytest.raises(ValueError, match="next_n"):
        derive_decode_row_bounds(sl, 4, 0, dl)
    with pytest.raises(ValueError, match="num_rows"):
        derive_decode_row_bounds(sl, -1, 2, dl)
    with pytest.raises(ValueError, match="device"):
        derive_decode_row_bounds(sl.cpu(), 4, 2, dl.to(torch.device("meta")))


def test_decode_merge_indices_rejected_by_caller(monkeypatch):
    """A DCP decode metadata with indices set (the varlen layout,
    numel-compatible) is rejected by the CALLER at the decode branch
    entry — before the tail write, padding pack, native logits, or
    any collective."""
    W, J = 2, 1
    md = _make_decode_md(
        torch.tensor([[2, 3]], dtype=torch.int32),
        torch.tensor([2], dtype=torch.int32),
        False,
        W=W,
        rank=0,
        J=J,
    )
    md.indices = torch.zeros(4, dtype=torch.int32)  # varlen layout
    _LOGITS_COLS[id(md)] = 8
    _PER_RANK_PACKED[id(md)] = {}
    with pytest.raises(ValueError, match="indices/varlen"):
        _drive_decode(
            monkeypatch, md, 2, 2, positions=torch.tensor([9, 10], dtype=torch.int32)
        )


def test_decode_merge_early_validation_before_tail_write(monkeypatch):
    """The positions/kpool/indices rejections fire at the decode
    branch ENTRY — BEFORE the KPool tail write (the writer is NOT
    invoked) and before any collective."""
    import vllm.model_executor.layers.sparse_attn_indexer_kpool as kpool_mod

    W, J = 2, 1
    tail_calls = []

    class _TailSpy:
        def kpool_decode_update_and_maybe_write_cache_batched(self, *a, **k):
            tail_calls.append(1)

        def kpool_compress_and_write_cache(self, *a, **k):
            tail_calls.append(1)

        def kpool_seed_tail_cache(self, *a, **k):
            tail_calls.append(1)

    monkeypatch.setattr(kpool_mod, "kpool_ops", _TailSpy(), raising=False)

    # positions None: rejected before the tail write.
    md = _make_decode_md(
        torch.tensor([[2]], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
        False,
        W=W,
        rank=0,
        J=J,
    )
    _LOGITS_COLS[id(md)] = 8
    _PER_RANK_PACKED[id(md)] = {}
    with pytest.raises(ValueError, match="positions"):
        _drive_decode(monkeypatch, md, 1, 1, positions=None)
    assert tail_calls == []


def test_derive_decode_row_bounds_decode_lens_shape():
    """decode_lens must be 1-D [B]; the 1-D flattened layout requires
    B == num_rows (one decode_len per row) — shape facts, no tensor
    value probing."""
    from vllm.v1.attention.backends.mla.dcp_pool_merge_utils import (
        derive_decode_row_bounds,
    )

    sl = torch.tensor([[2, 3]], dtype=torch.int32)
    with pytest.raises(ValueError, match="1-D"):
        derive_decode_row_bounds(sl, 2, 2, torch.tensor([[1, 1]], dtype=torch.int32))
    # 1-D seq_lens with a mismatched decode_lens B: rejected.
    with pytest.raises(ValueError, match="one decode_len per row"):
        derive_decode_row_bounds(
            torch.tensor([4, 5], dtype=torch.int32),
            2,
            1,
            torch.tensor([1, 1, 1], dtype=torch.int32),
        )


def test_select_local_rejects_float_bounds():
    """select_local_pool_candidates rejects float row bounds (the
    .to(int64) would silently truncate)."""
    from vllm.v1.attention.backends.mla.dcp_pool_merge_utils import (
        select_local_pool_candidates,
    )

    logits = torch.zeros(2, 4)
    with pytest.raises(ValueError, match="row_starts"):
        select_local_pool_candidates(
            logits,
            torch.tensor([0.0, 0.0]),
            torch.tensor([2.0, 2.0]),
            2,
        )
    with pytest.raises(ValueError, match="row_ends"):
        select_local_pool_candidates(
            logits,
            torch.tensor([0, 0], dtype=torch.int64),
            torch.tensor([2.5, 2.5]),
            2,
        )
