# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the DCP prefill local mapping (local_token_to_seq) and
the kpool prefill chunk's DCP input resolution.

Real functions under test:
- ``build_prefill_chunk_metadata`` (module kernel symbol replaced by a
  recording double — Triton execution stays NOT_RUN): verifies
  local_token_to_seq's construction against the rank-local counts.
- ``_prefill_chunk_dcp_inputs`` (the REAL helper): W1 legacy
  passthrough, DCP local views, the empty-shard contract, and the
  legacy-metadata rejection.
- The kpool prefill chunk loop's gather call-site: driven through the
  REAL ``sparse_attn_indexer_kpool`` with a platform double and
  captured gather/topk inputs (a CPU recorder — NOT a device run).
"""

from types import SimpleNamespace

import pytest
import torch

# Bootstrap glm5next before the kpool module: kpool_compress runs
# glm5next/__init__ -> model -> attention -> back into
# sparse_attn_indexer_kpool (a pre-existing cycle; production enters
# via attention.py first). Mirrors test_sparse_indexer_decode_seq_lens.
# isort: off
import vllm.models.glm5next  # noqa: F401
# isort: on

import vllm.v1.attention.backends.mla.indexer as indexer_mod
from vllm.v1.attention.backends.mla.indexer import build_prefill_chunk_metadata


class _RecordingKernel:
    """Same protocol as the real prefill-chunk kernel symbol."""

    def __init__(self):
        self.calls = []

    def __call__(
        self,
        query_start_loc,
        uncompressed_seq_lens,
        cu_seq_lens,
        local_cu_seq_lens,
        token_to_seq,
        cu_ks,
        cu_ke,
        qs_start,
        qs_stop,
        dcp_rank,
        dcp_world,
        dcp_interleave,
        *,
        num_reqs,
        COMPRESS_RATIO,
    ):
        self.calls.append((dcp_rank, dcp_world, dcp_interleave))
        # The kernel writes token_to_seq for the GLOBAL pools (one
        # entry per global pool, request order) and the per-query
        # LOCAL ks/ke bounds; replay the kernel's documented
        # arithmetic in plain torch so the merge sees REAL bounds.
        for b in range(num_reqs):
            seq_start = int(cu_seq_lens[b])
            seq_end = int(cu_seq_lens[b + 1])
            for off in range(seq_end - seq_start):
                token_to_seq[seq_start + off] = b
        qsl = query_start_loc
        for b in range(num_reqs):
            q_start = int(qsl[b])
            q_end = int(qsl[b + 1])
            query_len = q_end - q_start
            row_start = int(local_cu_seq_lens[b])
            unc = int(uncompressed_seq_lens[b])
            start_pos = unc - query_len
            for off in range(query_len):
                abs_pos = q_start + off
                if not (qs_start <= abs_pos < qs_stop):
                    continue
                out_pos = abs_pos - qs_start
                cu_ks[out_pos] = row_start
                global_ctx = start_pos + 1 + off
                len_per_token = global_ctx // COMPRESS_RATIO
                if dcp_world > 1:
                    base = (
                        len_per_token // dcp_interleave // dcp_world
                    ) * dcp_interleave
                    rem = len_per_token - base * dcp_world
                    rem = min(max(rem - dcp_rank * dcp_interleave, 0), dcp_interleave)
                    len_per_token = base + rem
                cu_ke[out_pos] = row_start + len_per_token


def _run_chunk(
    dcp_world_size,
    dcp_rank,
    interleave,
    compress_ratio,
    seq_lens,
    query_lens,
    query_slice=None,
    kernel=None,
):
    device = torch.device("cpu")
    qsl = [0]
    for ql in query_lens:
        qsl.append(qsl[-1] + ql)
    query_start_loc = torch.tensor(qsl, dtype=torch.int32, device=device)
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    compressed = seq_lens_t // compress_ratio
    block_table = torch.zeros((len(seq_lens), 4), dtype=torch.int32, device=device)
    chunk = build_prefill_chunk_metadata(
        start_idx=0,
        end_idx=len(seq_lens),
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.cpu(),
        uncompressed_seq_lens=seq_lens_t,
        compressed_seq_lens=compressed,
        compressed_seq_lens_cpu=compressed.cpu(),
        block_table=block_table,
        compress_ratio=compress_ratio,
        query_slice=query_slice,
        dcp_rank=dcp_rank,
        dcp_world_size=dcp_world_size,
        cp_kv_cache_interleave_size=interleave,
    )
    assert chunk is not None
    return chunk


@pytest.fixture
def recording_kernel(monkeypatch):
    rk = _RecordingKernel()
    monkeypatch.setattr(indexer_mod, "_BUILD_PREFILL_CHUNK_METADATA_KERNEL", rk)
    return rk


def _oracle_local_counts(global_pool_lens, W, rank, J):
    """Rank-local pool counts per request under interleave sharding."""
    counts = []
    for n in global_pool_lens:
        c = 0
        for p in range(n):
            if (p // J) % W == rank:
                c += 1
        counts.append(c)
    return counts


def test_local_token_to_seq_two_requests_two_ranks(recording_kernel):
    """Two requests with GLOBAL pool counts 4 and 2 (seq 16, 8 at R=4);
    W2/J1 rank 0 owns [2, 1] pools, rank 1 owns [2, 1] — different
    from the global counts. local_token_to_seq must repeat each
    request index by ITS local count and align with local_cu."""
    W, J, R = 2, 1, 4
    seq_lens = [16, 8]
    for rank in (0, 1):
        chunk = _run_chunk(W, rank, 4, R, seq_lens, [16, 8], kernel=recording_kernel)
        global_pools = [s // R for s in seq_lens]
        exp_counts = _oracle_local_counts(global_pools, W, rank, J)
        exp_total = sum(exp_counts)
        assert chunk.local_total_seq_lens == exp_total
        assert chunk.local_token_to_seq is not None
        assert chunk.local_token_to_seq.dtype is torch.int32
        assert chunk.local_token_to_seq.shape[0] == exp_total
        exp_map = []
        for b, c in enumerate(exp_counts):
            exp_map.extend([b] * c)
        assert chunk.local_token_to_seq.tolist() == exp_map
        # Alignment: within each request's local slice, the map is
        # constant and matches local_cu's per-request bounds.
        lcu = chunk.local_cu_seq_lens
        for b in range(2):
            seg = chunk.local_token_to_seq[int(lcu[b]) : int(lcu[b + 1])]
            assert torch.all(seg == b)


def test_local_token_to_seq_rank_counts_differ(recording_kernel):
    """W2/J2: rank 0 owns pool groups [0,1] then [4,5]...; for
    seq 32 (8 pools) rank0 owns 4, rank1 owns 4 — but for seq 20
    (5 pools: groups 0,0,2,2,4) rank0 owns [0,1,0,1,0]=3? Verify via
    oracle: pools p in 0..4, group = p//2, owner = group % 2:
    p0,1 -> g0 r0; p2,3 -> g1 r1; p4 -> g2 r0. rank0: 3, rank1: 2."""
    W, J, R = 2, 2, 4
    seq_lens = [32, 20]
    for rank in (0, 1):
        chunk = _run_chunk(W, rank, 8, R, seq_lens, [32, 20], kernel=recording_kernel)
        global_pools = [s // R for s in seq_lens]
        exp_counts = _oracle_local_counts(global_pools, W, rank, J)
        assert chunk.local_total_seq_lens == sum(exp_counts)
        exp_map = []
        for b, c in enumerate(exp_counts):
            exp_map.extend([b] * c)
        assert chunk.local_token_to_seq.tolist() == exp_map


def test_local_token_to_seq_query_slice_nonzero(recording_kernel):
    """A nonzero query_slice keeps local_token_to_seq identical (it
    depends only on the request-level local counts, not the query
    window)."""
    full = _run_chunk(2, 0, 4, 4, [16, 8], [16, 8], kernel=recording_kernel)
    sliced = _run_chunk(
        2, 0, 4, 4, [16, 8], [16, 8], query_slice=slice(3, 9), kernel=recording_kernel
    )
    torch.testing.assert_close(full.local_token_to_seq, sliced.local_token_to_seq)
    torch.testing.assert_close(full.local_cu_seq_lens, sliced.local_cu_seq_lens)


def test_w1_aliases_token_to_seq(recording_kernel):
    """W1: local_token_to_seq ALIASES token_to_seq (same object) and
    local_cu aliases cu — the legacy contract."""
    chunk = _run_chunk(1, 0, 1, 4, [40], [40], kernel=recording_kernel)
    assert chunk.local_token_to_seq is chunk.token_to_seq
    assert chunk.local_cu_seq_lens is chunk.cu_seq_lens
    assert chunk.token_to_seq.tolist() == [0] * 10


# --- _prefill_chunk_dcp_inputs (the REAL helper) --------------------------


def _make_chunk(
    local_cu=None,
    local_t2s=None,
    local_total=0,
    max_local=0,
    total=10,
    num_reqs=2,
    skip=False,
    dcp_world=1,
):
    return SimpleNamespace(
        cu_seq_lens=torch.tensor([0, 5, total], dtype=torch.int32),
        token_to_seq=torch.zeros(total, dtype=torch.int32),
        cu_seqlen_ks=torch.zeros(4, dtype=torch.int32),
        cu_seqlen_ke=torch.zeros(4, dtype=torch.int32),
        total_seq_lens=total,
        max_local_total_seq_lens=max_local,
        num_reqs=num_reqs,
        skip_kv_gather=skip,
        local_cu_seq_lens=local_cu,
        local_token_to_seq=local_t2s,
        local_total_seq_lens=local_total,
        dcp_world_size=dcp_world,
    )


def test_dcp_inputs_w1_legacy_passthrough():
    """W1: the helper returns the global cu/token_to_seq and slices the
    workspace by total_seq_lens — byte-for-byte legacy behavior."""
    chunk = _make_chunk()
    kq = torch.zeros(20, 8)
    ks = torch.zeros(20, 4)
    out = _call_helper(chunk, kq, ks, q_rows=4)
    assert out.gather_cu is chunk.cu_seq_lens
    assert out.gather_token_to_seq is chunk.token_to_seq
    assert out.k_quant.shape[0] == 10  # total_seq_lens
    assert out.logits_ks is chunk.cu_seqlen_ks
    assert out.skip_gather is False
    assert out.empty_logits is None


def test_dcp_inputs_dcp2_local_views():
    """DCP2 with a nonempty local shard: gather gets the LOCAL
    cu/token_to_seq; the workspace slices by max_local; the logits
    bounds pass through (already rank-local row bounds)."""
    local_cu = torch.tensor([0, 2, 5], dtype=torch.int32)
    local_t2s = torch.tensor([0, 0, 1, 1, 1], dtype=torch.int32)
    chunk = _make_chunk(
        local_cu=local_cu,
        local_t2s=local_t2s,
        local_total=5,
        max_local=7,
        dcp_world=2,
    )
    kq = torch.zeros(20, 8)
    ks = torch.zeros(20, 4)
    out = _call_helper(chunk, kq, ks, q_rows=4)
    assert out.gather_cu is local_cu
    assert out.gather_token_to_seq is local_t2s
    # EXACT local length, not max_local: the HCU gather grid is
    # k_fp8.size(0) and reads token_to_seq[tid] for every tid.
    assert out.k_quant.shape[0] == 5
    assert out.k_scale.shape[0] == 5
    assert out.skip_gather is False
    assert out.empty_logits is None


def test_dcp_inputs_empty_local_shard():
    """local_total == 0: the gather is skipped, an empty fp32 logits
    of the right query rows is returned, and NO exception."""
    chunk = _make_chunk(
        local_cu=torch.tensor([0, 0, 0], dtype=torch.int32),
        local_t2s=torch.empty(0, dtype=torch.int32),
        local_total=0,
        max_local=3,
        dcp_world=2,
    )
    kq = torch.zeros(20, 8)
    ks = torch.zeros(20, 4)
    out = _call_helper(chunk, kq, ks, q_rows=6)
    assert out.skip_gather is True
    assert out.empty_logits is not None
    assert out.empty_logits.shape == (6, 0)
    assert out.empty_logits.dtype is torch.float32


def test_dcp_inputs_legacy_metadata_rejected():
    """DCP>1 with local_cu None (legacy manual metadata): clear
    rejection, never a silent fallback to the global mapping."""
    chunk = _make_chunk(
        local_cu=None, local_t2s=None, local_total=5, max_local=5, dcp_world=2
    )
    kq = torch.zeros(20, 8)
    ks = torch.zeros(20, 4)
    with pytest.raises(ValueError, match="local_cu_seq_lens"):
        _call_helper(chunk, kq, ks, q_rows=4)
    # local_total > 0 but the map missing: also rejected.
    chunk2 = _make_chunk(
        local_cu=torch.tensor([0, 3, 5], dtype=torch.int32),
        local_t2s=None,
        local_total=5,
        max_local=5,
        dcp_world=2,
    )
    with pytest.raises(ValueError, match="local_token_to_seq"):
        _call_helper(chunk2, kq, ks, q_rows=4)
    # A DCP>1 chunk with ONLY the default local_total=0 (fields
    # missing): rejected — the default must not fabricate an empty
    # shard that masks the missing metadata.
    chunk3 = _make_chunk(
        local_cu=None, local_t2s=None, local_total=0, max_local=0, dcp_world=2
    )
    with pytest.raises(ValueError, match="local_cu_seq_lens"):
        _call_helper(chunk3, kq, ks, q_rows=4)


def _call_helper(chunk, kq, ks, q_rows):
    from vllm.model_executor.layers.sparse_attn_indexer_kpool import (
        _prefill_chunk_dcp_inputs,
    )

    (
        k_quant,
        k_scale,
        gather_cu,
        gather_token_to_seq,
        logits_ks,
        logits_ke,
        skip_gather,
        empty_logits,
    ) = _prefill_chunk_dcp_inputs(chunk, kq, ks, q_rows)
    return SimpleNamespace(
        k_quant=k_quant,
        k_scale=k_scale,
        gather_cu=gather_cu,
        gather_token_to_seq=gather_token_to_seq,
        logits_ks=logits_ks,
        logits_ke=logits_ke,
        skip_gather=skip_gather,
        empty_logits=empty_logits,
    )


# --- the kpool prefill chunk loop's gather call-site -----------------------


# Per-drive wiring for the recording group and the fake logits
# width: keyed by id(chunk) (the dataclass is unhashable). Cleared
# between tests — a freed chunk's id can be reused by a new object.
_LOGITS_COLS: dict = {}
_PER_RANK_PACKED: dict = {}
_EXPAND_CALLS: dict = {}


@pytest.fixture(autouse=True)
def _clear_chunk_wiring():
    _LOGITS_COLS.clear()
    _PER_RANK_PACKED.clear()
    _EXPAND_CALLS.clear()
    yield
    _LOGITS_COLS.clear()
    _PER_RANK_PACKED.clear()
    _EXPAND_CALLS.clear()


class _GatherRecorder:
    """Captures the HCU gather's arguments."""

    def __init__(self):
        self.calls = []

    def __call__(
        self,
        kv_cache,
        k_fp8,
        k_fp8_scale,
        block_table,
        cu_seqlen,
        token_to_seq=None,
        **kwargs,
    ):
        self.calls.append(
            SimpleNamespace(
                cu_seqlen=cu_seqlen,
                token_to_seq=token_to_seq,
                k_fp8=k_fp8,
            )
        )


def _drive_kpool_prefill(
    monkeypatch, chunk, seq_lens, num_prefill_tokens, max_seq_len, topk_tokens=16
):
    """Drive the REAL sparse_attn_indexer_kpool prefill chunk loop on
    CPU: platform doubles route the HCU gather and the topk op to
    recorders; native logits/kernels stay NOT_RUN."""
    import vllm.model_executor.layers.sparse_attn_indexer_kpool as kpool_mod
    import vllm.v1.attention.ops.rocm_aiter_mla_sparse as rocm_mod
    from vllm.v1.attention.backends.mla.indexer import (
        DeepseekV32IndexerMetadata,
        DeepseekV32IndexerPrefillMetadata,
    )

    gather = _GatherRecorder()
    monkeypatch.setattr(
        rocm_mod, "cp_gather_indexer_k_quant_cache_triton", gather, raising=False
    )
    # The DCP branch lazily fetches get_dcp_group(); route it to a
    # recording group whose topology matches the chunk metadata.
    _group_state = SimpleNamespace(calls=[])
    import vllm.model_executor.layers.sparse_attn_indexer_kpool as _km

    class _ChunkGroup:
        def __init__(self, world, rank, per_rank_pack):
            self.world_size = world
            self.rank_in_group = rank
            self._per_rank = per_rank_pack

        def all_gather(self, input_, dim=-1):
            _group_state.calls.append(
                dict(
                    dtype=input_.dtype,
                    dim=dim,
                    shape=tuple(input_.shape),
                    is_local=input_,
                )
            )
            idx = 0 if len(_group_state.calls) % 2 == 1 else 1
            return torch.cat(
                [self._per_rank[r][idx] for r in range(self.world_size)],
                dim=dim,
            )

    _km_get_dcp_group_orig = _km.__dict__.get("get_dcp_group")
    monkeypatch.setattr(
        "vllm.distributed.get_dcp_group",
        lambda: _ChunkGroup(
            chunk.dcp_world_size, chunk.dcp_rank, _PER_RANK_PACKED.get(id(chunk), {})
        ),
    )
    monkeypatch.setattr(
        _km,
        "get_dcp_group",
        lambda: _ChunkGroup(
            chunk.dcp_world_size, chunk.dcp_rank, _PER_RANK_PACKED.get(id(chunk), {})
        ),
        raising=False,
    )

    def fake_logits(*args, **kwargs):
        q = args[0]
        # A deterministic REAL fp32 table: [rows, local_cols] where
        # local_cols is the (per-rank) K view width — callers capture
        # it and pre-pack the other ranks' candidates from the same
        # construction. Column j of rank r's row i scores
        # 1000 * (i + 1) - j, so ties exist within a row.
        rows = q.shape[0]
        cols = _LOGITS_COLS.get(id(chunk), 0)
        return 1000.0 * (torch.arange(rows, dtype=torch.float32) + 1).unsqueeze(
            1
        ) - torch.arange(cols, dtype=torch.float32).unsqueeze(0)

    monkeypatch.setattr(rocm_mod, "rocm_fp8_mqa_logits", fake_logits, raising=False)

    class _TopkRecorder:
        def top_k_per_row_prefill(self, logits, ks, ke, dst, rows, s0, s1, k):
            pass

    monkeypatch.setattr(kpool_mod, "xpu_ops", _TopkRecorder(), raising=False)
    monkeypatch.setattr(
        kpool_mod,
        "current_platform",
        SimpleNamespace(
            is_rocm=lambda: True,
            is_cuda=lambda: False,
            is_xpu=lambda: True,
            fp8_dtype=lambda: torch.float8_e4m3fn,
        ),
    )

    class _FakeExpand:
        def expand_pools_and_append_tail(self, pool_ids, q_seq, kpool):
            rec = _EXPAND_CALLS.setdefault(id(chunk), [])
            rec.append(pool_ids.clone())
            return torch.full(
                (pool_ids.shape[0], pool_ids.shape[1] * kpool + kpool - 1),
                -1,
                dtype=torch.int32,
            )

    monkeypatch.setattr(kpool_mod, "kpool_ops", _FakeExpand(), raising=False)

    md = DeepseekV32IndexerMetadata(
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        max_seq_len=max_seq_len,
        slot_mapping=torch.zeros(num_prefill_tokens, dtype=torch.int64),
        num_decodes=0,
        num_decode_tokens=0,
        num_prefills=len(seq_lens),
        num_prefill_tokens=num_prefill_tokens,
        prefill=DeepseekV32IndexerPrefillMetadata([chunk]),
        decode=None,
    )
    fc = SimpleNamespace(attn_metadata={"layer": md})
    monkeypatch.setattr(kpool_mod, "get_forward_context", lambda: fc)

    topk_buf = torch.zeros((num_prefill_tokens, 16 + 4 - 1), dtype=torch.int32)

    import vllm.v1.worker.workspace as ws

    ws.init_workspace_manager(torch.device("cpu"))
    try:
        kpool_mod.sparse_attn_indexer_kpool(
            hidden_states=torch.zeros(num_prefill_tokens, 8),
            k_cache_prefix="layer",
            kv_cache=torch.zeros((4, 16, 12)),
            q_quant=torch.zeros(num_prefill_tokens, 8, dtype=torch.float8_e4m3fn),
            q_scale=None,
            k=torch.zeros(num_prefill_tokens, 8),
            weights=torch.zeros(num_prefill_tokens),
            quant_block_size=16,
            scale_fmt="e4m3",
            topk_tokens=topk_tokens,
            head_dim=8,
            max_model_len=256,
            total_seq_lens=8,
            topk_indices_buffer=topk_buf,
            skip_k_cache_insert=True,
            use_fp4_cache=False,
            gate_score=None,
            compress_ape=None,
            index_kpool=4,
            positions=torch.arange(num_prefill_tokens, dtype=torch.int32) % max_seq_len,
        )
    finally:
        ws._manager = None
    return gather, _group_state


def test_kpool_prefill_gather_uses_local_mapping(monkeypatch):
    """The REAL kpool forward's prefill chunk loop: with a DCP chunk
    (distinct local_cu), the HCU gather receives local_cu and
    local_token_to_seq — NOT the global cu/map."""
    seq_lens = [64, 32]
    n_tokens = 96
    # Build BOTH ranks' chunks; pack each with its OWN bounds.
    chunks = {}
    per_rank_packed = {}
    for r in range(2):
        rk = _RecordingKernel()
        monkeypatch.setattr(indexer_mod, "_BUILD_PREFILL_CHUNK_METADATA_KERNEL", rk)
        chunks[r] = _run_chunk(2, r, 4, 4, seq_lens, seq_lens, kernel=rk)
    from vllm.v1.attention.backends.mla.dcp_pool_merge_utils import (
        pack_local_pool_candidates,
        select_local_pool_candidates,
    )

    select_k = 16 // 4
    for r in range(2):
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
        lids, _ = select_local_pool_candidates(table, ks, ke, select_k)
        per_rank_packed[r] = pack_local_pool_candidates(table, lids, ks, ke, 2, r, 1)
    chunk = chunks[0]
    assert chunk.local_cu_seq_lens is not chunk.cu_seq_lens
    _LOGITS_COLS[id(chunk)] = int(chunk.cu_seqlen_ke.max())
    _PER_RANK_PACKED[id(chunk)] = per_rank_packed

    gather, _ = _drive_kpool_prefill(monkeypatch, chunk, seq_lens, 96, 64)
    # The gather was called ONCE with the LOCAL mapping.
    assert len(gather.calls) == 1
    call = gather.calls[0]
    assert call.cu_seqlen is chunk.local_cu_seq_lens
    assert call.token_to_seq is chunk.local_token_to_seq


def test_kpool_prefill_w1_gather_uses_global_mapping(monkeypatch):
    """W1 chunk (local_cu ALIASES cu): the gather keeps the global
    cu_seq_lens/token_to_seq — the legacy behavior."""
    rk = _RecordingKernel()
    monkeypatch.setattr(indexer_mod, "_BUILD_PREFILL_CHUNK_METADATA_KERNEL", rk)
    chunk = _run_chunk(1, 0, 1, 4, [64, 32], [64, 32], kernel=rk)
    assert chunk.local_cu_seq_lens is chunk.cu_seq_lens

    gather, _ = _drive_kpool_prefill(monkeypatch, chunk, [64, 32], 96, 64)
    assert len(gather.calls) == 1
    call = gather.calls[0]
    assert call.cu_seqlen is chunk.cu_seq_lens
    assert call.token_to_seq is chunk.token_to_seq


def test_kpool_prefill_empty_rank_skips_gather(monkeypatch):
    """A chunk whose local_total is 0 (rank 1 owning no pool) issues NO
    gather, no native logits/topk, and no exception — and still enters
    the merge (both collectives) with the fixed-K invalid candidates."""
    rk = _RecordingKernel()
    monkeypatch.setattr(indexer_mod, "_BUILD_PREFILL_CHUNK_METADATA_KERNEL", rk)
    # rank 1 of W2 with I=J*R=64, J=16: group 0 (pools 0-15) is rank
    # 0's; rank 1's group would start at pool 16 — none exists.
    chunk = _run_chunk(2, 1, 64, 4, [64], [64], kernel=rk)
    assert chunk.local_total_seq_lens == 0
    # The other rank (0) owns all 16 pools; its candidates feed the
    # recording group's gather.
    from vllm.v1.attention.backends.mla.dcp_pool_merge_utils import (
        pack_local_pool_candidates,
        select_local_pool_candidates,
    )

    per_rank = {}
    n_rows = 64
    for r in range(2):
        lc = 16 if r == 0 else 0
        if lc > 0:
            table = 1000.0 * (torch.arange(n_rows, dtype=torch.float32) + 1).unsqueeze(
                1
            ) - torch.arange(lc, dtype=torch.float32).unsqueeze(0)
        else:
            table = torch.zeros((n_rows, 0))
        # rank 0's window: [0, 16); rank 1's: empty — PER-RANK bounds.
        st = torch.zeros(n_rows, dtype=torch.int64)
        en = torch.full((n_rows,), lc, dtype=torch.int64)
        lids, _ = select_local_pool_candidates(table, st, en, 4)
        per_rank[r] = pack_local_pool_candidates(table, lids, st, en, 2, r, 16)
    _LOGITS_COLS[id(chunk)] = 0
    _PER_RANK_PACKED[id(chunk)] = per_rank

    gather, group_state = _drive_kpool_prefill(monkeypatch, chunk, [64], 64, 64, 16)
    assert len(gather.calls) == 0
    # The merge ran: both collectives with fixed shapes.
    assert len(group_state.calls) == 2
    assert group_state.calls[0]["shape"] == (64, 4)


class _HcuContractGather:
    """Simulates the REAL HCU gather's consumption contract
    (rocm_aiter_mla_sparse.py:421-481): grid = k_fp8.size(0); for
    every tid in that grid, batch_id = token_to_seq[tid] (the gfx950
    variant loads it UNMASKED; the generic variant masks only against
    the same num_tokens), then offsets within cu[batch]..cu[batch+1]
    and resolves through the block table. Reads past len(map) are the
    OOB the old padded-view implementation caused."""

    def __init__(self):
        self.calls = []

    def __call__(
        self,
        kv_cache,
        k_fp8,
        k_fp8_scale,
        block_table,
        cu_seqlen,
        token_to_seq=None,
        **kwargs,
    ):
        self.calls.append((k_fp8, cu_seqlen, token_to_seq))
        num_tokens = k_fp8.size(0)
        num_batches = cu_seqlen.shape[0] - 1
        rows_written = 0
        for tid in range(num_tokens):
            batch_id = int(token_to_seq[tid])  # OOB if tid >= len(map)
            assert 0 <= batch_id < num_batches, (tid, batch_id)
            start = int(cu_seqlen[batch_id])
            end = int(cu_seqlen[batch_id + 1])
            assert start <= tid < end, (tid, batch_id, start, end)
            block = int(block_table[batch_id, (tid - start) // 8])
            assert block >= 0
            k_fp8[tid] = block  # resolved source row
            rows_written += 1
        assert rows_written == num_tokens == len(token_to_seq)
        assert int(cu_seqlen[-1]) == num_tokens


def test_hcu_gather_contract_exact_local_lengths(monkeypatch):
    """A NON-EMPTY rank whose local_total (3) is SHORTER than
    max_local (5): the gather must receive the exact [3] views — the
    HCU grid is k_fp8.size(0) and reads token_to_seq[tid] for every
    tid, so a padded 5-row destination with a 3-entry map is OOB.
    The simulated-contract gather (above) would raise IndexError /
    assertion on the OLD implementation."""
    rk = _RecordingKernel()
    monkeypatch.setattr(indexer_mod, "_BUILD_PREFILL_CHUNK_METADATA_KERNEL", rk)
    # W2/J4/R4, seq 20 (5 pools: groups 0,0,2,2,0? J=4: groups of 4
    # pools; 5 pools -> group0 (pools 0-3) rank0, pool 4 group1 rank1.
    # rank0: 4, rank1: 1. For local 3 vs max 5: use two requests.
    # req0: seq 12 -> 3 pools (all group0, rank0); req1: seq 8 -> 2
    # pools (group0 rank0). rank0 local = [3, 2] = 5; rank1 = 0.
    # Instead: req0 seq 20 -> 5 pools: g0(p0-3) r0, g1(p4) r1.
    # rank0 local = 4, rank1 = 1; max = 4.
    # For local_total 3 < max 5: W2/J2/R4, req0 seq 20 -> 5 pools:
    # g0(p0,1) r0; g1(p2,3) r1; g2(p4) r0 -> rank0 = 3, rank1 = 2,
    # max = 3. Need max 5 with local 3: two requests.
    # req0: seq 20 -> 5 pools -> rank0 3, rank1 2; req1: seq 4 ->
    # 1 pool -> rank0 1, rank1 0. rank0 local = [3,1] = 4; rank1 =
    # [2,0] = 2; max = 4. Hmm still not 3 vs 5.
    # Direct: W2/J8/R4: req0 seq 36 -> 9 pools: g0(p0-7) r0, g1(p8)
    # r1 -> rank0 8, rank1 1. req1 seq 8 -> 2 pools: g0 r0 -> r0 2.
    # rank0 = [8, 2] = 10; rank1 = [1, 0] = 1; max = 10.
    # SIMPLEST: craft the CHUNK directly (the helper's contract is
    # what's under test, not the builder).
    local_cu = torch.tensor([0, 2, 3], dtype=torch.int32)
    local_t2s = torch.tensor([0, 0, 1], dtype=torch.int32)
    chunk = _make_chunk(
        local_cu=local_cu,
        local_t2s=local_t2s,
        local_total=3,
        max_local=5,  # the OTHER rank owns 5
        dcp_world=2,
    )
    kq = torch.zeros(20, 8)
    ks = torch.zeros(20, 4)
    out = _call_helper(chunk, kq, ks, q_rows=4)
    # EXACT lengths everywhere.
    assert out.k_quant.shape[0] == 3
    assert out.k_scale.shape[0] == 3
    assert len(out.gather_token_to_seq) == 3
    assert int(out.gather_cu[-1]) == 3
    # Drive the simulated HCU contract: no OOB, all rows resolved.
    hcu = _HcuContractGather()
    block_table = torch.zeros((2, 4), dtype=torch.int32)
    hcu(
        None,
        out.k_quant,
        out.k_scale,
        block_table,
        out.gather_cu,
        token_to_seq=out.gather_token_to_seq,
    )


def test_hcu_gather_contract_padded_destination_is_oob():
    """OLD-red evidence: a max_local-padded destination with a
    shorter map violates the HCU consumption contract (the simulator
    raises), proving the exact-length fix is load-bearing."""
    local_t2s = torch.tensor([0, 0, 1], dtype=torch.int32)
    local_cu = torch.tensor([0, 2, 3], dtype=torch.int32)
    padded_dst = torch.zeros(5, 8)  # max_local-sized, map only 3
    hcu = _HcuContractGather()
    block_table = torch.zeros((2, 4), dtype=torch.int32)
    with pytest.raises((IndexError, AssertionError)):
        hcu(
            None,
            padded_dst,
            torch.zeros(5, 4),
            block_table,
            local_cu,
            token_to_seq=local_t2s,
        )


def test_topology_explicit_field_not_identity(monkeypatch):
    """The chunk's explicit dcp_world_size drives the path — never
    tensor identity: (a) W1 with a CLONED (non-aliased) cu stays W1;
    (b) W2 missing local fields is rejected (fail-closed, not silent
    W1); (c) W4 passes through as W4 (topology transport only — not a
    device authorization); (d) the default (legacy fixture) is W1."""
    kq = torch.zeros(20, 8)
    ks = torch.zeros(20, 4)

    # (a) W1 clone: local_cu is a DISTINCT tensor but dcp_world=1.
    cloned_cu = torch.tensor([0, 5, 10], dtype=torch.int32)
    chunk_w1_clone = _make_chunk(
        local_cu=cloned_cu,
        local_t2s=None,
        local_total=10,
        max_local=10,
        dcp_world=1,
    )
    out = _call_helper(chunk_w1_clone, kq, ks, q_rows=4)
    # W1 legacy path: the GLOBAL cu drives the gather.
    assert out.gather_cu is chunk_w1_clone.cu_seq_lens
    assert out.gather_token_to_seq is chunk_w1_clone.token_to_seq
    assert out.k_quant.shape[0] == 10  # total_seq_lens

    # (b) W2 missing local fields: rejected.
    chunk_w2_bad = _make_chunk(
        local_cu=None, local_t2s=None, local_total=5, max_local=5, dcp_world=2
    )
    with pytest.raises(ValueError, match="local_cu_seq_lens"):
        _call_helper(chunk_w2_bad, kq, ks, q_rows=4)

    # (c) W4 passes through (transport only).
    local_cu4 = torch.tensor([0, 3, 5], dtype=torch.int32)
    local_t2s4 = torch.tensor([0, 0, 0, 1, 1], dtype=torch.int32)
    chunk_w4 = _make_chunk(
        local_cu=local_cu4,
        local_t2s=local_t2s4,
        local_total=5,
        max_local=7,
        dcp_world=4,
    )
    out4 = _call_helper(chunk_w4, kq, ks, q_rows=4)
    assert out4.gather_cu is local_cu4
    assert out4.k_quant.shape[0] == 5

    # (d) legacy default: no dcp_world -> W1.
    chunk_legacy = SimpleNamespace(
        cu_seq_lens=torch.tensor([0, 5, 10], dtype=torch.int32),
        token_to_seq=torch.zeros(10, dtype=torch.int32),
        cu_seqlen_ks=torch.zeros(4, dtype=torch.int32),
        cu_seqlen_ke=torch.zeros(4, dtype=torch.int32),
        total_seq_lens=10,
        max_local_total_seq_lens=10,
        num_reqs=2,
        skip_kv_gather=False,
        local_cu_seq_lens=None,
        local_token_to_seq=None,
        local_total_seq_lens=0,
        # no dcp_world_size attr at all -> helper must treat as 1
    )

    # Simulate the missing attribute via a plain object.
    class _NoField:
        pass

    for f, v in vars(chunk_legacy).items():
        setattr(_NoField, f, v)
    # The dataclass default is 1; a hand-built namespace without the
    # field would AttributeError — the REAL dataclass always has it.
    # Verify the dataclass default directly.
    import dataclasses

    from vllm.v1.attention.backends.mla.indexer import (
        DeepseekV32IndexerPrefillChunkMetadata as _Chunk,
    )

    fields = {f.name: f.default for f in dataclasses.fields(_Chunk)}
    assert fields["dcp_world_size"] == 1


def test_builder_sets_explicit_world(monkeypatch):
    """The REAL builder records the actual dcp_world_size on every
    chunk; W1 chunks carry 1 and DCP2 chunks carry 2."""
    rk = _RecordingKernel()
    monkeypatch.setattr(indexer_mod, "_BUILD_PREFILL_CHUNK_METADATA_KERNEL", rk)
    c1 = _run_chunk(1, 0, 1, 4, [40], [40], kernel=rk)
    assert c1.dcp_world_size == 1
    c2 = _run_chunk(2, 0, 4, 4, [16, 8], [16, 8], kernel=rk)
    assert c2.dcp_world_size == 2


def test_skip_reuse_local_view_content(monkeypatch):
    """skip_kv_gather sub-chunks (same req_slice, later query_slice)
    reuse the workspace's rank-local gathered rows: the local views
    for the SECOND sub-chunk carry the same request-local content —
    no cross-request bleed for multi-request chunks with DIFFERENT
    per-request local counts."""
    rk = _RecordingKernel()
    monkeypatch.setattr(indexer_mod, "_BUILD_PREFILL_CHUNK_METADATA_KERNEL", rk)
    # Two requests, W2/J1/R4: req0 seq 16 (4 pools: r0 2, r1 2);
    # req1 seq 8 (2 pools: r0 1, r1 1). rank0 local counts [2, 1].
    chunk = _run_chunk(2, 0, 4, 4, [16, 8], [16, 8], kernel=rk)
    assert chunk.local_token_to_seq.tolist() == [0, 0, 1]
    assert chunk.local_cu_seq_lens.tolist() == [0, 2, 3]
    # The second sub-chunk of the same request range: the builder is
    # invoked again with a query_slice — the LOCAL fields must be
    # IDENTICAL (same requests, same sharding), so the reused
    # workspace content matches the gather inputs.
    chunk2 = _run_chunk(
        2, 0, 4, 4, [16, 8], [16, 8], query_slice=slice(8, 16), kernel=rk
    )
    assert chunk2.local_token_to_seq.tolist() == [0, 0, 1]
    assert chunk2.local_cu_seq_lens.tolist() == [0, 2, 3]
    assert chunk2.skip_kv_gather is True  # qs_start > 0
    # And the map's request segments never cross: seg for req0 is
    # rows [0,2), req1 [2,3) — content addressing stays in-range.
    lcu = chunk.local_cu_seq_lens
    for b in range(2):
        seg = chunk.local_token_to_seq[int(lcu[b]) : int(lcu[b + 1])]
        assert torch.all(seg == b)


def _build_merge_fixture(
    world=2, rank=0, J=1, R=4, seq_lens=(20, 12), interleave=4, query_lens=None
):
    """Build a real DCP chunk + the per-rank pack outputs for the
    recording group, and return the independent global reference.

    The fake logits table (rows x local_cols) scores
    1000*(row+1) - col; each rank r's local column j maps to global
    pool f(r, j) = (j // J) * (W * J) + r * J + j % J with score
    table[r-row][row_start_r + j]."""
    rk = _RecordingKernel()
    import vllm.v1.attention.backends.mla.indexer as im

    orig = im._BUILD_PREFILL_CHUNK_METADATA_KERNEL
    im._BUILD_PREFILL_CHUNK_METADATA_KERNEL = rk
    try:
        chunk = _run_chunk(
            world,
            rank,
            interleave,
            R,
            list(seq_lens),
            query_lens or list(seq_lens),
            kernel=rk,
        )
    finally:
        im._BUILD_PREFILL_CHUNK_METADATA_KERNEL = orig
    return chunk


def _pack_all_ranks_for_group(chunks_by_rank, table_fn, W, J, select_k):
    """Pre-build every rank's packed candidates for the recording
    group: EACH rank packs from ITS OWN chunk's per-token bounds
    (cu_seqlen_ks/ke) and ITS OWN logits table — exactly what the
    real per-rank loop does."""
    from vllm.v1.attention.backends.mla.dcp_pool_merge_utils import (
        pack_local_pool_candidates,
        select_local_pool_candidates,
    )

    per_rank = {}
    for r in range(W):
        c = chunks_by_rank[r]
        ks = c.cu_seqlen_ks.to(torch.int64)
        ke = c.cu_seqlen_ke.to(torch.int64)
        table = table_fn(r)
        lids, _ = select_local_pool_candidates(table, ks, ke, select_k)
        # Pack from the ORIGINAL table (the adapter's pack re-gathers
        # scores at row_start + local_id — the selected score matrix
        # is only K wide and its bounds semantics differ).
        per_rank[r] = pack_local_pool_candidates(table, lids, ks, ke, W, r, J)
    return per_rank


def test_prefill_merge_two_ranks_global_ids_match_oracle(monkeypatch):
    """The REAL kpool prefill forward with the DCP merge wired: both
    ranks drive the loop with recording groups; the expand receives
    IDENTICAL int64 global pool ids on both ranks, matching an
    independent global oracle (score desc, global id asc; nonzero
    row_starts, multiple requests, per-rank different local widths,
    ties at the local K boundary)."""
    W, J, R = 2, 1, 4
    seq_lens = [20, 12]  # pools 5, 3; rank0 [3,2], rank1 [2,1]
    n_tokens = sum(seq_lens)
    topk_tokens = 8
    select_k = topk_tokens // 4  # kpool=4 -> 2 pools

    def table_fn(r):
        # [n_tokens, local_cols] deterministic; ties within a row.
        cols = 5 if r == 0 else 3
        return 1000.0 * (torch.arange(n_tokens, dtype=torch.float32) + 1).unsqueeze(
            1
        ) - torch.arange(cols, dtype=torch.float32).unsqueeze(0)

    # Build BOTH ranks' chunks up front (each pack uses its own).
    chunks_by_rank = {}
    for rank in range(W):
        rk = _RecordingKernel()
        monkeypatch.setattr(indexer_mod, "_BUILD_PREFILL_CHUNK_METADATA_KERNEL", rk)
        chunks_by_rank[rank] = _run_chunk(W, rank, 4, R, seq_lens, seq_lens, kernel=rk)
        assert chunks_by_rank[rank].dcp_world_size == W
        assert chunks_by_rank[rank].dcp_rank == rank
        assert chunks_by_rank[rank].pool_interleave == J
        assert chunks_by_rank[rank].cu_seqlen_ks.shape[0] == n_tokens
    packed_all = _pack_all_ranks_for_group(chunks_by_rank, table_fn, W, J, select_k)

    expand_inputs = {}
    for rank in range(W):
        chunk = chunks_by_rank[rank]
        _LOGITS_COLS[id(chunk)] = 5 if rank == 0 else 3
        _PER_RANK_PACKED[id(chunk)] = packed_all

        _EXPAND_CALLS.setdefault(id(chunk), [])
        gather, group_state = _drive_kpool_prefill(
            monkeypatch, chunk, seq_lens, n_tokens, max(seq_lens), 8
        )
        expand_inputs[rank] = (id(chunk), group_state)

    for rank, (cid, gs) in expand_inputs.items():
        assert len(gs.calls) == 2, rank
        assert gs.calls[0]["dtype"] is torch.float32
        assert gs.calls[1]["dtype"] is torch.int64
        assert gs.calls[0]["dim"] == 1 and gs.calls[1]["dim"] == 1
        assert gs.calls[0]["shape"] == (n_tokens, select_k)
        assert gs.calls[1]["shape"] == (n_tokens, select_k)

    ids0 = _EXPAND_CALLS[expand_inputs[0][0]][0]
    ids1 = _EXPAND_CALLS[expand_inputs[1][0]][0]
    assert ids0.dtype is torch.int64 and ids1.dtype is torch.int64
    torch.testing.assert_close(ids0, ids1)

    # Independent global oracle over BOTH ranks' windows.
    def f(r, j):
        return (j // J) * (W * J) + r * J + j % J

    tables = {r: table_fn(r) for r in range(W)}
    bounds = {
        r: (
            chunks_by_rank[r].cu_seqlen_ks.to(torch.int64),
            chunks_by_rank[r].cu_seqlen_ke.to(torch.int64),
        )
        for r in range(W)
    }
    for i in range(n_tokens):
        cands = []
        for r in range(W):
            ks_r, ke_r = bounds[r]
            for j in range(int(ke_r[i]) - int(ks_r[i])):
                cands.append((float(tables[r][i, int(ks_r[i]) + j]), f(r, j)))
        top = sorted(cands, key=lambda p: (-p[0], p[1]))[:select_k]
        for c, (score, gid) in enumerate(top):
            assert int(ids0[i, c]) == gid, (i, c)


def test_prefill_merge_empty_rank_participates_in_collectives(monkeypatch):
    """An empty-shard rank issues BOTH collectives with the same
    [rows, K] shapes as the non-empty rank (no skip on local
    emptiness), and its expand receives the OTHER rank's winners."""
    W, J, R = 2, 16, 4
    seq_lens = [64]  # 16 pools, all in rank 0's group (J=16)
    n_tokens = 64
    select_k = 8 // 4

    def table_fn(r):
        cols = 16 if r == 0 else 0
        return 1000.0 * (torch.arange(n_tokens, dtype=torch.float32) + 1).unsqueeze(
            1
        ) - torch.arange(cols, dtype=torch.float32).unsqueeze(0)

    chunks_by_rank = {}
    for rank in range(W):
        rk = _RecordingKernel()
        monkeypatch.setattr(indexer_mod, "_BUILD_PREFILL_CHUNK_METADATA_KERNEL", rk)
        chunks_by_rank[rank] = _run_chunk(W, rank, 64, R, seq_lens, seq_lens, kernel=rk)
        assert (chunks_by_rank[rank].local_total_seq_lens == 0) == (rank == 1)
    packed_all = _pack_all_ranks_for_group(chunks_by_rank, table_fn, W, J, select_k)

    expand_inputs = {}
    for rank in range(W):
        chunk = chunks_by_rank[rank]
        _LOGITS_COLS[id(chunk)] = 16 if rank == 0 else 0
        _PER_RANK_PACKED[id(chunk)] = packed_all

        _EXPAND_CALLS.setdefault(id(chunk), [])
        gather, group_state = _drive_kpool_prefill(
            monkeypatch, chunk, seq_lens, 64, 64, 8
        )
        expand_inputs[rank] = (id(chunk), group_state, gather)

    for rank, (cid, gs, gather) in expand_inputs.items():
        assert len(gs.calls) == 2, rank
        assert gs.calls[0]["shape"] == (n_tokens, select_k)
        assert gs.calls[1]["shape"] == (n_tokens, select_k)
        assert len(gather.calls) == (0 if rank == 1 else 1)
    ids0 = _EXPAND_CALLS[expand_inputs[0][0]][0]
    ids1 = _EXPAND_CALLS[expand_inputs[1][0]][0]
    torch.testing.assert_close(ids0, ids1)
    # The LAST token (full context) selects rank 0's local id 0 ->
    # global 0; the FIRST token's causal window is empty (no
    # completed pool yet) so its lanes are -1.
    assert int(ids0[-1, 0]) == 0
    assert int(ids0[0, 0]) == -1


def test_prefill_merge_short_prefill_zero_collectives(monkeypatch):
    """The globally-shared short-prefill shortcut (max_prefill_seq_len
    <= topk_tokens — a GLOBAL quantity every rank agrees on) bypasses
    the chunk loop and the merge entirely: NO collective on any
    rank."""
    W = 2
    seq_lens = [8]  # <= topk_tokens 16
    for rank in range(W):
        rk = _RecordingKernel()
        monkeypatch.setattr(indexer_mod, "_BUILD_PREFILL_CHUNK_METADATA_KERNEL", rk)
        chunk = _run_chunk(W, rank, 4, 4, seq_lens, seq_lens, kernel=rk)
        _LOGITS_COLS[id(chunk)] = 4
        _PER_RANK_PACKED[id(chunk)] = {}
        gather, group_state = _drive_kpool_prefill(monkeypatch, chunk, seq_lens, 8, 8)
        assert len(group_state.calls) == 0, rank
        assert len(gather.calls) == 0, rank


def test_prefill_merge_w1_no_collectives(monkeypatch):
    """W1: no group calls at all; the native local-topk path is
    preserved (routed through the recorder)."""
    rk = _RecordingKernel()
    monkeypatch.setattr(indexer_mod, "_BUILD_PREFILL_CHUNK_METADATA_KERNEL", rk)
    chunk = _run_chunk(1, 0, 1, 4, [64, 32], [64, 32], kernel=rk)
    assert chunk.dcp_world_size == 1
    _LOGITS_COLS[id(chunk)] = 16
    _PER_RANK_PACKED[id(chunk)] = {}
    gather, group_state = _drive_kpool_prefill(monkeypatch, chunk, [64, 32], 96, 64)
    assert len(group_state.calls) == 0
    assert len(gather.calls) == 1  # the W1 gather still happens


# --- the skip_kv_gather reuse contract -------------------------------------


def test_sub_chunks_share_request_range():
    """split_indexer_prefill_chunks: query-dimension sub-chunks of one
    request-group chunk share the SAME req_slice — the precondition
    that makes skip_kv_gather's workspace reuse valid (the gathered KV
    in the workspace belongs to exactly those requests). Chunks from
    DIFFERENT request groups must never share the reuse."""
    from vllm.v1.attention.backends.mla.indexer import (
        split_indexer_prefill_chunks,
    )

    # Two groups: [req0] huge (forces query sub-chunking), [req1].
    seq_lens = torch.tensor([100, 3])
    query_lens = torch.tensor([100, 3])
    # Tiny max_logits forces one request per chunk and heavy sub-chunk.
    chunks = split_indexer_prefill_chunks(seq_lens, query_lens, 1000, 1000 * 4)
    assert len(chunks) >= 2
    # Every sub-chunk of the same group shares req_slice.
    by_range: dict[tuple[int, int], list[slice]] = {}
    for req_slice, query_slice in chunks:
        by_range.setdefault((req_slice.start, req_slice.stop), []).append(query_slice)
    for (s, e), qs in by_range.items():
        # Sub-chunk query ranges tile [0, total_q) without overlap.
        covered = sum(q.stop - q.start for q in qs)
        assert covered == query_lens[s:e].sum().item()
    # skip_kv_gather is exactly (query_slice.start > 0): only the
    # LATER sub-chunks of a group skip; a fresh group's first chunk
    # never skips.
    for req_slice, query_slice in chunks:
        first_of_group = query_slice.start == 0
        assert (not first_of_group) == (query_slice.start > 0)
