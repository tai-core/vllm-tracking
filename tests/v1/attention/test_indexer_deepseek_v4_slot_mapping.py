# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.backends.mla.compressor_utils import (
    get_compressed_slot_mapping,
)


def _import_builder_deps():
    """Lazily import the builder stack (its transitive deps need uvloop,
    which is absent on minimal CPU hosts; the pure slot-mapping reference
    below must stay collectable there)."""
    from tests.v1.attention.utils import create_vllm_config
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.attention.backends.mla.indexer import (
        BuildPrefillChunkMetadataKernel,
        DeepseekV32IndexerMetadataBuilder,
    )
    from vllm.v1.kv_cache_interface import MLAAttentionSpec
    from vllm.v1.worker.block_table import get_block_table_width

    return (
        create_vllm_config,
        CommonAttentionMetadata,
        BuildPrefillChunkMetadataKernel,
        DeepseekV32IndexerMetadataBuilder,
        MLAAttentionSpec,
        get_block_table_width,
    )


@pytest.mark.skipif(
    os.environ.get("VLLM_TEST_SKIP_BUILDER_DEPS") == "1",
    reason="builder-stack deps (uvloop etc.) unavailable on this host",
)
def test_indexer_warmup_normalizes_zero_compress_ratios():
    BuildPrefillChunkMetadataKernel = _import_builder_deps()[2]
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(compress_ratios=[0, 0, 4, 128, 0])
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            cp_kv_cache_interleave_size=1,
        ),
    )

    keys = BuildPrefillChunkMetadataKernel().get_warmup_keys(config)

    assert {key.COMPRESS_RATIO for key in keys} == {1, 4, 128}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_indexer_builder_deepseek_v4_compressed_slot_mapping_uses_num_states():
    """Regression test: DeepseekV4 compression path must compute slot_mapping from
    compressed positions, not reuse the uncompressed common metadata mapping.
    """
    device = torch.device("cuda")

    (
        create_vllm_config,
        CommonAttentionMetadata,
        _BuildKernel,
        DeepseekV32IndexerMetadataBuilder,
        MLAAttentionSpec,
        get_block_table_width,
    ) = _import_builder_deps()

    # num_states = block_size // tokens_per_state = 256 // 4 = 64
    kv_cache_spec = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        tokens_per_state=4,
    )
    vllm_config = create_vllm_config(max_model_len=1024)
    max_num_blocks = kv_cache_spec.max_num_blocks_per_req(vllm_config, 1024)
    block_table_width = get_block_table_width(max_num_blocks, kv_cache_spec.block_size)
    builder = DeepseekV32IndexerMetadataBuilder(
        kv_cache_spec=kv_cache_spec,
        layer_names=["dummy"],
        vllm_config=vllm_config,
        device=device,
        block_table_width=block_table_width,
    )

    # Construct a single request where:
    # - num_computed = 240 (=> compressed_pos_start = 60)
    # - query_len = 40 (=> num_groups = 10)
    # => compressed positions are 60..69 which cross the storage block boundary at 64.
    query_start_loc = torch.tensor([0, 40], dtype=torch.int32, device=device)
    query_start_loc_cpu = query_start_loc.cpu()
    seq_lens = torch.tensor([280], dtype=torch.int32, device=device)  # 240 + 40

    # Two blocks: compressed positions 0..63 map to block 5, 64..127 map to block 7.
    block_table_tensor = torch.tensor([[5, 7]], dtype=torch.int32, device=device)

    # Dummy uncompressed slot mapping (length == uncompressed num_actual_tokens).
    slot_mapping = torch.full((40,), -123, dtype=torch.int64, device=device)

    common = CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=seq_lens,
        seq_lens_cpu_upper_bound=seq_lens.cpu(),
        num_reqs=1,
        num_actual_tokens=40,
        max_query_len=40,
        max_seq_len=280,
        block_table_tensor=block_table_tensor,
        slot_mapping=slot_mapping,
        causal=True,
    )

    md = builder.build(common_prefix_len=0, common_attn_metadata=common)

    # The compressed slot_mapping retains the original uncompressed size (40).
    # Only every compress_ratio-th position gets a valid slot; the rest are -1.
    assert md.slot_mapping.numel() == 40
    valid_slots = md.slot_mapping[md.slot_mapping >= 0]
    assert valid_slots.numel() == 10  # 40 tokens / compress_ratio 4

    storage_bs = kv_cache_spec.num_states  # 64
    # Compressed positions 60..63 land in block 5, positions 64..69 in block 7.
    expected = torch.tensor(
        [
            5 * storage_bs + 60,
            5 * storage_bs + 61,
            5 * storage_bs + 62,
            5 * storage_bs + 63,
        ]
        + [
            7 * storage_bs + 0,
            7 * storage_bs + 1,
            7 * storage_bs + 2,
            7 * storage_bs + 3,
            7 * storage_bs + 4,
            7 * storage_bs + 5,
        ],
        dtype=torch.int64,
        device=device,
    )
    torch.testing.assert_close(valid_slots, expected)


def _oracle_slot_ids_by_enumeration(
    query_start_loc: list[int],
    seq_lens: list[int],
    block_table: list[list[int]],
    block_size: int,
    compress_ratio: int,
    dcp_world_size: int,
    dcp_rank: int,
    pool_interleave: int,
    num_tokens: int,
) -> list[int]:
    """Derive expected slot ids by ENUMERATING pools, not by mirroring the
    kernel's arithmetic.

    For each request: enumerate every global pool 0..ceil(seq_len/R)-1; a
    pool is OWNED by this rank iff its position in the interleaved group
    ordering selects this rank. Independently of any closed-form local-id
    formula, the rank's local pool sequence is the subsequence of owned
    pools in global order, and a pool's LOCAL id is its ORDINAL in that
    subsequence (counting owned pools from the sequence start, including
    pools completed before the query window). A query token gets the slot
    of its pool iff the token is the pool's LAST token (position % R ==
    R - 1) and the position is a real (non-padding, non-negative) row; the
    slot reads the rank's block table at local_id // block_size.
    """
    R = compress_ratio
    W = dcp_world_size
    J = pool_interleave
    result: list[int] = [-1] * num_tokens
    for req in range(len(seq_lens)):
        seq_len = seq_lens[req]
        # Enumerate global pools; build the rank's local id per pool.
        local_id_of_pool: dict[int, int] = {}
        for pool in range(-(-seq_len // R)):  # ceil
            # Group ordinal of this pool among ALL pools, then rank.
            owned = W == 1 or (pool // J) % W == dcp_rank
            if owned:
                local_id_of_pool[pool] = len(local_id_of_pool)
        q_start = query_start_loc[req]
        q_len = query_start_loc[req + 1] - q_start
        start_pos = seq_len - q_len
        for off in range(q_len):
            pos = start_pos + off
            slot = -1
            if pos >= 0 and pos % R == R - 1:
                pool = pos // R
                if pool in local_id_of_pool:
                    local_id = local_id_of_pool[pool]
                    blk = block_table[req][local_id // block_size]
                    slot = blk * block_size + local_id % block_size
            result[q_start + off] = slot
    return result


# --- Shared fixtures: the SAME cases run as the CPU oracle audit and as the
# CUDA kernel tests, so the CPU audit covers every native fixture too. ----


def _shared_cases():
    """(name, qsl, seq_lens, table_by_rank, block_size, R, W, J)

    table_by_rank: {rank: rows} so each rank resolves distinct physical
    blocks; the table width is chosen per-case to cover the maximum LOCAL
    pool id of any rank plus a safety margin.
    """
    cases = []

    # W1 legacy-shaped case with a non-zero start and a block crossing:
    # seq 40, query tokens 16..39 -> pools 4..9; block_size 4 -> local ids
    # 4..9 cross the block boundary at 8 (2 blocks needed + margin).
    cases.append(
        (
            "w1_crossing",
            [0, 24],
            [40],
            {0: [[10, 11, 12]]},
            4,
            4,
            1,
            1,
        )
    )

    # W2, J in {1, 2, 4}: 3 requests, non-zero starts, a partial tail,
    # remote pools, and (for the multi-request case) different lengths.
    # seq 32 (8 pools), 10 (2 pools + partial), 24 (6 pools).
    for J in (1, 2, 4):
        width = 8
        cases.append(
            (
                f"w2_j{J}_mixed",
                [0, 24, 34, 42],
                [32, 10, 24],
                {
                    0: [
                        [100 + i for i in range(width)],
                        [110 + i for i in range(width)],
                        [120 + i for i in range(width)],
                    ],
                    1: [
                        [200 + i for i in range(width)],
                        [210 + i for i in range(width)],
                        [220 + i for i in range(width)],
                    ],
                },
                4,
                4,
                2,
                J,
            )
        )

    # All-remote window for rank 1 with a genuinely tiny table: seq 804
    # (201 pools); W2 J201 -> (p // 201) % 2 == 0 for every pool 0..200,
    # so rank 0 owns ALL pools and rank 1 owns NONE (the one-column table
    # is capacity-safe for zero owned pools; rank 0 owns all 201 pools,
    # local ids 0..200 -> blocks 0..50 at block_size 4, hence its width
    # of 51). The query window [800, 804) completes pool 200 (rank 0's):
    # if the ownership mask were applied after the load, rank 1 would
    # compute local id (200 // (2*201))*201 + 200 % 201 = 200 -> block
    # 50, far beyond the one-column table — the masked load
    # (mask & is_valid, False for every lane) is what guarantees the row
    # is never read. The proof of correctness is the ORACLE agreement
    # plus the mask semantics, not merely "it did not crash".
    cases.append(
        (
            "w2_j201_all_remote_rank1",
            [0, 4],
            [804],
            {0: [[i for i in range(51)]], 1: [[999]]},
            4,
            4,
            2,
            201,
        )
    )

    # Negative-start (padded) request: seq_lens smaller than the query
    # window makes start_pos negative; every pos < 0 must be -1.
    cases.append(
        (
            "negative_start_padding",
            [0, 8],
            [4],
            {0: [[10, 11, 12]]},
            4,
            4,
            1,
            1,
        )
    )
    return cases


def _case_tables(case):
    _, _, _, table_by_rank, _, _, W, _ = case
    return table_by_rank


def _audit_case(case) -> None:
    """CPU capacity audit of one shared case for every participating rank.

    Checks (independent of any kernel run): every OWNED local id indexes
    within the rank's table width; the expected valid count matches the
    oracle; each completed global pool has exactly one owner across ranks
    (joint over all ranks); partial/padded positions are -1 for every rank;
    and (W>1) the two ranks' expected slot sets are disjoint given
    distinct physical block numbers.
    """
    name, qsl, seq_lens, table_by_rank, block_size, R, W, J = case
    num_tokens = qsl[-1]
    ranks = sorted(table_by_rank)
    per_rank = {
        r: _oracle_slot_ids_by_enumeration(
            qsl, seq_lens, table_by_rank[r], block_size, R, W, r, J, num_tokens
        )
        for r in ranks
    }
    # 1. Owned local ids stay in the table (width audit via the oracle's
    # own enumeration).
    for req, seq_len in enumerate(seq_lens):
        n_pools = -(-seq_len // R)
        for r in ranks:
            width = len(table_by_rank[r][req])
            owned = [p for p in range(n_pools) if W == 1 or (p // J) % W == r]
            max_local = (len(owned) - 1) if owned else 0
            assert max_local // block_size < width, (
                f"{name}: rank {r} req {req} needs local block "
                f"{max_local // block_size}, table width {width}"
            )
    # 2. Expected valid counts: for each request, the union over ranks of
    # valid slots equals the number of complete pools inside the query
    # window with pos >= 0.
    for req in range(len(seq_lens)):
        q_len = qsl[req + 1] - qsl[req]
        start_pos = seq_lens[req] - q_len
        complete = sum(
            1
            for off in range(q_len)
            if start_pos + off >= 0 and (start_pos + off) % R == R - 1
        )
        union_valid = 0
        for r in ranks:
            row = per_rank[r][qsl[req] : qsl[req + 1]]
            union_valid += sum(1 for s in row if s >= 0)
        assert union_valid == complete, (
            f"{name}: req {req} union valid {union_valid} != complete {complete}"
        )
        # 3. Exactly one owner per completed pool: the per-rank valid
        # counts partition `complete` (no double assignment).
        per_rank_valid = [
            sum(1 for s in per_rank[r][qsl[req] : qsl[req + 1]] if s >= 0)
            for r in ranks
        ]
        assert sum(per_rank_valid) == complete
    # 4. Partial/padded positions are -1 for every rank.
    for r in ranks:
        for i, s in enumerate(per_rank[r]):
            if s >= 0:
                # A valid slot must correspond to a completing position.
                req = max(j for j in range(len(qsl) - 1) if qsl[j] <= i)
                off = i - qsl[req]
                pos = seq_lens[req] - (qsl[req + 1] - qsl[req]) + off
                assert pos >= 0 and pos % R == R - 1, (
                    f"{name}: rank {r} token {i} got slot {s} at pos {pos}"
                )
    # 5. Cross-rank disjointness under distinct physical numbers.
    if W > 1:
        s0 = {s for s in per_rank[ranks[0]] if s >= 0}
        s1 = {s for s in per_rank[ranks[1]] if s >= 0}
        assert not (s0 & s1), f"{name}: overlapping slots {s0 & s1}"


def test_shared_case_cpu_audit():
    """Every shared fixture passes the independent CPU audit (no kernel)."""
    for case in _shared_cases():
        _audit_case(case)


def test_oracle_w2_joint_ownership():
    """Joint over ranks: each completed pool stored exactly once, and the
    slot's PHYSICAL block comes from the owning rank's own table.

    Uses the W2/J2 mixed case; verifies by GLOBAL TOKEN POSITION (not by
    cross-rank numeric disjointness alone — equal physical ids across
    devices would also be disjoint-looking).
    """
    case = next(c for c in _shared_cases() if c[0] == "w2_j2_mixed")
    name, qsl, seq_lens, table_by_rank, block_size, R, W, J = case
    num_tokens = qsl[-1]
    outs = {
        r: _oracle_slot_ids_by_enumeration(
            qsl, seq_lens, table_by_rank[r], block_size, R, W, r, J, num_tokens
        )
        for r in (0, 1)
    }
    for req, seq_len in enumerate(seq_lens):
        q_len = qsl[req + 1] - qsl[req]
        start_pos = seq_len - q_len
        for off in range(q_len):
            pos = start_pos + off
            i = qsl[req] + off
            if pos >= 0 and pos % R == R - 1:
                pool = pos // R
                owner = (pool // J) % W
                assigned = [r for r in (0, 1) if outs[r][i] >= 0]
                assert assigned == [owner], (
                    f"pool {pool} (pos {pos}) assigned to {assigned}, owner {owner}"
                )
                slot = outs[owner][i]
                # The slot's physical block must come from the owner's row.
                local_ordinal = sum(1 for p in range(pool) if (p // J) % W == owner)
                blk = table_by_rank[owner][req][local_ordinal // block_size]
                assert slot // block_size == blk


def test_oracle_w1_legacy_boundary_hand_computed():
    """The W1 crossing case against a HAND-COMPUTED expectation.

    seq 40, query tokens 16..39, R=4, block_size=4, table [10, 11, 12]:
    completing positions 19, 23, 27, 31, 35, 39 -> pools 4..9. Local ids
    4..7 sit in storage block 1 (table[1] = 11, offsets 0..3), ids 8..9
    in block 2 (table[2] = 12, offsets 0..1): slots 44..47 then 48..49,
    in token order, -1 elsewhere. (Hand arithmetic: pool p ->
    table[p // 4] * 4 + p % 4.)
    """
    case = next(c for c in _shared_cases() if c[0] == "w1_crossing")
    _, qsl, seq_lens, tables, block_size, R, W, J = case
    got = _oracle_slot_ids_by_enumeration(
        qsl, seq_lens, tables[0], block_size, R, W, 0, J, qsl[-1]
    )
    expected = [-1] * 24
    expected[3] = 11 * 4 + 0  # pos 19 -> pool 4  (table[1], offset 0)
    expected[7] = 11 * 4 + 1  # pos 23 -> pool 5
    expected[11] = 11 * 4 + 2  # pos 27 -> pool 6
    expected[15] = 11 * 4 + 3  # pos 31 -> pool 7
    expected[19] = 12 * 4 + 0  # pos 35 -> pool 8  (table[2], offset 0)
    expected[23] = 12 * 4 + 1  # pos 39 -> pool 9
    assert got == expected


# --- CUDA kernel tests (NOT_RUN without a GPU; same shared fixtures). -----


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("case", _shared_cases(), ids=[c[0] for c in _shared_cases()])
def test_compressed_slot_mapping_kernel_matches_oracle(case):
    """The kernel agrees with the enumeration oracle on every shared case,
    run once per participating rank (rank 0 for W1 cases, both for W2)."""
    name, qsl, seq_lens, table_by_rank, block_size, R, W, J = case
    device = torch.device("cuda")
    for rank, rows in sorted(table_by_rank.items()):
        qsl_t = torch.tensor(qsl, dtype=torch.int32, device=device)
        sls_t = torch.tensor(seq_lens, dtype=torch.int32, device=device)
        bt_t = torch.tensor(rows, dtype=torch.int32, device=device)
        out = get_compressed_slot_mapping(
            qsl[-1],
            qsl_t,
            sls_t,
            bt_t,
            block_size,
            R,
            dcp_world_size=W,
            dcp_rank=rank,
            pool_interleave=J,
        )
        expected = _oracle_slot_ids_by_enumeration(
            qsl, seq_lens, rows, block_size, R, W, rank, J, qsl[-1]
        )
        torch.testing.assert_close(out.tolist(), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "case",
    [c for c in _shared_cases() if c[6] == 2],
    ids=[c[0] for c in _shared_cases() if c[6] == 2],
)
def test_compressed_slot_mapping_kernel_joint_ownership(case):
    """Joint W2 kernel check by GLOBAL TOKEN POSITION: each completing
    token is assigned by exactly its owning rank (physical provenance from
    that rank's own table), matching the oracle's joint property."""
    name, qsl, seq_lens, table_by_rank, block_size, R, W, J = case
    device = torch.device("cuda")
    outs = {}
    for rank, rows in sorted(table_by_rank.items()):
        qsl_t = torch.tensor(qsl, dtype=torch.int32, device=device)
        sls_t = torch.tensor(seq_lens, dtype=torch.int32, device=device)
        bt_t = torch.tensor(rows, dtype=torch.int32, device=device)
        outs[rank] = (
            get_compressed_slot_mapping(
                qsl[-1],
                qsl_t,
                sls_t,
                bt_t,
                block_size,
                R,
                dcp_world_size=W,
                dcp_rank=rank,
                pool_interleave=J,
            )
            .cpu()
            .tolist(),
            rows,
        )
    for req, seq_len in enumerate(seq_lens):
        q_len = qsl[req + 1] - qsl[req]
        start_pos = seq_len - q_len
        for off in range(q_len):
            pos = start_pos + off
            i = qsl[req] + off
            if pos >= 0 and pos % R == R - 1:
                pool = pos // R
                owner = (pool // J) % W
                assigned = [r for r in outs if outs[r][0][i] >= 0]
                assert assigned == [owner]
                slot = outs[owner][0][i]
                local_ordinal = sum(1 for p in range(pool) if (p // J) % W == owner)
                blk = outs[owner][1][req][local_ordinal // block_size]
                assert slot // block_size == blk


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_compressed_slot_mapping_kernel_out_param_tail_cleared():
    """The `out` buffer is fully pre-cleared and the returned view shares
    its storage; entries beyond num_tokens and every non-completing
    position stay -1.

    NOTE: `res is out[:4]` is False in torch (each slice builds a new
    view object) — aliasing must be checked via storage identity
    (data_ptr / storage) or by observing a mutation through the other
    handle. The window is tokens 4..7 of an 8-token sequence: the only
    completing position is 7 (pos % 4 == 3) -> pool 1 (7 // 4), local
    id 1, block table [10, 11] -> block 0 -> table[0] = 10, offset 1
    -> slot 10*4 + 1 = 41.
    """
    device = torch.device("cuda")
    qsl = torch.tensor([0, 4], dtype=torch.int32, device=device)
    sls = torch.tensor([8], dtype=torch.int32, device=device)
    bt = torch.tensor([[10, 11]], dtype=torch.int32, device=device)
    out = torch.full((16,), -123, dtype=torch.int64, device=device)
    res = get_compressed_slot_mapping(4, qsl, sls, bt, 4, 4, out=out)
    # Storage alias: same underlying buffer, and mutations through `res`
    # are visible in `out` (a fresh `out[:4]` slice is a DIFFERENT python
    # object even when it aliases the same storage).
    assert res.data_ptr() == out.data_ptr()
    assert res.untyped_storage().data_ptr() == out.untyped_storage().data_ptr()
    assert res.shape == (4,)
    res[0] = 777
    assert out[0].item() == 777
    out[0] = -1
    # Beyond num_tokens: cleared by the wrapper's fill_(-1).
    assert (out[4:] == -1).all()
    # Non-completing positions inside the window: -1.
    assert out[0].item() == -1 and out[1].item() == -1 and out[2].item() == -1
    # pos 7 completes pool 1 -> local id 1 -> block 10, offset 1 -> 41.
    assert out[3].item() == 10 * 4 + 1


def test_compressed_slot_mapping_wrapper_contract_cpu(monkeypatch):
    """The wrapper's CONTRACT on CPU with a controlled fake launch.

    Only the module-level ``_compressed_slot_mapping_kernel`` symbol is
    monkeypatched (pytest-managed, restored after the test — no global
    pollution of the suite); the REAL wrapper code under test runs
    unmodified. The fake records the launch arguments and writes a
    sentinel into the slot_mapping buffer so every observable wrapper
    behavior is checked: the out-buffer pre-fill, the returned view's
    storage aliasing and length, parameter pass-through, validation
    BEFORE launch, and the out=None allocation path. The fake proves
    NOTHING about the Triton kernel itself (kernel execution stays
    NOT_RUN on this host).
    """
    import vllm.v1.attention.backends.mla.compressor_utils as cu

    launched = []

    class _FakeKernel:
        def __getitem__(self, grid):
            def _launch(*args, **kwargs):
                # Snapshot the slot_mapping AT LAUNCH TIME (before the
                # sentinel write) so the pre-clear is proven as seen by
                # the kernel, not only after the fact.
                launched.append((grid, args, kwargs, args[0].clone()))
                # Sentinel write into the slot_mapping tensor (arg 0).
                args[0].fill_(4242)

            return _launch

    monkeypatch.setattr(cu, "_compressed_slot_mapping_kernel", _FakeKernel())

    qsl = torch.tensor([0, 4], dtype=torch.int32)
    sls = torch.tensor([8], dtype=torch.int32)
    bt = torch.tensor([[10, 11]], dtype=torch.int32)

    # --- out buffer path: pre-fill, aliasing, pass-through ----------
    out = torch.full((16,), -123, dtype=torch.int64)
    res = cu.get_compressed_slot_mapping(
        4,
        qsl,
        sls,
        bt,
        4,
        4,
        out=out,
        dcp_world_size=2,
        dcp_rank=1,
        pool_interleave=2,
    )
    assert len(launched) == 1
    grid, args, kwargs, pre_launch = launched[0]
    assert grid == (1,)  # num_reqs from block_table.shape[0]
    # The WHOLE out buffer was already -1 when the kernel launched.
    assert (pre_launch == -1).all()
    # The wrapper pre-cleared `out` BEFORE the launch: the tail beyond
    # num_tokens still holds -1 (the original buffer was filled with
    # -123, so only the wrapper's fill_(-1) can account for it); the
    # [0, 4) window holds the sentinel the fake wrote at launch time.
    assert (out[4:] == -1).all()
    # ...then the fake's sentinel landed through the returned view.
    assert (res == 4242).all() and res.shape == (4,)
    assert res.data_ptr() == out.data_ptr()
    # Parameter pass-through to the launch (DCP scalars are positional).
    assert args[1] is qsl and args[2] is sls and args[3] is bt
    assert args[4] == bt.stride(0)
    assert args[5] == 4 and args[6] == 4  # block_size, compress_ratio
    assert args[7] == 2 and args[8] == 1 and args[9] == 2
    assert kwargs["PAD_ID"] == -1

    # --- out=None path: fresh allocation, correct dtype/device -------
    launched.clear()
    res2 = cu.get_compressed_slot_mapping(
        4,
        qsl,
        sls,
        bt,
        4,
        4,
        dcp_world_size=2,
        dcp_rank=0,
        pool_interleave=1,
    )
    assert len(launched) == 1
    assert res2.shape == (4,)
    assert res2.dtype == torch.int64
    assert res2.device == qsl.device
    assert (res2 == 4242).all()  # the fake wrote through the fresh buffer

    # --- validation rejects BEFORE any launch -----------------------
    launched.clear()
    for bad in (
        dict(dcp_world_size=0),
        dict(dcp_world_size=1, dcp_rank=1),
        dict(dcp_world_size=2, dcp_rank=2),
        dict(dcp_world_size=2, dcp_rank=0, pool_interleave=0),
        dict(dcp_world_size=True),
    ):
        with pytest.raises(ValueError):
            cu.get_compressed_slot_mapping(4, qsl, sls, bt, 4, 4, **bad)
    assert launched == []  # nothing launched on the rejected paths


def test_out_param_fixture_cpu_audit():
    """The out-param test's fixture passes the same CPU audit as the
    shared fixtures (it previously sat outside the shared list).

    Reuses the verified shared input shape: seq 8 / window 4 tokens /
    R 4 -> the only completing pool is pool 1, owned by rank 0 under
    W1; table [10, 11] -> slot 41 (the same arithmetic the CUDA test
    pins).
    """
    got = _oracle_slot_ids_by_enumeration([0, 4], [8], [[10, 11]], 4, 4, 1, 0, 1, 4)
    assert got == [-1, -1, -1, 41]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_compressed_slot_mapping_kernel_invalid_rank_rejects():
    """An out-of-range rank rejects at the wrapper (before any launch)."""
    device = torch.device("cuda")
    qsl = torch.tensor([0, 4], dtype=torch.int32, device=device)
    sls = torch.tensor([8], dtype=torch.int32, device=device)
    bt = torch.tensor([[0]], dtype=torch.int32, device=device)
    with pytest.raises(ValueError, match="dcp_rank"):
        get_compressed_slot_mapping(4, qsl, sls, bt, 4, 4, dcp_world_size=1, dcp_rank=1)
    with pytest.raises(ValueError, match="dcp_world_size"):
        get_compressed_slot_mapping(4, qsl, sls, bt, 4, 4, dcp_world_size=0)
    with pytest.raises(ValueError, match="pool_interleave"):
        get_compressed_slot_mapping(
            4,
            qsl,
            sls,
            bt,
            4,
            4,
            dcp_world_size=2,
            dcp_rank=0,
            pool_interleave=0,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_compressed_slot_mapping_kernel_w1_matches_legacy_formula():
    """W1 defaults vs the LEGACY formula computed independently here
    (slot = table[p // bs] * bs + p % bs over completing positions).

    The negative-pos guard only changes behavior for inputs the legacy
    path left to chance (padded rows with negative positions produced
    bogus block indices); on valid inputs (pos >= 0) the two agree, which
    is what this oracle pins. No byte-identical claim beyond that.
    """
    device = torch.device("cuda")
    qsl = [0, 24]
    seq_lens = [40]
    rows = [[10, 11, 12]]
    block_size, R = 4, 4
    qsl_t = torch.tensor(qsl, dtype=torch.int32, device=device)
    sls_t = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    bt_t = torch.tensor(rows, dtype=torch.int32, device=device)
    got = get_compressed_slot_mapping(
        qsl[-1], qsl_t, sls_t, bt_t, block_size, R
    ).tolist()
    legacy = [-1] * qsl[-1]
    q_len = qsl[1] - qsl[0]
    start_pos = seq_lens[0] - q_len
    for off in range(q_len):
        pos = start_pos + off
        if pos >= 0 and (pos + 1) % R == 0:
            p = pos // R
            legacy[off] = rows[0][p // block_size] * block_size + p % block_size
    assert got == legacy
