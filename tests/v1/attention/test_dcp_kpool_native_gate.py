# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NATIVE acceptance tests for the KPool tail-ring R/C split.

Calls the REAL AMD/HCU kernels — kpool_seed_tail_cache,
kpool_decode_update_and_maybe_write_cache_batched, and
get_compressed_slot_mapping — with NO recorder doubles. Runs ONLY on
an actual HCU/CUDA device (skipped elsewhere with a clear reason;
the module-level import performs NO GPU initialization). NOT RUN in
this session (no GPU): the executable package for the node-08 gate.

The oracle substrate lives in dcp_kpool_gate_oracle.py and is
independently verified by test_dcp_kpool_gate_selfcheck.py (CPU):
the butterfly Hadamard, the REAL fp8 dtype round-trip with the
platform dtype, the preshuffled page layout round-trip, the
per-element ULP-derived tolerance, and the request-separating token
vectors. The native assertions check ACTUAL VALUES.

Coverage (docs/design/kpool_tail_ring_capacity.md §6a): R=4 with C
in {4, 7, 8}. C=4 is the legacy layout — the ring-aliasing
counterexample is EXPECTED to reproduce and is REQUIRED to be
observed (a mismatch-free C=4 run REJECTS the gate: the attribution
would be unproven); C=8 (the contract) and C=7 (the primitive) MUST
match the oracle.
"""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dcp_kpool_gate_oracle import (  # noqa: E402
    BF16_SENTINEL,
    UINT8_SENTINEL,
    assert_import_root,
    expected_tolerance,
    fp8_max,
    hadamard128,
    read_compressed_page,
    token_vec,
)

# The gate requires BOTH the device AND the explicit env var.
_IS_NATIVE = (
    os.environ.get("VLLM_TEST_DCP_NATIVE", "0") == "1"
    and torch.accelerator.is_available()
    and torch.version.hip is not None
)

pytestmark = pytest.mark.skipif(
    not _IS_NATIVE,
    reason="requires an HCU (ROCm) device AND VLLM_TEST_DCP_NATIVE=1 "
    "(the node-08 native gate; not run in CPU-only sessions)",
)

R = 4
HEAD_DIM = 128


def _device():
    return torch.device("cuda")


def _sync_native():
    torch.accelerator.synchronize()


def _tid(req, round_idx, pos, input_idx):
    # A deterministic, request-separating identity (see the oracle).
    return (req, round_idx, pos, input_idx)


# --- the shared runner (ONE body; native and CPU both call it) --------------
#
# The runner takes a KERNEL DEPENDENCY INJECTION: a namespace with the
# three wrappers under test (seed / decode / slot). The native tests
# bind the REAL wrappers (imported at call time, inside the device
# gate, with inspect-verified signatures); the CPU self-check binds
# controlled source-contract simulations. The fixture construction,
# address allocation, oracle, and assertions are ONE copy — the CPU
# entry executes EXACTLY this body.


def _bind_native_kernels():
    """Import and bind the REAL wrappers (the device path). The
    signatures are verified with inspect so a drifted signature
    fails loudly before any launch."""
    import inspect

    import vllm.models.glm5next.amd.ops.kpool_compress as _amd
    import vllm.v1.attention.backends.mla.compressor_utils as _cu

    seed = _amd.kpool_seed_tail_cache
    decode = _amd.kpool_decode_update_and_maybe_write_cache_batched
    slot = _cu.get_compressed_slot_mapping
    # COMPLETE signature verification: bind each wrapper's FULL
    # parameter set exactly as the shared runners call it (plus the
    # runners' accepted defaults), so ANY drift — renamed, removed,
    # reordered, or newly-required parameters — fails before any
    # launch.
    inspect.signature(seed).bind(None, None, None, None, kpool=4, head_dim=128)
    inspect.signature(decode).bind(
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        pool_size=4,
        head_dim=128,
        round_scale=True,
    )
    inspect.signature(slot).bind(
        1,
        None,
        None,
        None,
        16,
        4,
        None,
        dcp_world_size=1,
        dcp_rank=0,
        pool_interleave=1,
    )
    return SimpleNamespace(seed=seed, decode=decode, slot=slot)


class _LaunchCounter:
    """Wraps each kernel, counting calls per name (for the exact
    launch-budget assertion)."""

    def __init__(self, kernels):
        self._kernels = kernels
        self.counts = {"seed": 0, "decode": 0, "slot": 0}

    def __getattr__(self, name):
        fn = getattr(self._kernels, name)

        def counted(*args, **kwargs):
            self.counts[name] += 1
            return fn(*args, **kwargs)

        return counted


# --- prefill seed -----------------------------------------------------------


def _run_seed(C, kernels, dev, sync):
    """The SEED acceptance body (shared by the native test and the CPU
    self-check — ONE copy). ``kernels`` carries the bound seed
    wrapper (the REAL one natively; a controlled simulation on CPU)."""
    num_blocks = 8
    tail = torch.full(
        (num_blocks, 2, C, HEAD_DIM), BF16_SENTINEL, dtype=torch.bfloat16, device=dev
    )

    req0_positions = list(range(0, 2 * C))  # a full C-wrap
    req1_positions = list(range(0, C - 1))
    n = len(req0_positions) + 1 + len(req1_positions)
    key = torch.zeros((n, HEAD_DIM), dtype=torch.bfloat16, device=dev)
    gate = torch.zeros((n, HEAD_DIM), dtype=torch.bfloat16, device=dev)
    tslot = torch.full((n,), -1, dtype=torch.int32, device=dev)
    expected = {}
    i = 0
    for pos in req0_positions:
        # Request-separating vectors (asserted distinct in the
        # self-check suite).
        key[i] = token_vec(0, 0, pos, 0).to(dev).to(torch.bfloat16)
        gate[i] = token_vec(0, 0, pos, 1).to(dev).to(torch.bfloat16)
        tslot[i] = 2 * C + pos % C
        if pos >= len(req0_positions) - C:  # the last-C window
            expected[(2, pos % C)] = i
        i += 1
    # The padding sentinel token.
    key[i] = 0
    gate[i] = 0
    tslot[i] = -1
    i += 1
    for pos in req1_positions:
        key[i] = token_vec(1, 0, pos, 0).to(dev).to(torch.bfloat16)
        gate[i] = token_vec(1, 0, pos, 1).to(dev).to(torch.bfloat16)
        tslot[i] = 5 * C + pos % C
        expected[(5, pos % C)] = i
        i += 1

    kernels.seed(tail, key, gate, tslot, kpool=R, head_dim=HEAD_DIM)
    sync()

    # The same-device comparison: expected rows gathered from the
    # same tensors the kernel read.
    for (blk, slot), row in expected.items():
        torch.testing.assert_close(tail[blk, 0, slot], key[row])
        torch.testing.assert_close(tail[blk, 1, slot], gate[row])
    # Every other entry stays the exactly-representable sentinel.
    mask = torch.ones_like(tail[..., 0], dtype=torch.bool)
    for blk, slot in expected:
        mask[blk, :, slot] = False
    assert torch.all(tail[mask] == BF16_SENTINEL), "sentinel overwritten out of window"


@pytest.mark.parametrize("C", [4, 7, 8])
def test_native_seed_tail_cache(C):
    """The REAL seed kernel (the shared body with the REAL wrapper
    bound; signatures inspect-verified)."""
    assert_import_root()
    _run_seed(C, _bind_native_kernels(), _device(), _sync_native)


# --- decode update ----------------------------------------------------------


def _decode_timeline(accepts, V, num_reqs, start):
    """The (req, pos) input sequence per round and the accepted-only
    reference positions — request-isolated histories."""
    positions = {r: start for r in range(num_reqs)}
    rounds = []
    for rnd, acc in enumerate(accepts):
        inputs = {r: [positions[r] + j for j in range(V)] for r in range(num_reqs)}
        keep = {r: [positions[r] + j for j in range(1 + acc)] for r in range(num_reqs)}
        rounds.append((inputs, keep))
        for r in range(num_reqs):
            positions[r] += 1 + acc
    flush = {r: [positions[r] + j for j in range(2 * R)] for r in range(num_reqs)}
    return rounds, flush


def _run_rank(C, accepts, V, num_reqs, start, rank, W, J, ape, kernels, dev, sync):
    """Run ONE rank's decode kernel over the shared timeline (the
    wrapper bound via ``kernels`` — REAL natively, simulated on CPU).
    Every token is stashed on this rank; the compressed write is
    owner-gated at call time with a UNIQUE physical location per
    (request, observed pool) — recorded immediately when written."""
    rounds, flush = _decode_timeline(accepts, V, num_reqs, start)

    # Per-rank per-request device buffers with distinct blocks.
    blocks = {0: 0, 1: 3}
    num_blocks = 8
    tail = torch.full(
        (num_blocks, 2, C, HEAD_DIM), BF16_SENTINEL, dtype=torch.bfloat16, device=dev
    )
    PAGE = 16  # tokens per kv page (the preshuffle tile basis)
    kv = torch.full(
        (num_blocks, PAGE, HEAD_DIM + 4),
        UINT8_SENTINEL,
        dtype=torch.uint8,
        device=dev,
    )
    # Per-request accepted histories (request-isolated; only ACCEPTED
    # inputs are recorded — never drafts or flush reruns overwriting
    # a reference). The reference vectors are CPU bf16 (the oracle
    # computes entirely on CPU; only the ACTUAL results are copied
    # back for comparison).
    histories: dict[int, dict[int, tuple[torch.Tensor, torch.Tensor]]] = {
        r: {} for r in range(num_reqs)
    }
    # Seed each request's prefix (the last C positions).
    for r in range(num_reqs):
        for pos in range(start - C, start):
            k = token_vec(r, 0, pos, 0).to(torch.bfloat16)
            g = token_vec(r, 0, pos, 1).to(torch.bfloat16)
            histories[r][pos] = (k, g)
            tail[blocks[r], 0, pos % C] = k.to(dev)
            tail[blocks[r], 1, pos % C] = g.to(dev)

    def owner(pos):
        return ((pos // R) // J) % W

    # Unique compressed locations: (req, pool) -> (page, token_off),
    # allocated incrementally, never reused; recorded at write time.
    compressed_loc = {}
    next_page = [0]

    def alloc_loc(req, pool):
        key = (req, pool)
        if key not in compressed_loc:
            page = next_page[0] // PAGE
            tok = next_page[0] % PAGE
            compressed_loc[key] = (page, tok)
            next_page[0] += 1
        return compressed_loc[key]

    observed = {}  # (req, pool) -> (page, tok) for THIS rank's writes
    # The WRITE SETS for the sentinel verification: every tail slot
    # the kernel may legitimately touch (ALL processed positions %C
    # — rejected drafts included; the tail is last-writer), and
    # every compressed location actually written (sm >= 0).
    tail_write_set = set()
    compressed_write_set = set()

    def run_round(inputs, keep=None):
        """Launch one round; when ``keep`` is given, record ONLY the
        accepted inputs into the (request-isolated) histories —
        captured from the exact vectors the kernel consumed."""
        B = num_reqs
        key = torch.zeros((B, V, HEAD_DIM), dtype=torch.bfloat16, device=dev)
        score = torch.zeros_like(key)
        tsm = torch.full((B, V), -1, dtype=torch.int32, device=dev)
        sm = torch.full((B, V), -1, dtype=torch.int32, device=dev)
        pos_t = torch.zeros((B, V), dtype=torch.int32, device=dev)
        for req in range(B):
            for j, p in enumerate(inputs[req]):
                k_cpu = token_vec(req, rnd, p, j).to(torch.bfloat16)
                g_cpu = token_vec(req, rnd, p, j + 500).to(torch.bfloat16)
                key[req, j] = k_cpu.to(dev)
                score[req, j] = g_cpu.to(dev)
                tsm[req, j] = blocks[req] * C + p % C
                pos_t[req, j] = p
                tail_write_set.add((blocks[req], p % C))
                if keep is not None and p in keep[req]:
                    histories[req][p] = (k_cpu.clone(), g_cpu.clone())
                if owner(p) == rank and p % R == R - 1:
                    page, tok = alloc_loc(req, p // R)
                    sm[req, j] = page * PAGE + tok
                    observed[(req, p // R)] = (page, tok)
                    compressed_write_set.add((page, tok))
        kernels.decode(
            kv,
            tail,
            tsm,
            key,
            score,
            ape,
            sm,
            pos_t,
            pool_size=R,
            head_dim=HEAD_DIM,
        )
        sync()

    rnd = 0
    for inputs, keep in rounds:
        run_round(inputs, keep)
        rnd += 1
    # The flush round: single-token calls (all accepted).
    for req in range(num_reqs):
        for p in flush[req]:
            k_cpu = token_vec(req, 999, p, 0).to(torch.bfloat16)
            g_cpu = token_vec(req, 999, p, 1).to(torch.bfloat16)
            histories[req][p] = (k_cpu.clone(), g_cpu.clone())
            k = k_cpu.to(dev)
            g = g_cpu.to(dev)
            key1 = torch.zeros((1, 1, HEAD_DIM), dtype=torch.bfloat16, device=dev)
            score1 = torch.zeros_like(key1)
            tsm1 = torch.full((1, 1), -1, dtype=torch.int32, device=dev)
            sm1 = torch.full((1, 1), -1, dtype=torch.int32, device=dev)
            pos1 = torch.zeros((1, 1), dtype=torch.int32, device=dev)
            key1[0, 0] = k
            score1[0, 0] = g
            tsm1[0, 0] = blocks[req] * C + p % C
            pos1[0, 0] = p
            tail_write_set.add((blocks[req], p % C))
            if owner(p) == rank and p % R == R - 1:
                page, tok = alloc_loc(req, p // R)
                sm1[0, 0] = page * PAGE + tok
                observed[(req, p // R)] = (page, tok)
                compressed_write_set.add((page, tok))
            kernels.decode(
                kv,
                tail,
                tsm1,
                key1,
                score1,
                ape,
                sm1,
                pos1,
                pool_size=R,
                head_dim=HEAD_DIM,
            )
            sync()
    return (
        tail,
        kv,
        histories,
        observed,
        PAGE,
        tail_write_set,
        compressed_write_set,
    )


def _expected_rounded_scale(prequant: torch.Tensor) -> float:
    """The kernel's theoretical scale for a pre-quantization vector:
    exp2(ceil(log2(absmax / fp8_max))) with the 1e-4 floor. The
    float32-weighted reduction + bf16 rounding in the kernel can sit
    on the other side of an exp2 boundary from the float64 oracle,
    so ONE binade of slack is allowed and documented (never a
    blanket tolerance)."""
    import math

    absmax = float(prequant.abs().amax())
    absmax = max(absmax, 1e-4)
    return math.ldexp(1.0, math.ceil(math.log2(absmax / fp8_max())))


def _verify_tail_write_set(tail, tail_write_set, C):
    """Every tail element OUTSIDE the legitimate write set (the
    processed positions' %C slots on each request's OWN block —
    drafts included, the tail is last-writer) must still carry the
    BF16 sentinel."""
    mask = torch.ones_like(tail[..., 0], dtype=torch.bool)
    for blk, slot in tail_write_set:
        mask[blk, :, slot] = False
    assert torch.all(tail[mask] == BF16_SENTINEL), (
        "tail sentinel overwritten outside the legitimate write set "
        f"(write set size {len(tail_write_set)}, C={C})"
    )


def _verify_compressed_write_set(kv, compressed_write_set, page):
    """Every kv byte OUTSIDE the legitimate compressed write set
    (the preshuffled K bytes + the 4 scale bytes at each sm>=0
    location) must still carry the UINT8 sentinel — a non-owner or
    out-of-bounds write fails here."""
    for pg in range(kv.shape[0]):
        page_t = kv[pg]
        written_toks = {tok for (p, tok) in compressed_write_set if p == pg}
        if not written_toks:
            assert torch.all(page_t == UINT8_SENTINEL), (
                f"kv page {pg} fully sentinel-expected but modified"
            )
            continue
        # The allowed byte set: each written token's preshuffled K
        # offsets + its 4 scale bytes.
        offs = torch.arange(HEAD_DIM)
        allowed: set[int] = set()
        for tok in written_toks:
            allowed.update(int(o) for o in _page_k_offsets(tok, offs, HEAD_DIM))
            s_byte = page * HEAD_DIM + tok * 4
            allowed.update(range(s_byte, s_byte + 4))
        flat = page_t.reshape(-1)
        sentinel_positions = torch.ones_like(flat, dtype=torch.bool)
        sentinel_positions[list(allowed)] = False
        assert torch.all(flat[sentinel_positions] == UINT8_SENTINEL), (
            f"kv page {pg} sentinel overwritten outside the legitimate "
            f"write set (tokens {sorted(written_toks)})"
        )


def _page_k_offsets(token_off, dim_off, head_dim):
    """The preshuffled K offsets (the oracle's k_offset_in_page,
    re-exposed for the write-set verification)."""
    from dcp_kpool_gate_oracle import k_offset_in_page

    return k_offset_in_page(token_off, dim_off, head_dim)


def _run_decode(C, kernels, dev, sync):
    """The DECODE acceptance body (shared: the native test and the
    CPU self-check call the SAME body). Returns the mismatch list.
    C=8/C=7 must be empty; C=4 must be NON-empty (the required
    observed ring-aliasing counterexample)."""
    V = 5
    accepts = [1, 0, 2]
    num_reqs = 2
    W, J = 2, 1
    start = 2 * C
    ape = (
        torch.randn(R, HEAD_DIM, generator=torch.Generator().manual_seed(7))
        .to(torch.float32)
        .to(dev)
    )
    results = {}
    oracles = {}
    for rank in range(W):
        (
            tail,
            kv,
            histories,
            observed,
            page,
            tail_write_set,
            compressed_write_set,
        ) = _run_rank(
            C, accepts, V, num_reqs, start, rank, W, J, ape, kernels, dev, sync
        )
        results[rank] = (
            tail,
            kv,
            observed,
            page,
            tail_write_set,
            compressed_write_set,
        )
        # The oracle: each OBSERVED (req, pool) compressed from the
        # accepted history's R consecutive entries.
        for (req, pool), (pg, tok) in observed.items():
            p = pool * R + R - 1
            entries = []
            for ps in range(R):
                x = p - R + 1 + ps
                k, g = histories[req][x]
                entries.append((k.double(), g.double()))
            K = torch.stack([e[0] for e in entries])
            G = torch.stack([e[1] for e in entries])
            A = ape.cpu().double()
            scores = G + A
            mx = scores.max(dim=0).values
            probs = torch.exp(scores - mx)
            avg = (K * probs).sum(0) / probs.sum(dim=0)
            # The kernel's order: bf16 round the Hadamard INPUT, then
            # Hadamard, then bf16, then quantize.
            h = hadamard128(avg.to(torch.bfloat16).double())
            h_bf16 = h.to(torch.bfloat16).double()
            oracles[(rank, req, pool)] = h_bf16

    mismatches = []
    for rank in range(W):
        (
            tail,
            kv,
            observed,
            page,
            tail_write_set,
            compressed_write_set,
        ) = results[rank]
        for (req, pool), (pg, tok) in sorted(observed.items()):
            # The ACTUAL result: read from the device page (copied
            # to CPU for the comparison — small fixed-size reads).
            page_cpu = kv[pg].cpu()
            fp8_vals, scale = read_compressed_page(page_cpu, tok, HEAD_DIM, page)
            dequant = fp8_vals.double() * scale
            # The SCALE contract (round_scale=True): finite, positive,
            # and a power of two; and the theoretical expected scale
            # for this pool's pre-quantization magnitude.
            import math

            m, e = math.frexp(scale)
            assert scale == scale and scale > 0.0 and math.isfinite(scale), (
                f"corrupt scale {scale!r} at rank {rank} pool {pool}"
            )
            assert m == 0.5, (
                f"non-power-of-two scale {scale!r} (frexp mantissa {m}) "
                f"at rank {rank} pool {pool}"
            )
            exp_scale = _expected_rounded_scale(oracles[(rank, req, pool)])
            assert scale in (
                exp_scale,
                exp_scale / 2,
                exp_scale * 2,
            ), (
                f"scale {scale!r} not the expected rounded scale "
                f"{exp_scale!r} (or one binade from the float32/"
                f"bf16-vs-float64 exp2 boundary) at rank {rank} pool {pool}"
            )
            # The EXPECTED value: an all-CPU oracle computation.
            exp = oracles[(rank, req, pool)]
            tol = expected_tolerance(exp)
            if not torch.all((dequant - exp).abs() <= tol + 1e-15):
                mismatches.append((rank, req, pool))
        # The WRITE-SET verification: every tail/kv element OUTSIDE
        # the legitimate write set must still carry its sentinel.
        _verify_tail_write_set(tail, tail_write_set, C)
        _verify_compressed_write_set(kv, compressed_write_set, page)
    return mismatches


@pytest.mark.parametrize("C", [8, 7, 4])
def test_native_decode_reject_rerun(C):
    """The REAL decode kernel over a reject-then-rerun timeline (the
    shared body with the REAL wrapper bound).

    C=8 (the contract capacity) and C=7 (the primitive) MUST match
    the accepted-history oracle at the ULP-derived tolerance. C=4
    (the legacy layout) is REQUIRED to show the observed mismatch
    (the ring-aliasing counterexample): a clean C=4 run REJECTS the
    gate — the attribution would be unproven — and a mismatching
    C=8 also rejects. No xfail, no unconditional pass."""
    assert_import_root()
    mismatches = _run_decode(C, _bind_native_kernels(), _device(), _sync_native)
    if C in (8, 7):
        assert not mismatches, (
            f"C={C}: {len(mismatches)} pool(s) diverged: {mismatches}"
        )
    else:
        # C=4 legacy: the ring-aliasing counterexample MUST be
        # observed natively — a clean run REJECTS the gate.
        assert mismatches, (
            "C=4 legacy run showed NO ring-aliasing divergence — the "
            "counterexample attribution is unproven on this device; "
            "the gate cannot accept this result"
        )


# --- the compressed slot mapping (native) ------------------------------------


def _run_slot(rank, kernels, dev, sync):
    """The SLOT acceptance body (shared; the CORRECT rank-local
    ordinal oracle). Fixture: qsl=[0,16], seq 64 -> the CONTINUOUS
    positions 48..63; only p%R==3 completes (51/55/59/63 -> pools
    12..15). W2/J1: rank 0 owns pools {12,14}, rank 1 {13,15}. The
    RANK-LOCAL ORDINAL is the pool's index within the rank's OWNED
    subsequence (12->6, 14->7 on rank 0; 13->6, 15->7 on rank 1) —
    computed by the EXISTING independent enumeration oracle
    (test_indexer_deepseek_v4_slot_mapping.
    _oracle_slot_ids_by_enumeration), not a hand formula. The block
    table [[0]] with block_size 16 gives the expected slots."""
    from test_indexer_deepseek_v4_slot_mapping import (
        _oracle_slot_ids_by_enumeration,
    )

    W, J = 2, 1
    num_tokens = 16
    seq_len = 64
    qsl = torch.tensor([0, num_tokens], dtype=torch.int32, device=dev)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=dev)
    block_table = torch.tensor([[0]], dtype=torch.int32, device=dev)
    out = torch.full((num_tokens,), -1, dtype=torch.int64, device=dev)
    res = kernels.slot(
        num_tokens,
        qsl,
        seq_lens,
        block_table,
        block_size=16,
        compress_ratio=R,
        out=out,
        dcp_world_size=W,
        dcp_rank=rank,
        pool_interleave=J,
    )
    sync()
    # The INDEPENDENT enumeration oracle (the same one the slot
    # primitive's own suite uses — never a formula mirroring the
    # kernel).
    expected = _oracle_slot_ids_by_enumeration(
        [0, num_tokens], [seq_len], [[0]], 16, R, W, rank, J, num_tokens
    )
    for t in range(num_tokens):
        assert int(res[t]) == expected[t], (rank, t, int(res[t]), expected[t])
    # The exact hand-pinned values (rank-local ordinals).
    if rank == 0:
        assert expected == [
            -1,
            -1,
            -1,
            6,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            7,
            -1,
            -1,
            -1,
            -1,
        ]
    else:
        assert expected == [
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            6,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            7,
        ]
    return expected


@pytest.mark.parametrize("rank", [0, 1])
def test_native_get_compressed_slot_mapping(rank):
    """The REAL get_compressed_slot_mapping under DCP2/J=1 on device
    for EACH rank, against the independent enumeration oracle (the
    same oracle the slot primitive's own suite uses) and the exact
    hand-pinned rank-local ordinals."""
    assert_import_root()
    _run_slot(rank, _bind_native_kernels(), _device(), _sync_native)
