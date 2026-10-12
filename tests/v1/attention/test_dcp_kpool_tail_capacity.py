# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Synthetic CPU state-model tests for the KPool tail-ring capacity
contract (docs/design/kpool_tail_ring_capacity.md).

REAL flow: replays the AMD kernel's documented order (per-request
programs, position-order iteration, completion reads the pool
window's ring slots BEFORE the unconditional stash at pos % C),
instrumented to RECORD every compressed-read position/value for
divergence attribution.

REFERENCE: fully INDEPENDENT — a per-request accepted_history dict
mapping position -> (K, gate, token identity). A pool completion
computes the weighted compress DIRECTLY from the history's R
consecutive positions; it never uses a ring, never takes mod C, and
never calls the kernel step.

Timeline (scheduler.py:1898-1917 + the spec scheduling :735-747): a
verify step computes KV for its ANCHOR input and all DRAFT inputs
(V = D+1 positions); sampling emits the bonus (the next input token)
plus one output per accepted draft prefix; num_accepted counts
matched DRAFTS; the rollback leaves num_computed = anchor + 1 +
accepted — the committed prefix is THIS round's anchor input plus
the accepted drafts' positions (whose sampled outputs become the
next inputs, with KV computed NEXT step). The model's keep = 1 +
accepted therefore counts already-computed inputs only; the bonus
output's KV is never assumed computed in the same round.

Prefix: start >= 2*C + phase so max(0, start - C)..start - 1 is a
REAL non-negative history; every mod-R phase is covered.

DCP2: two ranks with INDEPENDENT rings; every verify token is
stashed on BOTH ranks (the kernel has no rank filter on the stash);
a completion's compressed write happens ONLY on the pool's owner
rank ((pool_index // J) % W == rank, with J in POOL units — the old
comment's "J=R" with owner p%2 corresponded to J=1, corrected
here). Each rank's owner-subset of completions is compared against
the reference's corresponding subset; the union over ranks is the
complete global set, exactly once.
"""

import itertools
import random

import torch

from vllm.models.glm5next.tail_capacity import kpool_tail_ring_capacity

R = 4  # logical pool size (index_kpool)
DIM = 8


def _tid(req, round_idx, pos, input_idx):
    """A REVERSIBLE unique integer encoding (no Python hash
    randomness, no collision risk): the request index, the round, the
    position, and the input index each get a disjoint bit range."""
    return (
        (req << 40)
        | ((round_idx & 0xFFFFF) << 20)
        | ((pos & 0x3FFFF) << 1)
        | (input_idx & 1)
    )


def _tvec(tid):
    g = torch.Generator().manual_seed(1000 + (tid % (2**31)))
    return torch.randn(DIM, generator=g)


def _compress_from_entries(entries):
    """The reference compress: weighted average of R (K, gate)
    entries — computed directly from the history, no ring, no mod."""
    sm = torch.stack([g for _, g in entries])
    km = torch.stack([k for k, _ in entries])
    mx = sm.max(dim=0).values
    probs = torch.exp(sm - mx)
    return (km * probs).sum(0) / probs.sum(dim=0)


def _kernel_step(ring_k, ring_gate, C, pos, k_vec, g_vec, read_log):
    """One REAL-flow token: completion read (R-window, C-indexed,
    BEFORE the stash), then the unconditional stash. Records each
    compressed-read (position, value) for attribution."""
    slot = pos % C
    logical_start = pos - (pos % R)
    compressed = None
    if pos % R == R - 1:
        entries = []
        for ps in range(R):
            x = logical_start + ps
            if x == pos:
                entries.append((k_vec, g_vec))
                read_log.append((x, "current", None))
            else:
                phys = x % C
                entries.append((ring_k[phys], ring_gate[phys]))
                read_log.append((x, "ring", phys))
        compressed = _compress_from_entries(entries)
    ring_k[slot] = k_vec
    ring_gate[slot] = g_vec
    return compressed


def _run_real_flow(history, start, V, accepts, C, num_rounds):
    """The real flow over the verify rounds. Returns
    (compressed: dict[pos -> tensor], read_log)."""
    ring_k = [torch.zeros(DIM) for _ in range(C)]
    ring_gate = [torch.zeros(DIM) for _ in range(C)]
    read_log: list[tuple[int, str, int | None]] = []
    # Seed the ring with the last C history tokens (real positions).
    for x in range(max(0, start - C), start):
        k_vec, g_vec = history[x][0], history[x][1]
        _kernel_step(ring_k, ring_gate, C, x, k_vec, g_vec, read_log)
    compressed = {}
    pos = start
    for rnd, acc in enumerate(accepts):
        # This round's inputs: the anchor at pos + drafts at
        # pos+1..pos+V-1 — ALL have KV computed this round.
        for j in range(V):
            p = pos + j
            tid = (p, rnd, j)
            k_vec, g_vec = _tvec(hash(tid) % 100000), _tvec((hash(tid) + 5000) % 100000)
            history[p] = (k_vec, g_vec, tid)
            c = _kernel_step(ring_k, ring_gate, C, p, k_vec, g_vec, read_log)
            if c is not None:
                compressed[p] = c.clone()
        # keep = this round's ANCHOR input (already computed) + the
        # accepted drafts. The bonus output's KV is computed NEXT
        # round — not counted here.
        pos += 1 + acc
    # Flush: fully-accepted fresh tokens complete every pending pool.
    for j in range(2 * R):
        p = pos + j
        tid = (p, -1, j)
        k_vec, g_vec = _tvec(hash(tid) % 100000), _tvec((hash(tid) + 5000) % 100000)
        history[p] = (k_vec, g_vec, tid)
        c = _kernel_step(ring_k, ring_gate, C, p, k_vec, g_vec, read_log)
        if c is not None:
            compressed[p] = c.clone()
    return compressed, read_log


def _run_reference(history, start, V, accepts, C, num_rounds):
    """The INDEPENDENT accepted-only reference: maintains ONLY the
    accepted history (already seeded); computes each completion
    directly from R consecutive history entries — no ring, no mod C,
    no kernel call."""
    compressed = {}
    pos = start
    for rnd, acc in enumerate(accepts):
        keep = 1 + acc
        for j in range(keep):
            p = pos + j
            tid = (p, rnd, j)
            k_vec, g_vec = _tvec(hash(tid) % 100000), _tvec((hash(tid) + 5000) % 100000)
            history[p] = (k_vec, g_vec, tid)
            if p % R == R - 1:
                entries = [
                    (history[p - R + 1 + ps][0], history[p - R + 1 + ps][1])
                    for ps in range(R)
                ]
                compressed[p] = _compress_from_entries(entries)
        pos += keep
    for j in range(2 * R):
        p = pos + j
        tid = (p, -1, j)
        k_vec, g_vec = _tvec(hash(tid) % 100000), _tvec((hash(tid) + 5000) % 100000)
        history[p] = (k_vec, g_vec, tid)
        if p % R == R - 1:
            entries = [
                (history[p - R + 1 + ps][0], history[p - R + 1 + ps][1])
                for ps in range(R)
            ]
            compressed[p] = _compress_from_entries(entries)
    return compressed


def _make_history(start):
    """A real prefix history: unique ids for positions
    max(0, start-C)..start-1 (requires start >= 2*C+phase so the
    window is fully non-negative)."""
    history = {}
    for x in range(max(0, start - 64), start):
        tid = (x, "prefix", 0)
        history[x] = (
            _tvec(hash(tid) % 100000),
            _tvec((hash(tid) + 5000) % 100000),
            tid,
        )
    return history


def _scenario(start, V, accepts, C, num_rounds=3):
    """Returns (diverged, detail) for one scenario."""
    history_real = _make_history(start)
    history_ref = {p: v for p, v in history_real.items()}
    real_comp, read_log = _run_real_flow(history_real, start, V, accepts, C, num_rounds)
    ref_comp = _run_reference(history_ref, start, V, accepts, C, num_rounds)
    for p, rc in ref_comp.items():
        if p not in real_comp:
            return True, f"reference completion at {p} missing in real flow"
        if not torch.allclose(real_comp[p], rc, atol=1e-5):
            return True, f"divergence at {p}"
    return False, "match"


def _enumerate(V, C, num_rounds=3):
    fails = total = 0
    first_detail = None
    for phase in range(R):
        start = 2 * C + phase  # a real non-negative prefix window
        for accepts in itertools.product(range(V), repeat=num_rounds):
            total += 1
            d, detail = _scenario(start, V, list(accepts), C, num_rounds)
            fails += d
            if d and first_detail is None:
                first_detail = (start, accepts, detail)
    return fails, total, first_detail


def test_capacity_helper_values():
    """C = R + S; no MTP keeps C = R; rejections on bad inputs."""
    assert kpool_tail_ring_capacity(4, 0) == 4
    assert kpool_tail_ring_capacity(4, 4) == 8  # R4 / MTP k4
    assert kpool_tail_ring_capacity(16, 0) == 16
    assert kpool_tail_ring_capacity(16, 3) == 19
    import pytest

    with pytest.raises(ValueError):
        kpool_tail_ring_capacity(0, 0)
    with pytest.raises(ValueError):
        kpool_tail_ring_capacity(4, -1)
    with pytest.raises(ValueError):
        kpool_tail_ring_capacity(True, 0)


def test_capacity_C_equals_R_old_counterexample_red():
    """RED: at C = R the canonical counterexample persists under the
    independent reference and the real prefix — with attribution of
    the polluted read."""
    C = R
    # start phase 0, V=3, round-1 accepts 1, round-2 rejects its
    # drafts (keep = anchor only), round-3 mixed.
    d, detail = _scenario(2 * C + 0, 3, [1, 0, 2], C)
    assert d, "the C=R counterexample must persist (red)"
    fails, total, detail = _enumerate(3, C)
    assert fails > 0, f"C=R must fail for V=3 ({fails}/{total})"
    assert "divergence" in detail[2] or "missing" in detail[2]


def test_capacity_contract_green_exhaustive():
    """GREEN: the contract C = R + S heals every enumerated scenario
    for V in 1..5 (real prefixes, all mod-R phases, all accept
    prefixes, 3 rounds)."""
    for V in range(1, 6):
        C = kpool_tail_ring_capacity(R, V - 1)
        fails, total, detail = _enumerate(V, C)
        assert fails == 0, (V, C, fails, total, detail)


def test_capacity_randomized_stress():
    """4-round randomized stress at the contract C: clean across
    phases, accept prefixes, and round counts."""
    random.seed(0)
    for V in range(1, 6):
        C = kpool_tail_ring_capacity(R, V - 1)
        for _ in range(120):
            phase = random.randrange(R)
            start = 2 * C + phase
            accepts = [random.randrange(V) for _ in range(4)]
            d, detail = _scenario(start, V, accepts, C, num_rounds=4)
            assert not d, (V, C, start, accepts, detail)


def test_capacity_below_contract_diverges():
    """TIGHTNESS EVIDENCE: one below the empirical minimum
    (R + V - 2) diverges for V >= 3; at the minimum it heals. The
    contract keeps one slot of margin over the minimum."""
    for V in (3, 4, 5):
        below = R + V - 3
        fails, total, _ = _enumerate(V, below)
        assert fails > 0, (V, below, fails, total)
        at_min = R + V - 2
        fails_min, _, _ = _enumerate(V, at_min)
        assert fails_min == 0, (V, at_min, fails_min)


def test_capacity_no_mtp_unchanged():
    """V <= 2 (no MTP / k=1): C = R suffices — unchanged behavior."""
    for V in (1, 2):
        fails, total, _ = _enumerate(V, R)
        assert fails == 0, (V, fails, total)


def test_capacity_dcp2_two_rank_owner_split():
    """DCP2 with pool-unit interleave J and owner
    (pool_index // J) % W: TWO RANKS with INDEPENDENTLY ALLOCATED
    rings (separate ring_k/ring_gate per rank per request), every
    verify input EXPLICITLY stashed on BOTH ranks at its own
    execution, and the compressed write GATED at write time by the
    owner condition (not filtered afterwards). Two requests share
    the same POSITION RANGE but use different token ids and
    different tail blocks; the two requests' steps are INTERLEAVED
    so any shared-ring contamination between requests or ranks is
    observable. Each rank's owner-subset of completions is compared
    to the reference's subset; the union is complete and unique."""
    W = 2
    J = 1  # pool units: owner = pool_index % 2 (J=1)
    V = 5
    C = kpool_tail_ring_capacity(R, V - 1)

    def owner(pos):
        return ((pos // R) // J) % W

    for phase in range(R):
        start = 2 * C + phase
        for accepts in ([0, 0, 0], [4, 4, 4], [1, 0, 2], [2, 3, 0]):
            # Two requests: the SAME position range but DIFFERENT
            # tail blocks (block ids 10 and 20) and disjoint token
            # id spaces.
            reqs = [0, 1]
            # Independent per-rank per-request rings.
            rings = {
                (rank, req): (
                    [torch.zeros(DIM) for _ in range(C)],
                    [torch.zeros(DIM) for _ in range(C)],
                )
                for rank in range(W)
                for req in reqs
            }
            # Independent accepted histories per request.
            histories = {}
            for req in reqs:
                h = {}
                for x in range(max(0, start - 64), start):
                    tid = _tid(req, 0, x, 0)
                    h[x] = (_tvec(tid), _tvec(tid + 1), tid)
                histories[req] = h
            # Per-rank owner-gated compressed outputs.
            rank_comp: dict[int, dict[int, torch.Tensor]] = {
                rank: {} for rank in range(W)
            }
            read_log: list[tuple] = []

            def kernel_step_rank(
                rank,
                req,
                pos,
                k_vec,
                g_vec,
                _rings=rings,
                _log=read_log,
                _comp=rank_comp,
            ):
                ring_k, ring_gate = _rings[(rank, req)]
                phys_slot = pos % C
                logical_start = pos - (pos % R)
                compressed = None
                if pos % R == R - 1:
                    entries = []
                    for ps in range(R):
                        x = logical_start + ps
                        if x == pos:
                            entries.append((k_vec, g_vec))
                            _log.append((rank, req, x, "current"))
                        else:
                            phys = x % C
                            entries.append((ring_k[phys], ring_gate[phys]))
                            _log.append((rank, req, x, "ring"))
                    sm = torch.stack([g for _, g in entries])
                    km = torch.stack([k for k, _ in entries])
                    mx = sm.max(dim=0).values
                    probs = torch.exp(sm - mx)
                    compressed = (km * probs).sum(0) / probs.sum(dim=0)
                ring_k[phys_slot] = k_vec
                ring_gate[phys_slot] = g_vec
                if compressed is not None and owner(pos) == rank:
                    # The owner gate at WRITE time.
                    _comp[rank][pos] = compressed.clone()

            # Seed each rank's ring with the request's last C
            # history tokens (a real prefill tail).
            for req in reqs:
                h = histories[req]
                for x in range(max(0, start - C), start):
                    k_vec, g_vec = h[x][0], h[x][1]
                    for rank in range(W):
                        ring_k, ring_gate = rings[(rank, req)]
                        ring_k[x % C] = k_vec
                        ring_gate[x % C] = g_vec

            # Interleave the two requests' rounds: process round r of
            # BOTH requests before round r+1 (a shared-ring bug would
            # cross-contaminate within a round).
            positions = {req: start for req in reqs}
            for rnd, acc in enumerate(accepts):
                for j in range(V):
                    for req in reqs:
                        p = positions[req] + j
                        tid = _tid(req, rnd + 1, p, j)
                        k_vec, g_vec = _tvec(tid), _tvec(tid + 1)
                        histories[req][p] = (k_vec, g_vec, tid)
                        for rank in range(W):
                            kernel_step_rank(rank, req, p, k_vec, g_vec)
                for req in reqs:
                    positions[req] += 1 + accepts[rnd]
            # Flush both requests.
            for req in reqs:
                for j in range(2 * R):
                    p = positions[req] + j
                    tid = _tid(req, 999, p, j)
                    k_vec, g_vec = _tvec(tid), _tvec(tid + 1)
                    histories[req][p] = (k_vec, g_vec, tid)
                    for rank in range(W):
                        kernel_step_rank(rank, req, p, k_vec, g_vec)

            # The independent reference per request.
            for req in reqs:
                h = {p: v for p, v in histories[req].items()}
                # Recompute the reference from accepted positions
                # only (replay with the accepted timeline).
                ref_comp = {}
                pos_r = start
                for rnd, acc in enumerate(accepts):
                    keep = 1 + acc
                    for j in range(keep):
                        p = pos_r + j
                        if p % R == R - 1:
                            entries = [
                                (h[p - R + 1 + ps][0], h[p - R + 1 + ps][1])
                                for ps in range(R)
                            ]
                            sm = torch.stack([g for _, g in entries])
                            km = torch.stack([k for k, _ in entries])
                            mx = sm.max(dim=0).values
                            probs = torch.exp(sm - mx)
                            ref_comp[p] = (km * probs).sum(0) / probs.sum(dim=0)
                    pos_r += keep
                for j in range(2 * R):
                    p = pos_r + j
                    if p % R == R - 1:
                        entries = [
                            (h[p - R + 1 + ps][0], h[p - R + 1 + ps][1])
                            for ps in range(R)
                        ]
                        sm = torch.stack([g for _, g in entries])
                        km = torch.stack([k for k, _ in entries])
                        mx = sm.max(dim=0).values
                        probs = torch.exp(sm - mx)
                        ref_comp[p] = (km * probs).sum(0) / probs.sum(dim=0)

                for rank in range(W):
                    ref_subset = {p: c for p, c in ref_comp.items() if owner(p) == rank}
                    real_subset = {p: c for p, c in rank_comp[rank].items()}
                    assert set(ref_subset) == set(real_subset), (
                        phase,
                        accepts,
                        req,
                        rank,
                    )
                    for p, rc in ref_subset.items():
                        assert torch.allclose(real_subset[p], rc, atol=1e-5), (
                            phase,
                            accepts,
                            req,
                            rank,
                            p,
                        )
                # The union over ranks equals the complete reference
                # for THIS request — each completion exactly once.
                union = set()
                for rank in range(W):
                    union |= set(rank_comp[rank])
                assert union == set(ref_comp)
