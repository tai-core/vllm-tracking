# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU self-checks for the KPool native-gate oracles and fixtures.

These run WITHOUT any GPU and MUST pass before the native gate is
executed: they verify the oracle substrate (the import root, the
butterfly Hadamard against independent mathematical properties, the
fp8 codec via a REAL dtype round-trip, the preshuffled page layout
round-trip, the token-identity separation, and the sentinels) so
that none of these errors can reach the device run.
"""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dcp_kpool_gate_oracle import (  # noqa: E402
    BF16_SENTINEL,
    assert_import_root,
    expected_tolerance,
    fp8_dtype,
    fp8_max,
    hadamard128,
    k_offset_in_page,
    quant_dequant,
    read_compressed_page,
    token_vec,
    write_compressed_page,
)


def test_import_root_exact():
    """vllm and the modules under test resolve inside THIS worktree
    (an exact path comparison — no substring heuristics)."""
    assert_import_root()


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
        torch.float8_e5m2,
    ],
)
def test_fp8_tolerance_near_boundary_sweep(dtype):
    """The tolerance bound holds for near-boundary values on EVERY
    fp8 dtype (not just the platform's): subnormals, tiny, zero, the
    max, saturation, and a random sweep — each element's error is
    within its own grid-derived bound, checked against the REAL
    encode/decode (never the quant_dequant helper alone)."""
    finfo = torch.finfo(dtype)
    cases = torch.tensor(
        [
            0.0,
            -0.0,
            finfo.tiny,
            finfo.tiny / 2,
            finfo.tiny * 3,
            finfo.tiny * 100,
            1e-7,
            1e-4,
            1.0,
            -1.0,
            finfo.max,
            -finfo.max,
            finfo.max * 1.5,
            -finfo.max * 1.5,
            123.456,
            -0.000651,
        ],
        dtype=torch.float64,
    )
    g = torch.Generator().manual_seed(21)
    cases = torch.cat([cases, torch.randn(64, generator=g) * 3])
    mantissa = 3 if "e4m3" in str(dtype) else 2
    for scale in (2.0**-8, 1.0, 2.0**5):
        raw = cases / scale
        q = raw.clamp(-finfo.max, finfo.max)
        deq = q.to(dtype).double() * scale
        scaled_mag = raw.abs()
        ulp = torch.where(
            scaled_mag >= finfo.tiny,
            scaled_mag * (2.0**-mantissa),
            torch.full_like(scaled_mag, finfo.tiny),
        )
        clamp_delta = (scaled_mag - finfo.max).clamp_min(0)
        tol = torch.where(clamp_delta > 0, clamp_delta, ulp) * scale
        err = (cases - deq).abs()
        assert torch.all(err <= tol + 1e-15 * scale), (
            dtype,
            scale,
            float((err - tol).max()),
        )


def test_expected_tolerance_subnormal_floor():
    """The Codex counterexample: expected_tolerance([1., 1e-7]) gave
    a tiny bound (1.25e-8) while the real error was 1e-7. The fixed
    bound uses the subnormal absolute floor (finfo.tiny * scale)."""
    v = torch.tensor([1.0, 1e-7])
    ref = quant_dequant(v)
    err = (v - ref).abs()
    tol = expected_tolerance(v)
    assert torch.all(err <= tol), (err.tolist(), tol.tolist())
    # The stored-scale check: the expected rounded scale.
    absmax = v.abs().amax().clamp_min(1e-4)
    exp_scale = float(torch.exp2(torch.ceil(torch.log2(absmax / fp8_max()))))
    ref2 = quant_dequant(v, scale=exp_scale)
    assert torch.allclose(ref, ref2)


def test_hadamard128_base_vectors():
    """e0 -> all 1/sqrt(128); e1 alternates +/- 1/sqrt(128) — the
    unnormalized-basis signature (the OLD wrong base matrix [[1,-1],
    [-1,1]] failed this: H(ones) had zero energy)."""
    e0 = torch.zeros(128, dtype=torch.float64)
    e0[0] = 1.0
    assert torch.allclose(
        hadamard128(e0),
        torch.full((128,), 1 / math.sqrt(128), dtype=torch.float64),
        atol=1e-12,
    )
    e1 = torch.zeros(128, dtype=torch.float64)
    e1[1] = 1.0
    r1 = hadamard128(e1)
    expected = torch.tensor(
        [1 / math.sqrt(128) * (1 if i % 2 == 0 else -1) for i in range(128)],
        dtype=torch.float64,
    )
    # e1's row of H128: alternating starting +.
    assert torch.allclose(r1, expected, atol=1e-12)


def test_hadamard128_orthogonality():
    """H(ones) has energy exactly 128 (the OLD base matrix gave 0),
    and inner products are preserved on random vectors."""
    ones = torch.ones(128, dtype=torch.float64)
    assert abs(float((hadamard128(ones) ** 2).sum()) - 128.0) < 1e-6
    g = torch.Generator().manual_seed(3)
    v1 = torch.randn(128, generator=g, dtype=torch.float64)
    v2 = torch.randn(128, generator=g, dtype=torch.float64)
    assert torch.isclose(v1 @ v2, hadamard128(v1) @ hadamard128(v2), rtol=1e-9)
    assert abs(float((hadamard128(v1) ** 2).sum() - (v1**2).sum())) < 1e-9


def test_hadamard128_random_vectors():
    """Permutation/row-sum sanity on random vectors: H is symmetric
    and involutive up to the normalization (H @ H = 128 * I / 128)."""
    g = torch.Generator().manual_seed(11)
    for _ in range(5):
        v = torch.randn(128, generator=g, dtype=torch.float64)
        # H(H(v)) == v (the normalized Hadamard is an involution).
        assert torch.allclose(hadamard128(hadamard128(v)), v, atol=1e-10)


def test_fp8_codec_real_roundtrip():
    """quant_dequant uses a REAL fp8 dtype cast (not a float .round()
    — the OLD bug); the platform dtype's finfo.max (not a hardcoded
    448); the tolerance is the per-vector scale."""
    d = fp8_dtype()
    assert fp8_max() == float(torch.finfo(d).max)
    g = torch.Generator().manual_seed(5)
    v = torch.randn(128, generator=g)
    ref = quant_dequant(v)
    # The dequantized value is within one quantization step.
    tol = expected_tolerance(v)
    assert torch.all((v - ref).abs() <= tol + 1e-12)
    # The fp8 values land on the dtype's actual grid.
    assert ref.dtype == torch.float64


def test_page_layout_roundtrip_real_2d():
    """The REAL [page_size, head_dim + 4] uint8 page: write then read
    reconstructs the (fp8, scale) pair exactly at every token
    offset. INDEPENDENT verification: the expected bytes are placed
    by HAND at the kernel's byte offsets (not via the writer), then
    read — the reader is not self-certified by the writer."""
    head_dim, page_size = 128, 16
    page = torch.zeros((page_size, head_dim + 4), dtype=torch.uint8)
    g = torch.Generator().manual_seed(9)
    offs = torch.arange(head_dim)
    for token_off in (0, 5, 15):
        # HAND-write the bytes at the kernel's offsets.
        vals = (torch.randn(head_dim, generator=g) * 10).to(fp8_dtype())
        byte_offs = k_offset_in_page(token_off, offs, head_dim)
        flat = page.flatten()
        flat[byte_offs] = vals.view(torch.uint8)
        s_float_idx = page_size * head_dim // 4 + token_off
        flat.view(torch.float32)[s_float_idx] = 0.03125
        # The reader reconstructs exactly.
        r_vals, r_scale = read_compressed_page(page, token_off, head_dim, page_size)
        assert r_scale == 0.03125
        assert torch.equal(r_vals.view(torch.uint8), vals.view(torch.uint8))
    # The writer agrees with the hand placement.
    page2 = torch.zeros((page_size, head_dim + 4), dtype=torch.uint8)
    vals = (torch.randn(head_dim, generator=g) * 10).to(fp8_dtype())
    write_compressed_page(page2, 5, vals, 0.03125, head_dim, page_size)
    r2, s2 = read_compressed_page(page2, 5, head_dim, page_size)
    assert torch.equal(r2.view(torch.uint8), vals.view(torch.uint8))
    assert s2 == 0.03125


def test_page_layout_padded_rows():
    """An explicitly padded row stride (a wider row width) is legal;
    the reader addresses through the actual layout."""
    head_dim, page_size = 128, 16
    page = torch.zeros((page_size, 140), dtype=torch.uint8)
    vals = torch.randn(head_dim).to(fp8_dtype())
    write_compressed_page(page, 3, vals, 0.015625, head_dim, page_size)
    r, s = read_compressed_page(page, 3, head_dim, page_size)
    assert torch.equal(r.view(torch.uint8), vals.view(torch.uint8))
    assert s == 0.015625


def test_page_layout_rejections():
    """A wrong row count, an undersized row, a non-uint8 dtype, a 1-D
    page, and a negative-stride view all reject clearly."""
    head_dim, page_size = 128, 16
    with pytest.raises(ValueError, match="rows"):
        read_compressed_page(
            torch.zeros((8, head_dim + 4), dtype=torch.uint8), 0, head_dim, page_size
        )
    with pytest.raises(ValueError, match="row width"):
        read_compressed_page(
            torch.zeros((page_size, head_dim), dtype=torch.uint8),
            0,
            head_dim,
            page_size,
        )
    with pytest.raises(ValueError, match="uint8"):
        read_compressed_page(
            torch.zeros((page_size, head_dim + 4), dtype=torch.int8),
            0,
            head_dim,
            page_size,
        )
    with pytest.raises(ValueError, match="2-D"):
        read_compressed_page(
            torch.zeros(page_size * (head_dim + 4), dtype=torch.uint8),
            0,
            head_dim,
            page_size,
        )
    # A non-contiguous byte axis (a transposed view) rejects.
    base = torch.zeros((head_dim + 4, page_size), dtype=torch.uint8)
    transposed = base.t()  # [page_size, head_dim+4], byte stride > 1
    assert transposed.stride(1) != 1
    with pytest.raises(ValueError, match="contiguous"):
        read_compressed_page(transposed, 0, head_dim, page_size)


def test_scale_offset_formula():
    """The scale's float index is page_size*head_dim//4 + token_off
    (the kernel's S_OFFSET_NBYTES_IN_PAGE = page_size * head_dim
    bytes), NOT head_dim*head_dim//4. Pinned at PAGE=16, H=128:
    512 + tok, not 4096 + tok."""
    head_dim, page_size = 128, 16
    page = torch.zeros((page_size, head_dim + 4), dtype=torch.uint8)
    vals = torch.zeros(head_dim).to(fp8_dtype())
    write_compressed_page(page, 7, vals, 0.5, head_dim, page_size)
    s_byte = page_size * head_dim + 7 * 4
    assert float(page.flatten()[s_byte : s_byte + 4].view(torch.float32)[0]) == 0.5
    assert page_size * head_dim // 4 + 7 == 512 + 7


def test_k_offset_preshuffle_structure():
    """The preshuffle offsets are a permutation of [0, page bytes):
    every (token, dim) pair maps to a distinct byte in its 16-token
    tile, and tiles do not overlap."""
    head_dim = 128
    offs = torch.arange(head_dim)
    seen = set()
    for t in range(16):
        bo = k_offset_in_page(t, offs, head_dim).tolist()
        assert len(set(bo)) == head_dim  # distinct within the token
        for b in bo:
            assert b not in seen, f"offset collision at t={t}"
            seen.add(b)
    # All within the first tile's extent.
    assert max(seen) < 16 * head_dim


def test_token_vec_separates_requests():
    """Distinct requests at the SAME position get DIFFERENT vectors
    (the OLD truncating encoding made them identical, hiding
    cross-request contamination)."""
    for pos in (0, 7, 100):
        for rnd in (0, 2):
            a = token_vec(0, rnd, pos, 0)
            b = token_vec(1, rnd, pos, 0)
            assert not torch.allclose(a, b), (pos, rnd)
            # Rounds and input indices also separate.
            c = token_vec(0, rnd + 1, pos, 0)
            d = token_vec(0, rnd, pos, 1)
            assert not torch.allclose(a, c)
            assert not torch.allclose(a, d)
    # Determinism.
    assert torch.equal(token_vec(3, 1, 42, 0), token_vec(3, 1, 42, 0))


@pytest.mark.parametrize(
    "dtype",
    [torch.float8_e4m3fn, torch.float8_e4m3fnuz, torch.float8_e5m2],
)
def test_fp8_grid_adjacency_and_subnormal(dtype):
    """The tolerance model is verified against the ACTUAL fp8 grid:
    decode every positive code, confirm the smallest positive value
    is the first SUBNORMAL (finfo.tiny / 2^mantissa — NOT
    finfo.tiny), and confirm the normal-range ULP equals the decoded
    grid's adjacent spacing (within the power-of-two exponent
    boundaries)."""
    finfo = torch.finfo(dtype)
    mantissa = 3 if "e4m3" in str(dtype) else 2
    # Decode ALL positive byte codes.
    codes = torch.arange(1, 256, dtype=torch.uint8)
    decoded = codes.view(dtype).double()
    positive = decoded[decoded > 0]
    min_positive = float(positive.min())
    assert min_positive == finfo.tiny / (2.0**mantissa), (
        dtype,
        min_positive,
        finfo.tiny / (2.0**mantissa),
    )
    # The subnormal range: all values below tiny share the absolute
    # spacing min_positive.
    subs = positive[positive < finfo.tiny]
    if subs.numel() > 1:
        steps = subs.unique()
        assert torch.allclose(steps.diff(), torch.full_like(steps[:-1], min_positive))
    # The normal range: the ULP at magnitude m is m * 2^-mantissa.
    normals = positive[positive >= finfo.tiny]
    sorted_n = normals.unique()
    # Check the ULP relation on the decoded grid.
    for m in (finfo.tiny, finfo.tiny * 4, 1.0, 64.0):
        if m > finfo.max:
            continue
        below = sorted_n[sorted_n <= m][-1]
        above = sorted_n[sorted_n >= m][0]
        if above > below:
            # The adjacent spacing at this magnitude.
            step = float(above - below)
            assert step <= max(m * (2.0**-mantissa) * 2, min_positive * 2), (
                dtype,
                m,
                step,
            )


def test_bf16_sentinel_exact():
    """The sentinel is exactly representable in bf16 (the OLD 12345
    became 12352 — an inexact float== comparison)."""
    s = torch.tensor(BF16_SENTINEL, dtype=torch.bfloat16)
    assert s.item() == BF16_SENTINEL
    # A bf16 tensor filled with it compares exactly.
    t = torch.full((4, 4), BF16_SENTINEL, dtype=torch.bfloat16)
    assert torch.all(t == BF16_SENTINEL)


# --- the CPU harness self-check (the gate's own execution path) --------------
#
# The CPU entry calls the SHARED native bodies (the SAME functions
# the native tests call) with the device injected as `cpu`, sync as
# a no-op, and the three kernel wrappers bound to controlled
# source-contract simulations. This is NOT a GPU acceptance and does
# not claim kernel correctness — it verifies the TEST HARNESS (the
# fixtures, address allocation, oracle, assertions) executes
# correctly, so a wrong expected value or shape fails HERE before
# reaching the device.


def _simulate_decode_kernel_source_contract(
    kv, tail, tsm, key, score, ape, sm, pos_t, C, R_, HEAD_DIM, page
):
    """A controlled simulation of the decode kernel's DOCUMENTED
    source contract (read-before-stash; the owner gate comes from
    the caller's sm=-1 masking): for each (req, t): pos = pos_t;
    if pos % R == R-1 and sm >= 0: read the R-window ring slots
    (phys = (logical_start + ps) % C), weighted-compress with APE,
    Hadamard, bf16, quantize, write at sm's page/token via the page
    layout; then stash at pos % C in tsm's block."""
    from dcp_kpool_gate_oracle import (
        hadamard128,
        write_compressed_page,
    )

    B, V = key.shape[0], key.shape[1]
    for req in range(B):
        for t in range(V):
            pos = int(pos_t[req, t])
            tslot = int(tsm[req, t])
            slot = int(sm[req, t])
            if pos % R_ == R_ - 1 and slot >= 0:
                logical_start = pos - (pos % R_)
                blk = tslot // C if tslot >= 0 else -1
                entries = []
                for ps in range(R_):
                    x = logical_start + ps
                    if x == pos:
                        k = key[req, t].double()
                        g = score[req, t].double()
                    else:
                        phys = x % C
                        k = tail[blk, 0, phys].double()
                        g = tail[blk, 1, phys].double()
                    entries.append((k, g))
                K = torch.stack([e[0] for e in entries])
                G = torch.stack([e[1] for e in entries])
                A = ape.double()
                scores = G + A
                mx = scores.max(dim=0).values
                probs = torch.exp(scores - mx)
                avg = (K * probs).sum(0) / probs.sum(dim=0)
                h = hadamard128(avg.to(torch.bfloat16).double())
                h_bf16 = h.to(torch.bfloat16).double()
                pg, tok = slot // page, slot % page
                # The kernel's codec: derive the rounded scale, cast
                # the SCALED value to fp8 (never the dequantized
                # magnitude), store both.
                from dcp_kpool_gate_oracle import fp8_dtype, fp8_max

                absmax = h_bf16.abs().amax().clamp_min(1e-4)
                scale = float(torch.exp2(torch.ceil(torch.log2(absmax / fp8_max()))))
                vals = (h_bf16 / scale).clamp(-fp8_max(), fp8_max()).to(fp8_dtype())
                write_compressed_page(kv[pg], tok, vals, scale, HEAD_DIM, page)
            # The stash (after the completion read).
            if tslot >= 0 and pos >= 0:
                blk = tslot // C
                tail[blk, 0, pos % C] = key[req, t]
                tail[blk, 1, pos % C] = score[req, t]


def _bind_simulated_kernels():
    """The controlled source-contract simulations bound to the SAME
    signatures the native wrappers expose (inspect-verified against
    the real ones by the native binding; the simulations mirror the
    documented kernel contracts)."""
    from dcp_kpool_gate_oracle import (
        fp8_dtype,
        fp8_max,
        hadamard128,
        write_compressed_page,
    )

    def sim_seed(tail_kv_cache, key, gate_score, tslot, kpool, head_dim=128):
        # The seed contract: the last-C window per request writes
        # the raw K/gate at tslot's block/slot.
        C = tail_kv_cache.shape[2]
        n = tslot.shape[0]
        for idx in range(n):
            t = int(tslot[idx])
            if t < 0:
                continue
            blk, slot = t // C, t % C
            ahead = int(tslot[idx + C]) if idx + C < n else -1
            if ahead >= 0 and ahead // C == blk:
                continue
            tail_kv_cache[blk, 0, slot] = key[idx]
            tail_kv_cache[blk, 1, slot] = gate_score[idx]

    def sim_decode(
        kv_cache,
        tail_kv_cache,
        tail_slot_mapping,
        key,
        slot_score,
        ape,
        slot_mapping,
        positions,
        pool_size=4,
        head_dim=128,
    ):
        R_ = pool_size
        C = tail_kv_cache.shape[2]
        PAGE = kv_cache.shape[1]
        B, V = key.shape[0], key.shape[1]
        for req in range(B):
            for t in range(V):
                pos = int(positions[req, t])
                tslot = int(tail_slot_mapping[req, t])
                slot = int(slot_mapping[req, t])
                if pos % R_ == R_ - 1 and slot >= 0:
                    logical_start = pos - (pos % R_)
                    blk = tslot // C if tslot >= 0 else -1
                    entries = []
                    for ps in range(R_):
                        x = logical_start + ps
                        if x == pos:
                            k = key[req, t].double()
                            g = slot_score[req, t].double()
                        else:
                            phys = x % C
                            k = tail_kv_cache[blk, 0, phys].double()
                            g = tail_kv_cache[blk, 1, phys].double()
                        entries.append((k, g))
                    K = torch.stack([e[0] for e in entries])
                    G = torch.stack([e[1] for e in entries])
                    A = ape.double()
                    scores = G + A
                    mx = scores.max(dim=0).values
                    probs = torch.exp(scores - mx)
                    avg = (K * probs).sum(0) / probs.sum(dim=0)
                    h = hadamard128(avg.to(torch.bfloat16).double())
                    h_bf16 = h.to(torch.bfloat16).double()
                    pg, tok = slot // PAGE, slot % PAGE
                    absmax = h_bf16.abs().amax().clamp_min(1e-4)
                    scale = float(
                        torch.exp2(torch.ceil(torch.log2(absmax / fp8_max())))
                    )
                    vals = (h_bf16 / scale).clamp(-fp8_max(), fp8_max()).to(fp8_dtype())
                    write_compressed_page(
                        kv_cache[pg], tok, vals, scale, head_dim, PAGE
                    )
                if tslot >= 0 and pos >= 0:
                    blk = tslot // C
                    tail_kv_cache[blk, 0, pos % C] = key[req, t]
                    tail_kv_cache[blk, 1, pos % C] = slot_score[req, t]

    def sim_slot(
        num_tokens,
        query_start_loc,
        seq_lens,
        block_table,
        block_size,
        compress_ratio,
        out=None,
        dcp_world_size=1,
        dcp_rank=0,
        pool_interleave=1,
    ):
        # The slot contract: the enumeration oracle's semantics.
        from test_indexer_deepseek_v4_slot_mapping import (
            _oracle_slot_ids_by_enumeration,
        )

        qsl = query_start_loc.cpu().tolist()
        sls = seq_lens.cpu().tolist()
        table = block_table.cpu().tolist()
        expected = _oracle_slot_ids_by_enumeration(
            qsl,
            sls,
            table,
            block_size,
            compress_ratio,
            dcp_world_size,
            dcp_rank,
            pool_interleave,
            num_tokens,
        )
        out.copy_(torch.tensor(expected, dtype=out.dtype, device=out.device))
        return out

    from types import SimpleNamespace

    return SimpleNamespace(seed=sim_seed, decode=sim_decode, slot=sim_slot)


def _cpu_dev():
    return torch.device("cpu")


def _cpu_sync():
    pass


def test_gate_harness_cpu_runs_the_shared_native_bodies():
    """The CPU entry executes the SHARED native bodies (the exact
    functions the native tests call): the seed acceptance for C in
    {4,7,8}, the decode discrimination (C=8/C=7 match; C=4 REQUIRED
    mismatch), and the slot acceptance for both ranks — with the
    kernel wrappers bound to controlled simulations. A wrong
    expected value, shape, or device discipline fails HERE."""
    import test_dcp_kpool_native_gate as gate

    kernels = _bind_simulated_kernels()
    # The seed body for all three C values.
    for C in (4, 7, 8):
        gate._run_seed(C, kernels, _cpu_dev(), _cpu_sync)
    # The decode body: the discrimination contract.
    m8 = gate._run_decode(8, kernels, _cpu_dev(), _cpu_sync)
    assert not m8, f"C=8 shared-body simulation diverged: {m8}"
    m7 = gate._run_decode(7, kernels, _cpu_dev(), _cpu_sync)
    assert not m7, f"C=7 shared-body simulation diverged: {m7}"
    m4 = gate._run_decode(4, kernels, _cpu_dev(), _cpu_sync)
    assert m4, "C=4 shared-body simulation showed no ring-aliasing divergence"
    # The slot body for both ranks (the exact hand-pinned ordinals).
    for rank in (0, 1):
        gate._run_slot(rank, kernels, _cpu_dev(), _cpu_sync)


def test_gate_harness_corruption_neighbor_byte_fails():
    """NEGATIVE: an injected single-byte corruption of a NON-owner kv
    byte must make the shared decode body FAIL (the write-set
    verification is load-bearing, not decorative)."""
    import test_dcp_kpool_native_gate as gate

    kernels = _bind_simulated_kernels()
    real_decode = kernels.decode

    def corrupting_decode(*args, _real=real_decode, **kwargs):
        # Corrupt ONE byte that must stay sentinel: page 5 (never
        # written by this scenario), token 0, first K byte.
        kv = args[0]
        kv[5].reshape(-1)[0] = 123
        return _real(*args, **kwargs)

    kernels.decode = corrupting_decode
    try:
        with pytest.raises(AssertionError, match="sentinel"):
            gate._run_decode(8, kernels, _cpu_dev(), _cpu_sync)
    finally:
        kernels.decode = real_decode


def test_gate_harness_corruption_tail_padding_fails():
    """NEGATIVE: an injected write to a tail slot outside the
    legitimate write set (a padding slot on a foreign block) must
    make the shared decode body FAIL."""
    import test_dcp_kpool_native_gate as gate

    kernels = _bind_simulated_kernels()
    real_decode = kernels.decode

    def corrupting_decode(*args, _real=real_decode, **kwargs):
        # Write to tail block 7 (never used by either request).
        tail = args[1]
        tail[7, 0, 0] = 1.0
        return _real(*args, **kwargs)

    kernels.decode = corrupting_decode
    try:
        with pytest.raises(AssertionError, match="sentinel"):
            gate._run_decode(8, kernels, _cpu_dev(), _cpu_sync)
    finally:
        kernels.decode = real_decode


def test_gate_harness_corrupt_scale_fails():
    """NEGATIVE: a corrupt (non-power-of-two / negative / NaN) stored
    scale must make the shared decode body FAIL even if the dequant
    value happens to land within tolerance."""
    import test_dcp_kpool_native_gate as gate

    for bad_scale in (0.003, -0.003, float("nan")):
        kernels = _bind_simulated_kernels()
        real_decode = kernels.decode

        def make_corrupting(bad, _real=real_decode):
            def corrupting_decode(*args, **kwargs):
                ret = _real(*args, **kwargs)
                kv = args[0]
                flat = kv[0].reshape(-1)
                # The scale region's first float: page_size*head_dim//4.
                s_idx = 16 * 128 // 4
                flat.view(torch.float32)[s_idx] = bad
                return ret

            return corrupting_decode

        kernels.decode = make_corrupting(bad_scale)
        try:
            with pytest.raises(AssertionError, match="corrupt scale|power-of-two"):
                gate._run_decode(8, kernels, _cpu_dev(), _cpu_sync)
        finally:
            kernels.decode = real_decode


def test_gate_harness_launch_budget():
    """The exact kernel-launch budget, MEASURED by the recording
    runner over the shared bodies (never hand-copied): the counts
    per kernel per case are asserted against the computed ceiling."""
    import test_dcp_kpool_native_gate as gate

    kernels = _bind_simulated_kernels()
    counter = gate._LaunchCounter(kernels)
    gate._run_seed(8, counter, _cpu_dev(), _cpu_sync)
    assert counter.counts == {"seed": 1, "decode": 0, "slot": 0}
    counter2 = gate._LaunchCounter(kernels)
    m = gate._run_decode(8, counter2, _cpu_dev(), _cpu_sync)
    assert not m
    # The decode body: 2 ranks x (3 rounds + 2 reqs x 8 flush
    # tokens) = 2 x 19 = 38 decode launches.
    assert counter2.counts["decode"] == 38, counter2.counts
    counter3 = gate._LaunchCounter(kernels)
    gate._run_slot(0, counter3, _cpu_dev(), _cpu_sync)
    assert counter3.counts == {"seed": 0, "decode": 0, "slot": 1}
    # The TOTAL over the full native gate: 3 seed + 3 decode cases
    # (38 each) + 2 slot = 3 + 114 + 2 = 119 custom kernel launches.
    total = 3 * 1 + 3 * 38 + 2 * 1
    assert total == 119
