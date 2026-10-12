# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared CPU oracles and fixtures for the KPool native gate.

Everything here is CPU-only and runs WITHOUT any GPU: this module is
the self-check suite's substrate AND the native gate's oracle. It
deliberately reproduces the REAL kernel's arithmetic from source
(the butterfly Hadamard, the fp8 codec with the platform dtype, the
preshuffled page layout) — verified against independent mathematical
properties, never against the kernels themselves.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# --- import root (exact, no substring heuristics) ---------------------------
# tests/v1/attention/THIS.py -> parents[3] is the worktree root.
_THIS_FILE = Path(__file__).resolve()
_WORKTREE_ROOT = _THIS_FILE.parents[3]
_VLLM_ROOT = _WORKTREE_ROOT / "vllm"
if _VLLM_ROOT.is_dir() and str(_WORKTREE_ROOT) not in sys.path:
    # Put the worktree root FIRST so `import vllm` resolves here even
    # if an installed package exists. We do NOT reload already-
    # imported modules — the caller asserts the identity instead.
    sys.path.insert(0, str(_WORKTREE_ROOT))


def assert_import_root() -> None:
    """Assert that vllm and the three modules under test all resolve
    inside THIS worktree (an exact root comparison — no substring
    matching; a differently-named staged directory is fine as long
    as it is the actual source root)."""
    import vllm

    actual = Path(vllm.__file__).resolve().parent
    expected = _VLLM_ROOT.resolve()
    assert actual == expected, f"vllm resolves to {actual}, expected {expected}"

    import vllm.models.glm5next.amd.ops.kpool_compress as _amd
    import vllm.v1.attention.backends.mla.compressor_utils as _cu

    for mod in (_amd, _cu):
        mod_file = Path(mod.__file__).resolve()
        # The module file must live under the worktree's vllm root.
        assert _VLLM_ROOT.resolve() in mod_file.parents, (
            f"{mod.__name__} resolves outside the worktree: {mod.__file__}"
        )


# --- the butterfly Hadamard-128 (mirrors _hadamard128 exactly) --------------


def _had_stage(x: torch.Tensor, groups: int, stride: int) -> torch.Tensor:
    x3 = x.reshape(groups, 2, stride)
    x3 = x3.transpose(1, 2)  # (G, S, 2)
    a, b = x3[..., 0], x3[..., 1]
    j = torch.stack([a + b, a - b], dim=-1)  # (G, S, 2)
    j = j.transpose(1, 2)  # (G, 2, S)
    return j.reshape(-1)


def hadamard128(x: torch.Tensor) -> torch.Tensor:
    """The kernel's exact butterfly (7 stages) + 1/sqrt(128), in
    float64 throughout (the caller's dtype is preserved at the
    boundary by the tests' explicit casts)."""
    x = x.detach().clone().double()
    for groups, stride in (
        (64, 1),
        (32, 2),
        (16, 4),
        (8, 8),
        (4, 16),
        (2, 32),
        (1, 64),
    ):
        x = _had_stage(x, groups, stride)
    return x * 0.08838834764831845


# --- the preshuffled kv page layout (mirrors _cache_k_offset) ---------------
#
# The REAL kv cache page is [page_size, head_dim + 4] uint8: the fp8 K
# region occupies the first head_dim bytes of each row (addressed by
# the kernel's preshuffled _cache_k_offset within the FLAT page), and
# the fp32 scale region occupies the trailing 4 bytes per row. The
# kernel's S_OFFSET_NBYTES_IN_PAGE = page_size * head_dim, so the
# scale's FLOAT index is page_size * head_dim // 4 + token_off.


def k_offset_in_page(
    token_off: int, dim_off: torch.Tensor, head_dim: int
) -> torch.Tensor:
    """The ROCm preshuffle layout: (t//16)*16*head_dim + (d//16)*256
    + (t%16)*16 + d%16 (16-token tiles)."""
    t = token_off
    d = dim_off
    return (t // 16) * 16 * head_dim + (d // 16) * 16 * 16 + (t % 16) * 16 + d % 16


def _page_flat(page: torch.Tensor, page_size: int, head_dim: int) -> torch.Tensor:
    """Validate and flatten a kv page to its flat byte view.

    Accepts the REAL [page_size, head_dim + 4] uint8 layout (the
    page_size derived STRICTLY from the argument or the shape's
    first dim — never from head_dim). An explicitly padded row
    stride is allowed via ``row_stride``; a negative stride or an
    illegal size rejects.
    """
    if page.dtype != torch.uint8:
        raise ValueError(f"kv page must be uint8, got {page.dtype}")
    if page.ndim == 2:
        if page.shape[0] != page_size:
            raise ValueError(
                f"page has {page.shape[0]} rows, expected page_size {page_size}"
            )
        if page.shape[1] < head_dim + 4:
            raise ValueError(
                f"page row width {page.shape[1]} < head_dim + 4 ({head_dim} + 4)"
            )
        row_stride = page.stride(0)
        if row_stride < 0:
            raise ValueError("negative page row stride is not supported")
        if page.stride(1) != 1:
            raise ValueError("the page's byte axis must be contiguous")
        # A padded row stride is legal (explicit padding); the flat
        # view follows the actual stride via flatten.
        return page.flatten()
    raise ValueError(
        f"kv page must be 2-D [page_size, head_dim + 4], got {page.ndim}-D"
    )


def write_compressed_page(
    page: torch.Tensor,
    token_off: int,
    values_fp8: torch.Tensor,
    scale: float,
    head_dim: int,
    page_size: int,
) -> None:
    """Write one compressed entry into a [page_size, head_dim + 4]
    kv page following the kernel's exact layout: the fp8 K at the
    preshuffled offsets within the FLAT K region (page_size *
    head_dim bytes), the fp32 scale at byte offset
    S_OFFSET_NBYTES_IN_PAGE + token_off * 4."""
    flat = _page_flat(page, page_size, head_dim)
    offs = torch.arange(head_dim)
    byte_offs = k_offset_in_page(token_off, offs, head_dim)
    if int(byte_offs.max()) >= page_size * head_dim:
        raise ValueError(
            f"token offset {token_off} outside the K region "
            f"({int(byte_offs.max())} >= {page_size * head_dim})"
        )
    flat[byte_offs] = values_fp8.view(torch.uint8).cpu()
    # The scale region: FLAT byte offset page_size*head_dim + tok*4,
    # reinterpreted as float32 on the flattened view (the 2D page's
    # row-major float view would misalign).
    flat.view(torch.float32)[page_size * head_dim // 4 + token_off] = scale


def read_compressed_page(
    page: torch.Tensor, token_off: int, head_dim: int, page_size: int
) -> tuple[torch.Tensor, float]:
    """The inverse reader: reconstruct (fp8 values, scale) exactly as
    the kernel wrote them, from the REAL [page_size, head_dim + 4]
    layout."""
    flat = _page_flat(page, page_size, head_dim)
    offs = torch.arange(head_dim)
    byte_offs = k_offset_in_page(token_off, offs, head_dim)
    if int(byte_offs.max()) >= page_size * head_dim:
        raise ValueError(
            f"token offset {token_off} outside the K region "
            f"({int(byte_offs.max())} >= {page_size * head_dim})"
        )
    fp8 = flat[byte_offs].view(fp8_dtype())
    scale = float(flat.view(torch.float32)[page_size * head_dim // 4 + token_off])
    return fp8, scale


# --- the fp8 codec (the platform dtype; finfo.max; rounded scale) -----------


def fp8_dtype() -> torch.dtype:
    from vllm.platforms import current_platform

    return current_platform.fp8_dtype()


def fp8_max() -> float:
    return float(torch.finfo(fp8_dtype()).max)


def quant_dequant(x: torch.Tensor, scale: float | None = None) -> torch.Tensor:
    """The kernel's codec: absmax scale (power-of-two rounded when
    ROUND_SCALE) + a REAL fp8 cast round-trip. The BF16 rounding of
    the Hadamard output happens BEFORE this, at the caller, matching
    the kernel's order. The scale may be forced (for verifying a
    STORED scale) or derived."""
    d = fp8_dtype()
    x = x.double()
    if scale is None:
        absmax = x.abs().amax().clamp_min(1e-4)
        scale = float(torch.exp2(torch.ceil(torch.log2(absmax / fp8_max()))))
    q = (x / scale).clamp(-fp8_max(), fp8_max())
    fp8 = q.to(d)
    return fp8.double() * scale


def _min_subnormal() -> float:
    """The dtype's smallest POSITIVE representable value: the first
    SUBNORMAL. finfo.tiny is the smallest NORMAL (e4m3: 2^-6; the
    first subnormal is tiny / 2^mantissa — e4m3: tiny/8, e5m2:
    tiny/4). Verified against the actual decode of the smallest
    positive code by the self-check suite."""
    d = fp8_dtype()
    mantissa = 3 if "e4m3" in str(d) else 2
    return float(torch.finfo(d).tiny) / (2.0**mantissa)


def expected_tolerance(x: torch.Tensor) -> torch.Tensor:
    """A per-element, interpretable bound on the codec's round-trip
    error, derived from the fp8 grid itself (no blanket fixed atol):

    - For an element at scaled magnitude m >= finfo.tiny, the grid
      step is m * 2^-mantissa_bits; round-to-nearest error <= half
      of that. We use ONE full ULP (conservative, covering the
      saturation clamp's delta).
    - For m >= finfo.tiny (the normal range), the grid step is m *
      2^-mantissa_bits; we use ONE full ULP (conservative).
    - For m < finfo.tiny (the subnormal range), the grid spacing is
      the ABSOLUTE subnormal step (tiny / 2^mantissa); the error is
      bounded by that step (values below half the first subnormal
      round to zero).
    - Saturated elements (|x/scale| > fp8_max after the codec's own
      scale ceiling) carry the EXACT clamp delta.
    - Zero is exact.
    """
    x = x.double()
    absmax = x.abs().amax().clamp_min(1e-4)
    scale = float(torch.exp2(torch.ceil(torch.log2(absmax / fp8_max()))))
    d = fp8_dtype()
    mantissa = 3 if "e4m3" in str(d) else 2
    tiny = float(torch.finfo(d).tiny)
    sub_step = tiny / (2.0**mantissa)
    scaled = x / scale
    scaled_mag = scaled.abs()
    ulp = torch.where(
        scaled_mag >= tiny,
        scaled_mag * (2.0**-mantissa),
        torch.full_like(scaled_mag, sub_step),
    )
    clamp_delta = (scaled_mag - fp8_max()).clamp_min(0)
    tol = torch.where(clamp_delta > 0, clamp_delta, ulp) * scale
    return tol


# --- token identity (no truncating encodings) --------------------------------


def token_vec(req: int, round_idx: int, pos: int, input_idx: int, dim: int = 128):
    """A deterministic vector per (req, round, pos, input_idx). The
    seed mixes the FULL tuple (no bit-range truncation): distinct
    requests at the same position get DIFFERENT vectors — asserted
    by the self-check suite."""
    seed = ((req * 1_000_003) + (round_idx * 10_007) + (pos * 101) + input_idx) % (
        2**31 - 1
    )
    g = torch.Generator().manual_seed(seed)
    return torch.randn(dim, generator=g)


def token_id(req: int, round_idx: int, pos: int, input_idx: int) -> tuple:
    """The logging identity: the exact TUPLE — no int encoding, no
    truncation, no collision claims needed."""
    return (req, round_idx, pos, input_idx)


# --- sentinels (exactly representable in bf16) --------------------------------

BF16_SENTINEL = 10240.0  # 10 * 2^10 — exact in bf16 (asserted in tests)
UINT8_SENTINEL = 200
