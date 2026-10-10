# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""IR operations for gated packed-QKV projection post-processing."""

import torch
from torch import Tensor

from ..op import register_op


def _rms_norm_with_offset(
    x: Tensor,
    weight: Tensor,
    epsilon: float,
    weight_offset: float,
) -> Tensor:
    """RMSNorm with an optional additive weight offset.

    The conversion back to the activation dtype is intentional: RoPE observes
    the stored low-precision result in the unfused model path.
    """
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + epsilon)
    weight = weight.to(torch.float32) + weight_offset
    return (x * weight).to(orig_dtype)


def _select_mrope_cache(
    cache: Tensor,
    mrope_section: list[int],
    mrope_interleaved: bool,
) -> Tensor:
    """Select one modality for each rotary frequency."""
    if not mrope_section:
        raise ValueError("mrope_section is required for multi-axis positions")
    if cache.shape[0] != len(mrope_section):
        raise ValueError(
            "positions first dimension must equal the number of MRoPE sections"
        )
    if sum(mrope_section) != cache.shape[-1]:
        raise ValueError("MRoPE sections must cover the rotary half dimension")

    if not mrope_interleaved:
        chunks = cache.split(mrope_section, dim=-1)
        return torch.cat([chunk[i] for i, chunk in enumerate(chunks)], dim=-1)

    # Match MRotaryEmbedding.apply_interleaved_rope. Axis zero supplies the
    # default frequency. Later axes replace their interleaved frequencies only
    # for the number of entries assigned to that axis.
    selected = cache[0]
    frequency = torch.arange(cache.shape[-1], device=cache.device)
    for axis, section_size in enumerate(mrope_section[1:], start=1):
        axis_mask = (frequency.remainder(len(mrope_section)) == axis) & (
            frequency < section_size * len(mrope_section)
        )
        selected = torch.where(axis_mask, cache[axis], selected)
    return selected


def _apply_partial_rope(
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
    rotary_dim: int,
    is_neox_style: bool,
) -> Tensor:
    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)

    if is_neox_style:
        x1, x2 = x_rot.chunk(2, dim=-1)
        x_rot = torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
    else:
        x1 = x_rot[..., ::2]
        x2 = x_rot[..., 1::2]
        x_rot = torch.stack((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
        x_rot = x_rot.flatten(-2)

    return torch.cat((x_rot, x_pass), dim=-1)


@register_op(activations=["packed_qkv"])
def gated_qkv_rms_norm_rope(
    packed_qkv: Tensor,
    q_weight: Tensor,
    k_weight: Tensor,
    cos_sin_cache: Tensor,
    positions: Tensor,
    epsilon: float,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    mrope_section: list[int],
    mrope_interleaved: bool,
    is_neox_style: bool,
    weight_offset: float,
) -> tuple[Tensor, Tensor]:
    """Post-process a gated packed-QKV projection.

    ``packed_qkv`` uses the layout
    ``[Q0, gate0, ..., Qn, gaten, K..., V...]``. This functional op extracts
    Q and K, applies per-head RMSNorm to both, and applies partial RoPE. Gate
    and V remain in ``packed_qkv`` and are intentionally not returned so their
    consumers can retain zero-copy views of the packed projection.

    ``positions`` may be one-dimensional or ``[axes, tokens]``. In the latter
    case, ``mrope_section`` and ``mrope_interleaved`` define which position axis
    supplies each rotary frequency. ``weight_offset=1`` provides Gemma-style
    RMSNorm; ``weight_offset=0`` provides conventional RMSNorm weights.
    """
    if packed_qkv.ndim != 2:
        raise ValueError("packed_qkv must have shape [tokens, packed_width]")
    if num_q_heads <= 0 or num_kv_heads <= 0:
        raise ValueError("num_q_heads and num_kv_heads must be positive")
    if head_dim <= 0:
        raise ValueError(f"head_dim must be positive, got {head_dim}")
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2:
        raise ValueError(
            "rotary_dim must be a positive even integer no larger than head_dim"
        )
    if positions.ndim not in (1, 2):
        raise ValueError("positions must have shape [tokens] or [axes, tokens]")

    num_tokens = packed_qkv.shape[0]
    if positions.shape[-1] != num_tokens:
        raise ValueError("positions token dimension must match packed_qkv")

    q_gate_size = num_q_heads * 2 * head_dim
    k_size = num_kv_heads * head_dim
    expected_size = q_gate_size + 2 * k_size
    if packed_qkv.shape[-1] != expected_size:
        raise ValueError(
            f"packed_qkv last dimension must be {expected_size}, "
            f"got {packed_qkv.shape[-1]}"
        )
    if q_weight.shape != (head_dim,) or k_weight.shape != (head_dim,):
        raise ValueError("Q/K RMSNorm weights must have shape [head_dim]")
    if cos_sin_cache.ndim != 2 or cos_sin_cache.shape[-1] != rotary_dim:
        raise ValueError("cos_sin_cache must have shape [max_position, rotary_dim]")

    q_gate, key, _value = packed_qkv.split([q_gate_size, k_size, k_size], dim=-1)
    q_gate = q_gate.view(num_tokens, num_q_heads, 2 * head_dim)
    query, _gate = q_gate.split(head_dim, dim=-1)
    key = key.view(num_tokens, num_kv_heads, head_dim)

    query = _rms_norm_with_offset(query, q_weight, epsilon, weight_offset)
    key = _rms_norm_with_offset(key, k_weight, epsilon, weight_offset)

    cache = cos_sin_cache.to(device=packed_qkv.device, dtype=packed_qkv.dtype)
    cos_sin = cache[positions]
    cos, sin = cos_sin.chunk(2, dim=-1)
    if positions.ndim == 2:
        cos = _select_mrope_cache(cos, mrope_section, mrope_interleaved)
        sin = _select_mrope_cache(sin, mrope_section, mrope_interleaved)

    query = _apply_partial_rope(query, cos, sin, rotary_dim, is_neox_style).reshape(
        num_tokens, num_q_heads * head_dim
    )
    key = _apply_partial_rope(key, cos, sin, rotary_dim, is_neox_style).reshape(
        num_tokens, num_kv_heads * head_dim
    )
    return query, key


# Provider implementations may fuse reductions and BF16 RoPE, changing
# accumulation order; cancellation can exceed the default elementwise tolerance.
gated_qkv_rms_norm_rope.override_tolerance(torch.bfloat16, atol=2e-2, rtol=1.6e-2)


@gated_qkv_rms_norm_rope.register_input_generator
def _gated_qkv_rms_norm_rope_input_generator(
    num_tokens: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    dtype: torch.dtype,
    epsilon: float = 1e-6,
    mrope_section: list[int] | None = None,
    mrope_interleaved: bool = False,
    is_neox_style: bool = True,
    weight_offset: float = 1.0,
) -> tuple:
    if mrope_section is None:
        mrope_section = []
    packed_size = (2 * num_q_heads + 2 * num_kv_heads) * head_dim
    packed_qkv = torch.randn(num_tokens, packed_size, dtype=dtype)
    q_weight = torch.randn(head_dim, dtype=dtype)
    k_weight = torch.randn(head_dim, dtype=dtype)
    max_position = max(128, num_tokens + 1)
    cos_sin_cache = torch.randn(max_position, rotary_dim, dtype=dtype)
    if mrope_section:
        positions = torch.randint(
            max_position, (len(mrope_section), num_tokens), dtype=torch.int64
        )
    else:
        positions = torch.randint(max_position, (num_tokens,), dtype=torch.int64)
    return (
        packed_qkv,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        epsilon,
        num_q_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        mrope_section,
        mrope_interleaved,
        is_neox_style,
        weight_offset,
    )
