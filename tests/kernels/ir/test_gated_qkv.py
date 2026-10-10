# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm import ir


def _reference(
    packed_qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cache: torch.Tensor,
    positions: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    *,
    is_neox_style: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens = packed_qkv.shape[0]
    q_gate_width = num_q_heads * 2 * head_dim
    kv_width = num_kv_heads * head_dim
    q_gate, key, _value = packed_qkv.split([q_gate_width, kv_width, kv_width], dim=-1)
    q_gate = q_gate.view(num_tokens, num_q_heads, 2 * head_dim)
    query, _gate = q_gate.split(head_dim, dim=-1)
    key = key.view(num_tokens, num_kv_heads, head_dim)

    def norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        x_float *= torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        return (x_float * (weight.float() + 1.0)).to(x.dtype)

    query = norm(query, q_weight)
    key = norm(key, k_weight)
    cos_sin = cache[positions]
    cos, sin = cos_sin.chunk(2, dim=-1)
    if positions.ndim == 2:
        selected_cos = cos[0].clone()
        selected_sin = sin[0].clone()
        selected_cos[..., 1:33:3] = cos[1, ..., 1:33:3]
        selected_sin[..., 1:33:3] = sin[1, ..., 1:33:3]
        selected_cos[..., 2:30:3] = cos[2, ..., 2:30:3]
        selected_sin[..., 2:30:3] = sin[2, ..., 2:30:3]
        cos, sin = selected_cos, selected_sin
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)

    def rope(x: torch.Tensor) -> torch.Tensor:
        rotary, passthrough = x[..., :rotary_dim], x[..., rotary_dim:]
        if is_neox_style:
            x1, x2 = rotary.chunk(2, dim=-1)
            rotary = torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
        else:
            x1, x2 = rotary[..., ::2], rotary[..., 1::2]
            rotary = torch.stack(
                (x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1
            ).flatten(-2)
        return torch.cat((rotary, passthrough), dim=-1)

    return (
        rope(query).reshape(num_tokens, num_q_heads * head_dim),
        rope(key).reshape(num_tokens, num_kv_heads * head_dim),
    )


@pytest.mark.parametrize("num_q_heads", [2, 3])
@pytest.mark.parametrize("num_tokens", [1, 7])
def test_gated_qkv_rms_norm_rope_native_interleaved_mrope(
    num_q_heads: int,
    num_tokens: int,
) -> None:
    torch.manual_seed(1234)
    num_kv_heads = 1
    head_dim = 256
    rotary_dim = 64
    packed_width = (2 * num_q_heads + 2 * num_kv_heads) * head_dim
    packed_qkv = torch.randn(num_tokens, packed_width, dtype=torch.bfloat16)
    packed_before = packed_qkv.clone()
    q_weight = torch.randn(head_dim, dtype=torch.bfloat16)
    k_weight = torch.randn(head_dim, dtype=torch.bfloat16)
    cache = torch.randn(128, rotary_dim, dtype=torch.bfloat16)
    index = torch.arange(num_tokens)
    positions = torch.stack((index + 3, index // 2 + 1, index * 2 + 5))
    args = (
        packed_qkv,
        q_weight,
        k_weight,
        cache,
        positions,
        1e-6,
        num_q_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        [11, 11, 10],
        True,
        True,
        1.0,
    )

    native = ir.ops.gated_qkv_rms_norm_rope.impls["native"].impl_fn
    actual = native(*args)
    expected = _reference(
        packed_qkv,
        q_weight,
        k_weight,
        cache,
        positions,
        num_q_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        is_neox_style=True,
    )

    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)
        assert actual_tensor.is_contiguous()
    assert torch.equal(packed_qkv, packed_before)
    packed_storage = packed_qkv.untyped_storage().data_ptr()
    assert actual[0].untyped_storage().data_ptr() != packed_storage
    assert actual[1].untyped_storage().data_ptr() != packed_storage


def test_gated_qkv_rms_norm_rope_native_one_dimensional_gptj() -> None:
    args = ir.ops.gated_qkv_rms_norm_rope.generate_inputs(
        num_tokens=4,
        num_q_heads=2,
        num_kv_heads=1,
        head_dim=32,
        rotary_dim=16,
        dtype=torch.float32,
        is_neox_style=False,
    )
    native = ir.ops.gated_qkv_rms_norm_rope.impls["native"].impl_fn
    query, key = native(*args)
    assert query.shape == (4, 64)
    assert key.shape == (4, 32)
    assert query.dtype == key.dtype == torch.float32


def test_gated_qkv_rms_norm_rope_rejects_invalid_packed_width() -> None:
    args = list(
        ir.ops.gated_qkv_rms_norm_rope.generate_inputs(
            num_tokens=1,
            num_q_heads=2,
            num_kv_heads=1,
            head_dim=32,
            rotary_dim=16,
            dtype=torch.float32,
        )
    )
    args[0] = args[0][..., :-1]
    native = ir.ops.gated_qkv_rms_norm_rope.impls["native"].impl_fn
    with pytest.raises(ValueError, match="packed_qkv last dimension"):
        native(*args)
