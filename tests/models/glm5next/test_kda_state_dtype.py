# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``--mamba-ssm-cache-dtype`` must reach GLM-5.3-Flash's KDA recurrent state
(it used to be dropped, leaving the state float32 unconditionally)."""

from types import SimpleNamespace

import pytest
import torch

from vllm.models.glm5next.common.kda import Glm5NextLinearAttention
from vllm.models.glm5next.common.model import (
    Glm5NextForCausalLM,
    Glm5NextForConditionalGeneration,
)


@pytest.mark.parametrize(
    ("ssm_dtype", "expected"),
    [("auto", torch.float32), ("float32", torch.float32), ("bfloat16", torch.bfloat16)],
)
@pytest.mark.parametrize("recover", [False, True])
def test_kda_recurrent_state_dtype_follows_ssm_cache_dtype(
    ssm_dtype, expected, recover
):
    model_config = SimpleNamespace(dtype=torch.bfloat16)
    cache_config = SimpleNamespace(
        mamba_cache_dtype="auto",
        mamba_ssm_cache_dtype=ssm_dtype,
        use_kda_recoverssm=recover,
    )
    vllm_config = SimpleNamespace(model_config=model_config, cache_config=cache_config)
    want = (torch.bfloat16, expected)
    if recover:
        want += (torch.float32, torch.float32)

    assert Glm5NextForCausalLM.get_mamba_state_dtype_from_config(vllm_config) == want
    assert (
        Glm5NextForConditionalGeneration.get_mamba_state_dtype_from_config(vllm_config)
        == want
    )
    # The layer's own dtype must agree with the model-level one, since the
    # KV-cache spec and the kernels read them separately.
    layer = Glm5NextLinearAttention.__new__(Glm5NextLinearAttention)
    layer.model_config, layer.cache_config = model_config, cache_config
    assert layer.get_state_dtype() == want


@pytest.mark.parametrize("tp,pcp", [(1, 8), (8, 1)])
@pytest.mark.parametrize("recover", [False, True])
def test_kda_config_and_layer_agree_on_head_sharded_state(tp, pcp, recover):
    from vllm.config.kda_parallel import get_kda_layer_tp

    config = SimpleNamespace(
        additional_config={},
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp,
            kda_layer_tp=8,
            prefill_context_parallel_size=pcp,
            decode_context_parallel_size=1,
        ),
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(
                linear_num_heads=64,
                linear_head_dim=128,
                linear_conv_kernel_dim=4,
            )
        ),
        speculative_config=SimpleNamespace(num_speculative_tokens=3),
        cache_config=SimpleNamespace(use_kda_recoverssm=recover),
    )
    layer = Glm5NextLinearAttention.__new__(Glm5NextLinearAttention)
    layer.kda_layer_tp = get_kda_layer_tp(config)
    layer.num_heads, layer.head_dim, layer.conv_size, layer.num_spec = 64, 128, 4, 3
    layer.cache_config = config.cache_config
    expected = layer.get_state_shape()
    assert expected[1] == (8, 128, 128)
    if recover:
        assert expected[2:] == ((8, 4, 128), (8, 4, 256))
    assert Glm5NextForCausalLM.get_mamba_state_shape_from_config(config) == expected
    assert (
        Glm5NextForConditionalGeneration.get_mamba_state_shape_from_config(config)
        == expected
    )
