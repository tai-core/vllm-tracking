# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check GLM's speculative outputs and committed state against its native path."""

from math import prod
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.config.compilation import CUDAGraphMode
from vllm.models.glm5next.nvidia import kda
from vllm.models.glm5next.nvidia.model import (
    Glm5NextForCausalLM,
    Glm5NextForConditionalGeneration,
)
from vllm.models.kimi_k3.nvidia.kda_metadata import KimiK3KDAMetadata
from vllm.models.kimi_k3.nvidia.ops import recoverssm
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.kv_cache_interface import MambaSpec


def _layer(recover, dtype, device, dim_first, num_blocks, num_spec=3):
    layer = object.__new__(kda.Glm5NextLinearAttention)
    nn.Module.__init__(layer)
    layer.prefix = "model.layers.0.self_attn"
    layer.model_config = SimpleNamespace(dtype=torch.bfloat16)
    layer.cache_config = SimpleNamespace(mamba_cache_dtype=str(dtype).split(".")[-1])
    layer.use_recoverssm = recover
    layer.tp_size = 1
    layer.num_heads = layer.local_num_heads = 2
    layer.head_dim = 128
    layer.local_projection_size = 256
    layer.conv_size = 4
    layer.num_spec = num_spec
    layer.kda_safe_gate = True
    layer.kda_lower_bound = -5.0
    layer._conv_state_dim_first = dim_first
    layer._merged_conv_weight = torch.randn(768, 4, device=device) * 0.1
    layer.q_conv1d = SimpleNamespace(bias=None)
    layer.A_log = nn.Parameter(torch.randn(1, 1, 2, 1, device=device) * 0.1)
    layer.dt_bias = nn.Parameter(torch.randn(256, device=device) * 0.1)
    conv_width = num_spec + 3
    conv_shape = (768, conv_width) if dim_first else (conv_width, 768)
    shapes = [conv_shape, (2, 128, 128)]
    dtypes = [torch.bfloat16, dtype]
    if recover:
        shapes += [(2, num_spec + 1, 128), (2, num_spec + 1, 256)]
        dtypes += [torch.float32, torch.bfloat16]
    # Packed cache pages have padding between blocks in the serving runtime.
    layer.kv_cache = tuple(
        torch.zeros(
            num_blocks,
            prod(shape) + 128,
            dtype=state_dtype,
            device=device,
        )[:, : prod(shape)].view(num_blocks, *shape)
        for shape, state_dtype in zip(shapes, dtypes)
    )
    return layer


@pytest.mark.parametrize("recover", [False, True])
def test_glm_layer_and_multimodal_model_agree_on_state_layout(recover):
    layer = _layer(recover, torch.bfloat16, "cpu", True, 1)
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            hf_config=SimpleNamespace(
                linear_num_heads=2, linear_head_dim=128, linear_conv_kernel_dim=4
            ),
        ),
        cache_config=SimpleNamespace(
            mamba_cache_dtype="bfloat16", use_kda_recoverssm=recover
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
        speculative_config=SimpleNamespace(num_speculative_tokens=3),
    )
    for model in (Glm5NextForCausalLM, Glm5NextForConditionalGeneration):
        assert (
            model.get_mamba_state_dtype_from_config(config) == layer.get_state_dtype()
        )
        assert (
            model.get_mamba_state_shape_from_config(config) == layer.get_state_shape()
        )
    assert len(layer.get_state_shape()) == (4 if recover else 2)
    if recover:
        assert layer.get_state_dtype()[2:] == (torch.float32, torch.bfloat16)
        assert layer.get_state_shape()[2:] == ((2, 4, 128), (2, 4, 256))
        assert layer.get_attn_backend().get_name() == "GLM5NEXT_RECOVERSSM"


def test_glm_metadata_build_binds_and_reuses_state_commit_context():
    """The first speculative batch must bind the group's real cache tensors."""
    layer = _layer(True, torch.bfloat16, "cpu", kda.is_conv_state_dim_first(), 1)
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.NONE,
            max_cudagraph_capture_size=None,
            static_forward_context={layer.prefix: layer},
        ),
        speculative_config=SimpleNamespace(
            num_speculative_tokens=3, parallel_drafting=False
        ),
        num_speculative_tokens=3,
        cache_config=SimpleNamespace(use_kda_recoverssm=True, mamba_cache_mode="none"),
        scheduler_config=SimpleNamespace(max_num_seqs=1),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(linear_key_head_dim=128)
        ),
        additional_config={"gdn_prefill_backend": "triton"},
    )
    builder = layer.get_attn_backend().get_builder_cls()(
        MambaSpec(
            block_size=16,
            shapes=layer.get_state_shape(),
            dtypes=layer.get_state_dtype(),
            mamba_cache_mode="none",
            num_speculative_blocks=0,
        ),
        [layer.prefix],
        config,
        torch.device("cpu"),
    )
    query_start = torch.tensor([0, 4], dtype=torch.int32)
    common = CommonAttentionMetadata(
        query_start_loc=query_start,
        query_start_loc_cpu=query_start,
        seq_lens=torch.tensor([12], dtype=torch.int32),
        num_reqs=1,
        num_actual_tokens=4,
        max_query_len=4,
        max_seq_len=12,
        block_table_tensor=torch.zeros((1, 1), dtype=torch.int32),
        slot_mapping=torch.zeros(4, dtype=torch.int64),
        is_prefilling=torch.tensor([False]),
    )
    accepted = torch.ones(1, dtype=torch.int32)
    draft_lengths = torch.tensor([3], dtype=torch.int32)
    metadata = builder.build(0, common, accepted, draft_lengths)
    context = metadata.recoverssm_context
    assert context is not None
    assert context.checkpoints[0] is layer.kv_cache[1]
    assert metadata.recoverssm_commit is not None
    assert (
        builder.build(0, common, accepted, draft_lengths).recoverssm_context is context
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA/HCU")
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("dim_first", [False, True])
@pytest.mark.parametrize("spec_query_len", [4, 5])
@pytest.mark.parametrize("align_mode", [False, True])
@torch.inference_mode()
def test_glm_recovery_matches_native_acceptance_and_slot_reuse(
    monkeypatch, state_dtype, dim_first, spec_query_len, align_mode
):
    """Exercise GLM's real conv/verify path, rollback, and a reused request slot."""
    torch.manual_seed(317)
    monkeypatch.setattr(recoverssm, "is_conv_state_dim_first", lambda: dim_first)
    native = _layer(
        False,
        state_dtype,
        "cuda",
        dim_first,
        3 * spec_query_len + 1,
        num_spec=spec_query_len - 1,
    )
    recovered = _layer(
        True,
        state_dtype,
        "cuda",
        dim_first,
        7 if align_mode else 4,
        num_spec=spec_query_len - 1,
    )
    recovered._merged_conv_weight.copy_(native._merged_conv_weight)
    recovered.A_log.copy_(native.A_log)
    recovered.dt_bias.copy_(native.dt_bias)
    native_indices = torch.arange(
        1, 3 * spec_query_len + 1, device="cuda", dtype=torch.int32
    ).view(3, spec_query_len)
    recover_indices = torch.arange(1, 4, device="cuda", dtype=torch.int32)
    initial_states = (
        torch.randn(3, 2, 128, 128, device="cuda", dtype=state_dtype) * 0.05
    )
    initial_conv = torch.randn_like(recovered.kv_cache[0][recover_indices.long()]) * 0.1
    native.kv_cache[1][native_indices[:, 0]] = initial_states
    recovered.kv_cache[1][recover_indices.long()] = initial_states
    native.kv_cache[0][native_indices[:, 0]] = initial_conv
    recovered.kv_cache[0][recover_indices.long()] = initial_conv
    previous_accepted = torch.ones(3, device="cuda", dtype=torch.int32)
    context = recoverssm.KDARecoverSSMCommitContext.create(
        [recovered], spec_query_len=spec_query_len, max_num_reqs=3
    )
    computed = [7, 13, 15]
    block_table = torch.tensor([[1, 4], [2, 5], [3, 6]], device="cuda")

    for round_idx, (lengths, accepted) in enumerate(
        [
            ([spec_query_len] * 3, [1, 2, spec_query_len]),
            ([spec_query_len, 1, spec_query_len - 1], [spec_query_len, 1, 2]),
            ([1, spec_query_len, spec_query_len], [1, spec_query_len, 1]),
        ]
    ):
        if round_idx == 2:
            # Release request zero, then assign that same slot to a new request.
            native.kv_cache[0][1].zero_()
            native.kv_cache[1][1].zero_()
            recovered.kv_cache[0][1].zero_()
            recovered.kv_cache[1][1].zero_()
            previous_accepted[0] = 1
            if align_mode:
                computed[0] = 0
                recover_indices[0] = 1
        qsl = torch.tensor(
            [0, *torch.tensor(lengths).cumsum(0).tolist()],
            dtype=torch.int32,
            device="cuda",
        )
        ntokens = sum(lengths)
        qkv = torch.randn(ntokens, 768, device="cuda", dtype=torch.bfloat16)
        gate = torch.randn(1, ntokens, 2, 128, device="cuda", dtype=torch.bfloat16)
        beta = torch.randn(1, ntokens, 2, device="cuda", dtype=torch.bfloat16)
        outputs = []
        initial_checkpoint = recovered.kv_cache[1].clone()
        for layer in (native, recovered):
            metadata_cls = (
                KimiK3KDAMetadata if layer.use_recoverssm else GDNAttentionMetadata
            )
            metadata = metadata_cls(
                num_prefills=0,
                num_prefill_tokens=0,
                num_decodes=0,
                num_decode_tokens=0,
                num_spec_decodes=3,
                num_spec_decode_tokens=ntokens,
                num_actual_tokens=ntokens,
                spec_query_start_loc=qsl,
                spec_state_indices_tensor=(
                    recover_indices[:, None] if layer.use_recoverssm else native_indices
                ),
                spec_sequence_masks=(
                    None
                    if layer.use_recoverssm
                    else torch.ones(3, device="cuda", dtype=torch.bool)
                ),
                num_accepted_tokens=(
                    torch.ones_like(previous_accepted)
                    if layer.use_recoverssm
                    else previous_accepted
                ),
            )
            monkeypatch.setattr(
                kda,
                "get_forward_context",
                lambda layer=layer, metadata=metadata: SimpleNamespace(
                    attn_metadata={layer.prefix: metadata}
                ),
            )
            out = torch.empty(1, ntokens, 2, 128, device="cuda", dtype=torch.bfloat16)
            layer._forward(qkv.clone(), gate, beta, out)
            outputs.append(out)

        torch.testing.assert_close(
            recovered.kv_cache[1], initial_checkpoint, rtol=0, atol=0
        )
        # Small state/output errors can change MTP argmax despite unchanged text.
        torch.testing.assert_close(outputs[1], outputs[0], atol=0, rtol=0)
        accepted_gpu = torch.tensor(accepted, device="cuda", dtype=torch.int32)
        if align_mode:
            context.commit(
                accepted_gpu,
                recover_indices,
                qsl,
                block_table=block_table,
                num_computed_tokens=torch.tensor(computed, device="cuda"),
                mamba_block_size=16,
            )
        else:
            context.commit(accepted_gpu, recover_indices, qsl)
        for row, count in enumerate(accepted):
            if align_mode:
                boundary_len = 16 - computed[row] % 16
                if count >= boundary_len:
                    torch.testing.assert_close(
                        recovered.kv_cache[1][block_table[row, computed[row] // 16]],
                        native.kv_cache[1][native_indices[row, boundary_len - 1]],
                        atol=0,
                        rtol=0,
                    )
                computed[row] += count
                recover_indices[row] = block_table[row, computed[row] // 16]
            torch.testing.assert_close(
                recovered.kv_cache[1][recover_indices[row]],
                native.kv_cache[1][native_indices[row, count - 1]],
                atol=0,
                rtol=0,
            )
            native_conv = native.kv_cache[0][native_indices[row, 0]]
            recover_conv = recovered.kv_cache[0][recover_indices[row]]
            if not dim_first:
                native_conv = native_conv.transpose(-1, -2)
                recover_conv = recover_conv.transpose(-1, -2)
            torch.testing.assert_close(
                recover_conv[..., :3],
                native_conv[..., count - 1 : count + 2],
                atol=0,
                rtol=0,
            )
        previous_accepted = accepted_gpu
