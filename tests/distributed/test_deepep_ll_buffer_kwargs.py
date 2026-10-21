# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import sys
from types import SimpleNamespace

import pytest
import torch

from vllm.distributed.device_communicators.all2all import DeepEPLLAll2AllManager


class LegacyBuffer:
    def __init__(self, group=None):
        pass

    @staticmethod
    def get_low_latency_rdma_size_hint(**kwargs):
        return 1024


class ShrinkBuffer(LegacyBuffer):
    def __init__(self, group=None, enable_shrink=False):
        pass


def buffer_kwargs(monkeypatch, buffer_type, fault_tolerance):
    monkeypatch.setitem(sys.modules, "deep_ep", SimpleNamespace(Buffer=buffer_type))
    manager = object.__new__(DeepEPLLAll2AllManager)
    manager.cpu_group = None
    manager.support_fault_tolerance = fault_tolerance
    return manager._make_all2all_kwargs(32, 4096, 8, 288, 36)


def test_legacy_buffer_omits_unsupported_shrink(monkeypatch):
    kwargs = buffer_kwargs(monkeypatch, LegacyBuffer, False)
    assert "enable_shrink" not in kwargs
    assert kwargs["low_latency_mode"] is True
    assert kwargs["num_qps_per_rank"] == 36


def test_legacy_buffer_rejects_requested_fault_tolerance(monkeypatch):
    with pytest.raises(ValueError, match="fault tolerance is unavailable"):
        buffer_kwargs(monkeypatch, LegacyBuffer, True)


@pytest.mark.parametrize("enabled", [False, True])
def test_supported_buffer_preserves_shrink_setting(monkeypatch, enabled):
    assert buffer_kwargs(monkeypatch, ShrinkBuffer, enabled)["enable_shrink"] is enabled


@pytest.mark.parametrize(
    "tokens,sp_size,pcp_size,expected",
    [
        (512, 4, 1, 128),
        (513, 4, 1, 129),
        (1, 4, 1, 1),
        (512, 1, 1, 512),
        (512, 4, 2, 512),
        (8192, 4, 1, 2048),
    ],
)
def test_ll_sp_capacity_reaches_buffer_and_experts(
    monkeypatch, tokens, sp_size, pcp_size, expected
):
    from vllm.model_executor.layers.fused_moe import all2all_utils

    expected = min(expected, 256)
    captured = {}

    def get_handle(kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        all2all_utils,
        "DeepEPLLPrepareAndFinalize",
        lambda handle, **kwargs: SimpleNamespace(**kwargs),
    )
    moe = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(
            use_passthrough_all2all=False,
            use_all2all_kernels=True,
        ),
        use_deepep_ht_kernels=False,
        use_deepep_ll_kernels=True,
        max_num_tokens=tokens,
        sp_size=sp_size,
        pcp_size=pcp_size,
        is_sequence_parallel=sp_size > 1,
        hidden_dim=4096,
        num_experts=288,
    )
    result = all2all_utils.maybe_make_prepare_finalize(
        moe,
        SimpleNamespace(quant_dtype=None),
        all2all_manager=SimpleNamespace(world_size=8, get_handle=get_handle),
    )
    assert captured["max_num_tokens_per_dp_rank"] == expected
    assert result.max_tokens_per_rank == expected


@pytest.mark.parametrize("local_tokens", [0, 2, 7])
def test_ll_chunks_keep_idle_ranks_in_collectives(monkeypatch, local_tokens):
    """Uneven ranks must issue identical rounds and preserve token order."""
    from vllm.model_executor.layers.fused_moe import modular_kernel as mk

    monkeypatch.setattr(mk, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(
        mk,
        "get_forward_context",
        lambda: SimpleNamespace(
            dp_metadata=SimpleNamespace(
                get_chunk_sizes_across_dp_rank=lambda: [0, 2, 7],
            )
        ),
    )
    impl = mk.FusedMoEKernelModularImpl(
        SimpleNamespace(max_num_tokens_per_rank=lambda: 3),
        SimpleNamespace(
            moe_config=SimpleNamespace(
                moe_parallel_config=SimpleNamespace(
                    use_deepep_ll_kernels=True,
                    dp_size=3,
                    use_ep=True,
                ),
            ),
        ),
    )
    calls = []
    x = torch.arange(local_tokens * 2).reshape(local_tokens, 2).float()
    ids = torch.arange(local_tokens).reshape(-1, 1)
    weights = ids.float() + 1
    shared = object()

    def apply_chunk(hidden_states, **kwargs):
        calls.append((hidden_states.shape[0], kwargs["shared_experts"]))
        if kwargs["shared_experts"] is not None:
            assert kwargs["shared_experts_input"] is x
        torch.testing.assert_close(
            kwargs["topk_weights"], kwargs["topk_ids"].float() + 1
        )
        return hidden_states + kwargs["topk_weights"]

    impl._apply = apply_chunk
    output = impl.apply(
        x, x, x, ids, weights, shared_experts=shared, shared_experts_input=x
    )
    torch.testing.assert_close(output, x + weights)
    assert [n for n, _ in calls] == [
        min(max(local_tokens - start, 0), 3) for start in (0, 3, 6)
    ]
    assert [s for _, s in calls] == [shared, None, None]
