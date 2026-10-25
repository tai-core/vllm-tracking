# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from vllm.config.kda_parallel import kda_tensor_parallel_size, kda_tp_over_pcp
from vllm.model_executor.model_loader import weight_utils


def config(tp=1, pcp=8, dcp=1, mode="tp_pcp"):
    return SimpleNamespace(
        additional_config={"kda_parallel_mode": mode},
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp,
            prefill_context_parallel_size=pcp,
            decode_context_parallel_size=dcp,
        ),
    )


def test_kda_effective_tp_matches_on_prefill_and_decode():
    assert kda_tp_over_pcp(config())
    assert kda_tensor_parallel_size(config()) == 8
    assert kda_tensor_parallel_size(config(tp=2, pcp=4)) == 8
    assert not kda_tp_over_pcp(config(tp=8, pcp=1))
    assert kda_tensor_parallel_size(config(tp=8, pcp=1)) == 8
    assert kda_tensor_parallel_size(config(mode="tp")) == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dcp": 8},
        {"mode": "unknown"},
    ],
)
def test_kda_parallel_rejects_unimplemented_layouts(kwargs):
    with pytest.raises(ValueError):
        kda_tensor_parallel_size(config(**kwargs))


@pytest.mark.parametrize("axis", [0, 2])
def test_kda_scalar_parameter_loader_uses_head_owner(monkeypatch, axis):
    monkeypatch.setattr(weight_utils, "get_tensor_model_parallel_rank", lambda: 0)
    shape = [1, 1, 1]
    shape[axis] = 64
    weight = torch.arange(64, dtype=torch.float32).reshape(shape)
    for rank in range(8):
        target_shape = list(shape)
        target_shape[axis] = 8
        target = torch.nn.Parameter(torch.empty(target_shape))
        weight_utils.sharded_weight_loader(axis, tp_rank=rank)(target, weight)
        torch.testing.assert_close(target, weight.narrow(axis, rank * 8, 8))


@pytest.mark.parametrize("tp_size,pcp_size", [(1, 8), (2, 4), (4, 2), (8, 1)])
def test_kda_projection_head_order_matches_pd(monkeypatch, tp_size, pcp_size):
    from vllm.models.glm5next.common import parallel

    for tp_rank in range(tp_size):
        for pcp_rank in range(pcp_size):
            tp = SimpleNamespace(world_size=tp_size, rank_in_group=tp_rank)
            pcp = SimpleNamespace(world_size=pcp_size, rank_in_group=pcp_rank)
            monkeypatch.setattr(parallel, "get_tp_group", lambda tp=tp: tp)
            monkeypatch.setattr(parallel, "get_pcp_group", lambda pcp=pcp: pcp)
            spec = parallel.get_kda_parallel_spec(config(tp=tp_size, pcp=pcp_size))
            assert spec.size == 8
            assert spec.rank == tp_rank * pcp_size + pcp_rank


@pytest.mark.parametrize("sequence_parallel", [False, True])
@pytest.mark.parametrize("mhc,is_mtp", [(False, False), (True, False), (True, True)])
def test_kda_output_reduces_tp_once(monkeypatch, sequence_parallel, mhc, is_mtp):
    from vllm.models.glm5next.common import model

    calls = []
    layout = SimpleNamespace(
        gather_to_global=lambda x: x.repeat(4, 1),
        reduce_scatter_to_local=lambda x: x.chunk(4)[0] * 4,
    )
    monkeypatch.setattr(
        model, "get_forward_context", lambda: SimpleNamespace(pcp_token_layout=layout)
    )

    def reduce_tp(x):
        calls.append("tp")
        return x * 2

    monkeypatch.setattr(model, "tensor_model_parallel_all_reduce", reduce_tp)

    class Attention:
        tp_over_pcp = True

        def __call__(self, hidden_states, positions):
            return hidden_states * 3

    layer = SimpleNamespace(
        layer_kind="kda",
        self_attn=Attention(),
        is_sequence_parallel=sequence_parallel,
        mhc=mhc,
        is_mtp_layer=is_mtp,
    )
    x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    out = model.Glm5NextDecoderLayer._run_attention(layer, x, torch.arange(2))
    deferred = sequence_parallel and mhc and not is_mtp
    torch.testing.assert_close(out, x * (12 if deferred else 24))
    assert calls == ([] if deferred else ["tp"])


@pytest.mark.parametrize("backend,eager,pcp", [
    ("mccl", True, 8), ("auto", False, 8), ("auto", True, 2),
])
def test_mapped_collective_fallback(monkeypatch, backend, eager, pcp):
    from vllm.models.glm5next.common.parallel import get_kda_pcp_collectives
    from vllm.platforms import current_platform
    monkeypatch.setattr(current_platform, "is_musa", lambda: True)
    cfg = config(pcp=pcp)
    cfg.additional_config["kda_pcp_collective"] = backend
    cfg.model_config = SimpleNamespace(enforce_eager=eager)
    assert get_kda_pcp_collectives(cfg, 4096) is None
