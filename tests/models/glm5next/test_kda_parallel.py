# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from vllm.config.kda_parallel import get_kda_layer_tp, kda_tp_over_pcp
from vllm.model_executor.model_loader import weight_utils


def config(tp=1, pcp=8, dcp=1, kda_tp=8, dp=1, sp=False):
    return SimpleNamespace(
        additional_config={},
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp,
            kda_layer_tp=kda_tp,
            data_parallel_size=dp,
            use_sequence_parallel_moe=sp,
            prefill_context_parallel_size=pcp,
            decode_context_parallel_size=dcp,
        ),
    )


def test_kda_effective_tp_matches_on_prefill_and_decode():
    assert kda_tp_over_pcp(config())
    assert get_kda_layer_tp(config()) == 8
    assert get_kda_layer_tp(config(tp=2, pcp=4)) == 8
    assert not kda_tp_over_pcp(config(tp=8, pcp=1))
    assert get_kda_layer_tp(config(tp=8, pcp=1)) == 8
    assert get_kda_layer_tp(config(kda_tp=None)) == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dcp": 8},
        {"kda_tp": 3},
    ],
)
def test_kda_parallel_rejects_unimplemented_layouts(kwargs):
    with pytest.raises(ValueError):
        get_kda_layer_tp(config(**kwargs))


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
            cfg = config(tp=tp_size, pcp=pcp_size)
            spec = parallel.get_kda_parallel_spec(cfg)
            assert spec.kda_layer_tp == 8
            assert spec.kda_layer_rank == tp_rank * pcp_size + pcp_rank


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


@pytest.mark.parametrize(
    "backend,eager,pcp",
    [
        ("mccl", True, 8),
        ("auto", False, 8),
        ("auto", True, 2),
    ],
)
def test_mapped_collective_fallback(monkeypatch, backend, eager, pcp):
    from vllm.models.glm5next.common.parallel import get_kda_pcp_collectives
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "is_musa", lambda: True)
    cfg = config(pcp=pcp, kda_tp=pcp)
    cfg.additional_config["kda_pcp_collective"] = backend
    cfg.model_config = SimpleNamespace(enforce_eager=eager)
    assert get_kda_pcp_collectives(cfg, 4096) is None


@pytest.mark.parametrize(
    "tp,pcp,dp,kda_tp,axis,pools",
    [
        (2, 4, 1, 8, "pcp", 1),
        (1, 8, 1, 8, "pcp", 1),
        (2, 4, 1, 2, "tp", 1),
        (1, 1, 8, 8, "dp", 8),
        (2, 1, 4, 8, "dp", 4),
        (8, 1, 1, 8, "tp", 1),
        (2, 4, 1, None, "tp", 1),
    ],
)
def test_explicit_kda_tp_layout_and_pool(tp, pcp, dp, kda_tp, axis, pools):
    from vllm.config.kda_parallel import get_kda_request_windows, kda_tp_over_dp

    cfg = config(tp=tp, pcp=pcp, dp=dp, kda_tp=kda_tp)
    assert kda_tp_over_pcp(cfg) == (axis == "pcp")
    assert kda_tp_over_dp(cfg) == (axis == "dp")
    assert get_kda_layer_tp(cfg) == (kda_tp or tp)
    assert get_kda_request_windows(cfg) == pools


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kda_tp": 0},
        {"kda_tp": -8},
        {"tp": 2, "kda_tp": 3},
        {"tp": 2, "pcp": 4, "kda_tp": 4},
        {"pcp": 1, "dp": 8, "kda_tp": 4},
        {"pcp": 4, "dp": 2, "kda_tp": 8},
        {"kda_tp": 8, "dcp": 2},
        {"pcp": 1, "dp": 8, "kda_tp": 8, "sp": True},
    ],
)
def test_explicit_kda_tp_rejects_unsupported_groups(kwargs):
    with pytest.raises(ValueError):
        get_kda_layer_tp(config(**kwargs))


@pytest.mark.parametrize("layout", ["pcp", "dp"])
def test_kda_degree_and_rank_are_independent_of_request_windows(monkeypatch, layout):
    from vllm.models.glm5next.common import parallel

    cfg = config(pcp=8 if layout == "pcp" else 1, dp=8 if layout == "dp" else 1)
    monkeypatch.setattr(
        parallel, "get_tp_group", lambda: SimpleNamespace(world_size=1, rank_in_group=0)
    )
    for rank in range(8):
        group = SimpleNamespace(world_size=8, rank_in_group=rank)
        monkeypatch.setattr(parallel, "get_pcp_group", lambda: group)
        monkeypatch.setattr(parallel, "get_ep_group", lambda: group)
        spec = parallel.get_kda_parallel_spec(cfg)
        assert spec.kda_layer_tp == 8
        assert spec.kda_layer_rank == rank
        assert spec.token_layout == layout
        assert spec.request_windows == (8 if layout == "dp" else 1)


def test_dp_decode_metadata_preserves_owner_windows_and_padding():
    from vllm.models.glm5next.common.kda import Glm5NextLinearAttention

    group = SimpleNamespace(all_gather=lambda tensor, dim: tensor.repeat(8))
    layer = SimpleNamespace(
        kda_parallel=SimpleNamespace(group=group),
        request_windows=8,
        kda_layer_tp=8,
        kv_cache=(None, torch.empty(32, 1)),
    )
    meta = SimpleNamespace(
        num_prefills=0,
        num_spec_decodes=0,
        non_spec_state_indices_tensor=torch.tensor([1, 2], dtype=torch.int32),
    )
    result = Glm5NextLinearAttention._dp_decode_metadata(layer, meta, 3)
    expected = (
        torch.tensor([1, 2, 0]).repeat(8) + torch.arange(8).repeat_interleave(3) * 4
    )
    torch.testing.assert_close(result.non_spec_state_indices_tensor, expected.int())
    torch.testing.assert_close(result.non_spec_query_start_loc, torch.arange(25).int())
    assert result.num_decodes == 24
    assert result.num_prefills == 0
    assert Glm5NextLinearAttention._dp_decode_metadata(layer, meta, 3) is result


@pytest.mark.parametrize("prefills,spec", [(1, 0), (0, 1)])
def test_dp_decode_metadata_rejects_non_decode_batches(prefills, spec):
    from vllm.models.glm5next.common.kda import Glm5NextLinearAttention

    meta = SimpleNamespace(num_prefills=prefills, num_spec_decodes=spec)
    with pytest.raises(AssertionError):
        Glm5NextLinearAttention._dp_decode_metadata(SimpleNamespace(), meta, 1)
