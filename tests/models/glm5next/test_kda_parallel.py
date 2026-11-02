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
        monkeypatch.setattr(parallel, "get_pcp_group", lambda group=group: group)
        monkeypatch.setattr(parallel, "get_ep_group", lambda group=group: group)
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
        shared_state_pool=False,
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


def test_dp_decode_metadata_global_ids_take_no_window_offset():
    """Shared pool: ids are already global and padding uses the null block 0."""
    from vllm.models.glm5next.common.kda import Glm5NextLinearAttention

    group = SimpleNamespace(all_gather=lambda tensor, dim: tensor.repeat(8))
    layer = SimpleNamespace(
        kda_parallel=SimpleNamespace(group=group),
        request_windows=8,
        kda_layer_tp=8,
        shared_state_pool=True,
        kv_cache=(None, torch.empty(257, 1)),
    )
    meta = SimpleNamespace(
        num_prefills=0,
        num_spec_decodes=0,
        non_spec_state_indices_tensor=torch.tensor([33, 34], dtype=torch.int32),
    )
    result = Glm5NextLinearAttention._dp_decode_metadata(layer, meta, 3)
    expected = torch.tensor([33, 34, 0]).repeat(8)
    torch.testing.assert_close(result.non_spec_state_indices_tensor, expected.int())
    torch.testing.assert_close(result.non_spec_query_start_loc, torch.arange(25).int())
    assert result.num_decodes == 24


def test_dp_decode_metadata_refreshes_persistent_graph_buffers():
    """A later build updates the storage captured by an earlier FULL graph."""
    from vllm.models.glm5next.common.kda import globalize_gdn_metadata_for_dp

    group = SimpleNamespace(all_gather=lambda tensor, dim: tensor.repeat(8))
    out_state = torch.full((24,), -1, dtype=torch.int32)
    out_qsl = torch.full((25,), -1, dtype=torch.int32)

    def build(state):
        meta = SimpleNamespace(
            num_prefills=0,
            num_spec_decodes=0,
            non_spec_state_indices_tensor=torch.tensor(state, dtype=torch.int32),
        )
        return globalize_gdn_metadata_for_dp(
            meta,
            group=group,
            request_windows=8,
            shared_state_pool=True,
            pool_num_blocks=257,
            max_tokens=3,
            out_state_indices=out_state,
            out_qsl=out_qsl,
        )

    first = build([33, 34])
    second = build([41])
    expected = torch.tensor([41, 0, 0]).repeat(8).int()
    torch.testing.assert_close(second.non_spec_state_indices_tensor, expected)
    torch.testing.assert_close(first.non_spec_state_indices_tensor, expected)
    assert second.non_spec_state_indices_tensor.data_ptr() == out_state.data_ptr()
    assert second.non_spec_query_start_loc.data_ptr() == out_qsl.data_ptr()
    torch.testing.assert_close(second.non_spec_query_start_loc, torch.arange(25).int())


def test_dp_decode_metadata_eager_overflow_skips_graph_buffers():
    """Batches beyond the capture width use fresh eager metadata tensors."""
    from vllm.models.glm5next.common.kda import globalize_gdn_metadata_for_dp

    meta = SimpleNamespace(
        num_prefills=0,
        num_spec_decodes=0,
        non_spec_state_indices_tensor=torch.tensor([33, 34], dtype=torch.int32),
    )
    out_state = torch.full((16,), -1, dtype=torch.int32)
    out_qsl = torch.full((17,), -1, dtype=torch.int32)
    result = globalize_gdn_metadata_for_dp(
        meta,
        group=SimpleNamespace(all_gather=lambda tensor, dim: tensor.repeat(8)),
        request_windows=8,
        shared_state_pool=True,
        pool_num_blocks=257,
        max_tokens=3,
        out_state_indices=out_state,
        out_qsl=out_qsl,
    )

    assert result.non_spec_state_indices_tensor.data_ptr() != out_state.data_ptr()
    assert result.non_spec_query_start_loc.data_ptr() != out_qsl.data_ptr()
    torch.testing.assert_close(out_state, torch.full_like(out_state, -1))
    torch.testing.assert_close(out_qsl, torch.full_like(out_qsl, -1))
    assert result.num_decodes == 24


def test_dp_spec_metadata_uses_fixed_capacity_real_then_padding_layout():
    from vllm.models.glm5next.common.kda import globalize_gdn_metadata_for_dp
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

    token_rows, request_rows, windows, slots = 5, 3, 4, 2
    num_global_requests = request_rows * windows
    num_global_tokens = token_rows * windows
    meta = GDNAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=2,
        num_spec_decode_tokens=3,
        num_actual_tokens=3,
        spec_query_start_loc=torch.tensor([0, 1, 3], dtype=torch.int32),
        spec_state_indices_tensor=torch.tensor(
            [[11, 111], [12, 112]], dtype=torch.int32
        ),
        num_accepted_tokens=torch.tensor([4, 1], dtype=torch.int32),
    )
    out_state = torch.full((num_global_requests * slots,), -1, dtype=torch.int32)
    out_qsl = torch.full((num_global_requests + 1,), -1, dtype=torch.int32)
    out_tokens = torch.full((num_global_tokens,), -1, dtype=torch.int64)
    out_accepted = torch.full((num_global_requests,), -1, dtype=torch.int32)
    out_count = torch.zeros(1, dtype=torch.int32)
    group = SimpleNamespace(
        all_gather=lambda tensor, dim: tensor.repeat(
            (windows,) + (1,) * (tensor.ndim - 1)
        )
    )

    result = globalize_gdn_metadata_for_dp(
        meta,
        group=group,
        request_windows=windows,
        shared_state_pool=True,
        pool_num_blocks=257,
        max_tokens=token_rows,
        max_requests=request_rows,
        out_state_indices=out_state,
        out_qsl=out_qsl,
        out_spec_token_indx=out_tokens,
        out_spec_accepted=out_accepted,
        out_spec_num_valid=out_count,
    )

    assert result.num_spec_decodes == num_global_requests
    assert result.num_actual_tokens == num_global_tokens
    assert result.spec_state_indices_tensor.shape == (num_global_requests, slots)
    assert result.spec_state_indices_tensor.data_ptr() == out_state.data_ptr()
    assert result.spec_query_start_loc.data_ptr() == out_qsl.data_ptr()
    assert result.spec_token_indx.data_ptr() == out_tokens.data_ptr()
    assert result.num_accepted_tokens.data_ptr() == out_accepted.data_ptr()
    torch.testing.assert_close(
        result.spec_query_start_loc.diff(),
        torch.tensor([1, 2, 0]).repeat(windows).int(),
    )
    torch.testing.assert_close(
        result.num_accepted_tokens,
        torch.tensor([4, 1, 0]).repeat(windows).int(),
    )
    expected_tokens = torch.tensor(
        [0, 1, 2, 5, 6, 7, 10, 11, 12, 15, 16, 17, 3, 4, 8, 9, 13, 14, 18, 19]
    )
    torch.testing.assert_close(result.spec_token_indx.cpu(), expected_tokens)
    torch.testing.assert_close(result._kda_dp_spec_num_valid, torch.tensor([12]).int())
    expected_state = torch.tensor([[11, 111], [12, 112], [0, 0]])
    torch.testing.assert_close(
        result.spec_state_indices_tensor, expected_state.repeat(windows, 1).int()
    )


def test_dp_plan_shares_spec_metadata_but_not_group_state(monkeypatch):
    from vllm.config import kda_parallel
    from vllm.models.glm5next.common import kda, parallel
    from vllm.models.glm5next.common.kda import globalize_gdn_metadata_for_dp_config
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

    windows, rows, token_rows = 4, 3, 5
    group = SimpleNamespace(
        all_gather=lambda tensor, dim: tensor.repeat(
            (windows,) + (1,) * (tensor.ndim - 1)
        )
    )

    class Plan:
        def __init__(self):
            self.max_tokens = token_rows
            self.max_requests = rows
            self.global_mode = 1
            self.spec_common = None
            self.spec_accepted_semantics = None
            self.gather_calls = 0

        def all_gather(self, tensor):
            self.gather_calls += 1
            return group.all_gather(tensor, 0)

        def track_consumers(self, *tensors):
            pass

    plan = Plan()
    vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=8),
    )
    kv_cache_spec = SimpleNamespace(
        page_size_bytes=4,
        max_memory_usage_bytes=lambda _: 4,
    )
    monkeypatch.setattr(kda, "is_forward_context_available", lambda: False)
    monkeypatch.setattr(kda_parallel, "get_kda_request_windows", lambda _: windows)
    monkeypatch.setattr(kda_parallel, "kda_mamba_shared_pool", lambda _: True)
    monkeypatch.setattr(
        parallel, "get_kda_parallel_spec", lambda _: SimpleNamespace(group=group)
    )

    def make_meta(state):
        return GDNAttentionMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decodes=0,
            num_decode_tokens=0,
            num_spec_decodes=2,
            num_spec_decode_tokens=3,
            num_actual_tokens=3,
            spec_query_start_loc=torch.tensor([0, 1, 3], dtype=torch.int32),
            spec_state_indices_tensor=torch.tensor(state, dtype=torch.int32),
            num_accepted_tokens=torch.tensor([4, 1], dtype=torch.int32),
        )

    def build(meta, plan, state, qsl, tokens, accepted, count):
        meta.kda_dp_plan = plan
        return globalize_gdn_metadata_for_dp_config(
            vllm_config,
            kv_cache_spec,
            meta,
            state,
            qsl,
            None,
            None,
            tokens,
            accepted,
            count,
        )

    state_a = torch.full((rows * windows,), -1, dtype=torch.int32)
    qsl_a = torch.full((rows * windows + 1,), -1, dtype=torch.int32)
    tokens_a = torch.full((token_rows * windows,), -1, dtype=torch.int64)
    accepted_a = torch.full((rows * windows,), -1, dtype=torch.int32)
    count_a = torch.zeros(1, dtype=torch.int32)
    first = build(
        make_meta([[11], [12]]), plan, state_a, qsl_a, tokens_a, accepted_a, count_a
    )
    assert plan.gather_calls == 3
    assert plan.spec_common is not None
    assert plan.spec_accepted_semantics == "forward_offsets"

    state_b = torch.full_like(state_a, -1)
    qsl_b = torch.full_like(qsl_a, -1)
    tokens_b = torch.full_like(tokens_a, -1)
    accepted_b = torch.full_like(accepted_a, -1)
    count_b = torch.zeros_like(count_a)
    second = build(
        make_meta([[21], [22]]), plan, state_b, qsl_b, tokens_b, accepted_b, count_b
    )

    assert plan.gather_calls == 4
    torch.testing.assert_close(second.spec_query_start_loc, first.spec_query_start_loc)
    torch.testing.assert_close(second.spec_token_indx, first.spec_token_indx)
    torch.testing.assert_close(second.num_accepted_tokens, first.num_accepted_tokens)
    torch.testing.assert_close(second._kda_dp_spec_num_valid, count_a)
    assert second.spec_state_indices_tensor.data_ptr() == state_b.data_ptr()
    assert second.spec_query_start_loc.data_ptr() == qsl_b.data_ptr()
    assert second.spec_token_indx.data_ptr() == tokens_b.data_ptr()
    assert second.num_accepted_tokens.data_ptr() == accepted_b.data_ptr()
    expected_state = torch.tensor([[21], [22], [0]]).repeat(windows, 1).int()
    torch.testing.assert_close(second.spec_state_indices_tensor, expected_state)


def test_dp_recoverssm_commit_gathers_local_accepted_counts():
    from vllm.models.kimi_k3.nvidia.kda_metadata import (
        KDARecoverSSMCommitMetadata,
        KimiK3KDAMetadata,
    )

    rows, windows = 5, 4
    num_global = rows * windows
    calls = []

    def commit(*args, **kwargs):
        calls.append((args, kwargs))

    group = SimpleNamespace(
        all_gather=lambda tensor, dim: tensor.repeat(
            (windows,) + (1,) * (tensor.ndim - 1)
        )
    )
    global_meta = KimiK3KDAMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=num_global,
        num_spec_decode_tokens=12,
        num_actual_tokens=num_global,
        recoverssm_commit=KDARecoverSSMCommitMetadata(
            state_indices=torch.ones(num_global, 1, dtype=torch.int32),
            query_start_loc=torch.zeros(num_global + 1, dtype=torch.int32),
            request_indices=None,
            align=None,
        ),
        recoverssm_context=SimpleNamespace(commit=commit),
    )
    global_meta._kda_dp_spec_group = group
    global_meta._kda_dp_spec_rows = rows
    local_meta = KimiK3KDAMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=2,
        num_spec_decode_tokens=3,
        num_actual_tokens=3,
    )
    local_meta._kda_global_metadata = global_meta

    local_meta.commit_recoverssm_state(torch.tensor([2, 1], dtype=torch.int32))
    global_meta.recoverssm_context._dp_last_commit = (
        global_meta.recoverssm_commit,
        group,
        rows,
        num_global,
    )
    from vllm.models.kimi_k3.nvidia.kda_metadata import (
        flush_recoverssm_dp_commit,
    )

    flush_recoverssm_dp_commit(global_meta.recoverssm_context)

    assert len(calls) == 1
    args, kwargs = calls[0]
    torch.testing.assert_close(
        args[0], torch.tensor([2, 1, 0, 0, 0]).repeat(windows).int()
    )
    assert args[1].shape == (num_global,)
    assert args[2].shape == (num_global + 1,)
    assert kwargs == {
        "request_indices": None,
        "block_table": None,
        "num_computed_tokens": None,
        "mamba_block_size": None,
    }


def test_dp_row_compaction_is_stable_and_restorable():
    from vllm.models.glm5next.common.kda import compact_gdn_rows_for_dp

    state = torch.tensor([11, 12, 0, 0, 0, 0, 21, 0, 0, 31, 32, 33], dtype=torch.int32)
    out_order = torch.full((12,), -1, dtype=torch.int64)
    out_count = torch.zeros(1, dtype=torch.int32)
    packed, order, count = compact_gdn_rows_for_dp(
        state, state > 0, out_order=out_order, out_num_valid=out_count
    )

    expected_order = torch.tensor([0, 1, 6, 9, 10, 11, 2, 3, 4, 5, 7, 8])
    torch.testing.assert_close(order.cpu(), expected_order)
    torch.testing.assert_close(count.cpu(), torch.tensor([6], dtype=torch.int32))
    torch.testing.assert_close(packed[:6], state[expected_order[:6]])
    torch.testing.assert_close(packed[6:], torch.zeros_like(packed[6:]))
    restored = torch.empty_like(state)
    restored.index_copy_(0, order, packed)
    torch.testing.assert_close(restored, state)
    assert order.data_ptr() == out_order.data_ptr()
    assert count.data_ptr() == out_count.data_ptr()


def test_dp_global_metadata_compaction_preserves_static_shape_and_owner_rows():
    from vllm.models.glm5next.common.kda import globalize_gdn_metadata_for_dp

    meta = SimpleNamespace(
        num_prefills=0,
        num_spec_decodes=0,
        non_spec_state_indices_tensor=torch.tensor([11, 12], dtype=torch.int32),
    )
    out_state = torch.empty(12, dtype=torch.int32)
    out_qsl = torch.empty(13, dtype=torch.int32)
    out_order = torch.empty(12, dtype=torch.int64)
    out_count = torch.empty(1, dtype=torch.int32)
    result = globalize_gdn_metadata_for_dp(
        meta,
        group=SimpleNamespace(all_gather=lambda tensor, dim: tensor.repeat(4)),
        request_windows=4,
        shared_state_pool=True,
        pool_num_blocks=129,
        max_tokens=3,
        out_state_indices=out_state,
        out_qsl=out_qsl,
        out_order=out_order,
        out_num_valid=out_count,
    )

    assert result.num_decodes == result.num_actual_tokens == 12
    torch.testing.assert_close(result.non_spec_query_start_loc, torch.arange(13).int())
    torch.testing.assert_close(result._kda_dp_num_valid, torch.tensor([8]).int())
    restored = torch.empty_like(result.non_spec_state_indices_tensor)
    restored.index_copy_(0, result._kda_dp_order, result.non_spec_state_indices_tensor)
    expected = torch.tensor([11, 12, 0]).repeat(4).int()
    torch.testing.assert_close(restored, expected)


def shared_pool_config(rank=0, max_num_seqs=16, global_num_seqs=None, dp=8):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            kda_layer_tp=8,
            data_parallel_size=dp,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            use_sequence_parallel_moe=False,
            kda_mamba_shared_pool=True,
            kda_mamba_global_num_seqs=global_num_seqs,
            data_parallel_rank=rank,
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
    )


@pytest.mark.parametrize("rank", [0, 1, 7])
def test_shared_pool_leases_partition_the_global_pool(rank):
    from vllm.config.kda_parallel import (
        get_kda_mamba_global_num_seqs,
        get_kda_mamba_lease,
        kda_mamba_shared_pool,
    )

    cfg = shared_pool_config(rank=rank)
    assert kda_mamba_shared_pool(cfg)
    assert get_kda_mamba_global_num_seqs(cfg) == 128
    start, num_blocks = get_kda_mamba_lease(cfg, blocks_per_request=2)
    assert (start, num_blocks) == (1 + rank * 32, 32)
    # Block 0 is the pool-wide null block; leases tile the rest exactly.
    assert start + num_blocks == 1 + (rank + 1) * 32


def test_small_global_bound_switches_to_admission_leases():
    from vllm.config.kda_parallel import (
        get_kda_mamba_lease,
        kda_mamba_admission_leases,
    )

    cfg = shared_pool_config(global_num_seqs=64)
    assert kda_mamba_admission_leases(cfg)
    assert get_kda_mamba_lease(cfg, 2) is None
    assert not kda_mamba_admission_leases(shared_pool_config())


def test_shared_pool_requires_kda_over_dp():
    from vllm.config.kda_parallel import kda_mamba_shared_pool

    cfg = shared_pool_config()
    cfg.parallel_config.kda_layer_tp = None
    assert not kda_mamba_shared_pool(cfg)


def test_dp_prefill_ignores_spec_only_dp_dummy():
    from vllm.models.glm5next.common.kda import globalize_gdn_metadata_for_dp
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

    meta = GDNAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=1,
        num_spec_decode_tokens=2,
        num_actual_tokens=2,
        spec_query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        spec_state_indices_tensor=torch.tensor([[7, 8]], dtype=torch.int32),
        num_accepted_tokens=torch.tensor([1], dtype=torch.int32),
    )
    other_tensors = (
        torch.tensor([3, 0], dtype=torch.int32),
        torch.tensor([2, 0], dtype=torch.int32),
        torch.tensor([0, 0], dtype=torch.int32),
    )
    calls = iter(other_tensors)
    group = SimpleNamespace(
        all_gather=lambda tensor, dim: torch.cat([tensor, next(calls)])
    )
    kwargs = dict(
        group=group,
        request_windows=2,
        shared_state_pool=True,
        pool_num_blocks=8,
        max_tokens=2,
        global_mode=2,
    )
    with pytest.raises(AssertionError, match="mixed prefill and decode"):
        globalize_gdn_metadata_for_dp(meta, **kwargs)

    meta.kda_dp_dummy = True
    result = globalize_gdn_metadata_for_dp(meta, **kwargs)

    assert result.num_prefills == 1
    assert result.num_prefill_tokens == 2
    assert result.num_spec_decodes == 0
    torch.testing.assert_close(
        result.non_spec_state_indices_tensor, torch.tensor([3], dtype=torch.int32)
    )
    torch.testing.assert_close(
        result.non_spec_query_start_loc, torch.tensor([0, 2], dtype=torch.int32)
    )
    assert result.non_spec_token_indx.shape == (4,)
    torch.testing.assert_close(
        result.non_spec_token_indx[:2], torch.tensor([2, 3], dtype=torch.int64)
    )
    assert int(result._kda_dp_prefill_num_valid[0]) == 2


@pytest.mark.parametrize("prefills,decodes,spec", [(1, 0, 0), (0, 1, 1)])
def test_dp_decode_metadata_rejects_non_decode_batches(prefills, decodes, spec):
    from vllm.models.glm5next.common.kda import Glm5NextLinearAttention

    meta = SimpleNamespace(
        num_prefills=prefills, num_decodes=decodes, num_spec_decodes=spec
    )
    layer = SimpleNamespace(
        kda_parallel=SimpleNamespace(group=None),
        request_windows=8,
        shared_state_pool=True,
        kv_cache=(None, torch.empty(257, 1)),
    )
    with pytest.raises(AssertionError):
        Glm5NextLinearAttention._dp_decode_metadata(layer, meta, 1)
