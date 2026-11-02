# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import vllm.v1.worker.gpu.kv_connector as kv_connector_module
from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorTransferResults
from vllm.v1.worker.gpu.kv_connector import ActiveKVConnector


def _make_connector(
    monkeypatch: pytest.MonkeyPatch,
    events: list[str],
    dp_decode_bench: bool = False,
    dp_mooncake: bool = False,
) -> ActiveKVConnector:
    backend = Mock()
    backend.handle_preemptions.side_effect = lambda _: events.append("handle")
    backend.bind_connector_metadata.side_effect = lambda _: events.append("bind")
    backend.start_load_kv.side_effect = lambda *_args, **_kwargs: events.append("start")
    backend.wait_for_save.side_effect = lambda: events.append("wait")
    backend.get_transfer_results.return_value = KVConnectorTransferResults()
    backend.get_block_ids_with_load_errors.return_value = set()
    backend.get_kv_connector_stats.return_value = None
    backend.get_kv_connector_kv_cache_events.return_value = None
    backend.build_connector_worker_meta.return_value = None
    backend.clear_connector_metadata.side_effect = lambda: events.append("clear")
    monkeypatch.setattr(kv_connector_module, "get_kv_transfer_group", lambda: backend)
    monkeypatch.setattr(
        kv_connector_module, "is_forward_context_available", lambda: True
    )
    monkeypatch.setattr(kv_connector_module, "get_forward_context", object)

    connector_name = "NixlConnector"
    if dp_decode_bench:
        connector_name = "DecodeBenchConnector"
    elif dp_mooncake:
        connector_name = "MooncakeConnector"
    kv_config = KVTransferConfig(
        kv_connector=connector_name,
        kv_role="kv_consumer",
        kv_buffer_device="cpu",
    )
    vllm_config = SimpleNamespace(kv_transfer_config=kv_config)
    if dp_decode_bench or dp_mooncake:
        vllm_config.parallel_config = SimpleNamespace(
            kda_layer_tp=8,
            kda_mamba_shared_pool=dp_mooncake,
            tensor_parallel_size=1,
            data_parallel_size=8,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            use_sequence_parallel_moe=False,
        )
    connector = ActiveKVConnector(vllm_config, {})
    events.clear()
    return connector


def _scheduler_output(has_sync_kv_loads: bool) -> SimpleNamespace:
    return SimpleNamespace(
        kv_connector_metadata=object(),
        finished_req_ids=set(),
        has_sync_kv_loads=has_sync_kv_loads,
    )


@pytest.mark.parametrize("has_sync_kv_loads", [False, True])
def test_load_start_phase(
    monkeypatch: pytest.MonkeyPatch,
    has_sync_kv_loads: bool,
):
    events: list[str] = []
    connector = _make_connector(monkeypatch, events)
    output = _scheduler_output(has_sync_kv_loads)

    request_indices = torch.tensor([3, 1])
    request_ids = ["first", "second"]
    input_batch = SimpleNamespace(
        idx_mapping=request_indices, req_ids=request_ids, num_tokens=5
    )
    attn_metadata = {"layer": object()}
    connector.pre_forward(output, input_batch, attn_metadata=attn_metadata)
    assert events == (
        ["handle", "bind", "start"] if has_sync_kv_loads else ["handle", "bind"]
    )

    connector.post_forward(set())
    assert events == ["handle", "bind", "start", "wait", "clear"]

    kwargs = connector.kv_connector.start_load_kv.call_args.kwargs
    assert kwargs["request_state_indices"] is request_indices
    assert kwargs["request_ids"] is request_ids
    assert kwargs["num_tokens"] == 5
    assert kwargs["scheduler_output"] is output
    assert kwargs["attn_metadata"] is attn_metadata

    # A subsequent step without a forward must not reuse the prior batch.
    connector.no_forward(_scheduler_output(False))
    assert connector.kv_connector.start_load_kv.call_count == 2
    assert connector.kv_connector.start_load_kv.call_args.kwargs == {}


def test_no_forward_starts_deferred_load_once(monkeypatch: pytest.MonkeyPatch):
    events: list[str] = []
    connector = _make_connector(monkeypatch, events)

    connector.no_forward(_scheduler_output(False))

    assert events == ["handle", "bind", "start", "wait", "clear"]


def test_dp_decode_bench_no_forward_leaves_fill_to_dummy(
    monkeypatch: pytest.MonkeyPatch,
):
    events: list[str] = []
    connector = _make_connector(monkeypatch, events, dp_decode_bench=True)

    connector.no_forward(_scheduler_output(False))
    assert events == []

    connector.set_disabled(True)
    dummy_output = _scheduler_output(False)
    dummy_output.kv_connector_metadata = None
    connector.pre_forward(dummy_output)

    assert events == ["handle", "bind", "start"]


def test_dp_mooncake_no_forward_defers_collectives_to_dummy(
    monkeypatch: pytest.MonkeyPatch,
):
    events: list[str] = []
    connector = _make_connector(monkeypatch, events, dp_mooncake=True)
    output = _scheduler_output(False)
    pending = output.kv_connector_metadata

    connector.no_forward(output)

    assert connector._dp_pending_metadata is pending
    assert events == ["wait", "clear"]

    def poll(_):
        events.append("poll")
        return KVConnectorTransferResults()

    connector.kv_connector.get_transfer_results.side_effect = poll
    dummy_output = _scheduler_output(False)
    dummy_output.kv_connector_metadata = None
    connector.pre_forward(dummy_output)

    assert connector._dp_pending_metadata is None
    connector.kv_connector.bind_connector_metadata.assert_called_once_with(pending)
    assert events == ["wait", "clear", "handle", "bind", "start", "poll"]


def test_dp_mooncake_no_forward_returns_previous_dummy_results(
    monkeypatch: pytest.MonkeyPatch,
):
    events: list[str] = []
    connector = _make_connector(monkeypatch, events, dp_mooncake=True)
    connector._dp_transfer_results = KVConnectorTransferResults(
        failed_recving={"req-1"}
    )

    output = connector.no_forward(_scheduler_output(False))

    assert output.kv_connector_output is not None
    assert output.kv_connector_output.failed_recving == {"req-1"}
    connector.kv_connector.get_transfer_results.assert_not_called()
