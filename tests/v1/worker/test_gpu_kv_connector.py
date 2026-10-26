# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import vllm.v1.worker.gpu.kv_connector as kv_connector_module
from vllm.config import KVTransferConfig
from vllm.v1.worker.gpu.kv_connector import ActiveKVConnector


def _make_connector(
    monkeypatch: pytest.MonkeyPatch,
    events: list[str],
) -> ActiveKVConnector:
    backend = Mock()
    backend.requires_pre_state_load = False
    backend.handle_preemptions.side_effect = lambda _: events.append("handle")
    backend.bind_connector_metadata.side_effect = lambda _: events.append("bind")
    backend.start_load_kv.side_effect = lambda _: events.append("start")
    backend.wait_for_save.side_effect = lambda: events.append("wait")
    backend.wait_for_pre_state_load.side_effect = lambda: events.append("state_wait")
    backend.get_finished.side_effect = lambda _: (set(), set())
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

    kv_config = KVTransferConfig(
        kv_connector="NixlConnector",
        kv_role="kv_consumer",
        kv_buffer_device="cpu",
    )
    connector = ActiveKVConnector(  # type: ignore[arg-type]
        SimpleNamespace(kv_transfer_config=kv_config), {}
    )
    events.clear()
    return connector


def _scheduler_output(has_sync_kv_loads: bool, num_tokens: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        kv_connector_metadata=object(),
        finished_req_ids=set(),
        has_sync_kv_loads=has_sync_kv_loads,
        total_num_scheduled_tokens=num_tokens,
    )


@pytest.mark.parametrize("has_sync_kv_loads", [False, True])
def test_load_start_phase(
    monkeypatch: pytest.MonkeyPatch,
    has_sync_kv_loads: bool,
):
    events: list[str] = []
    connector = _make_connector(monkeypatch, events)
    output = _scheduler_output(has_sync_kv_loads)

    connector.pre_forward(output)  # type: ignore[arg-type]
    assert events == (
        ["handle", "bind", "start"] if has_sync_kv_loads else ["handle", "bind"]
    )

    connector.post_forward(set())
    assert events == ["handle", "bind", "start", "wait", "clear"]


@pytest.mark.parametrize("has_sync_kv_loads", [False, True])
def test_no_forward_starts_load_once(
    monkeypatch: pytest.MonkeyPatch, has_sync_kv_loads: bool
):
    events: list[str] = []
    connector = _make_connector(monkeypatch, events)
    connector.kv_connector.requires_pre_state_load = True
    output = _scheduler_output(has_sync_kv_loads, num_tokens=0)

    connector.start_pre_state_load(output)
    connector.wait_for_pre_state_load()
    connector.no_forward(output)  # type: ignore[arg-type]

    assert events == ["handle", "bind", "start", "clear"]


def test_early_load_starts_once_and_resets_before_continuation(
    monkeypatch: pytest.MonkeyPatch,
):
    events: list[str] = []
    connector = _make_connector(monkeypatch, events)
    connector.kv_connector.requires_pre_state_load = True
    monkeypatch.setattr(
        kv_connector_module, "is_forward_context_available", lambda: False
    )
    context = Mock(return_value=nullcontext())
    monkeypatch.setattr(kv_connector_module, "set_forward_context", context)
    output = _scheduler_output(True)

    connector.start_pre_state_load(output)
    connector.start_pre_state_load(output)
    events.append("prepare_inputs")
    connector.wait_for_pre_state_load()
    events.append("read_state")
    connector.pre_forward(output)
    connector.post_forward(set())

    context.assert_called_once_with(None, connector.vllm_config)
    assert events == [
        "handle",
        "bind",
        "start",
        "prepare_inputs",
        "state_wait",
        "read_state",
        "wait",
        "clear",
    ]

    # The following chunk must bind fresh metadata and use the normal lifecycle.
    events.clear()
    continuation = _scheduler_output(False)
    connector.start_pre_state_load(continuation)
    connector.wait_for_pre_state_load()
    connector.pre_forward(continuation)
    connector.post_forward(set())
    assert events == ["handle", "bind", "start", "wait", "clear"]


@pytest.mark.parametrize(
    ("sync_load", "num_tokens", "requires_early"),
    [(False, 1, True), (True, 0, True), (True, 1, False)],
)
def test_early_load_requires_opt_in_and_scheduled_sync_tokens(
    monkeypatch: pytest.MonkeyPatch,
    sync_load: bool,
    num_tokens: int,
    requires_early: bool,
):
    events: list[str] = []
    connector = _make_connector(monkeypatch, events)
    connector.kv_connector.requires_pre_state_load = requires_early

    connector.start_pre_state_load(_scheduler_output(sync_load, num_tokens))
    connector.wait_for_pre_state_load()

    assert events == []


def test_disabled_connector_does_not_start_or_wait_for_early_load(
    monkeypatch: pytest.MonkeyPatch,
):
    events: list[str] = []
    connector = _make_connector(monkeypatch, events)
    connector.kv_connector.requires_pre_state_load = True
    monkeypatch.setattr(
        kv_connector_module.kv_transfer_state, "_KV_CONNECTOR_AGENT", None
    )
    connector.set_disabled(True)

    connector.start_pre_state_load(_scheduler_output(True))
    connector.wait_for_pre_state_load()

    assert events == []


def test_early_load_wait_failure_propagates(monkeypatch: pytest.MonkeyPatch):
    connector = _make_connector(monkeypatch, [])
    connector.kv_connector.requires_pre_state_load = True
    connector.kv_connector.wait_for_pre_state_load.side_effect = RuntimeError(
        "load failed"
    )
    connector.start_pre_state_load(_scheduler_output(True))

    with pytest.raises(RuntimeError, match="load failed"):
        connector.wait_for_pre_state_load()
