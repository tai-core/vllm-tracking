# SPDX-License-Identifier: Apache-2.0
"""CPU tests for intake progress and bounded restore-only RPCs during result waits."""

from collections import deque
from concurrent.futures import Future
from contextlib import nullcontext
from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.engine.core import EngineCore, EngineCoreProc
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, KVConnectorOutput


def make_core():
    core = EngineCore.__new__(EngineCore)
    core.progress_async_loads = True
    core.independent_async_loads = False
    core.batch_queue_size = 2
    core.batch_queue = deque(maxlen=3)
    core.scheduler = Mock()
    core.scheduler.has_requests.return_value = True
    core.scheduler.update_from_output.return_value = {}
    core.scheduler.schedule_kv_loads.return_value = SchedulerOutput.make_empty()
    core.model_executor = Mock()
    core.model_executor.execute_model.return_value = Future()
    core.is_ec_consumer = True
    core.is_pooling_model = True
    core.log_error_detail = lambda output: nullcontext()
    core.capture_iteration_details = lambda output: nullcontext()
    core._attach_iteration_details = Mock()
    core._process_aborts_queue = Mock()
    core._should_throttle_prefills = lambda: False
    for _ in range(2):
        future = Future()
        output = SimpleNamespace(total_num_scheduled_tokens=16)
        core.batch_queue.appendleft((future, output, future))
    return core


def test_full_compute_queue_can_submit_restore_without_blocking(monkeypatch):
    monkeypatch.setattr("vllm.v1.engine.core.time.sleep", lambda _: None)
    core = make_core()
    oldest = core.batch_queue[-1][0]
    assert core.step_with_batch_queue() == (None, False)
    core.scheduler.schedule_kv_loads.assert_called_once()
    core.scheduler.schedule.assert_not_called()
    assert len(core.batch_queue) == 3
    assert (
        sum(out.total_num_scheduled_tokens > 0 for _, out, _ in core.batch_queue) == 2
    )
    # The next pass is bounded: no fourth RPC and no premature result consumption.
    assert core.step_with_batch_queue() == (None, False)
    assert core.model_executor.execute_model.call_count == 1
    assert core.batch_queue[-1][0] is oldest
    # Input intake can run between polls while all compute slots remain occupied.
    core.has_work = lambda: True
    core.input_queue = Queue()
    core.input_queue.put(("new_request", "payload"))
    core._handle_client_request = Mock()
    EngineCoreProc._process_input_queue(core)
    core._handle_client_request.assert_called_once_with("new_request", "payload")
    oldest.set_result(EMPTY_MODEL_RUNNER_OUTPUT)
    assert core.step_with_batch_queue() == ({}, False)
    assert len(core.batch_queue) == 2


def test_scheduler_without_load_only_support_still_retires_results(monkeypatch):
    monkeypatch.setattr("vllm.v1.engine.core.time.sleep", lambda _: None)
    core = make_core()
    core.scheduler.schedule_kv_loads.return_value = None
    core.batch_queue[-1][0].set_result(EMPTY_MODEL_RUNNER_OUTPUT)
    assert core.step_with_batch_queue() == ({}, False)
    core.model_executor.execute_model.assert_not_called()
    assert len(core.batch_queue) == 1


def test_lookup_progress_continues_when_all_rpc_slots_are_full(monkeypatch):
    monkeypatch.setattr("vllm.v1.engine.core.time.sleep", lambda _: None)
    core = make_core()
    future = Future()
    core.batch_queue.appendleft(
        (future, SimpleNamespace(total_num_scheduled_tokens=0), future)
    )
    assert core.step_with_batch_queue() == (None, False)
    core.scheduler.progress_kv_lookups.assert_called_once()
    core.scheduler.schedule.assert_not_called()
    core.scheduler.schedule_kv_loads.assert_not_called()


def test_receive_completion_is_absorbed_while_forward_is_pending(monkeypatch):
    monkeypatch.setattr("vllm.v1.engine.core.time.sleep", lambda _: None)
    core = make_core()
    core.independent_async_loads = True
    core.scheduler.schedule_kv_loads.return_value = None
    received = KVConnectorOutput(finished_recving={"restored"})
    core.model_executor.poll_async_kv_loads.return_value = received
    core.scheduler.update_async_kv_loads.return_value = {}
    oldest = core.batch_queue[-1][0]
    assert core.step_with_batch_queue() == (None, False)
    core.scheduler.update_async_kv_loads.assert_called_once_with(received)
    assert not oldest.done()
    core.scheduler.update_from_output.assert_not_called()


def test_restore_submission_uses_independent_metadata_before_compute_rpc(monkeypatch):
    monkeypatch.setattr("vllm.v1.engine.core.time.sleep", lambda _: None)
    core = make_core()
    core.independent_async_loads = True
    core.model_executor.poll_async_kv_loads.return_value = None
    metadata, normal, loads = object(), object(), object()
    output = core.scheduler.schedule_kv_loads.return_value
    output.kv_connector_metadata = metadata
    connector = core.scheduler.get_kv_connector.return_value
    connector.split_async_loads.return_value = normal, loads
    calls = []
    core.model_executor.submit_async_kv_loads.side_effect = lambda value: calls.append(
        ("loads", value)
    )
    core.model_executor.execute_model.side_effect = lambda value, **_: (
        calls.append(("model", value.kv_connector_metadata)) or Future()
    )
    assert core.step_with_batch_queue() == (None, False)
    assert calls == [("loads", loads), ("model", normal)]


def test_control_failure_outputs_are_returned_without_retiring_forward():
    core = make_core()
    core.independent_async_loads = True
    received = KVConnectorOutput(invalid_block_ids={7}, finished_recving={"r"})
    core.model_executor.poll_async_kv_loads.return_value = received
    errors = {0: object()}
    core.scheduler.update_async_kv_loads.return_value = errors
    assert core.step_with_batch_queue() == (errors, False)
    assert len(core.batch_queue) == 2
    core.model_executor.execute_model.assert_not_called()


def test_control_head_retires_before_next_forward_or_grammar(monkeypatch):
    monkeypatch.setattr("vllm.v1.engine.core.time.sleep", lambda _: None)
    core = make_core()
    core.batch_queue.clear()
    control, previous = Future(), Future()
    control_out = SimpleNamespace(total_num_scheduled_tokens=0)
    previous_out = SimpleNamespace(total_num_scheduled_tokens=16)
    core.batch_queue.appendleft((control, control_out, control))
    core.batch_queue.appendleft((previous, previous_out, previous))
    # Pending control must not let a new grammar-dependent forward leap ahead.
    assert core.step_with_batch_queue() == (None, False)
    core.scheduler.schedule.assert_not_called()
    control.set_result(EMPTY_MODEL_RUNNER_OUTPUT)
    assert core.step_with_batch_queue() == ({}, False)
    core.scheduler.schedule.assert_not_called()
    assert core.batch_queue[-1][1] is previous_out
    core.is_pooling_model = False
    next_out = SimpleNamespace(
        total_num_scheduled_tokens=16, pending_structured_output_tokens=True
    )
    core.scheduler.schedule.return_value = next_out
    previous.set_result(EMPTY_MODEL_RUNNER_OUTPUT)
    seen = []
    core.scheduler.update_from_output.side_effect = lambda out, result: (
        seen.append(out) or {}
    )

    def grammar(out):
        assert out is next_out
        assert seen == [previous_out]
        return None

    core.scheduler.get_grammar_bitmask.side_effect = grammar
    core.check_for_draft_tokens = False
    core.model_executor.sample_tokens.return_value = Future()
    core.step_with_batch_queue()
    core.scheduler.get_grammar_bitmask.assert_called_once_with(next_out)


def independent_core():
    core = make_core()
    core.independent_async_loads = True
    core.model_executor.poll_async_kv_loads.return_value = None
    connector = core.scheduler.get_kv_connector.return_value
    connector.requires_worker_step.return_value = False
    normal, loads = object(), object()
    connector.split_async_loads.return_value = normal, loads
    output = core.scheduler.schedule_kv_loads.return_value
    output.kv_connector_metadata = object()
    core.scheduler.schedule.return_value = output
    return core, output, loads


def test_empty_restore_steps_submit_loads_without_using_model_fifo(monkeypatch):
    monkeypatch.setattr("vllm.v1.engine.core.time.sleep", lambda _: None)
    core, output, loads = independent_core()
    local_outputs = {0: object()}
    core.scheduler.update_from_output.return_value = local_outputs
    oldest = core.batch_queue[-1][0]
    for _ in range(3):
        assert core.step_with_batch_queue() == (local_outputs, False)
        assert len(core.batch_queue) == 2
    core.model_executor.submit_async_kv_loads.assert_called_with(loads)
    core.model_executor.execute_model.assert_not_called()
    core.scheduler.update_from_output.assert_called_with(
        output, EMPTY_MODEL_RUNNER_OUTPUT
    )
    # Repeated local stats/errors must not starve a newly completed forward.
    old_output = core.batch_queue[-1][1]
    oldest.set_result(EMPTY_MODEL_RUNNER_OUTPUT)
    assert core.step_with_batch_queue() == (local_outputs, False)
    assert len(core.batch_queue) == 1
    core.scheduler.update_from_output.assert_called_with(
        old_output, EMPTY_MODEL_RUNNER_OUTPUT
    )


def test_empty_restore_without_model_batches_delivers_local_outputs():
    core, output, _ = independent_core()
    core.batch_queue.clear()
    local_outputs = {0: object()}
    core.scheduler.update_from_output.return_value = local_outputs
    assert core.step_with_batch_queue() == (local_outputs, False)
    core.scheduler.update_from_output.assert_called_once_with(
        output, EMPTY_MODEL_RUNNER_OUTPUT
    )
    core.model_executor.execute_model.assert_not_called()
    assert not core.batch_queue


@pytest.mark.parametrize(
    "field,value",
    [
        ("finished_req_ids", {"finished"}),
        ("preempted_req_ids", {"preempted"}),
        ("free_encoder_mm_hashes", ["hash"]),
        ("scheduled_new_reqs", [object()]),
        ("scheduled_cached_reqs", SimpleNamespace(req_ids=["cached"])),
        ("scheduled_encoder_inputs", {"r": [0]}),
        ("new_block_ids_to_zero", [1]),
        ("kv_cache_block_copies", [object()]),
        ("has_sync_kv_loads", True),
        ("ec_connector_metadata", object()),
        ("ec_manager_metadata", object()),
        ("scheduled_spec_decode_tokens", {"r": [1]}),
        ("pending_structured_output_tokens", True),
        ("has_structured_output_requests", True),
    ],
)
def test_zero_token_worker_updates_are_not_discarded(monkeypatch, field, value):
    monkeypatch.setattr("vllm.v1.engine.core.time.sleep", lambda _: None)
    core, output, _ = independent_core()
    setattr(output, field, value)
    assert core.step_with_batch_queue() == (None, False)
    core.model_executor.execute_model.assert_called_once_with(output, non_block=True)
    assert len(core.batch_queue) == 3


def test_connector_work_keeps_zero_token_rpc(monkeypatch):
    monkeypatch.setattr("vllm.v1.engine.core.time.sleep", lambda _: None)
    core, output, _ = independent_core()
    connector = core.scheduler.get_kv_connector.return_value
    connector.requires_worker_step.return_value = True
    core.step_with_batch_queue()
    core.model_executor.execute_model.assert_called_once_with(output, non_block=True)
