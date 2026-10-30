# SPDX-License-Identifier: Apache-2.0
"""CPU tests for intake progress and bounded restore-only RPCs during result waits."""

from collections import deque
from concurrent.futures import Future
from contextlib import nullcontext
from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock

from vllm.v1.engine.core import EngineCore, EngineCoreProc
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT


def make_core():
    core = EngineCore.__new__(EngineCore)
    core.progress_async_loads = True
    core.batch_queue_size = 2
    core.batch_queue = deque(maxlen=3)
    core.scheduler = Mock()
    core.scheduler.has_requests.return_value = True
    core.scheduler.update_from_output.return_value = {}
    core.scheduler.schedule_kv_loads.return_value = SimpleNamespace(
        total_num_scheduled_tokens=0, pending_structured_output_tokens=False
    )
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
