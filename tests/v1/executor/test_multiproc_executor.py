# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import weakref
from types import SimpleNamespace
from typing import Any

import pytest

from vllm.v1.executor.multiproc_executor import WorkerProc


class _ExitWorkerLoop(RuntimeError):
    pass


class _RpcPayload:
    pass


class _PayloadLifetimeCheckingQueue:
    def __init__(self) -> None:
        self.payload_ref: weakref.ReferenceType[_RpcPayload] | None = None
        self.dequeue_count = 0

    def dequeue(self, *, indefinite: bool):
        assert indefinite
        self.dequeue_count += 1
        if self.dequeue_count == 1:
            payload = _RpcPayload()
            self.payload_ref = weakref.ref(payload)
            return "consume", (payload,), {}, None

        assert self.payload_ref is not None
        assert self.payload_ref() is None
        raise _ExitWorkerLoop


def test_worker_rpc_payload_released_before_next_dequeue():
    queue = _PayloadLifetimeCheckingQueue()
    worker_proc: Any = WorkerProc.__new__(WorkerProc)
    worker_proc.rpc_broadcast_mq = queue
    worker_proc.rank = 0
    worker_proc.worker = SimpleNamespace(consume=lambda payload: payload)
    worker_proc.handle_output = lambda output: None

    with pytest.raises(_ExitWorkerLoop):
        worker_proc.worker_busy_loop()

    assert queue.dequeue_count == 2


def test_execute_worker_rpc_returns_worker_exception():
    def fail():
        raise RuntimeError("test error")

    worker_proc: Any = WorkerProc.__new__(WorkerProc)
    worker_proc.rank = 0
    worker_proc.worker = SimpleNamespace(fail=fail)
    outputs: list[Any] = []
    worker_proc.handle_output = outputs.append

    worker_proc._execute_worker_rpc(("fail", (), {}, None))

    assert len(outputs) == 1
    assert isinstance(outputs[0], RuntimeError)
    assert str(outputs[0]) == "test error"


class _ResponseQueue:
    def __init__(self, value):
        self.ready = False
        self.value = value
        self.timeouts = []

    def is_ready(self):
        return self.ready

    def dequeue(self, timeout=None):
        assert self.ready, "poll must not consume an unavailable worker response"
        self.ready = False
        self.timeouts.append(timeout)
        return self.value


def _rpc_executor(queues):
    from collections import deque
    from unittest.mock import Mock

    return SimpleNamespace(
        rpc_broadcast_mq=Mock(),
        response_mqs=queues,
        futures_queue=deque(),
        is_failed=False,
    )


def test_poll_waits_for_all_eight_aggregator_replies():
    from unittest.mock import Mock

    from vllm.v1.executor.multiproc_executor import MultiprocExecutor

    queues = [
        _ResponseQueue((WorkerProc.ResponseStatus.SUCCESS, rank)) for rank in range(8)
    ]
    executor = _rpc_executor(queues)
    aggregator = Mock()
    aggregator.aggregate.side_effect = lambda outputs, output_rank: sum(outputs)
    future = MultiprocExecutor.collective_rpc(
        executor,
        "execute_model",
        non_block=True,
        unique_reply_rank=0,
        kv_output_aggregator=aggregator,
    )
    assert not future.poll()
    for queue in queues[:-1]:
        queue.ready = True
    assert not future.poll()
    assert not future.done()
    assert all(queue.timeouts == [0.0] for queue in queues[:-1])
    assert not queues[-1].timeouts
    aggregator.aggregate.assert_not_called()
    queues[-1].ready = True
    assert future.poll()
    assert future.result() == sum(range(8))
    aggregator.aggregate.assert_called_once_with(list(range(8)), output_rank=0)


def test_poll_respects_fifo_and_does_not_drain_later_futures():
    from collections import deque

    from vllm.v1.executor.multiproc_executor import FutureWrapper

    pending = deque()
    ready = [False, True]
    consumed = []
    first = FutureWrapper(
        pending, lambda: consumed.append(0), is_ready=lambda: ready[0]
    )
    second = FutureWrapper(
        pending, lambda: consumed.append(1), is_ready=lambda: ready[1]
    )
    assert not second.poll()
    assert consumed == []
    ready[0] = True
    assert first.poll()
    assert consumed == [0]
    assert not second.done()
    assert second.poll()
    assert consumed == [0, 1]


@pytest.mark.parametrize("failure_rank", [0, 1])
@pytest.mark.parametrize("timeout", [None, 0.0])
def test_poll_preserves_worker_error_with_unready_sibling(failure_rank, timeout):
    from vllm.v1.executor.multiproc_executor import MultiprocExecutor

    queue = _ResponseQueue((WorkerProc.ResponseStatus.FAILURE, "test failure"))
    queue.ready = True
    queues = [
        _ResponseQueue((WorkerProc.ResponseStatus.SUCCESS, None)) for _ in range(2)
    ]
    queues[failure_rank] = queue
    future = MultiprocExecutor.collective_rpc(
        _rpc_executor(queues), "execute_model", timeout=timeout, non_block=True
    )
    assert future.poll()
    with pytest.raises(RuntimeError, match="Worker failed with error 'test failure'"):
        future.result()


def test_poll_expired_deadline_fails_after_caching_ready_replies(monkeypatch):
    from vllm.v1.executor import multiproc_executor

    monkeypatch.setattr(multiproc_executor.time, "monotonic", lambda: 100.0)
    queues = [_ResponseQueue((WorkerProc.ResponseStatus.SUCCESS, 1)) for _ in range(8)]
    queues[0].ready = True
    future = multiproc_executor.MultiprocExecutor.collective_rpc(
        _rpc_executor(queues), "execute_model", timeout=1.0, non_block=True
    )
    assert not future.poll()
    monkeypatch.setattr(multiproc_executor.time, "monotonic", lambda: 102.0)
    assert future.poll()
    with pytest.raises(TimeoutError, match="RPC call to execute_model timed out"):
        future.result()
    assert queues[0].timeouts == [0.0]
    assert all(not queue.timeouts for queue in queues[1:])


def test_blocking_result_still_drains_fifo_without_readiness():
    from collections import deque

    from vllm.v1.executor.multiproc_executor import FutureWrapper

    pending = deque()
    consumed = []
    first = FutureWrapper(pending, lambda: consumed.append(0), is_ready=lambda: False)
    second = FutureWrapper(pending, lambda: consumed.append(1), is_ready=lambda: False)
    assert not second.poll()
    second.result()
    assert first.done()
    assert consumed == [0, 1]


def test_poll_ready_reply_after_deadline_keeps_zero_timeout(monkeypatch):
    from vllm.v1.executor import multiproc_executor

    monkeypatch.setattr(multiproc_executor.time, "monotonic", lambda: 100.0)
    queue = _ResponseQueue((WorkerProc.ResponseStatus.SUCCESS, 7))
    future = multiproc_executor.MultiprocExecutor.collective_rpc(
        _rpc_executor([queue]), "execute_model", timeout=1.0, non_block=True
    )
    monkeypatch.setattr(multiproc_executor.time, "monotonic", lambda: 102.0)
    queue.ready = True
    assert future.poll()
    assert future.result() == [7]
    assert queue.timeouts == [0.0]


def test_blocking_result_resumes_cached_partial_replies_including_none():
    from vllm.v1.executor.multiproc_executor import MultiprocExecutor

    queues = [
        _ResponseQueue((WorkerProc.ResponseStatus.SUCCESS, value))
        for value in [None, "second"]
    ]
    queues[0].ready = True
    future = MultiprocExecutor.collective_rpc(
        _rpc_executor(queues), "execute_model", non_block=True
    )
    assert not future.poll()
    assert queues[0].timeouts == [0.0]
    queues[1].ready = True
    assert future.result() == [None, "second"]
    assert queues[0].timeouts == [0.0]
    assert queues[1].timeouts == [None]


@pytest.mark.parametrize("failure_source", ["readiness", "response", "aggregate"])
def test_poll_propagates_earlier_fifo_failure_without_reading_later_rpc(failure_source):
    from collections import deque
    from unittest.mock import Mock

    from vllm.v1.executor.multiproc_executor import FutureWrapper

    pending = deque()
    error = RuntimeError("earlier RPC failed")
    readiness = Mock(return_value=True)
    response = Mock(return_value=None)
    aggregate = Mock(return_value=None)
    {"readiness": readiness, "response": response, "aggregate": aggregate}[
        failure_source
    ].side_effect = error
    first = FutureWrapper(pending, response, aggregate, is_ready=readiness)
    later_readiness = Mock(return_value=True)
    later_response = Mock()
    second = FutureWrapper(pending, later_response, is_ready=later_readiness)

    assert second.poll()
    for future in [first, second]:
        with pytest.raises(RuntimeError, match="earlier RPC failed") as raised:
            future.result()
        assert raised.value is error
    later_readiness.assert_not_called()
    later_response.assert_not_called()
    assert not pending


def test_poll_without_readiness_callback_fails_instead_of_spinning():
    from collections import deque

    from vllm.v1.executor.multiproc_executor import FutureWrapper

    future = FutureWrapper(deque(), lambda: 7)
    with pytest.raises(RuntimeError, match="no non-blocking readiness callback"):
        future.poll()
    assert future.result() == 7


def _kv_control_executor(connections):
    return SimpleNamespace(
        world_size=len(connections),
        local_world_size=len(connections),
        workers=[
            SimpleNamespace(rank=rank, kv_control_pipe=pipe)
            for rank, pipe in enumerate(connections)
        ],
        _kv_control_enabled=True,
        _kv_recv_ranks={},
        _kv_control_error=None,
    )


def test_restore_control_requires_all_cp8_ranks_without_duplicate_credit():
    """Control completion must not enter the normal ordered RPC reply stream."""
    import multiprocessing

    from vllm.v1.executor.multiproc_executor import MultiprocExecutor
    from vllm.v1.outputs import KVConnectorOutput

    pairs = [multiprocessing.Pipe() for _ in range(8)]
    executor = _kv_control_executor([parent for parent, _ in pairs])
    try:
        for _, child in pairs[:-1]:
            child.send(
                (
                    WorkerProc.ResponseStatus.SUCCESS,
                    KVConnectorOutput(finished_recving={"req"}),
                )
            )
        assert MultiprocExecutor.poll_async_kv_loads(executor) is None
        pairs[0][1].send(
            (
                WorkerProc.ResponseStatus.SUCCESS,
                KVConnectorOutput(finished_recving={"req"}, invalid_block_ids={3}),
            )
        )
        output = MultiprocExecutor.poll_async_kv_loads(executor)
        assert output.finished_recving is None
        assert output.invalid_block_ids == {3}
        pairs[-1][1].send(
            (
                WorkerProc.ResponseStatus.SUCCESS,
                KVConnectorOutput(finished_recving={"req"}),
            )
        )
        assert MultiprocExecutor.poll_async_kv_loads(executor).finished_recving == {
            "req"
        }
        assert executor._kv_recv_ranks == {}
        assert MultiprocExecutor.poll_async_kv_loads(executor) is None
    finally:
        for pair in pairs:
            for pipe in pair:
                pipe.close()


def test_restore_control_failure_remains_terminal():
    import multiprocessing

    from vllm.v1.executor.multiproc_executor import MultiprocExecutor

    parent, child = multiprocessing.Pipe()
    executor = _kv_control_executor([parent])
    try:
        child.send((WorkerProc.ResponseStatus.FAILURE, "restore failed"))
        for _ in range(2):
            with pytest.raises(
                RuntimeError, match="KV control worker 0: restore failed"
            ):
                MultiprocExecutor.poll_async_kv_loads(executor)
        with pytest.raises(RuntimeError, match="restore failed"):
            MultiprocExecutor.submit_async_kv_loads(executor, object())
    finally:
        parent.close()
        child.close()


def test_restore_control_progresses_while_forward_rpc_is_blocked():
    import multiprocessing
    import threading

    from vllm.v1.outputs import KVConnectorOutput

    parent, child = multiprocessing.Pipe()
    forward_entered, release_forward = threading.Event(), threading.Event()
    submitted = []
    responses = []

    def forward():
        forward_entered.set()
        assert release_forward.wait(5)
        return "forward result"

    def submit(metadata):
        submitted.append(metadata)
        return {"req"}

    worker: Any = WorkerProc.__new__(WorkerProc)
    worker.rank = 0
    worker.worker = SimpleNamespace(
        forward=forward,
        start_async_kv_loads=submit,
        poll_async_kv_loads=lambda: KVConnectorOutput(finished_recving={"req"}),
    )
    worker.handle_output = responses.append
    worker.start_kv_control_thread(child)
    main = threading.Thread(
        target=worker._execute_worker_rpc, args=(("forward", (), {}, None),)
    )
    main.start()
    try:
        assert forward_entered.wait(5)
        parent.send("restore metadata")
        assert parent.poll(5)
        status, output = parent.recv()
        assert status == WorkerProc.ResponseStatus.SUCCESS
        assert output.finished_recving == {"req"}
        assert submitted == ["restore metadata"]
        assert responses == []
        assert main.is_alive()
    finally:
        release_forward.set()
        main.join(5)
        worker._kv_control_stop.set()
        worker._kv_control_thread.join(5)
        parent.close()
        child.close()
    assert responses == ["forward result"]
    assert not worker._kv_control_thread.is_alive()


def test_restore_control_owner_waits_for_load_and_exits_on_parent_eof():
    import multiprocessing
    from unittest.mock import Mock

    parent, child = multiprocessing.Pipe()
    worker: Any = WorkerProc.__new__(WorkerProc)
    worker.worker = SimpleNamespace(
        start_async_kv_loads=Mock(), poll_async_kv_loads=Mock()
    )
    worker.start_kv_control_thread(child)
    parent.close()
    worker._kv_control_thread.join(5)
    child.close()
    assert not worker._kv_control_thread.is_alive()
    worker.worker.start_async_kv_loads.assert_not_called()
    worker.worker.poll_async_kv_loads.assert_not_called()


@pytest.mark.parametrize(
    "support, expected", [([False] * 8, False), ([True] * 8, True)]
)
def test_initialize_restore_control_requires_uniform_worker_support(support, expected):
    from unittest.mock import Mock

    from vllm.v1.executor.multiproc_executor import MultiprocExecutor

    executor = _kv_control_executor([object()] * 8)
    executor.collective_rpc = Mock(return_value=support)
    assert MultiprocExecutor.initialize_async_kv_loads(executor) is expected
    executor.collective_rpc.assert_called_once_with("initialize_async_kv_loads")
    executor.collective_rpc.return_value = [True, *([False] * 7)]
    with pytest.raises(RuntimeError, match="differs across workers"):
        MultiprocExecutor.initialize_async_kv_loads(executor)


def test_restore_control_worker_exception_does_not_use_rpc_reply_queue():
    import multiprocessing
    from unittest.mock import Mock

    parent, child = multiprocessing.Pipe()
    worker: Any = WorkerProc.__new__(WorkerProc)
    worker.worker = SimpleNamespace(
        start_async_kv_loads=Mock(side_effect=RuntimeError("restore submission failed"))
    )
    worker.handle_output = Mock()
    worker.start_kv_control_thread(child)
    try:
        parent.send("metadata")
        assert parent.poll(5)
        status, error = parent.recv()
        assert status == WorkerProc.ResponseStatus.FAILURE
        assert error == "restore submission failed"
        worker._kv_control_thread.join(5)
        assert not worker._kv_control_thread.is_alive()
        worker.handle_output.assert_not_called()
    finally:
        worker._kv_control_stop.set()
        worker._kv_control_thread.join(5)
        parent.close()
        child.close()


def test_worker_shutdown_stops_control_before_destroying_device(monkeypatch):
    import multiprocessing
    from unittest.mock import Mock

    from vllm.v1.executor import multiproc_executor as module

    parent, child = multiprocessing.Pipe()
    worker: Any = WorkerProc.__new__(WorkerProc)
    worker.worker = SimpleNamespace(shutdown=Mock())
    worker.rpc_broadcast_mq = Mock()
    worker.worker_response_mq = Mock()
    worker.start_kv_control_thread(child)

    def destroy():
        assert not worker._kv_control_thread.is_alive()
        assert child.closed

    monkeypatch.setattr(module, "destroy_model_parallel", destroy)
    monkeypatch.setattr(module, "destroy_distributed_environment", destroy)
    try:
        worker.shutdown()
        worker.worker.shutdown.assert_called_once_with()
    finally:
        parent.close()
        child.close()


def test_restore_control_preserves_pending_across_empty_polls():
    import multiprocessing
    from unittest.mock import Mock

    from vllm.v1.outputs import KVConnectorOutput

    parent, child = multiprocessing.Pipe()
    worker: Any = WorkerProc.__new__(WorkerProc)
    poll = Mock(
        side_effect=[None, None, None, KVConnectorOutput(finished_recving={"req"})]
    )
    worker.worker = SimpleNamespace(
        start_async_kv_loads=Mock(return_value={"req"}),
        poll_async_kv_loads=poll,
    )
    worker.start_kv_control_thread(child)
    try:
        parent.send("metadata")
        assert parent.poll(5)
        status, output = parent.recv()
        assert status == WorkerProc.ResponseStatus.SUCCESS
        assert output.finished_recving == {"req"}
        assert poll.call_count == 4
        worker.worker.start_async_kv_loads.assert_called_once_with("metadata")
        assert worker._kv_control_thread.is_alive()
        assert not parent.poll()
    finally:
        worker._kv_control_stop.set()
        worker._kv_control_thread.join(5)
        parent.close()
        child.close()
