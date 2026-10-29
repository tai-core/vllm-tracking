# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import tempfile

# Exercise the scheduler against the real socket broker: performance changes
# must preserve lease ownership, confirmed-token boundaries and cancellation.
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tests.v1.simple_kv_offload.test_scheduler import (
    BLOCK_SIZE,
    _alloc_and_register,
    _make_kv_cache_config,
    make_request,
    make_scheduler_output,
    simulate_store_completion,
)
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.shared_kv_offload import shared_pool as m
from vllm.v1.shared_kv_offload.shared_adapter import SharedOffloadScheduler


class PoolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.p = m.PoolState(self.tmp.name)
        for owner in ("a", "b"):
            self.op(owner, "hello", "namespace", 4)

    def tearDown(self):
        self.tmp.cleanup()

    def op(self, owner, op, *a):
        return self.p.dispatch(owner, op, a)

    def store(self, key):
        slot, gen = self.op("a", "reserve", 1)[0]
        self.op("a", "finish", [[slot, gen, {key: None}]])
        return slot, gen

    def test_uncommitted_invisible_and_cross_client_hit(self):
        slot, gen = self.op("a", "reserve", 1)[0]
        self.assertEqual(self.op("b", "acquire", ["x"]), [])
        self.op("a", "finish", [[slot, gen, {"x": None}]])
        self.assertEqual(self.op("b", "acquire", ["x"])[0][1:3], [slot, gen])

    def test_read_lease_prevents_reuse(self):
        slot, gen = self.store("x")
        self.op("b", "acquire", ["x"])
        refs = self.op("a", "reserve", 4)
        self.assertNotIn(slot, [r[0] for r in refs])
        self.assertEqual(len(refs), 2)
        self.op("b", "finish", [[slot, gen, {}]])
        self.assertEqual(self.op("a", "reserve", 1)[0][0], slot)

    def test_duplicate_store_does_not_replace_pinned_object(self):
        slot, gen = self.store("x")
        self.op("b", "acquire", ["x"])
        s, g = self.op("a", "reserve", 1)[0]
        self.op("a", "finish", [[s, g, {"x": None}]])
        self.assertEqual(self.op("a", "lookup", ["x"])[0][1], slot)
        self.assertEqual(self.p.stats["deduplicated"], 1)

    def test_disconnect_quarantines_live_dma(self):
        slot, gen = self.op("a", "reserve", 1)[0]
        self.p.disconnect("a")
        self.assertEqual(self.p.records[slot]["state"], "quarantined")
        self.assertNotIn(slot, [r[0] for r in self.op("b", "reserve", 5)])

    def test_flush_keeps_readers_and_invalidates_pending_store(self):
        slot, gen = self.store("x")
        self.op("b", "acquire", ["x"])
        s, g = self.op("a", "reserve", 1)[0]
        self.op("a", "flush")
        self.assertEqual(self.op("a", "lookup", ["x"]), [])
        self.op("a", "finish", [[s, g, {"y": None}]])
        self.assertEqual(self.op("a", "lookup", ["y"]), [])
        self.assertIn(slot, self.p.records)
        self.op("b", "finish", [[slot, gen, {}]])
        self.assertNotIn(slot, self.p.records)

    def test_stale_generation_cannot_release_new_object(self):
        s, g = self.store("x")
        self.op("a", "flush")
        self.op("a", "reserve", 3)
        with self.assertRaises(ValueError):
            self.op("a", "finish", [[s, g, {}]])

    def test_role_contracts_are_independent_but_peers_must_match(self):
        self.op("a", "contract", "scheduler", {"group": "summary"})
        self.op("b", "contract", "worker", {"group": "physical"})
        self.op("b", "contract", "scheduler", {"group": "summary"})
        with self.assertRaises(ValueError):
            self.op("b", "contract", "scheduler", {"group": "different"})
        with self.assertRaises(ValueError):
            self.op("b", "contract", "worker", {"group": "different"})

    def test_namespace_and_layout_mismatch_fail_closed(self):
        with self.assertRaises(ValueError):
            self.op("c", "hello", "different", 4)
        self.op("a", "attach", [["kv", 128]])
        with self.assertRaises(ValueError):
            self.op("b", "attach", [["kv", 256]])


@pytest.fixture
def shared_schedulers(tmp_path):
    server = m.PoolServer(str(tmp_path / "manager.sock"), str(tmp_path))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    clients = []

    def make(*, async_lookup=True):
        config = SimpleNamespace(
            kv_events_config=None,
            parallel_config=SimpleNamespace(
                world_size=1, decode_context_parallel_size=1
            ),
            model_config=SimpleNamespace(max_model_len=8192),
            speculative_config=None,
            max_in_flight_tokens=128,
            num_prefill_lookahead_tokens=0,
        )
        cache = _make_kv_cache_config(256)
        scheduler = SharedOffloadScheduler(
            config,
            cache,
            128 * 1024,
            BLOCK_SIZE,
            BLOCK_SIZE,
            {
                "cache_namespace": "tests",
                "manager_socket": str(tmp_path / "manager.sock"),
                "async_lookup": async_lookup,
            },
        )
        gpu = BlockPool(256, enable_caching=True, hash_block_size=BLOCK_SIZE)
        scheduler.bind_gpu_block_pool(gpu)
        clients.append(scheduler)
        return SimpleNamespace(scheduler=scheduler, gpu_block_pool=gpu)

    yield make, server.state
    for scheduler in clients:
        scheduler.reset()
        scheduler._client.close()
    server.shutdown()
    server.server_close()
    thread.join()


def _store_request(fix, request, count):
    scheduler = fix.scheduler
    blocks = _alloc_and_register(fix, request, count)
    scheduler.update_state_after_alloc(request, blocks, 0)
    output = make_scheduler_output(
        {request.request_id: count * BLOCK_SIZE},
        new_reqs={request.request_id: blocks.get_block_ids()},
    )
    meta = scheduler.build_connector_meta(output)
    if meta.store_event >= 0:
        simulate_store_completion(scheduler, meta.store_event)
    return meta


def test_shared_store_queries_only_new_confirmed_blocks(shared_schedulers):
    make, pool = shared_schedulers
    fix = make()
    request = make_request(num_blocks=64)
    _store_request(fix, request, 64)
    before = pool.stats.copy()
    output = make_scheduler_output(
        {request.request_id: 1}, cached_req_new_blocks={request.request_id: None}
    )
    meta = fix.scheduler.build_connector_meta(output)
    assert not meta.store_gpu_blocks
    assert pool.stats["admission_keys"] == before["admission_keys"] == 64
    assert pool.stats["rpc_admit"] == before["rpc_admit"] == 1
    assert fix.scheduler._reqs_to_store[request.request_id].num_stored_blocks == [64]


def test_shared_cached_store_advances_cursor_without_reserving(shared_schedulers):
    make, pool = shared_schedulers
    producer, consumer = make(), make()
    request = make_request(num_blocks=3)
    _store_request(producer, request, 3)
    before = pool.stats.copy()
    meta = _store_request(consumer, request, 3)
    assert not meta.store_gpu_blocks
    assert pool.stats["reserved"] == before["reserved"]
    assert consumer.scheduler._reqs_to_store[request.request_id].num_stored_blocks == [
        3
    ]
    output = make_scheduler_output(
        {request.request_id: 1}, cached_req_new_blocks={request.request_id: None}
    )
    before = pool.stats.copy()
    consumer.scheduler.build_connector_meta(output)
    assert pool.stats["admission_keys"] == before["admission_keys"]


@pytest.mark.parametrize("cancel", [False, True])
def test_shared_async_lookup_does_not_wait_and_releases_lease(
    shared_schedulers, cancel
):
    make, pool = shared_schedulers
    producer, consumer = make(), make()
    request = make_request(num_blocks=4)
    _store_request(producer, request, 4)
    scheduler = consumer.scheduler
    entered, release = threading.Event(), threading.Event()
    dispatch = pool.dispatch

    def delayed(owner, op, args):
        if op == "acquire":
            entered.set()
            assert release.wait(5)
        return dispatch(owner, op, args)

    with patch.object(pool, "dispatch", side_effect=delayed):
        try:
            assert scheduler.get_num_new_matched_tokens(request, 2 * BLOCK_SIZE) == (
                None,
                False,
            )
            assert entered.wait(2)
            future = scheduler._pending_lookups[request.request_id][1]
            assert scheduler.get_num_new_matched_tokens(request, 2 * BLOCK_SIZE) == (
                None,
                False,
            )
            if cancel:
                scheduler.request_finished(request, [])
            release.set()
            future.result(timeout=2)
            if not cancel:
                assert scheduler.get_num_new_matched_tokens(
                    request, 2 * BLOCK_SIZE
                ) == (2 * BLOCK_SIZE, True)
                # Allocation retries retain the same safe read lease.
                assert scheduler.get_num_new_matched_tokens(
                    request, 2 * BLOCK_SIZE
                ) == (2 * BLOCK_SIZE, True)
                scheduler.request_finished(request, [])
        finally:
            release.set()
    # Queue a barrier after any completion callback has released its lease.
    scheduler._client.submit("stats").result(timeout=2)
    assert pool.stats["acquire_keys"] == 2
    assert pool.stats["rpc_acquire"] == 1
    assert scheduler._client.call("stats")["held"] == 0


def test_shared_store_excludes_unconfirmed_tail(shared_schedulers):
    make, pool = shared_schedulers
    fix = make()
    request = make_request(num_blocks=6)
    blocks = _alloc_and_register(fix, request, 6)
    request.num_computed_tokens = 2 * BLOCK_SIZE
    fix.scheduler.update_state_after_alloc(request, blocks, 0)
    meta = fix.scheduler.build_connector_meta(
        make_scheduler_output(
            {request.request_id: BLOCK_SIZE},
            new_reqs={request.request_id: blocks.get_block_ids()},
        )
    )
    assert len(meta.store_gpu_blocks) == 2
    simulate_store_completion(fix.scheduler, meta.store_event)
    assert pool.stats["admission_keys"] == 2


@pytest.mark.parametrize("same_output", [False, True])
def test_shared_store_resumes_after_preemption(shared_schedulers, same_output):
    make, pool = shared_schedulers
    fix = make()
    request = make_request(num_blocks=3)
    _store_request(fix, request, 2)
    scheduler = fix.scheduler
    if not same_output:
        output = make_scheduler_output({})
        output.preempted_req_ids = {request.request_id}
        scheduler.build_connector_meta(output)
    blocks = _alloc_and_register(fix, request, 3)
    scheduler.update_state_after_alloc(request, blocks, 0)
    output = make_scheduler_output(
        {request.request_id: BLOCK_SIZE},
        cached_req_new_blocks={request.request_id: blocks.get_block_ids()},
    )
    output.scheduled_cached_reqs.resumed_req_ids.add(request.request_id)
    if same_output:
        output.preempted_req_ids = {request.request_id}
    meta = scheduler.build_connector_meta(output)
    assert meta.store_gpu_blocks == [blocks.blocks[0][2].block_id]
    simulate_store_completion(scheduler, meta.store_event)
    assert pool.stats["committed"] == 3


def test_shared_finish_stores_confirmed_tail(shared_schedulers):
    make, pool = shared_schedulers
    fix = make()
    request = make_request(num_blocks=4)
    blocks = _alloc_and_register(fix, request, 4)
    request.num_computed_tokens = 2 * BLOCK_SIZE
    scheduler = fix.scheduler
    scheduler.update_state_after_alloc(request, blocks, 0)
    meta = scheduler.build_connector_meta(
        make_scheduler_output(
            {request.request_id: BLOCK_SIZE},
            new_reqs={request.request_id: blocks.get_block_ids()},
        )
    )
    simulate_store_completion(scheduler, meta.store_event)
    request.num_computed_tokens = 4 * BLOCK_SIZE
    scheduler.request_finished_all_groups(request, blocks.get_block_ids())
    meta = scheduler.build_connector_meta(make_scheduler_output({}))
    assert len(meta.store_gpu_blocks) == 2
    simulate_store_completion(scheduler, meta.store_event)
    assert pool.stats["admission_keys"] == 4
    assert scheduler._client.call("stats")["held"] == 0


def test_shared_gpu_hit_skips_lookup(shared_schedulers):
    make, pool = shared_schedulers
    fix = make()
    request = make_request(num_blocks=4)
    assert fix.scheduler.get_num_new_matched_tokens(request, 4 * BLOCK_SIZE) == (
        0,
        False,
    )
    assert pool.stats["rpc_acquire"] == 0
