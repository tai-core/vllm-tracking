# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adapt native offload matching and event handling to a shared physical pool."""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import mmap
import time
from pathlib import Path
from typing import cast

import torch

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (
    BlockHashWithGroupId,
    KVCacheBlock,
    make_block_hash_with_group_id,
    resolve_block_hashes,
)
from vllm.v1.shared_kv_offload.shared_pool import PoolClient
from vllm.v1.simple_kv_offload.copy_backend import DmaCopyBackend
from vllm.v1.simple_kv_offload.cuda_mem_ops import pin_tensor
from vllm.v1.simple_kv_offload.manager import SimpleCPUOffloadScheduler
from vllm.v1.simple_kv_offload.worker import SimpleCPUOffloadWorker


def namespace(config, cache_config, extra):
    """Do not include DP identity or allocation size in content compatibility."""
    identity = extra.get("cache_namespace")
    if not identity:
        raise ValueError("shared cache_namespace must identify the weight revision")
    # Scheduler contracts summarize grouped specs; workers may resolve runtime
    # cache dtypes and retain per-layer UniformTypeKVCacheSpecs. Compare each
    # role to its peers, not their deliberately different Python configs.
    payload = {"revision": identity, "schema": 2}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def role_contract(cache_config):
    from vllm.v1.core.kv_cache_utils import generate_scheduler_kv_cache_config

    canonical = generate_scheduler_kv_cache_config([cache_config])
    return json.loads(
        json.dumps(
            [dataclasses.asdict(g) for g in canonical.kv_cache_groups],
            sort_keys=True,
            default=str,
        )
    )


class _HashView:
    def __init__(self):
        self.rows = {}

    def get_one_block(self, key):
        return self.rows.get(key)


class SharedBlockPool(BlockPool):
    """Sparse per-client references; allocation and ownership live in the broker."""

    def __init__(self, client, capacity, hash_block_size):
        # Intentionally do not construct capacity Python KVCacheBlock objects.
        self.client = client
        self.num_gpu_blocks = capacity
        self.hash_block_size = hash_block_size
        self.enable_caching = True
        self.metrics_collector = None
        self.medium = "CPU"
        # BlockPool methods used here only index blocks by ID.
        self.blocks = {0: KVCacheBlock(0, pool=self, is_null=True)}  # type: ignore[assignment]
        self.null_block = self.blocks[0]
        self.cached_block_hash_to_block = _HashView()  # type: ignore[assignment]
        self.cached_block_hashes_by_block = {}
        self.kv_event_queue = []
        self.enable_kv_cache_events = False
        self._refs = {}  # slot -> generation, writing, aliases
        self._credits = None

    def get_cached_block(self, block_hash, group_ids):
        result = [
            self.cached_block_hash_to_block.get_one_block(
                make_block_hash_with_group_id(block_hash, g)
            )
            for g in group_ids
        ]
        return result if all(result) else None

    def _from_row(self, slot, generation, primary=None, tokens=None):
        if slot in self._refs:
            assert self._refs[slot]["generation"] == generation
            return self.blocks[slot]
        b = KVCacheBlock(slot, pool=self)
        if primary is not None:
            b.set_block_hash(BlockHashWithGroupId(bytes.fromhex(primary)), tokens)
        self.blocks[slot] = b
        self._refs[slot] = {
            "generation": generation,
            "writing": primary is None,
            "aliases": {},
            "leases": 0,
        }
        return b

    @contextlib.contextmanager
    def lookup_scope(self, keys, acquire, *, rows=None):
        view = cast(_HashView, self.cached_block_hash_to_block).rows
        assert not view
        leased = []
        try:
            if rows is None:
                rows = (
                    self.client.call(
                        "acquire" if acquire else "lookup", [k.hex() for k in keys]
                    )
                    if keys
                    else []
                )
            seen = set()
            for key, slot, generation, primary, tokens in rows:
                if acquire:
                    b = self._from_row(slot, generation, primary, tokens)
                    if slot not in seen:
                        # The broker holds one lease; local touches share it.
                        b.ref_cnt += 1
                        self._refs[slot]["leases"] += 1
                        leased.append(b)
                        seen.add(slot)
                else:
                    b = KVCacheBlock(slot)
                view[bytes.fromhex(key)] = b
            yield
        finally:
            view.clear()
            if leased:
                self.free_blocks(leased)

    @contextlib.contextmanager
    def admission(self, keys, limit):
        assert self._credits is None
        view = cast(_HashView, self.cached_block_hash_to_block).rows
        assert not view
        self._credits = []
        try:
            if keys:
                result = self.client.call("admit", [k.hex() for k in keys], limit)
                for key, slot, *_ in result["rows"]:
                    view[bytes.fromhex(key)] = KVCacheBlock(slot)
                for slot, generation in result["reserved"]:
                    b = self._from_row(slot, generation)
                    b.ref_cnt = 1
                    self._refs[slot]["leases"] = 1
                    self._credits.append(b)
            yield
        finally:
            view.clear()
            unused = self._credits
            self._credits = None
            if unused:
                self.free_blocks(unused)

    def get_new_blocks(self, n):
        assert self._credits is not None and n <= len(self._credits)
        result = self._credits[:n]
        del self._credits[:n]
        return result

    def get_num_free_blocks(self):
        if self._credits is not None:
            return len(self._credits)
        s = self.client.call("stats")
        return s["free"] + s["evictable"]

    def touch(self, blocks):
        for b in blocks:
            if b.is_null:
                continue
            assert b.ref_cnt > 0 and b.block_id in self._refs
            b.ref_cnt += 1

    def free_blocks(self, blocks):
        finished = []
        for b in list(blocks):
            if b.is_null:
                continue
            assert b.ref_cnt > 0
            b.ref_cnt -= 1
            if b.ref_cnt == 0:
                r = self._refs.pop(b.block_id)
                finished.extend(
                    [
                        [b.block_id, r["generation"], r["aliases"]]
                        for _ in range(r["leases"])
                    ]
                )
                self.blocks.pop(b.block_id)
                self.cached_block_hashes_by_block.pop(b.block_id, None)
        if finished:
            self.client.call("finish", finished)

    def _insert_block_hash(self, key, block, num_tokens=None):
        r = self._refs[block.block_id]
        assert r["writing"]
        r["aliases"][key.hex()] = num_tokens
        if block.block_hash is None:
            block.set_block_hash(key, num_tokens)
        self.cached_block_hashes_by_block.setdefault(block.block_id, set()).add(key)

    def take_events(self):
        # GPU KV events remain native. Shared HOST publication needs a broker
        # invalidation stream before it can be advertised to Dynamo.
        return []

    def reset_prefix_cache(self):
        self.client.call("flush")
        return True


class SharedOffloadScheduler(SimpleCPUOffloadScheduler):
    cpu_block_pool: SharedBlockPool

    def __init__(
        self, config, cache_config, capacity_bytes, block_size, hash_size, extra
    ):
        self._extra = extra
        self._client: PoolClient
        self._pending_lookups = {}
        self._async_lookup = extra.get("async_lookup", True)
        self._bytes_per_slot = (
            cache_config.kv_cache_tensors[0].size // cache_config.num_blocks
        )
        self._max_store = max(
            1,
            int(extra.get("max_store_bytes_per_step", 128 << 20))
            // self._bytes_per_slot,
        )
        self._store_rate = int(extra.get("store_bytes_per_second_per_rank", 2 << 30))
        self._store_credit = float(self._max_store)
        self._credit_at = time.monotonic()
        ns = namespace(config, cache_config, extra)

        def factory(n):
            self._client = PoolClient(extra["manager_socket"], ns, n)
            self._client.call("contract", "scheduler", role_contract(cache_config))
            return SharedBlockPool(self._client, n, hash_size)

        super().__init__(
            config,
            cache_config,
            capacity_bytes,
            scheduler_block_size=block_size,
            hash_block_size=hash_size,
            lazy_offload=False,
            block_pool_factory=factory,
        )
        self.enable_kv_cache_events = False

    def _discard_lookup(self, request_id):
        pending = self._pending_lookups.pop(request_id, None)
        if pending is None:
            return
        future = pending[1]
        if future.cancel():
            return
        client = self._client

        def release(done):
            rows = done.result()
            refs = {(row[1], row[2]) for row in rows}
            if refs:
                client.call(
                    "finish", [[slot, generation, {}] for slot, generation in refs]
                )

        future.add_done_callback(release)

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        req_id = request.request_id
        pending_hit = self._pending_cpu_hits.get(req_id)
        if (
            pending_hit is not None
            and pending_hit[2] == num_computed_tokens
            and not request.skip_reading_prefix_cache
        ):
            return pending_hit[1], True
        if (
            request.skip_reading_prefix_cache
            or num_computed_tokens % self.block_size
            or request.num_tokens - 1 <= num_computed_tokens
        ):
            self._discard_lookup(req_id)
            with self.cpu_block_pool.lookup_scope([], True):
                return super().get_num_new_matched_tokens(request, num_computed_tokens)
        hashes = request.block_hashes[num_computed_tokens // self.hash_block_size :]
        signature = (num_computed_tokens, len(hashes), hashes[-1] if hashes else None)
        pending = self._pending_lookups.get(req_id)
        if pending is not None and pending[0] != signature:
            self._discard_lookup(req_id)
            pending = None
        if self._async_lookup and hashes:
            if pending is None:
                wire_keys = [
                    make_block_hash_with_group_id(h, g).hex()
                    for h in hashes
                    for g in self.prefix_cacheable_group_ids
                ]
                self._pending_lookups[req_id] = (
                    signature,
                    self._client.submit("acquire", wire_keys),
                )
                return None, False
            if not pending[1].done():
                return None, False
            rows = pending[1].result()
            del self._pending_lookups[req_id]
            with self.cpu_block_pool.lookup_scope([], True, rows=rows):
                return super().get_num_new_matched_tokens(request, num_computed_tokens)
        keys = [
            make_block_hash_with_group_id(h, g)
            for h in hashes
            for g in self.prefix_cacheable_group_ids
        ]
        with self.cpu_block_pool.lookup_scope(keys, True):
            return super().get_num_new_matched_tokens(request, num_computed_tokens)

    def request_finished(self, request, block_ids):
        self._discard_lookup(request.request_id)
        return super().request_finished(request, block_ids)

    def reset(self):
        for req_id in list(self._pending_lookups):
            self._discard_lookup(req_id)
        return super().reset()

    def _keys_for_ids(self, ids):
        gpu = self._gpu_block_pool
        if gpu is None:
            return []
        return list(
            dict.fromkeys(
                gpu.blocks[i].block_hash
                for i in ids
                if not gpu.blocks[i].is_null and gpu.blocks[i].block_hash is not None
            )
        )

    def _limit(self):
        now = time.monotonic()
        self._store_credit = min(
            self._max_store,
            self._store_credit
            + (now - self._credit_at) * self._store_rate / self._bytes_per_slot,
        )
        self._credit_at = now
        return int(self._store_credit)

    def _candidate_ids(self, state, block_ids, new_groups=(), *, reset=False):
        """Query only confirmed positional candidates beyond the store cursor."""
        if self._gpu_block_pool is None:
            return
        confirmed = (
            state.request.num_computed_tokens - state.request.num_output_placeholders
        )
        for g in self.prefix_cacheable_group_ids:
            manager = self.cpu_coordinator.single_type_managers[g]
            if not manager.has_positionally_stable_blocks:
                continue
            old = () if reset else block_ids[g]
            new = (new_groups[g] or ()) if g < len(new_groups) else ()
            ready = min(
                len(old) + len(new),
                self.cpu_coordinator.get_num_cacheable_tokens(confirmed, g)
                // self.group_block_sizes[g],
            )
            start = 0 if reset else state.num_stored_blocks[g]
            hashes = None
            for i in range(start, ready):
                bid = old[i] if i < len(old) else new[i - len(old)]
                block = self._gpu_block_pool.blocks[bid]
                if block.is_null:
                    if hashes is None:
                        hashes = resolve_block_hashes(
                            state.request.block_hashes,
                            self.hash_block_size,
                            self.group_block_sizes[g],
                        )
                    block = self._cached_gpu_block(hashes, i, g)
                    if block is None:
                        continue
                if block.block_id not in self._in_flight_store_gpu_blocks:
                    yield block.block_id

    def _prepare_eager_store_specs(self, output):
        from vllm.distributed.kv_transfer.kv_connector.utils import yield_req_data

        ids = []
        preempted = output.preempted_req_ids or ()
        for req_id, groups, _ in yield_req_data(output):
            state = self._reqs_to_store.get(req_id)
            if (
                state is None
                or state.finished
                or not output.num_scheduled_tokens.get(req_id, 0)
            ):
                continue
            ids.extend(
                self._candidate_ids(
                    state,
                    state.block_ids,
                    groups or (),
                    reset=req_id in preempted,
                )
            )
        if output.kv_connector_block_state is not None:
            for (
                req_id,
                entries,
            ) in output.kv_connector_block_state.boundary_state_offloads.items():
                state = self._reqs_to_store.get(req_id)
                if (
                    state is not None
                    and not state.finished
                    and req_id not in preempted
                    and req_id not in output.finished_req_ids
                ):
                    ids.extend(
                        x[1]
                        for x in entries
                        if x[1] not in self._in_flight_store_gpu_blocks
                    )
        with self.cpu_block_pool.admission(self._keys_for_ids(ids), self._limit()):
            result = super()._prepare_eager_store_specs(output)
            self._store_credit -= len(result[0])
            return result

    def _queue_finished_eager_store(self, request, block_ids):
        state = self._reqs_to_store.get(request.request_id)
        if state is None or self._gpu_block_pool is None:
            return
        ids = list(self._candidate_ids(state, block_ids))
        partial = self._find_fa_partial_tail_source(request, block_ids)
        if partial is not None:
            ids.append(partial)
        with self.cpu_block_pool.admission(self._keys_for_ids(ids), self._limit()):
            n = len(self._pending_finished_stores)
            super()._queue_finished_eager_store(request, block_ids)
            self._store_credit -= sum(
                len(t.gpu_block_ids) for t in self._pending_finished_stores[n:]
            )


class SharedOffloadWorker(SimpleCPUOffloadWorker):
    def __init__(self, config, cache_config, capacity_bytes, extra):
        self._extra = extra
        self._namespace = namespace(config, cache_config, extra)
        self._role_contract = role_contract(cache_config)
        self._mappings = []
        super().__init__(config, cache_config, capacity_bytes)

    def _init_cpu_mode(self, gpu_caches, total_bytes_per_block, device):
        self._pool_client = PoolClient(
            self._extra["manager_socket"], self._namespace, self.num_cpu_blocks
        )
        self._pool_client.call("contract", "worker", self._role_contract)
        layout = [
            [name, t.stride(0) * t.element_size()] for name, t in gpu_caches.items()
        ]
        info = self._pool_client.call("attach", layout)
        size = total_bytes_per_block * self.num_cpu_blocks
        with Path(info["file"]).open("r+b") as f:
            mapping = mmap.mmap(f.fileno(), size)
        if hasattr(mmap, "MADV_NOHUGEPAGE"):
            mapping.madvise(mmap.MADV_NOHUGEPAGE)
        self._mappings.append(mapping)
        offset = 0
        self.cpu_kv_caches = {}
        for name, gpu in gpu_caches.items():
            count = self.num_cpu_blocks * gpu.shape[1]
            tensor = torch.frombuffer(
                mapping, dtype=gpu.dtype, count=count, offset=offset
            ).view(self.num_cpu_blocks, *gpu.shape[1:])
            pin_tensor(tensor)
            self.cpu_kv_caches[name] = tensor
            offset += tensor.nbytes
        assert offset == size
        self._backend = DmaCopyBackend()
        self._backend.init(
            gpu_caches, self.cpu_kv_caches, device, self.load_stream, self.store_stream
        )


def preflight_shared_pool(config, cache_config):
    """Check the real scheduler/group contract before expensive GPU warmup."""
    transfer = config.kv_transfer_config
    if transfer is None:
        return
    connectors = (transfer.kv_connector_extra_config or {}).get("connectors", [])
    if transfer.kv_connector == "SharedCPUOffloadConnector":
        connectors = [
            {
                "kv_connector": transfer.kv_connector,
                "kv_connector_extra_config": transfer.kv_connector_extra_config,
            }
        ]
    for connector in connectors:
        if connector.get("kv_connector") != "SharedCPUOffloadConnector":
            continue
        from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes

        extra = connector["kv_connector_extra_config"]
        block, hashed = resolve_kv_cache_block_sizes(cache_config, config)
        scheduler = SharedOffloadScheduler(
            config, cache_config, int(extra["pool_bytes"]), block, hashed, extra
        )
        scheduler._client.close()
