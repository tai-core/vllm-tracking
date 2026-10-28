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

import torch

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import KVCacheBlock, make_block_hash_with_group_id
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
        self.blocks = {0: KVCacheBlock(0, pool=self, is_null=True)}
        self.null_block = self.blocks[0]
        self.cached_block_hash_to_block = _HashView()
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
            b.set_block_hash(bytes.fromhex(primary), tokens)
        self.blocks[slot] = b
        self._refs[slot] = {
            "generation": generation,
            "writing": primary is None,
            "aliases": {},
            "leases": 0,
        }
        return b

    @contextlib.contextmanager
    def lookup_scope(self, keys, acquire):
        view = self.cached_block_hash_to_block.rows
        assert not view
        leased = []
        try:
            rows = self.client.call(
                "acquire" if acquire else "lookup", [k.hex() for k in keys]
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
        self._credits = []
        try:
            with self.lookup_scope(keys, False):
                missing = sum(
                    k not in self.cached_block_hash_to_block.rows for k in keys
                )
                refs = (
                    self.client.call("reserve", min(missing, limit)) if missing else []
                )
                for slot, generation in refs:
                    b = self._from_row(slot, generation)
                    b.ref_cnt = 1
                    self._refs[slot]["leases"] = 1
                    self._credits.append(b)
                yield
        finally:
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
    def __init__(
        self, config, cache_config, capacity_bytes, block_size, hash_size, extra
    ):
        self._extra = extra
        self._client = None
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

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        keys = [
            make_block_hash_with_group_id(h, g)
            for h in request.block_hashes
            for g in self.prefix_cacheable_group_ids
        ]
        with self.cpu_block_pool.lookup_scope(keys, True):
            return super().get_num_new_matched_tokens(request, num_computed_tokens)

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

    def _prepare_eager_store_specs(self, output):
        ids = []
        for state in self._reqs_to_store.values():
            for group in state.block_ids:
                ids.extend(group)
        from vllm.distributed.kv_transfer.kv_connector.utils import yield_req_data

        for _, groups, _ in yield_req_data(output):
            for group in groups or ():
                ids.extend(group or ())
        if output.kv_connector_block_state is not None:
            for (
                entries
            ) in output.kv_connector_block_state.boundary_state_offloads.values():
                ids.extend(x[1] for x in entries)
        with self.cpu_block_pool.admission(self._keys_for_ids(ids), self._limit()):
            result = super()._prepare_eager_store_specs(output)
            self._store_credit -= len(result[0])
            return result

    def _queue_finished_eager_store(self, request, block_ids):
        ids = [i for group in block_ids for i in group]
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
