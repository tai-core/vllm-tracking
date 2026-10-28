# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Same-host shared CPU offload using native prefix matching and async DMA."""

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload_connector import (
    SimpleCPUOffloadConnector,
)
from vllm.v1.metrics.cache_hit_source import CacheHitSource
from vllm.v1.shared_kv_offload.shared_adapter import (
    SharedOffloadScheduler,
    SharedOffloadWorker,
)


class SharedCPUOffloadConnector(SimpleCPUOffloadConnector):
    def __init__(self, vllm_config, role, kv_cache_config):
        KVConnectorBase_V1.__init__(self, vllm_config, role, kv_cache_config)
        extra = self._kv_transfer_config.kv_connector_extra_config or {}
        if not vllm_config.cache_config.enable_prefix_caching:
            raise ValueError("Shared CPU offload requires prefix caching")
        if vllm_config.parallel_config.world_size != 1:
            raise ValueError("Shared CPU offload currently requires TP1/PP1/PCP1")
        capacity = int(extra["pool_bytes"])
        if (
            capacity <= 0
            or not extra.get("manager_socket")
            or not extra.get("cache_namespace")
        ):
            raise ValueError(
                "Positive pool_bytes, manager_socket and cache_namespace required"
            )
        self._cache_hit_source = CacheHitSource.HOST
        self.scheduler_manager = None
        self.worker_handler = None
        if role == KVConnectorRole.SCHEDULER:
            from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes

            block_size, hash_size = resolve_kv_cache_block_sizes(
                kv_cache_config, vllm_config
            )
            self.scheduler_manager = SharedOffloadScheduler(
                vllm_config, kv_cache_config, capacity, block_size, hash_size, extra
            )
        elif role == KVConnectorRole.WORKER:
            self.worker_handler = SharedOffloadWorker(
                vllm_config, kv_cache_config, capacity, extra
            )
