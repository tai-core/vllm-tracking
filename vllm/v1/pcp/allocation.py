# SPDX-License-Identifier: Apache-2.0
"""Worker-local physical allocation; scheduler block IDs remain logical."""

from copy import deepcopy
from dataclasses import dataclass

from vllm.config import VllmConfig
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    UniformTypeKVCacheSpecs,
    compute_layout_strides,
    is_mamba_group,
    iter_layer_specs,
)
from vllm.v1.pcp.layout import PCPPageLayout


def resolve_pcp_layout(config, groups):
    """Resolve layer-declared ownership after grouping; no model inference."""
    import math

    if any(
        spec.pcp is not None
        for group in groups
        if is_mamba_group(group)
        for spec in iter_layer_specs(group.kv_cache_spec)
    ):
        raise ValueError("PCP page placement cannot describe recurrent head state")
    attention = [
        (i, spec)
        for i, group in enumerate(groups)
        if not is_mamba_group(group)
        for spec in iter_layer_specs(group.kv_cache_spec)
    ]
    declared = [(i, spec) for i, spec in attention if spec.pcp is not None]
    if not declared:
        return None
    if len(declared) != len(attention):
        raise ValueError("PCP page groups cannot mix declared and undeclared caches")
    owners = {}
    for group_id, spec in declared:
        key = spec.pcp.ownership_group
        if key is not None:
            previous = owners.setdefault(key, (group_id, spec.block_size))
            if previous != (group_id, spec.block_size):
                raise ValueError(
                    f"PCP ownership group {key!r} must share one logical cache group"
                )
        if spec.dcp_sharded and config.parallel_config.decode_context_parallel_size > 1:
            raise ValueError("PCP page placement cannot also shard pages through DCP")
    return PCPPageLayout(
        config.parallel_config.prefill_context_parallel_size,
        max(spec.block_size for _, spec in declared),
        math.lcm(*(spec.pcp.query_alignment for _, spec in declared)),
    )


def _layer_specs(groups):
    return {
        name: (
            group.kv_cache_spec.kv_cache_specs[name]
            if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
            else group.kv_cache_spec
        )
        for group in groups
        for name in group.layer_names
    }


def make_pcp_physical_config(
    config: VllmConfig,
    logical: KVCacheConfig,
    size: int | None = None,
) -> KVCacheConfig:
    """Repack shared attention pages while retaining dedicated KDA pools.

    This result is only for allocation and binding. It must never replace
    the scheduler or metadata builder's logical KVCacheConfig.
    All ranks reserve the same rounded capacity; at most one page per rank
    is padding. Repacking preserves aliasing between cache groups.
    """
    if logical.pcp_layout is not None:
        if size is not None and size != logical.pcp_layout.size:
            raise ValueError("Worker PCP size disagrees with the resolved cache layout")
        size = logical.pcp_layout.size
    if size is None or size <= 0:
        raise ValueError("PCP size must be positive")
    mamba_groups = [group for group in logical.kv_cache_groups if is_mamba_group(group)]
    if mamba_groups and logical.musa_mamba_num_blocks is None:
        raise ValueError("PCP sharding requires separate KDA physical pools")
    attention_groups = [
        group for group in logical.kv_cache_groups if not is_mamba_group(group)
    ]
    result = deepcopy(logical)
    result.num_blocks = max(1, (logical.num_blocks + size - 1) // size)
    # Repack the existing tensor description. Core already resolved aliasing,
    # packing and state-pool ownership; the worker only changes the block count.
    layout = config.cache_config.get_resolved_kv_cache_layout()
    specs = _layer_specs(attention_groups)
    for tensor in result.kv_cache_tensors:
        if tensor.musa_mamba_pool:
            continue
        spec = specs[tensor.layers[0]]
        tensor.size = tensor.size // logical.num_blocks * result.num_blocks
        if not layout.is_block_outermost:
            tensor.offset = tensor.offset // logical.num_blocks * result.num_blocks
        layer_stride, block_stride, _, _, _ = compute_layout_strides(
            spec,
            result.num_blocks,
            len(tensor.layers),
            layout,
            fixed_strides=(
                None,
                tensor.block_stride
                if layout.is_block_outermost or layout.is_block_compact
                else None,
                None,
                None,
                None,
            ),
        )
        tensor.layer_stride = layer_stride
        tensor.block_stride = block_stride
    return result


@dataclass(frozen=True)
class PCPPageWorkspaceBudget:
    """Cache allocations and additional native-page staging, in bytes.

    Existing activation/current-token communication remains covered by the
    runner's memory profile. This budget covers the new history-page and CoW
    staging for the selected transport. KDA pools are counted separately and
    are never divided by PCP size.
    """

    attention_bytes: int
    kda_bytes: int
    history_bytes: int
    copy_bytes: int
    max_copy_pages: int
    staging_bytes: int = 0
    current_metadata_bytes: int = 0

    @property
    def total_bytes(self) -> int:
        # Block maintenance finishes before the model's layer execution.
        return (
            self.attention_bytes
            + self.kda_bytes
            + self.staging_bytes
            + self.current_metadata_bytes
            + max(self.history_bytes, self.copy_bytes)
        )


def current_staging_geometry(config, groups):
    specs = [
        spec
        for group in groups
        if not is_mamba_group(group)
        for spec in iter_layer_specs(group.kv_cache_spec)
    ]
    rows = (
        config.scheduler_config.max_num_batched_tokens
        + 2
        * max(spec.block_size for spec in specs)
        * config.scheduler_config.max_num_seqs
    )
    row_bytes = max(
        spec.page_size_bytes
        // (spec.block_size // max(1, getattr(spec, "tokens_per_state", 1)))
        for spec in specs
    )
    return rows, (row_bytes + 15) // 16 * 16


def plan_pcp_page_workspace(
    config, logical, physical, size, *, use_ipc=False, validate=True
):
    """Account for persistent pages and bounded staging in the worker budget."""
    if size <= 0:
        raise ValueError("PCP size must be positive")
    attention = [
        group for group in logical.kv_cache_groups if not is_mamba_group(group)
    ]
    max_copy_pages = min(
        logical.num_blocks, config.scheduler_config.max_num_seqs * len(attention)
    )
    # One partial tail per scheduled request per attention group. All source
    # snapshots coexist before any destination write, preserving CoW semantics.
    block_bytes = max(
        tensor.size // physical.num_blocks
        for tensor in physical.kv_cache_tensors
        if not tensor.musa_mamba_pool
    )
    owner_pages = (logical.num_blocks + size - 1) // size
    # Conservatively allow all cache kinds of one layer to coexist. Layers
    # execute serially; only descriptors, never page data, survive a layer.
    history_bytes = sum(
        ((size + 1) * owner_pages + logical.num_blocks)
        * max(spec.page_size_bytes for spec in iter_layer_specs(group.kv_cache_spec))
        for group in attention
    )
    copy_width = max(1, min(max_copy_pages, owner_pages))
    copy_bytes = ((size + 1) * copy_width + max_copy_pages) * block_bytes
    if use_ipc:
        # IPC reads directly into the dense output: no collective staging or
        # gathered replicas. Allow another dense-sized owner selection for
        # commit_pages; whole-pool CoW only keeps one immutable snapshot.
        history_bytes = (
            2
            * logical.num_blocks
            * sum(
                max(
                    spec.page_size_bytes
                    for spec in iter_layer_specs(group.kv_cache_spec)
                )
                for group in attention
            )
        )
        copy_bytes = max_copy_pages * block_bytes
    stage_rows, stage_bytes = current_staging_geometry(config, attention)
    # Layer-identical specs share forward metadata. Reserve two int64 values
    # per possible row for composition routing, plus bounded producer lists.
    metadata_rows = logical.num_blocks * sum(
        spec.block_size // max(1, getattr(spec, "tokens_per_state", 1))
        for group in attention
        for spec in set(iter_layer_specs(group.kv_cache_spec))
    )
    budget = PCPPageWorkspaceBudget(
        attention_bytes=max(
            tensor.size
            for tensor in physical.kv_cache_tensors
            if not tensor.musa_mamba_pool
        ),
        kda_bytes=sum(
            tensor.size
            for tensor in physical.kv_cache_tensors
            if tensor.musa_mamba_pool
        ),
        history_bytes=history_bytes,
        copy_bytes=copy_bytes,
        max_copy_pages=max_copy_pages,
        staging_bytes=stage_rows * stage_bytes + 65536,
        current_metadata_bytes=metadata_rows * 16 + stage_rows * size * 64,
    )
    original_bytes = (
        max(
            tensor.size
            for tensor in logical.kv_cache_tensors
            if not tensor.musa_mamba_pool
        )
        + budget.kda_bytes
    )
    if logical.pcp_memory_budget_bytes is not None:
        original_bytes = logical.pcp_memory_budget_bytes
    if validate and budget.total_bytes > original_bytes:
        raise ValueError(
            "PCP persistent cache and page workspace exceed the profiled "
            f"cache budget: need {budget.total_bytes}, have {original_bytes} bytes"
        )
    return budget


def pcp_required_blocks(config, groups):
    """Logical attention capacity for one max-length request, including null."""
    from vllm.utils.math_utils import cdiv
    from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs

    return 1 + sum(
        group.kv_cache_spec.max_memory_usage_pages(config)
        if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
        else cdiv(
            group.kv_cache_spec.max_memory_usage_bytes(config),
            group.kv_cache_spec.page_size_bytes,
        )
        for group in groups
        if not is_mamba_group(group)
    )


def bind_pcp_pages(config, logical, physical, forward_context, caches):
    """Bind native views and one whole-pool maintenance view after allocation."""
    import torch

    from vllm.distributed import get_pcp_group
    from vllm.v1.pcp.native_pages import PCPNativePages

    group = get_pcp_group()
    if logical.pcp_layout is not None and group.world_size != logical.pcp_layout.size:
        raise ValueError("PCP process group disagrees with the resolved cache layout")
    budget = (
        plan_pcp_page_workspace(
            config,
            logical,
            physical,
            group.world_size,
            use_ipc=any(
                isinstance(cache, torch.Tensor) and cache.device.type == "musa"
                for cache in caches.values()
            ),
        )
        if config is not None
        else None
    )
    maintenance = None
    ipc_reader = None
    for cache_group in logical.kv_cache_groups:
        if is_mamba_group(cache_group):
            continue
        for name in cache_group.layer_names:
            layer = forward_context[name]
            storage = layer.kv_cache
            placement = next(
                tensor for tensor in physical.kv_cache_tensors if name in tensor.layers
            )
            kernel_stride_bytes = storage.stride(0) * storage.element_size()
            factor, remainder = divmod(placement.block_stride, kernel_stride_bytes)
            valid_pages = storage.shape[0] - (physical.num_blocks - 1) * factor
            if remainder or factor < 1 or not 0 < valid_pages <= factor:
                raise ValueError(
                    f"Invalid PCP native geometry for {name}: "
                    f"shape={tuple(storage.shape)}, "
                    f"physical_blocks={physical.num_blocks}, "
                    f"manager_stride={placement.block_stride}, "
                    f"kernel_stride={kernel_stride_bytes}"
                )
            layer.pcp_pages = PCPNativePages(
                storage,
                logical.num_blocks,
                group,
                kernel_blocks_per_page=factor,
                max_materialize_pages=logical.num_blocks * valid_pages,
                kernel_pages_per_block=valid_pages,
                tokens_per_state=max(
                    1,
                    getattr(_layer_specs([cache_group])[name], "tokens_per_state", 1),
                ),
            )
            if maintenance is None:
                raw = torch.empty(0, dtype=torch.uint8, device=storage.device)
                raw.set_(storage.untyped_storage())
                raw = raw.view(physical.num_blocks, 1, -1)
                maintenance = PCPNativePages(
                    raw,
                    logical.num_blocks,
                    group,
                    max_materialize_pages=(
                        budget.max_copy_pages if budget else logical.num_blocks
                    ),
                )
                maintenance.page_workspace_budget = budget
                if storage.device.type == "musa":
                    from vllm_musa.distributed.device_communicators import (
                        pcp_native_pages,
                    )

                    ipc_reader = pcp_native_pages.PCPNativeIPC(raw, group)
                    if config is not None:
                        stage_rows, row_bytes = current_staging_geometry(
                            config, logical.kv_cache_groups
                        )
                        ipc_reader.register_current_staging(stage_rows, row_bytes)
                    if not hasattr(group, "_pcp_native_pools"):
                        group._pcp_native_pools = []
                    group._pcp_native_pools.append(ipc_reader)
                    maintenance.ipc_reader = ipc_reader
            layer.pcp_pages.ipc_reader = ipc_reader
            layer.pcp_maintenance = maintenance
    return maintenance


def apply_pcp_block_copies(logical, forward_context, maintenance, copies, group_ids):
    """Route state-pool CoW separately from the shared attention pool."""
    from vllm.v1.worker.utils import copy_kv_cache_blocks_inplace

    if group_ids is None or len(group_ids) != len(copies):
        raise ValueError("PCP CoW requires a cache-group ID for every copy")
    by_group = {}
    for group_id, pair in zip(group_ids, copies):
        if not 0 <= group_id < len(logical.kv_cache_groups):
            raise ValueError("Invalid CoW cache group")
        by_group.setdefault(group_id, []).append(pair)
    attention_copies = []
    for group_id, group_copies in by_group.items():
        group = logical.kv_cache_groups[group_id]
        if not is_mamba_group(group):
            attention_copies.extend(group_copies)
            continue
        if logical.musa_mamba_num_blocks is None:
            raise ValueError("PCP KDA copies require their independent state pool")
        states = []
        for name in group.layer_names:
            cache = forward_context[name].kv_cache
            states.extend(cache if isinstance(cache, (tuple, list)) else [cache])
        copy_kv_cache_blocks_inplace(
            states, logical.musa_mamba_num_blocks, group_copies
        )
    maintenance.copy_blocks(attention_copies)
