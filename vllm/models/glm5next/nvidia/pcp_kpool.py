# SPDX-License-Identifier: Apache-2.0
"""Encode complete rank-local kpool groups once, before PCP exchange."""

from __future__ import annotations

import torch

from vllm.models.glm5next.nvidia.ops import kpool_compress as kernels
from vllm.triton_utils import tl, triton
from vllm.v1.pcp.plan import get_pcp_plan_cache


def compress_local_pools(
    key: torch.Tensor,
    gate: torch.Tensor,
    ape: torch.Tensor,
    logical_pool_slots: torch.Tensor,
    pool_size: int,
    page_rows: int,
    *,
    round_scale: bool,
    positions: torch.Tensor | None = None,
    request_ids: torch.Tensor | None = None,
    plan_key=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return complete-pool slot IDs and opaque [FP8 K | FP32 scale] rows.

    Internal segment boundaries must be pool-aligned. Leading incomplete
    pools are handled by the persistent tail owner, not this function.
    Output contains no padding and no duplicate pool-completion candidates.
    """
    if key.ndim != 2 or gate.shape != key.shape:
        raise ValueError("Keys and gates must have equal [token, head] shapes")
    if key.shape[0] != logical_pool_slots.numel():
        raise ValueError("Each token must have one pool-completion slot")
    if pool_size <= 0 or page_rows <= 0:
        raise ValueError("Pool and page sizes must be positive")
    count, head = key.shape
    plans = get_pcp_plan_cache(plan_key)
    cached = plans.get(("pool_rows", plan_key)) if plans is not None else None
    if cached is None:
        end = torch.arange(count, device=key.device)
        valid = (logical_pool_slots >= 0) & (end >= pool_size - 1)
        if positions is not None:
            from vllm.v1.pcp.kpool_boundary import complete_pool_mask

            if request_ids is None:
                raise ValueError("Absolute pool positions require request identity")
            valid &= complete_pool_mask(
                logical_pool_slots, positions, request_ids, pool_size
            )
        end = end[valid]
        slots = logical_pool_slots[valid].to(torch.int64)
        if slots.unique().numel() != slots.numel():
            raise ValueError("Multiple local tokens complete the same pool")
        rows = end[:, None] - pool_size + 1 + torch.arange(pool_size, device=key.device)
        if plans is not None:
            plans[("pool_rows", plan_key)] = (slots, rows)
    else:
        slots, rows = cached
    # Only geometry is read when WRITE_CACHE=False; this is not a cache replica.
    geometry = torch.empty(
        (0, page_rows, head + 4), dtype=torch.uint8, device=key.device
    )
    values, scales = kernels.kpool_compress_and_write_cache(
        geometry,
        key[rows],
        gate[rows],
        ape,
        slots,
        pool_size=pool_size,
        head_dim=head,
        round_scale=round_scale,
        return_compressed=True,
        write_cache=False,
    )
    encoded = torch.cat(
        (values.view(torch.uint8), scales.view(torch.uint8).reshape(-1, 4)), dim=1
    )
    return slots, encoded


@triton.jit
def _scatter_encoded_pool_bytes(
    cache,
    slots,
    encoded,
    CACHE_STRIDE: tl.constexpr,
    ENCODED_STRIDE: tl.constexpr,
    PAGE_ROWS: tl.constexpr,
    HEAD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.load(slots + row).to(tl.int64)
    if slot >= 0:
        byte = tl.arange(0, BLOCK)
        page = slot // PAGE_ROWS
        offset = slot % PAGE_ROWS
        target = page * CACHE_STRIDE + tl.where(
            byte < HEAD,
            offset * HEAD + byte,
            PAGE_ROWS * HEAD + offset * 4 + byte - HEAD,
        )
        value = tl.load(
            encoded + row * ENCODED_STRIDE + byte,
            mask=byte < HEAD + 4,
            other=0,
        )
        tl.store(cache + target, value, mask=byte < HEAD + 4)


def write_compressed_pools(
    cache: torch.Tensor,
    slots: torch.Tensor,
    encoded: torch.Tensor,
) -> None:
    """Scatter byte rows to the native per-page K-plane/scale-plane layout."""
    if cache.dtype != torch.uint8 or cache.ndim != 3:
        raise ValueError("Expected uint8 paged index cache")
    head = cache.shape[-1] - 4
    if encoded.dtype != torch.uint8 or encoded.shape != (slots.numel(), head + 4):
        raise ValueError("Encoded pool rows do not match cache geometry")
    if slots.numel() == 0:
        return
    if cache.device.type != "cpu":
        _scatter_encoded_pool_bytes[(slots.numel(),)](
            cache,
            slots.contiguous(),
            encoded,
            CACHE_STRIDE=cache.stride(0),
            ENCODED_STRIDE=encoded.stride(0),
            PAGE_ROWS=cache.shape[1],
            HEAD=head,
            BLOCK=triton.next_power_of_2(head + 4),
        )
        return
    valid = slots >= 0
    slots = slots[valid]
    encoded = encoded[valid]
    pages = slots // cache.shape[1]
    offsets = slots % cache.shape[1]
    # The nominal last dimension is not an AoS row stride: scales follow
    # all values in each page. Preserve those bytes without requantizing.
    byte_pages = torch.as_strided(
        cache,
        (cache.shape[0], cache.shape[1] * (head + 4)),
        (cache.stride(0), 1),
    )
    key_offsets = offsets[:, None] * head + torch.arange(head, device=slots.device)
    scale_offsets = (
        cache.shape[1] * head
        + offsets[:, None] * 4
        + torch.arange(4, device=slots.device)
    )
    byte_pages[pages[:, None], key_offsets] = encoded[:, :head]
    byte_pages[pages[:, None], scale_offsets] = encoded[:, head:]


def materialized_index_cache(encoded_pages: torch.Tensor) -> torch.Tensor:
    """Convert owned-store rows into native K-plane/scale-plane page bytes."""
    if encoded_pages.dtype != torch.uint8 or encoded_pages.ndim != 3:
        raise ValueError("Expected opaque encoded [page, row, byte] data")
    pages, rows, width = encoded_pages.shape
    if width <= 4:
        raise ValueError("Index rows must contain values and an FP32 scale")
    return torch.cat(
        (
            encoded_pages[:, :, :-4].reshape(pages, rows * (width - 4)),
            encoded_pages[:, :, -4:].reshape(pages, rows * 4),
        ),
        dim=1,
    ).reshape(pages, rows, width)


def current_pool_descriptors(slots, group, plan_key=None):
    """Exchange immutable slot descriptors once; payload stays on its producer."""
    plans = get_pcp_plan_cache(plan_key)
    cache_key = ("current_pool_descriptors", plan_key, id(group))
    cached = plans.get(cache_key) if plans is not None else None
    if cached is not None:
        local_count, gathered_slots, width, sources = cached
        if local_count != slots.numel():
            raise RuntimeError("PCP current pool plan changed within one forward")
        return gathered_slots, width, sources
    counts = group.all_gather(
        torch.tensor([slots.numel()], device=slots.device, dtype=torch.int64), dim=0
    )
    width = int(counts.max().item())
    if width:
        padded = slots.new_full((width,), -1)
        padded[: slots.numel()] = slots
        gathered_slots = group.all_gather(padded, dim=0)
    else:
        gathered_slots = slots
    sources = torch.arange(gathered_slots.numel(), device=slots.device)
    if plans is not None:
        plans[cache_key] = (slots.numel(), gathered_slots, width, sources)
    return gathered_slots, width, sources


def exchange_compressed_pools(
    slots: torch.Tensor,
    encoded: torch.Tensor,
    group,
    plan_key=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exchange only completed encoded pools, including empty producer ranks."""
    plans = get_pcp_plan_cache(plan_key)
    cache_key = ("pool_exchange", plan_key, id(group))
    cached = plans.get(cache_key) if plans is not None else None
    if cached is None:
        counts = group.all_gather(
            torch.tensor([slots.numel()], device=slots.device, dtype=torch.int64), dim=0
        )
        width = int(counts.max().item())
        gathered_slots = None
    else:
        local_count, width, gathered_slots = cached
        if local_count != slots.numel():
            raise RuntimeError("PCP pool write plan changed within one forward")
    if width == 0:
        if plans is not None:
            plans[cache_key] = (0, 0, slots)
        return slots, encoded
    padded_slots = torch.full((width,), -1, device=slots.device, dtype=torch.int64)
    padded_bytes = encoded.new_zeros((width, encoded.shape[1]))
    padded_slots[: slots.numel()] = slots
    padded_bytes[: slots.numel()] = encoded
    if gathered_slots is None:
        gathered_slots = group.all_gather(padded_slots, dim=0)
        if plans is not None:
            plans[cache_key] = (slots.numel(), width, gathered_slots)
    return gathered_slots, group.all_gather(padded_bytes, dim=0)


def compress_pcp_pools(
    key: torch.Tensor,
    gate: torch.Tensor,
    ape: torch.Tensor,
    cache: torch.Tensor,
    local_slots: torch.Tensor,
    pool_size: int,
    group,
    *,
    round_scale: bool,
    page_store=None,
    positions=None,
    request_ids=None,
    plan_key=None,
) -> None:
    slots, encoded = compress_local_pools(
        key,
        gate,
        ape,
        local_slots,
        pool_size,
        cache.shape[1],
        round_scale=round_scale,
        positions=positions,
        request_ids=request_ids,
        plan_key=plan_key,
    )
    slots, encoded = exchange_compressed_pools(slots, encoded, group, plan_key=plan_key)
    if page_store is not None:
        slots = page_store.write_slots(slots, cache.shape[1])
    write_compressed_pools(cache, slots, encoded)


def read_compressed_pools(cache, slots):
    """Read native K-plane/scale-plane entries without numerical conversion."""
    head = cache.shape[-1] - 4
    pages, offsets = slots // cache.shape[1], slots % cache.shape[1]
    raw = torch.as_strided(
        cache,
        (cache.shape[0], cache.shape[1] * (head + 4)),
        (cache.stride(0), 1),
    )
    values = raw[
        pages[:, None],
        offsets[:, None] * head + torch.arange(head, device=slots.device),
    ]
    scales = raw[
        pages[:, None],
        cache.shape[1] * head
        + offsets[:, None] * 4
        + torch.arange(4, device=slots.device),
    ]
    return torch.cat((values, scales), dim=1)


def complete_boundary_pools(
    key,
    gate,
    ape,
    positions,
    global_slots,
    tail_slots,
    layout,
    index_cache,
    tail_cache,
    index_pages,
    tail_pages,
    pool_size,
    group,
    *,
    round_scale,
    plan_key=None,
    return_compressed=False,
):
    """Complete leading partial pools once, on each request's tail owner."""
    from vllm.v1.pcp.kpool_boundary import boundary_token_rows

    ring = tail_cache.shape[2]
    plans = get_pcp_plan_cache(plan_key)
    cache_key = ("boundary_metadata", plan_key, id(group), ring, pool_size)
    cached = plans.get(cache_key) if plans is not None else None
    if cached is None:
        consumed = torch.zeros_like(global_slots, dtype=torch.bool)
        if getattr(layout, "has_partial_pools", True):
            rows = boundary_token_rows(positions, tail_slots, pool_size, ring)
        else:
            rows = global_slots.new_empty((0, pool_size))
        prepared = None
        if rows.numel():
            if index_pages is None or tail_pages is None:
                raise RuntimeError(
                    "PCP boundary completion requires bound sharded pools"
                )
            valid = rows >= 0
            flat_rows = rows[valid]
            consumed[flat_rows] = True
            safe = rows.clamp_min(0)
            tail_locations = torch.where(valid, tail_slots[safe], -1)
            target_slots = torch.where(valid, global_slots[safe], -1)
            prefix_pages = torch.unique(tail_locations[valid] // ring)
            prefix_slots = tail_pages.remap_slots(prefix_pages, tail_locations, ring)
            request_owners, _ = tail_pages.locations(tail_locations[:, 0] // ring)
            own = request_owners == group.rank_in_group
            active = valid & own[:, None]
            token_positions = torch.where(active, positions[safe], -1).to(torch.int32)
            prefix_slots = torch.where(active, prefix_slots, -1)
            request_indices = torch.arange(
                rows.shape[0], device=key.device, dtype=torch.int64
            )
            scratch_slots = torch.where(
                active & (target_slots >= 0), request_indices[:, None], -1
            )
            completed = own & (target_slots >= 0).any(dim=1)
            completed_indices = request_indices[completed]
            destinations = target_slots.max(dim=1).values[completed]
            prepared = (
                valid,
                flat_rows,
                prefix_pages,
                prefix_slots,
                token_positions,
                scratch_slots,
                completed_indices,
                destinations,
            )
        if plans is not None:
            # Hold metadata alive, not layer-specific keys or cache contents.
            plans[cache_key] = (
                (layout, positions, global_slots, tail_slots),
                consumed,
                prepared,
            )
    else:
        _, consumed, prepared = cached
    if prepared is None:
        if return_compressed:
            return (
                consumed,
                global_slots.new_empty((0,), dtype=torch.int64),
                torch.empty(
                    (0, key.shape[-1] + 4), dtype=torch.uint8, device=key.device
                ),
            )
        return consumed
    (
        valid,
        flat_rows,
        prefix_pages,
        prefix_slots,
        token_positions,
        scratch_slots,
        completed_indices,
        destinations,
    ) = prepared
    count = valid.shape[0]
    head = key.shape[-1]
    keys = key.new_zeros((count, pool_size, head))
    gates = gate.new_zeros((count, pool_size, head))
    packed = layout.gather_rows(
        torch.stack((key, gate), dim=1),
        flat_rows,
        plan_key=("boundary", plan_key) if plan_key is not None else None,
    )
    keys[valid] = packed[:, 0]
    gates[valid] = packed[:, 1]
    prefix = tail_pages.materialize_ids(
        prefix_pages,
        plan_key=cache_key if plan_key is not None else None,
    )
    page_rows = index_cache.shape[1]
    scratch = torch.zeros(
        ((count + page_rows - 1) // page_rows, page_rows, head + 4),
        dtype=torch.uint8,
        device=key.device,
    )
    kernels.kpool_decode_update_and_maybe_write_cache_batched(
        scratch,
        prefix,
        prefix_slots,
        keys,
        gates,
        ape,
        scratch_slots,
        token_positions,
        pool_size,
        head,
        round_scale=round_scale,
    )
    tail_pages.commit_pages(prefix_pages, prefix)
    encoded = read_compressed_pools(scratch, completed_indices)
    if return_compressed:
        return consumed, destinations, encoded
    destinations, encoded = exchange_compressed_pools(
        destinations,
        encoded,
        group,
        plan_key=("boundary", plan_key) if plan_key is not None else None,
    )
    write_compressed_pools(
        index_cache, index_pages.write_slots(destinations, page_rows), encoded
    )
    return consumed
