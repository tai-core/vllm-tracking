# SPDX-License-Identifier: Apache-2.0
"""Select only leading incomplete pools, preserving request boundaries."""

import torch


def boundary_token_rows(
    positions: torch.Tensor,
    tail_slots: torch.Tensor,
    pool_size: int,
    ring_size: int,
    num_decode_tokens: int = 0,
) -> torch.Tensor:
    """Return [request, pool_size] global row IDs, padded with -1.

    A boundary group includes the new tokens of the already-started pool
    only. Its earlier tokens must be read from the persistent tail.
    """
    if positions.ndim != 1 or positions.shape != tail_slots.shape:
        raise ValueError("Global positions and tail slots must be matching vectors")
    if pool_size <= 0 or ring_size < pool_size or ring_size % pool_size:
        raise ValueError("Invalid pool/ring geometry")
    count = positions.numel()
    if not 0 <= num_decode_tokens <= count:
        raise ValueError("Invalid decode prefix")
    if count == 0:
        return positions.new_empty((0, pool_size), dtype=torch.int64)
    rows = torch.arange(count, device=positions.device, dtype=torch.int64)
    request_blocks = torch.div(tail_slots, ring_size, rounding_mode="floor")
    starts = torch.ones(count, dtype=torch.bool, device=positions.device)
    starts[1:] = request_blocks[1:] != request_blocks[:-1]
    start_rows = torch.cummax(torch.where(starts, rows, 0), dim=0).values
    start_positions = positions[start_rows]
    boundary = (
        (rows >= num_decode_tokens)
        & (tail_slots >= 0)
        & (positions >= 0)
        & (start_positions % pool_size != 0)
        & (positions // pool_size == start_positions // pool_size)
    )
    requests = torch.unique(start_rows[boundary])
    if requests.numel() == 0:
        return rows.new_empty((0, pool_size))
    result = rows.new_full((requests.numel(), pool_size), -1)
    selected = rows[boundary]
    request_rows = torch.searchsorted(requests, start_rows[selected])
    offsets = selected - start_rows[selected]
    result[request_rows, offsets] = selected
    return result


def complete_pool_mask(slots, positions, request_ids, pool_size):
    """A local pool must be contiguous in both request and absolute position."""
    if slots.shape != positions.shape or slots.shape != request_ids.shape:
        raise ValueError("Pool metadata shapes differ")
    if slots.ndim != 1 or pool_size <= 0:
        raise ValueError("Invalid complete-pool metadata")
    end = torch.arange(slots.numel(), device=slots.device)
    begin = (end - pool_size + 1).clamp_min(0)
    return (
        (slots >= 0)
        & (end >= pool_size - 1)
        & (positions[end] - positions[begin] == pool_size - 1)
        & (request_ids[end] == request_ids[begin])
        & (request_ids[end] >= 0)
    )
