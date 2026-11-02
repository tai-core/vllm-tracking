# SPDX-License-Identifier: Apache-2.0
"""Byte-preserving current-row composition and unique-owner cache writes."""

from dataclasses import dataclass

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.pcp.pages import PCPMaterializedPages


@dataclass(frozen=True)
class PCPCurrentRowPlan:
    logical_slots: torch.Tensor
    dense_slots: torch.Tensor
    full_current_pages: torch.Tensor | None


def plan_current_rows(pages, logical_slots, rows_per_page, *, include_full_pages=True):
    """Describe current rows without treating a partial page as fully current."""
    valid_slots = logical_slots[logical_slots >= 0]
    if valid_slots.unique().numel() != valid_slots.numel():
        raise ValueError("Current PCP rows must have unique producers")
    full_current_pages = None
    if include_full_pages:
        current_pages, counts = torch.unique(
            valid_slots // rows_per_page, return_counts=True
        )
        full_current_pages = current_pages[counts == rows_per_page]
    dense_slots = PCPMaterializedPages(pages, pages).remap_slots(
        logical_slots, rows_per_page
    )
    return PCPCurrentRowPlan(logical_slots, dense_slots, full_current_pages)


@triton.jit
def _scatter_current_rows(
    encoded,
    logical_slots,
    dense_slots,
    dense,
    persistent,
    ENCODED_STRIDE: tl.constexpr,
    STORAGE_STRIDE: tl.constexpr,
    ROW_BYTES: tl.constexpr,
    ROWS_PER_PAGE: tl.constexpr,
    FACTOR: tl.constexpr,
    RANK: tl.constexpr,
    SIZE: tl.constexpr,
    HEAD_BYTES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    logical = tl.load(logical_slots + row).to(tl.int64)
    target = tl.load(dense_slots + row).to(tl.int64)
    if logical >= 0:
        byte = tl.arange(0, BLOCK)
        value = tl.load(
            encoded + row * ENCODED_STRIDE + byte,
            mask=byte < ROW_BYTES,
            other=0,
        )
        offset = logical % ROWS_PER_PAGE
        if HEAD_BYTES == ROW_BYTES:
            in_page = offset * ROW_BYTES + byte
        else:
            in_page = tl.where(
                byte < HEAD_BYTES,
                offset * HEAD_BYTES + byte,
                ROWS_PER_PAGE * HEAD_BYTES
                + offset * (ROW_BYTES - HEAD_BYTES)
                + byte
                - HEAD_BYTES,
            )
        tl.store(
            dense + (target // ROWS_PER_PAGE) * ROWS_PER_PAGE * ROW_BYTES + in_page,
            value,
            mask=(byte < ROW_BYTES) & (target >= 0),
        )
        page = logical // ROWS_PER_PAGE
        manager_page = page // FACTOR
        owner = manager_page % SIZE
        physical = manager_page // SIZE * FACTOR + page % FACTOR
        tl.store(
            persistent + physical * STORAGE_STRIDE + in_page,
            value,
            mask=(byte < ROW_BYTES) & (owner == RANK),
        )


def scatter_current_rows(store, dense, plan, encoded, *, head_bytes=None):
    """One byte scatter fills attention input and writes only owned cache rows.

    With head_bytes < row_bytes, persistent pages contain separate data and
    scale planes; incoming current rows remain compact [data | scale] records.
    """
    rows = store.storage.shape[1]
    row_bytes = store.storage[0].numel() * store.storage.element_size() // rows
    head_bytes = row_bytes if head_bytes is None else head_bytes
    if not 0 < head_bytes <= row_bytes:
        raise ValueError("Invalid current-row byte layout")
    if encoded.dtype != torch.uint8 or encoded.shape != (
        plan.logical_slots.numel(),
        row_bytes,
    ):
        raise ValueError("Current rows do not match native page geometry")
    if store.ipc_reader is not None:
        store.ipc_reader.wait_for_reuse()
    if not encoded.shape[0]:
        return
    if encoded.device.type != "cpu":
        _scatter_current_rows[(encoded.shape[0],)](
            encoded,
            plan.logical_slots,
            plan.dense_slots,
            dense.view(torch.uint8),
            store.storage.view(torch.uint8),
            ENCODED_STRIDE=encoded.stride(0),
            STORAGE_STRIDE=store.storage.stride(0) * store.storage.element_size(),
            ROW_BYTES=row_bytes,
            ROWS_PER_PAGE=rows,
            FACTOR=store.factor,
            RANK=store.group.rank_in_group,
            SIZE=store.group.world_size,
            HEAD_BYTES=head_bytes,
            BLOCK=triton.next_power_of_2(row_bytes),
        )
        return
    # Byte-level correctness reference for both native MLA and index planes.
    byte = torch.arange(row_bytes)
    for row, logical in enumerate(plan.logical_slots.tolist()):
        if logical < 0:
            continue
        offset = logical % rows
        in_page = torch.where(
            byte < head_bytes,
            offset * head_bytes + byte,
            rows * head_bytes + offset * (row_bytes - head_bytes) + byte - head_bytes,
        )
        if head_bytes == row_bytes:
            in_page = offset * row_bytes + byte
        target = int(plan.dense_slots[row])
        dense[target // rows].view(torch.uint8).reshape(-1)[in_page] = encoded[row]
        owner, physical = store.locations(torch.tensor(logical // rows))
        if int(owner) == store.group.rank_in_group:
            store.storage[int(physical)].view(torch.uint8).reshape(-1)[in_page] = (
                encoded[row]
            )
