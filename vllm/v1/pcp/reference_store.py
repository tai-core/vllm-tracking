# SPDX-License-Identifier: Apache-2.0
"""Byte-preserving collective reference for exclusively owned PCP pages."""

from __future__ import annotations

from typing import Protocol

import torch

from vllm.v1.pcp.layout import PCPPageLayout
from vllm.v1.pcp.pages import PCPMaterializedPages


class PCPGatherGroup(Protocol):
    rank_in_group: int
    world_size: int

    def all_gather(self, tensor: torch.Tensor, dim: int = 0) -> torch.Tensor: ...


class PCPPageStore:
    """Persistent local bytes plus a bounded per-layer collective workspace.

    All ranks must pass the same sorted logical page list to materialize.
    write_rows accepts rank-local producers, padded to the same row count.
    Each valid logical row must have exactly one producer.
    """

    def __init__(
        self,
        layout: PCPPageLayout,
        logical_capacity: int,
        rows_per_page: int,
        row_bytes: int,
        device: torch.device,
        group: PCPGatherGroup,
        max_materialize_pages: int,
    ) -> None:
        if group.world_size != layout.size:
            raise ValueError("Layout and collective group sizes differ")
        if min(rows_per_page, row_bytes, max_materialize_pages) <= 0:
            raise ValueError("Storage geometry and workspace bound must be positive")
        if rows_per_page not in (
            layout.page_tokens,
            layout.page_tokens // layout.pool_tokens,
        ):
            raise ValueError("Rows must use token or compressed-pool coordinates")
        self.layout = layout
        self.group = group
        self.logical_capacity = logical_capacity
        self.rows_per_page = rows_per_page
        self.row_bytes = row_bytes
        self.max_materialize_pages = max_materialize_pages
        self.storage = torch.zeros(
            (
                layout.local_capacity(logical_capacity, group.rank_in_group),
                rows_per_page,
                row_bytes,
            ),
            dtype=torch.uint8,
            device=device,
        )

    @property
    def persistent_bytes(self) -> int:
        return self.storage.numel()

    def write_rows(
        self,
        logical_slots: torch.Tensor,
        row_bytes: torch.Tensor,
    ) -> None:
        if logical_slots.ndim != 1 or logical_slots.dtype != torch.int64:
            raise ValueError("Write slots must be a one-dimensional int64 tensor")
        if row_bytes.dtype != torch.uint8 or row_bytes.shape != (
            logical_slots.numel(),
            self.row_bytes,
        ):
            raise ValueError("Writes must supply one opaque byte row per slot")
        if row_bytes.device != self.storage.device or (
            logical_slots.device != self.storage.device
        ):
            raise ValueError("Write slots, bytes and storage must share a device")
        # Keep byte encodings intact: FP8 values/scales must never be summed.
        slots = self.group.all_gather(logical_slots.contiguous(), dim=0)
        values = self.group.all_gather(row_bytes.contiguous(), dim=0)
        valid = slots >= 0
        valid_slots = slots[valid]
        if bool((valid_slots >= self.logical_capacity * self.rows_per_page).any()):
            raise ValueError("Write exceeds the logical pool capacity")
        if valid_slots.unique().numel() != valid_slots.numel():
            raise ValueError("A PCP row has multiple producers")
        pages = torch.div(slots.clamp_min(0), self.rows_per_page, rounding_mode="floor")
        owned = valid & (pages.remainder(self.layout.size) == self.group.rank_in_group)
        physical = torch.div(pages[owned], self.layout.size, rounding_mode="floor")
        offsets = slots[owned].remainder(self.rows_per_page)
        self.storage[physical, offsets] = values[owned]

    def materialize(self, logical_pages: torch.Tensor) -> PCPMaterializedPages:
        if logical_pages.ndim != 1 or logical_pages.dtype != torch.int64:
            raise ValueError("Page list must be a one-dimensional int64 tensor")
        count = logical_pages.numel()
        if count > self.max_materialize_pages:
            raise ValueError("Materialize exceeds the configured workspace bound")
        if logical_pages.device != self.storage.device:
            raise ValueError("Page list must be on the storage device")
        if bool(((logical_pages < 0) | (logical_pages >= self.logical_capacity)).any()):
            raise ValueError("Page list exceeds the logical pool capacity")
        if count > 1 and bool((logical_pages[1:] <= logical_pages[:-1]).any()):
            raise ValueError("Materialize pages must be sorted and unique")
        if count == 0:
            return PCPMaterializedPages(
                logical_pages,
                self.storage.new_empty((0, self.rows_per_page, self.row_bytes)),
            )
        owners = logical_pages.remainder(self.layout.size)
        rank_indices = [
            torch.nonzero(owners == rank, as_tuple=True)[0]
            for rank in range(self.layout.size)
        ]
        # Gather only actual owner pages; never allocate CP complete copies.
        width = max(indices.numel() for indices in rank_indices)
        contribution = self.storage.new_zeros(
            (width, self.rows_per_page, self.row_bytes)
        )
        owned_indices = rank_indices[self.group.rank_in_group]
        physical = torch.div(
            logical_pages[owned_indices], self.layout.size, rounding_mode="floor"
        )
        contribution[: owned_indices.numel()] = self.storage[physical]
        gathered = self.group.all_gather(contribution, dim=0).view(
            self.layout.size, width, self.rows_per_page, self.row_bytes
        )
        dense = self.storage.new_empty((count, self.rows_per_page, self.row_bytes))
        for rank, indices in enumerate(rank_indices):
            dense[indices] = gathered[rank, : indices.numel()]
        return PCPMaterializedPages(logical_pages, dense)

    def zero_pages(self, logical_pages: tuple[int, ...]) -> None:
        """Clear newly allocated pages only on their physical owners."""
        if any(page < 0 or page >= self.logical_capacity for page in logical_pages):
            raise ValueError("Zeroing exceeds the logical pool capacity")
        physical = [
            page // self.layout.size
            for page in logical_pages
            if page % self.layout.size == self.group.rank_in_group
        ]
        if physical:
            indices = torch.tensor(physical, device=self.storage.device)
            self.storage[indices] = 0

    def copy_pages(self, copies: tuple[tuple[int, int], ...]) -> None:
        """Copy-on-write with snapshot semantics, including cross-rank cycles.

        Every rank supplies the same copy list. Sources are retained until
        the collective finishes; all reads precede all destination writes.
        """
        if not copies:
            return
        if len({destination for _, destination in copies}) != len(copies):
            raise ValueError("Copy destinations must be unique")
        if any(
            page < 0 or page >= self.logical_capacity
            for pair in copies
            for page in pair
        ):
            raise ValueError("Copy exceeds the logical pool capacity")
        source_pages = sorted({source for source, _ in copies})
        sources = torch.tensor(
            source_pages, device=self.storage.device, dtype=torch.int64
        )
        snapshot = self.materialize(sources)
        source_index = {page: index for index, page in enumerate(source_pages)}
        for source, destination in copies:
            owner, physical = self.layout.locate(destination)
            if owner == self.group.rank_in_group:
                self.storage[physical].copy_(snapshot.data[source_index[source]])
