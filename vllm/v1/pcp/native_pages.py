# SPDX-License-Identifier: Apache-2.0
"""Native-format PCP pages backed by the worker's shared physical pool."""

from __future__ import annotations

from copy import copy
from dataclasses import fields, is_dataclass

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.pcp.plan import get_pcp_forward_plan, get_pcp_plan_cache


def _map_block_tables(value, transform, seen=None):
    """Copy metadata containers, preserving all non-block-table tensors."""
    if seen is None:
        seen = {}
    if isinstance(value, torch.Tensor) or value is None:
        return value
    if id(value) in seen:
        return seen[id(value)]
    if isinstance(value, list):
        result = [_map_block_tables(item, transform, seen) for item in value]
    elif isinstance(value, tuple):
        result = tuple(_map_block_tables(item, transform, seen) for item in value)
    elif isinstance(value, dict):
        result = {
            name: transform(item, value)
            if name == "block_table"
            else _map_block_tables(item, transform, seen)
            for name, item in value.items()
        }
    elif is_dataclass(value):
        result = copy(value)
        seen[id(value)] = result
        for field in fields(value):
            item = getattr(value, field.name)
            mapped = (
                transform(item, value)
                if field.name == "block_table"
                else (_map_block_tables(item, transform, seen))
            )
            object.__setattr__(result, field.name, mapped)
    else:
        return value
    seen[id(value)] = result
    return result


@triton.jit
def _scatter_valid_page_ids(compact, offsets, output, stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    start = tl.load(offsets + row)
    count = tl.load(offsets + row + 1) - start
    for base in range(0, count, BLOCK):
        columns = base + tl.arange(0, BLOCK)
        values = tl.load(compact + start + columns, columns < count, other=0)
        tl.store(output + row * stride + columns, values, columns < count)


class PCPNativePages:
    """Borrow native kernel pages without changing values/scales byte layout.

    kernel_blocks_per_page preserves scheduler-page ownership when backends
    subdivide one manager block into multiple kernel blocks.
    """

    def __init__(
        self,
        storage,
        logical_blocks,
        group,
        kernel_blocks_per_page=1,
        max_materialize_pages=1,
        kernel_pages_per_block=None,
        tokens_per_state=1,
    ):
        if kernel_blocks_per_page <= 0 or max_materialize_pages <= 0:
            raise ValueError("Invalid PCP page geometry")
        if tokens_per_state <= 0:
            raise ValueError("PCP tokens_per_state must be positive")
        self.tokens_per_state = tokens_per_state
        self.storage = storage
        self.logical_blocks = logical_blocks
        self.group = group
        self.factor = kernel_blocks_per_page
        self.kernel_pages_per_block = (
            kernel_blocks_per_page
            if kernel_pages_per_block is None
            else kernel_pages_per_block
        )
        if not 0 < self.kernel_pages_per_block <= self.factor:
            raise ValueError("Invalid native pages per scheduler block")
        self.max_materialize_pages = max_materialize_pages
        self.ipc_reader = None

    def locations(self, kernel_pages):
        manager_pages = kernel_pages // self.factor
        owners = manager_pages % self.group.world_size
        physical = (
            manager_pages // self.group.world_size * self.factor
            + kernel_pages % self.factor
        )
        return owners, physical

    def write_slots(self, logical_slots, rows_per_kernel_page):
        if self.ipc_reader is not None:
            self.ipc_reader.wait_for_reuse()
        valid = logical_slots >= 0
        pages = logical_slots.clamp_min(0) // rows_per_kernel_page
        owners, physical = self.locations(pages)
        return torch.where(
            valid & (owners == self.group.rank_in_group),
            physical * rows_per_kernel_page + logical_slots % rows_per_kernel_page,
            -1,
        )

    def _valid_tables(self, metadata):
        """Read only each row's live prefix, never the capacity-sized tail."""
        descriptors = []

        def describe(table, container):
            if not isinstance(table, torch.Tensor):
                return table
            if table.ndim != 2:
                raise ValueError("PCP block tables must have two dimensions")

            def field(name):
                return (
                    container.get(name)
                    if isinstance(container, dict)
                    else getattr(container, name, None)
                )

            cumulative = field("local_cu_seq_lens")
            if cumulative is None:
                cumulative = field("cu_seq_lens")
            if cumulative is not None:
                # Indexer chunks already count completed compressed rows.
                lengths = cumulative[1:] - cumulative[:-1]
            else:
                lengths = field("seq_lens")
                if lengths is None:
                    # Dense MLA prefill reads only its paged history; current
                    # K/V comes from the projections. No history means no reads.
                    context = field("chunked_context")
                    if context is not None:
                        lengths = context.context_lens
                    elif (
                        "chunked_context" in container
                        if isinstance(container, dict)
                        else hasattr(container, "chunked_context")
                    ) and field("query_start_loc") is not None:
                        lengths = table.new_zeros(table.shape[0])
                    else:
                        raise ValueError("PCP block table is missing effective lengths")
                if lengths.ndim == 2:
                    # Decode can have one effective length per speculative token.
                    lengths = lengths.max(dim=-1).values
                if container is metadata:
                    # Indexer root metadata retains uncompressed token lengths.
                    # An unfinished pool lives in the separate tail cache.
                    lengths = lengths // self.tokens_per_state
            if lengths.ndim != 1 or lengths.numel() != table.shape[0]:
                raise ValueError("PCP block-table rows and effective lengths disagree")
            counts = (
                lengths.to(torch.int64) + self.storage.shape[1] - 1
            ) // self.storage.shape[1]
            counts = torch.where(lengths < 0, -1, counts)
            descriptors.append((container, table, counts))
            return table

        _map_block_tables(metadata, describe)
        if not descriptors:
            return {}
        # One small length transfer per geometry/forward, shared by all layers.
        # No table data (including stale out-of-range IDs) is inspected here.
        all_counts = torch.cat([counts for _, _, counts in descriptors]).tolist()
        offset = 0
        result = {}
        for container, table, _ in descriptors:
            counts = all_counts[offset : offset + table.shape[0]]
            offset += table.shape[0]
            if any(count < 0 or count > table.shape[1] for count in counts):
                raise ValueError("PCP effective length exceeds block-table capacity")
            pieces = [table[row, :count] for row, count in enumerate(counts) if count]
            compact = (
                torch.cat(pieces).to(torch.int64)
                if pieces
                else torch.empty(0, dtype=torch.int64, device=table.device)
            )
            result[id(container)] = (counts, compact)
        return result

    def gather_page_ids(self, metadata, *, valid_tables=None):
        if valid_tables is None:
            valid_tables = self._valid_tables(metadata)
        tables = [compact for _, compact in valid_tables.values()]
        local = (
            torch.unique(torch.cat(tables))
            if tables
            else torch.empty(0, dtype=torch.int64, device=self.storage.device)
        )
        local = local[local >= 0]
        counts = self.group.all_gather(
            torch.tensor(
                [local.numel()], device=self.storage.device, dtype=torch.int64
            ),
            dim=0,
        )
        width = max(1, int(counts.max().item()))
        padded = torch.full((width,), -1, device=self.storage.device, dtype=torch.int64)
        padded[: local.numel()] = local
        pages = torch.unique(self.group.all_gather(padded, dim=0))
        pages = pages[pages >= 0]
        if pages.numel() > self.max_materialize_pages:
            raise RuntimeError("PCP native materialization exceeds workspace capacity")
        if bool(
            (
                (pages >= self.logical_blocks * self.factor)
                | (pages % self.factor >= self.kernel_pages_per_block)
            ).any()
        ):
            raise ValueError("Metadata addresses a page outside the logical pool")
        return pages

    def _metadata_plan(self, metadata):
        plans = get_pcp_plan_cache()
        key = (
            "native_pages",
            id(metadata),
            id(self.group),
            self.factor,
            self.kernel_pages_per_block,
            self.max_materialize_pages,
            self.storage.shape[1],
            self.tokens_per_state,
        )
        if plans is not None:
            cached = plans.get(key)
            if cached is not None:
                _, pages, mapped, owners, physical = cached
                return pages, mapped, owners, physical

        valid_tables = self._valid_tables(metadata)
        pages = self.gather_page_ids(metadata, valid_tables=valid_tables)

        def remap(table, container):
            if not isinstance(table, torch.Tensor):
                return table
            counts, compact = valid_tables[id(container)]
            mapped = torch.searchsorted(pages, compact.clamp_min(0))
            mapped = torch.where(compact >= 0, mapped, -1).to(table.dtype)
            width = max(counts, default=0)
            result = table.new_zeros((len(counts), width))
            if not compact.numel():
                return result
            offsets = [0]
            for count in counts:
                offsets.append(offsets[-1] + count)
            if table.device.type == "cpu":
                for row, count in enumerate(counts):
                    result[row, :count].copy_(mapped[offsets[row] : offsets[row + 1]])
            else:
                row_offsets = torch.tensor(
                    offsets, dtype=torch.int64, device=table.device
                )
                _scatter_valid_page_ids[(len(counts),)](
                    mapped,
                    row_offsets,
                    result,
                    result.stride(0),
                    BLOCK=256,
                )
            return result

        mapped = _map_block_tables(metadata, remap)
        owners, physical = self.locations(pages)
        if plans is not None:
            # Own the original metadata for the forward's lifetime: tensor
            # allocation/address reuse must not turn into a stale cache hit.
            plans[key] = (metadata, pages, mapped, owners, physical)
        return pages, mapped, owners, physical

    def materialize(self, metadata):
        forward_plan = get_pcp_forward_plan()
        if forward_plan is not None:
            cached = forward_plan.pending_current.pop(id(self), None)
            if cached is not None:
                original, dense, mapped = cached
                if original is not metadata:
                    raise RuntimeError(
                        "PCP current rows consumed with different metadata"
                    )
                return dense, mapped
        pages, mapped, owners, physical = self._metadata_plan(metadata)
        return self._materialize_plan(pages, owners, physical), mapped

    def compose_current(
        self,
        metadata,
        logical_slots,
        encoded,
        *,
        head_bytes=None,
        sources=None,
        source_width=None,
    ):
        """Read historical bytes once and splice already encoded current rows."""
        from vllm.v1.pcp.current import plan_current_rows, scatter_current_rows

        forward_plan = get_pcp_forward_plan()
        if forward_plan is None:
            raise RuntimeError("PCP current composition requires a forward context")
        pending = forward_plan.pending_current
        if id(self) in pending:
            raise RuntimeError(
                "PCP current rows overwritten before attention consumes them"
            )
        plans = forward_plan.routes
        pages, mapped, owners, physical = self._metadata_plan(metadata)
        fused_ipc = sources is not None and self.ipc_reader is not None
        key = (
            "current_rows",
            id(metadata),
            id(self.group),
            self.factor,
            self.kernel_pages_per_block,
            self.storage.shape[1],
            self.tokens_per_state,
            source_width,
            encoded.shape[1],
            head_bytes,
            fused_ipc,
        )
        cached = plans.get(key)
        if cached is None:
            plan = plan_current_rows(
                pages,
                logical_slots,
                self.storage.shape[1],
                include_full_pages=not fused_ipc,
            )
            history_owners = None
            if not fused_ipc:
                history_owners = torch.where(
                    torch.isin(pages, plan.full_current_pages), -1, owners
                )
            source_map = None
            if sources is not None:
                if sources.shape != logical_slots.shape or not source_width:
                    raise ValueError("Invalid PCP producer-row descriptor")
                valid = logical_slots >= 0
                if bool(
                    (
                        valid
                        & (
                            (sources < 0)
                            | (sources >= source_width * self.group.world_size)
                        )
                    ).any()
                ):
                    raise ValueError("PCP producer row is outside staging capacity")
                source_map = logical_slots.new_full(
                    (pages.numel() * self.storage.shape[1],), -1
                )
                source_map[plan.dense_slots[valid]] = sources[valid]
            plans[key] = (metadata, plan, history_owners, source_map, sources)
        else:
            _, plan, history_owners, source_map, _ = cached
        if fused_ipc:
            row_bytes = (
                self.storage[0].numel()
                * self.storage.element_size()
                // self.storage.shape[1]
            )
            dense = self.ipc_reader.compose_current(
                self.storage,
                owners,
                physical,
                source_map,
                encoded,
                source_width,
                row_bytes if head_bytes is None else head_bytes,
            )
        else:
            if sources is not None:
                padding = encoded.new_zeros((source_width, encoded.shape[1]))
                padding[: encoded.shape[0]] = encoded
                encoded = self.group.all_gather(padding, dim=0)[sources.clamp_min(0)]
            dense = self._materialize_plan(pages, history_owners, physical)
            scatter_current_rows(self, dense, plan, encoded, head_bytes=head_bytes)
        pending[id(self)] = (metadata, dense, mapped)

    def workspace_bytes(self, page_count, max_owner_pages=None):
        """Upper bound including gather output, local selection and dense view."""
        if page_count < 0 or page_count > self.max_materialize_pages:
            raise ValueError("Invalid native materialization size")
        page_bytes = self.storage[0].numel() * self.storage.element_size()
        owner_bound = min(
            page_count,
            ((self.logical_blocks + self.group.world_size - 1) // self.group.world_size)
            * self.kernel_pages_per_block,
        )
        width = max(1, owner_bound if max_owner_pages is None else max_owner_pages)
        if width > max(1, owner_bound):
            raise ValueError("Owner contribution exceeds physical capacity")
        # During all-gather local staging and gathered pages coexist. During
        # packing, the gathered tensor and final dense tensor coexist.
        peak_pages = max(2 * width, (self.group.world_size + 1) * width + page_count)
        return peak_pages * page_bytes

    def materialize_ids(self, pages, *, plan_key=None):
        """Read fresh bytes; a plan key declares immutable IDs for this forward."""
        plans = get_pcp_plan_cache(plan_key)
        key = (
            "native_ids",
            plan_key,
            id(self.group),
            self.factor,
            self.kernel_pages_per_block,
            self.logical_blocks,
            self.max_materialize_pages,
        )
        if plans is not None:
            cached = plans.get(key)
            if cached is not None:
                original_pages, owners, physical = cached
                if original_pages.numel() != pages.numel():
                    raise RuntimeError("PCP page-ID plan changed within one forward")
                return self._materialize_plan(pages, owners, physical)
        if pages.numel() > self.max_materialize_pages:
            raise RuntimeError("PCP native materialization exceeds workspace capacity")
        if bool(
            (
                (pages < 0)
                | (pages >= self.logical_blocks * self.factor)
                | (pages % self.factor >= self.kernel_pages_per_block)
            ).any()
        ):
            raise ValueError("PCP materialization addresses outside the logical pool")
        owners, physical = self.locations(pages)
        if plans is not None:
            plans[key] = (pages, owners, physical)
        return self._materialize_plan(pages, owners, physical)

    def _materialize_plan(self, pages, owners, physical):
        """Read pages whose immutable descriptor was already validated."""
        if self.ipc_reader is not None:
            return self.ipc_reader.materialize(self.storage, owners, physical)
        indices = [
            torch.nonzero(owners == rank, as_tuple=True)[0]
            for rank in range(self.group.world_size)
        ]
        width = max(1, max(index.numel() for index in indices))
        # Copy native pages into contiguous staging, including opaque FP8 scales.
        page_shape = self.storage.shape[1:]
        page_bytes = self.storage[0].numel() * self.storage.element_size()
        staging = torch.zeros(
            (width, page_bytes), dtype=torch.uint8, device=self.storage.device
        )
        own = indices[self.group.rank_in_group]
        selected = self.storage.view(torch.uint8)[physical[own]].contiguous()
        staging[: own.numel()] = selected.view(torch.uint8).reshape(
            own.numel(), page_bytes
        )
        del selected
        gathered = self.group.all_gather(staging, dim=0).view(
            self.group.world_size, width, page_bytes
        )
        dense = torch.zeros(
            (pages.numel(), page_bytes), dtype=torch.uint8, device=self.storage.device
        )
        for rank, index in enumerate(indices):
            dense[index] = gathered[rank, : index.numel()]
        dense = dense.view(self.storage.dtype).reshape(pages.numel(), *page_shape)

        return dense

    def commit_pages(self, pages, dense):
        """Commit an identically updated working set on its unique owners."""
        if self.ipc_reader is not None:
            self.ipc_reader.wait_for_reuse()
        owners, physical = self.locations(pages)
        owned = torch.nonzero(owners == self.group.rank_in_group, as_tuple=True)[0]
        self.storage.view(torch.uint8)[physical[owned]] = dense.view(torch.uint8)[owned]

    @staticmethod
    def remap_slots(pages, slots, rows_per_page):
        from vllm.v1.pcp.pages import PCPMaterializedPages

        return PCPMaterializedPages(pages, pages).remap_slots(slots, rows_per_page)

    def zero_blocks(self, block_ids):
        if self.ipc_reader is not None:
            self.ipc_reader.wait_for_reuse()
        if any(block < 0 or block >= self.logical_blocks for block in block_ids):
            raise ValueError("PCP clear addresses outside the logical pool")
        own = [
            block // self.group.world_size
            for block in block_ids
            if block % self.group.world_size == self.group.rank_in_group
        ]
        if own:
            self.storage[torch.tensor(own, device=self.storage.device)] = 0

    def copy_blocks(self, copies):
        if not copies:
            return
        if self.factor != 1:
            raise ValueError("Whole-pool maintenance requires manager-sized pages")
        if any(
            block < 0 or block >= self.logical_blocks
            for pair in copies
            for block in pair
        ):
            raise ValueError("PCP copy addresses outside the logical pool")
        sources = sorted({source for source, _ in copies})
        pages = torch.tensor(sources, dtype=torch.int64, device=self.storage.device)
        snapshot = self.materialize_ids(pages)
        lookup = {source: offset for offset, source in enumerate(sources)}
        for source, destination in copies:
            if destination % self.group.world_size == self.group.rank_in_group:
                self.storage[destination // self.group.world_size].copy_(
                    snapshot[lookup[source]]
                )
