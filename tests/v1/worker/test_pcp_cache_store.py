# SPDX-License-Identifier: Apache-2.0
"""Exercise physical storage over a real two-process collective group."""

import tempfile
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.v1.pcp.layout import PCPPageLayout
from vllm.v1.pcp.reference_store import PCPPageStore


class Group:
    def __init__(self, rank, size):
        self.rank_in_group = rank
        self.world_size = size

    def all_gather(self, tensor, dim=0):
        parts = [torch.empty_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(parts, tensor)
        return torch.cat(parts, dim=dim)


def exercise_store(rank, size, rendezvous, device_kind):
    if device_kind == "musa":
        import torch_musa  # noqa: F401

        torch.musa.set_device(rank)
    device = (
        torch.device(device_kind, rank) if device_kind != "cpu" else torch.device("cpu")
    )
    dist.init_process_group(
        "gloo" if device_kind == "cpu" else "mccl",
        init_method=rendezvous,
        rank=rank,
        world_size=size,
    )
    try:
        layout = PCPPageLayout(size, 4, 2)
        store = PCPPageStore(layout, 5, 2, 9, device, Group(rank, size), 5)
        # Deliberately make producer ownership different from physical ownership.
        slots = torch.tensor(
            [0, 4, 8, -1] if rank else [2, 6, -1, -1],
            dtype=torch.int64,
            device=device,
        )
        values = (
            slots[:, None].clamp_min(0) * 17 + torch.arange(9, device=device)[None, :]
        ).to(torch.uint8)
        store.write_rows(slots, values)
        assert store.persistent_bytes == layout.local_capacity(5, rank) * 2 * 9
        pages = torch.arange(5, device=device, dtype=torch.int64)
        result = store.materialize(pages)
        expected = torch.zeros((5, 2, 9), dtype=torch.uint8, device=device)
        expected[:, 0] = (
            torch.arange(5, device=device)[:, None] * 34
            + torch.arange(9, device=device)[None, :]
        ).to(torch.uint8)
        torch.testing.assert_close(result.data, expected, rtol=0, atol=0)
        # Sparse working sets must remap by page identity, not source pool size.
        subset = store.materialize(pages[[1, 4]])
        mapped = subset.remap_slots(torch.tensor([2, 3, 8, 9, -1], device=device), 2)
        torch.testing.assert_close(
            mapped, torch.tensor([0, 1, 2, 3, -1], device=device)
        )
        torch.testing.assert_close(subset.data, expected[[1, 4]], rtol=0, atol=0)
        # COW is a simultaneous snapshot even when ownership crosses ranks.
        store.copy_pages(((0, 1), (1, 0)))
        expected[[0, 1]] = expected[[1, 0]]
        torch.testing.assert_close(
            store.materialize(pages).data, expected, rtol=0, atol=0
        )
        store.zero_pages((4,))
        expected[4] = 0
        torch.testing.assert_close(
            store.materialize(pages).data, expected, rtol=0, atol=0
        )
        with pytest.raises(ValueError, match="destinations"):
            store.copy_pages(((0, 1), (2, 1)))
        # Duplicate writers must fail uniformly, before changing persistent bytes.
        duplicate = torch.tensor([0], device=device, dtype=torch.int64)
        with pytest.raises(ValueError, match="multiple producers"):
            store.write_rows(duplicate, values[:1])
        torch.testing.assert_close(
            store.materialize(pages).data, expected, rtol=0, atol=0
        )
    finally:
        dist.destroy_process_group()


def test_collective_storage_routes_writes_and_materializes_exact_bytes(tmp_path):
    mp.spawn(
        exercise_store,
        args=(2, (tmp_path / "rendezvous").as_uri(), "cpu"),
        nprocs=2,
        join=True,
    )


def test_workspace_limit_is_enforced_before_collectives():
    store = PCPPageStore(
        PCPPageLayout(1, 4), 4, 4, 8, torch.device("cpu"), Group(0, 1), 1
    )
    with pytest.raises(ValueError, match="workspace bound"):
        store.materialize(torch.tensor([0, 1], dtype=torch.int64))


if __name__ == "__main__":
    import sys

    with tempfile.TemporaryDirectory(prefix="vllm-pcp-") as work:
        mp.spawn(
            exercise_store,
            args=(2, (Path(work) / "rendezvous").as_uri(), sys.argv[1]),
            nprocs=2,
            join=True,
        )
