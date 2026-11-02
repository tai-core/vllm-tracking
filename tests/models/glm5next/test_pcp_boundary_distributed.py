# SPDX-License-Identifier: Apache-2.0
"""Exact cross-chunk kpool completion with distinct compute/cache owners."""

import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


class Group:
    def __init__(self, rank, size, cpu):
        self.rank_in_group, self.world_size, self.cpu_group = rank, size, cpu

    def all_gather(self, tensor, dim=0):
        outputs = [torch.empty_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(outputs, tensor)
        return torch.cat(outputs, dim=dim)


def run_rank(rank, size, rendezvous):
    import torch_musa  # noqa: F401
    from vllm_musa.distributed.device_communicators.pcp_native_pages import PCPNativeIPC

    from vllm.forward_context import override_forward_context
    from vllm.models.glm5next.nvidia.ops import kpool_compress as kernels
    from vllm.models.glm5next.nvidia.pcp_kpool import (
        complete_boundary_pools,
        current_pool_descriptors,
    )
    from vllm.v1.pcp.native_pages import PCPNativePages
    from vllm.v1.worker.gpu import pcp_manager

    torch.musa.set_device(rank)
    device = torch.device("musa", rank)
    dist.init_process_group("mccl", init_method=rendezvous, rank=rank, world_size=size)
    cpu = dist.new_group(backend="gloo")
    group = Group(rank, size, cpu)
    pcp_manager.get_pcp_group = lambda: group
    readers = []
    try:
        torch.manual_seed(113)
        keys = torch.randn(3, 4, 128, dtype=torch.bfloat16).to(device)
        gates = torch.randn(3, 4, 128, dtype=torch.bfloat16).to(device)
        ape = torch.randn(4, 128).to(device)
        index_ids = [12, 14, 8]
        tail_ids = [11, 13, 15]
        baseline = torch.zeros(16, 16, 132, dtype=torch.uint8, device=device)
        kernels.kpool_compress_and_write_cache(
            baseline,
            keys,
            gates,
            ape,
            torch.tensor([page * 16 for page in index_ids], device=device),
            pool_size=4,
        )
        for use_ipc, position_base in (
            (False, 0),
            (True, 0),
            (False, 700),
            (True, 700),
        ):
            index = torch.zeros(2, 16, 132, dtype=torch.uint8, device=device)
            tail = torch.zeros(2, 2, 8, 128, dtype=torch.bfloat16, device=device)
            index_pages = PCPNativePages(index, 16, group, max_materialize_pages=16)
            tail_pages = PCPNativePages(tail, 16, group, max_materialize_pages=16)
            if use_ipc:
                readers = [PCPNativeIPC(index, group), PCPNativeIPC(tail, group)]
                index_pages.ipc_reader, tail_pages.ipc_reader = readers
                readers[0].register_current_staging(8, 656)
            prefix = [1, 2, 3]
            for request, start in enumerate(prefix):
                if tail_ids[request] % size == rank:
                    block = tail_ids[request] // size
                    offset = position_base % 8
                    tail[block, 0, offset : offset + start] = keys[request, :start]
                    tail[block, 1, offset : offset + start] = gates[request, :start]
            # First step intentionally does not finish two of the three pools.
            for step in range(2):
                pairs = [
                    (request, pos)
                    for request, start in enumerate(prefix)
                    for pos in range(start, min(4, start + 1) if step == 0 else 4)
                ]
                positions = torch.tensor(
                    [position_base + pos for _, pos in pairs], device=device
                )
                tail_slots = torch.tensor(
                    [
                        tail_ids[request] * 8 + (position_base + pos) % 8
                        for request, pos in pairs
                    ],
                    device=device,
                )
                pool_slots = torch.tensor(
                    [
                        index_ids[request] * 16 if pos == 3 else -1
                        for request, pos in pairs
                    ],
                    device=device,
                )
                local_key = torch.zeros(1, 128, dtype=keys.dtype, device=device)
                local_gate = torch.zeros_like(local_key)
                if rank < len(pairs):
                    request, pos = pairs[rank]
                    local_key[0] = keys[request, pos]
                    local_gate[0] = gates[request, pos]
                layout = pcp_manager.PCPTokenLayout(
                    len(pairs),
                    1,
                    torch.arange(len(pairs), device=device),
                    torch.empty(0, dtype=torch.int64, device=device),
                    torch.empty(0, dtype=torch.bool, device=device),
                )
                saved_index, saved_tail = index.clone(), tail.clone()
                with override_forward_context(SimpleNamespace()):
                    for provisional in (True, False):
                        # Reuse descriptors, but never data, across two layers.
                        # Restore persistent input before the real second layer.
                        index.copy_(saved_index)
                        tail.copy_(saved_tail)
                        step_key = (
                            torch.zeros_like(local_key) if provisional else local_key
                        )
                        step_gate = (
                            torch.zeros_like(local_gate) if provisional else local_gate
                        )
                        consumed, current_slots, current_rows = complete_boundary_pools(
                            step_key,
                            step_gate,
                            ape,
                            positions,
                            pool_slots,
                            tail_slots,
                            layout,
                            index,
                            tail,
                            index_pages,
                            tail_pages,
                            4,
                            group,
                            round_scale=True,
                            plan_key="leading-pool",
                            return_compressed=True,
                        )
                        slots, width, sources = current_pool_descriptors(
                            current_slots, group, plan_key="leading-index"
                        )
                        # A dictionary is a supported metadata container.
                        metadata = {
                            "block_table": torch.tensor(
                                [sorted(index_ids)], dtype=torch.int32, device=device
                            )
                        }
                        index_pages.compose_current(
                            metadata,
                            slots,
                            current_rows,
                            head_bytes=128,
                            sources=sources,
                            source_width=max(1, width),
                        )
                        index_pages.materialize(metadata)
                        assert bool(consumed.all())
                if step == 0:
                    prefix = [min(4, start + 1) for start in prefix]
            ids = torch.tensor(sorted(index_ids), device=device)
            actual = index_pages.materialize_ids(ids)
            torch.testing.assert_close(actual, baseline[ids], rtol=0, atol=0)
            ids = torch.tensor(sorted(tail_ids), device=device)
            actual_tail = tail_pages.materialize_ids(ids)
            for row, page in enumerate(sorted(tail_ids)):
                request = tail_ids.index(page)
                torch.testing.assert_close(
                    actual_tail[row, 0, position_base % 8 : position_base % 8 + 4],
                    keys[request],
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    actual_tail[row, 1, position_base % 8 : position_base % 8 + 4],
                    gates[request],
                    rtol=0,
                    atol=0,
                )
            for reader in reversed(readers):
                reader.close()
            readers = []
        if rank == 0:
            print(
                "PASS: CP8 two-step incomplete pools, MCCL and IPC, "
                "exact index/tail bytes",
                flush=True,
            )
    finally:
        for reader in reversed(readers):
            reader.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="vllm-pcp-") as directory:
        mp.spawn(
            run_rank,
            args=(8, (Path(directory) / "rendezvous").as_uri()),
            nprocs=8,
            join=True,
        )
