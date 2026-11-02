# SPDX-License-Identifier: Apache-2.0
"""Distributed kpool compression + unique storage against the fused baseline."""

import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.v1.pcp.layout import PCPPageLayout
from vllm.v1.pcp.reference_store import PCPPageStore


class Group:
    def __init__(self, rank, size):
        self.rank_in_group, self.world_size = rank, size

    def all_gather(self, tensor, dim=0):
        output = [torch.empty_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(output, tensor)
        return torch.cat(output, dim=dim)


def run_rank(rank, size, rendezvous):
    import torch_musa  # noqa: F401 -- registers the MUSA device backend

    from vllm.models.glm5next.nvidia.ops import kpool_compress as kernels
    from vllm.models.glm5next.nvidia.pcp_kpool import (
        compress_local_pools,
        compress_pcp_pools,
        materialized_index_cache,
    )

    torch.musa.set_device(rank)
    device = torch.device("musa", rank)
    dist.init_process_group("mccl", init_method=rendezvous, rank=rank, world_size=size)
    try:
        group = Group(rank, size)
        layout = PCPPageLayout(size, 64, 4)
        for count in (3, 65, 259, 1027):
            torch.manual_seed(97)
            key = torch.randn(count, 128, dtype=torch.bfloat16).to(device)
            gate = torch.randn(count, 128, dtype=torch.bfloat16).to(device)
            ape = torch.randn(4, 128, dtype=torch.float32).to(device)
            pages = (count + 63) // 64
            store = PCPPageStore(layout, pages, 16, 132, device, group, pages)
            local_ids = [
                token
                for span in layout.split_query(0, count)
                if span.compute_rank == rank
                for token in range(span.start, span.stop)
            ]
            ids = torch.tensor(local_ids, device=device, dtype=torch.int64)
            slots = torch.where(ids.remainder(4) == 3, ids // 4, -1)
            slots, rows = compress_local_pools(
                key[ids], gate[ids], ape, slots, 4, 16, round_scale=True
            )
            counts = group.all_gather(
                torch.tensor([slots.numel()], device=device, dtype=torch.int64)
            )
            assert counts.sum().item() == count // 4
            width = max(1, int(counts.max().item()))
            padded_slots = torch.full((width,), -1, device=device, dtype=torch.int64)
            padded_rows = torch.zeros(width, 132, device=device, dtype=torch.uint8)
            padded_slots[: slots.numel()] = slots
            padded_rows[: slots.numel()] = rows
            store.write_rows(padded_slots, padded_rows)
            result = materialized_index_cache(
                store.materialize(
                    torch.arange(pages, device=device, dtype=torch.int64)
                ).data
            )
            baseline = torch.zeros_like(result)
            pools = count // 4
            if pools:
                kernels.kpool_compress_and_write_cache(
                    baseline,
                    key[: pools * 4].reshape(-1, 4, 128),
                    gate[: pools * 4].reshape(-1, 4, 128),
                    ape,
                    torch.arange(pools, device=device, dtype=torch.int64),
                    pool_size=4,
                )
            torch.testing.assert_close(result, baseline, rtol=0, atol=0)
            # The exact helper called by the live indexer must also agree.
            native_result = torch.zeros_like(baseline)
            raw_slots = torch.where(ids.remainder(4) == 3, ids // 4, -1)
            from types import SimpleNamespace

            from vllm.forward_context import override_forward_context

            with override_forward_context(SimpleNamespace()):
                for multiplier in (1.0, 0.5):
                    changed_key = key * multiplier
                    native_result.zero_()
                    compress_pcp_pools(
                        changed_key[ids],
                        gate[ids],
                        ape,
                        native_result,
                        raw_slots,
                        4,
                        group,
                        round_scale=True,
                        plan_key=("test", count),
                    )
                    expected_layer = torch.zeros_like(baseline)
                    if pools:
                        kernels.kpool_compress_and_write_cache(
                            expected_layer,
                            changed_key[: pools * 4].reshape(-1, 4, 128),
                            gate[: pools * 4].reshape(-1, 4, 128),
                            ape,
                            torch.arange(pools, device=device, dtype=torch.int64),
                            pool_size=4,
                        )
                    torch.testing.assert_close(
                        native_result, expected_layer, rtol=0, atol=0
                    )
            from vllm.v1.worker.gpu import pcp_manager as manager_module

            manager_module.get_pcp_group = lambda: group
            all_ids = [
                [
                    token
                    for span in layout.split_query(0, count)
                    if span.compute_rank == owner
                    for token in range(span.start, span.stop)
                ]
                for owner in range(size)
            ]
            padded = max(map(len, all_ids))
            restore = torch.empty(count, dtype=torch.int64, device=device)
            for owner, token_ids in enumerate(all_ids):
                restore[torch.tensor(token_ids, dtype=torch.int64, device=device)] = (
                    torch.arange(len(token_ids), device=device) + owner * padded
                )
            token_layout = manager_module.PCPTokenLayout(
                count,
                padded,
                restore,
                torch.empty(0, dtype=torch.int64, device=device),
                torch.empty(0, dtype=torch.bool, device=device),
            )
            local_key = torch.zeros(padded, 128, dtype=key.dtype, device=device)
            local_key[: ids.numel()] = key[ids]
            tail_rows = torch.arange(max(0, count - 4), count, device=device)
            torch.testing.assert_close(
                token_layout.gather_rows(local_key, tail_rows),
                key[tail_rows],
                rtol=0,
                atol=0,
            )

            total_bytes = (
                group.all_gather(
                    torch.tensor(
                        [store.persistent_bytes], device=device, dtype=torch.int64
                    )
                )
                .sum()
                .item()
            )
            assert total_bytes == pages * 16 * 132
        if rank == 0:
            print(
                f"PASS: {size} MUSA ranks, unique compressed pages, exact bytes",
                flush=True,
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="vllm-pcp-") as directory:
        mp.spawn(
            run_rank,
            args=(8, (Path(directory) / "rendezvous").as_uri()),
            nprocs=8,
            join=True,
        )
