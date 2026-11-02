# SPDX-License-Identifier: Apache-2.0
"""Exercise real allocation, binding, native reads, and whole-pool maintenance."""

import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.v1.core.kv_cache_utils import _build_attention_kv_cache_tensors
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, MLAAttentionSpec
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.pcp.allocation import bind_pcp_pages, make_pcp_physical_config
from vllm.v1.worker.utils import allocate_kv_cache


@dataclass
class Metadata:
    block_table: torch.Tensor
    seq_lens: torch.Tensor


class Group:
    def __init__(self, rank, size):
        self.rank_in_group, self.world_size = rank, size

    def all_gather(self, tensor, dim=0):
        output = [torch.empty_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(output, tensor)
        return torch.cat(output, dim=dim)


def check_compose_alignment(group, device):
    """Compare native plane packing and owner writes with a CPU byte reference."""
    from vllm_musa.distributed.device_communicators.pcp_native_pages import PCPNativeIPC

    rank, size = group.rank_in_group, group.world_size
    # Vector records, split FP32 scales, odd pages, and unaligned byte records.
    for rows, row_bytes, head_bytes, staging_stride, offset in (
        (64, 656, 656, 656, 0),
        (64, 132, 128, 656, 0),
        (3, 132, 128, 656, 0),
        (5, 20, 16, 20, 4),
        (3, 15, 11, 17, 1),
    ):
        page_bytes = rows * row_bytes
        stride = 2 * page_bytes
        root = torch.full(
            (4 * stride + offset,), rank + 11, dtype=torch.uint8, device=device
        )
        reader = PCPNativeIPC(root, group)
        try:
            reader.register_current_staging(2, staging_stride)
            owners = torch.tensor([0, 1, -1], dtype=torch.int64)
            pages = torch.tensor([1, 2, 0], dtype=torch.int64)
            sources = torch.full((3, rows), -1, dtype=torch.int64)
            sources[0, 0] = (size - 1) * 2 + 1
            sources[1, -1] = 0
            output = torch.empty((3, page_bytes), dtype=torch.uint8, device=device)
            gpu_owners, gpu_pages, gpu_sources = [
                tensor.to(device) for tensor in (owners, pages, sources)
            ]
            for generation in range(2):
                encoded = (
                    torch.arange(2 * staging_stride).reshape(2, staging_stride)
                    + rank * 17
                    + generation
                ).to(torch.uint8)
                reader.current_staging.root[0].copy_(encoded)
                root.fill_(rank + 11)
                expected = torch.stack(
                    [
                        torch.full(
                            (page_bytes,),
                            owner + 11 if owner >= 0 else 0,
                            dtype=torch.uint8,
                        )
                        for owner in owners.tolist()
                    ]
                )
                persistent = root.cpu()
                for page, row in ((0, 0), (1, rows - 1)):
                    source = sources[page, row].item()
                    record = (
                        torch.arange(row_bytes)
                        + (source % 2) * staging_stride
                        + (source // 2) * 17
                        + generation
                    ).to(torch.uint8)
                    head = slice(row * head_bytes, (row + 1) * head_bytes)
                    tail_start = rows * head_bytes + row * (row_bytes - head_bytes)
                    tail = slice(tail_start, tail_start + row_bytes - head_bytes)
                    expected[page, head] = record[:head_bytes]
                    expected[page, tail] = record[head_bytes:]
                    if owners[page] == rank:
                        start = offset + pages[page].item() * stride
                        persistent[start : start + page_bytes] = expected[page]
                reader.module.pcp_compose_current_launch(
                    reader.pointers,
                    reader.current_staging.pointers,
                    reader.signal_pointers,
                    gpu_owners,
                    gpu_pages,
                    gpu_sources,
                    output,
                    rank,
                    size,
                    page_bytes,
                    stride,
                    offset,
                    rows,
                    row_bytes,
                    head_bytes,
                    2,
                    staging_stride,
                )
                torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
                torch.testing.assert_close(root.cpu(), persistent, rtol=0, atol=0)
            # Empty selections still participate in the publication/reuse barrier.
            reader.module.pcp_compose_current_launch(
                reader.pointers,
                reader.current_staging.pointers,
                reader.signal_pointers,
                gpu_owners[:0],
                gpu_pages[:0],
                gpu_sources[:0],
                output[:0],
                rank,
                size,
                page_bytes,
                stride,
                offset,
                rows,
                row_bytes,
                head_bytes,
                2,
                staging_stride,
            )
            torch.testing.assert_close(root.cpu(), persistent, rtol=0, atol=0)
        finally:
            reader.close()


def run_rank(rank, size, rendezvous, device_kind):
    if device_kind == "musa":
        import torch_musa  # noqa: F401

        torch.musa.set_device(rank)
    device = (
        torch.device(device_kind, rank) if device_kind != "cpu" else torch.device("cpu")
    )
    dist.init_process_group(
        "mccl" if device_kind == "musa" else "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=size,
    )
    try:
        import vllm.distributed

        group = Group(rank, size)
        if device_kind == "musa":
            group.cpu_group = dist.new_group(backend="gloo")
        vllm.distributed.get_pcp_group = lambda: group
        config = SimpleNamespace(
            cache_config=SimpleNamespace(
                get_resolved_kv_cache_layout=lambda: KVCacheLayout.BLHNC
            )
        )
        spec = MLAAttentionSpec(
            block_size=16,
            num_kv_heads=1,
            head_size=32,
            dtype=torch.uint8,
        )
        groups = [
            KVCacheGroupSpec(["layer.0", "layer.1"], spec),
            KVCacheGroupSpec(["alias.0"], spec),
        ]
        logical_count = size * 2 + 1
        tensors, _ = _build_attention_kv_cache_tensors(
            config, groups, 0, physical_num_blocks=logical_count
        )
        logical = KVCacheConfig(
            num_blocks=logical_count,
            kv_cache_tensors=tensors,
            kv_cache_groups=groups,
        )
        physical = make_pcp_physical_config(config, logical, size)
        caches = allocate_kv_cache(physical, device, KVCacheLayout.BLHNC)
        layers = {
            name: SimpleNamespace(kv_cache=cache.squeeze(1))
            for name, cache in caches.items()
        }
        # This tiny geometry tests native addressing, not model admission.
        maintenance = bind_pcp_pages(None, logical, physical, layers, caches)
        for physical_id in range(physical.num_blocks):
            logical_id = physical_id * size + rank
            maintenance.storage[physical_id].fill_(logical_id * 7)
        source_table = torch.tensor([[rank + 1]], dtype=torch.int32, device=device)
        for layer in layers.values():
            dense, metadata = layer.pcp_pages.materialize(
                Metadata(source_table, source_table.new_full((1,), 16))
            )
            assert dense.shape[0] == size
            assert metadata.block_table.item() == rank
            for page in range(size):
                assert bool((dense[page] == (page + 1) * 7).all())
        # Cross-owner copy must happen once for the aliased shared allocation.
        a, b = 1, size + 2
        maintenance.copy_blocks([(a, b), (b, a)])
        selected = torch.tensor([a, b], dtype=torch.int64, device=device)
        data = maintenance.materialize_ids(selected)
        assert bool((data[0] == b * 7).all())
        assert bool((data[1] == a * 7).all())
        maintenance.zero_blocks([b])
        data = maintenance.materialize_ids(selected)
        assert bool((data[0] == b * 7).all())
        assert bool((data[1] == 0).all())
        from vllm.forward_context import override_forward_context

        if maintenance.ipc_reader is not None:
            maintenance.ipc_reader.register_current_staging(32, 32)
        ids = torch.tensor([1, 2, 3], dtype=torch.int64, device=device)
        slots = torch.arange(16, 34, dtype=torch.int64, device=device)
        width = (slots.numel() + size - 1) // size
        row = torch.arange(slots.numel(), device=device)
        sources = (row % size) * width + row // size
        metadata = Metadata(ids[None].to(torch.int32), ids.new_tensor([48]))
        for head_bytes in (32, 28):
            with override_forward_context(SimpleNamespace()):
                for layer_id, layer in enumerate(layers.values()):
                    store = layer.pcp_pages
                    expected = store.materialize_ids(ids).clone()
                    encoded = (
                        torch.arange(slots.numel() * 32, device=device)
                        .reshape(-1, 32)
                        .to(torch.uint8)
                        + layer_id
                    )
                    # The CPU reference indexes native page data/scale planes.
                    byte = torch.arange(32, device=device)
                    for source_row, slot in enumerate(range(16, 34)):
                        offset = slot % 16
                        in_page = torch.where(
                            byte < head_bytes,
                            offset * head_bytes + byte,
                            16 * head_bytes
                            + offset * (32 - head_bytes)
                            + byte
                            - head_bytes,
                        )
                        if head_bytes == 32:
                            in_page = offset * 32 + byte
                        expected[slot // 16 - 1].reshape(-1)[in_page] = encoded[
                            source_row
                        ]
                    store.compose_current(
                        metadata,
                        slots,
                        encoded[rank::size].contiguous(),
                        head_bytes=head_bytes,
                        sources=sources,
                        source_width=width,
                    )
                    result, _ = store.materialize(metadata)
                    torch.testing.assert_close(result, expected, rtol=0, atol=0)
                    persisted = store.materialize_ids(ids)
                    torch.testing.assert_close(persisted, expected, rtol=0, atol=0)
        if device_kind == "musa":
            import vllm_musa
            from vllm_musa.distributed.device_communicators.pcp_native_pages import (
                PCPNativeIPC,
            )

            from vllm import _custom_ops as ops
            from vllm.v1.pcp.native_pages import PCPNativePages

            vllm_musa.register_custom_ops()
            torch.manual_seed(117)
            count = 7
            keys = torch.randn(size * count, 512, dtype=torch.bfloat16).to(device)
            rope = torch.randn(size * count, 64, dtype=torch.bfloat16).to(device)
            slots = torch.arange(64, 64 + size * count, device=device)
            reference = torch.full(
                (size * 2, 64, 656), 7, dtype=torch.uint8, device=device
            )
            scale = torch.ones(1, device=device)
            ops.concat_and_cache_mla(
                keys, rope, reference, slots, kv_cache_dtype="fp8_ds_mla", scale=scale
            )
            packed = torch.empty((1, 64, 656), dtype=torch.uint8, device=device)
            local_slice = slice(rank * count, (rank + 1) * count)
            ops.concat_and_cache_mla(
                keys[local_slice],
                rope[local_slice],
                packed,
                torch.arange(count, device=device),
                kv_cache_dtype="fp8_ds_mla",
                scale=scale,
            )
            raw = torch.full((2, 64, 656), 7, dtype=torch.uint8, device=device)
            reader = PCPNativeIPC(raw, group)
            group._pcp_native_pools.append(reader)
            reader.register_current_staging(64, 656)
            direct = reader.current_mla_buffer(raw, count)
            assert direct is not None
            assert direct.data_ptr() == reader.current_staging.root.data_ptr()
            ops.concat_and_cache_mla(
                keys[local_slice],
                rope[local_slice],
                direct,
                torch.arange(count, device=device),
                kv_cache_dtype="fp8_ds_mla",
                scale=scale,
            )
            torch.testing.assert_close(
                direct[0, :count], packed[0, :count], rtol=0, atol=0
            )
            packed = direct
            store = PCPNativePages(raw, size * 2, group, max_materialize_pages=size * 2)
            store.ipc_reader = reader
            metadata = Metadata(
                torch.tensor([[1]], dtype=torch.int32, device=device),
                torch.tensor([64], device=device),
            )
            with override_forward_context(SimpleNamespace()):
                store.compose_current(
                    metadata,
                    slots,
                    packed.reshape(-1, 656)[:count],
                    sources=torch.arange(size * count, device=device),
                    source_width=count,
                )
                result, _ = store.materialize(metadata)
                torch.testing.assert_close(result, reference[1:2], rtol=0, atol=0)
                persisted = store.materialize_ids(torch.tensor([1], device=device))
                torch.testing.assert_close(persisted, reference[1:2], rtol=0, atol=0)
        if device_kind == "musa":
            check_compose_alignment(group, device)
        if rank == 0:
            print(
                "PASS: native allocation, COW, current staging, fused owner writes; "
                f"{size} ranks",
                flush=True,
            )
    finally:
        for reader in reversed(getattr(locals().get("group"), "_pcp_native_pools", [])):
            reader.close()
        dist.destroy_process_group()


def test_native_binding_cpu(tmp_path):
    mp.spawn(
        run_rank,
        args=(2, (tmp_path / "rendezvous").as_uri(), "cpu"),
        nprocs=2,
        join=True,
    )


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="vllm-pcp-") as directory:
        mp.spawn(
            run_rank,
            args=(8, (Path(directory) / "rendezvous").as_uri(), "musa"),
            nprocs=8,
            join=True,
        )
