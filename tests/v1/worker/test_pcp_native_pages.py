# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.pcp.native_pages import PCPNativePages


@dataclass
class Metadata:
    block_table: torch.Tensor
    children: list
    seq_lens: torch.Tensor


def test_native_metadata_is_remapped_without_modifying_scheduler_tables():
    storage = (
        torch.arange(4 * 8 * 12, dtype=torch.int32).to(torch.uint8).reshape(4, 8, 12)
    )
    group = SimpleNamespace(
        world_size=1, rank_in_group=0, all_gather=lambda x, dim=0: x
    )
    pool = PCPNativePages(
        storage, 2, group, kernel_blocks_per_page=2, max_materialize_pages=4
    )
    table = torch.tensor([[2, 3]], dtype=torch.int32)
    child = Metadata(torch.tensor([[3]], dtype=torch.int32), [], torch.tensor([1]))
    original = Metadata(table, [child], torch.tensor([16]))
    dense, remapped = pool.materialize(original)
    torch.testing.assert_close(dense, storage[2:])
    torch.testing.assert_close(
        remapped.block_table, torch.tensor([[0, 1]], dtype=torch.int32)
    )
    torch.testing.assert_close(
        remapped.children[0].block_table, torch.tensor([[1]], dtype=torch.int32)
    )
    torch.testing.assert_close(table, torch.tensor([[2, 3]], dtype=torch.int32))
    assert remapped.seq_lens is original.seq_lens
    assert child.block_table.item() == 3


def test_kernel_block_subdivision_does_not_change_manager_page_owner():
    pool = PCPNativePages(
        torch.zeros(4, 8, 16, dtype=torch.uint8),
        16,
        SimpleNamespace(world_size=8, rank_in_group=3),
        kernel_blocks_per_page=2,
        max_materialize_pages=32,
    )
    owners, physical = pool.locations(torch.tensor([6, 7, 22, 23]))
    assert owners.tolist() == [3, 3, 3, 3]
    assert physical.tolist() == [0, 1, 2, 3]
    assert pool.write_slots(torch.tensor([48, 56, 176, 184, 0, -1]), 8).tolist() == [
        0,
        8,
        16,
        24,
        -1,
        -1,
    ]


def test_native_workspace_budget_includes_collective_and_dense_lifetimes():
    import pytest

    pool = PCPNativePages(
        torch.zeros(3, 4, 16, dtype=torch.uint8),
        17,
        SimpleNamespace(world_size=8, rank_in_group=0),
        max_materialize_pages=17,
    )
    assert pool.workspace_bytes(17) == (9 * 3 + 17) * 64
    assert pool.workspace_bytes(2) == (9 * 2 + 2) * 64
    assert pool.workspace_bytes(17, max_owner_pages=2) == (9 * 2 + 17) * 64
    with pytest.raises(ValueError):
        pool.workspace_bytes(18)
    with pytest.raises(ValueError):
        pool.workspace_bytes(2, max_owner_pages=3)


def test_page_descriptors_are_shared_across_layers_but_not_forward_calls():
    from vllm.forward_context import override_forward_context

    calls = []

    def gather(tensor, dim=0):
        calls.append(tensor.shape)
        return tensor

    group = SimpleNamespace(world_size=1, rank_in_group=0, all_gather=gather)
    first = torch.arange(4 * 8 * 12).to(torch.uint8).reshape(4, 8, 12)
    second = first + 1
    pools = [
        PCPNativePages(value, 4, group, max_materialize_pages=4)
        for value in (first, second)
    ]
    for pool in pools:
        pool.ipc_reader = SimpleNamespace(
            materialize=lambda storage, owners, physical: storage[physical]
        )
    metadata = Metadata(
        torch.tensor([[1, 3]], dtype=torch.int32), [], torch.tensor([16])
    )
    with override_forward_context(SimpleNamespace()):
        data_a, mapped_a = pools[0].materialize(metadata)
        data_b, mapped_b = pools[1].materialize(metadata)
        assert len(calls) == 2  # one count exchange and one page-id exchange
        assert mapped_a is mapped_b
        torch.testing.assert_close(data_a, first[[1, 3]])
        torch.testing.assert_close(data_b, second[[1, 3]])
    # Same metadata object with new contents must not reuse the old descriptor.
    metadata.block_table[:] = torch.tensor([[0, 2]], dtype=torch.int32)
    with override_forward_context(SimpleNamespace()):
        data, mapped = pools[0].materialize(metadata)
        assert len(calls) == 4
        assert mapped is not mapped_a
        torch.testing.assert_close(data, first[[0, 2]])


def test_page_id_plan_reuses_addresses_but_reads_each_layers_values():
    import pytest

    from vllm.forward_context import override_forward_context

    group = SimpleNamespace(world_size=1, rank_in_group=0)
    first = torch.arange(4 * 8, dtype=torch.uint8).reshape(4, 1, 8)
    second = first + 100
    pools = [
        PCPNativePages(data, 4, group, max_materialize_pages=4)
        for data in (first, second)
    ]
    for pool in pools:
        pool.ipc_reader = SimpleNamespace(
            materialize=lambda storage, owners, physical: storage[physical]
        )
    pages = torch.tensor([1, 3])
    with override_forward_context(SimpleNamespace()):
        torch.testing.assert_close(
            pools[0].materialize_ids(pages, plan_key="boundary"), first[pages]
        )
        torch.testing.assert_close(
            pools[1].materialize_ids(pages, plan_key="boundary"), second[pages]
        )
    pages[1] = 4
    with (
        override_forward_context(SimpleNamespace()),
        pytest.raises(ValueError, match="outside the logical pool"),
    ):
        pools[0].materialize_ids(pages, plan_key="boundary")


def test_current_rows_bypass_complete_pages_and_preserve_partial_history():
    import pytest

    from vllm.forward_context import override_forward_context

    for head_bytes in (8, 4):
        original = torch.arange(96, dtype=torch.uint8).reshape(3, 4, 8)
        storage = original.clone()
        reads = []
        group = SimpleNamespace(
            world_size=1, rank_in_group=0, all_gather=lambda x, dim=0: x
        )
        store = PCPNativePages(storage, 3, group, max_materialize_pages=3)

        def read(source, owners, physical, reads=reads):
            reads.append(owners.clone())
            result = torch.zeros(len(owners), 4, 8, dtype=torch.uint8)
            valid = owners >= 0
            result[valid] = source[physical[valid]]
            return result

        store.ipc_reader = SimpleNamespace(
            materialize=read, wait_for_reuse=lambda: None
        )
        metadata = Metadata(torch.tensor([[0, 1, 2]]), [], torch.tensor([10]))
        slots = torch.arange(4, 10)
        encoded = torch.arange(48, dtype=torch.uint8).reshape(6, 8) + 100
        expected = original.clone()
        for i, slot in enumerate(slots.tolist()):
            byte = torch.arange(8)
            offset = slot % 4
            locations = (
                offset * 8 + byte
                if head_bytes == 8
                else torch.where(
                    byte < 4, offset * 4 + byte, 16 + offset * 4 + byte - 4
                )
            )
            expected[slot // 4].reshape(-1)[locations] = encoded[i]
        with override_forward_context(SimpleNamespace()):
            store.compose_current(metadata, slots, encoded, head_bytes=head_bytes)
            with pytest.raises(RuntimeError, match="before attention consumes"):
                store.compose_current(
                    metadata, slots, encoded + 1, head_bytes=head_bytes
                )
            dense, mapped = store.materialize(metadata)
            assert len(reads) == 1
            assert reads[0].tolist() == [0, -1, 0]
            torch.testing.assert_close(dense, expected)
            torch.testing.assert_close(storage, expected)
            torch.testing.assert_close(mapped.block_table, metadata.block_table)


def test_fused_current_routes_reuse_across_layers_and_expire_between_forwards():
    """Shared addresses must still read each layer's newly produced bytes."""
    from vllm.forward_context import override_forward_context

    group = SimpleNamespace(
        world_size=1, rank_in_group=0, all_gather=lambda x, dim=0: x
    )
    metadata = Metadata(torch.tensor([[0, 1]]), [], torch.tensor([8]))
    slots = torch.tensor([1, 6])
    source_maps = []

    def compose(storage, owners, physical, source_map, encoded, width, head_bytes):
        source_maps.append(source_map)
        result = storage[physical].clone()
        current = source_map >= 0
        result.view(-1, 8)[current] = encoded[source_map[current]]
        return result

    pools = [
        PCPNativePages(
            torch.full((2, 4, 8), value, dtype=torch.uint8),
            2,
            group,
            max_materialize_pages=2,
        )
        for value in (10, 20)
    ]
    for pool in pools:
        pool.ipc_reader = SimpleNamespace(compose_current=compose)

    for iteration in range(2):
        with override_forward_context(SimpleNamespace()):
            for layer, pool in enumerate(pools):
                encoded = torch.full((2, 8), 100 + layer, dtype=torch.uint8)
                pool.compose_current(
                    metadata,
                    slots,
                    encoded,
                    sources=torch.arange(2),
                    source_width=2,
                )
                dense, _ = pool.materialize(metadata)
                expected = pool.storage.clone().view(-1, 8)
                expected[slots] = encoded
                torch.testing.assert_close(dense.view(-1, 8), expected)
        # Same objects, different addresses on the next chunk.
        slots.add_(1)
    assert source_maps[0] is source_maps[1]
    assert source_maps[2] is source_maps[3]
    assert source_maps[0] is not source_maps[2]


@pytest.mark.parametrize("device_type", ["cpu", "cuda", "musa"])
def test_collect_and_remap_only_live_prefixes_after_table_reuse(device_type):
    if device_type == "musa":
        pytest.importorskip("torch_musa")
    if device_type != "cpu":
        accelerator = getattr(torch, device_type, None)
        if accelerator is None or not accelerator.is_available():
            pytest.skip(f"{device_type} is unavailable")
    device = torch.device(device_type)
    group = SimpleNamespace(
        world_size=1, rank_in_group=0, all_gather=lambda x, dim=0: x
    )
    pool = PCPNativePages(
        torch.arange(8 * 4 * 8, device=device).to(torch.uint8).reshape(8, 4, 8),
        8,
        group,
        max_materialize_pages=8,
    )
    # Both legal stale IDs and deliberately out-of-range garbage must be ignored.
    table = torch.full((4, 1366), 123456789, dtype=torch.int32, device=device)
    table[0, :3] = torch.tensor([5, 2, 7], device=device)
    table[1, :3] = torch.tensor([6, 3, 1], device=device)
    table[2, :2] = torch.tensor([4, 2], device=device)
    original = table.clone()
    metadata = Metadata(table, [], torch.tensor([5, 1, 0, 0], device=device))
    ids = pool.gather_page_ids(metadata)
    assert ids.cpu().tolist() == [2, 5, 6]
    dense, mapped = pool.materialize(metadata)
    torch.testing.assert_close(dense, pool.storage[ids])
    assert mapped.block_table.shape == (4, 2)
    assert mapped.block_table.cpu().tolist() == [[1, 0], [2, 0], [0, 0], [0, 0]]
    torch.testing.assert_close(table, original)


def test_compressed_root_and_chunk_lengths_use_their_own_units():
    group = SimpleNamespace(
        world_size=1, rank_in_group=0, all_gather=lambda x, dim=0: x
    )
    pool = PCPNativePages(
        torch.zeros(8, 4, 8, dtype=torch.uint8),
        8,
        group,
        max_materialize_pages=8,
        tokens_per_state=4,
    )
    metadata = {
        "block_table": torch.tensor([[1, 2, 777], [3, 888, 999], [555, 666, 777]]),
        # 5, 1 and 0 complete pools, each native page holds 4 pools.
        "seq_lens": torch.tensor([23, 7, 3]),
        "prefill": {
            "chunks": [
                {
                    "block_table": torch.tensor([[1, 2, 111], [3, 222, 333]]),
                    "local_cu_seq_lens": torch.tensor([0, 5, 6]),
                }
            ]
        },
    }
    pages, mapped, _, _ = pool._metadata_plan(metadata)
    assert pages.tolist() == [1, 2, 3]
    assert mapped["block_table"].tolist() == [[0, 1], [2, 0], [0, 0]]
    assert mapped["prefill"]["chunks"][0]["block_table"].tolist() == [[0, 1], [2, 0]]


def test_empty_pages_and_dense_prefill_history():
    group = SimpleNamespace(
        world_size=1, rank_in_group=0, all_gather=lambda x, dim=0: x
    )
    pool = PCPNativePages(torch.zeros(8, 4, 8), 8, group, max_materialize_pages=8)
    empty = Metadata(torch.full((2, 128), 999), [], torch.zeros(2, dtype=torch.int32))
    assert pool.gather_page_ids(empty).numel() == 0
    assert pool._metadata_plan(empty)[1].block_table.shape == (2, 0)
    history = {
        "block_table": torch.tensor([[1, 2, 999]]),
        "query_start_loc": torch.tensor([0, 4]),
        "chunked_context": SimpleNamespace(context_lens=torch.tensor([5])),
    }
    assert pool.gather_page_ids(history).tolist() == [1, 2]
    history["chunked_context"] = None
    assert pool.gather_page_ids(history).numel() == 0


def test_missing_or_inconsistent_lengths_fail_without_reading_tail():
    group = SimpleNamespace(
        world_size=1, rank_in_group=0, all_gather=lambda x, dim=0: x
    )
    pool = PCPNativePages(torch.zeros(8, 4, 8), 8, group, max_materialize_pages=8)
    with pytest.raises(ValueError, match="missing effective lengths"):
        pool.gather_page_ids({"block_table": torch.tensor([[1, 999]])})
    with pytest.raises(ValueError, match="exceeds block-table capacity"):
        pool.gather_page_ids(Metadata(torch.tensor([[1]]), [], torch.tensor([5])))
    with pytest.raises(ValueError, match="rows and effective lengths disagree"):
        pool.gather_page_ids(Metadata(torch.tensor([[1]]), [], torch.tensor([1, 2])))


def test_identical_storage_geometry_does_not_alias_compression_units():
    from vllm.forward_context import override_forward_context

    group = SimpleNamespace(
        world_size=1, rank_in_group=0, all_gather=lambda x, dim=0: x
    )
    storage = torch.arange(4 * 4 * 8).reshape(4, 4, 8)
    full = PCPNativePages(storage, 4, group, max_materialize_pages=4)
    compressed = PCPNativePages(
        storage, 4, group, max_materialize_pages=4, tokens_per_state=4
    )
    metadata = Metadata(torch.tensor([[1, 2]]), [], torch.tensor([7]))
    with override_forward_context(SimpleNamespace()):
        assert full._metadata_plan(metadata)[0].tolist() == [1, 2]
        assert compressed._metadata_plan(metadata)[0].tolist() == [1]


def test_current_routes_do_not_alias_different_compression_units():
    from vllm.forward_context import override_forward_context

    group = SimpleNamespace(
        world_size=1, rank_in_group=0, all_gather=lambda x, dim=0: x
    )
    pools = [
        PCPNativePages(
            torch.zeros(3, 4, 8, dtype=torch.uint8),
            3,
            group,
            max_materialize_pages=3,
            tokens_per_state=ratio,
        )
        for ratio in (1, 4)
    ]
    source_sizes = []

    def compose(storage, owners, physical, source_map, encoded, width, head_bytes):
        source_sizes.append(source_map.numel())
        assert source_map.numel() == owners.numel() * storage.shape[1]
        return storage[physical].clone()

    for pool in pools:
        pool.ipc_reader = SimpleNamespace(compose_current=compose)
    metadata = Metadata(torch.tensor([[1, 2]]), [], torch.tensor([7]))
    with override_forward_context(SimpleNamespace()):
        for pool in pools:
            pool.compose_current(
                metadata,
                torch.tensor([4]),
                torch.zeros(1, 8, dtype=torch.uint8),
                sources=torch.tensor([0]),
                source_width=1,
            )
            pool.materialize(metadata)
    assert source_sizes == [8, 4]
