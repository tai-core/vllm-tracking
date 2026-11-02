# SPDX-License-Identifier: Apache-2.0
"""Bitwise kpool PCP encoding against the existing fused kernel."""

import pytest
import torch

from vllm.v1.pcp.layout import PCPPageLayout


def check_encoding(device):
    from vllm.models.glm5next.nvidia.ops import kpool_compress as kernels
    from vllm.models.glm5next.nvidia.pcp_kpool import (
        compress_local_pools,
        write_compressed_pools,
    )

    torch.manual_seed(41)
    for count in (3, 4, 63, 64, 65, 259):
        key = torch.randn((count, 128), dtype=torch.bfloat16).to(device)
        gate = torch.randn((count, 128), dtype=torch.bfloat16).to(device)
        ape = torch.randn((4, 128), dtype=torch.float32).to(device)
        slots = torch.arange(count, device=device, dtype=torch.int64)
        slots = torch.where(slots.remainder(4) == 3, slots // 4, -1)
        reference = torch.zeros((5, 16, 132), dtype=torch.uint8, device=device)
        result = torch.zeros_like(reference)
        full_pools = count // 4
        if full_pools:
            kernels.kpool_compress_and_write_cache(
                reference,
                key[: full_pools * 4].reshape(-1, 4, 128),
                gate[: full_pools * 4].reshape(-1, 4, 128),
                ape,
                torch.arange(full_pools, dtype=torch.int64, device=device),
                pool_size=4,
                head_dim=128,
            )
        completed = 0
        for span in PCPPageLayout(8, 64, 4).split_query(0, count):
            owned_slots, encoded = compress_local_pools(
                key[span.start : span.stop],
                gate[span.start : span.stop],
                ape,
                slots[span.start : span.stop],
                4,
                16,
                round_scale=True,
            )
            completed += owned_slots.numel()
            write_compressed_pools(result, owned_slots, encoded)
        assert completed == full_pools
        torch.testing.assert_close(result, reference, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_complete_pools_are_encoded_once_with_identical_cache_bytes():
    check_encoding(torch.device("cuda"))


if __name__ == "__main__":
    import torch_musa  # noqa: F401 -- registers the MUSA device backend

    torch.musa.set_device(0)
    check_encoding(torch.device("musa", 0))
    print("MUSA local kpool encoding: exact bytes for all cases", flush=True)


def test_pool_exchange_reuses_only_descriptors_within_one_forward():
    from types import SimpleNamespace

    from vllm.forward_context import override_forward_context
    from vllm.models.glm5next.nvidia.pcp_kpool import exchange_compressed_pools

    calls = []

    def gather(value, dim=0):
        calls.append(value.clone())
        return value

    group = SimpleNamespace(all_gather=gather)
    slots = torch.tensor([3, 8], dtype=torch.int64)
    encoded = torch.zeros(2, 132, dtype=torch.uint8)
    with override_forward_context(SimpleNamespace()):
        first_slots, first = exchange_compressed_pools(
            slots, encoded, group, plan_key=("group",)
        )
        second_slots, second = exchange_compressed_pools(
            slots, encoded + 1, group, plan_key=("group",)
        )
        assert len(calls) == 4
        assert first_slots is second_slots
        assert bool((first == 0).all())
        assert bool((second == 1).all())
    with override_forward_context(SimpleNamespace()):
        refreshed, _ = exchange_compressed_pools(
            slots + 1, encoded, group, plan_key=("group",)
        )
        assert len(calls) == 7
        assert refreshed.tolist() == [4, 9]
