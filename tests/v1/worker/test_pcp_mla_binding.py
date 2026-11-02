# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import torch

from vllm.v1.pcp.native_pages import PCPNativePages


def test_mla_cache_update_only_writes_owned_physical_slots(monkeypatch):
    from vllm.model_executor.layers.attention import mla_attention as module

    storage = torch.zeros(2, 4, 16)
    pages = PCPNativePages(
        storage,
        4,
        SimpleNamespace(world_size=2, rank_in_group=1),
        max_materialize_pages=4,
    )
    captured = []
    layer = SimpleNamespace(
        hisparse_cache=None,
        pcp_pages=pages,
        use_pcp=True,
        pcp_shard_decode_requests=False,
        impl=SimpleNamespace(do_kv_cache_update=lambda *args: captured.append(args)),
    )
    monkeypatch.setattr(
        module,
        "maybe_gather_mla_latent_cache_inputs",
        lambda key, rope, slots, *args, **kwargs: (key, rope, slots),
    )
    key = torch.zeros(4, 16)
    rope = torch.zeros(4, 0)
    slots = torch.tensor([0, 4, 8, 12], dtype=torch.int64)
    module.MLAAttention.update_kv_cache(
        layer,
        key,
        rope,
        storage,
        slots,
        SimpleNamespace(num_decode_tokens=0),
        "auto",
        torch.tensor(1.0),
    )
    assert len(captured) == 1
    assert captured[0][2] is storage
    assert captured[0][3].tolist() == [-1, 0, -1, 4]
    assert slots.tolist() == [0, 4, 8, 12]


def test_mla_encodes_only_producer_rows_and_publishes_native_bytes(monkeypatch):
    from vllm.forward_context import override_forward_context
    from vllm.model_executor.layers.attention import mla_attention as module
    from vllm.v1.worker.gpu.pcp_manager import PCPTokenLayout

    storage = torch.zeros(2, 64, 656, dtype=torch.uint8)
    encoded_calls, published = [], []
    pages = SimpleNamespace(
        group=SimpleNamespace(world_size=2, rank_in_group=0),
        compose_current=lambda *args, **kwargs: published.append((args, kwargs)),
    )

    def encode(key, rope, cache, slots, dtype, scale):
        encoded_calls.append((key.shape[0], slots.clone()))
        cache[0, 0].fill_(11 + len(encoded_calls) - 1)
        cache[0, 1].fill_(22 + len(encoded_calls) - 1)

    def forbidden_gather(*args, **kwargs):
        raise AssertionError("Raw MLA projections must not be all-gathered")

    monkeypatch.setattr(
        module, "maybe_gather_mla_latent_cache_inputs", forbidden_gather
    )
    layer = SimpleNamespace(
        hisparse_cache=None,
        pcp_pages=pages,
        use_pcp=True,
        pcp_shard_decode_requests=False,
        impl=SimpleNamespace(do_kv_cache_update=encode),
    )
    layout = PCPTokenLayout(
        4,
        2,
        torch.tensor([0, 2, 1, 3]),
        torch.tensor([0, 2, 1, 3]),
        torch.ones(4, dtype=torch.bool),
    )
    metadata = SimpleNamespace(num_decode_tokens=0)
    slots = torch.tensor([64, 65, 66, 67])
    for forward in range(2):
        with override_forward_context(SimpleNamespace(pcp_token_layout=layout)):
            for layer_index in range(2):
                module.MLAAttention.update_kv_cache(
                    layer,
                    torch.zeros(2, 512),
                    torch.zeros(2, 1, 64),
                    storage,
                    slots,
                    metadata,
                    "fp8_ds_mla",
                    torch.tensor(1.0),
                )
        slots.add_(64)
    assert len(encoded_calls) == len(published) == 4
    assert all(count == 2 and rows.tolist() == [0, 1] for count, rows in encoded_calls)
    assert published[0][0][1] is published[1][0][1]
    assert published[2][0][1] is published[3][0][1]
    assert published[2][0][1].tolist() == [128, 130, 129, 131]
    assert bool((published[1][0][2][0] == 12).all())
    args, kwargs = published[0]
    assert args[1].tolist() == [64, 66, 65, 67]
    assert args[2].shape == (2, 656)
    assert bool((args[2][0] == 11).all())
    assert bool((args[2][1] == 22).all())
    assert kwargs["sources"] is layout.restore_idx
    assert kwargs["source_width"] == 2
