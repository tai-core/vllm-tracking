# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for MooncakeConnector hybrid FA + GDN support.

GDN is represented as a MambaSpec in vLLM, so these tests exercise the
Mooncake MambaSpec path with mamba_type=GDN_ATTN. Mamba2 is intentionally not
validated by this test module.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from tests.v1.attention.utils import dense_kv_cache_views
from vllm.config import set_current_vllm_config
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake import (
    mooncake_connector as mooncake_connector_module,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
    KVConnectorRole,
    MooncakeConnector,
    MooncakeConnectorMetadata,
    MooncakeConnectorScheduler,
    MooncakeConnectorWorker,
    MooncakeXferMetadata,
    PullReqMeta,
    SendBlockMeta,
    TransferRegion,
    _compute_kda_dp_sender_transfer_plan,
)
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheLayout,
    MambaSpec,
)

from .test_mooncake_connector import patch_worker_dependencies
from .utils import create_request, create_vllm_config, maybe_update_block_size


def noop_shutdown():
    pass


def make_hybrid_gdn_kv_cache_config(block_size: int) -> KVCacheConfig:
    gdn_spec = MambaSpec(
        block_size=block_size,
        shapes=((6, 3), (1, 2, 2)),
        dtypes=(torch.float16, torch.float16),
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
    )
    assert gdn_spec.mamba_type == MambaAttentionBackendEnum.GDN_ATTN
    return KVCacheConfig(
        num_blocks=16,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["model.layers.0.self_attn"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float16,
                ),
            ),
            KVCacheGroupSpec(
                ["model.layers.1.linear_attn"],
                gdn_spec,
            ),
        ],
    )


def make_hybrid_gdn_scheduler(kv_role: str) -> MooncakeConnectorScheduler:
    vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector",
        kv_role=kv_role,
    )
    vllm_config.scheduler_config.disable_hybrid_kv_cache_manager = False
    return MooncakeConnectorScheduler(
        vllm_config=vllm_config,
        engine_id="test-engine",
        kv_cache_config=make_hybrid_gdn_kv_cache_config(
            vllm_config.cache_config.block_size
        ),
    )


@pytest.mark.cpu_test
def test_hybrid_gdn_remote_prefill_uses_mamba_n_minus_one():
    scheduler = make_hybrid_gdn_scheduler(kv_role="kv_consumer")
    request = create_request(num_tokens=10, do_remote_prefill=True)

    num_new_tokens, is_async = scheduler.get_num_new_matched_tokens(
        request, num_computed_tokens=0
    )

    assert num_new_tokens == request.num_prompt_tokens - 1
    assert is_async is True


@pytest.mark.cpu_test
def test_hybrid_gdn_remote_decode_truncates_prefill_before_cache_lookup():
    vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector",
        kv_role="kv_producer",
    )
    vllm_config.scheduler_config.disable_hybrid_kv_cache_manager = False
    connector = MooncakeConnector(
        vllm_config,
        KVConnectorRole.SCHEDULER,
        make_hybrid_gdn_kv_cache_config(vllm_config.cache_config.block_size),
    )
    scheduler = connector.connector_scheduler
    assert scheduler is not None
    request = create_request(num_tokens=10, do_remote_decode=True)
    original_tokens = list(request.prompt_token_ids)

    connector.on_new_request(request)
    assert request.prompt_token_ids == original_tokens[:-1]
    assert request._all_token_ids == original_tokens[:-1]
    assert request.num_prompt_tokens == len(original_tokens) - 1
    assert request.max_tokens == 1
    assert request.kv_transfer_params["_p_side_truncated"] is True

    # Re-adding or rescheduling the request must not truncate another token.
    scheduler.on_new_request(request)
    assert request.prompt_token_ids == original_tokens[:-1]

    # Prefix-cache matching must not mutate the request after its local lookup.
    with patch.object(
        scheduler,
        "_truncate_mamba_request_for_prefill",
        side_effect=AssertionError("late Mamba truncation"),
    ):
        num_new_tokens, is_async = scheduler.get_num_new_matched_tokens(
            request, num_computed_tokens=0
        )

    assert num_new_tokens == 0
    assert is_async is False


def test_register_kv_caches_emits_fa_and_gdn_regions(monkeypatch):
    monkeypatch.setenv("VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT", "5")
    vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector",
        kv_role="kv_consumer",
        block_size=maybe_update_block_size(16),
    )
    kv_cache_config = make_hybrid_gdn_kv_cache_config(
        vllm_config.cache_config.block_size
    )

    with set_current_vllm_config(vllm_config), patch_worker_dependencies():
        connector = MooncakeConnector(
            vllm_config,
            KVConnectorRole.WORKER,
            kv_cache_config,
        )
        worker = connector.connector_worker

        num_blocks = kv_cache_config.num_blocks
        fa_spec = kv_cache_config.kv_cache_groups[0].kv_cache_spec
        gdn_spec = kv_cache_config.kv_cache_groups[1].kv_cache_spec
        fa_raw = torch.empty(num_blocks * fa_spec.page_size_bytes, dtype=torch.int8)
        gdn_block_stride = gdn_spec.page_size_bytes + 64
        gdn_raw = torch.empty(num_blocks, gdn_block_stride, dtype=torch.int8)
        (fa_cache,) = dense_kv_cache_views(
            fa_raw, fa_spec, num_blocks, 1, KVCacheLayout.LBHNC
        )
        gdn_cache = gdn_raw[:, : gdn_spec.page_size_bytes]

        worker.register_kv_caches(
            {
                "model.layers.0.self_attn": fa_cache,
                "model.layers.1.linear_attn": gdn_cache,
            }
        )

        assert worker.transfer_topo.is_mamba is True
        assert worker.registered_layer_names == [
            "model.layers.0.self_attn",
            "model.layers.1.linear_attn",
        ]
        assert worker.kv_block_len_per_layer == [
            fa_spec.page_size_bytes,
            gdn_spec.page_size_bytes,
        ]
        assert worker.registered_group_indices == [0, 1]
        assert worker.kv_caches_base_addr == [
            fa_cache.data_ptr(),
            gdn_cache.data_ptr(),
        ]

        worker.shutdown()
        worker.shutdown = noop_shutdown
        connector.connector_worker = None


def test_register_kv_caches_scales_attention_len_to_kernel_block(monkeypatch):
    monkeypatch.setenv("VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT", "5")
    vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector",
        kv_role="kv_consumer",
        block_size=maybe_update_block_size(16),
    )
    kv_cache_config = make_hybrid_gdn_kv_cache_config(
        vllm_config.cache_config.block_size
    )

    with set_current_vllm_config(vllm_config), patch_worker_dependencies():
        connector = MooncakeConnector(
            vllm_config,
            KVConnectorRole.WORKER,
            kv_cache_config,
        )
        worker = connector.connector_worker
        factor = 4
        worker._physical_blocks_per_logical_kv_block = factor

        fa_spec = kv_cache_config.kv_cache_groups[0].kv_cache_spec
        physical_page_bytes = fa_spec.page_size_bytes // factor
        fa_cache = torch.empty(
            kv_cache_config.num_blocks * factor,
            physical_page_bytes,
            dtype=torch.uint8,
        )

        worker.register_kv_caches({"model.layers.0.self_attn": fa_cache})

        assert worker.block_len_per_layer == [physical_page_bytes]
        assert worker.kv_block_len_per_layer == [physical_page_bytes]

        worker.shutdown()
        worker.shutdown = noop_shutdown
        connector.connector_worker = None


@pytest.mark.asyncio
async def test_build_transfer_params_uses_non_overlapping_physical_pages():
    kv_cache_config = make_hybrid_gdn_kv_cache_config(block_size=16)
    worker = object.__new__(MooncakeConnectorWorker)
    worker.shutdown = noop_shutdown
    # __new__ skips __init__, where this cache is created. register_kv_caches clears it.
    worker._prepared_transfer_regions = {}
    worker.use_mla = False
    worker.engine = SimpleNamespace(batch_register_memory=lambda *_: 0)
    worker.is_kv_consumer = True
    worker.tp_rank = 0
    worker.tp_size = 1
    worker.transfer_topo = SimpleNamespace(local_replicates_kv_cache=False)
    worker.kv_cache_config = kv_cache_config
    worker._layer_specs = {
        layer_name: group.kv_cache_spec
        for group in kv_cache_config.kv_cache_groups
        for layer_name in group.layer_names
    }
    worker._layer_group_indices = {
        layer_name: group_index
        for group_index, group in enumerate(kv_cache_config.kv_cache_groups)
        for layer_name in group.layer_names
    }

    ratio = 4
    worker._physical_blocks_per_logical_kv_block = ratio
    fa_spec = kv_cache_config.kv_cache_groups[0].kv_cache_spec
    physical_page_bytes = fa_spec.page_size_bytes // ratio
    fa_cache = torch.empty(
        kv_cache_config.num_blocks * ratio,
        physical_page_bytes,
        dtype=torch.uint8,
    )
    worker.register_kv_caches({"model.layers.0.self_attn": fa_cache})

    local_regions = worker._get_transfer_regions(
        worker.kv_caches_base_addr,
        worker.block_len_per_layer,
        worker.kv_block_len_per_layer,
        worker.registered_layer_names,
        worker.registered_layer_indices,
        worker.registered_group_indices,
    )
    remote_base_addr = 0x100000
    remote_regions = [
        TransferRegion(
            layer_name="model.layers.0.self_attn",
            layer_index=0,
            base_addr=remote_base_addr,
            block_len=physical_page_bytes,
            kv_block_len=physical_page_bytes,
            group_index=0,
        )
    ]

    local_logical_block = 2
    remote_logical_block = 5
    transfer_id = "xfer-physical-attention-pages"
    send_meta = SendBlockMeta(
        p_req_id="p-physical-attention-pages",
        transfer_id=transfer_id,
        local_block_ids=[[local_logical_block], []],
        ready=asyncio.Event(),
    )
    xfer_meta = MooncakeXferMetadata(
        remote_hostname="consumer-host",
        remote_port=54321,
        remote_tp_size=1,
        remote_tp_rank=0,
        req_blocks={
            "d-physical-attention-pages": (
                transfer_id,
                [[remote_logical_block], []],
            )
        },
        kv_caches_base_addr=[],
        block_lens=[],
        kv_block_lens=[],
    )

    (
        src_ptrs,
        dst_ptrs,
        lengths,
        err_reqs,
        err_msg,
    ) = await worker._build_transfer_params(
        [("d-physical-attention-pages", send_meta)],
        xfer_meta,
        local_regions,
        remote_regions,
    )

    assert err_reqs == []
    assert err_msg is None
    assert src_ptrs == [
        fa_cache.data_ptr() + local_logical_block * ratio * physical_page_bytes
    ]
    assert dst_ptrs == [
        remote_base_addr + remote_logical_block * ratio * physical_page_bytes
    ]
    assert lengths == [ratio * physical_page_bytes]

    def split_contiguous_blocks(src_ids, dst_ids):
        return [[block_id] for block_id in src_ids], [
            [block_id] for block_id in dst_ids
        ]

    with patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
        "mooncake_connector.group_concurrent_contiguous",
        side_effect=split_contiguous_blocks,
    ):
        (
            src_ptrs,
            dst_ptrs,
            lengths,
            err_reqs,
            err_msg,
        ) = await worker._build_transfer_params(
            [("d-physical-attention-pages", send_meta)],
            xfer_meta,
            local_regions,
            remote_regions,
        )

    expected_src_ptrs = [
        fa_cache.data_ptr()
        + (local_logical_block * ratio + offset) * physical_page_bytes
        for offset in range(ratio)
    ]
    expected_dst_ptrs = [
        remote_base_addr + (remote_logical_block * ratio + offset) * physical_page_bytes
        for offset in range(ratio)
    ]
    assert err_reqs == []
    assert err_msg is None
    assert src_ptrs == expected_src_ptrs
    assert dst_ptrs == expected_dst_ptrs
    assert lengths == [physical_page_bytes] * ratio
    assert all(
        next_ptr - ptr == physical_page_bytes
        for ptr, next_ptr in zip(src_ptrs, src_ptrs[1:])
    )
    assert all(
        next_ptr - ptr == physical_page_bytes
        for ptr, next_ptr in zip(dst_ptrs, dst_ptrs[1:])
    )


def test_register_kv_caches_deduplicates_shared_backing_memory(monkeypatch):
    monkeypatch.setenv("VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT", "5")
    vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector",
        kv_role="kv_consumer",
        block_size=maybe_update_block_size(16),
    )
    kv_cache_config = make_hybrid_gdn_kv_cache_config(
        vllm_config.cache_config.block_size
    )

    with set_current_vllm_config(vllm_config), patch_worker_dependencies():
        connector = MooncakeConnector(
            vllm_config,
            KVConnectorRole.WORKER,
            kv_cache_config,
        )
        worker = connector.connector_worker

        fa_spec = kv_cache_config.kv_cache_groups[0].kv_cache_spec
        fa_cache_width = (
            fa_spec.page_size_bytes
            // torch.tensor([], dtype=torch.float16).element_size()
        )
        backing = torch.empty((4, fa_cache_width), dtype=torch.float16)
        fa_cache = backing[:2]
        gdn_cache = backing[:3]

        with patch.object(
            worker.engine, "batch_register_memory", return_value=0
        ) as batch_register_memory:
            worker.register_kv_caches(
                {
                    "model.layers.0.self_attn": fa_cache,
                    "model.layers.1.linear_attn": gdn_cache,
                }
            )

        assert worker.kv_caches_base_addr == [
            fa_cache.data_ptr(),
            gdn_cache.data_ptr(),
        ]
        batch_register_memory.assert_called_once()
        registered_ptrs, registered_lens = batch_register_memory.call_args[0]
        assert registered_ptrs == [backing.data_ptr()]
        assert registered_lens == [backing.untyped_storage().nbytes()]

        worker.shutdown()
        worker.shutdown = noop_shutdown
        connector.connector_worker = None


def test_hybrid_gdn_transfer_params_preserve_group_identity(monkeypatch):
    monkeypatch.setenv("VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT", "5")
    vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector",
        kv_role="kv_producer",
        block_size=maybe_update_block_size(16),
    )
    kv_cache_config = make_hybrid_gdn_kv_cache_config(
        vllm_config.cache_config.block_size
    )

    with set_current_vllm_config(vllm_config), patch_worker_dependencies():
        connector = MooncakeConnector(
            vllm_config,
            KVConnectorRole.WORKER,
            kv_cache_config,
        )
        worker = connector.connector_worker

        block_len = 0x100
        transfer_id = "xfer-hybrid-gdn"

        async def build_transfer_params():
            send_meta = SendBlockMeta(
                p_req_id="p-hybrid-gdn",
                transfer_id=transfer_id,
                local_block_ids=[
                    [10, 11],
                    [NULL_BLOCK_ID, 4],
                ],
                ready=asyncio.Event(),
            )
            return await worker._build_transfer_params(
                [("d-hybrid-gdn", send_meta)],
                xfer_meta,
                local_regions,
                remote_regions,
            )

        xfer_meta = MooncakeXferMetadata(
            remote_hostname="consumer-host",
            remote_port=54321,
            remote_tp_size=1,
            remote_tp_rank=0,
            req_blocks={
                "d-hybrid-gdn": (
                    transfer_id,
                    [
                        [30, 31],
                        [NULL_BLOCK_ID, 7],
                    ],
                )
            },
            kv_caches_base_addr=[],
            block_lens=[],
            kv_block_lens=[],
        )

        local_regions = [
            TransferRegion(
                layer_name="model.layers.1.linear_attn",
                layer_index=1,
                base_addr=0x5000,
                block_len=block_len,
                kv_block_len=block_len,
                group_index=1,
            ),
            TransferRegion(
                layer_name="model.layers.0.self_attn",
                layer_index=0,
                base_addr=0x1000,
                block_len=block_len,
                kv_block_len=block_len,
                group_index=0,
            ),
        ]
        remote_regions = [
            TransferRegion(
                layer_name="model.layers.1.linear_attn",
                layer_index=1,
                base_addr=0x6000,
                block_len=block_len,
                kv_block_len=block_len,
                group_index=1,
            ),
            TransferRegion(
                layer_name="model.layers.0.self_attn",
                layer_index=0,
                base_addr=0x2000,
                block_len=block_len,
                kv_block_len=block_len,
                group_index=0,
            ),
        ]

        src_ptrs, dst_ptrs, lengths, err_reqs, err_msg = asyncio.run(
            build_transfer_params()
        )

        assert err_reqs == []
        assert err_msg is None
        assert src_ptrs == [
            0x5000 + 4 * block_len,
            0x1000 + 10 * block_len,
        ]
        assert dst_ptrs == [
            0x6000 + 7 * block_len,
            0x2000 + 30 * block_len,
        ]
        assert lengths == [block_len, 2 * block_len]

        worker.shutdown()
        worker.shutdown = noop_shutdown
        connector.connector_worker = None


def test_logical_to_kernel_block_ids_expands_fa_not_gdn():
    worker = object.__new__(MooncakeConnectorWorker)
    worker.shutdown = noop_shutdown
    worker._physical_blocks_per_logical_kv_block = 17
    worker.kv_cache_config = make_hybrid_gdn_kv_cache_config(block_size=544)

    block_ids = [[2], [2]]
    kernel_block_ids = worker._logical_to_kernel_block_ids(block_ids)

    assert kernel_block_ids == [list(range(34, 51)), [2]]


def test_hybrid_gdn_keeps_packed_fa_and_gdn_regions_whole(
    monkeypatch,
):
    monkeypatch.setenv("VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT", "5")
    vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector",
        kv_role="kv_producer",
        block_size=maybe_update_block_size(16),
    )
    kv_cache_config = make_hybrid_gdn_kv_cache_config(
        vllm_config.cache_config.block_size
    )

    with set_current_vllm_config(vllm_config), patch_worker_dependencies():
        connector = MooncakeConnector(
            vllm_config,
            KVConnectorRole.WORKER,
            kv_cache_config,
        )
        worker = connector.connector_worker

        worker.transfer_topo = SimpleNamespace(is_kv_layout_blocks_first=False)
        regions = worker._get_transfer_regions(
            base_addrs=[0x1000, 0x2000],
            block_lens=[0x100, 0x100],
            kv_block_lens=[0x40, 0x100],
            layer_names=[
                "model.layers.0.self_attn",
                "model.layers.1.linear_attn",
            ],
            layer_indices=[0, 1],
            group_indices=[0, 1],
        )

        assert [
            (region.group_index, region.base_addr, region.kv_block_len)
            for region in regions
        ] == [
            (0, 0x1000, 0x40),
            (1, 0x2000, 0x100),
        ]

        worker.shutdown()
        worker.shutdown = noop_shutdown
        connector.connector_worker = None


def test_kda_dp_transfer_plan_only_matches_corresponding_rank():
    should_send, src_offset, dst_offset, length = _compute_kda_dp_sender_transfer_plan(
        2, 8, 2, 8, 128, 128
    )
    assert should_send
    assert (src_offset, dst_offset, length) == (0, 0, 128)

    should_send, *_ = _compute_kda_dp_sender_transfer_plan(1, 8, 2, 8, 128, 128)
    assert not should_send

    with pytest.raises(ValueError, match="matching producer/consumer KDA sizes"):
        _compute_kda_dp_sender_transfer_plan(1, 8, 1, 4, 128, 128)
    with pytest.raises(ValueError, match="state schema mismatch"):
        _compute_kda_dp_sender_transfer_plan(1, 8, 1, 8, 128, 64)


def test_kda_dp_producer_waits_for_every_state_consumer():
    worker = object.__new__(MooncakeConnectorWorker)
    worker.shutdown = noop_shutdown
    send_meta = SendBlockMeta(
        p_req_id="p-req",
        transfer_id="xfer-req",
        local_block_ids=[],
        ready=asyncio.Event(),
    )

    worker.resolve_need_send(send_meta, [0], remote_kda_tp_size=8)

    assert send_meta.need_send == 8


def test_kda_dp_transfer_params_do_not_duplicate_state_shards():
    kv_cache_config = make_hybrid_gdn_kv_cache_config(block_size=16)
    layer_specs = {
        layer_name: group.kv_cache_spec
        for group in kv_cache_config.transfer_groups
        for layer_name in group.layer_names
    }
    block_len = 0x100
    local_regions = [
        TransferRegion(
            layer_name="model.layers.1.linear_attn",
            layer_index=1,
            base_addr=0x5000,
            block_len=2 * block_len,
            kv_block_len=2 * block_len,
            content_len=block_len,
            group_index=1,
        ),
        TransferRegion(
            layer_name="model.layers.0.self_attn",
            layer_index=0,
            base_addr=0x1000,
            block_len=block_len,
            kv_block_len=block_len,
            group_index=0,
        ),
    ]
    remote_regions = [
        TransferRegion(
            layer_name=region.layer_name,
            layer_index=region.layer_index,
            base_addr=0x6000 if region.group_index == 1 else 0x2000,
            block_len=block_len if region.group_index == 1 else 8 * block_len,
            kv_block_len=block_len if region.group_index == 1 else 8 * block_len,
            group_index=region.group_index,
        )
        for region in local_regions
    ]

    def build_for_rank(local_rank: int):
        worker = object.__new__(MooncakeConnectorWorker)
        worker.shutdown = noop_shutdown
        worker.kv_cache_config = kv_cache_config
        worker._physical_blocks_per_logical_kv_block = 1
        worker._layer_specs = layer_specs
        worker.tp_rank = local_rank
        worker.tp_size = 8
        worker.kda_layer_rank = local_rank
        worker.kda_layer_tp = 8
        worker.use_mla = False
        worker.transfer_topo = SimpleNamespace(
            local_replicates_kv_cache=False, total_num_kv_heads=1
        )
        send_meta = SendBlockMeta(
            p_req_id="p-req",
            transfer_id="xfer-req",
            local_block_ids=[[10, 11], [4]],
            ready=asyncio.Event(),
        )
        agent_meta = MooncakeXferMetadata(
            remote_hostname="consumer-host",
            remote_port=54321,
            remote_tp_size=1,
            remote_tp_rank=0,
            remote_kda_tp_size=8,
            remote_kda_tp_rank=1,
            req_blocks={"d-req": ("xfer-req", [[30, 31], [7]])},
            kv_caches_base_addr=[],
            block_lens=[],
            kv_block_lens=[],
        )
        assert (
            worker._validate_kda_dp_regions(local_regions, remote_regions, agent_meta)
            is None
        )
        return asyncio.run(
            worker._build_transfer_params(
                [("d-req", send_meta)], agent_meta, local_regions, remote_regions
            )
        )

    src_ptrs, dst_ptrs, lengths, err_reqs, err_msg = build_for_rank(1)
    assert err_reqs == [] and err_msg is None
    assert src_ptrs == [
        0x5000 + 4 * (2 * block_len),
        0x1000 + 10 * block_len,
        0x1000 + 11 * block_len,
    ]
    assert dst_ptrs == [
        0x6000 + 7 * block_len,
        0x2000 + (30 * 8 + 1) * block_len,
        0x2000 + (31 * 8 + 1) * block_len,
    ]
    assert lengths == [block_len, block_len, block_len]

    src_ptrs, dst_ptrs, lengths, err_reqs, err_msg = build_for_rank(2)
    assert err_reqs == [] and err_msg is None
    assert src_ptrs == [0x1000 + 10 * block_len, 0x1000 + 11 * block_len]
    assert dst_ptrs == [
        0x2000 + (30 * 8 + 2) * block_len,
        0x2000 + (31 * 8 + 2) * block_len,
    ]
    assert lengths == [block_len, block_len]


def test_dp_fanout_non_owner_only_receives_mamba_blocks():
    kv_cache_config = make_hybrid_gdn_kv_cache_config(block_size=16)
    mamba_group = kv_cache_config.transfer_groups[1]
    transfer_groups = list(kv_cache_config.transfer_groups)
    transfer_groups[1] = SimpleNamespace(
        kv_cache_spec=SimpleNamespace(
            kv_cache_specs={
                layer_name: mamba_group.kv_cache_spec
                for layer_name in mamba_group.layer_names
            }
        )
    )
    kv_cache_config = SimpleNamespace(transfer_groups=transfer_groups)
    pull_meta = PullReqMeta(
        d_req_id="owner-req",
        local_block_ids=[[10, 11], [4]],
        remote_engine_id="p-engine",
        remote_bootstrap_addr="http://p-host:8998",
        transfer_id="xfer-req",
    )
    gathered = [[("p-engine", "owner-req", pull_meta)], []]

    def merge_for_rank(rank: int):
        worker = object.__new__(MooncakeConnectorWorker)
        worker.shutdown = noop_shutdown
        worker.dp_rank = rank
        worker.kv_cache_config = kv_cache_config
        worker._dp_req_owners = {}
        metadata = MooncakeConnectorMetadata()
        return worker._merge_dp_pull_reqs(metadata, gathered)

    owner_meta = merge_for_rank(0).reqs_to_recv["p-engine"]["xfer-req"]
    assert owner_meta.d_req_id == "xfer-req"
    assert owner_meta.local_block_ids == [[10, 11], [4]]

    non_owner_meta = merge_for_rank(1).reqs_to_recv["p-engine"]["xfer-req"]
    assert non_owner_meta.d_req_id == "xfer-req"
    assert non_owner_meta.local_block_ids == [[], [4]]

    gathered[0][0] = (
        "p-engine",
        "owner-req",
        PullReqMeta(
            d_req_id="owner-req",
            local_block_ids=[],
            remote_engine_id="p-engine",
            remote_bootstrap_addr="http://p-host:8998",
            transfer_id="xfer-req",
        ),
    )
    release_meta = merge_for_rank(1).reqs_to_recv["p-engine"]["xfer-req"]
    assert release_meta.local_block_ids == []


def test_dp_completion_waits_for_all_ranks_and_only_notifies_owner(monkeypatch):
    monkeypatch.setattr(
        mooncake_connector_module,
        "get_dp_group",
        lambda: SimpleNamespace(world_size=2, cpu_group=object()),
    )

    def make_worker(rank: int):
        worker = object.__new__(MooncakeConnectorWorker)
        worker.shutdown = noop_shutdown
        worker.dp_rank = rank
        worker._dp_req_owners = {"xfer-req": (0, "owner-req")}
        worker._dp_completed_reqs = set()
        worker._dp_failed_reqs = set()
        return worker

    def gather_packets(monkeypatch, packets):
        def fake_gather(gathered, _local, group):
            gathered[:] = packets

        monkeypatch.setattr(torch.distributed, "all_gather_object", fake_gather)

    owner = make_worker(0)
    gather_packets(monkeypatch, [({"xfer-req"}, set()), (set(), set())])
    result = owner._get_dp_transfer_results(None, {"xfer-req"}, set())
    assert result.finished_recving == set()
    assert owner._dp_completed_reqs == {"xfer-req"}

    gather_packets(
        monkeypatch,
        [({"xfer-req"}, set()), ({"xfer-req"}, set())],
    )
    result = owner._get_dp_transfer_results(None, None, set())
    assert result.finished_recving == {"owner-req"}
    assert owner._dp_completed_reqs == set()
    assert owner._dp_req_owners == {}

    non_owner = make_worker(1)
    gather_packets(
        monkeypatch,
        [({"xfer-req"}, set()), ({"xfer-req"}, set())],
    )
    result = non_owner._get_dp_transfer_results(None, {"xfer-req"}, set())
    assert result.finished_recving == set()
    assert non_owner._dp_completed_reqs == set()

    failed_owner = make_worker(0)
    gather_packets(monkeypatch, [(set(), {"xfer-req"}), (set(), set())])
    result = failed_owner._get_dp_transfer_results(None, None, set())
    assert result.finished_recving == set()
    assert result.failed_recving == {"owner-req"}
    assert failed_owner._dp_failed_reqs == set()
