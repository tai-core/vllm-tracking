# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import vllm.v1.worker.gpu.model_runner as model_runner_module
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.model_runner import GPUModelRunner


@pytest.mark.parametrize("enable_apc", [False, True])
@pytest.mark.parametrize("prefix", [8192, 17408, 24576])
def test_align_state_resume_index_does_not_depend_on_local_prefix_cache(
    monkeypatch, enable_apc, prefix
):
    from vllm.v1.worker.gpu.model_states.default import DefaultModelState
    from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState

    def initialize(self, *args):
        self.max_num_reqs = 1
        self.device = torch.device("cpu")

    monkeypatch.setattr(DefaultModelState, "__init__", initialize)
    monkeypatch.setattr(DefaultModelState, "add_request", lambda *args: None)
    config = SimpleNamespace(
        cache_config=SimpleNamespace(
            mamba_cache_mode="align",
            enable_prefix_caching=enable_apc,
            use_kda_recoverssm=False,
            block_size=1024,
        )
    )
    state = MambaHybridModelState(
        config, torch.nn.Identity(), None, torch.device("cpu")
    )
    assert state._align_mode
    state.add_request(0, SimpleNamespace(num_computed_tokens=prefix))
    assert state._mamba_state_idx_gpu.tolist() == [prefix // 1024 - 1]


@pytest.mark.parametrize(
    ("dummy_run", "num_tokens", "wait_fails"),
    [(False, 1, False), (False, 1, True), (False, 0, False), (True, 1, False)],
)
def test_execute_model_waits_for_restore_before_recurrent_state_read(
    monkeypatch, dummy_run, num_tokens, wait_fails
):
    events = []

    class BoundaryReached(Exception):
        pass

    def dispatch(*args, **kwargs):
        if dummy_run:
            raise BoundaryReached("dummy dispatch")
        return SimpleNamespace(num_tokens=num_tokens), None

    def wait_for_load():
        events.append("wait_load")
        if wait_fails:
            raise RuntimeError("restore failed")

    def read_state(*args):
        events.append("read_state")
        raise BoundaryReached("state read")

    empty_output = object()
    connector = SimpleNamespace(
        start_pre_state_load=lambda output: events.append("submit_load"),
        wait_for_pre_state_load=wait_for_load,
        no_forward=Mock(return_value=empty_output),
    )
    runner = SimpleNamespace(
        update_pp_decode_requests=Mock(),
        finish_requests=Mock(),
        free_states=Mock(),
        add_requests=Mock(),
        update_requests=Mock(),
        block_tables=SimpleNamespace(
            apply_staged_writes=lambda: events.append("table_writes")
        ),
        kv_connector=connector,
        _merge_ec_connector_no_forward=lambda output, empty: empty,
        gather_batch_req_state=lambda output, dummy: (
            SimpleNamespace(num_tokens=num_tokens),
            1,
        ),
        pcp_manager=None,
        lora_config=None,
        is_encoder_decoder=False,
        cudagraph_manager=None,
        dp_size=1,
        dp_rank=0,
        prepare_inputs=lambda *args: events.append("prepare_inputs"),
        prepare_attn=lambda batch: (events.append("prepare_attn"), None),
        model_state=SimpleNamespace(preprocess_state=read_state),
        kv_cache_config=object(),
        req_states=SimpleNamespace(num_computed_tokens=SimpleNamespace(gpu=None)),
    )
    output = SimpleNamespace(
        total_num_scheduled_tokens=num_tokens,
        num_scheduled_tokens={"restore": num_tokens} if num_tokens else {},
    )
    monkeypatch.setattr(model_runner_module, "dispatch_cg_and_sync_dp", dispatch)

    if num_tokens == 0:
        assert GPUModelRunner.execute_model(runner, output) is empty_output
        assert events == ["table_writes", "submit_load"]
        connector.no_forward.assert_called_once_with(output)
    elif dummy_run:
        with pytest.raises(BoundaryReached, match="dummy dispatch"):
            GPUModelRunner.execute_model(runner, output, dummy_run=True)
        assert events == []
    else:
        error, message = (
            (RuntimeError, "restore failed")
            if wait_fails
            else (BoundaryReached, "state read")
        )
        with pytest.raises(error, match=message):
            GPUModelRunner.execute_model(runner, output)
        assert events == [
            "table_writes",
            "submit_load",
            "prepare_inputs",
            "prepare_attn",
            "wait_load",
        ] + ([] if wait_fails else ["read_state"])


def test_qsa_circular_group_uses_custom_slot_mapping(monkeypatch):
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.max_model_len = 262144
    runner.is_encoder_decoder = False
    runner.dcp_size = 1
    runner.dcp_rank = 0
    runner.cp_interleave = 1
    runner.cache_config = SimpleNamespace(enable_prefix_caching=True)
    parallel_config = SimpleNamespace(
        decode_context_parallel_size=1,
        cp_kv_cache_interleave_size=1,
    )
    runner.parallel_config = parallel_config
    runner.vllm_config = SimpleNamespace(
        parallel_config=parallel_config,
        cache_config=SimpleNamespace(mamba_cache_mode="none"),
    )
    runner.model_state = SimpleNamespace(
        get_additional_cg_support=lambda: (),
        num_new_sampled_tokens_per_step=1,
    )
    runner.speculator = None
    runner.req_states = []
    runner.input_buffers = SimpleNamespace(query_start_loc=None)
    runner.vocab_size = 1
    runner.max_num_reqs = 1
    runner.max_num_tokens = 2
    runner.device = torch.device("cuda")

    raw_spec = CircularBufferSpec(
        block_size=8,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
    )
    compressed_spec = FullAttentionSpec(
        block_size=262144,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["raw"],
                kv_cache_spec=UniformTypeKVCacheSpecs(
                    block_size=8,
                    kv_cache_specs={"raw": raw_spec},
                ),
            ),
            KVCacheGroupSpec(layer_names=["compressed"], kv_cache_spec=compressed_spec),
        ],
    )

    class FakeAttnCGSupport:
        def narrow(self, *args):
            return self

    attn_cg_support = FakeAttnCGSupport()
    monkeypatch.setattr(
        model_runner_module,
        "init_attn_backend",
        lambda *args: ([], attn_cg_support, [8, 262144]),
    )
    monkeypatch.setattr(
        model_runner_module,
        "maybe_create_adaptive_verification_manager",
        lambda **kwargs: None,
    )

    captured = {}

    class BlockTablesCaptured(Exception):
        pass

    def capture_block_tables(**kwargs):
        captured.update(kwargs)
        raise BlockTablesCaptured

    monkeypatch.setattr(model_runner_module, "BlockTables", capture_block_tables)

    with pytest.raises(BlockTablesCaptured):
        runner.initialize_kv_cache(kv_cache_config)

    assert captured["max_num_blocks_per_group"] == [1, 1]
    assert captured["slot_mapping_enabled"] == [False, True]


@pytest.mark.parametrize(
    ("mamba_cache_mode", "num_speculative_blocks", "expected"),
    [
        pytest.param("align", 0, 65_536, id="align-prefix-cache"),
        pytest.param("none", 7, 8, id="no-prefix-cache-with-speculation"),
    ],
)
def test_initialize_kv_cache_does_not_dcp_shard_mamba_block_table(
    monkeypatch,
    mamba_cache_mode: str,
    num_speculative_blocks: int,
    expected: int,
):
    """Mamba/GDN block-table rows index global positions, unlike DCP KV."""

    max_model_len = 1_048_576
    attention_block_size = 1_536
    mamba_block_size = 16
    dcp_size = 8
    full_attention_spec = FullAttentionSpec(
        block_size=attention_block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.bfloat16,
    )
    mamba_spec = MambaSpec(
        shapes=((1,),),
        dtypes=(torch.bfloat16,),
        block_size=mamba_block_size,
        mamba_cache_mode=mamba_cache_mode,
        num_speculative_blocks=num_speculative_blocks,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["attention"], full_attention_spec),
            KVCacheGroupSpec(["kda"], mamba_spec),
        ],
    )
    parallel_config = SimpleNamespace(
        decode_context_parallel_size=dcp_size,
        cp_kv_cache_interleave_size=1,
    )
    vllm_config = SimpleNamespace(
        parallel_config=parallel_config,
        cache_config=SimpleNamespace(mamba_cache_mode=mamba_cache_mode),
    )
    runner = SimpleNamespace(
        max_model_len=max_model_len,
        is_encoder_decoder=False,
        vllm_config=vllm_config,
        parallel_config=parallel_config,
    )

    class _CapturedWidths(Exception):
        pass

    captured: list[int] = []

    def capture_width(max_num_blocks: int, *_args, **_kwargs) -> int:
        captured.append(max_num_blocks)
        if len(captured) == 2:
            raise _CapturedWidths
        return max_num_blocks

    monkeypatch.setattr(model_runner_module, "get_block_table_width", capture_width)

    with pytest.raises(_CapturedWidths):
        GPUModelRunner.initialize_kv_cache(runner, kv_cache_config)

    # Attention KV is local to one of eight DCP ranks; KDA state is replicated
    # and therefore needs one table entry for every global 16-token page.
    assert captured == [86, expected]


def test_append_block_ids_rejects_write_past_row_capacity():
    """Reject an oversized staged write before it can corrupt the next row."""

    class _BlockTable:
        gpu = torch.empty((2, 4), dtype=torch.int32)

        def stage_write(self, *_args):
            pytest.fail("an oversized write must not be staged")

    block_tables = BlockTables.__new__(BlockTables)
    block_tables.num_kv_cache_groups = 1
    block_tables.blocks_per_kv_block = [1]
    block_tables.block_tables = [_BlockTable()]
    block_tables.num_blocks = SimpleNamespace(
        np=torch.tensor([[0, 3]], dtype=torch.int32)
    )

    with pytest.raises(
        RuntimeError,
        match=r"request 1, group 0 exceeds row capacity \(5 > 4\)",
    ):
        block_tables.append_block_ids(
            req_index=1,
            new_block_ids=([4, 5],),
            overwrite=False,
        )

    assert block_tables.num_blocks.np[0, 1] == 3
