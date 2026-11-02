# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.spec_decode.dflash import speculator as dflash_speculator
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator
from vllm.v1.worker.gpu.spec_decode.speculator import BaseSpeculator


class _DefaultSpeculator(BaseSpeculator):
    def init_cudagraph_manager(self, cudagraph_mode):
        pass

    def capture(self):
        pass

    def propose(self, *args, **kwargs):
        raise NotImplementedError


class _StopDummyRun(Exception):
    pass


def _run_dummy_batch(
    speculator,
    *,
    max_num_reqs=512,
    is_profile=False,
    skip_attn=False,
):
    captured = {}

    def execute_model(scheduler_output, **kwargs):
        captured["scheduler_output"] = scheduler_output
        raise _StopDummyRun

    runner = SimpleNamespace(
        max_num_reqs=max_num_reqs,
        max_num_tokens=2048,
        decode_query_len=1,
        speculator=speculator,
        kv_connector=SimpleNamespace(set_disabled=lambda disabled: None),
        is_first_pp_rank=True,
        lora_config=None,
        maybe_dummy_run_with_lora=lambda *args, **kwargs: nullcontext(),
        execute_model=execute_model,
    )

    with pytest.raises(_StopDummyRun):
        GPUModelRunner._dummy_run(
            runner,
            num_tokens=2048,
            skip_attn=skip_attn,
            is_profile=is_profile,
        )

    return captured["scheduler_output"]


def test_default_speculator_query_width_is_one():
    assert _DefaultSpeculator().num_query_per_req == 1


@pytest.mark.parametrize("speculator_cls", [DFlashSpeculator, DSparkSpeculator])
@pytest.mark.parametrize(
    ("is_profile", "skip_attn"),
    [(False, False), (True, False), (True, True)],
)
def test_dummy_run_caps_parallel_draft_requests(speculator_cls, is_profile, skip_attn):
    speculator = speculator_cls.__new__(speculator_cls)
    speculator.num_query_per_req = 5

    scheduler_output = _run_dummy_batch(
        speculator,
        is_profile=is_profile,
        skip_attn=skip_attn,
    )
    assert len(scheduler_output.num_scheduled_tokens) == 409
    assert scheduler_output.total_num_scheduled_tokens == 2048


def test_dummy_run_keeps_already_fitting_parallel_draft_batch():
    speculator = DFlashSpeculator.__new__(DFlashSpeculator)
    speculator.num_query_per_req = 5

    scheduler_output = _run_dummy_batch(speculator, max_num_reqs=128)
    assert len(scheduler_output.num_scheduled_tokens) == 128


def test_dummy_run_keeps_default_speculator_batch():
    scheduler_output = _run_dummy_batch(_DefaultSpeculator())
    assert len(scheduler_output.num_scheduled_tokens) == 512


def test_dflash_dummy_propose_makes_sampling_rows_inert(monkeypatch):
    """DP dummy sampling must not overwrite real per-request proposals."""
    speculator = DFlashSpeculator.__new__(DFlashSpeculator)
    speculator.num_query_per_req = 8
    speculator.max_model_len = 64
    speculator.hidden_states = torch.zeros(8, 1)
    speculator.draft_kv_cache_group_id = 0
    speculator.draft_kv_cache_group_ids = []
    speculator.pcp_manager = None
    speculator.sample_idx_mapping = torch.zeros(8, dtype=torch.int64)
    speculator.draft_tokens = torch.zeros(1, 7, dtype=torch.int64)
    monkeypatch.setattr(speculator, "prepare_context_anchor", lambda *args: None)
    monkeypatch.setattr(speculator, "_precompute_context_kv", lambda *args: None)
    monkeypatch.setattr(speculator, "_prepare_eplb_forward", lambda *args: None)
    monkeypatch.setattr(speculator, "_build_uniform_attn_metadata", lambda **kwargs: {})
    speculator._group_causal = False
    speculator.query_cudagraph_manager = None
    speculator.dp_size = 1
    speculator.dp_rank = 0
    speculator.kv_cache_config = None
    speculator.block_tables = SimpleNamespace(
        slot_mappings=torch.zeros(1, 8, dtype=torch.int64)
    )
    sampled_mappings = []

    def generate_draft(*args, **kwargs):
        sampled_mappings.append(speculator.sample_idx_mapping.clone())

    monkeypatch.setattr(speculator, "_generate_draft", generate_draft)
    batch_desc = SimpleNamespace(cg_mode=CUDAGraphMode.NONE, num_tokens=8)
    monkeypatch.setattr(
        dflash_speculator,
        "dispatch_cg_and_sync_dp",
        lambda *args, **kwargs: (batch_desc, None),
    )
    monkeypatch.setattr(
        dflash_speculator, "build_slot_mappings_by_layer", lambda *args: {}
    )
    input_batch = SimpleNamespace(
        num_reqs=1,
        num_tokens=8,
        seq_lens_cpu_upper_bound=torch.tensor([8]),
    )

    speculator.propose(
        input_batch=input_batch,
        attn_metadata={},
        slot_mappings={},
        last_hidden_states=torch.zeros(8, 1),
        aux_hidden_states=None,
        num_sampled=torch.ones(1, dtype=torch.int32),
        num_rejected=torch.zeros(1, dtype=torch.int32),
        last_sampled=torch.zeros(1, dtype=torch.int64),
        next_prefill_tokens=torch.zeros(1, dtype=torch.int64),
        temperature=torch.ones(1),
        seeds=torch.zeros(1, dtype=torch.int64),
        dummy_run=True,
    )

    assert len(sampled_mappings) == 1
    assert torch.all(sampled_mappings[0] == -1)
