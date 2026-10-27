# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the max_num_reqs gate on the V2 mixed prefill+decode warmup."""

from types import SimpleNamespace

import pytest

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker.gpu import warmup
from vllm.v1.worker.gpu.warmup import run_mixed_prefill_decode_warmup


def _fail(*args, **kwargs):
    raise AssertionError("worker callback must not run when warmup is skipped")


@pytest.mark.parametrize("max_num_reqs", [1, 0])
def test_mixed_warmup_skipped_for_single_seq(max_num_reqs):
    """A mixed prefill+decode step needs >=2 requests; with max_num_reqs < 2
    the warmup must be skipped without touching the worker callbacks."""
    runner = SimpleNamespace(is_pooling_model=False, max_num_reqs=max_num_reqs)

    assert (
        run_mixed_prefill_decode_warmup(
            runner,
            worker_execute_model=_fail,
            worker_sample_tokens=_fail,
            num_tokens=128,
        )
        is False
    )


def test_mixed_warmup_disables_watermarking():
    outputs: list[SchedulerOutput] = []
    runner = SimpleNamespace(
        is_pooling_model=False,
        max_num_reqs=2,
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=16))
            ],
            num_blocks=128,
            musa_mamba_num_blocks=None,
        ),
        vllm_config=SimpleNamespace(num_lookahead_tokens=0),
        max_model_len=128,
        num_speculative_steps=0,
        model_state=SimpleNamespace(max_encoder_len=0),
        kv_connector=SimpleNamespace(set_disabled=lambda disabled: None),
    )

    assert run_mixed_prefill_decode_warmup(
        runner,
        worker_execute_model=outputs.append,
        worker_sample_tokens=lambda grammar_output: None,
        num_tokens=32,
    )

    sampling_params = [
        request.sampling_params
        for output in outputs
        for request in output.scheduled_new_reqs
    ]
    assert len(sampling_params) == 2
    assert all(params.watermarking is False for params in sampling_params)


@pytest.mark.parametrize("fail_warmup", [False, True])
def test_kernel_warmup_restores_uncalibrated_adaptive_manager(monkeypatch, fail_warmup):
    """Startup must warm fixed drafts before calibration and retain its manager."""
    manager = SimpleNamespace(cost_tables=None)
    rejection_sampler = SimpleNamespace(enable_adaptive_verification=True)
    runner = SimpleNamespace(
        adaptive_verification=manager,
        rejection_sampler=rejection_sampler,
    )

    def run_steps(model_runner, execute, sample):
        assert model_runner.adaptive_verification is None
        assert not model_runner.rejection_sampler.enable_adaptive_verification
        if fail_warmup:
            raise RuntimeError("warmup failed")

    monkeypatch.setattr(warmup, "_warmup_kernels", run_steps)
    if fail_warmup:
        with pytest.raises(RuntimeError, match="warmup failed"):
            warmup.warmup_kernels(runner, _fail, _fail)
    else:
        warmup.warmup_kernels(runner, _fail, _fail)
    assert runner.adaptive_verification is manager
    assert manager.cost_tables is None
    assert rejection_sampler.enable_adaptive_verification


def test_dp_kda_warmup_starts_from_cached_prefix_and_only_decodes(monkeypatch):
    import torch

    from tests.v1.worker.test_gpu_warmup_blocks import (
        _attention_group,
        _make_runner,
        _mamba_group,
    )

    mamba = _mamba_group("align")
    # No speculative state is needed for DP KDA.
    from dataclasses import replace

    mamba.kv_cache_spec = replace(mamba.kv_cache_spec, num_speculative_blocks=0)
    runner = _make_runner([_attention_group(), mamba], 0, num_spec_steps=0)
    runner.vllm_config.parallel_config = SimpleNamespace(
        tensor_parallel_size=1,
        prefill_context_parallel_size=1,
        data_parallel_size=8,
        decode_context_parallel_size=1,
        kda_layer_tp=8,
    )
    runner.kv_cache_config.musa_mamba_num_blocks = 40
    runner.kv_cache_config.musa_mamba_alloc_num_blocks = 5
    outputs = []
    disabled = []
    runner.kv_connector.set_disabled = disabled.append
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    warmup._warmup_kernels(runner, outputs.append, lambda _: None)
    assert disabled == [True, False]
    active = [out for out in outputs if out.total_num_scheduled_tokens]
    assert active
    first = active[0]
    assert first.scheduled_new_reqs
    for req in first.scheduled_new_reqs:
        assert req.num_computed_tokens == 1
        assert len(req.prompt_token_ids) == 2
        assert req.block_ids[1]
        assert all(0 < block < 5 for block in req.block_ids[1])
    for out in active:
        assert set(out.num_scheduled_tokens.values()) == {1}
    assert outputs[-1].finished_req_ids == {
        req.req_id for req in first.scheduled_new_reqs
    }
    assert not warmup.run_mixed_prefill_decode_warmup(
        runner,
        _fail,
        _fail,
        num_tokens=32,
    )
