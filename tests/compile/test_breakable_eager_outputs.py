# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
from vllm.config import CUDAGraphMode
from vllm.model_executor.layers.fused_moe.runner import moe_runner
from vllm.utils import torch_utils


def make_capture(monkeypatch):
    capture = BreakableCUDAGraphCapture()
    monkeypatch.setattr(capture, "_end_segment", lambda: None)
    monkeypatch.setattr(capture, "_begin_segment", lambda: None)
    capture._capturing = True
    monkeypatch.setattr(BreakableCUDAGraphCapture, "current", lambda: capture)
    monkeypatch.setattr(
        moe_runner,
        "get_forward_context",
        lambda: SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE),
    )
    monkeypatch.setattr(torch_utils, "weak_ref_tensors", lambda inputs: inputs)
    return capture


def run_moe(forward):
    layer = SimpleNamespace(
        moe_config=SimpleNamespace(use_deepep_ht_kernels=True),
        _forward_impl=forward,
    )
    return moe_runner._run_moe_with_graph_boundary(layer)


@pytest.mark.parametrize("shared", [False, True])
def test_eager_outputs_keep_addresses_and_refresh_values(monkeypatch, shared):
    capture = make_capture(monkeypatch)
    x = torch.arange(12, dtype=torch.float32).reshape(3, 4)

    def forward():
        routed = x * 2
        return (routed, x + 3, None) if shared else routed

    output = run_moe(forward)
    routed = output[0] if shared else output
    pointers = (routed.data_ptr(), output[1].data_ptr() if shared else None)
    for _ in range(5):
        x.add_(1)
        capture.replay()
        torch.testing.assert_close(routed, x * 2)
        assert routed.data_ptr() == pointers[0]
        if shared:
            torch.testing.assert_close(output[1], x + 3)
            assert output[1].data_ptr() == pointers[1]
            assert output[2] is None


def test_eager_output_metadata_change_is_rejected(monkeypatch):
    capture = make_capture(monkeypatch)
    x = torch.ones(3, 4)
    run_moe(lambda: x.clone())
    x = torch.ones(4, 4)
    with pytest.raises(RuntimeError, match="metadata changed"):
        capture.replay()
