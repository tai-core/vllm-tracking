# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM MoE must let its internally routed runner own gate computation."""

import pytest
import torch
from torch import nn

from vllm.models.glm5next.nvidia import model


class _Gate(nn.Linear):
    def forward(self, hidden_states):
        return super().forward(hidden_states), None


class _InternalRouter(nn.Module):
    def __init__(self, gate):
        super().__init__()
        self.gate = gate
        self.register_buffer("expert_values", torch.arange(24).reshape(3, 8).float())

    def forward(self, hidden_states, router_logits):
        logits, _ = self.gate(hidden_states)
        return logits.softmax(-1) @ self.expert_values


@pytest.mark.parametrize("num_tokens", [2, 16, 128])
@pytest.mark.parametrize("parallel", ["none", "chunk", "already_chunked"])
def test_glm_moe_computes_gate_once_and_preserves_routed_output(
    monkeypatch, num_tokens, parallel
):
    layer = object.__new__(model.Glm5NextMoE)
    nn.Module.__init__(layer)
    layer.is_sequence_parallel = parallel != "none"
    layer.gate = _Gate(8, 3, bias=False)
    layer.experts = _InternalRouter(layer.gate)
    hidden_states = torch.arange(num_tokens * 8).reshape(num_tokens, 8).float() / 100
    if parallel == "chunk":
        hidden_states[num_tokens // 2 :] = hidden_states[: num_tokens // 2]
    monkeypatch.setattr(
        model, "sequence_parallel_chunk", lambda x: x[: x.shape[0] // 2]
    )
    monkeypatch.setattr(
        model, "tensor_model_parallel_all_gather", lambda x, dim: torch.cat([x, x])
    )
    calls = []
    handle = layer.gate.register_forward_hook(lambda *args: calls.append(1))
    try:
        result = layer(
            hidden_states, already_sequence_parallel=parallel == "already_chunked"
        )
    finally:
        handle.remove()
    expected_logits = nn.functional.linear(hidden_states, layer.gate.weight)
    expected = expected_logits.softmax(-1) @ layer.experts.expert_values
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert len(calls) == 1
