# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration shared by GLM KDA, cache planning and PD transfer."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig


def kda_tp_over_pcp(config: "VllmConfig") -> bool:
    extra = getattr(config, "additional_config", None)
    mode = extra.get("kda_parallel_mode", "tp") if isinstance(extra, dict) else "tp"
    if mode not in ("tp", "tp_pcp"):
        raise ValueError(f"Unknown kda_parallel_mode: {mode}")
    parallel = config.parallel_config
    enabled = mode == "tp_pcp" and parallel.prefill_context_parallel_size > 1
    if enabled and parallel.decode_context_parallel_size != 1:
        raise ValueError("KDA TP over PCP currently requires DCP=1")
    return enabled


def kda_tensor_parallel_size(config: "VllmConfig") -> int:
    parallel = config.parallel_config
    if kda_tp_over_pcp(config):
        return parallel.tensor_parallel_size * parallel.prefill_context_parallel_size
    return parallel.tensor_parallel_size
