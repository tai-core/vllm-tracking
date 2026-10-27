# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration shared by GLM KDA, cache planning and PD transfer."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.config.parallel import ParallelConfig


def kda_layer_tp_axis(parallel: "ParallelConfig") -> str:
    """Resolve explicit KDA TP to an existing complete process group."""
    size = getattr(parallel, "kda_layer_tp", None)
    if size is None:
        return "tp"
    tp = parallel.tensor_parallel_size
    if size <= 0 or size % tp:
        raise ValueError(
            "kda_layer_tp must be a positive multiple of tensor_parallel_size"
        )
    if size == tp:
        return "tp"
    pcp = parallel.prefill_context_parallel_size
    dp = getattr(parallel, "data_parallel_size", 1)
    if pcp > 1 and dp > 1:
        raise ValueError(
            "kda_layer_tp beyond TP with both PCP and DP requires an explicit "
            "group layout, which is not implemented"
        )
    axis, group_size = ("pcp", pcp * tp) if pcp > 1 else ("dp", dp * tp)
    if size != group_size:
        raise ValueError(
            f"kda_layer_tp ({size}) must equal TP ({tp}) or the complete "
            f"TP x {axis.upper()} group ({group_size}); partial groups are not implemented"
        )
    if axis == "pcp" and parallel.decode_context_parallel_size != 1:
        raise ValueError("KDA TP over PCP currently requires DCP=1")
    if axis == "dp" and getattr(parallel, "use_sequence_parallel_moe", False):
        raise ValueError("KDA TP over DP is incompatible with sequence-parallel MoE")
    return axis


def kda_tp_over_pcp(config: "VllmConfig") -> bool:
    return kda_layer_tp_axis(config.parallel_config) == "pcp"


def kda_tp_over_dp(config: "VllmConfig") -> bool:
    return kda_layer_tp_axis(getattr(config, "parallel_config", None)) == "dp"


def get_kda_layer_tp(config: "VllmConfig") -> int:
    parallel = config.parallel_config
    kda_layer_tp_axis(parallel)
    return getattr(parallel, "kda_layer_tp", None) or parallel.tensor_parallel_size


def get_kda_request_windows(config: "VllmConfig") -> int:
    """Only DP contributes independent request pools; PCP shares requests."""
    if kda_tp_over_dp(config):
        return get_kda_layer_tp(config) // config.parallel_config.tensor_parallel_size
    return 1
