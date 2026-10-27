# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

from vllm.config import VllmConfig
from vllm.config.kda_parallel import (
    get_kda_layer_tp,
    get_kda_request_windows,
    kda_layer_tp_axis,
    kda_tp_over_pcp,
)
from vllm.distributed.parallel_state import (
    GroupCoordinator,
    get_ep_group,
    get_pcp_group,
    get_tp_group,
)


@dataclass(frozen=True)
class KDAParallelSpec:
    group: GroupCoordinator
    kda_layer_tp: int
    kda_layer_rank: int
    token_layout: str
    request_windows: int


def get_kda_parallel_spec(config: VllmConfig) -> KDAParallelSpec:
    tp = get_tp_group()
    layout = kda_layer_tp_axis(config.parallel_config)
    kda_layer_tp = get_kda_layer_tp(config)
    if layout == "pcp":
        group = get_pcp_group()
        rank = tp.rank_in_group * group.world_size + group.rank_in_group
    elif layout == "dp":
        group = get_ep_group()
        assert group.world_size == kda_layer_tp
        rank = group.rank_in_group
    else:
        group = tp
        rank = tp.rank_in_group
    return KDAParallelSpec(
        group=group,
        kda_layer_tp=kda_layer_tp,
        kda_layer_rank=rank,
        token_layout=layout,
        request_windows=get_kda_request_windows(config),
    )


def get_kda_pcp_collectives(config: VllmConfig, hidden_size: int):
    """One staging allocation per PCP group, shared by all KDA layers."""
    from vllm.platforms import current_platform

    extra = config.additional_config
    backend = extra.get("kda_pcp_collective", "auto")
    if backend not in ("auto", "mccl"):
        raise ValueError(f"Unknown kda_pcp_collective: {backend}")
    if (
        backend == "mccl"
        or not kda_tp_over_pcp(config)
        or not current_platform.is_musa()
        or not config.model_config.enforce_eager
        or config.parallel_config.prefill_context_parallel_size not in (4, 8)
    ):
        return None
    import torch
    from vllm_musa.distributed.device_communicators.pcp_collectives import (
        PCPCollectives,
    )

    dtype = config.model_config.dtype
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return None
    element_size = torch.empty((), dtype=dtype).element_size()
    if hidden_size * element_size % 16:
        return None
    group = get_pcp_group()
    if not hasattr(group, "_kda_pcp_collectives"):
        scheduler = config.scheduler_config
        # Reserve global rows plus DualChunkSwap padding. This is allocated
        # before memory profiling so the KV cache planner sees its cost.
        rows = scheduler.max_num_batched_tokens + (
            2 * group.world_size * scheduler.max_num_seqs
        )
        group._kda_pcp_collectives = PCPCollectives(
            group, capacity=rows * hidden_size * element_size
        )
        from vllm.logger import init_logger

        init_logger(__name__).info(
            "KDA mapped Gather/ReduceScatter enabled: PCP=%d, staging=%.2f MiB",
            group.world_size,
            rows * hidden_size * element_size / 2**20,
        )
    return group._kda_pcp_collectives
