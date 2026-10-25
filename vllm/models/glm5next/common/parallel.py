# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

from vllm.config import VllmConfig
from vllm.config.kda_parallel import kda_tp_over_pcp
from vllm.distributed.parallel_state import (
    GroupCoordinator,
    get_pcp_group,
    get_tp_group,
)


@dataclass(frozen=True)
class KDAParallelSpec:
    group: GroupCoordinator
    tp_size: int = 1
    tp_rank: int = 0

    @property
    def size(self) -> int:
        return self.tp_size * self.group.world_size

    @property
    def rank(self) -> int:
        return self.tp_rank * self.group.world_size + self.group.rank_in_group


def get_kda_parallel_spec(config: VllmConfig) -> KDAParallelSpec:
    tp = get_tp_group()
    if kda_tp_over_pcp(config):
        return KDAParallelSpec(get_pcp_group(), tp.world_size, tp.rank_in_group)
    return KDAParallelSpec(tp)


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
            group.world_size, rows * hidden_size * element_size / 2**20,
        )
    return group._kda_pcp_collectives
