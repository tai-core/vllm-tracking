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
