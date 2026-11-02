# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DPSyncState:
    """What a `dispatch_cg_and_sync_dp` call agreed across DP ranks.

    Every field is identical on every rank, so callers can branch on them
    without disagreeing about whether to run a collective. Never add a per-rank
    value here.
    """

    num_tokens_across_dp: torch.Tensor
    uniform_token_count: int | None
    eager: bool
    num_reqs: int
    max_num_reqs_across_dp: int | None = None
    max_spec_reqs_across_dp: int | None = None
    global_mode: int = 0
    use_plan: bool = False
    fallback_reason: str | None = None


class ExecutionKind(enum.Enum):
    REAL = "real"
    DUMMY = "dummy"
    CAPTURE = "capture"


@dataclass(frozen=True)
class DPBatchPlanKey:
    epoch: int
    dp_size: int
    global_mode: int


@dataclass
class DPBatchShapePlan:
    key: DPBatchPlanKey
    work: dist.Work | None
    tensor_cpu: torch.Tensor
    batch_desc: Any | None = None
    sync_state: DPSyncState | None = None
    use_plan: bool = True
    fallback_reason: str | None = None
    shape_wait_ms: float = 0.0


@dataclass
class DPBatchPlan:
    key: DPBatchPlanKey
    execution_kind: ExecutionKind
    shape: DPBatchShapePlan
    previous_pending_global: bool = False
    kda: Any | None = None
    consumed: bool = False
