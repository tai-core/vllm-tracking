# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.distributed as dist

from vllm.config.kda_parallel import kda_tp_over_dp
from vllm.logger import init_logger
from vllm.models.glm5next.common.parallel import get_kda_parallel_spec
from vllm.v1.worker.gpu.dp_batch_plan import DPSyncState

logger = init_logger(__name__)


@dataclass
class KdaDPSpecCommon:
    query_start_loc: torch.Tensor
    spec_token_indx: torch.Tensor
    num_valid: torch.Tensor
    accepted: torch.Tensor


def get_kda_dp_request_bound(
    sync_state: DPSyncState, *, use_padded_requests: bool
) -> int:
    if sync_state.global_mode != 1:
        return int(sync_state.num_tokens_across_dp.max().item())
    if sync_state.max_spec_reqs_across_dp is not None:
        return sync_state.max_spec_reqs_across_dp
    if use_padded_requests:
        return sync_state.num_reqs
    if sync_state.max_num_reqs_across_dp is not None:
        return sync_state.max_num_reqs_across_dp
    return sync_state.num_reqs


class KdaDPMetadataPlan:
    """One scheduler batch's shared KDA metadata and its device-side work."""

    def __init__(
        self,
        vllm_config,
        sync_state: DPSyncState,
        previous,
        device,
        *,
        use_padded_requests: bool = False,
    ):
        self.sync_state = sync_state
        self.group = get_kda_parallel_spec(vllm_config).group
        self.max_tokens = int(sync_state.num_tokens_across_dp.max().item())
        self.max_requests = get_kda_dp_request_bound(
            sync_state, use_padded_requests=use_padded_requests
        )
        self.global_mode = sync_state.global_mode
        self.spec_common: KdaDPSpecCommon | None = None
        self.spec_accepted_semantics: str | None = None
        self.gather_inputs: list[torch.Tensor] = []
        self.gather_outputs: list[torch.Tensor] = []
        self.consumer_tensors: list[torch.Tensor] = []
        self.plan_stream = (
            previous.plan_stream if previous is not None else torch.cuda.Stream(device)
        )
        self.inputs_ready_event = torch.cuda.Event()
        self.inputs_ready_event.record(torch.cuda.current_stream())
        self.plan_stream.wait_event(self.inputs_ready_event)
        if previous is not None and previous.buffers_free_event is not None:
            self.plan_stream.wait_event(previous.buffers_free_event)
        self.ready_event: torch.cuda.Event | None = None
        self.buffers_free_event: torch.cuda.Event | None = None

    @contextmanager
    def use_stream(self):
        with torch.cuda.stream(self.plan_stream):
            yield

    def all_gather(self, input_: torch.Tensor) -> torch.Tensor:
        output = torch.empty(
            (self.group.world_size * input_.shape[0], *input_.shape[1:]),
            dtype=input_.dtype,
            device=input_.device,
        )
        gather_input = input_.contiguous()
        gather_input.record_stream(self.plan_stream)
        # The synchronous C10d API enqueues MCCL on the current stream and
        # returns without host synchronization. No Work wait is needed.
        dist.all_gather_into_tensor(
            output,
            gather_input,
            group=self.group.device_group,
        )
        self.gather_inputs.append(gather_input)
        self.gather_outputs.append(output)
        return output

    def track_consumers(self, *tensors: torch.Tensor | None) -> None:
        self.consumer_tensors.extend(
            tensor for tensor in tensors if tensor is not None and tensor.is_cuda
        )

    def finish(self) -> None:
        main_stream = torch.cuda.current_stream()
        for tensor in (*self.gather_outputs, *self.consumer_tensors):
            tensor.record_stream(main_stream)
        self.ready_event = torch.cuda.Event()
        self.ready_event.record(self.plan_stream)
        main_stream.wait_event(self.ready_event)

    def record_buffers_free(self) -> None:
        self.buffers_free_event = torch.cuda.Event()
        self.buffers_free_event.record(torch.cuda.current_stream())


def make_kda_dp_plan(
    vllm_config,
    sync_state: DPSyncState | None,
    previous: KdaDPMetadataPlan | None,
    device,
    *,
    use_padded_requests: bool = False,
) -> KdaDPMetadataPlan | None:
    if sync_state is None or not sync_state.use_plan:
        return None
    if not kda_tp_over_dp(vllm_config) or sync_state.global_mode not in (0, 1):
        return None
    group = get_kda_parallel_spec(vllm_config).group
    if group.world_size != sync_state.num_tokens_across_dp.numel():
        raise RuntimeError("KDA/DP group mismatch must be rejected before planning")
    if previous is not None and previous.buffers_free_event is None:
        raise RuntimeError("Previous KDA DP plan buffers are still owned by a step")
    logger.info_once(
        "KDA DP metadata plan enabled (DP=%d, mode=%d)",
        group.world_size,
        sync_state.global_mode,
    )
    return KdaDPMetadataPlan(
        vllm_config,
        sync_state,
        previous,
        device,
        use_padded_requests=use_padded_requests,
    )
