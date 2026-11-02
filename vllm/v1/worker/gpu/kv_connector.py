# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING, Any

import torch

from vllm.config import VllmConfig
from vllm.config.kda_parallel import kda_mamba_shared_pool, kda_tp_over_dp
from vllm.distributed.kv_transfer import (
    get_kv_transfer_group,
    has_kv_transfer_group,
    kv_transfer_state,
)
from vllm.distributed.kv_transfer.kv_connector.utils import copy_kv_blocks
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorTransferResults,
)
from vllm.distributed.kv_transfer.kv_connector.v1.decode_bench_connector import (
    DecodeBenchConnectorMetadata,
)
from vllm.forward_context import (
    get_forward_context,
    is_forward_context_available,
    set_forward_context,
)
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    KVConnectorOutput,
    ModelRunnerOutput,
)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu.input_batch import InputBatch


class KVConnector:
    """KVConnector interface used by GPUModelRunner."""

    def pre_forward(
        self,
        scheduler_output: "SchedulerOutput",
        input_batch: "InputBatch | None" = None,
        attn_metadata: dict[str, Any] | None = None,
    ) -> None:
        pass

    def finish_forward(self) -> None:
        pass

    def post_forward(self, finished_req_ids: set[str]) -> KVConnectorOutput | None:
        return None

    def no_forward(self, scheduler_output: "SchedulerOutput") -> ModelRunnerOutput:
        return EMPTY_MODEL_RUNNER_OUTPUT

    def set_disabled(self, disabled: bool) -> None:
        pass

    def reset_capture_state(self) -> None:
        pass


class ActiveKVConnector(KVConnector):
    def __init__(
        self, vllm_config: VllmConfig, kv_caches_dict: dict[str, torch.Tensor]
    ):
        self.vllm_config = vllm_config
        kv_transfer_config = vllm_config.kv_transfer_config
        self._dp_global_state_fill = (
            kv_transfer_config is not None
            and kv_transfer_config.kv_connector == "DecodeBenchConnector"
            and kda_tp_over_dp(vllm_config)
        )
        self._dp_global_state_transfer = (
            kv_transfer_config is not None
            and kv_transfer_config.kv_connector == "MooncakeConnector"
            and kda_mamba_shared_pool(vllm_config)
        )
        self._dp_transfer_results: KVConnectorTransferResults | None = None
        self._dp_pending_metadata: KVConnectorMetadata | None = None
        self.kv_connector = get_kv_transfer_group()
        # Register kv caches with KV Connector if applicable.
        self.kv_connector.register_kv_caches(kv_caches_dict)
        self.kv_connector.set_host_xfer_buffer_ops(copy_kv_blocks)

        self._pending_load_kwargs: dict[str, Any] | None = None
        self._disabled = False

    @staticmethod
    def _merge_dp_metadata(target: Any, source: Any) -> Any:
        for engine_id, reqs in source.reqs_to_recv.items():
            target.reqs_to_recv[engine_id].update(reqs)
        target.reqs_to_send.update(source.reqs_to_send)
        target.reqs_not_processed.update(source.reqs_not_processed)
        return target

    def pre_forward(
        self,
        scheduler_output: "SchedulerOutput",
        input_batch: "InputBatch | None" = None,
        attn_metadata: dict[str, Any] | None = None,
    ) -> None:
        if self._disabled:
            return

        kv_connector_metadata = scheduler_output.kv_connector_metadata
        if self._dp_global_state_transfer and self._dp_pending_metadata is not None:
            pending = self._dp_pending_metadata
            self._dp_pending_metadata = None
            kv_connector_metadata = (
                pending
                if kv_connector_metadata is None
                else self._merge_dp_metadata(kv_connector_metadata, pending)
            )
        if kv_connector_metadata is None and self._dp_global_state_fill:
            kv_connector_metadata = DecodeBenchConnectorMetadata(reqs_to_fill={})
        if kv_connector_metadata is None and self._dp_global_state_transfer:
            from vllm.distributed.kv_transfer.kv_connector.v1.mooncake import (
                mooncake_connector,
            )

            kv_connector_metadata = mooncake_connector.MooncakeConnectorMetadata()
        assert kv_connector_metadata is not None
        self.kv_connector.handle_preemptions(kv_connector_metadata)
        self.kv_connector.bind_connector_metadata(kv_connector_metadata)
        self._pending_load_kwargs = self._build_load_kwargs(
            scheduler_output, input_batch, attn_metadata
        )

        if (
            scheduler_output.has_sync_kv_loads
            or self._dp_global_state_fill
            or self._dp_global_state_transfer
        ):
            # Sync loads and DP-wide state work run before this step's forward.
            self._start_load_kv()
        if self._dp_global_state_transfer:
            # DP dummies do not reach post_forward. Poll and vote here so every
            # rank joins the same collectives before the next model step.
            self._dp_transfer_results = self.kv_connector.get_transfer_results(set())
        # Otherwise start the async load after forward to keep submission
        # off the critical path.

    @staticmethod
    def _build_load_kwargs(
        scheduler_output: "SchedulerOutput",
        input_batch: "InputBatch | None",
        attn_metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Keyword args for the connector's start_load_kv()."""
        if input_batch is None:
            # No forward this step.
            return {}
        load_kwargs: dict[str, Any] = dict(
            scheduler_output=scheduler_output,
            request_state_indices=input_batch.idx_mapping,
            request_ids=input_batch.req_ids,
            num_tokens=input_batch.num_tokens,
        )
        if attn_metadata is not None:
            load_kwargs["attn_metadata"] = attn_metadata
        return load_kwargs

    def _start_load_kv(self) -> None:
        load_kwargs = self._pending_load_kwargs
        assert load_kwargs is not None
        self._pending_load_kwargs = None
        # TODO: sort out KV Connectors' use of forward_context
        if is_forward_context_available():
            self.kv_connector.start_load_kv(get_forward_context(), **load_kwargs)
        else:
            with set_forward_context(None, self.vllm_config):
                self.kv_connector.start_load_kv(get_forward_context(), **load_kwargs)

    def finish_forward(self) -> None:
        if not self._disabled:
            self.kv_connector.finish_forward()

    def reset_capture_state(self) -> None:
        self.kv_connector.reset_capture_state()

    def post_forward(self, finished_req_ids: set[str]) -> KVConnectorOutput | None:
        if self._disabled:
            return None

        if self._pending_load_kwargs is not None:
            self._start_load_kv()

        output = KVConnectorOutput()
        self.kv_connector.wait_for_save()
        if self._dp_global_state_transfer:
            transfer_results = self._dp_transfer_results
            self._dp_transfer_results = None
            if transfer_results is None:
                transfer_results = KVConnectorTransferResults()
        else:
            transfer_results = self.kv_connector.get_transfer_results(finished_req_ids)
        output.finished_sending = transfer_results.finished_sending or None
        output.finished_recving = transfer_results.finished_recving or None
        output.failed_recving = transfer_results.failed_recving
        output.invalid_block_ids = self.kv_connector.get_block_ids_with_load_errors()
        output.kv_connector_stats = self.kv_connector.get_kv_connector_stats()
        output.kv_cache_events = self.kv_connector.get_kv_connector_kv_cache_events()
        output.kv_connector_worker_meta = (
            self.kv_connector.build_connector_worker_meta()
        )
        self.kv_connector.clear_connector_metadata()
        return output

    def no_forward(self, scheduler_output: "SchedulerOutput") -> ModelRunnerOutput:
        if self._disabled or self._dp_global_state_fill:
            # The DP dummy that follows a zero-token step joins the state fill.
            return EMPTY_MODEL_RUNNER_OUTPUT

        if self._dp_global_state_transfer:
            # Only ranks with local requests run the real zero-token step. Do
            # not join a DP collective here: the following dummy step is the
            # first point where every rank can fan out and vote together.
            metadata = scheduler_output.kv_connector_metadata
            if metadata is not None:
                self._dp_pending_metadata = (
                    metadata
                    if self._dp_pending_metadata is None
                    else self._merge_dp_metadata(self._dp_pending_metadata, metadata)
                )
        else:
            self.pre_forward(scheduler_output)
        self.finish_forward()
        finished_req_ids = scheduler_output.finished_req_ids
        kv_connector_output = self.post_forward(finished_req_ids)
        return ModelRunnerOutput.with_kv_conn_output_only(kv_connector_output)

    def set_disabled(self, disabled: bool) -> None:
        if disabled and (self._dp_global_state_fill or self._dp_global_state_transfer):
            # Dummy ranks must join the DP-wide state work.
            return
        # Ensure that layer-wise connector hooks aren't called when disabled.
        kv_transfer_state._KV_CONNECTOR_AGENT = None if disabled else self.kv_connector
        self._disabled = disabled


NO_OP_KV_CONNECTOR = KVConnector()


def get_kv_connector(
    vllm_config: VllmConfig, kv_caches_dict: dict[str, torch.Tensor]
) -> KVConnector:
    if not has_kv_transfer_group():
        # No-op connector.
        return NO_OP_KV_CONNECTOR

    return ActiveKVConnector(vllm_config, kv_caches_dict)
