# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash KDA layer with separate convolutions and a bounded safe gate."""

import functools
from dataclasses import replace

import torch
import torch.distributed
from torch import nn
from torch.nn import functional as F
from transformers import Glm5NextTextConfig

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.config.kda_parallel import kda_dp_compact_decode, kda_mamba_shared_pool
from vllm.distributed import divide
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.mamba.checkpoint import (
    MambaPrefillCheckpointMetadata,
)
from vllm.model_executor.layers.mamba.gdn.base import GatedDeltaNetAttention
from vllm.model_executor.layers.mamba.kda_checkpoint import (
    FlashKDAPrefillCheckpointExporter,
    kda_prefill_checkpoint_alignment,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
    is_conv_state_dim_first,
)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from vllm.model_executor.layers.mamba.ops.gather_initial_states import (
    gather_initial_states,
)
from vllm.model_executor.layers.mamba.ops.scatter_states import scatter_states
from vllm.model_executor.model_loader.weight_utils import sharded_weight_loader
from vllm.model_executor.utils import (
    maybe_disable_graph_partition,
    set_weight_attrs,
)
from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops.kda import FusedRMSNormGated
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.kv_cache_interface import MambaSpec, UniformTypeKVCacheSpecs
from vllm.v1.worker.workspace import current_workspace_manager

from .parallel import get_kda_parallel_spec

if current_platform.is_rocm():
    from vllm.models.glm5next.amd.ops.third_party.kda import (
        chunk_kda_with_fused_gate,
        fused_recurrent_kda,
    )
else:
    from vllm.models.glm5next.nvidia.ops.third_party.kda import (
        chunk_kda_with_fused_gate,
        fused_recurrent_kda,
    )


class _Glm5NextMergedColumnParallelLinear(MergedColumnParallelLinear):
    """Merged projection with multiple replicated output shards.

    Extends K3's ``_KimiGDNMergedColumnParallelLinear`` to support two
    replicated shards (f_a, g_a) instead of one. Pre-multiplies each
    replicated entry's output_size by tp_size so the per-rank shard
    divides back to the full size, and forces tp_rank=0 during weight
    loading for replicated shards.
    """

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        replicated_shard_ids: tuple[int, ...],
        tp_size: int,
        tp_rank: int | None = None,
        **kwargs,
    ) -> None:
        self.replicated_shard_ids = set(replicated_shard_ids)
        output_sizes = output_sizes.copy()
        for sid in self.replicated_shard_ids:
            output_sizes[sid] *= tp_size
        super().__init__(
            input_size, output_sizes, tp_size=tp_size, tp_rank=tp_rank, **kwargs
        )

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: tuple[int, ...] | int | None = None,
    ) -> None:
        tp_rank = self.tp_rank
        param_tp_rank = getattr(param, "tp_rank", None)
        if loaded_shard_id in self.replicated_shard_ids:
            self.tp_rank = 0
            if param_tp_rank is not None:
                param.tp_rank = 0
        try:
            super().weight_loader(param, loaded_weight, loaded_shard_id)
        finally:
            self.tp_rank = tp_rank
            if param_tp_rank is not None:
                param.tp_rank = param_tp_rank

    def weight_loader_v2(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: tuple[int, ...] | int | None = None,
    ) -> None:
        tp_rank = self.tp_rank
        param_tp_rank = getattr(param, "tp_rank", None)
        if loaded_shard_id in self.replicated_shard_ids:
            self.tp_rank = 0
            if param_tp_rank is not None:
                param.tp_rank = 0
        try:
            super().weight_loader_v2(param, loaded_weight, loaded_shard_id)
        finally:
            self.tp_rank = tp_rank
            if param_tp_rank is not None:
                param.tp_rank = param_tp_rank


@torch.compile(
    dynamic=True,
    backend=current_platform.simple_compile_backend,
    options=maybe_disable_graph_partition(current_platform.simple_compile_backend),
)
def _cast_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Fuse the fp32 cast + sigmoid into one Inductor kernel."""
    return x.float().sigmoid()


def _resolve_kda_prefill_backend(
    backend: str, head_dim: int, dtype: torch.dtype, lower_bound: float | None
) -> str:
    """Pick FlashKDA on supported accelerators (including MUSA MP31) or the Triton
    ``chunk_kda_with_fused_gate`` path. ``backend`` comes from
    ``additional_config.kda_prefill_backend`` (auto / triton / flashkda)."""
    if backend not in ("auto", "triton", "flashkda"):
        raise ValueError(f"Unsupported KDA prefill backend: {backend}")
    is_musa = current_platform.device_type == "musa"
    if is_musa:
        # Query the native architecture, not a compatibility capability.
        import torch_musa  # noqa: F401

        architecture_supported = torch.musa.get_device_capability() == (3, 1)
    else:
        capability = current_platform.get_device_capability()
        architecture_supported = (
            current_platform.is_cuda()
            and capability is not None
            and capability.major in (9, 10, 12)
        )
    supported = (
        architecture_supported
        and head_dim == 128
        and dtype == torch.bfloat16
        and lower_bound is not None
        and (not is_musa or -5.0 <= lower_bound <= 0.0)
    )
    if is_musa and supported and backend != "triton":
        try:
            import flash_kda  # noqa: F401

            _ = flash_kda.fwd
        except (ImportError, AttributeError) as exc:
            if backend == "flashkda":
                raise RuntimeError(
                    "MUSA FlashKDA is not installed; build FlashKDA for MP31."
                ) from exc
            supported = False
    if backend == "flashkda" and not supported:
        raise RuntimeError(
            "FlashKDA requires a supported accelerator, bfloat16, head_dim=128 "
            "and a bounded KDA gate in [-5, 0]."
        )
    return "flashkda" if supported and backend != "triton" else "triton"


def compact_gdn_rows_for_dp(
    state_indices: torch.Tensor,
    valid_rows: torch.Tensor,
    out_order: torch.Tensor | None = None,
    out_num_valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stably move real rows first without reading their count on the host."""
    num_rows = state_indices.shape[0]
    device = state_indices.device
    valid_rows = valid_rows.to(torch.bool)
    num_valid = valid_rows.sum(dim=0, keepdim=True, dtype=torch.int32)

    real_positions = valid_rows.to(torch.int64).cumsum(dim=0) - 1
    padding_positions = (
        num_valid.to(torch.int64) + (~valid_rows).to(torch.int64).cumsum(dim=0) - 1
    )
    destinations = torch.where(valid_rows, real_positions, padding_positions)
    source_rows = torch.arange(num_rows, dtype=torch.int64, device=device)
    order = torch.empty_like(source_rows)
    order.scatter_(dim=0, index=destinations, src=source_rows)

    if out_order is not None and out_order.numel() >= num_rows:
        out_order[:num_rows].copy_(order)
        order = out_order[:num_rows]
    if out_num_valid is not None and out_num_valid.numel() >= 1:
        out_num_valid[:1].copy_(num_valid)
        num_valid = out_num_valid[:1]
    return state_indices.index_select(0, order), order, num_valid


@functools.lru_cache(maxsize=8)
def _get_kda_masked_gemm(tile_m: int, tile_n: int, stages: int, warps: int):
    from mate.jit.gemm.deep_gemm.gemm import (
        GEMM_TYPE_M_GROUPED_MASKED,
        GemmJitConfig,
        get_deep_gemm_gemm_module,
    )

    dispatch, module = get_deep_gemm_gemm_module(
        kind="bf16",
        gemm_type=GEMM_TYPE_M_GROUPED_MASKED,
        config=GemmJitConfig(tile_m, tile_n, stages, warps),
    )
    return module.get_function(dispatch)


def masked_bf16_linear_for_dp(
    x: torch.Tensor,
    weight: torch.Tensor,
    num_valid: torch.Tensor,
    *,
    zero_tail: bool = False,
) -> torch.Tensor:
    """Single-group BF16 GEMM with a device-side valid row count."""
    num_rows = x.shape[0]
    output = torch.empty((1, num_rows, weight.shape[0]), dtype=x.dtype, device=x.device)
    if num_rows <= 128:
        if num_rows <= 32:
            config = (32, 64, 4, 1)
        elif num_rows <= 64:
            config = (64, 64, 4, 1)
        else:
            config = (64, 128, 4, 1)
        tile_m, tile_n, stages, warps = config
        kernel = _get_kda_masked_gemm(tile_m, tile_n, stages, warps)
        kernel(
            x.unsqueeze(0),
            weight.unsqueeze(0),
            output,
            num_valid,
            tile_m,
            0,
        )
    else:
        import deep_gemm

        deep_gemm.bf16_m_grouped_gemm_nt_masked(
            x.unsqueeze(0), weight.unsqueeze(0), output, num_valid, num_rows
        )
    output = output[0]
    if zero_tail:
        valid = torch.arange(num_rows, dtype=torch.int32, device=x.device)
        valid = valid < num_valid[0]
        output = torch.where(valid.unsqueeze(1), output, output.new_zeros(()))
    return output


def _globalize_gdn_prefill_metadata_for_dp(
    meta: GDNAttentionMetadata,
    *,
    group,
    request_windows: int,
    shared_state_pool: bool,
    pool_num_blocks: int,
    max_tokens: int,
    plan=None,
) -> GDNAttentionMetadata:
    is_dp_dummy = getattr(meta, "kda_dp_dummy", False)
    assert is_dp_dummy or (meta.num_spec_decodes == 0 and meta.num_decodes == 0), (
        "KDA across DP ranks does not support mixed prefill and decode: "
        f"prefills={meta.num_prefills}, decodes={meta.num_decodes}, "
        f"spec_decodes={meta.num_spec_decodes}"
    )
    assert meta.checkpoint is None, (
        "KDA across DP ranks does not support aligned prefill checkpoints"
    )
    k = request_windows
    rows = max_tokens
    if is_dp_dummy:
        reference = meta.non_spec_state_indices_tensor
        if reference is None:
            reference = meta.spec_state_indices_tensor
        assert reference is not None
        device = reference.device
        local_state = torch.empty(0, dtype=torch.int32, device=device)
        local_qsl = torch.zeros(1, dtype=torch.int32, device=device)
        local_has_initial = torch.empty(0, dtype=torch.bool, device=device)
    else:
        local_state = meta.non_spec_state_indices_tensor
        local_qsl = meta.non_spec_query_start_loc
        local_has_initial = meta.has_initial_state
        assert local_state is not None
        assert local_qsl is not None
        if local_has_initial is None:
            local_has_initial = torch.empty(
                0, dtype=torch.bool, device=local_state.device
            )
        device = local_state.device

    def pad_rows(t: torch.Tensor, fill: int) -> torch.Tensor:
        n = t.shape[0]
        assert n <= rows
        if n == rows:
            return t
        extra = t.new_full((rows - n,) + t.shape[1:], fill)
        return torch.cat([t, extra], 0)

    local_query_lens = (local_qsl[1:] - local_qsl[:-1]).to(torch.int32)
    gathered_state = (
        plan.all_gather(pad_rows(local_state, 0).contiguous().to(torch.int32))
        if plan is not None
        else group.all_gather(pad_rows(local_state, 0).contiguous().to(torch.int32), 0)
    )
    gathered_query_lens = group.all_gather(
        pad_rows(local_query_lens, 0).contiguous(), 0
    )
    gathered_has_initial = group.all_gather(
        pad_rows(local_has_initial, 0).contiguous().to(torch.int32), 0
    )
    if not shared_state_pool:
        assert pool_num_blocks % k == 0
        rank_offsets = torch.arange(k, dtype=torch.int32, device=device)
        rank_offsets = rank_offsets.repeat_interleave(rows) * (pool_num_blocks // k)
        gathered_state = gathered_state + rank_offsets

    local_token_totals = gathered_query_lens.view(k, rows).sum(dim=1)
    request_mask = gathered_query_lens > 0
    gathered_state = gathered_state[request_mask]
    gathered_has_initial = gathered_has_initial[request_mask].to(torch.bool)
    gathered_query_lens = gathered_query_lens[request_mask]
    num_reqs = int(request_mask.sum().item())
    num_tokens = int(gathered_query_lens.sum().item())
    query_start_loc = gathered_query_lens.new_zeros(num_reqs + 1)
    torch.cumsum(gathered_query_lens, dim=0, out=query_start_loc[1:])
    query_start_loc_cpu = query_start_loc.cpu()

    num_global = rows * k
    token_positions = torch.arange(rows, dtype=torch.int32, device=device).repeat(k)
    valid_tokens = token_positions < local_token_totals.repeat_interleave(rows)
    token_rows = torch.arange(num_global, dtype=torch.int64, device=device)
    _, non_spec_token_indx, num_valid = compact_gdn_rows_for_dp(
        token_rows, valid_tokens
    )

    from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata

    nums_dict, batch_ptr, token_chunk_offset_ptr = compute_causal_conv1d_metadata(
        query_start_loc_cpu, device=device
    )
    global_meta = replace(
        meta,
        num_prefills=num_reqs,
        num_prefill_tokens=num_tokens,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=num_global,
        has_initial_state=gathered_has_initial,
        checkpoint=None,
        chunk_indices=None,
        chunk_offsets=None,
        prefill_query_start_loc=query_start_loc,
        prefill_state_indices=gathered_state,
        prefill_has_initial_state=gathered_has_initial,
        aiter_prefill_metadata=None,
        spec_query_start_loc=None,
        non_spec_query_start_loc=query_start_loc,
        non_spec_query_start_loc_cpu=query_start_loc_cpu,
        spec_state_indices_tensor=None,
        non_spec_state_indices_tensor=gathered_state,
        spec_sequence_masks=None,
        spec_sequence_masks_cpu=None,
        spec_token_indx=None,
        non_spec_token_indx=non_spec_token_indx,
        num_accepted_tokens=None,
        uniform_spec_sequence_length=None,
        nums_dict=nums_dict,
        batch_ptr=batch_ptr,
        token_chunk_offset_ptr=token_chunk_offset_ptr,
    )
    global_meta._kda_dp_order = None
    global_meta._kda_dp_num_valid = None
    global_meta._kda_dp_prefill_num_valid = num_valid
    meta._kda_global_metadata = global_meta
    return global_meta


def _globalize_gdn_spec_metadata_for_dp(
    meta: GDNAttentionMetadata,
    *,
    group,
    request_windows: int,
    shared_state_pool: bool,
    pool_num_blocks: int,
    max_tokens: int,
    max_requests: int,
    out_state_indices: torch.Tensor | None,
    out_qsl: torch.Tensor | None,
    out_spec_token_indx: torch.Tensor | None,
    out_spec_accepted: torch.Tensor | None,
    out_num_valid: torch.Tensor | None,
    plan=None,
) -> GDNAttentionMetadata:
    assert meta.num_decodes == 0, (
        "KDA across DP ranks does not support mixed spec and non-spec decode"
    )
    k = request_windows
    rows = max_requests
    num_local = meta.num_spec_decodes
    assert meta.spec_state_indices_tensor is not None
    assert meta.spec_query_start_loc is not None
    assert meta.num_accepted_tokens is not None
    local_state = meta.spec_state_indices_tensor[:num_local]
    local_qsl = meta.spec_query_start_loc
    local_accepted = meta.num_accepted_tokens[:num_local]
    num_slots = local_state.shape[1]
    device = local_state.device

    def pad_rows(t: torch.Tensor, fill: int) -> torch.Tensor:
        n = t.shape[0]
        assert n <= rows
        if n == rows:
            return t
        extra = t.new_full((rows - n,) + t.shape[1:], fill)
        return torch.cat([t, extra], 0)

    if plan is None:
        gathered_state = group.all_gather(
            pad_rows(local_state, 0).contiguous().to(torch.int32), 0
        )

    shared = plan.spec_common if plan is not None else None
    if shared is None:
        local_query_lens = (local_qsl[1 : num_local + 1] - local_qsl[:num_local]).to(
            torch.int32
        )
        gathered_query_lens = (
            plan.all_gather(pad_rows(local_query_lens, 0).contiguous())
            if plan is not None
            else group.all_gather(pad_rows(local_query_lens, 0).contiguous(), 0)
        )
        gathered_accepted = (
            plan.all_gather(pad_rows(local_accepted, 0).contiguous().to(torch.int32))
            if plan is not None
            else group.all_gather(
                pad_rows(local_accepted, 0).contiguous().to(torch.int32), 0
            )
        )
    else:
        gathered_accepted = shared.accepted
        query_start_loc = shared.query_start_loc
        spec_token_indx = shared.spec_token_indx
        num_valid = shared.num_valid

    if plan is not None:
        gathered_state = plan.all_gather(
            pad_rows(local_state, 0).contiguous().to(torch.int32)
        )
    if not shared_state_pool:
        assert pool_num_blocks % k == 0
        rank_offsets = torch.arange(k, dtype=torch.int32, device=device)
        rank_offsets = rank_offsets.repeat_interleave(rows) * (pool_num_blocks // k)
        gathered_state = gathered_state + rank_offsets.unsqueeze(1)

    num_global = rows * k
    num_global_tokens = max_tokens * k
    if shared is None:
        query_start_loc = gathered_query_lens.new_zeros(num_global + 1)
        torch.cumsum(gathered_query_lens, dim=0, out=query_start_loc[1:])

        token_totals = gathered_query_lens.view(k, rows).sum(dim=1)
        token_positions = torch.arange(
            max_tokens, dtype=torch.int32, device=device
        ).repeat(k)
        valid_tokens = token_positions < token_totals.repeat_interleave(max_tokens)
        token_rows = torch.arange(num_global_tokens, dtype=torch.int64, device=device)
        _, spec_token_indx, num_valid = compact_gdn_rows_for_dp(
            token_rows,
            valid_tokens,
            out_order=out_spec_token_indx,
            out_num_valid=out_num_valid,
        )

    if out_state_indices is not None and out_state_indices.numel() >= (
        num_global * num_slots
    ):
        state_buffer = out_state_indices[: num_global * num_slots].view(
            num_global, num_slots
        )
        state_buffer.copy_(gathered_state)
        gathered_state = state_buffer
    if out_qsl is not None and out_qsl.numel() >= num_global + 1:
        out_qsl[: num_global + 1].copy_(query_start_loc)
        query_start_loc = out_qsl[: num_global + 1]
    if shared is not None:
        if out_spec_token_indx is not None and out_spec_token_indx.numel() >= (
            num_global_tokens
        ):
            out_spec_token_indx[:num_global_tokens].copy_(spec_token_indx)
            spec_token_indx = out_spec_token_indx[:num_global_tokens]
        if out_num_valid is not None and out_num_valid.numel() >= 1:
            out_num_valid[:1].copy_(num_valid)
            num_valid = out_num_valid[:1]
    if out_spec_accepted is not None and out_spec_accepted.numel() >= num_global:
        out_spec_accepted[:num_global].copy_(gathered_accepted)
        gathered_accepted = out_spec_accepted[:num_global]

    extra_fields = {}
    recoverssm_commit = getattr(meta, "recoverssm_commit", None)
    if recoverssm_commit is not None:
        assert recoverssm_commit.align is None, (
            "KDA across DP ranks does not support RecoverSSM align metadata"
        )
        extra_fields["recoverssm_commit"] = replace(
            recoverssm_commit,
            state_indices=gathered_state,
            query_start_loc=query_start_loc,
            request_indices=None,
            align=None,
        )
    if hasattr(meta, "spec_token_start"):
        extra_fields["spec_token_start"] = None
        extra_fields["non_spec_token_start"] = None

    global_meta = replace(
        meta,
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=num_global,
        num_spec_decode_tokens=num_global_tokens,
        num_actual_tokens=num_global_tokens,
        has_initial_state=None,
        checkpoint=None,
        spec_query_start_loc=query_start_loc,
        non_spec_query_start_loc=None,
        non_spec_query_start_loc_cpu=None,
        spec_state_indices_tensor=gathered_state,
        non_spec_state_indices_tensor=None,
        spec_sequence_masks=None,
        spec_sequence_masks_cpu=None,
        spec_token_indx=spec_token_indx,
        non_spec_token_indx=None,
        num_accepted_tokens=gathered_accepted,
        uniform_spec_sequence_length=None,
        **extra_fields,
    )
    global_meta._kda_dp_order = None
    global_meta._kda_dp_num_valid = None
    global_meta._kda_dp_spec_num_valid = num_valid
    global_meta._kda_dp_spec_group = group
    global_meta._kda_dp_spec_rows = rows
    recoverssm_context = getattr(global_meta, "recoverssm_context", None)
    if recoverssm_context is not None and recoverssm_commit is not None:
        recoverssm_context._dp_last_commit = (
            global_meta.recoverssm_commit,
            group,
            rows,
            num_global,
        )
    meta._kda_global_metadata = global_meta
    return global_meta


def globalize_gdn_metadata_for_dp(
    meta: GDNAttentionMetadata,
    *,
    group,
    request_windows: int,
    shared_state_pool: bool,
    pool_num_blocks: int,
    max_tokens: int,
    max_requests: int | None = None,
    out_state_indices: torch.Tensor | None = None,
    out_qsl: torch.Tensor | None = None,
    out_order: torch.Tensor | None = None,
    out_num_valid: torch.Tensor | None = None,
    out_spec_token_indx: torch.Tensor | None = None,
    out_spec_accepted: torch.Tensor | None = None,
    out_spec_num_valid: torch.Tensor | None = None,
    global_mode: int = 0,
    plan=None,
) -> GDNAttentionMetadata:
    """Expand rank-local decode metadata to the gathered KDA batch.

    This runs outside graph capture (the GDN metadata builder calls it), so the
    KDA forward pass only reads the resulting tensors. Every rank contributes
    exactly ``max_tokens`` request slots; slots past a rank's own token count
    carry that rank's null block and their outputs are dropped on the way back.
    """
    cached = getattr(meta, "_kda_global_metadata", None)
    if cached is not None:
        return cached

    if global_mode == 2:
        return _globalize_gdn_prefill_metadata_for_dp(
            meta,
            group=group,
            request_windows=request_windows,
            shared_state_pool=shared_state_pool,
            pool_num_blocks=pool_num_blocks,
            max_tokens=max_tokens,
            plan=plan,
        )

    assert meta.num_prefills == 0, (
        "KDA across DP ranks supports decode batches only; "
        "prefill token metadata is not globalised"
    )
    if meta.num_spec_decodes > 0:
        assert out_order is None, (
            "KDA DP row compaction is not supported with speculative decoding"
        )
        assert max_requests is not None
        return _globalize_gdn_spec_metadata_for_dp(
            meta,
            group=group,
            request_windows=request_windows,
            shared_state_pool=shared_state_pool,
            pool_num_blocks=pool_num_blocks,
            max_tokens=max_tokens,
            max_requests=max_requests,
            out_state_indices=out_state_indices,
            out_qsl=out_qsl,
            out_spec_token_indx=out_spec_token_indx,
            out_spec_accepted=out_spec_accepted,
            out_num_valid=out_spec_num_valid,
            plan=plan,
        )

    k = request_windows
    assert meta.non_spec_state_indices_tensor is not None
    device = meta.non_spec_state_indices_tensor.device
    rows = max_tokens

    def pad_rows(t: torch.Tensor, fill: int) -> torch.Tensor:
        n = t.shape[0]
        if n >= rows:
            return t[:rows]
        extra = t.new_full((rows - n,) + t.shape[1:], fill)
        return torch.cat([t, extra], 0)

    local_state = meta.non_spec_state_indices_tensor
    assert local_state.shape[0] <= rows, (
        f"request axis ({local_state.shape[0]}) exceeds the gathered token "
        f"stride ({rows}); a decode batch has at most one request per token"
    )

    # Window-local pools shift each rank's rows into its own window so padding
    # lands on that rank's null block. A globally addressed pool hands out
    # global ids already, so no offset applies and every rank's padding uses
    # the pool-wide null block 0.
    gathered_state = (
        plan.all_gather(pad_rows(local_state, 0).contiguous().to(torch.int32))
        if plan is not None
        else group.all_gather(pad_rows(local_state, 0).contiguous().to(torch.int32), 0)
    )
    if shared_state_pool:
        state_indices = gathered_state
    else:
        assert pool_num_blocks % k == 0, (
            f"mamba state pool holds {pool_num_blocks} blocks, which does not "
            f"split into {k} request windows; the pool was sized without "
            "kda_layer_tp"
        )
        state_indices = gathered_state + torch.arange(
            k, dtype=torch.int32, device=device
        ).repeat_interleave(rows) * (pool_num_blocks // k)

    order = num_valid = None
    if out_order is not None:
        state_indices, order, num_valid = compact_gdn_rows_for_dp(
            state_indices,
            gathered_state > 0,
            out_order=out_order,
            out_num_valid=out_num_valid,
        )

    qsl = torch.arange(rows * k + 1, dtype=torch.int32, device=device)
    if out_state_indices is not None and out_state_indices.numel() >= rows * k:
        # FULL graph: write into the persistent buffer the graph captured.
        out_state_indices[: rows * k].copy_(state_indices)
        state_indices = out_state_indices[: rows * k]
    if out_qsl is not None and out_qsl.numel() >= rows * k + 1:
        out_qsl[: rows * k + 1].copy_(qsl)
        qsl = out_qsl[: rows * k + 1]
    global_meta = GDNAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=rows * k,
        num_decode_tokens=rows * k,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=rows * k,
        has_initial_state=None,
        spec_query_start_loc=None,
        non_spec_query_start_loc=qsl,
        spec_state_indices_tensor=None,
        non_spec_state_indices_tensor=state_indices,
        spec_sequence_masks=None,
        spec_token_indx=None,
        non_spec_token_indx=None,
        num_accepted_tokens=None,
        uniform_spec_sequence_length=None,
    )
    global_meta._kda_dp_order = order
    global_meta._kda_dp_num_valid = num_valid
    meta._kda_global_metadata = global_meta
    return global_meta


def globalize_gdn_metadata_for_dp_config(
    vllm_config,
    kv_cache_spec,
    meta,
    out_state_indices: torch.Tensor | None = None,
    out_qsl: torch.Tensor | None = None,
    out_order: torch.Tensor | None = None,
    out_num_valid: torch.Tensor | None = None,
    out_spec_token_indx: torch.Tensor | None = None,
    out_spec_accepted: torch.Tensor | None = None,
    out_spec_num_valid: torch.Tensor | None = None,
):
    """Config-driven entry point used by the GDN metadata builder."""
    from vllm.config.kda_parallel import (
        get_kda_request_windows,
        kda_mamba_shared_pool,
    )
    from vllm.models.glm5next.common.parallel import get_kda_parallel_spec
    from vllm.utils.math_utils import cdiv

    group = get_kda_parallel_spec(vllm_config).group
    k = get_kda_request_windows(vllm_config)
    shared = kda_mamba_shared_pool(vllm_config)
    plan = getattr(meta, "kda_dp_plan", None) or getattr(meta, "_kda_dp_plan", None)
    if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
        kv_cache_spec = kv_cache_spec.first_spec
    blocks_per_request = cdiv(
        kv_cache_spec.max_memory_usage_bytes(vllm_config),
        kv_cache_spec.page_size_bytes,
    )
    pool_num_blocks = vllm_config.scheduler_config.max_num_seqs * blocks_per_request + 1

    fc_available = is_forward_context_available()
    cache = (
        get_forward_context().additional_kwargs.setdefault("kda_group_max", {})
        if fc_available
        else {}
    )
    if plan is not None:
        max_tokens = plan.max_tokens
        cache["tokens"] = max_tokens
    else:
        max_tokens = cache.get("tokens")
        if max_tokens is None:
            dp_metadata = get_forward_context().dp_metadata if fc_available else None
            if (
                dp_metadata is not None
                and group.world_size == dp_metadata.num_tokens_across_dp_cpu.numel()
            ):
                max_tokens = int(dp_metadata.num_tokens_across_dp_cpu.max())
            else:
                t = torch.tensor([meta.num_actual_tokens], dtype=torch.int32)
                torch.distributed.all_reduce(
                    t, op=torch.distributed.ReduceOp.MAX, group=group.cpu_group
                )
                max_tokens = int(t[0])
            cache["tokens"] = max_tokens

    if plan is not None:
        global_mode = plan.global_mode
        cache["mode"] = global_mode
    else:
        global_mode = cache.get("mode")
        if global_mode is None:
            local_mode = 2 if meta.num_prefills > 0 else int(meta.num_spec_decodes > 0)
            mode = torch.tensor([local_mode], dtype=torch.int32)
            torch.distributed.all_reduce(
                mode, op=torch.distributed.ReduceOp.MAX, group=group.cpu_group
            )
            global_mode = int(mode[0])
            cache["mode"] = global_mode

    max_requests = None
    if global_mode == 1:
        if plan is not None:
            max_requests = plan.max_requests
            cache["spec_requests"] = max_requests
        else:
            max_requests = cache.get("spec_requests")
            if max_requests is None:
                t = torch.tensor([meta.num_spec_decodes], dtype=torch.int32)
                torch.distributed.all_reduce(
                    t, op=torch.distributed.ReduceOp.MAX, group=group.cpu_group
                )
                max_requests = int(t[0])
                cache["spec_requests"] = max_requests

    global_meta = globalize_gdn_metadata_for_dp(
        meta,
        group=group,
        request_windows=k,
        shared_state_pool=shared,
        pool_num_blocks=pool_num_blocks,
        max_tokens=max_tokens,
        max_requests=max_requests,
        out_state_indices=out_state_indices,
        out_qsl=out_qsl,
        out_order=out_order,
        out_num_valid=out_num_valid,
        out_spec_token_indx=out_spec_token_indx,
        out_spec_accepted=out_spec_accepted,
        out_spec_num_valid=out_spec_num_valid,
        global_mode=global_mode,
        plan=plan,
    )
    if plan is not None:
        plan.track_consumers(
            global_meta.spec_state_indices_tensor,
            global_meta.non_spec_state_indices_tensor,
            global_meta.spec_query_start_loc,
            global_meta.non_spec_query_start_loc,
            global_meta.spec_token_indx,
            global_meta.non_spec_token_indx,
            global_meta.num_accepted_tokens,
            getattr(global_meta, "_kda_dp_order", None),
            getattr(global_meta, "_kda_dp_num_valid", None),
            getattr(global_meta, "_kda_dp_spec_num_valid", None),
        )
        global_meta._kda_dp_plan = plan
        if global_mode == 1:
            accepted_semantics = (
                "recoverssm"
                if getattr(global_meta, "recoverssm_context", None) is not None
                else "forward_offsets"
            )
            if plan.spec_accepted_semantics is None:
                plan.spec_accepted_semantics = accepted_semantics
            elif plan.spec_accepted_semantics != accepted_semantics:
                raise RuntimeError(
                    "KDA DP plan cannot share accepted metadata across builders "
                    "with different RecoverSSM semantics"
                )
        if plan.spec_common is None and global_mode == 1:
            from vllm.models.glm5next.common.kda_dp_plan import KdaDPSpecCommon

            plan.spec_common = KdaDPSpecCommon(
                query_start_loc=global_meta.spec_query_start_loc,
                spec_token_indx=global_meta.spec_token_indx,
                num_valid=global_meta._kda_dp_spec_num_valid,
                accepted=global_meta.num_accepted_tokens,
            )
    return global_meta


class Glm5NextLinearAttention(GatedDeltaNetAttention):
    head_dim: int
    num_heads: int
    conv_size: int

    def get_attn_backend(self) -> type[AttentionBackend]:
        if self.cache_config.use_kda_recoverssm:
            from .kda_metadata import Glm5NextRecoverSSMAttentionBackend

            return Glm5NextRecoverSSMAttentionBackend
        return super().get_attn_backend()

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> MambaSpec:
        spec = super().get_kv_cache_spec(vllm_config)
        assert isinstance(spec, MambaSpec)
        alignment = kda_prefill_checkpoint_alignment(self.kda_prefill_backend)
        return replace(
            spec,
            num_prefill_checkpoint_blocks=int(alignment is not None),
            prefill_checkpoint_alignment=alignment,
        )

    @property
    def tp_over_pcp(self) -> bool:
        return self.kda_parallel.token_layout == "pcp"

    @property
    def tp_over_dp(self) -> bool:
        return self.kda_parallel.token_layout == "dp"

    def get_state_dtype(
        self,
    ) -> tuple[torch.dtype, ...]:
        if self.model_config is None or self.cache_config is None:
            raise ValueError("model_config and cache_config must be set")
        base_dtypes = MambaStateDtypeCalculator.kda_state_dtype(
            self.model_config.dtype,
            self.cache_config.mamba_cache_dtype,
            self.cache_config.mamba_ssm_cache_dtype,
        )

        if self.cache_config.use_kda_recoverssm:
            return MambaStateDtypeCalculator.append_kda_recoverssm_record(base_dtypes)
        return base_dtypes

    def get_state_shape(
        self,
    ) -> tuple[tuple[int, ...], ...]:
        # conv_state width must include num_spec so the spec-decode conv update
        # (causal_conv1d_update with num_accepted_tokens + max_query_len) can
        # slide the window across the draft-verify tokens without reading past
        # the allocated width. Matches qwen_gdn_linear_attn.get_state_shape.
        base_shapes = MambaStateShapeCalculator.kda_state_shape(
            self.kda_layer_tp,
            self.num_heads,
            self.head_dim,
            conv_kernel_size=self.conv_size,
            num_spec=self.num_spec,
        )

        if self.cache_config.use_kda_recoverssm:
            return MambaStateShapeCalculator.append_kda_recoverssm_record(
                base_shapes,
                self.num_heads,
                self.head_dim,
                self.kda_layer_tp,
                1 + self.num_spec,
            )
        return base_shapes

    def __init__(
        self,
        config: Glm5NextTextConfig,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        # KDA projections remain BF16 because fp8 checkpoints omit their scales.
        saved_quant_config = vllm_config.quant_config
        try:
            vllm_config.quant_config = None
            super().__init__(config, vllm_config, prefix)
        finally:
            vllm_config.quant_config = saved_quant_config

        self.use_recoverssm = self.cache_config.use_kda_recoverssm
        self.kda_parallel = get_kda_parallel_spec(vllm_config)
        self.kda_layer_tp = self.kda_parallel.kda_layer_tp
        self.kda_layer_rank = self.kda_parallel.kda_layer_rank
        self.request_windows = self.kda_parallel.request_windows
        self.shared_state_pool = kda_mamba_shared_pool(vllm_config)
        self.dp_compact_decode = kda_dp_compact_decode(vllm_config)
        if self.tp_over_dp:
            # Each rank holds the head slice of every request in the group, so
            # the Mamba state pool must be one window per rank wide. That only
            # exists with MUSA's dedicated Mamba pool: a shared pool sizes the
            # state tensor from the attention block count instead, and the
            # window offset would run off the end of it.
            from vllm.v1.core.kv_cache_utils import (
                musa_mamba_separate_pool_enabled,
            )

            assert musa_mamba_separate_pool_enabled(), (
                "kda_layer_tp > tensor_parallel_size requires the dedicated "
                "MUSA Mamba state pool."
            )
        self.head_dim = config.linear_head_dim
        self.num_heads = config.linear_num_heads
        self.conv_size = config.linear_conv_kernel_dim
        assert self.num_heads % self.kda_layer_tp == 0
        self.local_num_heads = divide(self.num_heads, self.kda_layer_tp)

        projection_size = self.head_dim * self.num_heads
        self.local_projection_size = divide(projection_size, self.kda_layer_tp)

        _, recurrent_state_shape, *_ = self.get_state_shape()
        _, recurrent_state_dtype, *_ = self.get_state_dtype()
        # The scatter covers every gathered request, so its index tensor is
        # request_windows times the local batch.
        scatter_states.register_warmup(
            state_shape=recurrent_state_shape,
            state_dtype=recurrent_state_dtype,
            indices_dtype=torch.int32,
            max_num_tokens=(
                vllm_config.scheduler_config.max_num_seqs * self.request_windows
            ),
        )

        # Merge q, k, v, b, f_a, g_a projections into one GEMM (6→1 launches).
        # Order matches checkpoint's fused_qkvbfg_a_proj convention.
        # Shards 4 (f_a) and 5 (g_a) are replicated across TP ranks.
        self.in_proj_qkvbfg_a = _Glm5NextMergedColumnParallelLinear(
            self.hidden_size,
            [
                projection_size,  # q (shard 0)
                projection_size,  # k (shard 1)
                projection_size,  # v (shard 2)
                self.num_heads,  # b (shard 3)
                self.head_dim,  # f_a (shard 4, replicated)
                self.head_dim,  # g_a (shard 5, replicated)
            ],
            replicated_shard_ids=(4, 5),
            tp_size=self.kda_layer_tp,
            tp_rank=self.kda_layer_rank,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.in_proj_qkvbfg_a",
        )

        self.f_b_proj = ColumnParallelLinear(
            self.head_dim,
            projection_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.f_b_proj",
            tp_rank=self.kda_layer_rank,
            tp_size=self.kda_layer_tp,
        )
        self.dt_bias = nn.Parameter(
            torch.empty(divide(projection_size, self.kda_layer_tp), dtype=torch.float32)
        )

        set_weight_attrs(
            self.dt_bias,
            {"weight_loader": sharded_weight_loader(0, tp_rank=self.kda_layer_rank)},
        )

        self.q_conv1d = ColumnParallelLinear(
            input_size=self.conv_size,
            output_size=projection_size,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.q_conv1d",
            tp_rank=self.kda_layer_rank,
            tp_size=self.kda_layer_tp,
        )
        self.k_conv1d = ColumnParallelLinear(
            input_size=self.conv_size,
            output_size=projection_size,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.k_conv1d",
            tp_rank=self.kda_layer_rank,
            tp_size=self.kda_layer_tp,
        )
        self.v_conv1d = ColumnParallelLinear(
            input_size=self.conv_size,
            output_size=projection_size,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.v_conv1d",
            tp_rank=self.kda_layer_rank,
            tp_size=self.kda_layer_tp,
        )
        # unsqueeze to fit conv1d weights shape into the linear weights shape.
        # Can't do this in `weight_loader` since it already exists in
        # `ColumnParallelLinear` and `set_weight_attrs`
        # doesn't allow to override it
        self.q_conv1d.weight.data = self.q_conv1d.weight.data.unsqueeze(1)
        self.k_conv1d.weight.data = self.k_conv1d.weight.data.unsqueeze(1)
        self.v_conv1d.weight.data = self.v_conv1d.weight.data.unsqueeze(1)
        # Lazily-built merged q|k|v conv weight (built on first forward, after
        # weights are loaded). See _forward.
        self._merged_conv_weight: torch.Tensor | None = None

        self.A_log = nn.Parameter(
            torch.empty(1, 1, self.local_num_heads, 1, dtype=torch.float32)
        )
        set_weight_attrs(
            self.A_log,
            {"weight_loader": sharded_weight_loader(2, tp_rank=self.kda_layer_rank)},
        )

        self.g_b_proj = ColumnParallelLinear(
            self.head_dim,
            projection_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.g_b_proj",
            tp_rank=self.kda_layer_rank,
            tp_size=self.kda_layer_tp,
        )
        self.o_norm = FusedRMSNormGated(self.head_dim, activation="sigmoid")
        self.o_proj = RowParallelLinear(
            projection_size,
            self.hidden_size,
            bias=False,
            quant_config=self.quant_config,
            reduce_results=self.kda_parallel.token_layout == "tp",
            prefix=f"{prefix}.o_proj",
            tp_rank=self.kda_layer_rank,
            tp_size=self.kda_layer_tp,
            parallel_group=self.kda_parallel.group,
        )

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

        # Checkpoints store A_log as 1-D; the model parameter is 4-D.
        def _a_log_weight_loader(param, loaded_weight):
            if loaded_weight.dim() == 1:
                loaded_weight = loaded_weight.view([1, 1, -1, 1])
            loader = sharded_weight_loader(2, tp_rank=self.kda_layer_rank)
            return loader(param, loaded_weight)

        self.A_log.weight_loader = _a_log_weight_loader

        # GLM-5.3-Flash uses a bounded sigmoid gate instead of the default
        # unbounded softplus gate.
        self.kda_safe_gate = True
        self.kda_lower_bound = config.linear_lower_bound
        # Process-global conv-state layout, resolved once here instead of on
        # every _forward call (it reads an env-derived flag each time).
        self._conv_state_dim_first = is_conv_state_dim_first()

        additional_config = vllm_config.additional_config
        self.kda_prefill_backend = _resolve_kda_prefill_backend(
            additional_config.get("kda_prefill_backend", "auto")
            if isinstance(additional_config, dict)
            else "auto",
            self.head_dim,
            vllm_config.model_config.dtype,
            self.kda_lower_bound,
        )
        self._flashkda_buffer_specs: (
            tuple[tuple[tuple[int, ...], torch.dtype], ...] | None
        ) = None
        self._checkpoint_exporter: FlashKDAPrefillCheckpointExporter | None = None
        if self.kda_prefill_backend == "flashkda":
            if current_platform.device_type == "musa":
                import flash_kda  # noqa: F401

                flashkda_ops = flash_kda
            else:
                import vllm._flashkda_C  # noqa: F401

                flashkda_ops = torch.ops._flashkda_C

            max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
            max_seqs = vllm_config.scheduler_config.max_num_seqs
            workspace_size = flashkda_ops.get_workspace_size(
                max_tokens, self.local_num_heads, max_seqs
            )
            self._flashkda_buffer_specs = (
                (
                    (max_seqs, self.local_num_heads, self.head_dim, self.head_dim),
                    self.get_state_dtype()[1],
                ),
                (
                    (max_seqs, self.local_num_heads, self.head_dim, self.head_dim),
                    self.get_state_dtype()[1],
                ),
                ((workspace_size,), torch.uint8),
                # Output buffer for steps that also carry spec-decode tokens:
                # the non-spec tokens are then scattered by non_spec_token_indx.
                (
                    (1, max_tokens, self.local_num_heads, self.head_dim),
                    vllm_config.model_config.dtype,
                ),
            )
            self._checkpoint_exporter = FlashKDAPrefillCheckpointExporter(
                state_len=self.conv_size - 1
            )

    def _flashkda_prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        cu_seqlens: torch.Tensor,
        out: torch.Tensor | None,
        checkpoint: MambaPrefillCheckpointMetadata | None = None,
        raw_qkv: torch.Tensor | None = None,
        conv_state: torch.Tensor | None = None,
        recurrent_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fused KDA chunked prefill (FlashKDA). Takes the raw gate logits ``g``
        and raw ``beta`` logits, l2-normalizes q/k in-kernel and applies the
        bounded gate ``lower_bound * sigmoid(exp(A_log) * (g + dt_bias))``,
        matching ``chunk_kda_with_fused_gate(..., safe_gate=True)``. Writes the
        attention output into ``out`` (a workspace buffer when ``None``) and
        returns ``(out, final_state)``."""
        assert self._flashkda_buffer_specs is not None
        final_state, checkpoint_state, workspace, workspace_out = (
            current_workspace_manager().get_simultaneous(*self._flashkda_buffer_specs)
        )
        final_state = final_state[: initial_state.shape[0]]
        checkpoint_state = (
            checkpoint_state[: initial_state.shape[0]]
            if checkpoint is not None
            else None
        )
        if out is None:
            out = workspace_out[:, : q.shape[1]]
        # FlashKDA hardcodes dense q/k/v/g strides; beta may be row-strided.
        if current_platform.device_type == "musa":
            import flash_kda as flashkda_ops
        else:
            flashkda_ops = torch.ops._flashkda_C
        flashkda_ops.fwd(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            g.contiguous(),
            beta,
            self.head_dim**-0.5,
            out,
            workspace=workspace,
            A_log=self.A_log.view(-1),
            dt_bias=self.dt_bias.view(-1, self.head_dim),
            lower_bound=self.kda_lower_bound,
            initial_state=initial_state.contiguous(),
            final_state=final_state,
            cu_seqlens=cu_seqlens.contiguous(),
            checkpoint_state=checkpoint_state,
            checkpoint_offsets=checkpoint.checkpoint_offsets.contiguous()
            if checkpoint is not None
            else None,
        )
        if checkpoint is not None:
            assert self._checkpoint_exporter is not None
            assert raw_qkv is not None and conv_state is not None
            assert recurrent_state is not None
            self._checkpoint_exporter.export(
                checkpoint,
                raw_qkv=raw_qkv,
                conv_state=conv_state,
                recurrent_checkpoint=checkpoint_state,
                recurrent_state=recurrent_state,
                cu_seqlens=cu_seqlens,
            )
        return out, final_state

    def _dp_compact_metadata(
        self,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        metadata = get_forward_context().attn_metadata
        if not isinstance(metadata, dict):
            return None, None
        layer_metadata = metadata.get(self.prefix)
        global_meta = getattr(layer_metadata, "_kda_global_metadata", None)
        if global_meta is None:
            return None, None
        return global_meta._kda_dp_order, global_meta._kda_dp_num_valid

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        # With ``kda_layer_tp`` the KDA layers shard over several DP ranks, so
        # each rank computes its own head slice for *every* request. Gather the
        # rank-local tokens in rank order; the reduce_scatter at the end hands
        # back the tokens this rank started with.
        gather = self.tp_over_dp
        num_local_tokens = hidden_states.size(0)
        dp_order = dp_num_valid = None
        if gather:
            # Ranks can hold different token counts on an eager step, and the
            # collective needs equal shapes. Pad up to the group maximum; the
            # extra rows still form one-token sequences, but they carry the
            # null state block and their outputs are dropped on the way back.
            max_tokens = self._group_max("tokens", num_local_tokens)
            if max_tokens > num_local_tokens:
                hidden_states = F.pad(
                    hidden_states, (0, 0, 0, max_tokens - num_local_tokens)
                )
            hidden_states = self.kda_parallel.group.all_gather(hidden_states, 0)
            if self.dp_compact_decode:
                dp_order, dp_num_valid = self._dp_compact_metadata()
                if dp_order is not None:
                    hidden_states = hidden_states.index_select(0, dp_order)
        num_tokens = hidden_states.size(0)

        def project(layer, inputs, *, zero_tail=False):
            if dp_num_valid is None:
                return layer(inputs)[0]
            assert layer.bias is None
            return masked_bf16_linear_for_dp(
                inputs, layer.weight, dp_num_valid, zero_tail=zero_tail
            )

        # One merged GEMM for q, k, v, b, f_a, g_a (replaces 6 separate GEMMs).
        projected = project(self.in_proj_qkvbfg_a, hidden_states)
        qkv, beta_raw, f_a, g_a = projected.split(
            [
                3 * self.local_projection_size,
                self.local_num_heads,
                self.head_dim,
                self.head_dim,
            ],
            dim=-1,
        )

        # Beta stays raw (bf16) here: the recurrent kernel sigmoids it in fp32
        # at load (SIGMOID_BETA), and only the chunked prefill path needs the
        # pre-computed fp32 sigmoid — computed lazily in _forward. Pure decode
        # / spec-verify steps then skip the _cast_sigmoid kernel and its fp32
        # intermediate entirely.
        beta = beta_raw.unsqueeze(0)
        g1 = project(self.f_b_proj, f_a)
        g1 = g1.reshape(1, -1, self.local_num_heads, self.head_dim)

        g_proj_states = project(self.g_b_proj, g_a)
        # Must stay 3D: rms_norm_gated reads H from g.shape[-2].
        g2 = g_proj_states.reshape(-1, self.local_num_heads, self.head_dim)

        shape = (1, num_tokens, self.local_num_heads, self.head_dim)
        # Zeroed, not empty, when the batch is shared: the recurrent kernel
        # returns early on the null block, so a padded row that lands on block
        # 0 is never written and would otherwise reach the projection as
        # uninitialized memory.
        alloc = torch.zeros if gather else torch.empty
        core_attn_out = alloc(
            shape, dtype=hidden_states.dtype, device=hidden_states.device
        )
        # Call the decorated eager break directly so host-side prefill branches
        # are not captured by PIECEWISE CUDA graphs.
        self._forward(
            qkv_proj_states=qkv,
            g1=g1,
            beta=beta,
            core_attn_out=core_attn_out,
        )
        core_attn_out = self.o_norm(core_attn_out, g2)
        core_attn_out = core_attn_out.reshape(core_attn_out.size(1), -1)
        core_attn_out = project(self.o_proj, core_attn_out, zero_tail=True)
        if dp_order is not None:
            unpacked = torch.empty_like(core_attn_out)
            unpacked.index_copy_(0, dp_order, core_attn_out)
            core_attn_out = unpacked
        if gather:
            # o_proj left an unreduced partial sum over the KDA group; complete
            # it and hand each rank back the tokens it owns. The gather above
            # is rank-ordered, so rank r's rows are exactly chunk r.
            core_attn_out = self.kda_parallel.group.reduce_scatter(core_attn_out, 0)
            if core_attn_out.size(0) > num_local_tokens:
                core_attn_out = core_attn_out[:num_local_tokens]
        return core_attn_out

    def _group_max(self, key: str, value: int) -> int:
        """Largest ``value`` across the KDA group, computed once per step.

        Ranks run different numbers of tokens, but a collective that shares the
        whole batch needs a shape every rank agrees on. The DP coordination
        already publishes the per-rank token counts, so read the maximum from
        there instead of running another collective; the result is cached on
        the forward context, which all 34 KDA layers share.
        """
        cache = get_forward_context().additional_kwargs.setdefault("kda_group_max", {})
        cached = cache.get(key)
        if cached is not None:
            return cached

        dp_metadata = get_forward_context().dp_metadata
        kda_world_size = self.kda_parallel.group.world_size
        if (
            key == "tokens"
            and dp_metadata is not None
            and kda_world_size == dp_metadata.num_tokens_across_dp_cpu.numel()
        ):
            # The KDA group spans the whole DP group here, so the DP token
            # counts ARE its counts -- read the maximum off the host copy the
            # DP coordination already published instead of a new collective.
            cached = int(dp_metadata.num_tokens_across_dp_cpu.max())
        else:
            t = torch.tensor([value], dtype=torch.int32)
            torch.distributed.all_reduce(
                t,
                op=torch.distributed.ReduceOp.MAX,
                group=self.kda_parallel.group.cpu_group,
            )
            cached = int(t[0])
        cache[key] = cached
        return cached

    def _dp_decode_metadata(
        self, meta: GDNAttentionMetadata, max_tokens: int
    ) -> GDNAttentionMetadata:
        """Expand rank-local metadata to describe the gathered batch.

        The gathered token buffer is rank-major with a stride of ``max_tokens``,
        so token row ``t`` on rank ``j`` lands at ``j * max_tokens + t``. The
        decode kernels index requests by token row, so the request axis is laid
        out the same way: every rank contributes exactly ``max_tokens`` request
        slots, of which the leading ones are real. A slot past a rank's own
        token count still gets its own one-token segment -- ``cu_seqlens``
        cannot express a gap, and a zero-length segment would make the next
        rank's segment span the hole -- but it carries that rank's null block
        and its output is dropped when the row is scattered back.
        """
        cached = getattr(meta, "_kda_global_metadata", None)
        if cached is not None:
            return cached

        pool_blocks = self.kv_cache[1].shape[0]
        k = self.request_windows
        if not self.shared_state_pool:
            assert pool_blocks % k == 0, (
                f"mamba state pool holds {pool_blocks} blocks, which does not "
                f"split into {k} request windows (kda_layer_tp={self.kda_layer_tp}); "
                "the pool was sized without kda_layer_tp"
            )
        return globalize_gdn_metadata_for_dp(
            meta,
            group=self.kda_parallel.group,
            request_windows=k,
            shared_state_pool=self.shared_state_pool,
            pool_num_blocks=pool_blocks,
            max_tokens=max_tokens,
            max_requests=(
                (max_tokens + getattr(self, "num_spec", 7))
                // (1 + getattr(self, "num_spec", 7))
                if meta.num_spec_decodes > 0
                else max_tokens
            ),
        )

    def _forward(
        self,
        qkv_proj_states: torch.Tensor,
        g1: torch.Tensor,
        beta: torch.Tensor,
        core_attn_out: torch.Tensor,
    ) -> None:
        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata

        if attn_metadata_raw is None:
            return

        assert isinstance(attn_metadata_raw, dict)
        attn_metadata_narrowed = attn_metadata_raw.get(self.prefix)
        if attn_metadata_narrowed is None:
            # Profile/warmup dummy runs may omit mamba-family metadata.
            return
        assert isinstance(attn_metadata_narrowed, GDNAttentionMetadata)
        if self.tp_over_dp:
            # The GDN metadata builder already expanded this to the gathered
            # batch, so the captured forward only reads tensors -- no
            # collectives and no host-side branching happen here.
            attn_metadata_narrowed = getattr(
                attn_metadata_narrowed, "_kda_global_metadata", attn_metadata_narrowed
            )
        has_initial_state = attn_metadata_narrowed.has_initial_state
        non_spec_query_start_loc = attn_metadata_narrowed.non_spec_query_start_loc
        non_spec_state_indices_tensor = (
            attn_metadata_narrowed.non_spec_state_indices_tensor
        )  # noqa: E501
        num_actual_tokens = attn_metadata_narrowed.num_actual_tokens
        # Spec-decode metadata (all None when speculative decoding is disabled).
        spec_query_start_loc = attn_metadata_narrowed.spec_query_start_loc
        spec_state_indices_tensor = attn_metadata_narrowed.spec_state_indices_tensor
        spec_token_indx = attn_metadata_narrowed.spec_token_indx
        non_spec_token_indx = attn_metadata_narrowed.non_spec_token_indx
        num_accepted_tokens = attn_metadata_narrowed.num_accepted_tokens
        num_spec_decodes = attn_metadata_narrowed.num_spec_decodes
        use_spec = num_spec_decodes > 0
        dp_spec_num_valid = getattr(
            attn_metadata_narrowed, "_kda_dp_spec_num_valid", None
        )
        dp_prefill_num_valid = getattr(
            attn_metadata_narrowed, "_kda_dp_prefill_num_valid", None
        )
        # Safe-gate checkpoints use the bounded sigmoid variant.
        safe_gate = self.kda_safe_gate
        lower_bound = self.kda_lower_bound
        constant_caches = self.kv_cache

        qkv_proj_states = qkv_proj_states[:num_actual_tokens]
        g1 = g1[:, :num_actual_tokens]
        beta = beta[:, :num_actual_tokens]

        conv_state, recurrent_state, *recoverssm_records = constant_caches
        # conv_state must be (..., dim, width-1) for the conv kernels.
        # DS layout stores it that way directly; SD layout needs a transpose.
        # Layout is process-global and resolved once at init (see __init__).
        if not self._conv_state_dim_first:
            conv_state = conv_state.transpose(-1, -2)

        # One merged short-conv over q|k|v instead of three separate calls. The
        # 1D conv is independent per channel, so concatenating q/k/v along the
        # channel dim and running a single causal_conv1d is bit-identical to
        # three calls. The merged weight is q|k|v conv weights concatenated;
        # built once and cached (params are fixed after load). conv_state is
        # already stored as the merged q|k|v state, so it is used directly.
        if self._merged_conv_weight is None:

            def _w(m):
                return m.weight.view(m.weight.size(0), m.weight.size(2))

            self._merged_conv_weight = torch.cat(
                [_w(self.q_conv1d), _w(self.k_conv1d), _w(self.v_conv1d)],
                dim=0,
            ).contiguous()
        conv_weights = self._merged_conv_weight
        conv_bias = self.q_conv1d.bias

        # Split projections / gating into spec (draft-verify) and non-spec token
        # groups when speculative decoding is active. Native verification stores
        # one state per token; RecoverSSM keeps a checkpoint and correction
        # records, then commits the accepted prefix after sampling. Projections are
        # [n, *] (token dim 0); g1/beta are [1, n, h, d] (token dim 1).
        if use_spec:
            # In a pure spec-verify step (no non-spec tokens) the metadata
            # builder sets spec_token_indx = arange(num_actual_tokens), making
            # the index_select calls below identity copies. Skip them on this
            # steady-state decode hot path. The outputs alias the inputs here;
            # the downstream conv/recurrent kernels read them without mutating
            # in place, so the aliasing is safe.
            if dp_spec_num_valid is not None:
                assert spec_token_indx is not None
                qkv_spec = qkv_proj_states.index_select(0, spec_token_indx)
                g1_spec = g1.index_select(1, spec_token_indx)
                beta_spec = beta.index_select(1, spec_token_indx)
            elif non_spec_token_indx is None or non_spec_token_indx.numel() == 0:
                qkv_spec = qkv_proj_states
                g1_spec = g1
                beta_spec = beta
            else:
                qkv_spec = qkv_proj_states.index_select(0, spec_token_indx)
                g1_spec = g1.index_select(1, spec_token_indx)
                beta_spec = beta.index_select(1, spec_token_indx)
            if non_spec_token_indx is not None and non_spec_token_indx.numel() > 0:
                qkv_ns = qkv_proj_states.index_select(0, non_spec_token_indx)
                g1_ns = g1.index_select(1, non_spec_token_indx)
                beta_ns = beta.index_select(1, non_spec_token_indx)
            else:
                qkv_ns = g1_ns = beta_ns = None
        else:
            qkv_spec = g1_spec = beta_spec = None
            qkv_ns, g1_ns, beta_ns = qkv_proj_states, g1, beta

        # --- causal conv1d: spec (draft-verify) path ---
        if use_spec:
            assert spec_state_indices_tensor is not None
            assert num_accepted_tokens is not None
            conv_idx = spec_state_indices_tensor[:, 0][:num_spec_decodes]
            conv_mql = (
                1 + self.num_spec
                if self.use_recoverssm
                else spec_state_indices_tensor.size(-1)
            )
            qkv_spec = causal_conv1d_update(
                qkv_spec,
                conv_state,
                conv_weights,
                conv_bias,
                activation="silu",
                conv_state_indices=conv_idx,
                num_accepted_tokens=num_accepted_tokens,
                query_start_loc=spec_query_start_loc,
                max_query_len=conv_mql,
            )
            q_spec, k_spec, v_spec = qkv_spec.split(self.local_projection_size, dim=-1)

        # --- causal conv1d: non-spec path (prefill or plain decode) ---
        if dp_prefill_num_valid is not None:
            assert not use_spec and non_spec_token_indx is not None
            qkv_ns = qkv_proj_states.index_select(0, non_spec_token_indx)
            g1_ns = g1.index_select(1, non_spec_token_indx)
            beta_ns = beta.index_select(1, non_spec_token_indx)
        raw_qkv_ns = qkv_ns
        q_ns = k_ns = v_ns = None
        if attn_metadata_narrowed.num_prefills > 0:
            assert qkv_ns is not None
            qkv_ns = causal_conv1d_fn(
                qkv_ns.transpose(0, 1),
                conv_weights,
                conv_bias,
                activation="silu",
                conv_states=conv_state,
                has_initial_state=has_initial_state,
                cache_indices=non_spec_state_indices_tensor,
                query_start_loc=non_spec_query_start_loc,
                metadata=attn_metadata_narrowed,
            ).transpose(0, 1)
            q_ns, k_ns, v_ns = qkv_ns.split(self.local_projection_size, dim=-1)
        elif attn_metadata_narrowed.num_decodes > 0:
            assert non_spec_state_indices_tensor is not None
            decode_conv_indices = non_spec_state_indices_tensor[
                : attn_metadata_narrowed.num_decodes
            ]
            qkv_ns = causal_conv1d_update(
                qkv_ns,
                conv_state,
                conv_weights,
                conv_bias,
                activation="silu",
                conv_state_indices=decode_conv_indices,
            )
            q_ns, k_ns, v_ns = qkv_ns.split(self.local_projection_size, dim=-1)

        def _rearr(x):
            return x.reshape(1, -1, self.local_num_heads, self.head_dim)

        # --- core attention: spec (draft-verify) path ---
        core_attn_out_spec = None
        # In a pure spec-verify step (no non-spec tokens) the recurrent kernel
        # can write straight into the layer output buffer, skipping the
        # fresh allocation + copy below. Mixed steps must scatter via
        # spec_token_indx, so they keep the kernel-managed output.
        spec_out = (
            core_attn_out[0, :num_actual_tokens].unsqueeze(0)
            if dp_spec_num_valid is None
            and (non_spec_token_indx is None or non_spec_token_indx.numel() == 0)
            else None
        )
        if use_spec:
            assert spec_state_indices_tensor is not None
            assert num_accepted_tokens is not None
            assert spec_query_start_loc is not None
            # Gate computed inside the recurrent kernel (COMPUTE_GATE) from
            # raw g1 — replicates fused_kda_gate's arithmetic bit-for-bit and
            # skips its launch + fp32 [n, H, D] intermediate per layer.
            if self.use_recoverssm:
                from vllm.models.kimi_k3.nvidia.ops.recoverssm import (
                    kda_recoverssm_verify,
                )

                core_attn_out_spec = kda_recoverssm_verify(
                    q=_rearr(q_spec),
                    k=_rearr(k_spec),
                    v=_rearr(v_spec),
                    raw_g=g1_spec,
                    raw_beta=beta_spec,
                    A_log=self.A_log.view(-1),
                    dt_bias=self.dt_bias,
                    lower_bound=lower_bound,
                    checkpoint_state=recurrent_state,
                    correction_cache=recoverssm_records[0],
                    kg_cache=recoverssm_records[1],
                    query_start_loc=spec_query_start_loc[: num_spec_decodes + 1],
                    state_indices=spec_state_indices_tensor[:num_spec_decodes, 0],
                    spec_query_len=1 + self.num_spec,
                    out=spec_out,
                )
            else:
                core_attn_out_spec, _ = fused_recurrent_kda(
                    q=_rearr(q_spec),
                    k=_rearr(k_spec),
                    v=_rearr(v_spec),
                    g=g1_spec,
                    beta=beta_spec,
                    initial_state=recurrent_state,
                    use_qk_l2norm_in_kernel=True,
                    cu_seqlens=spec_query_start_loc[: num_spec_decodes + 1],
                    ssm_state_indices=spec_state_indices_tensor,
                    num_accepted_tokens=num_accepted_tokens,
                    out=spec_out,
                    sigmoid_beta=True,
                    a_log=self.A_log,
                    g_bias=self.dt_bias,
                    compute_gate=True,
                    lower_bound=lower_bound,
                )

        # --- core attention: non-spec path (prefill or plain decode) ---
        core_attn_out_non_spec = None
        # Only the plain-decode recurrent kernel can write straight into the
        # layer output buffer; the chunked prefill kernel cannot, so this
        # stays None there and the merge copy below runs as before.
        ns_out = None
        if attn_metadata_narrowed.num_prefills > 0:
            assert q_ns is not None
            assert non_spec_state_indices_tensor is not None
            assert has_initial_state is not None
            initial_state = gather_initial_states(
                recurrent_state, non_spec_state_indices_tensor, has_initial_state
            )
            if self.kda_prefill_backend == "flashkda":
                # Non-spec step: write straight into the layer output buffer
                # (dense token order, no merge copy). Step with spec-decode
                # tokens: write to the workspace buffer and scatter below.
                ns_out = (
                    None
                    if use_spec or dp_prefill_num_valid is not None
                    else core_attn_out[:, :num_actual_tokens]
                )
                core_attn_out_non_spec, last_recurrent_state = self._flashkda_prefill(
                    q=_rearr(q_ns),
                    k=_rearr(k_ns),
                    v=_rearr(v_ns),
                    g=g1_ns,
                    beta=beta_ns,
                    initial_state=initial_state,
                    cu_seqlens=non_spec_query_start_loc,
                    out=ns_out,
                    checkpoint=attn_metadata_narrowed.checkpoint,
                    raw_qkv=raw_qkv_ns,
                    conv_state=conv_state,
                    recurrent_state=recurrent_state,
                )
            else:
                (
                    core_attn_out_non_spec,
                    last_recurrent_state,
                ) = chunk_kda_with_fused_gate(
                    q=_rearr(q_ns),
                    k=_rearr(k_ns),
                    v=_rearr(v_ns),
                    raw_g=g1_ns,
                    # Chunk path wants the pre-sigmoided fp32 beta (its
                    # kernels don't sigmoid); beta_ns is raw bf16 from forward.
                    beta=_cast_sigmoid(beta_ns.squeeze(0)).unsqueeze(0),
                    A_log=self.A_log,
                    g_bias=self.dt_bias,
                    initial_state=initial_state,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=True,
                    cu_seqlens=non_spec_query_start_loc,
                    safe_gate=safe_gate,
                    lower_bound=lower_bound,
                )
            # Init cache
            scatter_states(
                recurrent_state,
                last_recurrent_state,
                non_spec_state_indices_tensor,
            )
        elif attn_metadata_narrowed.num_decodes > 0:
            assert non_spec_query_start_loc is not None
            assert non_spec_state_indices_tensor is not None
            # Plain decode step (no spec tokens): token order is dense, so the
            # kernel can write straight into the layer output buffer. A mixed
            # step scatters non-spec output via non_spec_token_indx instead.
            # Gate computed in-kernel (COMPUTE_GATE), beta sigmoided in-kernel.
            if not use_spec:
                ns_out = spec_out
            core_attn_out_non_spec, _ = fused_recurrent_kda(
                q=_rearr(q_ns),
                k=_rearr(k_ns),
                v=_rearr(v_ns),
                g=g1_ns,
                beta=beta_ns,
                initial_state=recurrent_state,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=non_spec_query_start_loc[
                    : attn_metadata_narrowed.num_decodes + 1
                ],
                ssm_state_indices=non_spec_state_indices_tensor,
                out=ns_out,
                sigmoid_beta=True,
                a_log=self.A_log,
                g_bias=self.dt_bias,
                compute_gate=True,
                lower_bound=lower_bound,
            )

        # --- merge spec / non-spec outputs back into token order ---
        if dp_spec_num_valid is not None:
            assert core_attn_out_spec is not None
            valid_tokens = (
                torch.arange(
                    core_attn_out_spec.shape[1], device=core_attn_out_spec.device
                )
                < dp_spec_num_valid[0]
            )
            core_attn_out_spec.masked_fill_(~valid_tokens.view(1, -1, 1, 1), 0)
        if dp_prefill_num_valid is not None:
            assert core_attn_out_non_spec is not None
            valid_tokens = (
                torch.arange(
                    core_attn_out_non_spec.shape[1],
                    device=core_attn_out_non_spec.device,
                )
                < dp_prefill_num_valid[0]
            )
            core_attn_out_non_spec.masked_fill_(~valid_tokens.view(1, -1, 1, 1), 0)
        if use_spec and core_attn_out_non_spec is not None:
            assert core_attn_out_spec is not None
            core_attn_out.index_copy_(1, spec_token_indx, core_attn_out_spec)
            core_attn_out.index_copy_(1, non_spec_token_indx, core_attn_out_non_spec)
        elif use_spec:
            assert core_attn_out_spec is not None
            if spec_out is None:
                if dp_spec_num_valid is not None:
                    assert spec_token_indx is not None
                    core_attn_out.index_copy_(1, spec_token_indx, core_attn_out_spec)
                else:
                    core_attn_out[0, :num_actual_tokens] = core_attn_out_spec.squeeze(0)
        else:
            assert core_attn_out_non_spec is not None
            if ns_out is None:
                if dp_prefill_num_valid is not None:
                    assert non_spec_token_indx is not None
                    core_attn_out.index_copy_(
                        1, non_spec_token_indx, core_attn_out_non_spec
                    )
                else:
                    core_attn_out[0, :num_actual_tokens] = core_attn_out_non_spec[
                        0, :num_actual_tokens
                    ]
