# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash KDA layer with separate convolutions and a bounded safe gate."""

from dataclasses import replace

import torch
import torch.distributed
from torch import nn
from torch.nn import functional as F
from transformers import Glm5NextTextConfig

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import divide
from vllm.forward_context import get_forward_context
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
from vllm.v1.kv_cache_interface import MambaSpec
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
        num_tokens = hidden_states.size(0)
        # One merged GEMM for q, k, v, b, f_a, g_a (replaces 6 separate GEMMs).
        projected = self.in_proj_qkvbfg_a(hidden_states)[0]
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
        g1 = self.f_b_proj(f_a)[0]
        g1 = g1.reshape(1, -1, self.local_num_heads, self.head_dim)

        g_proj_states = self.g_b_proj(g_a)[0]
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
        core_attn_out = self.o_proj(core_attn_out)[0]
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

        assert meta.num_prefills == 0, (
            "KDA across DP ranks supports decode batches only; "
            "prefill token metadata is not globalised"
        )
        assert meta.num_spec_decodes == 0, (
            "kda_layer_tp does not support speculative decoding yet"
        )

        group = self.kda_parallel.group
        k = self.request_windows
        device = meta.non_spec_state_indices_tensor.device
        rows = max_tokens

        def pad_rows(t: torch.Tensor, fill: int) -> torch.Tensor:
            """Pad the request axis to ``rows`` with ``fill``."""
            n = t.shape[0]
            if n >= rows:
                return t[:rows]
            extra = t.new_full((rows - n,) + t.shape[1:], fill)
            return torch.cat([t, extra], 0)

        # Mamba block ids are per-rank: MUSA gives every rank its own pool, so
        # each rank's rows shift into its own window of the combined slot
        # space. A window is one whole local pool wide, null slot included, so
        # each rank's padding lands on *its own* null block rather than racing
        # with the next rank's real state.
        blocks_per_rank = self.kv_cache[1].shape[0]
        assert blocks_per_rank % k == 0, (
            f"mamba state pool holds {blocks_per_rank} blocks, which does not "
            f"split into {k} request windows (kda_layer_tp={self.kda_layer_tp}); "
            "the pool was "
            f"sized without kda_layer_tp"
        )
        blocks_per_rank //= k

        local_state = meta.non_spec_state_indices_tensor
        # One token per row is what makes the arange below valid. It holds for
        # any decode batch: with no prefill and no spec rows every request
        # schedules exactly one token (or none, for a graph-padded slot).
        assert local_state.shape[0] <= rows, (
            f"request axis ({local_state.shape[0]}) exceeds the gathered token "
            f"stride ({rows}); a decode batch has at most one request per token"
        )
        state_indices = (
            group.all_gather(pad_rows(local_state, 0).contiguous().to(torch.int32), 0)
            + torch.arange(k, dtype=torch.int32, device=device).repeat_interleave(rows)
            * blocks_per_rank
        )

        # Cumulative query lengths mirror the gathered buffer exactly: every
        # row of it is a one-token sequence, real requests and padding slots
        # alike, so the global ``cu_seqlens`` is just ``arange(k * rows + 1)``.
        # Padding slots carry that rank's null block, which the recurrent
        # kernel treats as "no state" and their outputs are dropped by the
        # slice back to the local token count.
        qsl = torch.arange(rows * k + 1, dtype=torch.int32, device=device)

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
        meta._kda_global_metadata = global_meta
        return global_meta

    @eager_break_during_capture
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
            # Every rank holds a head slice for *all* requests, so the kernels
            # must see the whole gathered batch, not just this rank's slice.
            # The object is shared by all KDA layers and rebuilt each step, so
            # caching on it keeps this to one round of collectives per step.
            max_tokens = self._group_max(
                "tokens", attn_metadata_narrowed.num_actual_tokens
            )
            attn_metadata_narrowed = self._dp_decode_metadata(
                attn_metadata_narrowed, max_tokens
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
            if non_spec_token_indx is None or non_spec_token_indx.numel() == 0:
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
            if non_spec_token_indx is None or non_spec_token_indx.numel() == 0
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
                ns_out = None if use_spec else core_attn_out[:, :num_actual_tokens]
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
        if use_spec and core_attn_out_non_spec is not None:
            assert core_attn_out_spec is not None
            core_attn_out.index_copy_(1, spec_token_indx, core_attn_out_spec)
            core_attn_out.index_copy_(1, non_spec_token_indx, core_attn_out_non_spec)
        elif use_spec:
            assert core_attn_out_spec is not None
            if spec_out is None:
                core_attn_out[0, :num_actual_tokens] = core_attn_out_spec.squeeze(0)
        else:
            assert core_attn_out_non_spec is not None
            if ns_out is None:
                core_attn_out[0, :num_actual_tokens] = core_attn_out_non_spec[
                    0, :num_actual_tokens
                ]
