# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash KDA layer with separate convolutions and a bounded safe gate."""

import torch
from torch import nn

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import divide
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.mamba.gdn.base import GatedDeltaNetAttention
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
from vllm.transformers_utils.configs.glm5_next import Glm5NextConfig
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

logger = init_logger(__name__)

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
        **kwargs,
    ) -> None:
        self.replicated_shard_ids = set(replicated_shard_ids)
        output_sizes = output_sizes.copy()
        for sid in self.replicated_shard_ids:
            output_sizes[sid] *= tp_size
        super().__init__(input_size, output_sizes, **kwargs)

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


class Glm5NextLinearAttention(GatedDeltaNetAttention):
    head_dim: int
    num_heads: int
    conv_size: int

    def get_attn_backend(self) -> type[AttentionBackend]:
        if self.use_recoverssm:
            from .kda_metadata import Glm5NextRecoverSSMAttentionBackend

            return Glm5NextRecoverSSMAttentionBackend
        return super().get_attn_backend()

    @property
    def gate_lower_bound(self) -> float:
        return self.kda_lower_bound

    def get_state_dtype(
        self,
    ) -> tuple[torch.dtype, ...]:
        if self.model_config is None or self.cache_config is None:
            raise ValueError("model_config and cache_config must be set")
        dtypes = MambaStateDtypeCalculator.kda_state_dtype(
            self.model_config.dtype, self.cache_config.mamba_cache_dtype
        )
        if self.use_recoverssm:
            return MambaStateDtypeCalculator.append_kda_recoverssm_record(
                dtypes, self.model_config.dtype
            )
        return dtypes

    def get_state_shape(
        self,
        *,
        head_shards: int = 1,
    ) -> tuple[tuple[int, ...], ...]:
        # conv_state width must include num_spec so the spec-decode conv update
        # (causal_conv1d_update with num_accepted_tokens + max_query_len) can
        # slide the window across the draft-verify tokens without reading past
        # the allocated width. Matches qwen_gdn_linear_attn.get_state_shape.
        if head_shards < 1 or self.num_heads % (self.tp_size * head_shards):
            raise ValueError("KDA heads must be divisible by TP * head_shards")
        if head_shards > 1 and (self.use_recoverssm or self.num_spec):
            raise ValueError("Head-sharded KDA state supports prefill only")
        shapes = MambaStateShapeCalculator.kda_state_shape(
            self.tp_size * head_shards,
            self.num_heads,
            self.head_dim,
            conv_kernel_size=self.conv_size,
            num_spec=self.num_spec,
        )
        if self.use_recoverssm:
            return MambaStateShapeCalculator.append_kda_recoverssm_record(
                shapes,
                self.num_heads,
                self.head_dim,
                tp_world_size=self.tp_size,
                spec_query_len=1 + self.num_spec,
            )
        return shapes

    def __init__(
        self,
        config: Glm5NextConfig,
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
        if self.use_recoverssm:
            logger.info_once(
                "GLM5Next KDA RecoverSSM enabled: verifying speculative windows "
                "from one checkpoint and committing accepted state after sampling."
            )

        self.head_dim = config.linear_head_dim
        self.num_heads = config.linear_num_heads
        self.conv_size = config.linear_conv_kernel_dim
        assert self.num_heads % self.tp_size == 0
        self.local_num_heads = divide(self.num_heads, self.tp_size)

        projection_size = self.head_dim * self.num_heads
        self.local_projection_size = divide(projection_size, self.tp_size)

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
            tp_size=self.tp_size,
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
        )
        self.dt_bias = nn.Parameter(
            torch.empty(divide(projection_size, self.tp_size), dtype=torch.float32)
        )

        set_weight_attrs(self.dt_bias, {"weight_loader": sharded_weight_loader(0)})

        self.q_conv1d = ColumnParallelLinear(
            input_size=self.conv_size,
            output_size=projection_size,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.q_conv1d",
        )
        self.k_conv1d = ColumnParallelLinear(
            input_size=self.conv_size,
            output_size=projection_size,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.k_conv1d",
        )
        self.v_conv1d = ColumnParallelLinear(
            input_size=self.conv_size,
            output_size=projection_size,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.v_conv1d",
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
        set_weight_attrs(self.A_log, {"weight_loader": sharded_weight_loader(2)})

        self.g_b_proj = ColumnParallelLinear(
            self.head_dim,
            projection_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.g_b_proj",
        )
        self.o_norm = FusedRMSNormGated(self.head_dim, activation="sigmoid")
        self.o_proj = RowParallelLinear(
            projection_size,
            self.hidden_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.o_proj",
        )

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

        # Checkpoints store A_log as 1-D; the model parameter is 4-D.
        def _a_log_weight_loader(param, loaded_weight):
            if loaded_weight.dim() == 1:
                loaded_weight = loaded_weight.view([1, 1, -1, 1])
            return sharded_weight_loader(2)(param, loaded_weight)

        self.A_log.weight_loader = _a_log_weight_loader

        # GLM-5.3-Flash uses a bounded sigmoid gate instead of the default
        # unbounded softplus gate.
        self.kda_safe_gate = True
        self.kda_lower_bound = config.linear_lower_bound
        # Process-global conv-state layout, resolved once here instead of on
        # every _forward call (it reads an env-derived flag each time).
        self._conv_state_dim_first = is_conv_state_dim_first()

    def project_tokens(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project token-local inputs, retaining all TP-local heads."""
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

        return qkv, g1, beta, g2

    def output_tokens(
        self, core_attn_out: torch.Tensor, g2: torch.Tensor
    ) -> torch.Tensor:
        core_attn_out = self.o_norm(core_attn_out, g2)
        core_attn_out = core_attn_out.reshape(core_attn_out.size(1), -1)
        return self.o_proj(core_attn_out)[0]

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens = hidden_states.size(0)
        qkv, g1, beta, g2 = self.project_tokens(hidden_states)
        core_attn_out = torch.empty(
            (1, num_tokens, self.local_num_heads, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        # Call the decorated eager break directly so host-side prefill branches
        # are not captured by PIECEWISE CUDA graphs.
        self._forward(
            qkv_proj_states=qkv,
            g1=g1,
            beta=beta,
            core_attn_out=core_attn_out,
        )
        return self.output_tokens(core_attn_out, g2)

    def merged_conv_weight(self) -> torch.Tensor:
        if self._merged_conv_weight is None:
            self._merged_conv_weight = torch.cat(
                [
                    module.weight.view(module.weight.size(0), -1)
                    for module in (self.q_conv1d, self.k_conv1d, self.v_conv1d)
                ],
                dim=0,
            ).contiguous()
        return self._merged_conv_weight

    def prefill_core(
        self,
        qkv: torch.Tensor,
        g1: torch.Tensor,
        beta: torch.Tensor,
        metadata: GDNAttentionMetadata,
        *,
        head_start: int = 0,
        head_count: int | None = None,
    ) -> torch.Tensor:
        """Advance chronological prefill for an explicit TP-local head interval.

        Projection ownership stays TP-local. Cache ownership must match the
        supplied interval: recurrent state is [request, head, value, key], and
        conv channels are q|k|v for the same heads. Padding is excluded by the
        canonical request metadata before either kernel advances state.
        """
        if metadata.num_decodes or metadata.num_spec_decodes:
            raise ValueError("Head-local KDA supports prefill only")
        if head_count is None:
            head_count = self.local_num_heads
        if (
            head_count <= 0
            or head_start < 0
            or head_start + head_count > self.local_num_heads
        ):
            raise ValueError("Invalid KDA local head interval")
        conv_state, recurrent_state = self.kv_cache[:2]
        if not self._conv_state_dim_first:
            conv_state = conv_state.transpose(-1, -2)
        size = head_count * self.head_dim
        if (
            recurrent_state.shape[1:] != (head_count, self.head_dim, self.head_dim)
            or conv_state.shape[1] != 3 * size
        ):
            raise ValueError("KDA cache shape does not match local recurrence heads")
        if (
            qkv.shape[-1] != 3 * size
            or g1.shape[2:] != (head_count, self.head_dim)
            or beta.shape[2] != head_count
        ):
            raise ValueError("KDA projected inputs do not match local recurrence heads")
        start, stop = (
            head_start * self.head_dim,
            (head_start + head_count) * self.head_dim,
        )
        weights = self.merged_conv_weight().reshape(3, self.local_projection_size, -1)
        weights = weights[:, start:stop].reshape(3 * size, -1).contiguous()
        count = metadata.num_actual_tokens
        indices = metadata.non_spec_state_indices_tensor
        has_initial = metadata.has_initial_state
        qsl = metadata.non_spec_query_start_loc
        convolved = causal_conv1d_fn(
            qkv[:count].transpose(0, 1),
            weights,
            self.q_conv1d.bias,
            activation="silu",
            conv_states=conv_state,
            has_initial_state=has_initial,
            cache_indices=indices,
            query_start_loc=qsl,
            metadata=metadata,
        ).transpose(0, 1)
        q, k, v = (
            x.reshape(1, count, head_count, self.head_dim)
            for x in convolved.split(size, dim=-1)
        )
        initial = gather_initial_states(recurrent_state, indices, has_initial)
        output, final = chunk_kda_with_fused_gate(
            q=q,
            k=k,
            v=v,
            raw_g=g1[:, :count],
            beta=_cast_sigmoid(beta[:, :count].squeeze(0)).unsqueeze(0),
            A_log=self.A_log[:, :, head_start : head_start + head_count],
            g_bias=self.dt_bias[start:stop],
            initial_state=initial,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=qsl,
            safe_gate=self.kda_safe_gate,
            lower_bound=self.kda_lower_bound,
        )
        scatter_states(recurrent_state, final, indices)
        return output

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
        if self.kv_cache[1].shape[1] != self.local_num_heads:
            raise ValueError(
                "Head-sharded KDA state requires PCP prefill or explicit state "
                "export before ordinary decode"
            )
        if (
            attn_metadata_narrowed.num_prefills > 0
            and attn_metadata_narrowed.num_decodes == 0
            and attn_metadata_narrowed.num_spec_decodes == 0
        ):
            count = attn_metadata_narrowed.num_actual_tokens
            core_attn_out[:, :count] = self.prefill_core(
                qkv_proj_states, g1, beta, attn_metadata_narrowed
            )
            return
        has_initial_state = attn_metadata_narrowed.has_initial_state
        non_spec_query_start_loc = attn_metadata_narrowed.non_spec_query_start_loc
        non_spec_state_indices_tensor = (
            attn_metadata_narrowed.non_spec_state_indices_tensor
        )  # noqa: E501
        num_actual_tokens = attn_metadata_narrowed.num_actual_tokens
        # Spec-decode metadata (all None when speculative decoding is disabled).
        spec_sequence_masks = attn_metadata_narrowed.spec_sequence_masks
        spec_query_start_loc = attn_metadata_narrowed.spec_query_start_loc
        spec_state_indices_tensor = attn_metadata_narrowed.spec_state_indices_tensor
        spec_token_indx = attn_metadata_narrowed.spec_token_indx
        non_spec_token_indx = attn_metadata_narrowed.non_spec_token_indx
        num_accepted_tokens = attn_metadata_narrowed.num_accepted_tokens
        num_spec_decodes = attn_metadata_narrowed.num_spec_decodes
        use_spec = num_spec_decodes > 0 and (
            self.use_recoverssm or spec_sequence_masks is not None
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

        conv_weights = self.merged_conv_weight()
        conv_bias = self.q_conv1d.bias

        # Split projections / gating into spec (draft-verify) and non-spec token
        # groups when speculative decoding is active. Spec tokens carry
        # num_spec+1 recurrent-state columns each and are advanced with
        # num_accepted_tokens for rejection-sampling rollback; non-spec tokens
        # are one-per-request. Mirrors olmo_gdn_linear_attn.py. Projections are
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

                if len(recoverssm_records) != 2:
                    raise ValueError(
                        "GLM RecoverSSM requires correction and key/gate caches"
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

        # --- core attention: non-spec path ---
        # Mixed non-spec batch WITHOUT speculation (plain decodes + prefills
        # together): split the core so each request kind keeps its own
        # execution path regardless of batch composition. Plain decodes run
        # the recurrent kernel (raw beta + SIGMOID_BETA/COMPUTE_GATE, exactly
        # like the pure-decode step, updating their own state slots in place
        # and writing straight into their output-buffer rows); prefills run
        # the chunked kernel (fp32 pre-sigmoided beta) on the decode-suffix
        # slices, with cu_seqlens rebased to the suffix token count and
        # gather/scatter touching ONLY the prefill-suffix slots so the decode
        # states are never overwritten. With speculation the GDN builder
        # already reclassifies decodes as prefills, so the spec branch below
        # is untouched. (Port of the HCU mixed-KDA split fix; the conv path
        # above is unchanged — causal_conv1d_fn is per-request via
        # query_start_loc, so mixed conv routing is already correct.)
        num_decodes_ns = attn_metadata_narrowed.num_decodes
        num_prefills_ns = attn_metadata_narrowed.num_prefills
        num_decode_tokens_ns = attn_metadata_narrowed.num_decode_tokens
        mixed_ns = (
            not use_spec
            and num_decodes_ns > 0
            and num_prefills_ns > 0
            and num_decode_tokens_ns == num_decodes_ns
        )
        core_attn_out_non_spec = None
        # Only the plain-decode recurrent kernel can write straight into the
        # layer output buffer; the chunked prefill kernel cannot, so this
        # stays None there and the merge copy below runs as before.
        ns_out = None
        if mixed_ns:
            assert non_spec_query_start_loc is not None
            assert non_spec_state_indices_tensor is not None
            # Decodes are the decode-first prefix rows [0, num_decode_tokens)
            # of the non-spec token stream; prefills are the suffix.
            nd = num_decodes_ns
            ntok_ns = qkv_ns.shape[0]
            # --- decode prefix: recurrent kernel over its own rows ---
            qkv_d = qkv_ns[:num_decode_tokens_ns]
            q_d, k_d, v_d = qkv_d.split(self.local_projection_size, dim=-1)
            decode_out = core_attn_out[0, :num_decode_tokens_ns].unsqueeze(0)
            fused_recurrent_kda(
                q=_rearr(q_d),
                k=_rearr(k_d),
                v=_rearr(v_d),
                g=g1_ns[:, :num_decode_tokens_ns],
                beta=beta_ns[:, :num_decode_tokens_ns],
                initial_state=recurrent_state,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=non_spec_query_start_loc[: nd + 1],
                ssm_state_indices=non_spec_state_indices_tensor[:nd],
                out=decode_out,
                sigmoid_beta=True,
                a_log=self.A_log,
                g_bias=self.dt_bias,
                compute_gate=True,
                lower_bound=lower_bound,
            )
            # --- prefill suffix: chunked kernel on rebased slices ---
            # Rebase the cu_seqlens to the suffix token count (mirrors the
            # GDN builder's prefill_query_start_loc derivation:
            # non_spec_query_start_loc[num_decodes:] - num_decode_tokens).
            prefill_qsl = non_spec_query_start_loc[nd:] - num_decode_tokens_ns
            prefill_state_indices = non_spec_state_indices_tensor[nd:]
            prefill_has_initial_state = (
                has_initial_state[nd:] if has_initial_state is not None else None
            )
            qkv_p = qkv_ns[num_decode_tokens_ns:ntok_ns]
            q_p, k_p, v_p = qkv_p.split(self.local_projection_size, dim=-1)
            initial_state = gather_initial_states(
                recurrent_state,
                prefill_state_indices,
                prefill_has_initial_state,
            )
            (
                prefill_out,
                last_recurrent_state,
            ) = chunk_kda_with_fused_gate(
                q=_rearr(q_p),
                k=_rearr(k_p),
                v=_rearr(v_p),
                raw_g=g1_ns[:, num_decode_tokens_ns:ntok_ns],
                # Chunk path wants the pre-sigmoided fp32 beta (its kernels
                # don't sigmoid); the slice is raw bf16 from forward.
                beta=_cast_sigmoid(
                    beta_ns[:, num_decode_tokens_ns:ntok_ns].squeeze(0)
                ).unsqueeze(0),
                A_log=self.A_log,
                g_bias=self.dt_bias,
                initial_state=initial_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=prefill_qsl,
                safe_gate=safe_gate,
                lower_bound=lower_bound,
            )
            # Scatter ONLY the prefill suffix's final states; the decode
            # prefix already updated its own slots in place above and must
            # not be overwritten.
            scatter_states(
                recurrent_state,
                last_recurrent_state,
                prefill_state_indices,
            )
            # Merge back into token order: the decode prefix rows are already
            # in core_attn_out (decode_out aliases it); copy the prefill
            # suffix rows.
            core_attn_out[0, num_decode_tokens_ns:ntok_ns] = prefill_out[
                0, : ntok_ns - num_decode_tokens_ns
            ]
            core_attn_out_non_spec = core_attn_out
            ns_out = decode_out  # decode rows wrote in place; no merge copy
        elif attn_metadata_narrowed.num_prefills > 0:
            assert q_ns is not None
            assert non_spec_state_indices_tensor is not None
            assert has_initial_state is not None
            initial_state = gather_initial_states(
                recurrent_state, non_spec_state_indices_tensor, has_initial_state
            )
            (
                core_attn_out_non_spec,
                last_recurrent_state,
            ) = chunk_kda_with_fused_gate(
                q=_rearr(q_ns),
                k=_rearr(k_ns),
                v=_rearr(v_ns),
                raw_g=g1_ns,
                # Chunk path wants the pre-sigmoided fp32 beta (its kernels
                # don't sigmoid); beta_ns is raw bf16 from forward.
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
            merged = torch.empty(
                (1, num_actual_tokens, *core_attn_out_spec.shape[2:]),
                dtype=core_attn_out_non_spec.dtype,
                device=core_attn_out_non_spec.device,
            )
            merged.index_copy_(1, spec_token_indx, core_attn_out_spec)
            merged.index_copy_(1, non_spec_token_indx, core_attn_out_non_spec)
            core_attn_out[0, :num_actual_tokens] = merged.squeeze(0)
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
