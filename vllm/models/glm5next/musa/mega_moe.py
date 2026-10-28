# SPDX-License-Identifier: Apache-2.0
"""MUSA (MATE) MegaMoE adapter for GLM-5.3-Flash.

MATE's ``fp8_fp8_mega_moe`` fuses, in one kernel launch, the expert-parallel
dispatch, the two expert GEMMs, SwiGLU and the combine.  Unlike the SM100
DeepGEMM path it consumes the checkpoint's own quantization format: E4M3
weights with one float32 scale per ``weight_block_size`` block (128x128 for
GLM-5.3-Flash), so no MXFP8/UE8M0 requantization is needed -- the weights are
only re-ordered by ``transform_weights_for_mega_moe``.

Requirements (same as the SM100 backend):
  * expert parallelism, ``tensor_parallel_size == 1`` (tokens are already
    partitioned across ranks by prefill context parallelism),
  * a single expert group,
  * a block-FP8 checkpoint.
"""

from __future__ import annotations
import os

from typing import Optional

import torch
from torch import nn

from vllm.logger import init_logger

logger = init_logger(__name__)

# The symmetric workspace is scratch: one per process, shared by every MoE
# layer.  Creating it per layer would multiply its footprint by the layer
# count (~430 MiB each at 1024 tokens per rank).
_SYMM_BUFFERS: dict[tuple, object] = {}

# MegaMoE kernel-layout tensors keyed by the checkpoint parameter objects they
# came from.  The MTP layer shares the main model's expert parameters, so the
# second adapter for the same tensors must reuse the first transform instead of
# translating already-released (empty) tensors.
_TRANSFORM_CACHE: dict[tuple, tuple] = {}

def torch_cat_pad(tensor, pad: int, hidden: int):
    """Append ``pad`` zero rows (keeps ``all_gather`` shapes equal)."""
    if pad <= 0:
        return tensor
    rows = tensor.new_zeros((pad, hidden))
    return torch.cat([tensor, rows], dim=0)


def _shared_symm_buffer(
    deep_gemm,
    group,
    num_experts: int,
    num_max_tokens_per_rank: int,
    top_k: int,
    hidden_size: int,
    intermediate_size: int,
):
    key = (
        id(group),
        num_experts,
        num_max_tokens_per_rank,
        top_k,
        hidden_size,
        intermediate_size,
    )
    buffer = _SYMM_BUFFERS.get(key)
    if buffer is None:
        buffer = deep_gemm.get_symm_buffer_for_mega_moe(
            group,
            num_experts,
            num_max_tokens_per_rank,
            top_k,
            hidden_size,
            intermediate_size,
        )
        _SYMM_BUFFERS[key] = buffer
        logger.info(
            "MUSA MegaMoE: allocated shared symmetric workspace "
            "(experts=%d top_k=%d max_tokens_per_rank=%d)",
            num_experts, top_k, num_max_tokens_per_rank,
        )
    return buffer


class MusaMegaMoEExperts(nn.Module):
    """Runs the routed experts of one GLM-5.3-Flash MoE layer on MATE MegaMoE.

    The module owns the kernel-side tensors only: the transformed grouped
    weights and the symmetric IPC workspace.  Checkpoint loading is left to
    the surrounding ``FusedMoE`` layer, whose per-rank block-FP8 weights are
    handed to :meth:`build` once loading finishes.
    """

    def __init__(
        self,
        *,
        num_experts: int,
        num_max_tokens_per_rank: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        ep_group,
        tp_world: int = 1,
        activation_clamp: Optional[float] = None,
    ) -> None:
        super().__init__()
        from vllm.utils.deep_gemm import _import_deep_gemm

        self.deep_gemm = _import_deep_gemm()
        self.num_experts = num_experts
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.activation_clamp = activation_clamp
        self.ep_group = ep_group
        self.ep_world = ep_group.size()
        self.ep_rank = ep_group.rank()
        self.tp_world = int(tp_world)

        logger.info(
            "MUSA MegaMoE init: ep_world=%d ep_rank=%d tp=%d sp=%s mt_per_rank=%d",
            self.ep_world, self.ep_rank,
            int(self.ep_world), str(getattr(self, "_is_sp", "?")), num_max_tokens_per_rank,
        )
        self.sym_buffer = _shared_symm_buffer(
            self.deep_gemm,
            ep_group,
            num_experts,
            num_max_tokens_per_rank,
            top_k,
            hidden_size,
            intermediate_size,
        )
        self.l1_weights = None
        self.l2_weights = None
        self._build_attempted = False
        logger.info(
            "MUSA MegaMoE: experts=%d top_k=%d hidden=%d intermediate=%d "
            "max_tokens_per_rank=%d clamp=%s",
            num_experts, top_k, hidden_size, intermediate_size,
            num_max_tokens_per_rank, activation_clamp,
        )

    # ------------------------------------------------------------------
    # weight preparation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def build(
        self,
        w13: torch.Tensor,
        w13_scale: torch.Tensor,
        w2: torch.Tensor,
        w2_scale: torch.Tensor,
    ) -> None:
        """Transform this rank's block-FP8 experts into the kernel layout.

        Args:
            w13: ``(num_local_experts, 2 * intermediate, hidden)`` E4M3.
            w13_scale: ``(num_local_experts, 2 * intermediate // 128, hidden // 128)``
                float32 block scales.
            w2: ``(num_local_experts, hidden, intermediate)`` E4M3.
            w2_scale: ``(num_local_experts, hidden // 128, intermediate // 128)``
                float32 block scales.
        """
        if self._build_attempted:
            return
        self._build_attempted = True
        cache_key = (id(w13), id(w13_scale), id(w2), id(w2_scale))
        cached = _TRANSFORM_CACHE.get(cache_key)
        if cached is not None:
            self.l1_weights, self.l2_weights = cached
            logger.info("MUSA MegaMoE build: reused transform for shared experts")
            return
        logger.info(
            "MUSA MegaMoE build: w13%s %s, w13_scale%s %s, w2%s %s, w2_scale%s %s",
            tuple(w13.shape), w13.dtype,
            tuple(w13_scale.shape), w13_scale.dtype,
            tuple(w2.shape), w2.dtype,
            tuple(w2_scale.shape), w2_scale.dtype,
        )
        assert w13.dtype == torch.float8_e4m3fn, w13.dtype
        assert w2.dtype == torch.float8_e4m3fn, w2.dtype
        assert w13_scale.dtype == torch.float32, w13_scale.dtype
        assert w2_scale.dtype == torch.float32, w2_scale.dtype
        l1 = (w13.contiguous(), w13_scale.contiguous())
        l2 = (w2.contiguous(), w2_scale.contiguous())
        self.l1_weights, self.l2_weights = (
            self.deep_gemm.transform_weights_for_mega_moe(l1, l2)
        )
        _TRANSFORM_CACHE[cache_key] = (self.l1_weights, self.l2_weights)
        # ``transform_weights_for_mega_moe`` copies the tensors whose layout it
        # has to re-order (w13: gate/up interleave) and returns views for the
        # rest (w2).  A copied source is dead weight once its kernel-side twin
        # exists, and keeping it costs ~576 MiB per MoE layer -- 35 GiB over the
        # model.  Release only the sources that no kernel tensor aliases.
        out_ptrs = {t.data_ptr() for t in (*self.l1_weights, *self.l2_weights)}
        released = []
        for name, src in (
            ("w13", w13),
            ("w13_scale", w13_scale),
            ("w2", w2),
            ("w2_scale", w2_scale),
        ):
            if src.numel() == 0 or src.data_ptr() in out_ptrs:
                continue
            released.append(name)
            src.data = torch.empty(0, dtype=src.dtype, device=src.device)
        logger.info(
            "MUSA MegaMoE build done: released %s (kernel tensors alias the rest)",
            released or "nothing",
        )

    @property
    def is_ready(self) -> bool:
        return self.l1_weights is not None and self.l2_weights is not None

    # ------------------------------------------------------------------
    # execution
    # ------------------------------------------------------------------
    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Run the routed experts over this rank's share of the tokens.

        The kernel is expert-parallel: every rank owns a disjoint subset of the
        experts, exchanges the tokens that route to them and returns the result
        for the tokens it contributed.  MoE layers here are handed the whole
        batch, so the batch is split across the expert-parallel ranks and the
        per-rank results are gathered back.
        """
        import torch.distributed as dist
        import torch.nn.functional as F

        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            per_token_group_quant_fp8,
        )

        if not self.is_ready:
            raise RuntimeError("MusaMegaMoEExperts.build() was not called")

        num_tokens, hidden = hidden_states.shape

        # Under prefill context parallelism every rank holds the same tokens and
        # MATE's fused kernel expects each rank to submit them all (splitting the
        # batch across ranks produces wrong results, verified).  Submit the whole
        # tensor; the cost is that every rank runs its experts over the full
        # dispatch pool.
        if self.tp_world > 1 and num_tokens > 1:
            per = (num_tokens + self.ep_world - 1) // self.ep_world
            start = self.ep_rank * per
            stop = min(start + per, num_tokens)
            n_real = max(0, stop - start)
            x_sub = hidden_states.new_zeros((per, hidden))
            w_sub = topk_weights.new_zeros((per, topk_weights.shape[-1]))
            i_sub = topk_ids.new_zeros((per, topk_ids.shape[-1]))
            if n_real:
                x_sub[:n_real] = hidden_states[start:stop]
                w_sub[:n_real] = topk_weights[start:stop]
                i_sub[:n_real] = topk_ids[start:stop]
            y_local = self._run(x_sub, w_sub, i_sub)
            chunks = [torch.empty_like(y_local) for _ in range(self.ep_world)]
            dist.all_gather(chunks, y_local.contiguous(), group=self.ep_group)
            return torch.cat(chunks, dim=0)[:num_tokens]

        cap = self.num_max_tokens_per_rank
        if num_tokens <= cap:
            return self._run(hidden_states, topk_weights, topk_ids)
        outs = []
        for start in range(0, num_tokens, cap):
            stop = min(start + cap, num_tokens)
            outs.append(
                self._run(
                    hidden_states[start:stop],
                    topk_weights[start:stop],
                    topk_ids[start:stop],
                )
            )
        return torch.cat(outs, dim=0)

    @torch.no_grad()
    def _run(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            per_token_group_quant_fp8,
        )

        num_tokens, hidden = hidden_states.shape
        if num_tokens > self.num_max_tokens_per_rank:
            raise ValueError(
                f"MegaMoE got {num_tokens} tokens per rank but the workspace was "
                f"built for {self.num_max_tokens_per_rank}"
            )
        # Per-token, 128-column groups: the layout ``sym_buffer.x_sf`` expects.
        x, x_scale = per_token_group_quant_fp8(hidden_states, 128)
        sym = self.sym_buffer
        sym.x[:num_tokens].copy_(x)
        sym.x_sf[:num_tokens].copy_(x_scale.to(sym.x_sf.dtype))
        sym.topk_idx[:num_tokens].copy_(topk_ids.to(sym.topk_idx.dtype))
        sym.topk_weights[:num_tokens].copy_(topk_weights.to(sym.topk_weights.dtype))
        y = torch.empty(
            (num_tokens, hidden), dtype=torch.bfloat16, device=hidden_states.device
        )
        self.deep_gemm.fp8_fp8_mega_moe(
            y=y,
            l1_weights=self.l1_weights,
            l2_weights=self.l2_weights,
            sym_buffer=sym,
            cumulative_local_expert_recv_stats=None,
            activation_clamp=self.activation_clamp,
            fast_math=True,
        )
        return y
