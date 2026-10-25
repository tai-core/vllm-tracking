# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the GLM kpool indexer's top-k backend wiring."""

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.platforms import current_platform


def _require_deep_gemm() -> None:
    from vllm.utils.deep_gemm import has_deep_gemm

    if not has_deep_gemm():
        pytest.skip("kpool indexer requires DeepGEMM")


def _build(indexer_cls, backend: str):
    cfg = VllmConfig(kernel_config={"sparse_indexer_topk_backend": backend})
    with set_current_vllm_config(cfg):
        return indexer_cls(
            k_cache=None,
            quant_block_size=128,
            scale_fmt="ue8m0",
            topk_tokens=2048,
            head_dim=128,
            max_pool_len=4096,
            max_total_seq_len=8192,
            topk_indices_buffer=torch.empty(8, 2176, dtype=torch.int32, device="cuda"),
        )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA-only dispatch")
@pytest.mark.parametrize("backend", ["auto", "persistent", "cooperative", "torch"])
def test_kpool_indexer_dispatches_through_shared_topk_backend(backend: str) -> None:
    """The kpool indexer must read kernel_config.sparse_indexer_topk_backend
    and hand it to the shared SparseIndexerTopk dispatcher, rather than
    hard-coding its own cooperative/persistent/per_row choice."""
    _require_deep_gemm()
    from vllm.models.glm5next.nvidia.sparse_indexer import SparseAttnIndexerKpool

    assert _build(SparseAttnIndexerKpool, backend).topk_backend == backend


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm-only dispatch")
@pytest.mark.parametrize("backend", ["auto", "aiter", "per_row", "torch"])
def test_kpool_indexer_dispatches_through_shared_topk_backend_rocm(
    backend: str,
) -> None:
    """The AMD kpool indexer must go through the same dispatcher, so the
    AITER decode top-k is reachable from kernel_config."""
    from vllm.models.glm5next.amd.sparse_indexer import SparseAttnIndexerKpool

    assert _build(SparseAttnIndexerKpool, backend).topk_backend == backend


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm-only dispatch")
@pytest.mark.parametrize(
    "configured,full_cudagraph,expected",
    [
        ("auto", False, "aiter"),
        ("auto", True, "auto"),
        ("aiter", True, "aiter"),
        ("per_row", False, "per_row"),
    ],
)
def test_kpool_decode_topk_backend_never_auto_selects_aiter_under_full_cudagraph(
    monkeypatch: pytest.MonkeyPatch,
    configured: str,
    full_cudagraph: bool,
    expected: str,
) -> None:
    """A FULL cudagraph bakes in one kernel for every replay, so "auto" cannot
    use the context length to pick AITER and must stay on the in-tree kernel.
    Explicit backends keep working, and outside FULL the long-context heuristic
    still narrows to AITER."""
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.models.glm5next.amd import sparse_indexer

    monkeypatch.setattr(
        rocm_aiter_ops, "is_indexer_top_k_enabled", lambda: True, raising=False
    )
    monkeypatch.setattr(
        rocm_aiter_ops,
        "is_indexer_top_k_supported",
        lambda **kwargs: True,
        raising=False,
    )

    assert (
        sparse_indexer._kpool_decode_topk_backend(
            configured,
            num_rows=8,
            max_valid_seq_len=64 * 1024,
            select_k=512,
            index_kpool=4,
            full_cudagraph=full_cudagraph,
        )
        == expected
    )


@pytest.mark.parametrize("local_tokens", [0, 1])
def test_pcp_kpool_writes_global_pools_and_scores_local_rows(monkeypatch, local_tokens):
    from types import SimpleNamespace

    from vllm.models.glm5next.nvidia import sparse_indexer as indexer

    # Rank order is deliberately not token order, and the query rank may be empty.
    restore = torch.tensor([0, 2, 4, 6, 7, 5, 3, 1])
    local_k, local_gate = torch.zeros(1, 128), torch.ones(1, 128)
    local_positions = torch.tensor([3])
    global_k = torch.arange(8 * 128).reshape(8, 128).float()
    global_gate = global_k + 1000
    global_positions = torch.arange(8)
    values = {
        id(local_k): global_k,
        id(local_gate): global_gate,
        id(local_positions): global_positions,
    }
    layout = SimpleNamespace(
        restore_idx=restore,
        global_num_tokens=8,
        gather_to_global=lambda tensor: values[id(tensor)],
    )
    slots = torch.arange(8)
    meta = SimpleNamespace(
        slot_mapping=slots,
        num_decodes=0,
        num_decode_tokens=0,
        num_prefills=local_tokens,
        num_prefill_tokens=local_tokens,
        prefill=SimpleNamespace(max_prefill_seq_len=8, chunks=[]),
    )
    tail = SimpleNamespace(
        slot_mapping=torch.arange(8) + 32, num_decode_tokens=0, num_decodes=0
    )
    monkeypatch.setattr(indexer, "DeepseekV32IndexerMetadata", SimpleNamespace)
    monkeypatch.setattr(
        indexer,
        "get_forward_context",
        lambda: SimpleNamespace(
            attn_metadata={"k": meta, "tail": tail}, pcp_token_layout=layout
        ),
    )
    writes, seeds = [], []
    monkeypatch.setattr(
        indexer, "_kpool_compress_insert", lambda *args, **kwargs: writes.append(args)
    )
    monkeypatch.setattr(
        indexer.kpool_ops, "kpool_seed_tail_cache", lambda *args: seeds.append(args)
    )
    monkeypatch.setattr(indexer, "_gather_workspace_shapes", lambda *args: ((0,), (0,)))
    monkeypatch.setattr(
        indexer,
        "current_workspace_manager",
        lambda: SimpleNamespace(
            get_simultaneous=lambda *args: (torch.empty(0), torch.empty(0))
        ),
    )
    topk = torch.full((8, 16), -1, dtype=torch.int32)
    result = indexer.sparse_attn_indexer_kpool(
        torch.zeros(1, 4),
        "k",
        torch.empty(0),
        torch.empty(1, 32, 128),
        None,
        local_k,
        torch.empty(1, 32),
        128,
        "ue8m0",
        16,
        128,
        16,
        8,
        topk,
        False,
        gate_score=local_gate,
        compress_ape=torch.empty(4, 128),
        index_kpool=4,
        positions=local_positions,
        tail_kv_cache=torch.empty(0),
        tail_prefix="tail",
    )
    torch.testing.assert_close(writes[0][0], global_k)
    torch.testing.assert_close(writes[0][1], global_gate)
    torch.testing.assert_close(writes[0][4], slots[restore])
    torch.testing.assert_close(seeds[0][3], tail.slot_mapping)
    if local_tokens:
        assert result[0, :5].tolist() == [0, 1, 2, 3, -1]
    assert (result[local_tokens:] == -1).all()
