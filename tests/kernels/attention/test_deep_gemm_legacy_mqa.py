# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest
import torch

import vllm.utils.deep_gemm as deep_gemm


@pytest.fixture(autouse=True)
def _reset_mqa_implementations(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(deep_gemm, "_lazy_init", lambda: None)
    monkeypatch.setattr(deep_gemm, "_fp8_fp4_mqa_logits_impl", None)
    monkeypatch.setattr(deep_gemm, "_fp8_fp4_paged_mqa_logits_impl", None)
    monkeypatch.setattr(deep_gemm, "_fp8_mqa_logits_impl", None)
    monkeypatch.setattr(deep_gemm, "_fp8_paged_mqa_logits_impl", None)


def test_legacy_fp8_mqa_fallback_unwraps_query(monkeypatch: pytest.MonkeyPatch):
    legacy_impl = Mock(return_value=torch.empty(0))
    monkeypatch.setattr(deep_gemm, "_fp8_mqa_logits_impl", legacy_impl)
    q = torch.empty(2, 3, 4)
    kv = (torch.empty(5, 4), torch.empty(5))
    weights = torch.empty(2, 3)
    starts = torch.zeros(2, dtype=torch.int32)
    ends = torch.ones(2, dtype=torch.int32)

    deep_gemm.fp8_fp4_mqa_logits(
        (q, None), kv, weights, starts, ends, clean_logits=True
    )

    legacy_impl.assert_called_once_with(
        q, kv, weights, starts, ends, clean_logits=True
    )


def test_legacy_fp8_mqa_fallback_rejects_fp4_query(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(deep_gemm, "_fp8_mqa_logits_impl", Mock())
    q = torch.empty(2, 3, 4)
    q_scale = torch.empty(2, 3)

    with pytest.raises(RuntimeError, match="required APIs"):
        deep_gemm.fp8_fp4_mqa_logits(
            (q, q_scale),
            (torch.empty(5, 4), torch.empty(5)),
            torch.empty(2, 3),
            torch.zeros(2, dtype=torch.int32),
            torch.ones(2, dtype=torch.int32),
            clean_logits=False,
        )


def test_legacy_fp8_paged_mqa_fallback_unwraps_query_and_normalizes_stride(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, object] = {}

    def legacy_impl(*args):
        captured["args"] = args
        return torch.empty(0)

    monkeypatch.setattr(deep_gemm, "_fp8_paged_mqa_logits_impl", legacy_impl)
    q = torch.empty(1, 1, 2, 4)
    kv_cache = torch.empty(2, 64, 1, 8, dtype=torch.uint8)
    weights = torch.empty(1, 2)
    context_lens = torch.ones(1, dtype=torch.int32)
    block_tables = torch.empty(1, 2, dtype=torch.int32).t()
    assert block_tables.shape == (2, 1)
    assert block_tables.stride(-1) != 1
    schedule_metadata = torch.empty(2, 2, dtype=torch.int32)

    deep_gemm.fp8_fp4_paged_mqa_logits(
        (q, None),
        kv_cache,
        weights,
        context_lens,
        block_tables,
        schedule_metadata,
        2048,
        clean_logits=True,
    )

    args = captured["args"]
    assert isinstance(args, tuple)
    assert args[0] is q
    assert args[4].stride(-1) == 1
    assert args[6:] == (2048, True)


@pytest.mark.parametrize("with_scale,with_indices", [(True, False), (False, True)])
def test_legacy_fp8_paged_mqa_fallback_rejects_unsupported_inputs(
    monkeypatch: pytest.MonkeyPatch,
    with_scale: bool,
    with_indices: bool,
):
    monkeypatch.setattr(deep_gemm, "_fp8_paged_mqa_logits_impl", Mock())
    q = torch.empty(1, 1, 2, 4)
    q_scale = torch.empty(1) if with_scale else None
    indices = torch.zeros(1, dtype=torch.int32) if with_indices else None

    with pytest.raises(RuntimeError, match="required APIs"):
        deep_gemm.fp8_fp4_paged_mqa_logits(
            (q, q_scale),
            torch.empty(2, 64, 1, 8, dtype=torch.uint8),
            torch.empty(1, 2),
            torch.ones(1, dtype=torch.int32),
            torch.zeros(1, 1, dtype=torch.int32),
            torch.empty(2, 2, dtype=torch.int32),
            2048,
            clean_logits=False,
            indices=indices,
        )
