# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E402
"""The GLM KPool indexer cache-layout resolver tests (CPU-only).

The single source of truth for the K-region page layout (preshuffle
16x16 tiles vs linear) lives in
vllm/models/glm5next/amd/ops/kpool_compress.py; every writer and
reader must consult it so both sides always agree. These tests pin
the resolver contract and the writers' dispatch on it:

- the default (auto, no env) is preshuffle — existing deployments
  unchanged;
- auto + the lightop paged-MQA backend selects linear;
- the explicit preshuffle/linear selections override;
- an invalid selection fails closed with the named env and options;
- single-slot pages are always linear (nothing to tile);
- the write-cache entry points consult the resolver (a mismatched
  linear layout on a multi-slot page skips the 16x16 tile assert
  path — the kernel receives PRESHUFFLE=False).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


def _load_ops(platform, monkeypatch):
    # Load only the ops module: importing the model package pulls GPU/model
    # dependencies into this CPU-only layout suite.
    name = f"vllm.models.glm5next.{platform}.ops.kpool_compress"
    path = (
        Path(__file__).parents[3]
        / f"vllm/models/glm5next/{platform}/ops/kpool_compress.py"
    )
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, mod)
    spec.loader.exec_module(mod)
    return mod


def _fresh_resolver(monkeypatch, layout_env=None, backend_env=None):
    """Import/reload the resolver with the given env (functools.cache
    cleared so each probe sees the fresh selection)."""
    monkeypatch.setenv("VLLM_HCU_GLM_INDEXER_CACHE_LAYOUT", layout_env)
    if backend_env is None:
        monkeypatch.delenv("VLLM_HCU_GLM_PAGED_MQA_BACKEND", raising=False)
    else:
        monkeypatch.setenv("VLLM_HCU_GLM_PAGED_MQA_BACKEND", backend_env)
    mod = _load_ops("amd", monkeypatch)
    mod.indexer_cache_layout.cache_clear()
    return mod


class TestLayoutResolver:
    def test_default_auto_is_preshuffle(self, monkeypatch):
        """No env: auto resolves preshuffle — every existing deployment
        is unchanged."""
        mod = _fresh_resolver(monkeypatch, layout_env="auto")
        assert mod.indexer_cache_layout() == "preshuffle"
        assert mod.indexer_cache_preshuffle(64) is True

    def test_auto_with_lightop_backend_is_linear(self, monkeypatch):
        """auto + VLLM_HCU_GLM_PAGED_MQA_BACKEND=lightop => linear
        (the LightOp paged_mqa_logits contract on gfx936)."""
        mod = _fresh_resolver(monkeypatch, layout_env="auto", backend_env="lightop")
        assert mod.indexer_cache_layout() == "linear"
        assert mod.indexer_cache_preshuffle(64) is False

    def test_auto_with_other_backends_stays_preshuffle(self, monkeypatch):
        """auto + aiter/torch (the existing defaults) => preshuffle."""
        for backend in ("aiter", "torch", ""):
            mod = _fresh_resolver(
                monkeypatch, layout_env="auto", backend_env=backend or None
            )
            assert mod.indexer_cache_layout() == "preshuffle", backend

    def test_explicit_selections_override(self, monkeypatch):
        """The explicit preshuffle/linear selections override the
        backend-derived auto choice."""
        mod = _fresh_resolver(
            monkeypatch, layout_env="preshuffle", backend_env="lightop"
        )
        assert mod.indexer_cache_layout() == "preshuffle"
        mod = _fresh_resolver(monkeypatch, layout_env="linear", backend_env="aiter")
        assert mod.indexer_cache_layout() == "linear"

    def test_invalid_selection_fails_closed(self, monkeypatch):
        """An invalid layout value is rejected with the named env and
        the valid options — never a silent fallback."""
        mod = _fresh_resolver(monkeypatch, layout_env="bogus")
        with pytest.raises(ValueError, match="VLLM_HCU_GLM_INDEXER_CACHE_LAYOUT"):
            mod.indexer_cache_layout()

    def test_single_slot_page_always_linear(self, monkeypatch):
        """page_size == 1: nothing to tile — linear regardless of the
        resolved layout."""
        mod = _fresh_resolver(monkeypatch, layout_env="preshuffle")
        assert mod.indexer_cache_preshuffle(1) is False
        mod = _fresh_resolver(monkeypatch, layout_env="linear")
        assert mod.indexer_cache_preshuffle(1) is False


class TestWritersConsultResolver:
    @staticmethod
    def _writer_args(page_size=64, n_pools=2, pool_size=4, head_dim=128, num_blocks=8):
        """The real kpool_compress_and_write_cache argument shapes."""
        kv_cache = torch.zeros(num_blocks, page_size, head_dim + 4, dtype=torch.uint8)
        slot_k = torch.randn(n_pools, pool_size, head_dim, dtype=torch.bfloat16)
        slot_score = torch.randn(n_pools, pool_size, head_dim, dtype=torch.bfloat16)
        ape = torch.randn(pool_size, head_dim, dtype=torch.float32)
        loc = torch.zeros(n_pools, dtype=torch.int64)
        return kv_cache, slot_k, slot_score, ape, loc, pool_size

    def test_write_cache_dispatch_uses_resolver(self, monkeypatch):
        """kpool_compress_and_write_cache derives PRESHUFFLE from the
        resolver (not the raw page_size): a linear layout on a
        multi-slot page launches the kernel with PRESHUFFLE=False (the
        historical page_size>1 default is gone)."""
        mod = _fresh_resolver(monkeypatch, layout_env="linear")
        captured = {}

        class _FakeKernel:
            def __getitem__(self, grid):
                def _launch(*args, **kwargs):
                    captured["PRESHUFFLE"] = kwargs.get("PRESHUFFLE")

                return _launch

        monkeypatch.setattr(
            mod, "_kpool_softmax_rotate_write_cache_kernel", _FakeKernel()
        )
        # The CPU triton placeholder lacks next_power_of_2 (the launch
        # kwargs consult it after the kernel object):
        monkeypatch.setattr(
            mod.triton,
            "next_power_of_2",
            lambda n: 1 << (n - 1).bit_length(),
            raising=False,
        )
        args = self._writer_args(page_size=64)
        mod.kpool_compress_and_write_cache(*args)
        assert captured["PRESHUFFLE"] is False, (
            "a linear layout on a multi-slot page must write linear "
            "(PRESHUFFLE=False), not the historical page_size>1 default"
        )

    def test_preshuffle_layout_keeps_tile_dispatch(self, monkeypatch):
        """A preshuffle layout on a multi-slot page keeps the
        historical PRESHUFFLE=True dispatch."""
        mod = _fresh_resolver(monkeypatch, layout_env="preshuffle")
        captured = {}

        class _FakeKernel:
            def __getitem__(self, grid):
                def _launch(*args, **kwargs):
                    captured["PRESHUFFLE"] = kwargs.get("PRESHUFFLE")

                return _launch

        monkeypatch.setattr(
            mod, "_kpool_softmax_rotate_write_cache_kernel", _FakeKernel()
        )
        monkeypatch.setattr(
            mod.triton,
            "next_power_of_2",
            lambda n: 1 << (n - 1).bit_length(),
            raising=False,
        )
        args = self._writer_args(page_size=64)
        mod.kpool_compress_and_write_cache(*args)
        assert captured["PRESHUFFLE"] is True

    def test_preshuffle_asserts_non_multiple_of_16(self, monkeypatch):
        """A preshuffle layout on a page_size not divisible by 16 (50)
        keeps the historical assertion (fail-closed)."""
        mod = _fresh_resolver(monkeypatch, layout_env="preshuffle")
        args = self._writer_args(page_size=50)  # not a multiple of 16
        with pytest.raises(AssertionError, match="16-token tiles"):
            mod.kpool_compress_and_write_cache(*args)


@pytest.mark.parametrize(
    "platform,layout,page_size",
    [
        ("nvidia", "linear", 16),
        ("amd", "linear", 16),
        ("amd", "preshuffle", 32),
        ("amd", "preshuffle", 1),
    ],
)
def test_precompressed_insert_preserves_bytes_scales_and_page_stride(
    monkeypatch, platform, layout, page_size
):
    """Owner-produced bytes must survive replication without requantization."""
    _load_ops("nvidia", monkeypatch)
    mod = _load_ops(platform, monkeypatch)
    if platform == "amd":
        monkeypatch.setenv("VLLM_HCU_GLM_INDEXER_CACHE_LAYOUT", layout)
        mod.indexer_cache_layout.cache_clear()
    storage = torch.full((3, page_size * 132 + 20), 193, dtype=torch.uint8)
    cache = storage[:, : page_size * 132].view(3, page_size, 132)
    values = torch.arange(3 * 128).remainder(256).byte().reshape(3, 128)
    scales = torch.tensor([0.25, 8.0, 0.03125])
    slots = torch.tensor([0, page_size + page_size - 1, page_size * 2])
    assert hasattr(mod, "kpool_insert_compressed"), (
        "precompressed cache boundary required"
    )
    mod.kpool_insert_compressed(cache, values, scales, slots)
    expected = torch.full_like(storage, 193)
    for row, slot in enumerate(slots.tolist()):
        block, token = divmod(slot, page_size)
        for d in range(128):
            offset = (
                (
                    (token // 16) * 16 * 128
                    + (d // 16) * 256
                    + (token % 16) * 16
                    + d % 16
                )
                if platform == "amd" and layout == "preshuffle" and page_size > 1
                else token * 128 + d
            )
            expected[block, offset] = values[row, d]
        expected[
            block, page_size * 128 + token * 4 : page_size * 128 + (token + 1) * 4
        ] = scales[row : row + 1].view(torch.uint8)
    torch.testing.assert_close(storage, expected)
    mod.kpool_insert_compressed(cache, values[:0], scales[:0], slots[:0])
    torch.testing.assert_close(storage, expected)
