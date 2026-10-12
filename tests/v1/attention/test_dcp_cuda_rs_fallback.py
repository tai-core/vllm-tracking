# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E501
"""The CudaCommunicator.reduce_scatter fallback tests (CPU-only, the
ACTUAL method under test with clearly-labeled fakes; no native
claim).

THE DEFECT (the device-verified FAIL_NUMERIC root): the old
reduce_scatter asserted pynccl_comm is not None but never checked
DISABLED — it allocated torch.empty(output_shape) and called
PyNcclCommunicator.reduce_scatter, which returns IMMEDIATELY when
disabled=True => the UNWRITTEN buffer is returned (the actual-method
CPU extraction verified a disabled PyNccl leaves a NaN sentinel
unchanged and the actual CudaRS returns it). all_gather already had
the uniform fallback (missing OR disabled => super()); reduce_scatter
did not. The matched DCP2 diagnostic environment runs
VLLM_DISABLE_PYNCCL=1."""

from __future__ import annotations

from typing import Any

import torch  # noqa: E402

from vllm.distributed.device_communicators.cuda_communicator import (
    CudaCommunicator,
)


class _FakePyNccl:
    """The CLEARLY-LABELED fake PyNcclCommunicator: records calls; the
    disabled instance is a NO-OP (the real behavior — returns without
    touching the output buffer)."""

    def __init__(self, disabled: bool):
        self.disabled = disabled
        self.calls: list[tuple[Any, Any]] = []

    def reduce_scatter(self, output, input_):
        self.calls.append(("reduce_scatter", input_))
        if self.disabled:
            return  # the real no-op: the output stays UNWRITTEN
        # The enabled fake: a deterministic sum-scatter (rank r gets
        # the sum of chunk r across "ranks" — here a single-process
        # stand-in writing the chunk sum):
        ws = input_.shape[0] // output.shape[0]
        chunk = input_.shape[0] // ws
        output.copy_(input_[0:chunk] * ws)


def _make_comm(pynccl: _FakePyNccl | None, world_size: int = 2) -> CudaCommunicator:
    """Build a REAL CudaCommunicator instance whose pynccl_comm is the
    labeled fake (the method under test is the REAL class method)."""
    comm = CudaCommunicator.__new__(CudaCommunicator)
    comm.pynccl_comm = pynccl
    comm.world_size = world_size
    comm.rank_in_group = 0
    comm.device_group = "FAKE_DEVICE_GROUP"
    return comm


class TestReduceScatterFallback:
    def test_disabled_pynccl_uses_base_not_noop(self, monkeypatch):
        """OLD-RED/NEW-GREEN: a DISABLED PyNccl must NOT receive the
        reduce_scatter call (the no-op would leave the output
        unwritten) and MUST route through the REAL base
        torch.distributed.reduce_scatter_tensor with the device
        group."""
        fake = _FakePyNccl(disabled=True)
        comm = _make_comm(fake)
        x = torch.ones(4, 4)  # dim=-1 => [4,4] moved [4,4]; 4 % 2 == 0
        # Poison the base route's output deterministically: intercept
        # torch.distributed.reduce_scatter_tensor with a labeled fake
        # that writes a SENTINEL (deterministic against unwritten
        # torch.empty):
        calls: list[dict] = []

        def fake_rs_tensor(output, input_, group=None, **kw):
            calls.append(
                {
                    "out_shape": tuple(output.shape),
                    "in_shape": tuple(input_.shape),
                    "group": group,
                }
            )
            output.fill_(7.0)  # the deterministic sentinel

        monkeypatch.setattr(torch.distributed, "reduce_scatter_tensor", fake_rs_tensor)
        out = comm.reduce_scatter(x, dim=-1)
        # The base route ran (NOT the no-op PyNccl):
        assert len(fake.calls) == 0, "the disabled PyNccl was called!"
        assert len(calls) == 1
        assert calls[0]["group"] == "FAKE_DEVICE_GROUP"
        assert calls[0]["out_shape"] == (2, 4)
        assert calls[0]["in_shape"] == (4, 4)
        # The output is the base-route sentinel (deterministic — never
        # an unwritten torch.empty); the movedim-back maps [2,4]->[4,2]:
        assert torch.equal(out, torch.full((4, 2), 7.0))

    def test_missing_pynccl_uses_base(self, monkeypatch):
        """A MISSING (None) PyNccl also falls back to the base (the
        old assert pynccl_comm is not None crashed)."""
        comm = _make_comm(None)
        calls: list[tuple] = []

        def fake_rs_tensor(output, input_, group=None, **kw):
            calls.append((tuple(output.shape), tuple(input_.shape), group))
            output.fill_(3.0)

        monkeypatch.setattr(torch.distributed, "reduce_scatter_tensor", fake_rs_tensor)
        out = comm.reduce_scatter(torch.ones(4, 4), dim=-1)
        assert len(calls) == 1
        assert torch.equal(out, torch.full((4, 2), 3.0))

    def test_enabled_pynccl_route_unchanged(self, monkeypatch):
        """An ENABLED PyNccl keeps the existing route (the PyNccl call
        receives the contiguous input; the base is NOT invoked)."""
        fake = _FakePyNccl(disabled=False)
        comm = _make_comm(fake)
        base_calls: list[tuple] = []

        def fake_rs_tensor(output, input_, group=None, **kw):  # pragma: no cover
            base_calls.append((1,))
            output.fill_(99.0)

        monkeypatch.setattr(torch.distributed, "reduce_scatter_tensor", fake_rs_tensor)
        x = torch.ones(4, 4)
        out = comm.reduce_scatter(x, dim=-1)
        assert len(fake.calls) == 1, "the enabled PyNccl was NOT called"
        assert len(base_calls) == 0, "the base was wrongly invoked"
        # The enabled fake wrote the chunk sum (input[0:2] * ws =
        # 2.0 — deterministic); the movedim-back maps [2,4]->[4,2]:
        assert torch.equal(out, torch.full((4, 2), 2.0))

    def test_world1_identity(self):
        """The single-rank identity: the base returns the input
        unchanged (no collective)."""
        fake = _FakePyNccl(disabled=True)
        comm = _make_comm(fake, world_size=1)
        x = torch.randn(3, 2)
        out = comm.reduce_scatter(x, dim=-1)
        assert out is x  # the identity, no copy

    def test_dims_and_strided_inputs(self, monkeypatch):
        """dim 0, dim 1, NEGATIVE dim, and STRIDED inputs: the base
        route packs/moves exactly (the movedim/contiguous contract)
        with the exact head/rank chunk order."""
        fake = _FakePyNccl(disabled=True)
        comm = _make_comm(fake)
        calls: list[dict] = []

        def fake_rs_tensor(output, input_, group=None, **kw):
            calls.append(
                {
                    "out": output,
                    "out_shape": tuple(output.shape),
                    "in_shape": tuple(input_.shape),
                }
            )
            # Write the rank-0 chunk of the (moved) input so the
            # return-mapping is verifiable:
            output.copy_(input_[0 : output.shape[0]])

        monkeypatch.setattr(torch.distributed, "reduce_scatter_tensor", fake_rs_tensor)
        # dim=0 on a strided (transposed) input: [6, 4] with dim=0
        # => 3 chunks of rows (6 // 2), each [3, 4]:
        base = torch.arange(24, dtype=torch.float32).reshape(4, 6)
        strided = base.t()  # shape (6, 4), non-contiguous
        out = comm.reduce_scatter(strided, dim=0)
        assert len(calls) == 1
        assert calls[0]["out_shape"] == (3, 4)  # 6 // 2 chunks on dim 0
        # The returned chunk maps the FIRST dim-0 chunk of the moved
        # view (the first 3 rows of strided):
        assert torch.equal(out, strided[0:3].contiguous())
        # dim=1 on [4, 6]: the moved view [6,4] chunks to [3,4]; the
        # return maps back to [4,3] == base[:, 0:3]:
        calls.clear()
        out1 = comm.reduce_scatter(base, dim=1)
        assert calls[0]["out_shape"] == (3, 4)
        assert torch.equal(out1, base[:, 0:3].contiguous())
        # dim=-1 (equivalent to 1 here):
        calls.clear()
        outn = comm.reduce_scatter(base, dim=-1)
        assert torch.equal(outn, out1)

    def test_real_base_class_math(self, monkeypatch):
        """The REAL base-class implementation used unmocked except the
        torch.distributed call: an END-TO-END actual-method check —
        the base's packing/movedim logic with a labeled torch RS
        double that sums the two halves (the exact reduce-scatter
        semantics for ws=2):"""
        fake = _FakePyNccl(disabled=True)
        comm = _make_comm(fake)

        def fake_rs_tensor(output, input_, group=None, **kw):
            ws = input_.shape[0] // output.shape[0]
            chunk = input_.shape[0] // ws
            # The exact sum-over-ranks reduce-scatter:
            output.copy_(sum(input_[r * chunk : (r + 1) * chunk] for r in range(ws)))

        monkeypatch.setattr(torch.distributed, "reduce_scatter_tensor", fake_rs_tensor)
        x = torch.arange(8, dtype=torch.float32).reshape(4, 2)
        out = comm.reduce_scatter(x, dim=0)
        # chunks [0:2] + [2:4] summed:
        expected = x[0:2] + x[2:4]
        assert torch.equal(out, expected)
