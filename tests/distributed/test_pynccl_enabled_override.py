# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the PyNcclCommunicator per-communicator `enabled` override.

These tests exercise the REAL constructor control flow of
``PyNcclCommunicator.__init__`` on CPU. The NCCL library boundary
(``NCCLLibrary``) and the device/stream boundary are MOCKED and labeled;
the tests assert the guard/library-init routing decisions (available/
disabled flags and whether ``NCCLLibrary`` was attempted), never a fake
GPU-available PASS.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from vllm.distributed.device_communicators import (  # noqa: E402
    pynccl as pynccl_mod,
)
from vllm.distributed.device_communicators.pynccl import (  # noqa: E402
    PyNcclCommunicator,
)


class _FakeStatelessGroup(pynccl_mod.StatelessProcessGroup):
    """StatelessProcessGroup stand-in (rank/world_size only)."""

    def __init__(self, rank: int, world_size: int):
        # bypass the real TCPStore-based __init__ entirely
        self.rank = rank
        self.world_size = world_size


class _FakeNCCLLibrary:
    """MOCK_NCCL_LIBRARY: records the construction attempt, then raises.

    On CPU there is no real NCCL; the honest behavior is that the library
    boundary is unavailable. Recording the attempt lets tests assert the
    GUARD ROUTING (was NCCLLibrary attempted at all) without faking a
    GPU-available PASS.
    """

    instances: list[_FakeNCCLLibrary] = []

    def __init__(self, library_path=None):
        self.library_path = library_path
        _FakeNCCLLibrary.instances.append(self)
        raise RuntimeError("MOCK_NCCL_LIBRARY: unavailable on CPU")


@pytest.fixture()
def nccl_boundary(monkeypatch):
    """MOCK the NCCL library boundary; reset the record each test."""
    _FakeNCCLLibrary.instances = []
    monkeypatch.setattr(pynccl_mod, "NCCLLibrary", _FakeNCCLLibrary)
    return _FakeNCCLLibrary


@pytest.fixture()
def env_flag(monkeypatch):
    """Control envs.VLLM_DISABLE_PYNCCL without touching os.environ."""

    def set_flag(value: bool):
        monkeypatch.setattr(pynccl_mod.envs, "VLLM_DISABLE_PYNCCL", value)

    return set_flag


def _make(world_size: int = 2, **kwargs) -> PyNcclCommunicator:
    group = _FakeStatelessGroup(rank=0, world_size=world_size)
    return PyNcclCommunicator(group, device="cpu", **kwargs)


class TestDefaultBehavior:
    def test_default_none_global0_attempts_library(self, nccl_boundary, env_flag):
        set_flag = env_flag
        set_flag(False)
        # default None, global 0: proceeds past the guard and attempts
        # NCCLLibrary. On CPU the (honest) unavailable mock then fails
        # closed — the ROUTING assertion is that the attempt happened.
        comm = _make(enabled=None)
        assert nccl_boundary.instances, "library should be attempted"
        assert comm.available is False
        assert comm.disabled is True

    def test_default_none_global1_disabled(self, nccl_boundary, env_flag):
        set_flag = env_flag
        set_flag(True)
        comm = _make(enabled=None)
        assert comm.available is False
        assert comm.disabled is True
        assert not nccl_boundary.instances, "library must NOT be attempted"

    def test_positional_compat_library_path(self, nccl_boundary, env_flag):
        """Existing positional library_path argument still works."""
        set_flag = env_flag
        set_flag(False)
        group = _FakeStatelessGroup(0, 2)
        comm = PyNcclCommunicator(group, "cpu", "some/path/libnccl.so")
        assert nccl_boundary.instances
        assert nccl_boundary.instances[0].library_path == "some/path/libnccl.so"
        assert comm.disabled is True  # honest CPU unavailable path


class TestExplicitOverride:
    def test_explicit_true_overrides_env_disable(self, nccl_boundary, env_flag):
        set_flag = env_flag
        set_flag(True)
        comm = _make(enabled=True)
        # the env-disable decision is overridden for THIS communicator only:
        # the guard is passed and the library boundary is attempted
        assert nccl_boundary.instances, "library should be attempted"
        # honest CPU: the unavailable boundary then fails closed
        assert comm.available is False
        assert comm.disabled is True

    def test_explicit_false_disables_despite_env0(self, nccl_boundary, env_flag):
        set_flag = env_flag
        set_flag(False)
        comm = _make(enabled=False)
        assert comm.available is False
        assert comm.disabled is True
        assert not nccl_boundary.instances

    def test_explicit_true_env_unchanged(self, nccl_boundary, env_flag):
        set_flag = env_flag
        set_flag(True)
        _make(enabled=True)
        # the envs attribute the rest of the process reads is untouched
        assert pynccl_mod.envs.VLLM_DISABLE_PYNCCL is True


class TestWorldSizeOne:
    def test_world_size1_never_initializes_even_with_enabled_true(
        self, nccl_boundary, env_flag
    ):
        set_flag = env_flag
        set_flag(False)
        comm = _make(world_size=1, enabled=True)
        assert comm.available is False
        assert comm.disabled is True
        assert not nccl_boundary.instances

    def test_world_size1_default_disabled(self, nccl_boundary, env_flag):
        comm = _make(world_size=1)
        assert comm.available is False
        assert comm.disabled is True


class TestValidation:
    def test_invalid_enabled_type_rejected(self, nccl_boundary, env_flag):
        with pytest.raises(TypeError, match="enabled must be bool"):
            _make(enabled="yes")
        with pytest.raises(TypeError, match="enabled must be bool"):
            _make(enabled=1)

    def test_kwarg_only(self, nccl_boundary, env_flag):
        """`enabled` must not be positionally bindable."""
        set_flag = env_flag
        set_flag(True)
        with pytest.raises(TypeError):
            PyNcclCommunicator(_FakeStatelessGroup(0, 2), "cpu", None, True)


class TestLibraryUnavailable:
    def test_library_unavailable_still_fail_closed(self, nccl_boundary, env_flag):
        set_flag = env_flag
        set_flag(False)
        comm = _make(enabled=True)
        # the honest unavailable boundary always fails closed
        assert comm.available is False
        assert comm.disabled is True


class TestOtherCommunicatorsUntouched:
    def test_override_does_not_affect_new_instances(self, nccl_boundary, env_flag):
        set_flag = env_flag
        set_flag(True)
        _make(enabled=True)
        assert nccl_boundary.instances, "first (overridden) attempts library"
        nccl_boundary.instances.clear()
        # a second communicator with default None still honors the env
        second = _make(enabled=None)
        assert not nccl_boundary.instances, "second must not attempt library"
        assert second.available is False
        assert second.disabled is True
        # and the env decision source is unchanged
        assert pynccl_mod.envs.VLLM_DISABLE_PYNCCL is True
