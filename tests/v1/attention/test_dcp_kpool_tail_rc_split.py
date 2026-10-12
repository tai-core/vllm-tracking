# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the kpool R/C tail-ring split in the AMD/NVIDIA
kpool_compress primitives.

Triton kernels cannot execute on this host (NOT_RUN); these tests
verify the WRAPPER contract via a launch recorder that captures the
kernel's constexpr arguments, and the wrapper's shape validation:

- the decode wrapper derives the PHYSICAL capacity C from
  tail_kv_cache.shape[2] (a single source of truth) and passes
  TAIL_CAPACITY=C to the kernel; POOL_SIZE stays the LOGICAL R;
- C >= R is enforced; C < R rejects;
- the seed wrapper keeps the legacy kpool (R) signature, derives C
  from the tail shape, and passes both;
- C == R reproduces the legacy constexpr set exactly (POOL_SIZE ==
  TAIL_CAPACITY).

The accepted-history oracle for the KERNEL's arithmetic is the
capacity suite (test_dcp_kpool_tail_capacity.py); native kernel
execution stays NOT_RUN pending node-08.
"""

import pytest
import torch

import vllm.models.glm5next  # noqa: F401  (import cycle bootstrap)
from vllm.triton_utils import triton  # the compat shim

# The CPU triton stub lacks next_power_of_2; the wrappers call it at
# launch time. Provide it for the recorder tests (kernel execution
# stays NOT_RUN).
if not hasattr(triton, "next_power_of_2"):

    def _np2(n):
        return 1 << max(0, (n - 1)).bit_length()

    triton.next_power_of_2 = _np2

R = 4
HEAD_DIM = 64


class _KernelRecorder:
    """Records a Triton kernel launch's constexpr kwargs."""

    def __init__(self, monkeypatch, module, kernel_name):
        self.consts: dict = {}
        self.positional = None
        self.launches = 0
        recorder = self

        class _Rec:
            def __getitem__(self, grid):
                def _launch(*args, **kwargs):
                    recorder.launches += 1
                    recorder.consts.update(
                        {k: v for k, v in kwargs.items() if not k.startswith("_")}
                    )
                    recorder.positional = args

                return _launch

        monkeypatch.setattr(module, kernel_name, _Rec(), raising=False)


def _make_decode_inputs(C, W=2, next_n=3):
    """Valid decode wrapper inputs with a [*, 2, C, HEAD_DIM] tail."""
    num_requests = W
    tail = torch.zeros((8, 2, C, HEAD_DIM), dtype=torch.bfloat16)
    key = torch.zeros((num_requests, next_n, HEAD_DIM), dtype=torch.bfloat16)
    slot_score = torch.zeros_like(key)
    ape = torch.zeros((R, HEAD_DIM), dtype=torch.float32)
    tail_slot_mapping = torch.zeros((num_requests, next_n), dtype=torch.int32)
    slot_mapping = torch.zeros((num_requests, next_n), dtype=torch.int32)
    positions = torch.zeros((num_requests, next_n), dtype=torch.int32)
    kv_cache = torch.zeros((4, 16, HEAD_DIM + 4), dtype=torch.uint8)
    return (
        kv_cache,
        tail,
        tail_slot_mapping,
        key,
        slot_score,
        ape,
        slot_mapping,
        positions,
    )


@pytest.mark.parametrize("platform", ["amd", "nvidia"])
def test_decode_wrapper_passes_tail_capacity(monkeypatch, platform):
    """The decode wrapper derives C from the tail shape and passes
    TAIL_CAPACITY=C with POOL_SIZE=R (the logical size) unchanged."""
    mod = __import__(
        f"vllm.models.glm5next.{platform}.ops.kpool_compress",
        fromlist=["kpool_decode_update_and_maybe_write_cache_batched"],
    )
    rec = _KernelRecorder(monkeypatch, mod, "_kpool_decode_update_batched_kernel")
    C = 8  # > R
    (
        kv_cache,
        tail,
        tsm,
        key,
        score,
        ape,
        sm,
        pos,
    ) = _make_decode_inputs(C)
    mod.kpool_decode_update_and_maybe_write_cache_batched(
        kv_cache,
        tail,
        tsm,
        key,
        score,
        ape,
        sm,
        pos,
        pool_size=R,
        head_dim=HEAD_DIM,
    )
    assert rec.launches == 1
    assert rec.consts["POOL_SIZE"] == R  # the LOGICAL size
    assert rec.consts["TAIL_CAPACITY"] == C  # from the tail shape
    assert rec.consts["HEAD_DIM"] == HEAD_DIM


@pytest.mark.parametrize("platform", ["amd", "nvidia"])
def test_decode_wrapper_legacy_C_equals_R(monkeypatch, platform):
    """C == R (the current runtime allocation): TAIL_CAPACITY ==
    POOL_SIZE — the legacy constexpr set."""
    mod = __import__(
        f"vllm.models.glm5next.{platform}.ops.kpool_compress",
        fromlist=["kpool_decode_update_and_maybe_write_cache_batched"],
    )
    rec = _KernelRecorder(monkeypatch, mod, "_kpool_decode_update_batched_kernel")
    (
        kv_cache,
        tail,
        tsm,
        key,
        score,
        ape,
        sm,
        pos,
    ) = _make_decode_inputs(R)
    mod.kpool_decode_update_and_maybe_write_cache_batched(
        kv_cache,
        tail,
        tsm,
        key,
        score,
        ape,
        sm,
        pos,
        pool_size=R,
        head_dim=HEAD_DIM,
    )
    assert rec.consts["TAIL_CAPACITY"] == rec.consts["POOL_SIZE"] == R


@pytest.mark.parametrize("platform", ["amd", "nvidia"])
def test_decode_wrapper_rejects_C_below_R(platform):
    """A tail ring smaller than the logical pool size rejects."""
    mod = __import__(
        f"vllm.models.glm5next.{platform}.ops.kpool_compress",
        fromlist=["kpool_decode_update_and_maybe_write_cache_batched"],
    )
    (
        kv_cache,
        tail,
        tsm,
        key,
        score,
        ape,
        sm,
        pos,
    ) = _make_decode_inputs(R - 1)
    with pytest.raises(AssertionError, match="must be >= pool_size"):
        mod.kpool_decode_update_and_maybe_write_cache_batched(
            kv_cache,
            tail,
            tsm,
            key,
            score,
            ape,
            sm,
            pos,
            pool_size=R,
            head_dim=HEAD_DIM,
        )


@pytest.mark.parametrize("platform", ["amd", "nvidia"])
def test_decode_wrapper_non_multiple_C(monkeypatch, platform):
    """C = 7 with R = 4 (a non-multiple): accepted — only C >= R is
    required, no integrality constraint."""
    mod = __import__(
        f"vllm.models.glm5next.{platform}.ops.kpool_compress",
        fromlist=["kpool_decode_update_and_maybe_write_cache_batched"],
    )
    rec = _KernelRecorder(monkeypatch, mod, "_kpool_decode_update_batched_kernel")
    (
        kv_cache,
        tail,
        tsm,
        key,
        score,
        ape,
        sm,
        pos,
    ) = _make_decode_inputs(7)
    mod.kpool_decode_update_and_maybe_write_cache_batched(
        kv_cache,
        tail,
        tsm,
        key,
        score,
        ape,
        sm,
        pos,
        pool_size=R,
        head_dim=HEAD_DIM,
    )
    assert rec.consts["TAIL_CAPACITY"] == 7
    assert rec.consts["POOL_SIZE"] == 4


@pytest.mark.parametrize("platform", ["amd", "nvidia"])
def test_seed_wrapper_passes_tail_capacity(monkeypatch, platform):
    """The seed wrapper keeps the legacy kpool (R) signature, derives
    C from the tail shape, and passes both to the kernel."""
    mod = __import__(
        f"vllm.models.glm5next.{platform}.ops.kpool_compress",
        fromlist=["kpool_seed_tail_cache"],
    )
    rec = _KernelRecorder(monkeypatch, mod, "_kpool_tail_seed_kernel")
    C = 8
    tail = torch.zeros((8, 2, C, HEAD_DIM), dtype=torch.bfloat16)
    key = torch.zeros((16, HEAD_DIM), dtype=torch.bfloat16)
    gate = torch.zeros_like(key)
    # tslot encodes block * C + pos % C per the contract.
    tslot = torch.zeros(16, dtype=torch.int32)
    mod.kpool_seed_tail_cache(tail, key, gate, tslot, kpool=R, head_dim=HEAD_DIM)
    assert rec.launches == 1
    assert rec.consts["KPOOL"] == R
    assert rec.consts["TAIL_CAPACITY"] == C


@pytest.mark.parametrize("platform", ["amd", "nvidia"])
def test_seed_wrapper_legacy_C_equals_R(monkeypatch, platform):
    """C == R: TAIL_CAPACITY == KPOOL — the legacy set."""
    mod = __import__(
        f"vllm.models.glm5next.{platform}.ops.kpool_compress",
        fromlist=["kpool_seed_tail_cache"],
    )
    rec = _KernelRecorder(monkeypatch, mod, "_kpool_tail_seed_kernel")
    tail = torch.zeros((8, 2, R, HEAD_DIM), dtype=torch.bfloat16)
    key = torch.zeros((16, HEAD_DIM), dtype=torch.bfloat16)
    gate = torch.zeros_like(key)
    tslot = torch.zeros(16, dtype=torch.int32)
    mod.kpool_seed_tail_cache(tail, key, gate, tslot, kpool=R, head_dim=HEAD_DIM)
    assert rec.consts["TAIL_CAPACITY"] == rec.consts["KPOOL"] == R


@pytest.mark.parametrize("platform", ["amd", "nvidia"])
def test_seed_wrapper_rejects_C_below_R(platform):
    mod = __import__(
        f"vllm.models.glm5next.{platform}.ops.kpool_compress",
        fromlist=["kpool_seed_tail_cache"],
    )
    tail = torch.zeros((8, 2, R - 1, HEAD_DIM), dtype=torch.bfloat16)
    key = torch.zeros((16, HEAD_DIM), dtype=torch.bfloat16)
    gate = torch.zeros_like(key)
    tslot = torch.zeros(16, dtype=torch.int32)
    with pytest.raises(AssertionError, match="must be >= kpool"):
        mod.kpool_seed_tail_cache(tail, key, gate, tslot, kpool=R, head_dim=HEAD_DIM)


@pytest.mark.parametrize("platform", ["amd", "nvidia"])

# --- wrapper layout validation (real wrappers, CPU; kernels NOT_RUN) --------


def _make_seed_inputs(C, n=16, head_dim=HEAD_DIM):
    tail = torch.zeros((8, 2, C, head_dim), dtype=torch.bfloat16)
    key = torch.zeros((n, head_dim), dtype=torch.bfloat16)
    gate = torch.zeros_like(key)
    tslot = torch.zeros(n, dtype=torch.int32)
    return tail, key, gate, tslot


@pytest.mark.parametrize("platform", ["amd", "nvidia"])
def test_seed_wrapper_rejects_bad_layouts(platform):
    """The real seed wrapper rejects (before launch): non-contiguous
    key rows; a longer tslot than key rows; a tail whose head dims
    are strided; (NVIDIA) a non-contiguous tail."""
    mod = __import__(
        f"vllm.models.glm5next.{platform}.ops.kpool_compress",
        fromlist=["kpool_seed_tail_cache"],
    )

    # Non-contiguous key rows (a column slice of a wider tensor).
    tail, key, gate, tslot = _make_seed_inputs(8)
    big = torch.zeros(16, HEAD_DIM * 2, dtype=torch.bfloat16)
    key_strided = big[:, :HEAD_DIM]
    assert not key_strided.is_contiguous()
    with pytest.raises(AssertionError, match="contiguous"):
        mod.kpool_seed_tail_cache(
            tail, key_strided, gate, tslot, kpool=R, head_dim=HEAD_DIM
        )

    # A longer tslot than key rows -> OOB read.
    tail, key, gate, tslot = _make_seed_inputs(8, n=16)
    tslot_long = torch.zeros(32, dtype=torch.int32)
    with pytest.raises(AssertionError, match="out of bounds"):
        mod.kpool_seed_tail_cache(
            tail, key, gate, tslot_long, kpool=R, head_dim=HEAD_DIM
        )

    # A tail whose slot stride != head_dim (a head-strided tail).
    tail, key, gate, tslot = _make_seed_inputs(8)
    tail_strided = torch.zeros(8, 2, 8, HEAD_DIM, 2, dtype=torch.bfloat16)[
        :, :, :, :, 0
    ]
    assert tail_strided.stride(3) == 2 and tail_strided.stride(2) == HEAD_DIM * 2
    with pytest.raises(AssertionError, match="head-contiguous"):
        mod.kpool_seed_tail_cache(
            tail_strided, key, gate, tslot, kpool=R, head_dim=HEAD_DIM
        )

    # NVIDIA additionally: a non-contiguous tail (block-padded
    # strides) rejects; AMD accepts legal padded strides.
    tail, key, gate, tslot = _make_seed_inputs(8)
    tail_padded = torch.zeros(8, 3, 8, HEAD_DIM, dtype=torch.bfloat16)[:, :2]
    assert not tail_padded.is_contiguous()
    assert tail_padded.stride(3) == 1 and tail_padded.stride(2) == HEAD_DIM
    if platform == "nvidia":
        with pytest.raises(AssertionError, match="contiguous"):
            mod.kpool_seed_tail_cache(
                tail_padded, key, gate, tslot, kpool=R, head_dim=HEAD_DIM
            )
    else:
        # AMD accepts legal padded block/head strides; patch the
        # kernel symbol so the launch is a recorded no-op (the test's
        # point is that VALIDATION passes, not kernel execution).
        class _Noop:
            def __getitem__(self, grid):
                return lambda *a, **k: None

        orig = mod._kpool_tail_seed_kernel
        object.__setattr__(mod, "_kpool_tail_seed_kernel", _Noop())
        try:
            mod.kpool_seed_tail_cache(
                tail_padded, key, gate, tslot, kpool=R, head_dim=HEAD_DIM
            )
        finally:
            object.__setattr__(mod, "_kpool_tail_seed_kernel", orig)


@pytest.mark.parametrize("platform", ["amd", "nvidia"])
def test_decode_wrapper_rejects_strided_heads(platform):
    """The decode wrapper rejects key/score rows whose last dim is
    not contiguous (the kernel's head accesses assume stride 1)."""
    mod = __import__(
        f"vllm.models.glm5next.{platform}.ops.kpool_compress",
        fromlist=["kpool_decode_update_and_maybe_write_cache_batched"],
    )
    (
        kv_cache,
        tail,
        tsm,
        key,
        score,
        ape,
        sm,
        pos,
    ) = _make_decode_inputs(8)
    # A [B, next_n, head] view whose last dim is strided: take
    # every other element of a wider contiguous tensor.
    wide = torch.zeros(2, 3, HEAD_DIM * 2, dtype=torch.bfloat16)
    key2 = wide[:, :, ::2]
    assert key2.shape == key.shape and key2.stride(2) == 2
    with pytest.raises(AssertionError, match="head-contiguous"):
        mod.kpool_decode_update_and_maybe_write_cache_batched(
            kv_cache,
            tail,
            tsm,
            key2,
            score,
            ape,
            sm,
            pos,
            pool_size=R,
            head_dim=HEAD_DIM,
        )


@pytest.mark.parametrize("platform", ["amd", "nvidia"])
def test_seed_wrapper_non_multiple_C(monkeypatch, platform):
    """C = 7 with R = 4: the seed's ahead-window and tslot decode use
    C (7), not R."""
    mod = __import__(
        f"vllm.models.glm5next.{platform}.ops.kpool_compress",
        fromlist=["kpool_seed_tail_cache"],
    )
    rec = _KernelRecorder(monkeypatch, mod, "_kpool_tail_seed_kernel")
    tail = torch.zeros((8, 2, 7, HEAD_DIM), dtype=torch.bfloat16)
    key = torch.zeros((16, HEAD_DIM), dtype=torch.bfloat16)
    gate = torch.zeros_like(key)
    tslot = torch.zeros(16, dtype=torch.int32)
    mod.kpool_seed_tail_cache(tail, key, gate, tslot, kpool=R, head_dim=HEAD_DIM)
    assert rec.consts["TAIL_CAPACITY"] == 7
    assert rec.consts["KPOOL"] == 4
