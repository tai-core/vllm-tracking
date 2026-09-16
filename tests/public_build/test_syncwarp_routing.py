# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Focused HOST compiler tests for the vllm_syncwarp dispatcher.

SYNTHETIC routing/sequence coverage — NOT GPU synchronization
correctness. The REAL dispatcher block is EXTRACTED VERBATIM from
csrc/libtorch_stable/fused_qknorm_rope_kernel.cu and compiled on the
HOST with preprocessor scaffolding (synthetic
__builtin_amdgcn_fence/wave_barrier intrinsics that TRACE, and a
synthetic SDK-native __syncwarp ALWAYS declared to prove no global
redefinition and unambiguous routing):

  1. legacy ROCm (USE_ROCM, HIP_VERSION < 70000000) with the synthetic
     native present: the PRESERVED fallback sequence runs IN ORDER
     (release-fence -> wave-barrier -> acquire-fence); the native is
     NOT called; no redefinition error (real compile proves it).
  2. modern HIP (USE_ROCM, HIP_VERSION >= 70000000): delegates to the
     native intrinsic.
  3. CUDA (no USE_ROCM): delegates to the native intrinsic.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
KERNEL = ROOT / "csrc" / "libtorch_stable" / "fused_qknorm_rope_kernel.cu"
CXX = shutil.which("clang++") or shutil.which("g++")

requires_cxx = pytest.mark.skipif(CXX is None, reason="no host C++ compiler")


def _extract_dispatcher() -> str:
    """Slice the ROCm/CUDA helper region VERBATIM from the kernel file:
    from the `#ifdef USE_ROCM` that precedes FINAL_MASK through the
    `#endif` closing the CUDA-branch helper."""
    s = KERNEL.read_text()
    i = s.index("#ifdef USE_ROCM")
    # the region ends at the endif after the CUDA-branch helper
    j = s.index("::__syncwarp();", s.index("CUDA (and any non-ROCm build)"))
    j = s.index("#endif", j) + len("#endif")
    return s[i:j]


SCAFFOLD = """
// ---- synthetic host scaffolding (NOT an SDK copy) ----
#include <cstdio>
#define __device__
#define __builtin_amdgcn_fence(order, scope) \\
    printf("FENCE_%s_%s\\n", (order == __ATOMIC_RELEASE ? "REL" : "ACQ"), scope)
#define __builtin_amdgcn_wave_barrier() printf("WAVE_BARRIER\\n")
// synthetic SDK-NATIVE __syncwarp — ALWAYS declared (all profiles) to
// prove the legacy dispatcher does NOT redefine or pick it
static inline void __syncwarp() { printf("NATIVE_SYNCWARP\\n"); }
"""

MAIN_TMPL = """
int main() {
    vllm_syncwarp();
    printf("ROUTING_DONE\\n");
    return 0;
}
"""


def _build_and_run(tmp_path: Path, defines: list[str]) -> list[str]:
    src = tmp_path / "tu.cpp"
    src.write_text(SCAFFOLD + "\n" + _extract_dispatcher() + "\n" + MAIN_TMPL)
    exe = tmp_path / "tu"
    r = subprocess.run(
        [CXX, "-std=c++17", *defines, str(src), "-o", str(exe)],
        capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, (
        "COMPILATION FAILED (redefinition or ambiguity):\n" + r.stderr[-1500:]
    )
    r2 = subprocess.run([str(exe)], capture_output=True, text=True,
                        timeout=60)
    assert r2.returncode == 0, r2.stderr
    return [l for l in r2.stdout.splitlines() if l.strip()]


@requires_cxx
def test_legacy_rocm_with_native_present_uses_fallback_sequence(tmp_path):
    """Legacy ROCm (< 7.0) with the synthetic SDK-native __syncwarp
    DECLARED: compilation succeeds (no global redefinition) and the
    PRESERVED fallback sequence runs IN ORDER; the native is not
    called."""
    lines = _build_and_run(
        tmp_path,
        ["-DUSE_ROCM", "-DHIP_VERSION=6000000000" if False else
         "-DHIP_VERSION=60000000"])
    assert lines == [
        "FENCE_REL_wavefront",
        "WAVE_BARRIER",
        "FENCE_ACQ_wavefront",
        "ROUTING_DONE",
    ], lines


@requires_cxx
def test_modern_hip_delegates_to_native(tmp_path):
    lines = _build_and_run(
        tmp_path, ["-DUSE_ROCM", "-DHIP_VERSION=70000000"])
    assert lines == ["NATIVE_SYNCWARP", "ROUTING_DONE"], lines


@requires_cxx
def test_cuda_delegates_to_native(tmp_path):
    lines = _build_and_run(tmp_path, [])
    assert lines == ["NATIVE_SYNCWARP", "ROUTING_DONE"], lines


def test_kernel_callsites_all_routed():
    """Source contract: every former __syncwarp() call goes through the
    named helper; no unqualified __syncwarp() call remains; the exact
    release->barrier->acquire sequence is preserved verbatim."""
    src = KERNEL.read_text()
    calls = re.findall(r'(?<![::\w])__syncwarp\(\);', src)
    assert not calls, calls
    assert src.count("vllm_syncwarp();") == 4
    # preserved sequence, in order, inside the legacy branch
    i = src.index("__device__ inline void vllm_syncwarp()")
    body = src[i:i + 800]
    assert body.index('__ATOMIC_RELEASE, "wavefront"') < \
        body.index("__builtin_amdgcn_wave_barrier()") < \
        body.index('__ATOMIC_ACQUIRE, "wavefront"')
    # no masks/math changed: FINAL_MASK values untouched
    assert "#define FINAL_MASK 0xffffffffffffffffULL" in src
    assert "#define FINAL_MASK 0xffffffff\n" in src
