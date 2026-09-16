# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Focused HOST C++ call-ROUTING tests for the GPTQ atomicAdd dispatchers.

SYNTHETIC call-resolution coverage — NOT GPU arithmetic accuracy. The
REAL csrc/libtorch_stable/quantization/gptq/compat.cuh is compiled on
the HOST with preprocessor scaffolding (synthetic half/half2 types,
__device__ neutralized, stub intrinsics) and synthetic SDK NATIVE
atomicAdd overloads injected into the global namespace. Three profiles:

  1. old-HIP (USE_ROCM, HIP 6.x) WITH synthetic native overloads
     present: gptq_atomicAdd must select the CAS FALLBACK (the
     intended existing choice) — proving unambiguity even when the SDK
     backports native half/half2 atomicAdd.
  2. modern HIP (USE_ROCM, HIP >= 7.13): native ::atomicAdd selected.
  3. modern CUDA (__CUDA_ARCH__ >= 700, no USE_ROCM): native selected
     (half2 requires >= 600; both native at 700).

Each profile is a separate real clang++ compilation and execution.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
COMPAT = ROOT / "csrc" / "libtorch_stable" / "quantization" / "gptq" / "compat.cuh"
CXX = shutil.which("clang++") or shutil.which("g++")

requires_cxx = pytest.mark.skipif(CXX is None, reason="no host C++ compiler")

SCAFFOLD = """
// ---- synthetic host scaffolding (NOT an SDK copy) ----
#include <cstdint>
#include <cstdio>
struct half {
    unsigned short x;
    operator unsigned short() const { return x; }
};
struct half2 { unsigned int x; };
struct __half_raw {
    unsigned short x;
    __half_raw() = default;
    explicit __half_raw(unsigned short v) : x(v) {}
    explicit __half_raw(const half& h);
};
inline __half_raw::__half_raw(const half& h) : x(h.x) {}
static inline half __hadd(half a, half b) { return a; }
static inline half __hadd(__half_raw a, half b) {
    half r; r.x = a.x; return r;
}
static inline half2 __hadd2(half2 a, half2 b) { return a; }
static inline unsigned int atomicCAS(unsigned int* p, unsigned int cmp,
                                      unsigned int val) {
    FALLBACK_TRACE();
    return val;
}
#define __device__
#define __forceinline__ inline
// synthetic SDK NATIVE overloads — ALWAYS DECLARED (both profiles) to
// prove the old-HIP dispatcher still picks the fallback despite them
void atomicAdd(half* address, half val) { NATIVE_TRACE("half"); }
void atomicAdd(half2* address, half2 val) { NATIVE_TRACE("half2"); }
"""

MAIN_TMPL = """
int main() {
    ::half h{};
    ::half2 h2{};
    vllm::gptq::gptq_atomicAdd(&h, h);
    vllm::gptq::gptq_atomicAdd(&h2, h2);
    printf("ROUTING_DONE\\n");
    return 0;
}
"""


def _build_and_run(tmp_path: Path, profile_defines: list[str],
                   expect_half: str, expect_half2: str) -> None:
    """expect_half/expect_half2 are 'FALLBACK' or 'NATIVE' — the routing
    the ORIGINAL guards imply for that profile, per type."""
    src = tmp_path / "tu.cpp"
    body = SCAFFOLD
    body = body.replace("FALLBACK_TRACE();", 'printf("FALLBACK\\n");')
    body = body.replace('NATIVE_TRACE("half");', 'printf("NATIVE\\n");')
    body = body.replace('NATIVE_TRACE("half2");', 'printf("NATIVE\\n");')
    src.write_text(
        "#include \"" + str(COMPAT) + "\"\n" if False else
        body + "\n#include \"" + str(COMPAT) + "\"\n" + MAIN_TMPL
    )
    exe = tmp_path / "tu"
    r = subprocess.run(
        [CXX, "-std=c++17", *profile_defines, str(src), "-o", str(exe)],
        capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-1500:]
    r2 = subprocess.run([str(exe)], capture_output=True, text=True,
                        timeout=60)
    assert r2.returncode == 0, r2.stderr
    lines = [l for l in r2.stdout.splitlines() if l.strip()]
    assert lines == [expect_half, expect_half2, "ROUTING_DONE"], lines


@requires_cxx
def test_old_hip_with_native_backport_selects_fallback(tmp_path):
    """Older-HIP profile (USE_ROCM, HIP 6.2) with synthetic SDK native
    overloads PRESENT: the intended existing CAS fallback is selected —
    unambiguous despite the backport."""
    _build_and_run(
        tmp_path,
        ["-DUSE_ROCM", "-DHIP_VERSION_MAJOR=6", "-DHIP_VERSION_MINOR=2"],
        expect_half="FALLBACK", expect_half2="FALLBACK")


@requires_cxx
def test_modern_hip_selects_native(tmp_path):
    """Modern HIP (>= 7.13): native ::atomicAdd selected."""
    _build_and_run(
        tmp_path,
        ["-DUSE_ROCM", "-DHIP_VERSION_MAJOR=7", "-DHIP_VERSION_MINOR=13"],
        expect_half="NATIVE", expect_half2="NATIVE")


@requires_cxx
def test_modern_cuda_selects_native(tmp_path):
    """Modern CUDA (CC >= 700): native selected (both half and half2
    are native at 700)."""
    _build_and_run(
        tmp_path,
        ["-D__CUDA_ARCH__=700"],
        expect_half="NATIVE", expect_half2="NATIVE")


@requires_cxx
def test_old_cuda_cc_selects_fallback(tmp_path):
    """Old CUDA (CC < 700, no USE_ROCM): the original guard still
    selects the CAS fallback for half (CC<700) — preserving existing
    CUDA behavior."""
    _build_and_run(
        tmp_path,
        # original guards: half falls back below CC 700; half2 falls
        # back only below CC 600 — at 600, half2 is native
        ["-D__CUDA_ARCH__=600"],
        expect_half="FALLBACK", expect_half2="NATIVE")


def test_callsites_all_use_named_dispatcher():
    """Source contract: q_gemm.cu contains NO unqualified atomicAdd(
    callsites — every GPTQ atomic goes through the named dispatcher."""
    import re
    src = (ROOT / "csrc/libtorch_stable/quantization/gptq/q_gemm.cu").read_text()
    unqualified = re.findall(r'(?<![::\w])atomicAdd\(', src)
    assert not unqualified, unqualified
    assert src.count("gptq_atomicAdd(") == 10


def test_fallback_bodies_preserved():
    """The CAS fallback bodies are preserved verbatim in structure:
    atomicCAS loops with __hadd/__hadd2."""
    src = COMPAT.read_text()
    assert "atomicCAS(address_as_ui, assumed, old)" in src
    assert "atomicCAS(address_as_ui, assumed, *((unsigned int*)&new_val))" in src
    assert "__hadd(hsum, val)" in src
    assert "__hadd2(old_val, val)" in src
    # no unqualified `void atomicAdd(` overload remains in the namespace
    # (only the named dispatchers + _half helpers)
    assert "void atomicAdd(" not in src
