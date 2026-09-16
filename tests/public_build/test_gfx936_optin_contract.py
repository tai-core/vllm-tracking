# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Build-contract tests for the EXPERIMENTAL HCU gfx936 opt-in.

Source-level assertions on CMakeLists.txt + the build recipe. No cmake,
GPU, or build execution required (structural + semantic checks).
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CMAKE = ROOT / "CMakeLists.txt"
DOCKERFILE = ROOT / "docker" / "public-build" / "Dockerfile.glm53-v029"
UTILS = ROOT / "cmake" / "utils.cmake"


def _cmake_src():
    return CMAKE.read_text()


def test_option_default_off():
    src = _cmake_src()
    m = re.search(
        r"option\(VLLM_HCU_EXPERIMENTAL_GFX936\s*\n?.*?OFF\)", src)
    assert m, "option must exist and default to OFF"
    assert 'OFF)' in m.group(0)


def test_default_arch_list_untouched():
    """The upstream HIP_SUPPORTED_ARCHS list is NOT modified — gfx936 is
    only APPENDED when the option is ON."""
    src = _cmake_src()
    base = re.search(
        r'set\(HIP_SUPPORTED_ARCHS "([^"]+)"\)', src).group(1)
    assert "gfx936" not in base
    assert base.startswith("gfx906;gfx908;gfx90a;gfx942")
    # append happens only inside the opt-in branch
    assert re.search(
        r"if\(VLLM_HCU_EXPERIMENTAL_GFX936\).*?"
        r"list\(APPEND HIP_SUPPORTED_ARCHS gfx936\)",
        src, re.S)


def test_compiler_probe_fails_closed():
    """The configure-time HIP compiler acceptance probe must FATAL_ERROR
    on rejection (no silent arch drop)."""
    src = _cmake_src()
    assert "--offload-arch=gfx936" in src
    # the opt-in block spans from its `if(` to the `list(APPEND ... gfx936)`
    start = src.index("if(VLLM_HCU_EXPERIMENTAL_GFX936)")
    end = src.index("list(APPEND HIP_SUPPORTED_ARCHS gfx936)")
    probe_block = src[start:end]
    assert "execute_process" in probe_block
    assert "FATAL_ERROR" in probe_block
    assert "Refusing to silently drop" in probe_block
    # the append comes AFTER the probe failure check
    assert "_gfx936_probe_result EQUAL 0" in probe_block


def test_no_relabeling():
    """gfx936 is never relabeled as gfx90a/gfx942 or any other arch
    (checked on non-comment code lines)."""
    src = _cmake_src()
    code_lines = [l for l in src.splitlines()
                  if not l.strip().startswith("#")]
    code = "\n".join(code_lines)
    assert "gfx936" in code  # the opt-in references it
    for pat in (r"gfx936[^)\n]{0,40}gfx90a", r"gfx936[^)\n]{0,40}gfx942",
                r"gfx90a[^)\n]{0,40}gfx936", r"gfx942[^)\n]{0,40}gfx936"):
        assert not re.search(pat, code), pat


def test_inconsistent_non_hip_rejected():
    src = _cmake_src()
    m = re.search(
        r'if\(VLLM_HCU_EXPERIMENTAL_GFX936 AND NOT VLLM_GPU_LANG STREQUAL "HIP"\)(.*?)endif\(\)',
        src, re.S)
    assert m and "FATAL_ERROR" in m.group(1)
    assert "inconsistent" in m.group(1)


def test_no_global_validation_disable():
    """override_gpu_arches and its empty-result FATAL_ERROR are
    untouched (global arch validation stays enabled)."""
    utils = UTILS.read_text()
    assert 'message(FATAL_ERROR' in utils
    assert "None of the detected ROCm architectures" in utils


def test_version_warnings_untouched():
    """The torch version warnings remain present."""
    src = _cmake_src()
    assert "expected for CUDA build, saw" in src
    assert "expected for ROCm build, saw" in src


def test_recipe_enables_optin_only_for_vllm_build():
    """The public HCU recipe passes the opt-in + PYTORCH_ROCM_ARCH=gfx936
    for the vLLM wheel build; the plugin build does not."""
    df = DOCKERFILE.read_text()
    vllm_block = re.search(
        r"--- vLLM:.*?--- vLLM-HCU plugin", df, re.S).group(0)
    assert "-DVLLM_HCU_EXPERIMENTAL_GFX936=ON" in vllm_block
    assert 'PYTORCH_ROCM_ARCH="gfx936"' in vllm_block
    plugin_block = re.search(
        r"--- vLLM-HCU plugin.*?--- synthetic", df, re.S).group(0)
    assert "VLLM_HCU_EXPERIMENTAL_GFX936" not in plugin_block


def test_optin_documents_experimental_status():
    src = _cmake_src()
    assert "EXPERIMENTAL" in src
    assert "No runtime support is implied" in src


def test_probe_source_self_contained():
    """The generated probe is a valid, self-contained HIP translation
    unit: <hip/hip_runtime.h> is included BEFORE the __global__ kernel
    (no reliance on incidental implicit declarations)."""
    src = _cmake_src()
    assert '#include <hip/hip_runtime.h>' in src
    i_write = src.index('file(WRITE "${_gfx936_probe_src}"')
    seg = src[i_write:i_write + 400]
    assert seg.index('#include <hip/hip_runtime.h>') < seg.index('__global__')
    # fail-closed + compiler path preserved
    assert "CMAKE_HIP_COMPILER" in seg or "CMAKE_HIP_COMPILER" in src
    assert "Refusing to silently drop" in src


def test_early_non_rocm_rejection_before_cpu_return():
    """The early guard (option ON + VLLM_TARGET_DEVICE != rocm →
    FATAL_ERROR) sits between the option definition and the
    cpu/other-target early-return, so cpu+opt-in is REJECTED instead
    of returning silently."""
    src = _cmake_src()
    i_opt = src.index('option(VLLM_HCU_EXPERIMENTAL_GFX936')
    i_guard = src.index(
        'if(VLLM_HCU_EXPERIMENTAL_GFX936 AND NOT VLLM_TARGET_DEVICE '
        'STREQUAL "rocm")')
    i_cpuret = src.index('VLLM_TARGET_DEVICE STREQUAL "cpu")')
    assert i_opt < i_guard < i_cpuret
    guard_block = src[i_guard:src.index('endif()', i_guard)]
    assert 'FATAL_ERROR' in guard_block
    # the later HIP-language consistency guard is retained as well
    assert ('VLLM_HCU_EXPERIMENTAL_GFX936 AND NOT VLLM_GPU_LANG '
            'STREQUAL "HIP"') in src
