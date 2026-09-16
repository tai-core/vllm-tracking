# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""REAL CMake test for the HCU_MACRO_HEADER cache declaration + guards.

The declaration/guard block is EXTRACTED VERBATIM from the project's
CMakeLists.txt (single source of truth — no corrected copy in a
fixture) and executed by a real `cmake` configure in three scenarios:

  1. fresh default          -> HCU_MACRO_HEADER is an empty STRING cache
                               entry; no guard fires (exit 0).
  2. explicit -D + profile  -> the exact -D input is RETAINED and the
                               cache TYPE is STRING.
  3. nonempty + profile OFF -> the inconsistent-combination guard
                               rejects (nonzero exit, FATAL_ERROR text).

Requires a real cmake binary (skips otherwise). --noconftest friendly.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CMAKELISTS = ROOT / "CMakeLists.txt"
CMAKE = shutil.which("cmake") or str(Path(__import__("sys").prefix) / "bin" / "cmake")

requires_cmake = pytest.mark.skipif(
    not Path(CMAKE).exists(), reason="cmake not available")


def _extract_block() -> str:
    """Slice the declaration + guard block VERBATIM from CMakeLists.txt:
    from the `set(HCU_MACRO_HEADER ...)` line through the closing
    `endif()` of the path-validation guard."""
    s = CMAKELISTS.read_text()
    i = s.index("set(HCU_MACRO_HEADER")
    # balanced extraction: track if/endif pairs from the outer
    # if(HCU_MACRO_HEADER) to its matching endif()
    import re as _re
    depth = 0
    pos = s.index("if(HCU_MACRO_HEADER)", i)
    for m in _re.finditer(r'^\s*(if|endif)\(', s[pos:], _re.M):
        kw = m.group(1)
        if kw == "if":
            depth += 1
        else:
            depth -= 1
            if depth == 0:
                end = pos + m.end()
                end = s.index(")", end) + 1
                return s[i:end]
    raise AssertionError("unbalanced if/endif in the extracted block")


def _configure(tmp_path: Path, extra_args: list[str]) -> subprocess.CompletedProcess:
    proj = tmp_path / "proj"
    proj.mkdir(parents=True)
    (proj / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.25)\n"
        "project(hcu_macro_header_decl NONE)\n"
        'option(VLLM_HCU_EXPERIMENTAL_GFX936 "experimental profile" OFF)\n'
        + _extract_block()
        + "\nmessage(STATUS \"HCU_MACRO_HEADER=[${HCU_MACRO_HEADER}]\")\n"
    )
    build = tmp_path / "build"
    return subprocess.run(
        [CMAKE, "-S", str(proj), "-B", str(build)] + extra_args,
        capture_output=True, text=True, timeout=300)


def _cache_entry(build: Path, name: str):
    """Parse the CMakeCache.txt entry for NAME -> (type, value)."""
    for line in (build / "CMakeCache.txt").read_text().splitlines():
        if line.startswith(f"{name}:"):
            lhs, value = line.split("=", 1)
            _, typ = lhs.split(":", 1)
            return typ, value
    return None, None


@requires_cmake
def test_fresh_default_empty_string_cache(tmp_path):
    r = _configure(tmp_path, [])
    assert r.returncode == 0, r.stderr[-800:]
    typ, value = _cache_entry(tmp_path / "build", "HCU_MACRO_HEADER")
    assert typ == "STRING", f"cache type is {typ!r}, expected STRING"
    assert value == "", f"default value is {value!r}, expected empty"


@requires_cmake
def test_explicit_dash_d_retained_as_string(tmp_path):
    r = _configure(
        tmp_path,
        ["-DHCU_MACRO_HEADER=synthetic/vendor.h",
         "-DVLLM_HCU_EXPERIMENTAL_GFX936=ON"])
    assert r.returncode == 0, r.stderr[-800:]
    typ, value = _cache_entry(tmp_path / "build", "HCU_MACRO_HEADER")
    assert typ == "STRING", f"cache type is {typ!r}, expected STRING"
    assert value == "synthetic/vendor.h", (
        f"-D input not retained: {value!r}")
    # the declaration did not clobber the explicit input with the
    # docstring/list content
    assert "EXPERIMENTAL" not in value


@requires_cmake
def test_inconsistent_nonempty_profile_off_rejects(tmp_path):
    r = _configure(tmp_path, ["-DHCU_MACRO_HEADER=synthetic/vendor.h"])
    assert r.returncode != 0, "inconsistent combination must reject"
    combined = r.stdout + r.stderr
    assert "requires the experimental HCU HIP profile" in combined
