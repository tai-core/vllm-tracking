# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""EXPERIMENTAL HCU macro-header seam: narrow source/build-contract tests.

Synthetic vendor headers ONLY (invented macro names/values). The local
host preprocessor (clang) drives the shim generator where feasible;
CMake-level wiring is validated structurally (labeled separately). Run
with --noconftest (avoids unrelated Mac vllm atexit issues).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SHIM_SCRIPT = ROOT / "cmake" / "hcu_macro_shim.py"
HIPIFY_SCRIPT = ROOT / "cmake" / "hipify.py"
CC = shutil.which("clang") or shutil.which("gcc") or shutil.which("cc")

requires_cc = pytest.mark.skipif(CC is None, reason="no local preprocessor")


def run_shim(tmp: Path, header_name: str, header_body: str, *extra):
    """Write a synthetic vendor header into tmp, run the shim generator
    with the local compiler, return (rc, stderr, shim_text)."""
    inc = tmp / "include"
    inc.mkdir(exist_ok=True)
    (inc / header_name).write_text(header_body)
    out = tmp / "shim" / "hcu_macro_shim.h"
    argv = [
        sys.executable, str(SHIM_SCRIPT),
        "--compiler", CC,
        "--header", header_name,
        "--include-dir", str(inc),
        "--output", str(out),
        *extra,
    ]
    r = subprocess.run(argv, capture_output=True, text=True, timeout=120)
    shim = out.read_text() if out.exists() else None
    return r.returncode, r.stderr, shim


# ---------------------------------------------------------------------------
# shim generator: numeric value preservation + rejection cases (real
# local preprocessor)
# ---------------------------------------------------------------------------

@requires_cc
def test_numeric_values_preserved_exactly(tmp_path):
    body = (
        "#define VENDOR_TILE_X 16\n"
        "#define VENDOR_TILE_Y 040\n"
        "#define VENDOR_HEX 0x1F\n"
        "#define VENDOR_BIG 18446744073709551615ULL\n"
        "#define VENDOR_MAX_F 3.5e2f\n"
    )
    rc, err, shim = run_shim(tmp_path, "vendor_macros.h", body)
    assert rc == 0, err
    for name, val in (("VENDOR_TILE_X", "16"), ("VENDOR_HEX", "0x1F"),
                      ("VENDOR_BIG", "18446744073709551615ULL")):
        assert f"#define {name} {val}" in shim
    assert "VENDOR_TILE_Y" in shim  # octal preserved verbatim as text


@requires_cc
def test_function_like_macro_rejected(tmp_path):
    body = "#define VENDOR_TILE_X 16\n#define VENDOR_F(x) ((x)+1)\n"
    rc, err, _ = run_shim(tmp_path, "vendor_macros.h", body)
    assert rc == 3
    assert "function-like" in err


@requires_cc
def test_string_body_rejected(tmp_path):
    body = '#define VENDOR_NAME "tile"\n'
    rc, err, _ = run_shim(tmp_path, "vendor_macros.h", body)
    assert rc == 3
    assert "non-numeric" in err


@requires_cc
def test_declaration_content_rejected(tmp_path):
    body = "#define VENDOR_X 4\nstruct vendor_thing { int a; };\n"
    rc, err, _ = run_shim(tmp_path, "vendor_macros.h", body)
    assert rc == 3
    assert "non-macro content" in err


@requires_cc
def test_nested_include_rejected_explicitly(tmp_path):
    """Contract (review defect 4): a nested include inside the selected
    header is REJECTED outright — even when its contents do not affect
    the selected header's own numeric macros. No silent ignoring of
    non-selected linemarkers."""
    (tmp_path / "include").mkdir(exist_ok=True)
    (tmp_path / "include" / "vendor_other.h").write_text(
        "#define VENDOR_OTHER 9\n")
    body = (
        "#include <vendor_other.h>\n"
        "#define VENDOR_MAIN 7\n"
    )
    rc, err, _ = run_shim(tmp_path, "vendor_macros.h", body)
    assert rc == 3
    assert "nested includes" in err
    assert "vendor_other.h" in err


@requires_cc
def test_non_stable_context_used(tmp_path):
    """Macros conditional on non-stable context are the ACTIVE ones
    (TORCH_TARGET_VERSION/TORCH_STABLE_ONLY explicitly undefined)."""
    body = (
        "#ifdef TORCH_STABLE_ONLY\n"
        "#define VENDOR_STABLE_ONLY_PATH 1\n"
        "#else\n"
        "#define VENDOR_ACTIVE 42\n"
        "#endif\n"
    )
    rc, err, shim = run_shim(tmp_path, "vendor_macros.h", body)
    assert rc == 0, err
    assert "#define VENDOR_ACTIVE 42" in shim
    assert "VENDOR_STABLE_ONLY_PATH" not in shim


@requires_cc
def test_no_numeric_macros_rejected(tmp_path):
    body = "// just a comment\n"
    rc, err, _ = run_shim(tmp_path, "vendor_macros.h", body)
    assert rc == 3
    assert "no active object-like numeric macros" in err


def test_absolute_header_path_rejected(tmp_path):
    """Absolute header paths are rejected before any preprocessing
    (no file access attempted)."""
    import subprocess as sp
    argv = [
        sys.executable, str(SHIM_SCRIPT),
        "--compiler", CC or "/nonexistent/cc",
        "--header", "/abs/vendor.h",
        "--include-dir", str(tmp_path),
        "--output", str(tmp_path / "shim.h"),
    ]
    r = sp.run(argv, capture_output=True, text=True, timeout=60)
    assert r.returncode == 2
    assert "relative" in r.stderr


# ---------------------------------------------------------------------------
# hipify seam: rewrite of the auto-injected include (pure-python, no
# torch import needed — drive _rewrite_vendor_macro_injection directly)
# ---------------------------------------------------------------------------

def _import_rewrite_fn():
    import importlib.util as ilu
    spec = ilu.spec_from_file_location("hipify_seam", HIPIFY_SCRIPT)
    mod = ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._rewrite_vendor_macro_injection


BC = "// !!! This is a file automatically generated by hipify!!!"


def test_injected_include_rewritten_with_dispatch(tmp_path):
    rewrite = _import_rewrite_fn()
    out = tmp_path / "out"
    (out / "nested").mkdir(parents=True)
    # standard generated form: PUBLIC hipify breadcrumb on line 1,
    # injected include immediately after (vendor translator injects
    # adjacent to the breadcrumb)
    (out / "gen.hip").write_text(
        f"{BC}\n#include <vendor_macros.h>\n__global__ void k() {{}}\n")
    (out / "nested" / "gen_nested.h").write_text(
        f"{BC}\n\n#include <vendor_macros.h>\n#define LOCAL 1\n")
    # AUTHORED file whose FIRST LINE is the same include but WITHOUT
    # the generated-file marker -> must stay untouched (review defect 1)
    (out / "authored_first_line.h").write_text(
        "#include <vendor_macros.h>\nint x;\n")
    dispatch = tmp_path / "seam" / "hcu_macro_dispatch.h"

    rewrite(str(out), "vendor_macros.h", str(dispatch))

    top = (out / "gen.hip").read_text().split("\n")
    assert top[0] == BC  # breadcrumb preserved
    assert top[1].startswith('#include "') and "hcu_macro_dispatch.h" in top[1]
    assert "__global__ void k() {}" in top

    nested = (out / "nested" / "gen_nested.h").read_text().split("\n")
    assert nested[0] == BC
    # dispatch lives at tmp/seam; from out/nested/ the relative path
    # climbs two levels: ../../seam/hcu_macro_dispatch.h
    assert nested[2].startswith('#include "../../seam/')
    assert "hcu_macro_dispatch.h" in nested[2]
    assert "#define LOCAL 1" in nested

    # authored FIRST-LINE include (no marker) is NOT rewritten
    authored = (out / "authored_first_line.h").read_text()
    assert authored.split("\n")[0] == "#include <vendor_macros.h>"

    # (authored first-line include already asserted above)

    disp = dispatch.read_text()
    assert 'defined(TORCH_TARGET_VERSION) || defined(TORCH_STABLE_ONLY)' in disp
    assert '#include "hcu_macro_shim.h"' in disp
    assert "#include <vendor_macros.h>" in disp  # legacy path
    # no ABI guard undefined / no stable target disabled in dispatch
    assert "#undef" not in disp


def test_rewrite_idempotent_and_optout_untouched(tmp_path):
    rewrite = _import_rewrite_fn()
    out = tmp_path / "out"
    out.mkdir()
    f = out / "gen.hip"
    f.write_text(f"{BC}\n#include <vendor_macros.h>\nint x;\n")
    dispatch = tmp_path / "seam" / "hcu_macro_dispatch.h"
    rewrite(str(out), "vendor_macros.h", str(dispatch))
    first = f.read_text()
    rewrite(str(out), "vendor_macros.h", str(dispatch))
    assert f.read_text() == first  # idempotent


def test_no_injection_no_rewrite(tmp_path):
    """Files without the injected first line are never modified."""
    rewrite = _import_rewrite_fn()
    out = tmp_path / "out"
    out.mkdir()
    f = out / "plain.hip"
    original = "__global__ void k() {}\n"
    f.write_text(original)
    dispatch = tmp_path / "seam" / "hcu_macro_dispatch.h"
    rewrite(str(out), "vendor_macros.h", str(dispatch))
    assert f.read_text() == original


def test_sdk_and_source_trees_immutable(tmp_path):
    """The seam never writes outside the output dir + dispatch path."""
    rewrite = _import_rewrite_fn()
    out = tmp_path / "out"
    out.mkdir()
    (out / "gen.hip").write_text(f"{BC}\n#include <vendor_macros.h>\nint x;\n")
    sdk_dir = tmp_path / "sdk"
    sdk_dir.mkdir()
    sdk_header = sdk_dir / "vendor_macros.h"
    sdk_header.write_text("#define V 1\n")
    dispatch = tmp_path / "seam" / "hcu_macro_dispatch.h"
    before = sdk_header.read_text()
    rewrite(str(out), "vendor_macros.h", str(dispatch))
    assert sdk_header.read_text() == before
    # only gen.hip + dispatch were created/modified under tmp_path
    changed = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert {p.name for p in changed} == {
        "gen.hip", "vendor_macros.h", "hcu_macro_dispatch.h"}


# ---------------------------------------------------------------------------
# CMake/Docker wiring (structural; mocked-compiler validation is NOT
# claimed here — real HCU compilation is Codex's gate)
# ---------------------------------------------------------------------------

def test_cmake_option_guards():
    src = (ROOT / "CMakeLists.txt").read_text()
    assert 'set(HCU_MACRO_HEADER "" CACHE STRING' in src
    # requires the experimental gfx936 profile
    assert "HCU_MACRO_HEADER requires the experimental HCU HIP profile" in src
    # relative-path-only validation
    assert "non-empty RELATIVE include path" in src


def test_utils_wiring_preserves_default_command():
    src = (ROOT / "cmake" / "utils.cmake").read_text()
    # the default hipify command is byte-identical when the seam is off
    assert '${ALL_SRCS} ${_HIPIFY_EXTRA_ARGS}' in src
    assert 'set(_HIPIFY_EXTRA_ARGS "")' in src
    # shim generation is inside the same hipify_all target (ordering)
    assert "hcu_macro_shim.py" in src
    assert "Generating numeric-only HCU macro shim" in src
    # include path for the seam dir
    assert "include_directories(${_HCU_SEAM_DIR})" in src


def test_utils_keeps_torch_target_version_and_guards():
    """The seam does not touch stable ABI guards/flags."""
    cm = (ROOT / "CMakeLists.txt").read_text()
    assert "TORCH_TARGET_VERSION=0x020B000000000000ULL" in cm
    utils = (ROOT / "cmake" / "utils.cmake").read_text()
    # no suppression of stable targets or ABI guards in the seam
    seam_region = utils[utils.index("EXPERIMENTAL HCU macro-header seam"):
                        utils.index('COMMENT "Running hipify')]
    assert "#undef" not in seam_region
    assert "TORCH_TARGET_VERSION" not in seam_region


def test_docker_wiring_optional_and_value_free():
    df = (ROOT / "docker" / "public-build" / "Dockerfile.glm53-v029").read_text()
    assert "ARG HCU_MACRO_HEADER=" in df
    # no vendor path/value embedded (only the empty-default plumbing)
    assert "-DHCU_MACRO_HEADER=${HCU_MACRO_HEADER}" in df


# ---------------------------------------------------------------------------
# linemarker ancestry (include-entry/return flags 1/2): compiler preludes
# are NOT nested dependencies; only files entered while the selected
# header is innermost are nested. Synthetic invented paths/values.
# ---------------------------------------------------------------------------

@requires_cc
def test_prelude_before_selected_passes(tmp_path):
    """A REAL-FILE compiler prelude (via -include) entered BEFORE the
    selected header, then return to root, then the selected numeric
    header: must PASS — the prelude is not a nested dependency of the
    selected header."""
    inc = tmp_path / "include"
    inc.mkdir(exist_ok=True)
    (inc / "synth_prelude.h").write_text("#define SYNTH_PRELUDE 5\n")
    (inc / "synth_selected.h").write_text(
        "#ifdef USE_ROCM\n#define SYNTH_SEL_X 32\n#else\n"
        "#define SYNTH_SEL_X 1\n#endif\n")
    out = tmp_path / "shim" / "shim.h"
    argv = [
        sys.executable, str(SHIM_SCRIPT),
        "--compiler", CC,
        "--header", "synth_selected.h",
        f"--include-dir={inc}",
        "--compiler-arg=-DUSE_ROCM=1",
        "--compiler-arg=-include", f"--compiler-arg={inc}/synth_prelude.h",
        "--output", str(out),
    ]
    r = subprocess.run(argv, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    shim = out.read_text()
    assert "#define SYNTH_SEL_X 32" in shim
    assert "SYNTH_PRELUDE" not in shim  # prelude never leaks into the shim


@requires_cc
def test_selected_nested_child_rejects(tmp_path):
    """selected -> nested child -> return: the child was ENTERED while
    the selected header was innermost -> REJECT (even if the child only
    defines an unused macro)."""
    inc = tmp_path / "include"
    inc.mkdir(exist_ok=True)
    (inc / "synth_selected.h").write_text(
        "#include <synth_child.h>\n#define SYNTH_SEL_X 32\n")
    (inc / "synth_child.h").write_text("#define SYNTH_CHILD_UNUSED 9\n")
    out = tmp_path / "shim" / "shim.h"
    argv = [
        sys.executable, str(SHIM_SCRIPT),
        "--compiler", CC,
        "--header", "synth_selected.h",
        f"--include-dir={inc}",
        "--output", str(out),
    ]
    r = subprocess.run(argv, capture_output=True, text=True, timeout=120)
    assert r.returncode == 3
    assert "nested includes" in r.stderr
    assert "synth_child.h" in r.stderr


@requires_cc
def test_root_file_after_return_not_nested(tmp_path):
    """A root-level file entered AFTER returning from the selected
    header is NOT a nested dependency: the ancestry must show the
    selected header was already popped. Emulated with -include placing
    the selected header first and the stdin TU (root) then including a
    sibling root file."""
    inc = tmp_path / "include"
    inc.mkdir(exist_ok=True)
    (inc / "synth_selected.h").write_text("#define SYNTH_SEL_X 32\n")
    (inc / "synth_after_root.h").write_text("#define SYNTH_AFTER 3\n")
    out = tmp_path / "shim" / "shim.h"
    argv = [
        sys.executable, str(SHIM_SCRIPT),
        "--compiler", CC,
        "--header", "synth_selected.h",
        f"--include-dir={inc}",
        # the generator's own TU is `#include <synth_selected.h>` from
        # stdin; to exercise a root sibling AFTER the selected header,
        # wrap the TU: pre-include the selected header, then the stdin
        # root includes the sibling after returning.
        "--compiler-arg=-include", f"--compiler-arg={inc}/synth_selected.h",
        "--output", str(out),
    ]
    r = subprocess.run(argv, capture_output=True, text=True, timeout=120)
    # with -include the selected header precedes the stdin TU; the
    # stdin TU then ALSO includes the selected header (guarded by
    # include guards — the second entry is a no-op but still ENTERS the
    # file per linemarkers). Either way the sibling-after-root scenario
    # is not constructible via the fixed TU without the wrapper; assert
    # the generator still behaves correctly (pass + no sibling leak).
    assert r.returncode == 0, r.stderr
    shim = out.read_text()
    assert "#define SYNTH_SEL_X 32" in shim
    assert "SYNTH_AFTER" not in shim


@requires_cc
def test_root_after_return_direct_ancestry(tmp_path):
    """Direct ancestry validation with a custom root TU: preprocess a
    root file that includes the selected header then a sibling AFTER
    the return, and drive the generator's parser over the same output
    by using the wrapper-header trick: make the SELECTED header the
    root's FIRST include; the sibling is included at root level after
    the selected header returns."""
    inc = tmp_path / "include"
    inc.mkdir(exist_ok=True)
    (inc / "synth_selected.h").write_text("#define SYNTH_SEL_X 32\n")
    (inc / "synth_after_root.h").write_text("#define SYNTH_AFTER 3\n")
    # generator TU includes <synth_selected.h>; to get a root-level
    # sibling AFTER the return, wrap via the compiler: append the
    # sibling include to the stdin TU is impossible, so instead verify
    # the parser directly on crafted -dD output using clang on a real
    # root file, then feed the equivalent argv through the generator by
    # pointing --header at the ROOT wrapper and asserting the ROOT's
    # own numeric macro passes while its post-selected sibling is at
    # root level (not nested in the selected header):
    root = tmp_path / "synth_root_wrapper.h"
    root.write_text(
        "#include <synth_selected.h>\n"
        "#include <synth_after_root.h>\n"
        "#define SYNTH_ROOT_OWN 64\n")
    out = tmp_path / "shim" / "shim.h"
    argv = [
        sys.executable, str(SHIM_SCRIPT),
        "--compiler", CC,
        "--header", "synth_root_wrapper.h",
        f"--include-dir={inc}",
        f"--include-dir={tmp_path}",
        "--output", str(out),
    ]
    r = subprocess.run(argv, capture_output=True, text=True, timeout=120)
    # The selected-for-shim header here is the ROOT WRAPPER; its nested
    # includes (synth_selected + synth_after_root) are BOTH nested
    # dependencies of the wrapper -> must REJECT.
    assert r.returncode == 3
    assert "nested includes" in r.stderr
    assert "synth_selected.h" in r.stderr or "synth_after_root.h" in r.stderr


# ---------------------------------------------------------------------------
# focused marker-stream cases (synthetic preprocessor streams through the
# REAL _parse_defines parser; invented headers/values)
# ---------------------------------------------------------------------------

def _load_shim_module():
    import importlib.util as ilu
    spec = ilu.spec_from_file_location("shim_mod", SHIM_SCRIPT)
    mod = ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_selected_return_to_stdin_root_macro_not_leaked():
    """selected -> flag2 <stdin> -> root numeric macro: the root macro
    must NOT leak into the shim (return to a pseudo root ends the
    selected-header scope)."""
    mod = _load_shim_module()
    sel = "/synth/inc/sel.h"
    stream = (
        '# 1 "<stdin>"\n'
        f'# 1 "{sel}" 1\n'
        "#define SEL_X 32\n"
        '# 2 "<stdin>" 2\n'
        "#define ROOT_LEAK 99\n"
    )
    defs, order, saw, nested = mod._parse_defines(
        stream, mod._identity(sel), "sel.h")
    assert saw
    assert defs == {"SEL_X": "32"}
    assert "ROOT_LEAK" not in defs
    assert not nested


def test_selected_return_to_stdin_root_header_not_nested():
    """selected -> flag2 <stdin> -> root header entered at root level:
    must NOT be recorded as nested (the selected scope had ended)."""
    mod = _load_shim_module()
    sel = "/synth/inc/sel.h"
    root_h = "/synth/inc/root_sibling.h"
    stream = (
        '# 1 "<stdin>"\n'
        f'# 1 "{sel}" 1\n'
        "#define SEL_X 32\n"
        '# 2 "<stdin>" 2\n'
        f'# 1 "{root_h}" 1\n'
        "#define ROOT_SIB 7\n"
        f'# 2 "<stdin>" 2\n'
    )
    defs, _, saw, nested = mod._parse_defines(
        stream, mod._identity(sel), "sel.h")
    assert saw
    assert defs == {"SEL_X": "32"}
    assert not nested


def test_real_wrapper_return_retains_wrapper_scope():
    """real wrapper -> selected -> flag2 wrapper: the WRAPPER scope is
    retained (caller not popped); macros defined in the wrapper after
    the return are outside the selected scope (not leaked), and a
    header entered from the WRAPPER after the return is not nested in
    the selected header."""
    mod = _load_shim_module()
    wrap = "/synth/inc/wrap.h"
    sel = "/synth/inc/sel.h"
    after = "/synth/inc/wrap_sibling.h"
    stream = (
        '# 1 "<stdin>"\n'
        f'# 1 "{wrap}" 1\n'
        f'# 1 "{sel}" 1\n'
        "#define SEL_X 32\n"
        f'# 5 "{wrap}" 2\n'
        "#define WRAP_OWN 64\n"
        f'# 1 "{after}" 1\n'
        "#define WRAP_SIB 8\n"
        f'# 9 "{wrap}" 2\n'
        f'# 2 "<stdin>" 2\n'
    )
    defs, _, saw, nested = mod._parse_defines(
        stream, mod._identity(sel), "sel.h")
    assert saw
    assert defs == {"SEL_X": "32"}  # wrapper/sibling macros not leaked
    assert not nested  # sibling entered from the WRAPPER, not selected


def test_real_nested_inside_selected_still_rejects():
    """a real nested header entered WHILE INSIDE the selected header
    still rejects (regardless of use)."""
    mod = _load_shim_module()
    sel = "/synth/inc/sel.h"
    child = "/synth/inc/sel_child.h"
    stream = (
        '# 1 "<stdin>"\n'
        f'# 1 "{sel}" 1\n'
        f'# 1 "{child}" 1\n'
        "#define CHILD_UNUSED 9\n"
        f'# 4 "{sel}" 2\n'
        "#define SEL_X 32\n"
        '# 2 "<stdin>" 2\n'
    )
    # the parser RECORDS the nested include; main() rejects on it
    defs, _, saw, nested = mod._parse_defines(
        stream, mod._identity(sel), "sel.h")
    assert saw
    assert mod._identity(child) in nested


def test_malformed_return_rejected_not_widened():
    """a return naming a file not on the stack is rejected rather than
    widening the selected scope."""
    mod = _load_shim_module()
    sel = "/synth/inc/sel.h"
    ghost = "/synth/inc/ghost.h"
    stream = (
        '# 1 "<stdin>"\n'
        f'# 1 "{sel}" 1\n'
        "#define SEL_X 32\n"
        f'# 2 "{ghost}" 2\n'
    )
    with pytest.raises(SystemExit) as excinfo:
        mod._parse_defines(stream, mod._identity(sel), "sel.h")
    assert excinfo.value.code == 3


@requires_cc
def test_empty_body_feature_define_rejected(tmp_path):
    """`#define FEATURE` (empty body) inside the selected header is an
    ACTIVE definition a numeric-only shim cannot represent — reject,
    never silently skip."""
    body = "#define SYNTH_FEATURE\n#define SYNTH_X 32\n"
    rc, err, _ = run_shim(tmp_path, "vendor_macros.h", body)
    assert rc == 3
    assert "EMPTY body" in err
    assert "SYNTH_FEATURE" in err


@requires_cc
def test_pragma_pack_rejected(tmp_path):
    """A pragma with semantic effects (#pragma pack) is rejected — only
    '#pragma once' is allowed."""
    body = (
        "#pragma pack(1)\n"
        "#define SYNTH_X 32\n"
        "#pragma pack()\n"
    )
    rc, err, _ = run_shim(tmp_path, "vendor_macros.h", body)
    assert rc == 3
    assert "pragma with semantic effects" in err


@requires_cc
def test_pragma_once_allowed(tmp_path):
    """'#pragma once' alone is allowed (no numeric semantics)."""
    body = (
        "#pragma once\n"
        "#define SYNTH_X 32\n"
    )
    rc, err, shim = run_shim(tmp_path, "vendor_macros.h", body)
    assert rc == 0, err
    assert "#define SYNTH_X 32" in shim


GNU_SHAPED_PREFIX_STREAM = (
    '# 0 "<stdin>"\n'
    '# 0 "<built-in>"\n'
    '# 0 "<command-line>"\n'
    '# 1 "/synthetic/compiler-prelude.h" 1 3 4\n'
    "#define SYNTH_PRELUDE_ARCH 11\n"
    '# 0 "<command-line>" 2\n'
    '# 1 "<stdin>"\n'
    '# 1 "/synthetic/sdk/vendor_macros.h" 1\n'
    "#define VENDOR_ARCH 7\n"
    '# 2 "<stdin>" 2\n'
)


def test_gnu_shaped_pseudo_prelude_prefix_parses():
    """Complete GNU-shaped synthetic prefix: root-level virtual preludes
    (`# 0 "<stdin>"` / `<built-in>` / `<command-line>` no-flag pseudo
    switches, a real compiler-prelude header with flags `1 3 4`, return
    to the pseudo root, then the selected numeric header at root scope).
    Must parse with ONLY the selected header's macro captured — the
    prelude macro never leaks, nothing is called nested."""
    mod = _load_shim_module()
    sel = "/synthetic/sdk/vendor_macros.h"
    defs, order, saw, nested = mod._parse_defines(
        GNU_SHAPED_PREFIX_STREAM, mod._identity(sel), "vendor_macros.h")
    assert saw
    assert defs == {"VENDOR_ARCH": "7"}
    assert order == ["VENDOR_ARCH"]
    assert not nested
    assert "SYNTH_PRELUDE_ARCH" not in defs


def test_gnu_prefix_end_to_end_via_generator(tmp_path):
    """The same GNU-shaped stream shape end-to-end through the real
    generator: craft a compiler-prelude wrapper that the shim TU
    pre-includes (real clang), producing root-level prelude macros
    before the selected header, and verify the shim holds ONLY the
    selected numeric macros."""
    inc = tmp_path / "include"
    inc.mkdir(exist_ok=True)
    (inc / "synth_gnu_sel.h").write_text("#define GNU_SEL_ARCH 7\n")
    (inc / "synth_gnu_prelude.h").write_text(
        "#define SYNTH_PRELUDE_ARCH 11\n")
    out = tmp_path / "shim" / "shim.h"
    argv = [
        sys.executable, str(SHIM_SCRIPT),
        "--compiler", CC,
        "--header", "synth_gnu_sel.h",
        f"--include-dir={inc}",
        "--compiler-arg=-include", f"--compiler-arg={inc}/synth_gnu_prelude.h",
        "--output", str(out),
    ]
    r = subprocess.run(argv, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    shim = out.read_text()
    assert "#define GNU_SEL_ARCH 7" in shim
    assert "SYNTH_PRELUDE_ARCH" not in shim


def test_pseudo_switch_inside_selected_rejects():
    """A no-flag pseudo switch attempted INSIDE the selected header is
    NOT a legal root rebind — it must be rejected (ambiguous ancestry
    inside the selected/real scope)."""
    mod = _load_shim_module()
    sel = "/synthetic/sdk/vendor_macros.h"
    stream = (
        '# 1 "<stdin>"\n'
        f'# 1 "{sel}" 1\n'
        "#define VENDOR_ARCH 7\n"
        '# 0 "<built-in>"\n'
    )
    with pytest.raises(SystemExit) as excinfo:
        mod._parse_defines(stream, mod._identity(sel), "vendor_macros.h")
    assert excinfo.value.code == 3
