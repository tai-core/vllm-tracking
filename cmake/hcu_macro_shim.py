#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""EXPERIMENTAL HCU build seam: generate a numeric-only macro shim.

A vendor translator may prepend a NON-STABLE SDK macro header to
generated HIP sources. Stable-libtorch TUs (TORCH_TARGET_VERSION or
TORCH_STABLE_ONLY defined) must not include that header; they instead
include a shim containing ONLY the selected ACTIVE object-like NUMERIC
macro definitions, with their exact preprocessed values.

This script runs on the BUILD MACHINE at configure/build time:
  1. Preprocess the selected installed SDK header in NON-STABLE context
     (TORCH_TARGET_VERSION / TORCH_STABLE_ONLY explicitly UNDEFINED)
     with explicit compiler argv (no shell string).
  2. Parse the preprocessed output: every line of the form
     `# <lineno> "<header>"` guards which definitions come from the
     SELECTED header; within that region only
     `#define NAME value` with a NUMERIC literal body is accepted.
  3. REJECT (exit 3, no guessing, no silent omission): function-like
     macros, declarations/types, nested includes pulled into the shim,
     string or function-like or non-numeric bodies, ambiguous or
     differing values for the same name, or any unsupported layout.

The generated shim is build-directory-only; SDK constants are generated
on the node and never sent to OMP nor versioned. No SDK header contents
are pasted into project source; only the header PATH is an input.

Usage:
  python cmake/hcu_macro_shim.py \
      --compiler <path> [--compiler-arg ARG]... \
      --header <relative-include-name> \
      --include-dir <dir> [...] \
      --output <shim-path>

Exit codes: 0 ok; 2 usage/config error; 3 rejection (unsupported
layout — see stderr).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

_DEFINE_RE = re.compile(r"^#define\s+([A-Za-z_][A-Za-z0-9_]*)\s+(.+)$")
_FUNClike_RE = re.compile(r"^#define\s+([A-Za-z_][A-Za-z0-9_]*)\(")
# numeric literal body: integer/float/hex (with optional suffixes),
# optionally signed; char literals rejected (not numeric)
_NUMERIC_RE = re.compile(
    r"^[+-]?(0[xX][0-9a-fA-F]+|[0-9]+\.?[0-9]*(?:[eE][+-]?[0-9]+)?"
    r"|0[bB][01]+)(?:[uUlLfF]*)$")


def _reject(reason: str) -> "None":
    print(f"[hcu-macro-shim] REJECT: {reason}", file=sys.stderr)
    raise SystemExit(3)


def _normalize(path: str) -> str:
    p = PurePosixPath(path)
    return str(p)

def _identity(path: str) -> str:
    # canonical identity: realpath (resolves symlinks such as the
    # macOS /var -> /private/var mismatch) with a normalized
    # posix form; falls back to the normalized path when the file
    # is not accessible.
    try:
        return str(Path(path).resolve()).replace(os.sep, "/")
    except OSError:
        return _normalize(path)


def _parse_defines(proc_stdout: str, resolved_selected: str,
                   header: str):
    """Parse preprocessed -dD output under the unified include-stack
    scope model (standard linemarker flag2 semantics).

    The stack tracks REAL files AND pseudo roots (<stdin>/<built-in>/
    <command line>) as identities, because a return to a pseudo root
    must END the selected-header scope:

    - flag set containing 1: ENTER this file (push). If the current
      top is the selected header and the entered file is not the
      selected header itself (and is a real file), the entered file
      is a NESTED include of the selected header — recorded for
      rejection, used or not.
    - flag set containing 2: RETURN TO this file — the include(s)
      that just ended are popped and the NAMED CALLER is RETAINED as
      the active file (the caller itself is never popped). Returning
      to a pseudo root therefore ends the selected-header scope.
    - no flags: a line-number/location reset. With an empty stack it
      is the main-file declaration (establishes the root identity);
      otherwise it must name the current top.

    Marker flag sets are parsed as sets (compilers may emit e.g.
    ``1 3`` or ``2 3``). Ambiguous or malformed transitions are
    REJECTED rather than widening the selected-header scope.

    Returns (defs, order, saw_selected, nested_files); raises
    SystemExit via _reject on malformed/unsupported constructs.
    """
    include_stack: list[str] = []
    active_file = None
    saw_selected = False
    nested_files: set[str] = set()
    defs: dict[str, str] = {}
    order: list[str] = []

    def _is_pseudo(identity: str) -> bool:
        return identity.startswith("<") and identity.endswith(">")

    def _marker_identity(raw: str) -> str:
        # pseudo roots keep their literal angle-bracket identity; real
        # files use the resolved identity.
        if raw.startswith("<") and raw.endswith(">"):
            return raw
        return _identity(raw)

    for line in proc_stdout.splitlines():
        lm = re.match(r'^#\s+(\d+)\s+"([^"]+)"((?:\s+\d+)*)', line)
        if lm:
            raw = lm.group(2)
            flag_set = set(lm.group(3).split())
            ident = _marker_identity(raw)
            if "1" in flag_set and "2" in flag_set:
                _reject(
                    f"malformed linemarker (both enter and return "
                    f"flags) for {raw!r} — refusing to guess ancestry")
            if "1" in flag_set:
                # ENTER this file.
                if (include_stack
                        and include_stack[-1] == resolved_selected
                        and ident != resolved_selected
                        and not ident.startswith("<")):
                    nested_files.add(ident)
                include_stack.append(ident)
            elif "2" in flag_set:
                # RETURN TO this file: pop the completed include(s)
                # above the named caller; RETAIN the caller as active.
                if not include_stack:
                    _reject(
                        f"malformed linemarker: return to {raw!r} "
                        f"with an empty include stack — refusing to "
                        f"guess ancestry")
                if include_stack[-1] == ident:
                    # self-return (the named caller is already the
                    # active top, e.g. a redundant marker): RETAIN it.
                    pass
                elif ident in include_stack:
                    # returning to a deeper caller: pop everything
                    # above it and RETAIN it (missing end markers)
                    while include_stack and include_stack[-1] != ident:
                        include_stack.pop()
                else:
                    _reject(
                        f"malformed linemarker: return to {raw!r} "
                        f"which is not on the include stack "
                        f"({include_stack[-3:]}) — refusing to widen "
                        f"the selected-header scope")
            else:
                # No flags: a line-number/location reset. With an
                # empty stack this is the MAIN-FILE DECLARATION (e.g.
                # the opening `# 1 "<stdin>"`): establish the root
                # identity. GNU preprocessors additionally emit
                # root-level virtual preludes that rebind the pseudo
                # root with no flags (e.g. `# 0 "<stdin>"` followed
                # by `# 0 "<built-in>"` then `# 0 "<command-line>"`):
                # a no-flag PSEUDO-to-PSEUDO switch at ROOT scope (the
                # stack is exactly one pseudo root, nothing real
                # entered) is a legal root REBIND. Any OTHER file
                # change with no flags — including inside the selected
                # header or a real include ancestry — is ambiguous and
                # rejected.
                if not include_stack:
                    include_stack.append(ident)
                elif include_stack[-1] == ident:
                    pass  # same-file location reset
                elif (
                    len(include_stack) == 1
                    and _is_pseudo(include_stack[-1])
                    and _is_pseudo(ident)
                ):
                    # legal GNU root-level pseudo rebind
                    include_stack[0] = ident
                else:
                    _reject(
                        f"ambiguous no-flag linemarker: active file is "
                        f"{include_stack[-1]!r} but the marker names "
                        f"{raw!r} — refusing to guess ancestry")
            active_file = include_stack[-1] if include_stack else None
            if active_file == resolved_selected:
                saw_selected = True
            continue
        if active_file is None or active_file != resolved_selected:
            continue
        if _FUNClike_RE.match(line):
            _reject(f"function-like macro in {header}: "
                    f"{line.split('(')[0]}(...) — unsupported layout")
        # EMPTY-BODY define check first: `#define NAME` with no body
        # does not match _DEFINE_RE (which requires a body) and must
        # not fall through to the directive branch silently.
        _empty_def = re.match(
            r"^#define\s+([A-Za-z_][A-Za-z0-9_]*)\s*$", line)
        if _empty_def:
            _reject(f"macro {_empty_def.group(1)} in {header} has an "
                    f"EMPTY body (feature-flag define) — a numeric-only "
                    f"shim cannot represent it")
        m = _DEFINE_RE.match(line)
        if not m:
            # non-define content inside the selected header — reject:
            # the shim must contain numeric macro definitions only, and
            # nothing may be silently dropped.
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                _reject(f"non-macro content in {header}: "
                        f"{stripped[:80]!r}")
            if stripped.startswith("#"):
                if stripped.startswith("#pragma"):
                    # ONLY `#pragma once` is allowed (an include guard
                    # optimization with no numeric semantics). Any other
                    # pragma — e.g. pack, visibility, clang attribute
                    # push — has semantic effects a numeric-only shim
                    # cannot represent: reject.
                    tokens = stripped.split()
                    if tokens[1:2] != ["once"] or len(tokens) > 2:
                        _reject(f"pragma with semantic effects in "
                                f"{header}: {stripped[:60]!r} (only "
                                f"'#pragma once' is allowed)")
                elif not stripped.startswith(("#define", "#endif",
                                              "#ifndef", "#ifdef",
                                              "#else", "#if")):
                    # conditional-inclusion directives are structural
                    # and already resolved by the preprocessor; any
                    # OTHER directive is unsupported
                    _reject(f"unsupported directive in {header}: "
                            f"{stripped[:60]!r}")
            continue
        name, body = m.group(1), m.group(2).strip()
        if not _NUMERIC_RE.match(body):
            _reject(f"macro {name} has a non-numeric body: {body[:60]!r}")
        if name in defs and defs[name] != body:
            _reject(f"macro {name} redefined with a DIFFERENT value "
                    f"({defs[name]!r} -> {body!r}) — ambiguous")
        if name not in defs:
            order.append(name)
        defs[name] = body

    return defs, order, saw_selected, nested_files


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiler", required=True,
                        help="C/C++ preprocessor-capable compiler path")
    parser.add_argument("--compiler-arg", action="append", default=[],
                        help="extra compiler argument (repeatable)")
    parser.add_argument("--header", required=True,
                        help="relative include name of the SDK macro header")
    parser.add_argument("--include-dir", action="append", default=[],
                        help="include search dir (repeatable)")
    parser.add_argument("--output", required=True,
                        help="build-directory output shim path")
    args = parser.parse_args()

    # The header must be a RELATIVE include path with no traversal,
    # no newline/control chars, and no quote/angle-bracket injection.
    header = args.header
    if (header.startswith("/") or header.startswith("\\")
            or ".." in header.split("/")
            or any(ord(c) < 0x20 for c in header)
            or '"' in header or "'" in header or ">" in header
            or header.strip() != header or not header):
        print("[hcu-macro-shim] --header must be a clean relative include "
              "name (no absolute path, traversal, whitespace, quotes, "
              "angle brackets, or control characters)",
              file=sys.stderr)
        return 2

    # Preprocess in NON-STABLE context. Undefined macros via -U; the
    # vendor SDK context must be the plain (non-stable) one so the
    # ACTIVE definitions are the ones legacy TUs see. Explicit argv
    # list (never a shell string); stdin carries an empty TU whose only
    # content includes the selected header.
    import subprocess

    argv = [
        args.compiler,
        *args.compiler_arg,
        "-E",  # preprocess only
        "-U", "TORCH_TARGET_VERSION",
        "-U", "TORCH_STABLE_ONLY",
        "-dD",  # keep #define directives in the output
    ]
    for d in args.include_dir:
        argv += ["-I", d]
    argv += ["-x", "c++", "-"]
    empty_tu = f"#include <{args.header}>\n"
    proc = subprocess.run(
        argv, input=empty_tu, capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        print(f"[hcu-macro-shim] preprocessor failed rc={proc.returncode}",
              file=sys.stderr)
        print(proc.stderr[:2000], file=sys.stderr)
        return 3

    # Parse: track the "current file" via linemarkers; only #define
    # lines attributed to the SELECTED header are candidates. Nested
    # includes are NOT followed into the shim (their definitions belong
    # to their own headers — the dispatch keeps the original include
    # for legacy TUs, so nested content is only needed by stable TUs if
    # the selected header relies on it; if the header's own active
    # object-like macros reference nested names, the value text is
    # preserved verbatim and compilation will surface any gap — no
    # guessing here).
    # EXACT resolved-header identity: normalize the linemarker path
    # and compare against the resolved selected header (resolved via
    # the same include dirs). Suffix matching is not used.
    from pathlib import PurePosixPath

    resolved_selected = None
    for d in args.include_dir:
        cand = Path(d) / header
        if cand.is_file():
            resolved_selected = _identity(str(cand))
            break
    if resolved_selected is None:
        # the preprocessor may search system paths too; fall back to
        # the normalized include name itself
        resolved_selected = _normalize(header)

    selected = header
    # Include-stack ancestry model (rigorous selected-header scope):
    # preprocessor linemarkers carry flags — `# LINE "FILE" 1` ENTERS a
    # file (push), `# LINE "FILE" 2` RETURNS to a file after an include
    # ends (the stack unwinds to it). The selected header's scope is
    # exactly the regions where it is the innermost active real file;
    # a file ENTERED while the selected header is innermost is a NESTED
    # include of that header. Files entered BEFORE the selected header
    # (compiler preludes, root TU files) or AFTER returning from it are
    # NOT nested dependencies of the selected header.
    defs, order, saw_selected, nested_files = _parse_defines(
        proc.stdout, resolved_selected, header)
    if not saw_selected:
        _reject(f"the selected header {header} was never entered during "
                f"preprocessing (not found via the given include dirs)")
    if nested_files:
        # A nested include inside the selected header is REJECTED
        # outright — even when its contents do not affect the selected
        # header's own numeric macros. The shim must be numeric-only
        # and self-contained; silently ignoring non-selected
        # linemarkers is not acceptable.
        _reject(
            f"the selected header {header} contains nested includes "
            f"({sorted(nested_files)[:3]}); a numeric-only shim cannot "
            f"represent them — rejecting instead of guessing")

    if not defs:
        _reject(f"no active object-like numeric macros found in "
                f"{selected} — nothing to shim")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    guard = "VLLM_HCU_MACRO_SHIM_H_"
    lines = [
        "// GENERATED at build time by cmake/hcu_macro_shim.py —",
        "// numeric-only compatibility shim for the vendor SDK macro",
        f"// header <{selected}>. Build-directory-only; do not version.",
        f"#ifndef {guard}",
        f"#define {guard}",
        "// Selected ACTIVE object-like numeric macro definitions with",
        "// their exact preprocessed values (stable-TU safe: no types,",
        "// no declarations, no function-like macros, no includes).",
    ]
    for name in order:
        lines.append(f"#define {name} {defs[name]}")
    lines.append(f"#endif // {guard}")
    out.write_text("\n".join(lines) + "\n")
    print(f"[hcu-macro-shim] generated {out} ({len(defs)} numeric macros)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
