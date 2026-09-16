# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Focused tests for the smoke script's status model: classification,
aggregation, and verdict semantics — driven with SYNTHETIC outcomes
(no vllm/vllm_hcu/model imports; the smoke module is loaded with stub
dependencies so no real check body executes)."""

from __future__ import annotations

import importlib.util as ilu
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SMOKE = ROOT / "scripts" / "public-build-smoke" / "smoke_import_and_kernels.py"


def _load_smoke_module():
    """Load the smoke script WITHOUT executing its @check bodies' real
    imports: the check bodies only run when CALLED, and this loader
    never calls them. Module-level imports in the script are stdlib
    only (sys/traceback), so a plain load is side-effect free."""
    spec = ilu.spec_from_file_location("smoke_under_test", SMOKE)
    mod = ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fresh_state(mod):
    mod.RESULTS.clear()
    for k in mod.GATE:
        mod.GATE[k].clear()


def _add(mod, name, gate, status, detail=""):
    mod.RESULTS.append((name, status, detail))
    mod.GATE[gate].append((name, status))


def test_exit_codes_are_three_way():
    mod = _load_smoke_module()
    assert mod.EXIT_GREEN == 0
    assert mod.EXIT_FAIL == 2
    assert mod.EXIT_INCOMPLETE == 3
    assert mod.EXIT_FAIL != mod.EXIT_INCOMPLETE


def test_all_pass_is_green():
    mod = _load_smoke_module()
    _fresh_state(mod)
    _add(mod, "a", "IMPORT", "PASS")
    _add(mod, "b", "CPU", "PASS")
    rc = mod.main()
    assert rc == mod.EXIT_GREEN


def test_any_fail_is_fail_even_with_gaps():
    mod = _load_smoke_module()
    _fresh_state(mod)
    _add(mod, "a", "IMPORT", "PASS")
    _add(mod, "b", "GPU", "NOT_RUN")
    _add(mod, "c", "CPU", "FAIL", "boom")
    rc = mod.main()
    assert rc == mod.EXIT_FAIL  # FAIL dominates gaps


def test_gaps_without_fail_is_incomplete_not_green():
    """NOT_RUN/UNSUPPORTED without any FAIL must report INCOMPLETE —
    never a blanket GREEN/complete claim."""
    mod = _load_smoke_module()
    _fresh_state(mod)
    _add(mod, "a", "IMPORT", "PASS")
    _add(mod, "b", "IMPORT", "PASS")
    _add(mod, "c", "GPU", "NOT_RUN", "not implemented")
    rc = mod.main()
    assert rc == mod.EXIT_INCOMPLETE


def test_unsupported_alone_is_incomplete():
    mod = _load_smoke_module()
    _fresh_state(mod)
    _add(mod, "a", "IMPORT", "PASS")
    _add(mod, "b", "GPU", "UNSUPPORTED", "no device")
    rc = mod.main()
    assert rc == mod.EXIT_INCOMPLETE


def test_check_decorator_classifies_gate_and_records():
    mod = _load_smoke_module()
    _fresh_state(mod)

    @mod.check("synthetic ok", "IMPORT")
    def _ok():
        return "PASS", "fine"

    @mod.check("synthetic gap", "GPU")
    def _gap():
        return "NOT_RUN", "placeholder"

    @mod.check("synthetic boom", "CPU")
    def _boom():
        raise RuntimeError("synthetic error")

    assert ("synthetic ok", "PASS", "fine") in mod.RESULTS
    assert ("synthetic gap", "NOT_RUN", "placeholder") in mod.RESULTS
    boom = next(r for r in mod.RESULTS if r[0] == "synthetic boom")
    assert boom[1] == "FAIL"
    assert "synthetic error" in boom[2]
    assert ("synthetic ok", "PASS") in mod.GATE["IMPORT"]
    assert ("synthetic gap", "NOT_RUN") in mod.GATE["GPU"]


def test_gate_labels_import_only_where_no_kernel_runs():
    """Source-level contract: every check body that only imports a
    module (no kernel invocation) must be registered under IMPORT (or
    CPU for pure logic), never COMPILE — and the COMPILE gate carries
    no PASS-claiming check in this script (kernel compilation is
    NOT_RUN)."""
    src = SMOKE.read_text()
    import re
    # find every @check registration and its gate
    regs = re.findall(
        r'@check\("([^"]+)",\s*"(\w+)"\)', src)
    assert regs, "no checks found"
    gates = {name: gate for name, gate in regs}
    # kernel-module-import checks must be IMPORT, not COMPILE
    assert gates["GLM amd kpool_compress kernel module import"] == "IMPORT"
    assert gates["GLM amd kda kernel module import"] == "IMPORT"
    # the INT8 execution probe is NOT_RUN (source: literal status)
    assert '"INT8 execution probe", "GPU"' in src
    assert 'return "NOT_RUN"' in src
    # no check claims COMPILE in this script
    assert "COMPILE" not in {g for g in gates.values()}, gates


def test_int8_placeholder_is_not_a_capability_decision():
    src = SMOKE.read_text()
    i = src.index('"INT8 execution probe"')
    seg = src[i:i + 900]
    assert "NOT_RUN" in seg
    assert "NOT a capability decision" in seg
    assert "speculative API" in seg


def test_platform_check_reports_class_not_only_is_rocm():
    src = SMOKE.read_text()
    i = src.index('"platform identification (actual class + enum)"')
    seg = src[i:i + 1400]
    assert "type(current_platform)" in seg
    assert "_enum" in seg
    assert "HCUPlatform" in seg
    assert "do not" in seg.lower() and "infer" in seg.lower()


def test_bf16_reference_is_independent_cpu_fp32():
    src = SMOKE.read_text()
    i = src.index('"BF16 GEMM (deterministic')
    seg = src[i:i + 2400]
    assert "manual_seed" in seg          # deterministic input
    assert "Generator" in seg
    assert "a_bf16.float() @ b_bf16.float()" in seg  # CPU FP32 reference
    assert ".to(torch.bfloat16)" in seg   # reference cast to BF16
    # finite-only checking is gone: a real bound is asserted
    assert "0.03125" in seg or "BF16-eps" in seg
    assert "isfinite" in seg  # still checked, but not alone
    # result moved to host (synchronizing copy) before comparison
    assert "c = c_d.cpu()" in seg


def test_compatibility_gate_wording_is_source_policy():
    src = SMOKE.read_text()
    assert "SOURCE POLICY" in src
    assert "NOT a runtime compatibility certification" in src
    # stale 0.28.1 wording removed from the check title
    assert "release-line 0.28.1" not in src


# ---------------------------------------------------------------------------
# initialization order + device-co-located comparison (SYNTHETIC/mocked —
# no GPU claim from these source tests)
# ---------------------------------------------------------------------------

def test_torch_imported_before_extension_checks():
    """Source contract: the smoke script imports torch at MODULE level
    BEFORE any check body can import a Torch-linked native extension.
    Verified by source order: the module-level `import torch` precedes
    the first `import vllm_hcu` inside any check body."""
    src = SMOKE.read_text()
    i_torch = src.index("import torch  # noqa: F401  (initialization contract")
    i_first_ext = src.index("import vllm_hcu.hcu_ops")
    assert i_torch < i_first_ext, (
        "torch must be imported before the first native-extension check"
    )
    # the torch import is at module level (not indented inside a def)
    line = src[:i_torch].splitlines()[-1]
    assert not line.startswith(" "), "module-level import expected"


def test_bf16_result_moved_to_cpu_before_comparison():
    """Source contract: the GPU result is moved to the host
    (.cpu() — a synchronizing copy) BEFORE the independent CPU
    reference comparison; both comparison operands are then
    host-co-located."""
    src = SMOKE.read_text()
    i = src.index('"BF16 GEMM (deterministic')
    seg = src[i:i + 2000]
    assert "c = c_d.cpu()" in seg
    i_cpu = seg.index("c = c_d.cpu()")
    i_diff = seg.index("diff = (c.float() - ref.float())")
    assert i_cpu < i_diff, "result must be on CPU before the comparison"
    # the device tensor is NOT used in the comparison afterwards
    after = seg[i_diff:]
    assert "c_d.float()" not in after and "c_d -" not in after


def test_core_extension_check_uses_real_cmake_targets(monkeypatch):
    """The core-extension check imports the REAL CMake install targets
    (vllm._C, vllm._C_stable_libtorch, vllm._moe_C_stable_libtorch,
    vllm._rocm_C, vllm.cumem_allocator) and reports module identity —
    no symbol counting, no hcu_ops-as-core claim. Driven with a
    SYNTHETIC import environment (mocked importlib.import_module);
    labeled mock coverage, not GPU/build proof."""
    import types
    mod = _load_smoke_module()
    # find the check function by its title
    src = SMOKE.read_text()
    assert '"built core extension imports (real CMake install targets)"' in src
    # extract and exec the check body with mocked importlib
    i = src.index('"built core extension imports (real CMake install targets)"')
    j = src.index("@check(", i + 10)
    body = src[i:j]
    # build a controlled namespace
    imported = []
    fake_vllm = types.ModuleType("vllm")

    def fake_import(name):
        imported.append(name)
        if name.startswith("vllm."):
            m = types.ModuleType(name)
            m.__file__ = f"/synthetic/site-packages/{name.replace('.', '/')}.so"
            return m
        if name == "vllm":
            return fake_vllm
        if name == "importlib":
            raise ImportError("synthetic env")
        raise ImportError(name)

    class FakeImportlib:
        @staticmethod
        def import_module(name):
            return fake_import(name)

    ns = {"importlib": FakeImportlib, "vllm": fake_vllm,
          "__name__": "__synthetic__"}
    # exec the def-line + body (it is a plain def _(): ...)
    k = body.index("def _():")
    code = "import importlib\nimport vllm\n" + body[k:]
    import builtins
    real_import = builtins.__import__
    builtins.__import__ = lambda n, *a, **k: (
        FakeImportlib if n == "importlib" else
        fake_vllm if n == "vllm" else real_import(n, *a, **k))
    try:
        exec(compile(code, "<check>", "exec"), ns)
        status, detail = ns["_"]()
    finally:
        builtins.__import__ = real_import
    assert status == "PASS"
    for expected in ("vllm._C", "vllm._C_stable_libtorch",
                     "vllm._moe_C_stable_libtorch", "vllm._rocm_C",
                     "vllm.cumem_allocator"):
        assert expected in imported, imported
    # module identity reported, no symbol count
    assert "module identity only" in detail
    assert "/synthetic/site-packages/vllm/_C.so" in detail


def test_core_extension_check_fails_on_missing_module():
    """SYNTHETIC: one real target missing -> FAIL with the module name
    (actionable), not a silent skip."""
    import types
    src = SMOKE.read_text()
    i = src.index('"built core extension imports (real CMake install targets)"')
    j = src.index("@check(", i + 10)
    body = src[i:j]
    k = body.index("def _():")
    code = "import importlib\nimport vllm\n" + body[k:]

    class FakeImportlib:
        @staticmethod
        def import_module(name):
            if name == "vllm._rocm_C":
                raise ImportError("synthetic missing")
            m = types.ModuleType(name)
            m.__file__ = "/synthetic/x.so"
            return m

    fake_vllm = types.ModuleType("vllm")
    ns = {"importlib": FakeImportlib, "vllm": fake_vllm,
          "__name__": "__synthetic__"}
    import builtins
    real_import = builtins.__import__
    builtins.__import__ = lambda n, *a, **k: (
        FakeImportlib if n == "importlib" else
        fake_vllm if n == "vllm" else real_import(n, *a, **k))
    try:
        exec(compile(code, "<check>", "exec"), ns)
        status, detail = ns["_"]()
    finally:
        builtins.__import__ = real_import
    assert status == "FAIL"
    assert "vllm._rocm_C" in detail and "missing" in detail
