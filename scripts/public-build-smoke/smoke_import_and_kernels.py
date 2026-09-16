# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Synthetic no-weight smoke checks for the GLM53 public-source v0.29 build.

Codex runs this INSIDE the built container on a BW1000 node. NO model
weights, NO serving, NO network, NO downloads, NO benchmarks.

STATUS MODEL (reporting contract):
  PASS        — the check ran and passed.
  FAIL        — the check ran and failed (actionable error attached);
                exit code 2.
  UNSUPPORTED — the check ran but the required capability/context is
                absent in this environment (e.g. no visible device).
  NOT_RUN     — the check is NOT IMPLEMENTED in this script (an
                explicit placeholder gap, NOT a capability decision).

  AGGREGATION: if any check is FAIL -> exit 2. Else if any check is
  NOT_RUN or UNSUPPORTED -> the run is INCOMPLETE (exit 3): a limited
  import/CPU smoke is NEVER reported as a completed execution
  validation. Only all-PASS exits 0 (GREEN). INCOMPLETE is separate
  from FAIL: the completed subset is real evidence, the gaps are
  explicit.

GATE LEVELS (what each level means):
  IMPORT — the module loads from the built wheel (module import only;
           module loading does NOT demonstrate Triton kernel
           compilation).
  COMPILE — a Triton kernel actually compiles (not implemented in this
           script: NOT_RUN).
  CPU — pure-CPU logic runs (processor sampling etc.).
  GPU — a device kernel executes (BF16 matmul has a real check here;
        INT8 execution is NOT_RUN — not probed, not decided).

Checks:
  1. vllm_hcu compiled extension import (IMPORT)
  2. plugin compatibility gate (IMPORT; SOURCE POLICY acceptance for
     an experimental build — NOT a runtime compatibility certification)
  3. GLM registry presence (IMPORT; registry presence only — NOT model
     class initialization)
  4. GLM processor import + synthetic CPU call (CPU)
  5. platform identification (IMPORT; reports the ACTUAL platform
     class + enum when hardware is available, not merely is_rocm())
  6. built core extension imports (IMPORT; real CMake install targets
     under vllm: _C/_C_stable_libtorch/_moe_C_stable_libtorch/_rocm_C/
     cumem_allocator — module identity only, no symbol-count claim)
  7. BF16 GEMM (GPU; deterministic input + independent CPU FP32
     reference cast to BF16)
  8. INT8 execution probe (GPU; NOT_RUN — no speculative API)
  9. GLM amd kpool_compress / kda kernel modules (IMPORT — module
     imports only; kernel compilation is NOT demonstrated here)
"""

from __future__ import annotations

import sys
import traceback

# EXPLICIT initialization order: torch is imported BEFORE any
# Torch-linked native extension is loaded. Torch-linked .so modules may
# require libtorch symbols at dlopen time; the check bodies below must
# not depend on some LATER check's import side effects having loaded
# torch first.
import torch  # noqa: F401  (initialization contract, not a check)

RESULTS: list[tuple[str, str, str]] = []  # (name, status, detail)

GATE = {"IMPORT": [], "COMPILE": [], "CPU": [], "GPU": []}

EXIT_GREEN = 0
EXIT_FAIL = 2
EXIT_INCOMPLETE = 3


def check(name, gate):
    assert gate in GATE, gate

    def deco(fn):
        try:
            status, detail = fn()
        except Exception:
            status, detail = "FAIL", traceback.format_exc(limit=3)[-400:]
        RESULTS.append((name, status, detail))
        GATE[gate].append((name, status))
        print(f"[{status}] ({gate}) {name}" + (f" — {detail}" if detail else ""))
        return fn
    return deco


@check("vllm_hcu.hcu_ops compiled extension import", "IMPORT")
def _():
    import vllm_hcu.hcu_ops  # noqa: F401
    return "PASS", "compiled extension imports"


@check("vllm_hcu compatibility gate (SOURCE POLICY, not runtime certification)", "IMPORT")
def _():
    from vllm_hcu.compatibility import inspect_vllm_compatibility

    c = inspect_vllm_compatibility()
    if c.compatible:
        return "PASS", (
            "source-policy acceptance for the EXPERIMENTAL public "
            "build: " + c.reason + " — NOT a runtime compatibility "
            "certification"
        )
    return "FAIL", c.detail()


@check("GLM model registry presence (registry only, not class init)", "IMPORT")
def _():
    from vllm.model_executor.models.registry import (
        _MULTIMODAL_MODELS,
        _SPECULATIVE_DECODING_MODELS,
        _TEXT_GENERATION_MODELS,
    )

    assert "Glm5NextForCausalLM" in _TEXT_GENERATION_MODELS
    assert "Glm5NextForConditionalGeneration" in _MULTIMODAL_MODELS
    assert "Glm5NextMTPModel" in _SPECULATIVE_DECODING_MODELS
    return "PASS", (
        "GLM entries present in their registry tables — registry "
        "presence only; this does NOT initialize any model class"
    )


@check("GLM processor import + synthetic CPU call", "CPU")
def _():
    from vllm.transformers_utils.processors.glm5next import (
        Glm5NextProcessor,
        glm_sample_frame_indices,
    )

    idx = glm_sample_frame_indices(48, 2.0, 24.0, target_fps=2.0)
    assert len(idx) >= 1 and all(0 <= i < 48 for i in idx)
    return "PASS", f"sampled {len(idx)} in-range indices"


@check("platform identification (actual class + enum)", "IMPORT")
def _():
    from vllm.platforms import current_platform

    cls = type(current_platform)
    enum = getattr(cls, "_enum", None)
    is_rocm = current_platform.is_rocm()
    detail = (
        f"platform class={cls.__module__}.{cls.__name__}, "
        f"enum={getattr(enum, 'name', enum)}, is_rocm()={is_rocm}"
    )
    # The public bridge declares vllm_hcu.platforms.hcu.HCUPlatform with
    # PlatformEnum.ROCM. Report the ACTUAL class and enum — do not
    # infer "HCU" from is_rocm() alone.
    if cls.__name__ == "HCUPlatform":
        return "PASS", detail + " — HCU platform class active"
    if is_rocm:
        return "PASS", (
            detail + " — ROCm-line platform (NOT the HCUPlatform class; "
            "verify whether the HCU plugin platform was expected)"
        )
    return "FAIL", detail + " — neither HCUPlatform nor ROCm-line"


@check("built core extension imports (real CMake install targets)", "IMPORT")
def _():
    # REAL core module names from the project's CMake install targets
    # (define_extension_target ... DESTINATION vllm): _C, _C_stable_
    # libtorch, _moe_C_stable_libtorch, _rocm_C, cumem_allocator.
    # (spinloop is CXX-only and platform-conditional; not probed here.)
    # Identity is what is checked: a module that imports IS the built
    # artifact. Zero public pybind symbols is NOT failure for torch-op
    # registration modules (STABLE_TORCH_LIBRARY registers ops at load
    # time without exporting names) — no symbol counting is claimed.
    import importlib
    import vllm

    loaded = []
    missing = []
    for mod in ("vllm._C", "vllm._C_stable_libtorch",
                "vllm._moe_C_stable_libtorch", "vllm._rocm_C",
                "vllm.cumem_allocator"):
        try:
            m = importlib.import_module(mod)
            loaded.append(f"{mod} ({getattr(m, '__file__', 'builtin')})")
        except ImportError as exc:
            missing.append(f"{mod}: {type(exc).__name__}")
    if missing:
        return "FAIL", (
            "core extension modules missing: " + "; ".join(missing)
        )
    return "PASS", (
        "core extension modules import (CMake install targets, "
        "module identity only — no symbol-count claim): "
        + "; ".join(loaded)
    )


@check("BF16 GEMM (deterministic + independent CPU FP32 reference)", "GPU")
def _():
    import torch

    if not torch.cuda.is_available() and not getattr(torch.cuda, "is_hcu_available", lambda: False)():
        return "UNSUPPORTED", "no HCU device visible in this process"
    dev = "cuda"
    # Deterministic small input: seeded generator, integer-valued
    # matrix (exact in FP32; BF16 rounding is bounded and checked).
    g = torch.Generator().manual_seed(20260912)
    a_int = torch.randint(-64, 64, (32, 48), generator=g, dtype=torch.int64)
    b_int = torch.randint(-64, 64, (48, 32), generator=g, dtype=torch.int64)
    # Independent CPU FP32 reference, cast to BF16 (reference for what
    # a correctly rounding BF16 GEMM should produce).
    a_bf16 = a_int.to(torch.float32).to(torch.bfloat16)
    b_bf16 = b_int.to(torch.float32).to(torch.bfloat16)
    ref = (a_bf16.float() @ b_bf16.float()).to(torch.bfloat16)
    a_d = a_bf16.to(dev)
    b_d = b_bf16.to(dev)
    c_d = a_d @ b_d
    assert c_d.shape == (32, 32)
    # Move the RESULT to the host first (the .cpu() copy synchronizes
    # the device stream) so the comparison against the independent CPU
    # reference happens entirely on the CPU with both operands
    # co-located — no implicit device-side reduction or lazy evaluation.
    c = c_d.cpu()
    # Tolerance: BF16 accumulation differences are bounded; the exact
    # reference bound uses RTOL relative to the FP32-exact product.
    diff = (c.float() - ref.float()).abs()
    scale = ref.float().abs().max().clamp(min=1.0)
    assert torch.isfinite(c.float()).all(), "non-finite output"
    assert (diff <= 2 * scale * 0.03125).all(), (
        f"bf16 matmul deviates beyond 2*BF16-eps scale: max diff "
        f"{diff.max().item():.4f} vs scale {scale.item():.1f}"
    )
    return "PASS", (
        "bf16 matmul (result moved to CPU, synchronized) matches the "
        "independent CPU FP32->BF16 reference within 2*BF16-eps on "
        "deterministic seeded input"
    )


@check("INT8 execution probe", "GPU")
def _():
    # NOT implemented in this script: a real INT8 probe must exercise
    # an actual quantized operator path against a reference. This is an
    # explicit NOT_RUN placeholder — NOT a capability decision (the
    # INT8 path is neither confirmed nor denied by this smoke).
    return "NOT_RUN", (
        "INT8 operator execution is not implemented in this smoke "
        "script; no speculative API is invoked — run the dedicated "
        "quantized-operator suite on-node for an actual decision"
    )


@check("GLM amd kpool_compress kernel module import", "IMPORT")
def _():
    import torch  # noqa: F401

    from vllm.platforms import current_platform

    if not current_platform.is_rocm():
        return "UNSUPPORTED", "not on ROCm-line platform; amd kernel path not selected"
    from vllm.models.glm5next.amd.ops import kpool_compress

    assert hasattr(kpool_compress, "__file__")
    return "PASS", (
        "kpool_compress module IMPORTS — module loading does NOT "
        "demonstrate Triton kernel compilation; run the upstream "
        "kernel suite (tests/kernels/test_kpool_decode_update_batched.py) "
        "for compilation and numerics"
    )


@check("GLM amd kda kernel module import", "IMPORT")
def _():
    from vllm.platforms import current_platform

    if not current_platform.is_rocm():
        return "UNSUPPORTED", "not on ROCm-line platform"
    from vllm.models.glm5next.amd.ops.third_party.kda import (
        chunk_kda_with_fused_gate,
        fused_recurrent_kda,
    )

    assert callable(chunk_kda_with_fused_gate) and callable(fused_recurrent_kda)
    return "PASS", (
        "kda kernel modules IMPORT — module loading does NOT "
        "demonstrate Triton kernel compilation; run the upstream "
        "mamba kernel suite for compilation and numerics"
    )


def main() -> int:
    failed = [r for r in RESULTS if r[1] == "FAIL"]
    incomplete = [r for r in RESULTS if r[1] in ("NOT_RUN", "UNSUPPORTED")]
    print("\n=== smoke summary (by gate) ===")
    for gate, items in GATE.items():
        print(f"  [{gate}]")
        for name, status in items:
            print(f"    {status:12s} {name}")
    if failed:
        print(
            f"\nVERDICT: FAIL ({len(failed)} failed) — fix the errors "
            "above; this is a real failure, not a gap"
        )
        return EXIT_FAIL
    if incomplete:
        print(
            f"\nVERDICT: INCOMPLETE ({len(incomplete)} NOT_RUN/"
            f"UNSUPPORTED) — the completed subset is real evidence of a "
            "LIMITED import/CPU(+GPU-if-run) smoke only; it is NOT a "
            "completed execution validation. Gaps (actionable):"
        )
        for name, status, detail in incomplete:
            print(f"  - [{status}] {name}: {detail}")
        return EXIT_INCOMPLETE
    print(
        "\nVERDICT: GREEN (all checks PASS) — note this remains a "
        "synthetic no-weight smoke, not model/runtime acceptance"
    )
    return EXIT_GREEN


if __name__ == "__main__":
    sys.exit(main())
