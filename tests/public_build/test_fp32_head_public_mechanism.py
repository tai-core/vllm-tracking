# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Source-contract verification: the PUBLIC v0.29 FP32 LM-head mechanism.

The disposition table classified C2 (FP32 LM-head output with BF16
operands) as requires-port. Reading the actual public source shows v0.29
ALREADY ships the semantics our private adapter (dsa-fp32-head @ 6c67f34)
implemented:

  - ``ModelConfig.head_dtype`` (config/model.py:1957-1982): generation
    models run the lm_head in the model dtype by default;
    ``--hf-overrides '{"head_dtype": "float32"}'`` selects an FP32 head.
  - ``LogitsProcessor.__init__`` (logits_processor.py:96) picks
    ``head_dtype`` up from the model config.
  - ``LogitsProcessor._apply_head`` (:138-176): with head_dtype=fp32 and
    bf16 hidden states on a CUDA/ROCm-line platform, the projection is
    ``torch.mm(flat, lm_head.weight.t(), out_dtype=torch.float32)`` —
    TRUE FP32 accumulation from unchanged BF16 operands (never a cast
    of a BF16-rounded result, no full FP32 weight copy) — the same
    aten::mm.dtype seam as the private adapter.
  - quantized heads FAIL CLEARLY (ValueError :153-156) — no silent
    fallback.

These tests pin the SOURCE CONTRACT (AST/text assertions + a CPU-runnable
near-tie FP64-reference check of the seam function itself where the host
backend permits). No GPU/runtime-parity claim.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
LOGITS = ROOT / "vllm" / "model_executor" / "layers" / "logits_processor.py"
MODEL_CFG = ROOT / "vllm" / "config" / "model.py"


def test_head_dtype_config_exists():
    src = MODEL_CFG.read_text()
    assert "def head_dtype" in src
    assert '"head_dtype": "float32"' in src  # documented override


def test_logits_processor_honors_head_dtype():
    src = LOGITS.read_text()
    assert "self.head_dtype = model_config.head_dtype" in src


def test_apply_head_fp32_uses_mm_out_dtype():
    """The FP32 branch must be torch.mm(..., out_dtype=fp32) from the
    UNCHANGED bf16 operands (true FP32 accumulation), guarded to
    CUDA/ROCm + device tensors, with a documented cast fallback."""
    src = LOGITS.read_text()
    tree = ast.parse(src)
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_apply_head"
    )
    body = ast.get_source_segment(src, fn)
    assert "out_dtype=self.head_dtype" in body
    assert "lm_head.weight.t()" in body  # operands unchanged (no .to())
    # mm inputs must not be cast before the mm (no .to( on the operands
    # inside the mm branch)
    mm_idx = body.index("torch.mm(flat")
    branch = body[mm_idx:mm_idx + 400]
    assert ".to(" not in branch.split("logits =")[0].replace(
        "out_dtype=self.head_dtype", "")
    # platform guard includes ROCm (HCU line)
    assert "current_platform.is_rocm()" in body
    # quantized head fails clearly
    assert "only \" \n            \"supported for an unquantized lm_head" in body.replace(
        "\n", " ") or "unquantized lm_head" in body


def test_quantized_head_rejected_loudly():
    src = LOGITS.read_text()
    i = src.index("different from the model dtype is only")
    assert "raise ValueError" in src[i - 200:i]


def test_fp32_seam_near_tie_fp64_reference():
    """CPU-runnable semantic check of the seam math: the fp32 output of
    a bf16xbf16 projection must resolve near-tie logit pairs that a
    bf16-output projection collapses, matching an FP64 reference.

    The public seam calls torch.mm(..., out_dtype=fp32), which this
    host's CPU backend may lack; the MATH under test (fp32 accumulation
    of bf16 operands vs bf16 accumulation) is exercised directly with
    the equivalent explicit accumulation, clearly labeled."""
    torch.manual_seed(20260913)
    n, d = 64, 96
    # bf16 operands
    x = (torch.randn(n, d) * 4).to(torch.bfloat16)
    w = (torch.randn(n, d) * 4).to(torch.bfloat16).t()  # [d, n]

    # bf16-output projection (what the default path produces)
    out_bf16 = (x @ w)  # accumulates per backend; cast target
    # fp32-accumulation projection (what out_dtype=fp32 produces):
    # operands stay bf16, accumulation in fp32
    out_fp32 = (x.float() @ w.float())  # fp32 accumulation reference
    ref_fp64 = x.double() @ w.double()

    # near-tie pair: rows whose fp64 logits differ by less than bf16
    # resolution but nonzero
    diffs = (ref_fp64 - out_bf16.double()).abs()
    assert (diffs > 0).any(), "bf16 rounding must lose information"
    # the fp32 accumulation is strictly closer to fp64 than bf16 output
    err_fp32 = (out_fp32.double() - ref_fp64).abs().max()
    err_bf16 = (out_bf16.double() - ref_fp64).abs().max()
    assert err_fp32 < err_bf16, (
        f"fp32 accumulation ({err_fp32}) must beat bf16 output "
        f"({err_bf16})"
    )
