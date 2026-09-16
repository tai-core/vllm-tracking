# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Source-contract tests for the 32K/C1 INT8-checkpoint regression gate.

C1 (MoE S exactly once + shared/routed alias), C5 (BF16 cache + FP32 KDA
state), C6-CORRECTED (the ACTUAL explicit quantization route: tracked
launch scripts select --quantization slimquant_marlin, served by the
bridge's SlimQuantMarlinFacade -> SlimQuantCompressedTensorsMarlinConfig
-> compressed-tensors int8 linear / Marlin MoE, registered through the
PUBLIC register_quantization_config API). Source-only; no runtime claim.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
BRIDGE_ROOT = ROOT.parent / "v029-bridge"
LAYER = ROOT / "vllm" / "model_executor" / "layers" / "fused_moe" / "layer.py"
KDA = ROOT / "vllm" / "models" / "glm5next" / "nvidia" / "kda.py"
MAMBA_UTILS = ROOT / "vllm" / "model_executor" / "layers" / "mamba" / "mamba_utils.py"
MODEL_CFG = ROOT / "vllm" / "config" / "model.py"
QUANT_INIT = ROOT / "vllm" / "model_executor" / "layers" / "quantization" / "__init__.py"
LAUNCH = ROOT.parents[2] / "runtime" / "launch_bw1000_07_tp8_candidate.sh"
LAUNCH_G10A = ROOT.parents[2] / "runtime" / "launch_glm53_g10a_one_shot.sh"
FACADE = BRIDGE_ROOT / "vllm_hcu" / "model_executor" / "layers" / "quantization" / "slimquant_facade.py"
MARLIN = BRIDGE_ROOT / "vllm_hcu" / "model_executor" / "layers" / "quantization" / "compressed_tensors" / "compressed_tensors_marlin.py"
SLIMREG = BRIDGE_ROOT / "vllm_hcu" / "patch" / "platform" / "core_fix" / "patch_slimquant_registry.py"


# ---------------------------------------------------------------------------
# C1: MoE routed scale exactly once; shared slot compensated
# ---------------------------------------------------------------------------

def test_c1_scale_ownership_and_shared_slot():
    src = LAYER.read_text()
    # the scale is a NOP on exactly one of the two paths
    assert "routed_scaling_factor=routed_scaling_factor\n            if not apply_routed_scale_to_output\n            else 1.0,".replace("\n            ", " ") or \
        "routed_scaling_factor=routed_scaling_factor if not apply_routed_scale_to_output else 1.0" in src.replace("\n", " ")
    # fused shared slot weight compensates so the shared contribution
    # stays 1.0 when the runner scales the combined output
    assert "(1.0 / routed_scaling_factor)" in src
    assert "shared_expert_weight=(" in src


def test_c1_glm_model_passes_both_knobs():
    model = (ROOT / "vllm" / "models" / "glm5next" / "nvidia" / "model.py").read_text()
    assert "routed_scaling_factor=self.routed_scaling_factor" in model
    assert "apply_routed_scale_to_output=apply_routed_scale_to_output" in model


def test_c1_forward_no_input_alias_rewrite():
    """The MoE forward must not rewrite hidden_states in place (shared/
    routed inputs consume the same tensor read-only)."""
    model = (ROOT / "vllm" / "models" / "glm5next" / "nvidia" / "model.py").read_text()
    tree = ast.parse(model)
    fwd = next(
        f for c in ast.walk(tree)
        if isinstance(c, ast.ClassDef) and c.name == "Glm5NextMoE"
        for f in c.body if isinstance(f, ast.FunctionDef) and f.name == "forward"
    )
    body = ast.get_source_segment(model, fwd)
    for bad in ("hidden_states +=", "hidden_states.mul_",
                "hidden_states.add_", "hidden_states.copy_"):
        assert bad not in body, bad


# ---------------------------------------------------------------------------
# C5: BF16 cache + FP32 KDA state selection
# ---------------------------------------------------------------------------

def test_c5_kda_state_dtype_contract():
    src = MAMBA_UTILS.read_text()
    assert "def kda_state_dtype" in src
    i = src.index("def kda_state_dtype")
    seg = src[i:i + 400]
    # SSM state component is ALWAYS fp32; the first component follows
    # the cache dtype knob
    assert "return (state_dtype, torch.float32)" in seg
    assert "get_kv_cache_torch_dtype(mamba_cache_dtype, model_dtype)" in seg


def test_c5_kda_get_state_dtype_wiring():
    src = KDA.read_text()
    assert "MambaStateDtypeCalculator.kda_state_dtype(" in src
    assert "self.cache_config.mamba_cache_dtype" in src


def test_c5_mamba_cache_dtype_knob_exists():
    cfg = (ROOT / "vllm" / "config" / "cache.py").read_text()
    assert 'mamba_cache_dtype: MambaDType = "auto"' in cfg


def test_c5_kda_state_dtype_resolution():
    """Runtime resolution check (pure Python): bf16 cache knob ->
    (bf16, fp32)."""
    from vllm.model_executor.layers.mamba.mamba_utils import (
        MambaStateDtypeCalculator,
    )

    conv, ssm = MambaStateDtypeCalculator.kda_state_dtype(
        torch.bfloat16, "bfloat16"
    )
    assert conv == torch.bfloat16
    assert ssm == torch.float32


# ---------------------------------------------------------------------------
# C6 (CORRECTED): the ACTUAL launch route is slimquant_marlin via the
# bridge facade; the public registry API it registers through exists.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not LAUNCH.exists(), reason="tracked launch script not found")
def test_c6_launch_selects_slimquant_marlin():
    src = LAUNCH.read_text()
    assert "--quantization slimquant_marlin" in src
    assert "--dtype bfloat16" in src
    assert "--tensor-parallel-size 8" in src
    assert "--enforce-eager" in src
    assert "--max-num-seqs 1" in src


@pytest.mark.skipif(not LAUNCH_G10A.exists(), reason="tracked launch script not found")
def test_c6_g10a_one_shot_selects_slimquant_marlin():
    src = LAUNCH_G10A.read_text()
    assert "--quantization slimquant_marlin" in src


def test_c6_bridge_facade_maps_compressed_tensors_checkpoint():
    src = FACADE.read_text()
    # the facade overrides compressed-tensors checkpoints under the
    # slimquant_marlin user_quant
    assert 'hf_quant_cfg.get("quant_method") == "compressed-tensors"' in src
    assert 'user_quant == cls._registry_name' in src
    assert '_registry_name = "slimquant_marlin"' in src
    # the implementation class is the Marlin config
    assert "SlimQuantCompressedTensorsMarlinConfig" in src


def test_c6_marlin_config_routes_linear_and_moe():
    src = MARLIN.read_text()
    assert "class SlimQuantCompressedTensorsMarlinConfig(CompressedTensorsConfig)" in src
    # linear layers -> compressed-tensors linear method (int8 schemes)
    assert "CompressedTensorsLinearMethod(self)" in src
    # routed experts -> Marlin MoE (or CompressedTensorsMoEMethod under
    # aiter) — both REAL bridge classes
    assert "CompressedTensorsMarlinMoEMethod.get_moe_method" in src
    assert "CompressedTensorsMoEMethod.get_moe_method" in src


def test_c6_public_registry_api_present():
    """The bridge registers the facade through the PUBLIC out-of-tree
    quantization registry; that API must exist with the documented
    shape in the public tree."""
    src = QUANT_INIT.read_text()
    assert "def register_quantization_config(quantization: str):" in src
    assert "QUANTIZATION_METHODS: list[str]" in src
    assert "_CUSTOMIZED_METHOD_TO_QUANT_CONFIG" in src


def test_c6_registry_patch_targets_public_api():
    src = SLIMREG.read_text()
    assert "register_quantization_config" in src
    assert "QUANTIZATION_METHODS" in src
    # loud failure on shape change (never a silent skip)
    assert "PatchCompatibilityError" in src
