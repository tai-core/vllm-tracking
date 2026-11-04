# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops.kda import FusedRMSNormGated


@pytest.mark.skipif(not current_platform.is_musa(), reason="MUSA-only dispatch")
@torch.inference_mode()
def test_kda_forward_oot_matches_native(default_vllm_config):
    device = torch.device(current_platform.device_type)
    module = FusedRMSNormGated(
        128,
        eps=1e-5,
        activation="sigmoid",
        device=device,
        dtype=torch.bfloat16,
    )
    module.weight.uniform_(-1, 1)
    x = torch.randn(1, 16, 4, 128, dtype=torch.bfloat16, device=device)
    g = torch.randn(16, 4, 128, dtype=torch.bfloat16, device=device)

    oot_out = module.forward_oot(x.clone(), g.clone())
    native_out = module.forward_native(x.clone(), g.clone())

    torch.testing.assert_close(oot_out, native_out, atol=1e-3, rtol=1e-2)
