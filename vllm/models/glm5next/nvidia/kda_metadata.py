# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM KDA metadata for single-checkpoint speculative state recovery."""

from vllm.models.kimi_k3.nvidia.kda_metadata import KimiK3KDAAttentionBackend


class Glm5NextRecoverSSMAttentionBackend(KimiK3KDAAttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "GLM5NEXT_RECOVERSSM"
