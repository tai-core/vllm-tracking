# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM RecoverSSM metadata shares the KDA checkpoint/commit protocol."""

import torch

from vllm.models.kimi_k3.nvidia.kda_metadata import (
    KimiK3KDAAttentionBackend,
    KimiK3KDAMetadata,
    KimiK3KDAMetadataBuilder,
)
from vllm.v1.attention.backend import CommonAttentionMetadata


class Glm5NextRecoverSSMMetadataBuilder(KimiK3KDAMetadataBuilder):
    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
        fast_build: bool = False,
    ) -> KimiK3KDAMetadata:
        metadata = super().build(
            common_prefix_len,
            common_attn_metadata,
            num_accepted_tokens,
            num_decode_draft_tokens_cpu,
            fast_build,
        )
        # GLM's recurrent kernel consumes offsets even for a packed decode
        # subgroup; K3's packed kernel uses one row per sequence instead.
        if metadata.num_decodes and metadata.non_spec_query_start_loc is None:
            metadata.non_spec_query_start_loc = torch.arange(
                metadata.num_decodes + 1,
                dtype=torch.int32,
                device=metadata.non_spec_state_indices_tensor.device,
            )
        return metadata


class Glm5NextRecoverSSMAttentionBackend(KimiK3KDAAttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "GLM5NEXT_RECOVERSSM"

    @staticmethod
    def get_builder_cls() -> type[Glm5NextRecoverSSMMetadataBuilder]:
        return Glm5NextRecoverSSMMetadataBuilder
