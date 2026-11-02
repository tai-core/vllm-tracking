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
    def build(  # type: ignore[override]
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
        fast_build: bool = False,
    ) -> KimiK3KDAMetadata:
        from vllm.models.kimi_k3.nvidia.kda_metadata import (
            flush_recoverssm_dp_commit,
        )

        flush_recoverssm_dp_commit(self.recoverssm_context)
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
            assert metadata.non_spec_state_indices_tensor is not None
            metadata.non_spec_query_start_loc = torch.arange(
                metadata.num_decodes + 1,
                dtype=torch.int32,
                device=metadata.non_spec_state_indices_tensor.device,
            )
        from vllm.config.kda_parallel import kda_tp_over_dp

        if kda_tp_over_dp(self.vllm_config):
            metadata.kda_dp_plan = common_attn_metadata.kda_dp_plan
            metadata.kda_dp_dummy = common_attn_metadata.kda_dp_dummy
            from vllm.models.glm5next.common.kda import (
                globalize_gdn_metadata_for_dp_config,
            )

            metadata = globalize_gdn_metadata_for_dp_config(
                self.vllm_config,
                self.kv_cache_spec,
                metadata,
                self.dp_global_state_indices,
                self.dp_global_qsl,
                self.dp_global_order,
                self.dp_global_num_valid,
                self.dp_global_spec_token_indx,
                self.dp_global_spec_accepted,
                self.dp_global_spec_num_valid,
            )
        return metadata


class Glm5NextRecoverSSMAttentionBackend(KimiK3KDAAttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "GLM5NEXT_RECOVERSSM"

    @staticmethod
    def get_builder_cls() -> type[Glm5NextRecoverSSMMetadataBuilder]:
        return Glm5NextRecoverSSMMetadataBuilder
