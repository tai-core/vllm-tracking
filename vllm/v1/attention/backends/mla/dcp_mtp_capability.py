# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Restricted MTP-with-interleave capability for the adapted HCU GLM candidate.

The node-08 model round (namespace dcp2-model4096-20261009T043219Z)
failed at startup on the global compatibility assert:

    AssertionError: MTP with cp_kv_cache_interleave_size > 1 is not
    supported in HcuFlashMLASparseImpl

The chain: ``check_attention_cp_compatibility`` reads
``supports_mtp_with_cp_non_trivial_interleave_size`` (AttentionImpl
default False) from every layer impl; the HCU sparse impl never set
it. This module grants the capability — as an INSTANCE flag computed
in the impl's real ``__init__`` from the REAL current VllmConfig —
only for the already-validated GLM kpool BF16 DCP candidate, using
the SAME conditions as the indexer's
``validate_dcp_compressed_indexer_support`` (real config identity,
kpool compression, pool-coherent interleave, PCP 1, explicit BF16,
no transfer, resolved S). No global True: every unadapted
combination (other models, FP8 MLA, PCP>1, misaligned I, DCP1
MTP, non-granted backends) keeps the default False and the original
rejection.

The INDEXER compress ratio R is read from the indexer cache's OWN
spec in the forward context (``tokens_per_state``), never from the
MLA latent spec (whose ``tokens_per_state`` is 1 — mistaking it for
R would wrongly grant the capability).
"""

from __future__ import annotations

import torch

from vllm.v1.kv_cache_interface import MLAAttentionSpec

_TORCH_UINT8_SENTINEL = torch.uint8


def _fp8_single_module_mtp4_profile(vllm_config) -> bool:
    """The EXACT single-module FP8 MTP4 speculative profile (fail-closed).

    The observed production MTP4 class is the SINGLE-MODULE
    MTPSpeculator (the BF16 live evidence; dispatch checks
    use_multi_module_mtp() FIRST — speculative.py:1874 — so the
    observed class implies the predicate was False at runtime).
    MultiModuleMTPSpeculator stays REJECTED for FP8: the class is
    NOT guessed from S=4 alone; only the exact profile below is
    admitted, and the actual speculator/draft identities are frozen
    during Codex startup before generation (never from CPU tests).

    Requires, on the RESOLVED SpeculativeConfig of the actual
    construction path:
      - method == 'mtp'
      - a resolved EXACT int S == 4 (num_speculative_tokens)
      - parallel_drafting is the ACTUAL bool False (None/unresolved
        rejects — truthiness is never enough)
      - draft_model_config present (the MTP construction requires it)
      - use_multi_module_mtp() returns the ACTUAL bool False (the
        single-module path; MultiModule is out of the FP8 scope; a
        None/exception return rejects — never guessed from S=4)
      - the draft KV override (speculative_config.kv_cache_dtype)
        is None (inherit the target fp8_ds_mla) or exactly
        'fp8_ds_mla' — load_eagle_model applies it as the cache
        override, so any OTHER override changes the draft's KV
        layout away from the validated DS-656 storage.
    Returns False (never raises) otherwise.
    """
    spec = vllm_config.speculative_config
    if spec is None:
        return False
    # The MISSING/MALFORMED-field handling (fail-closed): an
    # unsupported/partial config object (e.g. a SimpleNamespace
    # without the field) must return False with NO AttributeError —
    # every attribute read is guarded; a missing or malformed field
    # is a rejection, never a crash:
    if not hasattr(spec, "method") or spec.method != "mtp":
        return False
    num_spec = getattr(spec, "num_speculative_tokens", None)
    if type(num_spec) is not int or num_spec != 4:
        return False
    # The EXACT-RESOLVED flags: truthiness is NOT enough — None
    # (an unresolved/malformed config) must REJECT, not pass. Both
    # flags must be the actual bool False:
    parallel_drafting = getattr(spec, "parallel_drafting", None)
    if type(parallel_drafting) is not bool or parallel_drafting:
        return False
    if getattr(spec, "draft_model_config", None) is None:
        return False
    umm = getattr(spec, "use_multi_module_mtp", None)
    if not callable(umm):
        return False
    try:
        is_multi_module = umm()
    except Exception:
        # A malformed/unavailable use_multi_module_mtp rejects —
        # the single-module class is NEVER guessed from S=4 alone:
        return False
    if type(is_multi_module) is not bool or is_multi_module:
        return False
    draft_kv = getattr(spec, "kv_cache_dtype", None)
    return draft_kv is None or draft_kv == "fp8_ds_mla"


def _glm_kpool_fp8_mla_mtp_interleave_capable(
    vllm_config,
    fp8_mla_enabled: bool,
) -> bool:
    """Whether THIS deployment is the adapted GLM kpool FP8 DS-656
    DCP candidate with the EXACT single-module MTP4 profile.

    ``fp8_mla_enabled``: the backend's SEPARATE FP8 MLA grant (the
    default-False core flag only the wired HCU plugin indexer patch
    may set — the SAME grant the indexer validator reads). The
    legacy BF16 grant NEVER authorizes FP8: when the cache dtype is
    fp8_ds_mla this predicate requires the FP8 grant, the exact
    single-module S4 profile above, AND the full GLM kpool
    candidate conditions (real text-config identity, kpool
    compression, the indexer caches' own specs, pool-coherent
    interleave, PCP 1, TP2/8 x DCP2, ag_rs, eager compilation, no
    KV transfer) — everything the BF16 predicate checks except the
    dtype, which must be fp8_ds_mla.
    """
    parallel_config = vllm_config.parallel_config
    dcp = parallel_config.decode_context_parallel_size
    tp = parallel_config.tensor_parallel_size
    interleave = parallel_config.cp_kv_cache_interleave_size
    if interleave <= 1:
        return False
    # The EXACT topology: DCP2 only (the verified two-rank writer/
    # reader/collectives; a generic dcp>1 — e.g. TP8 x DCP4 — is
    # OUT OF SCOPE and must NOT receive this core grant), with the
    # VALID TP-group divisibility (DCP splits the TP group — the
    # pair members are adjacent TP ranks; tp % dcp == 0), and ONLY
    # the explicitly approved shapes: TP8/DCP2 (the approved model
    # candidate, world 8) or TP2/DCP2 (the verified two-rank micro
    # diagnostic, world 2):
    if type(dcp) is not int or dcp != 2:
        return False
    if type(tp) is not int or tp not in (2, 8):
        return False
    if tp % dcp != 0 or tp // dcp < 1:
        return False
    if parallel_config.prefill_context_parallel_size != 1:
        return False
    if not _fp8_single_module_mtp4_profile(vllm_config):
        return False
    if vllm_config.cache_config.cache_dtype != "fp8_ds_mla":
        return False
    if not fp8_mla_enabled:
        # The FP8 MLA grant is SEPARATE from the BF16 backend grant:
        # the legacy BF16 opt-in NEVER authorizes FP8.
        return False
    if vllm_config.kv_transfer_config is not None:
        return False
    if parallel_config.dcp_comm_backend != "ag_rs":
        return False
    from vllm.config import CompilationMode

    if vllm_config.compilation_config.mode != CompilationMode.NONE:
        return False
    # Real GLM text-config identity (same check as the BF16
    # predicate and the indexer validator).
    from vllm.transformers_utils.configs.glm5_next import Glm5NextTextConfig

    text_config = getattr(vllm_config.model_config, "hf_text_config", None)
    if not isinstance(text_config, Glm5NextTextConfig):
        return False
    if getattr(text_config, "model_type", None) != "glm5_next_text":
        return False
    if getattr(text_config, "index_kpool_compress", False) is not True:
        return False
    # The indexer caches' OWN specs (the SAME fail-closed scan as
    # the BF16 predicate; the actual uint8 fp8-naive 132-byte-row
    # container, R == kpool, pool-coherent I, block tiling).
    from vllm.models.glm5next.nvidia.attention import Glm5NextIndexerCache

    forward_context = vllm_config.compilation_config.static_forward_context
    indexer_caches = [
        layer
        for layer in forward_context.values()
        if isinstance(layer, Glm5NextIndexerCache)
    ]
    if not indexer_caches:
        return False
    for cache in indexer_caches:
        try:
            spec = cache.get_kv_cache_spec(vllm_config)
        except Exception:
            return False
        if not isinstance(spec, MLAAttentionSpec):
            return False
        # The EXACT indexer uint8 format: the actual spec's dtype
        # must BE torch.uint8 (the fp8-naive 132-byte-row container
        # the GLM indexer cache allocates). None/absent/malformed
        # REJECTS — never a silent acceptance:
        if getattr(spec, "dtype", None) is not _TORCH_UINT8_SENTINEL:
            return False
        tps = spec.tokens_per_state
        if type(tps) is not int or tps <= 1:
            return False
        if tps != getattr(text_config, "index_kpool", None):
            return False
        if interleave < tps or interleave % tps != 0:
            return False
        block = spec.block_size
        if type(block) is not int or block % interleave != 0:
            return False
    return True


def _glm_kpool_mtp_interleave_capable(vllm_config) -> bool:
    """Whether THIS deployment is the adapted GLM kpool BF16 DCP
    candidate that supports MTP with a non-trivial interleave.

    ``vllm_config``: the REAL current config (the impl's __init__
    reads it via get_current_vllm_config()). The MLA KV dtype is
    checked on the CONFIG's cache_dtype (the same value the impl's
    kv_cache_dtype resolves from); the indexer R comes from the
    indexer caches' specs in the forward context.

    Returns False (never raises) for every unadapted combination —
    the caller keeps the base-class default and the original
    compatibility rejection applies.
    """
    parallel_config = vllm_config.parallel_config
    dcp = parallel_config.decode_context_parallel_size
    interleave = parallel_config.cp_kv_cache_interleave_size
    pcp = parallel_config.prefill_context_parallel_size
    if dcp <= 1 or interleave <= 1:
        # DCP1 (any MTP) and trivial interleave are OUT OF SCOPE —
        # the capability is only about DCP>1 with I>1.
        return False
    if pcp != 1:
        return False
    if vllm_config.speculative_config is None:
        # No MTP: nothing to grant (the assert only fires with a
        # speculative config anyway).
        return False
    num_spec = vllm_config.speculative_config.num_speculative_tokens
    if type(num_spec) is not int or num_spec < 0:
        return False
    if vllm_config.speculative_config.parallel_drafting:
        # Parallel drafting is NOT in this candidate's adapted
        # scope — the per-token decode path was validated for plain
        # MTP verify rows only.
        return False
    # The candidate requires an EXPLICITLY authorized BF16 MLA KV.
    if vllm_config.cache_config.cache_dtype != "bfloat16":
        return False
    if vllm_config.kv_transfer_config is not None:
        return False
    # Real GLM text-config identity (same check as the indexer
    # validator: isinstance against the imported real class).
    from vllm.transformers_utils.configs.glm5_next import Glm5NextTextConfig

    text_config = getattr(vllm_config.model_config, "hf_text_config", None)
    if not isinstance(text_config, Glm5NextTextConfig):
        return False
    if getattr(text_config, "model_type", None) != "glm5_next_text":
        return False
    if getattr(text_config, "index_kpool_compress", False) is not True:
        return False
    # The INDEXER compress ratio R: read from the INDEXER caches'
    # OWN specs in the forward context. The caches are identified by
    # their REAL registered TYPE (Glm5NextIndexerCache — the class
    # the GLM model constructs), never by a prefix string: an
    # unrelated layer with a similar name cannot influence the
    # grant, and an unrelated layer's errors never block the target.
    from vllm.models.glm5next.nvidia.attention import Glm5NextIndexerCache

    forward_context = vllm_config.compilation_config.static_forward_context
    indexer_caches = [
        layer
        for layer in forward_context.values()
        if isinstance(layer, Glm5NextIndexerCache)
    ]
    if not indexer_caches:
        # No GLM indexer cache constructed (yet): the candidate's
        # real construction order builds the Indexer (and its cache)
        # BEFORE the MLA attention impl, so reaching here means the
        # deployment is not the candidate.
        return False
    # FAIL-CLOSED over EVERY indexer cache: a cache whose spec
    # raises, returns a non-MLAAttentionSpec, or mismatches the
    # candidate conditions rejects the whole grant — an invalid
    # cache is never skipped in favor of a valid sibling.
    for cache in indexer_caches:
        try:
            spec = cache.get_kv_cache_spec(vllm_config)
        except Exception:
            return False
        if not isinstance(spec, MLAAttentionSpec):
            return False
        tps = spec.tokens_per_state
        if type(tps) is not int or tps <= 1:
            return False
        if tps != getattr(text_config, "index_kpool", None):
            return False
        # Pool-coherent interleave: I >= R and I % R == 0 (the same
        # derivation the indexer validator applies).
        if interleave < tps or interleave % tps != 0:
            return False
        # The indexer spec's own block size must tile the interleave
        # (the actual-spec layout proof).
        block = spec.block_size
        if type(block) is not int or block % interleave != 0:
            return False
    return True


def builder_supports_dcp_varlen_with_interleave(
    vllm_config,
    backend_grants_capability: bool,
    fp8_mla_enabled: bool = False,
) -> bool:
    """Whether the metadata builder may keep varlen decodes under
    DCP>1 with a non-trivial interleave.

    Same restricted candidate as the impl capability (the real GLM
    kpool BF16 DCP conditions, including a resolved non-parallel-
    drafting MTP S) AND the backend's explicit grant. When True, the
    builder passes supports_dcp_with_varlen=True so the spec-as-
    decode reorder threshold covers 1 + S per-token rows and MTP
    verify rows route through the adapted decode path instead of
    being forced to threshold=1 (which would wrongly route them as
    prefills). Everything else keeps the original threshold=1.

    FP8 (fp8_ds_mla) additionally requires the SEPARATE
    ``fp8_mla_enabled`` grant (the default-False core flag only the
    wired HCU plugin indexer patch may set) and the EXACT
    single-module MTP4 profile; the legacy BF16 grant alone NEVER
    authorizes FP8.
    """
    if not backend_grants_capability:
        return False
    if vllm_config.cache_config.cache_dtype == "fp8_ds_mla":
        return _glm_kpool_fp8_mla_mtp_interleave_capable(vllm_config, fp8_mla_enabled)
    return _glm_kpool_mtp_interleave_capable(vllm_config)


def attention_impl_supports_mtp_with_cp_interleave(
    vllm_config,
    backend_grants_capability: bool,
    fp8_mla_enabled: bool = False,
) -> bool:
    """The instance-level capability for
    ``supports_mtp_with_cp_non_trivial_interleave_size``.

    Requires BOTH the restricted candidate conditions AND the
    backend's explicit grant (e.g. the HCU subclass sets
    ``supports_glm_kpool_bf16_dcp = True``): the core base classes
    pass ``backend_grants_capability=False`` and stay rejected.

    FP8 (fp8_ds_mla) additionally requires the SEPARATE
    ``fp8_mla_enabled`` grant and the EXACT single-module MTP4
    profile; the legacy BF16 grant alone NEVER authorizes FP8.
    """
    if not backend_grants_capability:
        return False
    if vllm_config.cache_config.cache_dtype == "fp8_ds_mla":
        return _glm_kpool_fp8_mla_mtp_interleave_capable(vllm_config, fp8_mla_enabled)
    return _glm_kpool_mtp_interleave_capable(vllm_config)
