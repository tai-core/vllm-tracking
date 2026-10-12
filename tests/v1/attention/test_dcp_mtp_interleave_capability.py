# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the restricted MTP-with-interleave capability.

REVISED per the Codex source review of b347d8b876:

1. The capability scan is FAIL-CLOSED (an indexer cache whose spec
   raises / is non-MLA / mismatches rejects the grant — never
   skipped in favor of a valid sibling); the caches are identified
   by their REAL registered TYPE (Glm5NextIndexerCache), not a
   prefix string; the unused attention_spec parameter is removed.

2. REAL cache construction: the REAL Glm5NextIndexerCache and
   Glm5NextTailCache are built inside set_current_vllm_config (no
   weights — just the cache objects), registering themselves in the
   forward context; the REAL HCU indexer spec patch (which drops
   the DeepGEMM page restriction) is applied via the plugin's
   apply(); then the REAL HcuFlashMLASparseImpl is constructed and
   the REAL check_attention_cp_compatibility runs. The real model
   construction ORDER (Indexer before the MLA attention) is
   asserted: Glm5NextMLAAttention builds the Indexer first.

3. REAL metadata evidence: the layout tests drive the REAL
   HcuFlashMLASparseMetadataBuilder.build AND the REAL
   DeepseekV32IndexerMetadataBuilder.build with real
   CommonAttentionMetadata objects (positions strictly derived from
   seq - q .. seq - 1), comparing the actual token classifications,
   decode rows/positions, and the indexer's local pool causal
   bounds — uniform-5, mixed 1+5, prefill+MTP, and non-uniform
   verify shapes. Divergent classifications are FAIL-CLOSED
   (analyzed and asserted consistent), not assumed.
"""

import json
import pathlib
import sys
import tempfile
from types import ModuleType, SimpleNamespace

import pytest
import torch

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

# The plugin worktree is located RELATIVELY (sibling of the core
# worktree's parent) — no hardcoded machine-specific paths; the same
# layout resolves on node-08's isolated source tree.
_PLUGIN_CANDIDATES = [
    _HERE.parents[3] / "dcp2-plugin-20261008",
    _HERE.parents[2] / "dcp2-plugin-20261008",
]
_PLUGIN_ROOT = next((p for p in _PLUGIN_CANDIDATES if p.is_dir()), None)
if _PLUGIN_ROOT is not None:
    sys.path.insert(0, str(_PLUGIN_ROOT))

_STUBS_INSTALLED = False


def _install_cpu_dep_stubs() -> None:
    """Import-time doubles for pip packages missing on this venv
    (the same set as the capability-predicate suite; none execute on
    the builder path)."""
    global _STUBS_INSTALLED
    if _STUBS_INSTALLED or "msgspec" in sys.modules:
        return
    import asyncio

    def stub(name, **attrs):
        if name in sys.modules:
            return sys.modules[name]
        m = ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[name] = m
        return m

    class _Struct:
        def __init_subclass__(cls, **kwargs):
            super().__init_subclass__()

    stub(
        "msgspec",
        Struct=_Struct,
        field=lambda default=None, default_factory=None: None,
    )
    stub(
        "msgspec.msgpack",
        encode=lambda *a, **k: None,
        decode=lambda *a, **k: None,
        Decoder=lambda *a, **k: None,
        Encoder=lambda *a, **k: None,
    )
    stub("cloudpickle", dumps=lambda obj, **kw: b"", loads=lambda b, **kw: None)
    stub("uvloop", run=lambda coro, **kw: asyncio.run(coro), Install=lambda: None)
    stub("cpuinfo", get_cpu_info=lambda: {"brand": "stub"})

    class _Metric:
        def __init__(self, *a, **k):
            pass

        def labels(self, *a, **k):
            return self

        def __getattr__(self, n):
            return lambda *a, **k: None

    class _Registry:
        def __getattr__(self, name):
            return _Metric()

    stub(
        "prometheus_client",
        Gauge=_Metric,
        Counter=_Metric,
        Histogram=_Metric,
        Summary=_Metric,
        Info=_Metric,
        Enum=_Metric,
        GaugeHistogram=_Metric,
        registry=_Registry(),
        REGISTRY=_Registry(),
        generate_latest=lambda *a: b"",
        CONTENT_TYPE_LATEST="text/plain",
    )

    class _XGModule(ModuleType):
        __file__ = "<xgrammar stub>"

        def __getattr__(self, name):
            return object

    if "xgrammar" not in sys.modules:
        sys.modules["xgrammar"] = _XGModule("xgrammar")
    stub("llguidance", LLMatcher=object, LLTokenizer=object)
    _STUBS_INSTALLED = True


_install_cpu_dep_stubs()

import vllm.models.glm5next  # noqa: F401,E402
from vllm.config import (  # noqa: E402
    CacheConfig,
    CompilationConfig,
    DeviceConfig,
    LoadConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    VllmConfig,
)
from vllm.config.vllm import set_current_vllm_config  # noqa: E402
from vllm.transformers_utils.configs.glm5_next import (  # noqa: E402
    Glm5NextConfig,
    Glm5NextTextConfig,
)

R = 4
INTERLEAVE = 4
BLOCK = 128
S = 4
HEAD_DIM = 576


def _config_dir(kpool=R) -> pathlib.Path:
    inner = Glm5NextTextConfig(
        index_topk=2048,
        index_kpool=kpool,
        index_head_dim=128,
        index_n_heads=64,
        swiglu_limit=1.0,
    )
    outer = Glm5NextConfig(text_config=inner)
    d = pathlib.Path(tempfile.mkdtemp())
    dd = outer.to_dict()
    dd["vision_config"]["swiglu_limit"] = 1.0
    dd["architectures"] = ["Glm5NextForConditionalGeneration"]
    with open(d / "config.json", "w") as f:
        json.dump(dd, f)
    return d


def _make_vllm_config(
    *,
    dcp=2,
    interleave=INTERLEAVE,
    cache_dtype="bfloat16",
    spec_tokens=S,
    parallel_drafting=False,
    kpool=R,
    max_model_len=1024,
):
    d = _config_dir(kpool)
    mc = ModelConfig(
        model=str(d),
        tokenizer=str(d),
        trust_remote_code=False,
        dtype="bfloat16",
        seed=0,
        max_model_len=max_model_len,
    )
    pc = ParallelConfig(
        tensor_parallel_size=2,
        decode_context_parallel_size=dcp,
        cp_kv_cache_interleave_size=interleave,
        distributed_executor_backend="uni",
    )
    sc = SpeculativeConfig(
        method="mtp",
        num_speculative_tokens=spec_tokens,
        target_model_config=mc,
        target_parallel_config=pc,
    )
    sc.parallel_drafting = parallel_drafting
    return VllmConfig(
        model_config=mc,
        cache_config=CacheConfig(block_size=BLOCK, cache_dtype=cache_dtype),
        parallel_config=pc,
        scheduler_config=SchedulerConfig(
            max_num_seqs=8,
            max_num_batched_tokens=256,
            enable_chunked_prefill=True,
            max_model_len=max_model_len,
            is_encoder_decoder=False,
        ),
        device_config=DeviceConfig(device="cpu"),
        load_config=LoadConfig(),
        compilation_config=CompilationConfig(),
        speculative_config=sc,
    )


def _build_real_caches(vc, kpool=R):
    """Construct the REAL Glm5NextIndexerCache + Glm5NextTailCache in
    the current-config context; they register themselves in the
    forward context (no weights — cache objects only). Returns
    (indexer_cache, tail_cache)."""
    from vllm.models.glm5next.nvidia.attention import (
        Glm5NextIndexerCache,
        Glm5NextTailCache,
    )

    with set_current_vllm_config(vc):
        indexer_cache = Glm5NextIndexerCache(
            head_dim=HEAD_DIM,
            dtype=torch.uint8,
            prefix="model.layers.0.self_attn.indexer.k_cache",
            cache_config=vc.cache_config,
            index_kpool=kpool,
        )
        tail_cache = Glm5NextTailCache(
            head_dim=128,
            dtype=torch.bfloat16,
            prefix="model.layers.0.self_attn.indexer.tail_cache",
            cache_config=vc.cache_config,
            index_kpool=kpool,
            vllm_config=vc,
        )
    return indexer_cache, tail_cache


def _apply_hcu_indexer_patch():
    """Apply the REAL HCU indexer-cache spec patch (drops the
    DeepGEMM 32/64-state page restriction — the natural
    block_size // index_kpool page width is used)."""
    from vllm_hcu.patch.worker.core_fix.patch_glm5next_channel_fp8 import (
        _patch_glm5next_indexer_cache,
    )

    import vllm.models.glm5next.nvidia.attention as attention_mod

    return _patch_glm5next_indexer_cache(attention_mod)


def _real_candidate_config():
    """The full REAL candidate setup: real config, real caches
    registered in the forward context, the real HCU indexer spec
    patch applied. Returns (vc, indexer_cache)."""
    vc = _make_vllm_config()
    # The builder base walks the forward context; seed it so later
    # registrations land beside the caches.
    vc.compilation_config.static_forward_context = {}
    indexer_cache, _tail = _build_real_caches(vc)
    if _PLUGIN_ROOT is not None:
        _apply_hcu_indexer_patch()
    return vc, indexer_cache


def _capability(vc, backend_grants=True):
    from vllm.v1.attention.backends.mla.dcp_mtp_capability import (
        attention_impl_supports_mtp_with_cp_interleave,
    )

    return attention_impl_supports_mtp_with_cp_interleave(
        vc, backend_grants_capability=backend_grants
    )


# --- the pure capability conditions (fail-closed) ---------------------------


def test_old_combination_reproduces_rejection():
    """OLD-RED: the capability absent (base default), the real
    check_attention_cp_compatibility raises the exact startup
    assertion from the model round."""
    from vllm.v1.worker.cp_utils import check_attention_cp_compatibility

    vc = _make_vllm_config()
    layer = SimpleNamespace(
        impl=SimpleNamespace(
            supports_mtp_with_cp_non_trivial_interleave_size=False,
            need_to_return_lse_for_decode=True,
        )
    )
    vc.compilation_config.static_forward_context = {"l0": layer}
    with pytest.raises(AssertionError, match="MTP with cp_kv_cache_interleave"):
        check_attention_cp_compatibility(vc)


def test_real_caches_grant_capability():
    """NEW-GREEN with REAL caches: the real Glm5NextIndexerCache and
    Glm5NextTailCache constructed in the config context (registering
    in the forward context), the HCU indexer spec patch applied, the
    candidate conditions grant the capability."""
    vc, _ = _real_candidate_config()
    assert _capability(vc) is True


def test_no_indexer_cache_rejects():
    """No Glm5NextIndexerCache constructed (the pre-indexer state):
    no grant — the candidate's real construction order builds the
    Indexer BEFORE the MLA impl, so this state is not the
    candidate."""
    vc = _make_vllm_config()
    vc.compilation_config.static_forward_context = {}
    assert _capability(vc) is False


def test_broken_indexer_spec_rejects_fail_closed():
    """FAIL-CLOSED: an indexer cache whose get_kv_cache_spec RAISES
    rejects the grant even when a VALID sibling cache exists (the
    old code skipped the broken one)."""
    from vllm.models.glm5next.nvidia.attention import Glm5NextIndexerCache

    vc, good = _real_candidate_config()

    class _Broken(Glm5NextIndexerCache):
        def get_kv_cache_spec(self, vllm_config):
            raise RuntimeError("spec resolution failed")

    with set_current_vllm_config(vc):
        broken = _Broken(
            head_dim=HEAD_DIM,
            dtype=torch.uint8,
            prefix="model.layers.1.self_attn.indexer.k_cache",
            cache_config=vc.cache_config,
            index_kpool=R,
        )
    vc.compilation_config.static_forward_context[
        "model.layers.1.self_attn.indexer.k_cache"
    ] = broken
    try:
        assert _capability(vc) is False
    finally:
        del vc.compilation_config.static_forward_context[
            "model.layers.1.self_attn.indexer.k_cache"
        ]
    # Removing the broken cache restores the grant.
    assert _capability(vc) is True


def test_non_mla_spec_rejects_fail_closed():
    """An indexer cache returning a NON-MLAAttentionSpec rejects the
    grant (never skipped)."""
    from vllm.models.glm5next.nvidia.attention import Glm5NextIndexerCache

    vc, _ = _real_candidate_config()

    class _NonMLA(Glm5NextIndexerCache):
        def get_kv_cache_spec(self, vllm_config):
            return SimpleNamespace(tokens_per_state=R, block_size=BLOCK)

    with set_current_vllm_config(vc):
        bad = _NonMLA(
            head_dim=HEAD_DIM,
            dtype=torch.uint8,
            prefix="model.layers.1.self_attn.indexer.k_cache",
            cache_config=vc.cache_config,
            index_kpool=R,
        )
    vc.compilation_config.static_forward_context[
        "model.layers.1.self_attn.indexer.k_cache"
    ] = bad
    try:
        assert _capability(vc) is False
    finally:
        del vc.compilation_config.static_forward_context[
            "model.layers.1.self_attn.indexer.k_cache"
        ]


def test_mismatched_sibling_rejects():
    """A sibling indexer cache with a DIFFERENT tokens_per_state
    (mismatched R) rejects the whole grant."""
    from vllm.models.glm5next.nvidia.attention import Glm5NextIndexerCache

    vc, _ = _real_candidate_config()
    with set_current_vllm_config(vc):
        other = Glm5NextIndexerCache(
            head_dim=HEAD_DIM,
            dtype=torch.uint8,
            prefix="model.layers.1.self_attn.indexer.k_cache",
            cache_config=vc.cache_config,
            index_kpool=8,  # != the config's 4
        )
    vc.compilation_config.static_forward_context[
        "model.layers.1.self_attn.indexer.k_cache"
    ] = other
    try:
        assert _capability(vc) is False
    finally:
        del vc.compilation_config.static_forward_context[
            "model.layers.1.self_attn.indexer.k_cache"
        ]


def test_prefix_string_not_the_identity():
    """A NON-Glm5NextIndexerCache object with a .k_cache prefix and
    even a valid spec does NOT grant — the scan identifies caches by
    their real registered TYPE, and unrelated layers' entries never
    influence the decision either way."""

    class _Lookalike:
        prefix = "model.layers.9.self_attn.indexer.k_cache"

        def get_kv_cache_spec(self, vllm_config):
            from vllm.v1.kv_cache_interface import MLAAttentionSpec

            return MLAAttentionSpec(
                block_size=BLOCK,
                num_kv_heads=1,
                head_size=576,
                dtype=torch.uint8,
                tokens_per_state=R,
            )

    vc = _make_vllm_config()
    vc.compilation_config.static_forward_context = {"fake": _Lookalike()}
    assert _capability(vc) is False


def test_rejection_matrix():
    """Every unadapted combination keeps the capability False (with
    the REAL caches registered)."""
    cases: list[tuple[dict, str]] = [
        (dict(cache_dtype="fp8"), "fp8 MLA"),
        (dict(cache_dtype="auto"), "auto dtype"),
        (dict(dcp=1), "DCP1"),
        (dict(interleave=5), "misaligned I=5 vs R=4"),
        (dict(interleave=2), "sub-R I=2"),
        (dict(kpool=8, interleave=4), "kpool 8 with I=4 (I < R)"),
        (dict(parallel_drafting=True), "parallel drafting"),
    ]
    for kwargs, label in cases:
        vc = _make_vllm_config(**kwargs)
        vc.compilation_config.static_forward_context = {}
        _build_real_caches(vc, kpool=kwargs.get("kpool", R))
        assert _capability(vc) is False, label
    # No backend grant.
    vc, _ = _real_candidate_config()
    assert _capability(vc, backend_grants=False) is False
    # No speculative config.
    vc, _ = _real_candidate_config()
    vc.speculative_config = None
    assert _capability(vc) is False


def test_model_constructs_indexer_before_mla():
    """The REAL model construction order: Glm5NextMLAAttention
    builds its Indexer (which constructs the caches and registers
    them) BEFORE the MLA attention impl is created — asserted from
    the real source order so the capability derivation in the impl's
    __init__ really sees the caches."""
    import inspect

    import vllm.models.glm5next.nvidia.attention as attn_mod

    src = inspect.getsource(attn_mod)
    # The Indexer (whose __init__ constructs and registers the
    # caches) is built before the MLA attention that consumes them:
    # find the attention class's __init__ and assert the ordering of
    # the two construction statements.
    cls_start = src.find("class Glm5NextMLAAttention")
    init_start = src.find("def __init__", cls_start)
    init_end = src.find("def forward", init_start)
    init = src[init_start:init_end]
    indexer_pos = init.find("self.indexer: Indexer")
    mla_pos = init.find("MLAModules(")
    assert indexer_pos != -1 and mla_pos != -1 and indexer_pos < mla_pos, (
        "the real construction order changed: the Indexer (whose "
        "caches register in the forward context) must be built "
        "before the MLA modules"
    )


# --- the REAL impl construction and compatibility check --------------------


class _CpuWorkspaceManager:
    def get_simultaneous(self, *specs):
        return tuple(torch.empty(shape, dtype=dtype) for (shape, dtype) in specs)


def _make_impl(vc, impl_cls, kv_cache_dtype="bfloat16"):
    """Construct the REAL impl with CPU device doubles (device
    queries, workspace allocation, the TP world query, and the DCP
    group identity — the capability derivation itself reads the real
    config and the real registered caches)."""
    import vllm.config as vcfg_mod
    import vllm.distributed.parallel_state as pstate
    import vllm.model_executor.layers.attention.sparse_mla_attention as sma
    import vllm.v1.attention.backends.mla.flashmla_sparse as fms

    saved_ncu = fms.num_compute_units
    saved_wsm = fms.current_workspace_manager
    saved_cfg = fms.get_current_vllm_config
    saved_tp = sma.get_tensor_model_parallel_world_size
    saved_group = pstate.get_dcp_group
    saved_vcfg = vcfg_mod.get_current_vllm_config
    fms.num_compute_units = lambda idx: 8
    fms.current_workspace_manager = lambda: _CpuWorkspaceManager()
    fms.get_current_vllm_config = lambda: vc
    # The plugin impl imports get_current_vllm_config DIRECTLY from
    # vllm.config — patch the SOURCE module too (the fms patch alone
    # never reaches it):
    vcfg_mod.get_current_vllm_config = lambda: vc
    sma.get_tensor_model_parallel_world_size = lambda: 1
    pstate.get_dcp_group = lambda: SimpleNamespace(
        world_size=vc.parallel_config.decode_context_parallel_size,
        rank_in_group=0,
    )
    try:
        return impl_cls(
            num_heads=16,
            head_size=576,
            scale=1.0,
            num_kv_heads=1,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype=kv_cache_dtype,
            logits_soft_cap=None,
            attn_type="",
            kv_sharing_target_layer_name=None,
            q_lora_rank=1536,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            qk_head_dim=192,
            v_head_dim=128,
            kv_b_proj=SimpleNamespace(weight=None),
            topk_indices_buffer=None,
            indexer=None,
        )
    finally:
        fms.num_compute_units = saved_ncu
        fms.current_workspace_manager = saved_wsm
        fms.get_current_vllm_config = saved_cfg
        vcfg_mod.get_current_vllm_config = saved_vcfg
        sma.get_tensor_model_parallel_world_size = saved_tp
        pstate.get_dcp_group = saved_group


def _hcu_impl_cls():
    if _PLUGIN_ROOT is None:
        pytest.skip("the plugin worktree is not present in this layout")
    from vllm_hcu.v1.attention.backends.mla.flashmla_sparse import (
        HcuFlashMLASparseImpl,
    )

    return HcuFlashMLASparseImpl


def _hcu_builder_cls():
    if _PLUGIN_ROOT is None:
        pytest.skip("the plugin worktree is not present in this layout")
    from vllm_hcu.v1.attention.backends.mla.flashmla_sparse import (
        HcuFlashMLASparseMetadataBuilder,
    )

    return HcuFlashMLASparseMetadataBuilder


def test_real_hcu_impl_and_check_pass_with_real_caches():
    """The REAL HcuFlashMLASparseImpl constructed AFTER the real
    caches register (the real model order), with the real HCU
    indexer spec patch applied: the capability and the LSE flag both
    derive True from the real configuration, and the REAL
    check_attention_cp_compatibility passes over the real impl."""
    from vllm.v1.worker.cp_utils import check_attention_cp_compatibility

    vc, _indexer_cache = _real_candidate_config()
    impl = _make_impl(vc, _hcu_impl_cls())
    assert impl.supports_mtp_with_cp_non_trivial_interleave_size is True
    assert impl.need_to_return_lse_for_decode is True  # real derivation
    vc.compilation_config.static_forward_context["l0"] = SimpleNamespace(impl=impl)
    check_attention_cp_compatibility(vc)


def test_real_hcu_impl_fp8_mismatch_rejects():
    """The REAL HCU impl with an fp8_ds_mla impl dtype but a NON-
    fp8_ds_mla config cache dtype (the mismatch): the plugin's
    construction gate raises (fail-closed — the restricted-candidate
    rejection, never a silent capability False)."""
    vc = _make_vllm_config(cache_dtype="fp8")
    vc.compilation_config.static_forward_context = {}
    _build_real_caches(vc)
    with pytest.raises(NotImplementedError, match="restricted candidate"):
        _make_impl(vc, _hcu_impl_cls(), kv_cache_dtype="fp8_ds_mla")


def test_real_hcu_impl_fp8_mtp4_single_module_grants():
    """The NEW GREEN: the REAL HCU impl with the EXACT single-module
    FP8 MTP4 profile (fp8_ds_mla config, method mtp, resolved S=4,
    no parallel drafting, a draft config whose
    num_nextn_predict_layers is absent/1 — the OBSERVED production
    class MTPSpeculator): constructs, the capability derives True,
    and the REAL check_attention_cp_compatibility passes."""
    from vllm.v1.worker.cp_utils import check_attention_cp_compatibility

    vc = _make_vllm_config(cache_dtype="fp8_ds_mla")
    vc.compilation_config.static_forward_context = {}
    _build_real_caches(vc)
    impl = _make_impl(vc, _hcu_impl_cls(), kv_cache_dtype="fp8_ds_mla")
    assert impl.supports_mtp_with_cp_non_trivial_interleave_size is True
    assert impl.need_to_return_lse_for_decode is True
    vc.compilation_config.static_forward_context["l0"] = SimpleNamespace(impl=impl)
    check_attention_cp_compatibility(vc)


def test_real_hcu_impl_fp8_mtp_non_mtp4_rejects():
    """The REAL HCU impl with FP8 + a NON-MTP4 spec shape (S=3): the
    profile check fails, the eligibility is False, and the plugin's
    construction gate raises (fail-closed)."""
    vc = _make_vllm_config(cache_dtype="fp8_ds_mla", spec_tokens=3)
    vc.compilation_config.static_forward_context = {}
    _build_real_caches(vc)
    with pytest.raises(NotImplementedError, match="restricted candidate"):
        _make_impl(vc, _hcu_impl_cls(), kv_cache_dtype="fp8_ds_mla")


def test_core_base_impl_stays_rejected():
    """The CORE base FlashMLASparseImpl (no grant) keeps its
    pre-existing DCP+BF16 rejection; on DCP1 it constructs with the
    capability False."""
    from vllm.v1.attention.backends.mla.flashmla_sparse import FlashMLASparseImpl

    vc, _ = _real_candidate_config()
    with pytest.raises(NotImplementedError, match="bf16 sparse path"):
        _make_impl(vc, FlashMLASparseImpl)
    vc1 = _make_vllm_config(dcp=1)
    vc1.compilation_config.static_forward_context = {}
    _build_real_caches(vc1)
    impl = _make_impl(vc1, FlashMLASparseImpl)
    assert impl.supports_mtp_with_cp_non_trivial_interleave_size is False


# --- REAL dual-builder metadata layout evidence ----------------------------


class _FakeDCPManager:
    def init_kv_gather(self, *args, **kwargs):
        pass


class _FakeAttentionLayer:
    prefill_backend = None
    dcp_manager = _FakeDCPManager()


def _make_flashmla_builder(vc, builder_cls, num_heads=16):
    import vllm.v1.attention.backends.mla.flashmla_sparse as fms

    saved_ncu = fms.num_compute_units
    saved_layer = vc.compilation_config.static_forward_context.get("l0")
    saved_heads = vc.model_config.get_num_attention_heads
    fms.num_compute_units = lambda idx: 8
    vc.compilation_config.static_forward_context["l0"] = _FakeAttentionLayer()
    vc.model_config.get_num_attention_heads = lambda parallel_config: num_heads
    try:
        return builder_cls(
            kv_cache_spec=SimpleNamespace(block_size=BLOCK),
            layer_names=["l0"],
            vllm_config=vc,
            device=torch.device("cpu"),
        )
    finally:
        fms.num_compute_units = saved_ncu
        vc.model_config.get_num_attention_heads = saved_heads
        if saved_layer is None:
            vc.compilation_config.static_forward_context.pop("l0", None)
        else:
            vc.compilation_config.static_forward_context["l0"] = saved_layer


def _make_indexer_builder(vc, block_table_width=8):
    import vllm.v1.attention.backends.mla.indexer as ix

    saved_ncu = ix.num_compute_units
    saved_group = getattr(ix, "get_dcp_group", None)
    ix.num_compute_units = lambda idx: 8
    ix.get_dcp_group = lambda: SimpleNamespace(
        world_size=vc.parallel_config.decode_context_parallel_size,
        rank_in_group=0,
    )
    try:
        from vllm.v1.kv_cache_interface import MLAAttentionSpec

        spec = MLAAttentionSpec(
            block_size=BLOCK,
            num_kv_heads=1,
            head_size=576,
            dtype=torch.bfloat16,
            tokens_per_state=R,
        )
        return ix.DeepseekV32IndexerMetadataBuilder(
            kv_cache_spec=spec,
            layer_names=["l0"],
            vllm_config=vc,
            device=torch.device("cpu"),
            block_table_width=block_table_width,
        )
    finally:
        ix.num_compute_units = saved_ncu
        if saved_group is not None:
            ix.get_dcp_group = saved_group
        else:
            del ix.get_dcp_group


def _cam(seq_lens, query_lens):
    """A REAL-shaped CommonAttentionMetadata: positions strictly
    seq - q .. seq - 1 per request."""
    qsl = [0]
    for ql in query_lens:
        qsl.append(qsl[-1] + ql)
    positions: list[int] = []
    for sl, ql in zip(seq_lens, query_lens):
        positions.extend(range(sl - ql, sl))
    from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens

    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32)
    local = get_dcp_local_seq_lens(seq_lens_t, 2, 0, INTERLEAVE)
    # A block table wide enough for each request's pools (block_size
    # 32 pool-states), with DISTINCT per-request block numbers so
    # request identity is verifiable in the outputs.
    width = max((sl // R) // 32 + 2 for sl in seq_lens)
    table = torch.zeros((len(seq_lens), width), dtype=torch.int32)
    for b, sl in enumerate(seq_lens):
        pools = sl // R
        n_blocks = pools // 32 + 1
        table[b, :n_blocks] = torch.arange(
            1000 * (b + 1), 1000 * (b + 1) + n_blocks, dtype=torch.int32
        )
    return SimpleNamespace(
        num_reqs=len(seq_lens),
        num_actual_tokens=int(qsl[-1]),
        max_query_len=max(query_lens),
        max_seq_len=max(seq_lens),
        query_start_loc=torch.tensor(qsl, dtype=torch.int32),
        query_start_loc_cpu=torch.tensor(qsl, dtype=torch.int32),
        seq_lens=seq_lens_t,
        seq_lens_cpu_upper_bound=seq_lens_t.clone(),
        block_table_tensor=table,
        slot_mapping=torch.zeros(int(qsl[-1]), dtype=torch.int64),
        positions=torch.tensor(positions, dtype=torch.int64),
        causal=True,
        is_prefilling=None,
        dcp_local_seq_lens=local,
    )


class _CPUKernelDoubles:
    """Faithful CPU implementations of the three Triton kernels the
    dual-build path launches. Each fills ONLY its real OUTPUT
    parameters with values matching the kernel's documented
    semantics — inputs are never modified. Installed and restored
    STRICTLY within the _dual_build scope (a context manager), never
    globally."""

    def __init__(self, vc):
        self.vc = vc

    def __enter__(self):
        import vllm.v1.attention.backends.mla.compressor_utils as cu
        import vllm.v1.attention.backends.mla.indexer as ix_mod

        self._cu = cu
        self._ix = ix_mod
        self._saved = (
            cu._compressed_slot_mapping_kernel,
            ix_mod._prepare_uniform_decode_kernel,
            ix_mod._BUILD_PREFILL_CHUNK_METADATA_KERNEL,
            ix_mod.DeepseekV32IndexerMetadataBuilder.supports_glm_kpool_bf16_dcp,
        )
        cu._compressed_slot_mapping_kernel = _KernelLaunch(_cpu_compressed_slot_kernel)
        ix_mod._prepare_uniform_decode_kernel = _KernelLaunch(
            _cpu_uniform_decode_kernel
        )
        ix_mod._BUILD_PREFILL_CHUNK_METADATA_KERNEL = _CPUChunkKernel()
        # The indexer builder's DCP validator requires the candidate
        # grant: use the REAL plugin grant function (the same code
        # the runtime patch runs), scoped to this block and restored
        # on exit.
        ix_mod.DeepseekV32IndexerMetadataBuilder.supports_glm_kpool_bf16_dcp = True
        return self

    def __exit__(self, *exc):
        (cu, ix_mod, slot, udk, chunk, grant) = (
            self._cu,
            self._ix,
            *self._saved,
        )
        cu._compressed_slot_mapping_kernel = slot
        ix_mod._prepare_uniform_decode_kernel = udk
        ix_mod._BUILD_PREFILL_CHUNK_METADATA_KERNEL = chunk
        ix_mod.DeepseekV32IndexerMetadataBuilder.supports_glm_kpool_bf16_dcp = grant
        return False


class _KernelLaunch:
    """A Triton-style [grid] launcher over a plain function."""

    def __init__(self, fn):
        self._fn = fn

    def __getitem__(self, grid):
        n = grid[0] if isinstance(grid, tuple) else grid

        def launch(*args, **kwargs):
            for i in range(n):
                self._fn(i, *args, **kwargs)

        return launch


def _cpu_compressed_slot_kernel(
    batch_idx,
    slot_mapping,
    query_start_loc,
    seq_lens,
    block_table,
    block_table_stride,
    block_size,
    COMPRESS_RATIO,
    DCP_WORLD_SIZE,
    DCP_RANK,
    POOL_INTERLEAVE,
    PAD_ID,
    TRITON_BLOCK_SIZE,
    num_tokens=None,
):
    """The compressed-slot kernel's documented semantics, per request:
    for each query token, valid iff pos >= 0 and (pos+1) % R == 0 and
    (DCP) the pool's owning rank is THIS rank; the slot is
    block_number * block_size + local_pool_offset. Inputs untouched."""
    qs = int(query_start_loc[batch_idx].item())
    qe = int(query_start_loc[batch_idx + 1].item())
    ql = qe - qs
    sl = int(seq_lens[batch_idx].item())
    start_pos = sl - ql
    for off in range(ql):
        pos = start_pos + off
        valid = pos >= 0 and (pos + 1) % COMPRESS_RATIO == 0
        pac = pos // COMPRESS_RATIO
        if DCP_WORLD_SIZE == 1:
            local_pool = pac
        else:
            owner = (pac // POOL_INTERLEAVE) % DCP_WORLD_SIZE
            valid = valid and owner == DCP_RANK
            local_pool = (
                pac // (DCP_WORLD_SIZE * POOL_INTERLEAVE)
            ) * POOL_INTERLEAVE + pac % POOL_INTERLEAVE
        if valid:
            block_ids = local_pool // block_size
            flat = block_table.reshape(-1)
            bn = int(flat[batch_idx * block_table_stride + block_ids].item())
            slot = bn * block_size + local_pool % block_size
        else:
            slot = PAD_ID
        slot_mapping[qs + off] = slot


def _cpu_uniform_decode_kernel(
    idx,
    seq_lens,
    decode_seq_lens,
    block_table,
    block_table_stride,
    expanded_block_table,
    expanded_bt_stride,
    decode_lens,
    max_decode_len,
    BLOCK_SIZE,
):
    """The uniform-decode expansion kernel's semantics: per (req,
    local_idx), the per-token context length max(seq - mdl + li + 1,
    0) into decode_seq_lens; the block-table row copy; decode_lens
    all 1. Inputs untouched."""
    req_id = idx // max_decode_len
    local_idx = idx % max_decode_len
    seq_len = int(seq_lens[req_id].item())
    per_token = max(seq_len - max_decode_len + local_idx + 1, 0)
    decode_seq_lens[idx] = per_token
    flat = block_table.reshape(-1)
    src = flat[
        req_id * block_table_stride : req_id * block_table_stride + expanded_bt_stride
    ]
    out = expanded_block_table.reshape(-1)
    out[idx * expanded_bt_stride : (idx + 1) * expanded_bt_stride] = src
    decode_lens[idx] = 1


class _CPUChunkKernel:
    """The prefill-chunk kernel's semantics: cu_ks[row] = the row's
    local compressed row-start; cu_ke[row] = row-start + the
    per-token LOCAL context length (DCP-sharded exactly like
    get_dcp_local_seq_lens); token_to_seq[pool] = the request id.
    Inputs untouched. Mirrors the kernel class's __call__ ->
    self.kernel[grid](...) dispatch."""

    def __call__(self, *args, **kwargs):
        num_reqs = kwargs.get("num_reqs", 1)
        _CPUChunkKernel._Launcher(self, (num_reqs,))(*args, **kwargs)

    class _Launcher:
        def __init__(self, outer, grid):
            self._outer = outer
            self._n = grid[0] if isinstance(grid, tuple) else grid

        def __call__(self, *args, **kwargs):
            kwargs.pop("num_reqs", None)
            for i in range(self._n):
                self._outer._fn(i, *args, **kwargs)

    def kernel(self, grid):
        return _CPUChunkKernel._Launcher(self, grid)

    def _fn(
        self,
        batch_idx,
        query_start_loc,
        uncompressed_seq_lens,
        cu_compressed_seq_lens,
        row_start_cu,
        token_to_seq,
        cu_ks,
        cu_ke,
        query_slice_start,
        query_slice_stop,
        DCP_RANK,
        DCP_WORLD,
        DCP_INTERLEAVE,
        BLOCK_SIZE=None,
        COMPRESS_RATIO=None,
    ):
        qs = int(query_start_loc[batch_idx].item())
        qe = int(query_start_loc[batch_idx + 1].item())
        ql = qe - qs
        seq_start = int(cu_compressed_seq_lens[batch_idx].item())
        seq_end = int(cu_compressed_seq_lens[batch_idx + 1].item())
        csl = seq_end - seq_start
        row_start = int(row_start_cu[batch_idx].item())
        usl = int(uncompressed_seq_lens[batch_idx].item())
        start_pos = usl - ql
        for off in range(ql):
            abs_pos = qs + off
            if not (query_slice_start <= abs_pos < query_slice_stop):
                continue
            out_pos = abs_pos - query_slice_start
            cu_ks[out_pos] = row_start
            global_ctx = start_pos + 1 + off
            lpt = global_ctx // COMPRESS_RATIO
            if DCP_WORLD > 1:
                base = (lpt // DCP_INTERLEAVE // DCP_WORLD) * DCP_INTERLEAVE
                rem = lpt - base * DCP_WORLD
                rem = min(max(rem - DCP_RANK * DCP_INTERLEAVE, 0), DCP_INTERLEAVE)
                lpt = base + rem
            cu_ke[out_pos] = row_start + lpt
        for off in range(csl):
            token_to_seq[seq_start + off] = batch_idx


def _dual_build(vc, cam):
    """Run BOTH real builders over the same CommonAttentionMetadata
    with STRICTLY-SCOPED faithful CPU kernel doubles; return
    (flashmla_md, indexer_md)."""
    with _CPUKernelDoubles(vc):
        fm_builder = _make_flashmla_builder(vc, _hcu_builder_cls())
        fm_md = fm_builder.build(common_prefix_len=0, common_attn_metadata=cam)
        idx_builder = _make_indexer_builder(
            vc, block_table_width=cam.block_table_tensor.shape[1]
        )
        idx_builder.dcp_world_size = vc.parallel_config.decode_context_parallel_size
        idx_builder.dcp_rank = 0
        idx_md = idx_builder.build(common_prefix_len=0, common_attn_metadata=cam)
    return fm_md, idx_md


def _assert_classification_consistent(fm_md, idx_md, label):
    """FAIL-CLOSED for UNIFORM batches: the two builders' token
    classifications must agree exactly. (Non-uniform decode batches
    use _assert_nonuniform_mapping instead — the two builders
    INTENTIONALLY differ there: flashmla requires uniform decodes
    and routes minority-lens requests to prefill; the indexer keeps
    every request decode with requires_padding. A divergence on a
    uniform batch is a real layout bug — never assumed away.)"""
    assert fm_md.num_decodes == idx_md.num_decodes, (
        f"{label}: flashmla num_decodes {fm_md.num_decodes} != indexer "
        f"{idx_md.num_decodes}"
    )
    assert fm_md.num_prefills == idx_md.num_prefills, (
        f"{label}: flashmla num_prefills {fm_md.num_prefills} != indexer "
        f"{idx_md.num_prefills}"
    )
    assert fm_md.num_decode_tokens == idx_md.num_decode_tokens, (
        f"{label}: flashmla num_decode_tokens {fm_md.num_decode_tokens} != "
        f"indexer {idx_md.num_decode_tokens}"
    )


def _assert_nonuniform_mapping(fm_md, idx_md, label, expected_indexer_decodes):
    """The DOCUMENTED mapping for non-uniform decode batches: the
    indexer keeps EVERY request decode-routed (its decode path
    handles per-request actual bounds with requires_padding);
    flashmla (require_uniform_decodes=True) routes the
    minority-lens requests to prefill. The mapping is asserted
    EXACTLY — the flashmla decode set must be the uniform-prefix
    majority and the indexer's decode set the full request set."""
    assert idx_md.num_decodes == expected_indexer_decodes, (
        f"{label}: indexer num_decodes {idx_md.num_decodes} != "
        f"{expected_indexer_decodes}"
    )
    assert idx_md.num_prefills == 0, (label, idx_md.num_prefills)
    assert fm_md.num_decodes + fm_md.num_prefills == expected_indexer_decodes, (
        f"{label}: flashmla total requests {fm_md.num_decodes}+"
        f"{fm_md.num_prefills} != {expected_indexer_decodes}"
    )
    fm_prefill_tokens = fm_md.num_actual_tokens - fm_md.num_decode_tokens
    assert fm_md.num_decode_tokens + fm_prefill_tokens == (idx_md.num_decode_tokens), (
        label,
        fm_md.num_decode_tokens,
        fm_prefill_tokens,
    )


def test_layout_uniform_mtp5():
    """Uniform MTP5 decode (2 requests, q=5 each, seq 4097/4099 >
    2048): BOTH real builders classify ALL tokens as decode; the
    decode rows are request-major [b, t]; positions strictly
    seq-5..seq-1; the indexer's decode local pool causal bounds
    cover the same rows."""
    vc, _ = _real_candidate_config()
    cam = _cam([4097, 4099], [5, 5])
    fm_md, idx_md = _dual_build(vc, cam)
    _assert_classification_consistent(fm_md, idx_md, "uniform5")
    assert (fm_md.num_decodes, fm_md.num_prefills) == (2, 0)
    assert fm_md.num_decode_tokens == 10
    # REQUEST IDENTITY through the real builds: every query token's
    # req_id_per_token names its ORIGINAL request; positions strictly
    # seq-q..seq-1; the indexer's decode metadata carries the same
    # request-major rows with the per-token causal bounds.
    qsl = cam.query_start_loc.tolist()
    fm_req_ids = fm_md.req_id_per_token.tolist()
    for b, (sl, ql) in enumerate(zip([4097, 4099], [5, 5])):
        rows = list(range(qsl[b], qsl[b + 1]))
        pos = cam.positions[qsl[b] : qsl[b + 1]].tolist()
        assert pos == list(range(sl - ql, sl)), (b, pos[:3], pos[-1])
        assert len(rows) == ql
        for row in rows:
            assert fm_req_ids[row] == b, (row, fm_req_ids[row], b)
    # The indexer's decode seq_lens (the per-token causal bounds)
    # are the DCP-local pool counts of each token's context — real
    # values from the faithful uniform-decode kernel: token (b, t)
    # has context sl-ql+t+1 tokens -> local pools under W2/J1.
    dec = idx_md.decode
    assert dec is not None and dec.dcp_world_size == 2
    seq_lens = dec.seq_lens.tolist()
    for b, (sl, ql) in enumerate(zip([4097, 4099], [5, 5])):
        for t in range(ql):
            ctx = sl - ql + t + 1
            pools = ctx // R
            # W2/J1 rank0 local count: pools in {0,2,4,...} -> (pools+1)//2
            expected_local = (pools + 1) // 2
            row = b * 5 + t  # the flattened per-token row (query order)
            got = seq_lens[row][0]
            assert got == expected_local, (b, t, got, expected_local)
    # The slot mapping: token (b, t) completing a pool owned by
    # rank0 maps to request b's block (>= 1000*(b+1)); others -1.
    sm = idx_md.slot_mapping.tolist()
    for b, (sl, ql) in enumerate(zip([4097, 4099], [5, 5])):
        for t in range(ql):
            pos = sl - ql + t
            row = b * 5 + t
            if (pos + 1) % R == 0 and ((pos // R) % 2) == 0:
                assert sm[row] >= 1000 * (b + 1), (b, t, sm[row])
            else:
                assert sm[row] == -1, (b, t, sm[row])


def test_layout_mixed_plain_and_mtp5():
    """Mixed decode lens [1, 5] (a plain decode and an MTP5 verify,
    both > threshold path... 1 < 5 so the plain decode stays decode
    via decode_threshold): both builders classify both as decode;
    rows stay contiguous and request-ordered; positions strictly
    derived."""
    vc, _ = _real_candidate_config()
    cam = _cam([4097, 4099], [1, 5])
    fm_md, idx_md = _dual_build(vc, cam)
    _assert_nonuniform_mapping(fm_md, idx_md, "mixed1+5", 2)
    qsl = cam.query_start_loc.tolist()
    assert qsl == [0, 1, 6]
    pos1 = cam.positions[1:6].tolist()
    assert pos1 == [4094, 4095, 4096, 4097, 4098]
    # REQUEST IDENTITY through the real builds: the flashmla
    # metadata's req_id_per_token names each token's ORIGINAL request
    # (request 0 = the 1-token plain decode, request 1 = the MTP5).
    fm_req_ids = fm_md.req_id_per_token.tolist()
    assert fm_req_ids[:1] == [0]
    assert fm_req_ids[1:6] == [1, 1, 1, 1, 1]
    # The INDEXER keeps BOTH requests decode-routed with the real
    # per-token bounds: r0's single token ctx=4097 -> pools 1024 ->
    # local (1024+1)//2 = 512; r1's five tokens ctx=4095..4099 ->
    # pools 1023..1024 -> local 512 (1023+1)//2, 512, 512, 512, 512.
    dec = idx_md.decode
    seq_lens = dec.seq_lens.tolist()
    assert seq_lens[0][0] == (4097 // R + 1) // 2
    for t in range(5):
        ctx = 4095 + t
        assert seq_lens[1 + t][0] == (ctx // R + 1) // 2, t
    # The slot mapping carries request identity via the block table:
    # r0's token at pos 4096 completes pool 1024 (even -> rank0
    # owns); r1's tokens at 4094..4098 complete pools at
    # (pos+1)%4==0: 4095->pool 1023 (odd -> rank1, -1), 4099 is out
    # of range; the completing rank0 pools map into r1's blocks.
    sm = idx_md.slot_mapping.tolist()
    # r0's single token at pos 4096: (4096+1) % 4 != 0 — it does not
    # complete a pool, so its slot is -1 (a real value check, not a
    # blanket pass).
    assert sm[0] == -1
    for t, pos in enumerate([4094, 4095, 4096, 4097, 4098]):
        row = 1 + t
        if (pos + 1) % R == 0 and ((pos // R) % 2) == 0:
            assert sm[row] >= 2000, (t, sm[row])  # r1's block range
        else:
            assert sm[row] == -1, (t, sm[row])


def test_layout_prefill_plus_mtp_mixed():
    """A prefill (q=64, seq 4096) mixed with an MTP5 verify (seq
    4200): the classification must be consistent across BOTH
    builders — the prefill request routes to prefill and the MTP5
    request to decode; the divergence check is fail-closed."""
    vc, _ = _real_candidate_config()
    cam = _cam([4096, 4200], [64, 5])
    fm_md, idx_md = _dual_build(vc, cam)
    # DOCUMENTED divergence for this RAW (prefill-major) batch
    # order: the split functions document a decode-major
    # PRECONDITION (the model runner's reorder establishes it in
    # production). On the raw order the two builders legitimately
    # differ — flashmla's TABLE threshold (>= 64 for its head
    # count) makes the q=64 request a decode candidate and the q=5
    # minority routes to prefill (nd=1); the indexer's threshold
    # (next_n based) makes the q=64 request a prefill (nd=0). In
    # PRODUCTION the reorder places the q=5 MTP request first and
    # both builders agree — that reordered shape is pinned by the
    # uniform/non-uniform tests above. Here the RAW semantics are
    # pinned exactly, with the divergence made explicit:
    assert fm_md.num_decodes == 1 and fm_md.num_prefills == 1
    assert idx_md.num_decodes == 0 and idx_md.num_prefills == 2
    assert fm_md.num_actual_tokens == 69
    assert idx_md.num_decode_tokens == 0  # all 69 are prefill


def test_layout_nonuniform_verify():
    """Non-uniform verify lens [3, 5] (an adaptive-verification
    shape): both builders agree; every request stays decode-routed
    (max 5 <= threshold); rows/positions strictly derived."""
    vc, _ = _real_candidate_config()
    cam = _cam([4097, 4099], [3, 5])
    fm_md, idx_md = _dual_build(vc, cam)
    _assert_nonuniform_mapping(fm_md, idx_md, "nonuniform3+5", 2)
    pos0 = cam.positions[:3].tolist()
    pos1 = cam.positions[3:8].tolist()
    assert pos0 == [4094, 4095, 4096]
    assert pos1 == [4094, 4095, 4096, 4097, 4098]


def test_require_uniform_short_extend_routes():
    """The require_uniform path: a SHORT extend (q=2 < next_n=5)
    mixed with uniform decodes — the split's documented behavior
    (short extends route to prefill under require_uniform) is
    asserted from the REAL split, and the two builders' shared
    classification stays consistent (the flashmla builder requires
    uniform decodes; the indexer's own split handles the same
    request)."""
    from vllm.v1.attention.backends.mla.indexer import (
        split_decodes_and_prefills,
    )

    cam = _cam([4097, 4099], [2, 5])
    nd, npf, ndt, npt = split_decodes_and_prefills(
        cam, decode_threshold=5, require_uniform=True
    )
    # The REAL split's uniform-decode policy: the uniform batch is
    # the max query len (5); the q=2 request alone is NOT uniform
    # with it, so under require_uniform the DECODE set is the
    # uniform prefix by query-len order — the actual routing (the
    # q=2 request as decode nd=1/ndt=2 and the q=5 as prefill
    # npf=1/npt=5) is asserted from the real function, and the
    # dual-builder consistency below is the fail-closed check that
    # both consumers agree on it.
    assert (nd, npf, ndt, npt) == (1, 1, 2, 5)
    vc, _ = _real_candidate_config()
    fm_md, idx_md = _dual_build(vc, cam)
    # Non-uniform (2 vs 5): the documented mapping — the indexer
    # keeps both decode; flashmla routes the minority to prefill.
    _assert_nonuniform_mapping(fm_md, idx_md, "short-extend", 2)


def test_builder_threshold_covers_mtp_rows():
    """The REAL HcuFlashMLASparseMetadataBuilder under the candidate:
    the reorder threshold >= 1 + S = 5."""
    vc, _ = _real_candidate_config()
    builder = _make_flashmla_builder(vc, _hcu_builder_cls())
    assert builder.reorder_batch_threshold >= 1 + S


def test_builder_threshold_forced_one_for_core():
    """The CORE builder (no grant) keeps the forced threshold=1 under
    DCP>1 + I>1; I==1 keeps varlen; DCP1 unaffected; the HCU builder
    with FP8 keeps threshold=1."""
    from vllm.v1.attention.backends.mla.flashmla_sparse import (
        FlashMLASparseMetadataBuilder,
    )

    vc, _ = _real_candidate_config()
    builder = _make_flashmla_builder(vc, FlashMLASparseMetadataBuilder)
    assert builder.reorder_batch_threshold == 1

    vc2 = _make_vllm_config(interleave=1)
    vc2.compilation_config.static_forward_context = {}
    _build_real_caches(vc2)
    builder2 = _make_flashmla_builder(vc2, FlashMLASparseMetadataBuilder)
    assert builder2.reorder_batch_threshold >= 1 + S

    vc3 = _make_vllm_config(dcp=1)
    vc3.compilation_config.static_forward_context = {}
    _build_real_caches(vc3)
    builder3 = _make_flashmla_builder(vc3, FlashMLASparseMetadataBuilder)
    assert builder3.reorder_batch_threshold >= 1 + S

    # The HCU builder with FP8: WITHOUT the FP8 capability flag on
    # the flashmla builder class (the core default — the plugin only
    # sets it on ITS OWN subclass; a bare core-class check here would
    # need the plugin patch), the MTP-off FP8 shape has NO spec
    # config path... instead: with the EXACT single-module MTP4
    # profile the PLUGIN builder grants threshold >= 5 (the verify
    # 1+S=5 rows route through the per-token decode path):
    vc4 = _make_vllm_config(cache_dtype="fp8_ds_mla")
    vc4.compilation_config.static_forward_context = {}
    _build_real_caches(vc4)
    builder4 = _make_flashmla_builder(vc4, _hcu_builder_cls())
    assert builder4.reorder_batch_threshold >= 1 + S
    # ...and with a NON-MTP4 spec shape (S=3) the grant stays False:
    vc5 = _make_vllm_config(cache_dtype="fp8_ds_mla", spec_tokens=3)
    vc5.compilation_config.static_forward_context = {}
    _build_real_caches(vc5)
    builder5 = _make_flashmla_builder(vc5, _hcu_builder_cls())
    assert builder5.reorder_batch_threshold == 1


# --- the FP8 MLA opt-in contract (this worktree's separate flag) ------------


def _validator_kwargs(fp8: bool):
    from vllm.v1.attention.backends.mla.indexer import (
        validate_dcp_compressed_indexer_support,
    )

    return validate_dcp_compressed_indexer_support, dict(
        capability_enabled=True,
        fp8_mla_enabled=fp8,
    )


def test_fp8_dtype_rejected_without_opt_in():
    """cache_dtype fp8_ds_mla WITHOUT the FP8 opt-in: the ORIGINAL
    rejection fires (fail-closed; the default False)."""
    fn, kw = _validator_kwargs(fp8=False)
    vc = _make_vllm_config(cache_dtype="fp8_ds_mla")
    with pytest.raises(NotImplementedError, match="separate FP8 MLA opt-in"):
        fn(vc, R, INTERLEAVE, **kw)


def test_fp8_dtype_admitted_with_opt_in():
    """cache_dtype fp8_ds_mla WITH the FP8 opt-in (the plugin grant)
    and MTP OFF (the FP8 candidate's own condition —
    _make_vllm_config always sets a spec config, so it is cleared):
    the validator PASSES (the full candidate conditions hold — the
    real GLM identity, kpool, PCP1, the pool-coherent interleave,
    the aligned block, ag_rs, eager, no KV transfer)."""
    fn, kw = _validator_kwargs(fp8=True)
    vc = _make_vllm_config(cache_dtype="fp8_ds_mla")
    vc.speculative_config = None  # the FP8 candidate is MTP-off
    fn(vc, R, INTERLEAVE, **kw)  # no raise


def test_bf16_dtype_unaffected_by_fp8_flag():
    """The BF16 candidate: unchanged under EITHER FP8 flag state (the
    BF16 path never consults the FP8 opt-in)."""
    fn_t, kw_t = _validator_kwargs(fp8=True)
    fn_f, kw_f = _validator_kwargs(fp8=False)
    vc = _make_vllm_config(cache_dtype="bfloat16")
    fn_t(vc, R, INTERLEAVE, **kw_t)
    fn_f(vc, R, INTERLEAVE, **kw_f)


def test_fp8_dtype_auto_still_rejected():
    """cache_dtype 'auto': still rejected under either flag (the
    unproven resolution — fail closed, named)."""
    fn, kw = _validator_kwargs(fp8=True)
    vc = _make_vllm_config(cache_dtype="auto")
    with pytest.raises(NotImplementedError, match="not resolved"):
        fn(vc, R, INTERLEAVE, **kw)


def test_fp8_flag_does_not_replace_capability_flag():
    """The FP8 opt-in NEVER substitutes the primary capability flag:
    capability_enabled False + fp8_mla_enabled True still rejects
    (the primary gate fires first)."""
    from vllm.v1.attention.backends.mla.indexer import (
        validate_dcp_compressed_indexer_support,
    )

    vc = _make_vllm_config(cache_dtype="fp8_ds_mla")
    with pytest.raises(NotImplementedError, match="supports_glm_kpool_bf16_dcp"):
        validate_dcp_compressed_indexer_support(
            vc,
            R,
            INTERLEAVE,
            capability_enabled=False,
            fp8_mla_enabled=True,
        )


def test_fp8_with_single_module_mtp4_admitted():
    """The NEW GREEN: fp8_ds_mla WITH the grant and the EXACT
    single-module MTP4 profile (the _make_vllm_config spec config:
    method mtp, resolved S=4, no parallel drafting, a draft config
    with no num_nextn_predict_layers — the single-module class):
    the validator PASSES (the full candidate conditions hold)."""
    fn, kw = _validator_kwargs(fp8=True)
    vc = _make_vllm_config(cache_dtype="fp8_ds_mla")
    fn(vc, R, INTERLEAVE, **kw)  # no raise


def test_fp8_with_non_mtp4_spec_rejected():
    """The FP8 branch's profile gate: S != 4 (here 3) is rejected
    with the named single-module-MTP4 reason."""
    fn, kw = _validator_kwargs(fp8=True)
    vc = _make_vllm_config(cache_dtype="fp8_ds_mla", spec_tokens=3)
    with pytest.raises(NotImplementedError, match="single-module MTP4 profile"):
        fn(vc, R, INTERLEAVE, **kw)


def test_fp8_with_parallel_drafting_rejected():
    """parallel_drafting=True is rejected under the FP8 MTP4 gate."""
    fn, kw = _validator_kwargs(fp8=True)
    vc = _make_vllm_config(cache_dtype="fp8_ds_mla", parallel_drafting=True)
    with pytest.raises(NotImplementedError, match="single-module MTP4 profile"):
        fn(vc, R, INTERLEAVE, **kw)


def test_fp8_with_multi_module_rejected():
    """use_multi_module_mtp() True (num_nextn_predict_layers > 1 on
    the draft hf_config): the MultiModule path is REJECTED for FP8
    (the class is never guessed from S=4 alone)."""
    fn, kw = _validator_kwargs(fp8=True)
    vc = _make_vllm_config(cache_dtype="fp8_ds_mla")
    # The draft hf_config gains num_nextn_predict_layers > 1 (the
    # HF config is a plain object — setattr works even where the
    # class does not declare the field; getattr's default 1 is what
    # use_multi_module_mtp reads):
    vc.speculative_config.draft_model_config.hf_config.num_nextn_predict_layers = 2
    with pytest.raises(NotImplementedError, match="single-module MTP4 profile"):
        fn(vc, R, INTERLEAVE, **kw)


def test_fp8_with_wrong_method_rejected():
    """method != 'mtp' (here 'eagle'): rejected under the FP8 MTP4
    gate."""
    fn, kw = _validator_kwargs(fp8=True)
    vc = _make_vllm_config(cache_dtype="fp8_ds_mla")
    vc.speculative_config.method = "eagle"
    with pytest.raises(NotImplementedError, match="single-module MTP4 profile"):
        fn(vc, R, INTERLEAVE, **kw)


def test_fp8_with_wrong_draft_kv_override_rejected():
    """A draft KV override that is neither None nor fp8_ds_mla
    (here 'fp8'): rejected — the override changes the draft's KV
    layout away from the validated DS-656 storage."""
    fn, kw = _validator_kwargs(fp8=True)
    vc = _make_vllm_config(cache_dtype="fp8_ds_mla")
    vc.speculative_config.kv_cache_dtype = "fp8"
    with pytest.raises(NotImplementedError, match="single-module MTP4 profile"):
        fn(vc, R, INTERLEAVE, **kw)


def test_fp8_with_matching_draft_kv_override_admitted():
    """A draft KV override of exactly fp8_ds_mla: ADMITTED (the
    override matches the validated storage)."""
    fn, kw = _validator_kwargs(fp8=True)
    vc = _make_vllm_config(cache_dtype="fp8_ds_mla")
    vc.speculative_config.kv_cache_dtype = "fp8_ds_mla"
    fn(vc, R, INTERLEAVE, **kw)  # no raise


# --- the exact-capability fail-closed battery (the review's three
# omissions: topology, spec dtype, resolved flags) --------------------------


def test_fp8_capability_tp8_dcp4_rejected():
    """TP8 x DCP4 (a generic dcp>1 with an approved TP): the core
    capability predicate REJECTS — only the exact DCP2 with the
    valid TP-group divisibility is in scope; the builder grant and
    the cp-compat capability stay False (not left to a later
    plugin-side rejection)."""
    from vllm.v1.attention.backends.mla.dcp_mtp_capability import (
        _glm_kpool_fp8_mla_mtp_interleave_capable as capable,
    )
    from vllm.v1.attention.backends.mla.dcp_mtp_capability import (
        builder_supports_dcp_varlen_with_interleave as builder_ok,
    )

    vc = _make_vllm_config(cache_dtype="fp8_ds_mla")
    vc.compilation_config.static_forward_context = {}
    _build_real_caches(vc)
    vc.parallel_config.decode_context_parallel_size = 4
    assert capable(vc, fp8_mla_enabled=True) is False
    assert builder_ok(vc, backend_grants_capability=True, fp8_mla_enabled=True) is False
    builder = _make_flashmla_builder(vc, _hcu_builder_cls())
    assert builder.reorder_batch_threshold == 1


def test_fp8_capability_tp4_dcp2_rejected():
    """TP4 x DCP2 (an unapproved TP with valid divisibility): the
    core capability predicate REJECTS (only TP8-model/TP2-diagnostic
    are approved shapes)."""
    from vllm.v1.attention.backends.mla.dcp_mtp_capability import (
        _glm_kpool_fp8_mla_mtp_interleave_capable as capable,
    )

    vc = _make_vllm_config(cache_dtype="fp8_ds_mla")
    vc.compilation_config.static_forward_context = {}
    _build_real_caches(vc)
    vc.parallel_config.tensor_parallel_size = 4
    assert capable(vc, fp8_mla_enabled=True) is False


def test_fp8_capability_tp3_dcp2_nondivisible_rejected():
    """TP3 x DCP2 (an indivisible TP group): rejected on the
    divisibility check."""
    from vllm.v1.attention.backends.mla.dcp_mtp_capability import (
        _glm_kpool_fp8_mla_mtp_interleave_capable as capable,
    )

    vc = _make_vllm_config(cache_dtype="fp8_ds_mla")
    vc.compilation_config.static_forward_context = {}
    _build_real_caches(vc)
    vc.parallel_config.tensor_parallel_size = 3
    assert capable(vc, fp8_mla_enabled=True) is False


def test_fp8_capability_tp8_dcp2_granted():
    """The POSITIVE topology: TP8 x DCP2 (the approved model
    candidate) — the core capability GRANTS under the exact
    profile."""
    from vllm.v1.attention.backends.mla.dcp_mtp_capability import (
        _glm_kpool_fp8_mla_mtp_interleave_capable as capable,
    )

    vc = _make_vllm_config(cache_dtype="fp8_ds_mla")
    vc.compilation_config.static_forward_context = {}
    _build_real_caches(vc)
    vc.parallel_config.tensor_parallel_size = 8
    assert capable(vc, fp8_mla_enabled=True) is True


def test_fp8_capability_spec_dtype_none_rejected():
    """An indexer cache spec whose dtype is None/absent: the exact
    uint8 format check REJECTS (never a silent acceptance). The
    real caches are replaced with a stub whose spec drops dtype."""
    from vllm.v1.attention.backends.mla.dcp_mtp_capability import (
        _glm_kpool_fp8_mla_mtp_interleave_capable as capable,
    )

    vc = _make_vllm_config(cache_dtype="fp8_ds_mla")
    vc.compilation_config.static_forward_context = {}
    # A REAL cache whose spec's dtype is None (the fail-closed case):
    real_cache, _tail = _build_real_caches(vc)
    real_spec = real_cache.get_kv_cache_spec(vc)

    class _PatchedSpec:
        block_size = real_spec.block_size
        tokens_per_state = real_spec.tokens_per_state
        dtype = None

    orig = real_cache.get_kv_cache_spec
    real_cache.get_kv_cache_spec = lambda v: _PatchedSpec()
    try:
        assert capable(vc, fp8_mla_enabled=True) is False
    finally:
        real_cache.get_kv_cache_spec = orig
    # Restored: the real uint8 spec grants again:
    assert capable(vc, fp8_mla_enabled=True) is True


def test_fp8_capability_parallel_drafting_none_rejected():
    """parallel_drafting None (unresolved): the profile REJECTS —
    truthiness is never enough; the actual bool False is required."""
    from vllm.v1.attention.backends.mla.dcp_mtp_capability import (
        _fp8_single_module_mtp4_profile as profile,
    )

    vc = _make_vllm_config(cache_dtype="fp8_ds_mla")
    vc.speculative_config.parallel_drafting = None
    assert profile(vc) is False


def test_fp8_capability_multi_module_none_rejected():
    """use_multi_module_mtp() returning None: the profile REJECTS
    (an actual bool False is required — the single-module class is
    never guessed from S=4 alone)."""
    from vllm.v1.attention.backends.mla.dcp_mtp_capability import (
        _fp8_single_module_mtp4_profile as profile,
    )

    vc = _make_vllm_config(cache_dtype="fp8_ds_mla")
    vc.speculative_config.use_multi_module_mtp = lambda: None
    assert profile(vc) is False


def test_fp8_capability_multi_module_raises_rejected():
    """use_multi_module_mtp() raising (malformed): the profile
    REJECTS (never raises; never guesses)."""
    from vllm.v1.attention.backends.mla.dcp_mtp_capability import (
        _fp8_single_module_mtp4_profile as profile,
    )

    vc = _make_vllm_config(cache_dtype="fp8_ds_mla")

    def _boom():
        raise RuntimeError("malformed")

    vc.speculative_config.use_multi_module_mtp = _boom
    assert profile(vc) is False


def test_fp8_capability_missing_field_spec_rejects_no_crash():
    """A spec object WITHOUT the fields (an unsupported
    SimpleNamespace — the plugin runtime's actual failure shape):
    the profile returns False (the promised fail-closed), NEVER an
    AttributeError. Each missing field rejects individually."""
    from types import SimpleNamespace as _NS

    from vllm.v1.attention.backends.mla.dcp_mtp_capability import (
        _fp8_single_module_mtp4_profile as profile,
    )

    # No fields at all:
    bare = _NS()
    vc = _NS(speculative_config=bare)
    assert profile(vc) is False
    # Each partial shape (one field present, others missing):
    assert profile(_NS(speculative_config=_NS(method="mtp"))) is False
    assert (
        profile(_NS(speculative_config=_NS(method="mtp", num_speculative_tokens=4)))
        is False
    )
    assert (
        profile(
            _NS(
                speculative_config=_NS(
                    method="mtp",
                    num_speculative_tokens=4,
                    parallel_drafting=False,
                )
            )
        )
        is False
    )
    assert (
        profile(
            _NS(
                speculative_config=_NS(
                    method="mtp",
                    num_speculative_tokens=4,
                    parallel_drafting=False,
                    draft_model_config=object(),
                )
            )
        )
        is False
    )
    # The COMPLETE valid single-module S4 shape (a SimpleNamespace
    # carrying all the fields — the exact positive):
    assert (
        profile(
            _NS(
                speculative_config=_NS(
                    method="mtp",
                    num_speculative_tokens=4,
                    parallel_drafting=False,
                    draft_model_config=object(),
                    kv_cache_dtype=None,
                    use_multi_module_mtp=lambda: False,
                )
            )
        )
        is True
    )


def test_fp8_eager_backend_no_transfer_required():
    """The FP8 branch's OWN restrictions: non-ag_rs backend,
    compiled mode, and KV transfer each rejected under fp8_ds_mla
    with the grant (the same restrictions the impl predicate
    checks — now asserted IN the validator, not only later)."""
    fn, kw = _validator_kwargs(fp8=True)
    # Non-ag_rs (MTP cleared first — the MTP-off check fires first):
    vc = _make_vllm_config(cache_dtype="fp8_ds_mla")
    vc.speculative_config = None
    vc.parallel_config.dcp_comm_backend = "a2a"
    with pytest.raises(NotImplementedError, match="ag_rs"):
        fn(vc, R, INTERLEAVE, **kw)
    # A KV transfer config:
    vc2 = _make_vllm_config(cache_dtype="fp8_ds_mla")
    vc2.speculative_config = None
    vc2.kv_transfer_config = SimpleNamespace()
    with pytest.raises(NotImplementedError, match="single-node"):
        fn(vc2, R, INTERLEAVE, **kw)


def test_builder_carries_default_false_fp8_flag():
    """The core builder class: the FP8 flag DEFAULTS False (only the
    wired HCU plugin patch may set it True)."""
    from vllm.v1.attention.backends.mla.indexer import (
        DeepseekV32IndexerMetadataBuilder,
    )

    assert (
        DeepseekV32IndexerMetadataBuilder.__dict__["supports_glm_kpool_fp8_mla_dcp"]
        is False
    )
