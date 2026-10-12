# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the DCP compressed-indexer capability predicate.

The REAL DeepseekV32IndexerMetadataBuilder constructor is driven with
REAL VllmConfig objects (real ModelConfig/CacheConfig/ParallelConfig/
SchedulerConfig/CompilationConfig; real Glm5NextTextConfig text
configs) on CPU. The only CPU doubles are: (a) pure-Python package
stubs for imports the acceptance venv lacks (msgspec/cloudpickle/
uvloop/cpuinfo/prometheus_client — none execute on this path), (b)
``num_compute_units`` (a device query), and (c) the two-rank gloo
initialization for the real DCP group in the positive case. No
kernel, no GPU, no model weights.

The predicate replaces the two former blanket guards (interleave>1
and compress_ratio>1 under DCP). Positive: Glm5Next BF16/PCP1/I4/R4
with the capability flag granted constructs to a REAL metadata
build (C = R + S = 8). Negative matrix: each named failure reason.
DCP1 with any (I, R) and the generic R1/I1 DCP path stay unrejected.
"""

import json
import multiprocessing as mp
import os
import pathlib
import sys
import tempfile
from types import ModuleType

import pytest

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

_STUBS_INSTALLED = False


def _install_cpu_dep_stubs() -> None:
    """Import-time doubles for pip packages missing on this venv.

    These stub ONLY missing pure-Python serialization/metrics/loop
    libraries that the vllm config stack imports at module scope.
    None of the stubbed functions run on the builder path.
    """
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
    stub(
        "cloudpickle",
        dumps=lambda obj, **kw: b"",
        loads=lambda b, **kw: None,
    )
    stub(
        "uvloop",
        run=lambda coro, **kw: asyncio.run(coro),
        Install=lambda: None,
    )
    stub("cpuinfo", get_cpu_info=lambda: {"brand": "stub"})
    stub("llguidance", LLMatcher=object, LLTokenizer=object)

    class _XGModule(ModuleType):
        # A real __file__ (inspect.getmodule walks sys.modules and
        # calls getabsfile on each during torch's fake-op registration).
        __file__ = "<xgrammar stub>"

        def __getattr__(self, name):
            # Any symbol the structured-output stack imports resolves
            # to a placeholder; none execute on the builder path.
            return object

    if "xgrammar" not in sys.modules:
        sys.modules["xgrammar"] = _XGModule("xgrammar")

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
    _STUBS_INSTALLED = True


_install_cpu_dep_stubs()

import torch  # noqa: E402

import vllm.models.glm5next  # noqa: F401,E402  (registry before config)
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
from vllm.transformers_utils.configs.glm5_next import (  # noqa: E402
    Glm5NextConfig,
    Glm5NextTextConfig,
)
from vllm.v1.kv_cache_interface import MLAAttentionSpec  # noqa: E402

R = 4
INTERLEAVE = 4
BLOCK = 128
HEAD_DIM = 576


def _write_config_dir() -> pathlib.Path:
    """A REAL on-disk Glm5NextTextConfig (offline AutoConfig load)."""
    cfg = Glm5NextTextConfig(
        index_topk=2048,
        index_kpool=R,
        index_head_dim=128,
        index_n_heads=64,
        swiglu_limit=1.0,
    )
    d = pathlib.Path(tempfile.mkdtemp())
    with open(d / "config.json", "w") as f:
        json.dump(cfg.to_dict(), f)
    return d


def _make_vllm_config(
    *,
    dcp=2,
    interleave=INTERLEAVE,
    cache_dtype="bfloat16",
    pcp=1,
    spec_tokens=4,
    kv_transfer=None,
    kpool=R,
    model_type="glm5_next_text",
    block_size=BLOCK,
):
    d = _write_config_dir()
    if kpool != R:
        # Rewrite with a different kpool for the mismatch case.
        cfg = Glm5NextTextConfig(
            index_topk=2048,
            index_kpool=kpool,
            index_head_dim=128,
            index_n_heads=64,
            swiglu_limit=1.0,
        )
        with open(d / "config.json", "w") as f:
            json.dump(cfg.to_dict(), f)
    mc = ModelConfig(
        model=str(d),
        tokenizer=str(d),
        trust_remote_code=False,
        dtype="bfloat16",
        seed=0,
        max_model_len=1024,
        hf_overrides={"architectures": ["Glm5NextForCausalLM"]},
    )
    if model_type != "glm5_next_text":
        # A non-GLM text config: patch the loaded instance's identity.
        mc.hf_text_config.model_type = model_type
    cc = CacheConfig(block_size=block_size, cache_dtype=cache_dtype)
    pc = ParallelConfig(
        tensor_parallel_size=2,
        decode_context_parallel_size=dcp,
        cp_kv_cache_interleave_size=interleave,
        prefill_context_parallel_size=pcp,
        distributed_executor_backend="uni",
    )
    sc = SchedulerConfig(
        max_num_seqs=4,
        max_num_batched_tokens=64,
        enable_chunked_prefill=True,
        max_model_len=1024,
        is_encoder_decoder=False,
    )
    return VllmConfig(
        model_config=mc,
        cache_config=cc,
        parallel_config=pc,
        scheduler_config=sc,
        device_config=DeviceConfig(device="cpu"),
        load_config=LoadConfig(),
        compilation_config=CompilationConfig(),
        kv_transfer_config=kv_transfer,
    )


def _make_spec(tokens_per_state=R):
    return MLAAttentionSpec(
        block_size=BLOCK,
        num_kv_heads=1,
        head_size=HEAD_DIM,
        dtype=torch.bfloat16,
        tokens_per_state=tokens_per_state,
    )


def _construct(vc, *, flag=False, spec=None):
    """Drive the REAL builder constructor with the CPU device doubles:
    num_compute_units (a device query) and the DCP rank query (the
    single-rank process has no initialized DCP group; the two-rank
    positive test initializes a REAL gloo group instead)."""
    import vllm.v1.attention.backends.mla.indexer as ix

    saved_ncu = ix.num_compute_units
    saved_flag = ix.DeepseekV32IndexerMetadataBuilder.supports_glm_kpool_bf16_dcp
    saved_group = ix.get_dcp_group
    ix.num_compute_units = lambda idx: 8
    ix.DeepseekV32IndexerMetadataBuilder.supports_glm_kpool_bf16_dcp = flag
    ix.get_dcp_group = lambda: type("G", (), {"rank_in_group": 0})()
    try:
        return ix.DeepseekV32IndexerMetadataBuilder(
            kv_cache_spec=spec if spec is not None else _make_spec(),
            layer_names=["l0"],
            vllm_config=vc,
            device=torch.device("cpu"),
            block_table_width=4,
        )
    finally:
        ix.num_compute_units = saved_ncu
        ix.DeepseekV32IndexerMetadataBuilder.supports_glm_kpool_bf16_dcp = saved_flag
        ix.get_dcp_group = saved_group


# --- the pure predicate (unit level) --------------------------------------


def test_predicate_dcp1_keeps_legacy_for_any_i_r():
    """DCP1: the predicate returns immediately for ANY (I, R)."""
    from vllm.v1.attention.backends.mla.indexer import (
        validate_dcp_compressed_indexer_support,
    )

    vc = _make_vllm_config(dcp=1, interleave=7)
    # Even nonsense (I, R) passes on DCP1 — the legacy contract.
    validate_dcp_compressed_indexer_support(vc, 3, 7, capability_enabled=False)


def test_predicate_r1_i1_dcp_passes_without_capability():
    """DCP>1 with R==1 and I==1: the long-supported path, no flag."""
    from vllm.v1.attention.backends.mla.indexer import (
        validate_dcp_compressed_indexer_support,
    )

    vc = _make_vllm_config(interleave=1)
    validate_dcp_compressed_indexer_support(vc, 1, 1, capability_enabled=False)


def test_predicate_flag_false_rejects_with_named_reason():
    from vllm.v1.attention.backends.mla.indexer import (
        validate_dcp_compressed_indexer_support,
    )

    vc = _make_vllm_config()
    with pytest.raises(NotImplementedError, match="capability flag"):
        validate_dcp_compressed_indexer_support(
            vc, R, INTERLEAVE, capability_enabled=False
        )


def test_predicate_positive_all_conditions():
    from vllm.v1.attention.backends.mla.indexer import (
        validate_dcp_compressed_indexer_support,
    )

    vc = _make_vllm_config()
    validate_dcp_compressed_indexer_support(vc, R, INTERLEAVE, capability_enabled=True)


def test_predicate_rejection_matrix_named_reasons():
    from vllm.v1.attention.backends.mla.indexer import (
        validate_dcp_compressed_indexer_support,
    )

    cases: list[tuple[dict, str]] = [
        # (kwargs, reason substring)
        (dict(cache_dtype="auto"), "bfloat16"),
        (dict(cache_dtype="fp8"), "bfloat16"),
        (dict(kpool=8), "index_kpool"),
        (dict(model_type="deepseek_v3"), "Glm5NextTextConfig"),
        (dict(interleave=5), "divisible"),  # I=5 not % R=4 -> derive raises
        (dict(interleave=2), "cp_kv_cache_interleave_size"),
        (dict(block_size=130), "block_size"),  # 130 % 4 != 0
    ]
    for kwargs, reason in cases:
        vc = _make_vllm_config(**kwargs)
        itl = int(kwargs.get("interleave", INTERLEAVE))
        with pytest.raises((NotImplementedError, ValueError)) as ei:
            validate_dcp_compressed_indexer_support(vc, R, itl, capability_enabled=True)
        assert reason in str(ei.value), (kwargs, str(ei.value))


def test_predicate_pcp_rejected():
    """PCP>1: the predicate rejects with the named reason. (The full
    VllmConfig stack rejects PCP earlier for V1; the predicate-level
    check uses the same real config with the field patched.)"""
    from vllm.v1.attention.backends.mla.indexer import (
        validate_dcp_compressed_indexer_support,
    )

    vc = _make_vllm_config()
    vc.parallel_config.prefill_context_parallel_size = 2
    with pytest.raises(NotImplementedError, match="prefill_context_parallel_size"):
        validate_dcp_compressed_indexer_support(
            vc, R, INTERLEAVE, capability_enabled=True
        )


def test_predicate_fake_same_named_class_rejected():
    """A fabricated type named 'Glm5NextTextConfig' with all the right
    attributes is NOT the real config: the isinstance check (against
    the imported real class) rejects it — the class-NAME string is
    not an identity proof (Codex's reproduced counterexample)."""
    from vllm.v1.attention.backends.mla.indexer import (
        validate_dcp_compressed_indexer_support,
    )

    vc = _make_vllm_config()
    fake = type("Glm5NextTextConfig", (), {})()
    fake.model_type = "glm5_next_text"
    fake.index_kpool_compress = True
    fake.index_kpool = R
    vc.model_config.hf_text_config = fake
    with pytest.raises(NotImplementedError, match="real Glm5NextTextConfig"):
        validate_dcp_compressed_indexer_support(
            vc, R, INTERLEAVE, capability_enabled=True
        )


def test_predicate_real_subclass_accepted():
    """A REAL subclass of Glm5NextTextConfig (HF semantics allow
    subclasses) passes the identity check."""
    from vllm.transformers_utils.configs.glm5_next import Glm5NextTextConfig
    from vllm.v1.attention.backends.mla.indexer import (
        validate_dcp_compressed_indexer_support,
    )

    class Sub(Glm5NextTextConfig):
        pass

    vc = _make_vllm_config()
    vc.model_config.hf_text_config = Sub(
        index_topk=2048,
        index_kpool=R,
        index_kpool_compress=True,
        index_head_dim=128,
        index_n_heads=64,
        swiglu_limit=1.0,
    )
    validate_dcp_compressed_indexer_support(vc, R, INTERLEAVE, capability_enabled=True)


def test_predicate_actual_spec_block_mismatch_rejected():
    """The ACTUAL spec layout proof: a spec whose block_size does not
    tile the interleave is rejected even when the CONFIG block_size
    would pass — the config value is not a layout proof."""
    from vllm.v1.attention.backends.mla.indexer import (
        validate_dcp_compressed_indexer_support,
    )

    vc = _make_vllm_config()  # config block_size 128, I=4: config OK
    bad_spec = MLAAttentionSpec(
        block_size=130,  # 130 % 4 != 0
        num_kv_heads=1,
        head_size=HEAD_DIM,
        dtype=torch.bfloat16,
        tokens_per_state=R,
    )
    with pytest.raises(NotImplementedError, match="actual kv_cache_spec"):
        validate_dcp_compressed_indexer_support(
            vc, R, INTERLEAVE, capability_enabled=True, actual_spec=bad_spec
        )


def test_predicate_actual_spec_tokens_per_state_mismatch_rejected():
    """The spec's tokens_per_state must agree with the compress_ratio
    the caller derived — a divergent spec is rejected."""
    from vllm.v1.attention.backends.mla.indexer import (
        validate_dcp_compressed_indexer_support,
    )

    vc = _make_vllm_config()
    bad_spec = MLAAttentionSpec(
        block_size=BLOCK,
        num_kv_heads=1,
        head_size=HEAD_DIM,
        dtype=torch.bfloat16,
        tokens_per_state=8,  # != R
    )
    with pytest.raises(NotImplementedError, match="tokens_per_state"):
        validate_dcp_compressed_indexer_support(
            vc, R, INTERLEAVE, capability_enabled=True, actual_spec=bad_spec
        )


def test_predicate_actual_spec_positive_passes():
    """The matching actual spec passes (with the same config)."""
    from vllm.v1.attention.backends.mla.indexer import (
        validate_dcp_compressed_indexer_support,
    )

    vc = _make_vllm_config()
    good_spec = MLAAttentionSpec(
        block_size=BLOCK,
        num_kv_heads=1,
        head_size=HEAD_DIM,
        dtype=torch.bfloat16,
        tokens_per_state=R,
    )
    validate_dcp_compressed_indexer_support(
        vc, R, INTERLEAVE, capability_enabled=True, actual_spec=good_spec
    )


def test_predicate_spec_tokens_mismatch_rejected_at_constructor():
    """The REAL constructor rejects a spec whose tokens_per_state
    diverges from the model config's index_kpool (the actual-spec
    consistency check bites end-to-end)."""
    vc = _make_vllm_config()
    with pytest.raises(NotImplementedError, match="index_kpool|tokens_per_state"):
        _construct(
            vc,
            flag=True,
            spec=MLAAttentionSpec(
                block_size=BLOCK,
                num_kv_heads=1,
                head_size=HEAD_DIM,
                dtype=torch.bfloat16,
                tokens_per_state=8,  # model says 4
            ),
        )


def test_predicate_unresolved_speculative_config_rejected():
    """A PRESENT speculative_config with num_speculative_tokens=None
    is rejected with the named reason (it would fail in the tail
    helper after buffers); NO speculative_config stays legal (S=0)."""
    from types import SimpleNamespace

    from vllm.v1.attention.backends.mla.indexer import (
        validate_dcp_compressed_indexer_support,
    )

    vc = _make_vllm_config()
    vc.speculative_config = SimpleNamespace(num_speculative_tokens=None)
    with pytest.raises(NotImplementedError, match="not resolved"):
        validate_dcp_compressed_indexer_support(
            vc, R, INTERLEAVE, capability_enabled=True
        )
    # No spec config at all: legal.
    vc2 = _make_vllm_config()
    vc2.speculative_config = None
    validate_dcp_compressed_indexer_support(vc2, R, INTERLEAVE, capability_enabled=True)


def test_predicate_kv_transfer_rejected():
    from vllm.config import KVTransferConfig
    from vllm.v1.attention.backends.mla.indexer import (
        validate_dcp_compressed_indexer_support,
    )

    ktc = KVTransferConfig(
        engine_id="test", kv_role="kv_producer", kv_connector="LMCacheConnectorV1"
    )
    vc = _make_vllm_config(kv_transfer=ktc)
    with pytest.raises(NotImplementedError, match="kv_transfer_config"):
        validate_dcp_compressed_indexer_support(
            vc, R, INTERLEAVE, capability_enabled=True
        )


# --- the REAL constructor (integration level) ------------------------------


def test_real_constructor_flag_false_rejects_i4():
    """Flag False (core default): I=4/R=4 DCP2 construction fails closed."""
    vc = _make_vllm_config()
    with pytest.raises(NotImplementedError, match="capability flag"):
        _construct(vc, flag=False)


def test_real_constructor_positive_glm_bf16_pcp1_i4_r4():
    """Flag True: the full positive candidate constructs on the REAL
    VllmConfig (single-rank DCP group simulation — the group is only
    queried for rank when world>1; a rank-0 default suffices for the
    constructor, whose DCP path reads parallel config fields)."""
    vc = _make_vllm_config()
    b = _construct(vc, flag=True)
    assert b.dcp_world_size == 2
    assert b.cp_kv_cache_interleave_size == INTERLEAVE
    assert b.compress_ratio == R
    # The compressed buffers ARE allocated (the compression path is on).
    assert b.compressed_slot_mapping_buffer is not None


def test_real_constructor_generic_r1_dcp_unchanged():
    """DCP2 with R=1/I=1 (the generic path): constructs WITHOUT the
    flag — legacy behavior preserved."""
    vc = _make_vllm_config(interleave=1)
    b = _construct(vc, flag=False, spec=_make_spec(tokens_per_state=1))
    assert b.compress_ratio == 1


def test_real_constructor_dcp1_r4_i4_no_rejection():
    """DCP1 with compression and interleave: never rejected (legacy)."""
    vc = _make_vllm_config(dcp=1)
    b = _construct(vc, flag=False)
    assert b.compress_ratio == R


def _spawn_bootstrap(rank: int, module_file: str) -> None:
    """The spawn child entry: imports the test module BY PATH (no
    dotted-name unpickle — immune to a sibling suite's plugin
    sys.path entries shadowing this repo's packages) and runs the
    two-rank body."""
    import importlib.util
    import sys

    repo = str(pathlib.Path(module_file).parents[3])
    sys.path[:] = [p for p in sys.path if "dcp2-plugin" not in p]
    if repo not in sys.path:
        sys.path.insert(0, repo)
    spec = importlib.util.spec_from_file_location(
        "test_dcp_capability_predicate_spawn", module_file
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import traceback as _tb

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29713")
    try:
        mod._two_rank_positive_body(rank)
    except BaseException:
        err = pathlib.Path(tempfile.gettempdir()) / f"dcp_two_rank_err_{rank}.txt"
        err.write_text(_tb.format_exc())
        raise


def _two_rank_positive_run(rank):
    """The spawned per-rank body of the two-rank positive test: a REAL
    gloo DCP group of 2, the REAL builder, and the REAL tail spec."""
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29713")
    # The spawned child needs the repo on sys.path (spawn re-imports
    # this module by file path under pytest's importlib mode). Any
    # PLUGIN worktree paths inherited from a sibling suite's run are
    # removed first: this spawn test exercises the CORE registry
    # alone, and a plugin-first path changes vllm's module identity
    # resolution (the observed Glm5NextTextModel failure).
    repo = str(_HERE.parents[2])
    sys.path[:] = [p for p in sys.path if "dcp2-plugin" not in p]
    if repo not in sys.path:
        sys.path.insert(0, repo)
    # spawn pickles the child fn by module reference; ensure THIS
    # module resolves after the path scrub (its parent packages are
    # already imported in the parent, and re-importing under the
    # scrubbed path keeps a single identity).

    global __package__
    _install_cpu_dep_stubs()
    import traceback as _tb

    try:
        _two_rank_positive_body(rank)
    except BaseException:
        # The child's traceback goes to a per-rank file the parent
        # reads on failure (spawn stderr is not reliably forwarded).
        err = pathlib.Path(tempfile.gettempdir()) / f"dcp_two_rank_err_{rank}.txt"
        err.write_text(_tb.format_exc())
        raise


def _two_rank_positive_body(rank):
    import torch
    import torch.distributed as dist

    dist.init_process_group("gloo", rank=rank, world_size=2)
    from vllm.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    init_distributed_environment(
        world_size=2,
        rank=rank,
        local_rank=rank,
        distributed_init_method="env://",
        backend="gloo",
    )
    # The NESTED outer Glm5NextConfig (the real checkpoint shape):
    # MTP resolution reads the OUTER model_type ("glm5_next" ->
    # "glm5_next_mtp" -> Glm5NextMTPModel).
    inner = Glm5NextTextConfig(
        index_topk=2048,
        index_kpool=R,
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
    pc = ParallelConfig(
        tensor_parallel_size=2,
        decode_context_parallel_size=2,
        cp_kv_cache_interleave_size=INTERLEAVE,
        distributed_executor_backend="uni",
    )
    mc = ModelConfig(
        model=str(d),
        tokenizer=str(d),
        trust_remote_code=False,
        dtype="bfloat16",
        seed=0,
        max_model_len=1024,
    )
    vc = VllmConfig(
        model_config=mc,
        cache_config=CacheConfig(block_size=BLOCK, cache_dtype="bfloat16"),
        parallel_config=pc,
        scheduler_config=SchedulerConfig(
            max_num_seqs=4,
            max_num_batched_tokens=64,
            enable_chunked_prefill=True,
            max_model_len=1024,
            is_encoder_decoder=False,
        ),
        device_config=DeviceConfig(device="cpu"),
        load_config=LoadConfig(),
        compilation_config=CompilationConfig(),
        speculative_config=SpeculativeConfig(
            method="mtp",
            num_speculative_tokens=4,
            target_model_config=mc,
            target_parallel_config=pc,
        ),
    )
    from vllm.config.vllm import set_current_vllm_config

    with set_current_vllm_config(vc):
        initialize_model_parallel(
            tensor_model_parallel_size=2,
            decode_context_model_parallel_size=2,
        )
    import vllm.v1.attention.backends.mla.indexer as ix

    ix.num_compute_units = lambda idx: 8
    ix.DeepseekV32IndexerMetadataBuilder.supports_glm_kpool_bf16_dcp = True
    b = ix.DeepseekV32IndexerMetadataBuilder(
        kv_cache_spec=_make_spec(),
        layer_names=["l0"],
        vllm_config=vc,
        device=torch.device("cpu"),
        block_table_width=4,
    )
    assert b.dcp_world_size == 2
    assert b.dcp_rank == rank
    assert b.compress_ratio == R
    # The REAL tail capacity from Glm5NextTailCache's contract:
    # S=4 -> C = R + S = 8.
    from vllm.models.glm5next.nvidia.attention import Glm5NextTailCache

    with set_current_vllm_config(vc):
        tc = Glm5NextTailCache(
            head_dim=128,
            dtype=torch.bfloat16,
            prefix="t.tail",
            cache_config=vc.cache_config,
            index_kpool=R,
            vllm_config=vc,
        )
    tail_spec = tc.get_kv_cache_spec(vc)
    assert tail_spec.block_size == R + 4, tail_spec.block_size
    print(f"rank{rank}: OK C={tail_spec.block_size}", flush=True)
    dist.destroy_process_group()


def test_real_constructor_positive_builds_real_metadata_c8():
    """The positive case drives a REAL two-rank gloo DCP group: the
    builder constructs on both ranks (J = I // R = 1; C = R + S = 8
    via the REAL tail spec)."""
    ctx = mp.get_context("spawn")
    module_file = str(pathlib.Path(__file__).resolve())
    # A sibling suite may have PREPENDED a plugin worktree to
    # sys.path; the spawn children unpickle this module by its
    # dotted name BEFORE any child code runs, and the plugin's own
    # tests package would shadow this repo's. Scrub the plugin
    # entries for the DURATION of the spawn only (restored right
    # after): the in-process imports of any sibling suite are
    # already resolved and unaffected.
    saved_path = list(sys.path)
    sys.path[:] = [p for p in sys.path if "dcp2-plugin" not in p]
    try:
        ps = [
            ctx.Process(target=_spawn_bootstrap, args=(r, module_file)) for r in (0, 1)
        ]
        for p_ in ps:
            p_.start()
    finally:
        sys.path[:] = saved_path
    for p_ in ps:
        p_.join(180)
    codes = [p_.exitcode for p_ in ps]
    if codes != [0, 0]:
        details = []
        for r in (0, 1):
            err = pathlib.Path(tempfile.gettempdir()) / f"dcp_two_rank_err_{r}.txt"
            if err.exists():
                details.append(f"--- rank {r} ---\n{err.read_text()[-2000:]}")
        raise AssertionError(f"spawned ranks failed: {codes}\n" + "\n".join(details))


def test_real_constructor_rejects_auto_cache_dtype():
    """'auto' cache_dtype: rejected with the BF16 reason (fail-closed,
    no silent resolution)."""
    vc = _make_vllm_config(cache_dtype="auto")
    with pytest.raises(NotImplementedError, match="bfloat16"):
        _construct(vc, flag=True)
