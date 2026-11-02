# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.core.kv_cache_utils import _build_attention_kv_cache_tensors
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MambaSpec,
    MLAAttentionSpec,
    PCPCacheSpec,
)
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.pcp.allocation import make_pcp_physical_config
from vllm.v1.worker.utils import allocate_kv_cache


@pytest.mark.parametrize("size", [1, 2, 8])
def test_physical_repacking_preserves_group_aliases_and_kda_capacity(size):
    layout = KVCacheLayout.BLHNC
    config = SimpleNamespace(
        cache_config=SimpleNamespace(
            get_resolved_kv_cache_layout=lambda: layout,
        )
    )
    mla = MLAAttentionSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=16,
        dtype=torch.bfloat16,
    )
    index = MLAAttentionSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=132,
        dtype=torch.uint8,
        tokens_per_state=4,
    )
    groups = [
        KVCacheGroupSpec(["mla.0", "mla.1"], mla),
        KVCacheGroupSpec(["index.0"], index),
    ]
    tensors, _ = _build_attention_kv_cache_tensors(
        config,
        groups,
        0,
        physical_num_blocks=17,
    )
    kda = MambaSpec(
        block_size=64,
        shapes=((8, 3), (1, 4, 4)),
        dtypes=(torch.float16, torch.float32),
    )
    kda_tensor = KVCacheTensor(
        size=kda.page_size_bytes * 9,
        layers=["kda.0"],
        layer_stride=kda.page_size_bytes * 9,
        block_stride=kda.page_size_bytes,
        musa_mamba_pool=True,
    )
    logical = KVCacheConfig(
        num_blocks=17,
        kv_cache_tensors=tensors + [kda_tensor],
        kv_cache_groups=groups + [KVCacheGroupSpec(["kda.0"], kda)],
        musa_mamba_num_blocks=9,
    )
    physical = make_pcp_physical_config(config, logical, size)
    assert logical.num_blocks == 17
    assert physical.num_blocks == (17 + size - 1) // size
    assert physical.musa_mamba_num_blocks == 9
    assert physical.kv_cache_tensors[-1] == kda_tensor
    original_bytes = tensors[0].size
    assert physical.kv_cache_tensors[0].size == (
        original_bytes // 17 * physical.num_blocks
    )
    views = allocate_kv_cache(physical, torch.device("cpu"), layout)
    assert views["mla.0"].shape[0] == physical.num_blocks
    assert views["index.0"].shape[0] == physical.num_blocks
    assert views["kda.0"].shape[0] == 9
    # Different attention groups own disjoint logical IDs, sharing one pool.
    assert views["mla.0"].untyped_storage().data_ptr() == (
        views["index.0"].untyped_storage().data_ptr()
    )
    assert views["kda.0"].untyped_storage().data_ptr() != (
        views["mla.0"].untyped_storage().data_ptr()
    )
    physical.kv_cache_tensors[0].layers.append("changed")
    assert "changed" not in logical.kv_cache_tensors[0].layers


def test_binding_supports_sparse_kernel_pages_between_packed_layers(monkeypatch):
    import vllm.distributed
    from vllm.v1.pcp.allocation import bind_pcp_pages

    group = SimpleNamespace(world_size=8, rank_in_group=3)
    monkeypatch.setattr(vllm.distributed, "get_pcp_group", lambda: group)
    spec = MLAAttentionSpec(
        block_size=12,
        num_kv_heads=1,
        head_size=4,
        dtype=torch.uint8,
    )
    raw = torch.zeros(3 * 128, dtype=torch.uint8)
    # The native reader exposes 3 valid kernel pages every 8 address slots.
    native = raw.as_strided((19, 4, 4), (16, 4, 1))
    placement = KVCacheTensor(
        size=raw.numel(),
        layers=["index"],
        layer_stride=48,
        block_stride=128,
    )
    groups = [KVCacheGroupSpec(["index"], spec)]
    physical = KVCacheConfig(3, [placement], groups)
    logical = KVCacheConfig(17, [placement], groups)
    layer = SimpleNamespace(kv_cache=native)
    bind_pcp_pages(None, logical, physical, {"index": layer}, {"index": native})
    assert layer.pcp_pages.factor == 8
    assert layer.pcp_pages.kernel_pages_per_block == 3
    owner, offset = layer.pcp_pages.locations(torch.tensor([24, 25, 26, 88]))
    assert owner.tolist() == [3, 3, 3, 3]
    assert offset.tolist() == [0, 1, 2, 8]


def test_cow_preserves_cache_group_identity_when_block_ids_overlap():
    from vllm.v1.core.kv_cache_manager import KVCacheManager

    source_a, target_a = SimpleNamespace(block_id=1), SimpleNamespace(block_id=2)
    source_b, target_b = SimpleNamespace(block_id=1), SimpleNamespace(block_id=2)
    manager = object.__new__(KVCacheManager)
    manager.coordinator = SimpleNamespace(
        single_type_managers=[
            SimpleNamespace(take_pending_cow_copies=lambda: [(source_a, target_a)]),
            SimpleNamespace(take_pending_cow_copies=lambda: [(source_b, target_b)]),
        ]
    )
    copies, retained = manager.take_kv_cache_block_copies()
    assert [(copy.src_block_id, copy.dst_block_id) for copy in copies] == [
        (1, 2),
        (1, 2),
    ]
    assert manager.take_kv_cache_copy_group_ids() == [0, 1]
    assert manager.take_kv_cache_copy_group_ids() == []
    assert retained == [source_a, target_a, source_b, target_b]


def test_kda_copy_cannot_be_routed_to_the_attention_pool():
    from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
    from vllm.v1.pcp.allocation import apply_pcp_block_copies

    mla = MLAAttentionSpec(block_size=4, num_kv_heads=1, head_size=4, dtype=torch.uint8)
    kda = MambaSpec(block_size=4, shapes=((4,),), dtypes=(torch.float32,))
    config = KVCacheConfig(
        16,
        [],
        [KVCacheGroupSpec(["mla"], mla), KVCacheGroupSpec(["kda"], kda)],
        musa_mamba_num_blocks=4,
    )
    state = torch.zeros(4, 1, 1, 16, dtype=torch.uint8)
    state[1] = 7
    routed = []
    apply_pcp_block_copies(
        config,
        {"kda": SimpleNamespace(kv_cache=state)},
        SimpleNamespace(copy_blocks=lambda pairs: routed.extend(pairs)),
        [KVCacheBlockCopy(2, 3), KVCacheBlockCopy(1, 2)],
        [0, 1],
    )
    assert routed == [KVCacheBlockCopy(2, 3)]
    assert bool((state[2] == 7).all())
    assert bool((state[3] == 0).all())


def test_scheduler_copy_group_identity_survives_worker_broadcast():
    import pickle

    from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
    from vllm.v1.core.sched.output import SchedulerOutput

    message = SchedulerOutput.make_empty()
    message.kv_cache_block_copies = [KVCacheBlockCopy(1, 2), KVCacheBlockCopy(1, 2)]
    message.kv_cache_block_copy_group_ids = [0, 2]
    encoded = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
    restored = pickle.loads(encoded)
    assert restored.kv_cache_block_copies == message.kv_cache_block_copies
    assert restored.kv_cache_block_copy_group_ids == [0, 2]


def test_sharded_glm_requires_the_matching_token_layout_and_execution_mode():
    from copy import deepcopy

    from vllm.config import CUDAGraphMode
    from vllm.models.glm5next.common.attention import glm_pcp_cache_spec

    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type="glm5_next_text")
        ),
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=8,
            tensor_parallel_size=1,
            decode_context_parallel_size=1,
            kda_layer_tp=8,
            data_parallel_size=1,
        ),
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE),
        speculative_config=None,
        attention_config=SimpleNamespace(hisparse_config=None),
        kv_transfer_config=None,
    )
    assert glm_pcp_cache_spec(config, 4)
    invalid = deepcopy(config)
    invalid.parallel_config.kda_layer_tp = 1
    with pytest.raises(ValueError, match="KDA TP"):
        glm_pcp_cache_spec(invalid, 4)
    invalid = deepcopy(config)
    invalid.compilation_config.cudagraph_mode = CUDAGraphMode.FULL
    with pytest.raises(ValueError, match="eager"):
        glm_pcp_cache_spec(invalid, 4)
    invalid = deepcopy(config)
    invalid.speculative_config = object()
    with pytest.raises(ValueError, match="speculative"):
        glm_pcp_cache_spec(invalid, 4)
    # The intended decode DP8/KDA TP8 topology has PCP=1 and is unaffected.
    config.parallel_config.prefill_context_parallel_size = 1
    config.parallel_config.data_parallel_size = 8
    assert not glm_pcp_cache_spec(config, 4)


def test_workspace_admission_counts_aliases_once_and_bounds_cow():
    from vllm.v1.pcp.allocation import plan_pcp_page_workspace

    spec = MLAAttentionSpec(
        block_size=8,
        num_kv_heads=1,
        head_size=16,
        dtype=torch.uint8,
    )
    groups = [KVCacheGroupSpec([f"mla.{i}" for i in range(16)], spec)]
    block_bytes = spec.page_size_bytes * 16
    logical = KVCacheConfig(
        128,
        [
            KVCacheTensor(
                size=128 * block_bytes,
                layers=groups[0].layer_names,
                layer_stride=spec.page_size_bytes,
                block_stride=block_bytes,
            )
        ],
        groups,
    )
    physical = KVCacheConfig(
        16,
        [
            KVCacheTensor(
                size=16 * block_bytes,
                layers=groups[0].layer_names,
                layer_stride=spec.page_size_bytes,
                block_stride=block_bytes,
            )
        ],
        groups,
    )
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=4, max_num_batched_tokens=16)
    )
    budget = plan_pcp_page_workspace(config, logical, physical, 8)
    assert budget.max_copy_pages == 4
    assert budget.attention_bytes == 16 * block_bytes
    assert budget.history_bytes == (9 * 16 + 128) * spec.page_size_bytes
    assert budget.copy_bytes == (9 * 4 + 4) * block_bytes
    assert budget.total_bytes <= logical.kv_cache_tensors[0].size

    # More possible CoW sources exceed this budget for collective staging.
    config.scheduler_config.max_num_seqs = 64
    with pytest.raises(ValueError, match="profiled cache budget"):
        plan_pcp_page_workspace(config, logical, physical, 8)

    # Scale the pool to represent the native multi-layer pages. IPC needs one
    # CoW snapshot, not an all-gather allocation multiplied by the rank count.
    logical.kv_cache_tensors[0].size *= 64
    physical.kv_cache_tensors[0].size *= 64
    block_bytes *= 64
    with pytest.raises(ValueError, match="profiled cache budget"):
        plan_pcp_page_workspace(config, logical, physical, 8)
    native = plan_pcp_page_workspace(config, logical, physical, 8, use_ipc=True)
    assert native.max_copy_pages == 64
    assert native.copy_bytes == 64 * block_bytes
    assert native.history_bytes == 2 * 128 * spec.page_size_bytes
    assert native.total_bytes <= logical.kv_cache_tensors[0].size
    config.scheduler_config.max_num_seqs = 128
    with pytest.raises(ValueError, match="profiled cache budget"):
        plan_pcp_page_workspace(config, logical, physical, 8, use_ipc=True)


def test_pcp_ownership_is_declared_without_layer_name_conventions():
    from vllm.v1.pcp.allocation import resolve_pcp_layout

    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=8, decode_context_parallel_size=1
        )
    )
    spec = MLAAttentionSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=16,
        dtype=torch.uint8,
        pcp=PCPCacheSpec(4, "paired"),
    )
    assert (
        resolve_pcp_layout(
            config, [KVCacheGroupSpec(["arbitrary.a", "b"], spec)]
        ).pool_tokens
        == 4
    )
    with pytest.raises(ValueError, match="share one logical cache group"):
        resolve_pcp_layout(
            config,
            [
                KVCacheGroupSpec(["arbitrary.a"], spec),
                KVCacheGroupSpec(["b"], spec),
            ],
        )


@pytest.fixture
def physical_planner_case(monkeypatch):
    from vllm.config import CUDAGraphMode
    from vllm.v1.core import kv_cache_utils

    monkeypatch.setattr(
        kv_cache_utils, "musa_mamba_separate_pool_enabled", lambda: True
    )
    config = SimpleNamespace(
        cache_config=SimpleNamespace(
            get_resolved_kv_cache_layout=lambda: KVCacheLayout.BLHNC,
            num_gpu_blocks_override=None,
            prefix_cache_retention_interval=None,
            mamba_cache_mode="align",
        ),
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=8,
            tensor_parallel_size=1,
            decode_context_parallel_size=1,
            kda_layer_tp=8,
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=4, max_num_batched_tokens=256),
        model_config=SimpleNamespace(
            max_model_len=8192,
            hf_text_config=SimpleNamespace(model_type="unrelated_model"),
        ),
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE),
        attention_config=SimpleNamespace(hisparse_config=None),
        speculative_config=None,
        kv_transfer_config=None,
    )
    mla = MLAAttentionSpec(
        pcp=PCPCacheSpec(4),
        block_size=64,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
    )
    index = MLAAttentionSpec(
        pcp=PCPCacheSpec(4),
        block_size=64,
        num_kv_heads=1,
        head_size=132,
        dtype=torch.uint8,
        tokens_per_state=4,
    )
    kda = MambaSpec(
        block_size=64,
        shapes=((8, 3), (1, 4, 4)),
        dtypes=(torch.float16, torch.float32),
    )
    groups = [
        KVCacheGroupSpec([f"mla.{i}" for i in range(24)], mla),
        KVCacheGroupSpec([f"index.{i}" for i in range(24)], index),
        KVCacheGroupSpec(["kda.0", "kda.1"], kda),
    ]
    return config, groups


@pytest.mark.parametrize("use_ipc", [False, True])
def test_physical_planner_maximizes_measured_budget(physical_planner_case, use_ipc):
    from vllm.v1.core import kv_cache_utils as utils
    from vllm.v1.core.kv_cache_utils import (
        _pcp_capacity_candidate,
        pcp_min_memory_bytes,
        plan_pcp_cache_config,
    )
    from vllm.v1.pcp.allocation import (
        pcp_required_blocks,
        plan_pcp_page_workspace,
    )

    config, groups = physical_planner_case
    minimum = pcp_min_memory_bytes(config, groups, use_ipc=use_ipc)
    replicated = utils._musa_mamba_pool_bytes(
        config, groups
    ) + utils._pool_bytes_per_block(config, groups[:2]) * pcp_required_blocks(
        config, groups
    )
    assert minimum < replicated
    available = minimum + 2_000_000
    logical = plan_pcp_cache_config(config, groups, available, use_ipc=use_ipc)
    physical = make_pcp_physical_config(config, logical, 8)
    budget = plan_pcp_page_workspace(config, logical, physical, 8, use_ipc=use_ipc)
    assert budget.total_bytes <= available
    assert logical.pcp_memory_budget_bytes == available
    assert (
        _pcp_capacity_candidate(
            config, groups, logical.num_blocks + 1, use_ipc=use_ipc
        )[1].total_bytes
        > available
    )
    smaller = plan_pcp_cache_config(
        config,
        groups,
        available,
        use_ipc=use_ipc,
        num_blocks=pcp_required_blocks(config, groups),
    )
    assert smaller.pcp_memory_budget_bytes == available
    assert utils.get_max_concurrency_for_kv_cache_config(config, smaller) == 1
    with pytest.raises(ValueError, match="exceed available"):
        plan_pcp_cache_config(config, groups, minimum - 1, use_ipc=use_ipc)


def test_physical_block_override_keeps_real_budget(physical_planner_case):
    from vllm.v1.core.kv_cache_utils import (
        pcp_min_memory_bytes,
        plan_pcp_cache_config,
    )
    from vllm.v1.pcp.allocation import (
        pcp_required_blocks,
    )

    config, groups = physical_planner_case
    required = pcp_required_blocks(config, groups)
    config.cache_config.num_gpu_blocks_override = required
    needed = pcp_min_memory_bytes(config, groups, use_ipc=True)
    result = plan_pcp_cache_config(config, groups, needed, use_ipc=True)
    assert result.num_blocks == required
    with pytest.raises(ValueError, match="exceed available"):
        plan_pcp_cache_config(config, groups, needed - 1, use_ipc=True)
    config.cache_config.num_gpu_blocks_override = required - 1
    assert pcp_min_memory_bytes(config, groups, use_ipc=True) == float("inf")
    with pytest.raises(ValueError, match="cannot hold"):
        plan_pcp_cache_config(config, groups, needed, use_ipc=True)


def test_engine_uses_physical_admission_and_allocation(
    physical_planner_case, monkeypatch
):
    from vllm.v1.core import kv_cache_utils as utils
    from vllm.v1.core.kv_cache_utils import pcp_min_memory_bytes

    config, groups = physical_planner_case
    monkeypatch.setattr(utils, "current_platform", SimpleNamespace(device_type="musa"))
    minimum = pcp_min_memory_bytes(config, groups, use_ipc=True)
    assert utils._max_memory_usage_bytes_from_groups(config, groups) == minimum
    logical = utils.get_kv_cache_config_from_groups(config, groups, minimum)
    assert logical.pcp_memory_budget_bytes == minimum
    assert utils.get_max_concurrency_for_kv_cache_config(config, logical) >= 1


def test_physical_length_estimator_respects_budget_and_restores_config(
    physical_planner_case,
    monkeypatch,
):
    from vllm.v1.core import kv_cache_utils as utils
    from vllm.v1.core.kv_cache_utils import pcp_min_memory_bytes

    config, groups = physical_planner_case
    monkeypatch.setattr(utils, "current_platform", SimpleNamespace(device_type="musa"))
    minimum = pcp_min_memory_bytes(config, groups, use_ipc=True)
    original_length = config.model_config.max_model_len
    assert (
        utils._estimate_max_model_len_from_groups(config, groups, minimum)
        == original_length
    )
    reduced = utils._estimate_max_model_len_from_groups(config, groups, minimum - 1)
    assert 0 < reduced < original_length
    assert config.model_config.max_model_len == original_length
    assert utils._estimate_max_model_len_from_groups(config, groups, 1) == 0
    assert config.model_config.max_model_len == original_length


@pytest.mark.parametrize("layout", list(KVCacheLayout))
def test_physical_repacking_matches_resolved_strides(layout):
    from vllm.v1.kv_cache_interface import FullAttentionSpec
    from vllm.v1.pcp.layout import PCPPageLayout

    config = SimpleNamespace(
        cache_config=SimpleNamespace(
            get_resolved_kv_cache_layout=lambda: layout,
        )
    )
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=8,
        dtype=torch.bfloat16,
    )
    groups = [KVCacheGroupSpec(["first", "second"], spec)]
    tensors, _ = _build_attention_kv_cache_tensors(
        config, groups, 0, physical_num_blocks=17
    )
    logical = KVCacheConfig(17, tensors, groups, pcp_layout=PCPPageLayout(8, 16, 4))
    expected, _ = _build_attention_kv_cache_tensors(
        config, groups, 0, physical_num_blocks=3
    )
    physical = make_pcp_physical_config(config, logical)
    assert physical.kv_cache_tensors == expected
    assert logical.kv_cache_tensors == tensors
    with pytest.raises(ValueError, match="disagrees"):
        make_pcp_physical_config(config, logical, 4)


def test_pcp_declaration_survives_spec_merge_and_config_roundtrip():
    import pickle
    from dataclasses import replace

    from vllm.v1.core.kv_cache_utils import generate_scheduler_kv_cache_config
    from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs
    from vllm.v1.pcp.allocation import resolve_pcp_layout

    declaration = PCPCacheSpec(4, "opaque")
    spec = MLAAttentionSpec(
        block_size=64,
        num_kv_heads=1,
        head_size=16,
        dtype=torch.uint8,
        pcp=declaration,
    )
    assert MLAAttentionSpec.merge([spec, spec]).pcp == declaration
    resized = spec.copy_with_new_block_size(128)
    assert resized.pcp == declaration
    wrapped = UniformTypeKVCacheSpecs.from_specs({"arbitrary.a": resized, "z": resized})
    group = KVCacheGroupSpec(["arbitrary.a", "z"], wrapped)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=8, decode_context_parallel_size=1
        )
    )
    layout = resolve_pcp_layout(config, [group])
    logical = KVCacheConfig(17, [], [group], pcp_layout=layout)
    # The budget is deliberately absent: placement must not depend on a byte field.
    restored = pickle.loads(pickle.dumps(logical))
    scheduler = generate_scheduler_kv_cache_config([restored])
    assert scheduler.pcp_layout == layout
    assert scheduler.kv_cache_groups[0].kv_cache_spec.pcp == declaration
    with pytest.raises(ValueError, match="undeclared"):
        resolve_pcp_layout(
            config,
            [
                KVCacheGroupSpec(["declared"], spec),
                KVCacheGroupSpec(["undeclared"], replace(spec, pcp=None)),
            ],
        )
    assert (
        resolve_pcp_layout(
            config,
            [
                KVCacheGroupSpec(
                    ["glm5_next_text.indexer.k_cache"], replace(spec, pcp=None)
                ),
            ],
        )
        is None
    )


def test_declared_attention_does_not_require_a_kda_or_musa_pool(
    physical_planner_case,
    monkeypatch,
):
    from vllm.v1.core import kv_cache_utils as utils
    from vllm.v1.pcp.allocation import pcp_required_blocks

    config, groups = physical_planner_case
    groups = groups[:2]
    monkeypatch.setattr(utils, "musa_mamba_separate_pool_enabled", lambda: False)
    del config.parallel_config.kda_layer_tp
    required = utils.pcp_min_memory_bytes(config, groups, use_ipc=False)
    result = utils.plan_pcp_cache_config(config, groups, required, use_ipc=False)
    assert result.pcp_layout.size == 8
    assert result.num_blocks >= pcp_required_blocks(config, groups)
    assert result.musa_mamba_num_blocks is None
    assert not any(t.musa_mamba_pool for t in result.kv_cache_tensors)


def test_full_and_sliding_specs_preserve_pcp_declarations():
    from dataclasses import replace

    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        SlidingWindowMLASpec,
        SlidingWindowSpec,
    )

    declaration = PCPCacheSpec(4, "shared")
    for cls, extra in (
        (FullAttentionSpec, {}),
        (SlidingWindowSpec, {"sliding_window": 64}),
        (SlidingWindowMLASpec, {"sliding_window": 64}),
    ):
        spec = cls(
            block_size=64,
            num_kv_heads=1,
            head_size=16,
            dtype=torch.uint8,
            pcp=declaration,
            **extra,
        )
        assert cls.merge([spec, spec]).pcp == declaration
        with pytest.raises(AssertionError):
            cls.merge([spec, replace(spec, pcp=None)])
