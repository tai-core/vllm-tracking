# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu import dp_utils
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor


class _Work:
    def __init__(self):
        self.wait_calls = 0

    def wait(self):
        self.wait_calls += 1


def _reduced(epoch=(7, 7), global_mode=(0, 0), fallback=(0, 0), dummy=(0, 0), rows=10):
    tensor = torch.zeros(rows, 2, dtype=torch.int32)
    tensor[0] = 16
    tensor[1] = CUDAGraphMode.NONE.value
    tensor[2] = 0
    tensor[3] = -1
    tensor[4] = 0
    tensor[5] = 2
    if rows == 10:
        tensor[6] = torch.tensor(epoch, dtype=torch.int32)
        tensor[7] = torch.tensor(global_mode, dtype=torch.int32)
        tensor[8] = torch.tensor(fallback, dtype=torch.int32)
        tensor[9] = torch.tensor(dummy, dtype=torch.int32)
    return tensor


def _desc():
    return BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.NONE,
        num_tokens=16,
        num_reqs=2,
    )


def _patch_reduce(reduced, work):
    def all_reduce(tensor, group, async_op=False):
        tensor.copy_(reduced)
        return work if async_op else tensor

    return (
        patch.object(dp_utils.dist, "all_reduce", all_reduce),
        patch.object(dp_utils, "get_dp_group", lambda: SimpleNamespace(cpu_group=None)),
    )


def test_issue_resolve_matches_legacy_and_waits_once():
    work = _Work()
    reduce_patch, group_patch = _patch_reduce(_reduced(), work)
    with reduce_patch, group_patch:
        plan = dp_utils.issue_dp_shape_agreement(
            None, _desc(), 16, 2, None, 2, 0, epoch=7
        )
        desc, sync = dp_utils.resolve_dp_shape_agreement(
            plan, None, _desc(), 16, 2, None, 0
        )
        assert plan.use_plan
        assert work.wait_calls == 1
        assert dp_utils.resolve_dp_shape_agreement(
            plan, None, _desc(), 16, 2, None, 0
        ) == (desc, sync)
        assert work.wait_calls == 1

    with (
        patch.object(
            dp_utils.dist,
            "all_reduce",
            lambda t, group: t.copy_(_reduced(rows=6)),
        ),
        patch.object(dp_utils, "get_dp_group", lambda: SimpleNamespace(cpu_group=None)),
    ):
        legacy_desc, legacy_sync = dp_utils.sync_cudagraph_and_dp_padding(
            None, _desc(), 16, 2, None, 2, 0
        )

    assert desc == legacy_desc
    assert sync is not None and legacy_sync is not None
    assert sync.uniform_token_count == legacy_sync.uniform_token_count
    assert sync.eager == legacy_sync.eager
    assert sync.num_reqs == legacy_sync.num_reqs
    assert sync.max_num_reqs_across_dp == legacy_sync.max_num_reqs_across_dp
    torch.testing.assert_close(
        sync.num_tokens_across_dp, legacy_sync.num_tokens_across_dp
    )


def test_full_graph_padding_does_not_change_unpadded_request_bound():
    work = _Work()
    reduced = _reduced(global_mode=(1, 1))
    reduced[1] = CUDAGraphMode.FULL.value
    reduced[5] = torch.tensor([2, 3], dtype=torch.int32)
    reduce_patch, group_patch = _patch_reduce(reduced, work)
    manager = SimpleNamespace(
        dispatch=lambda *args, **kwargs: BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.FULL,
            num_tokens=16,
            num_reqs=8,
        )
    )
    desired = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL,
        num_tokens=16,
        num_reqs=2,
    )
    with reduce_patch, group_patch:
        plan = dp_utils.issue_dp_shape_agreement(
            manager, desired, 16, 2, None, 2, 0, epoch=7, global_mode=1
        )
        _, sync = dp_utils.resolve_dp_shape_agreement(
            plan, manager, desired, 16, 2, None, 0
        )

    assert sync is not None
    assert sync.num_reqs == 8
    assert sync.max_num_reqs_across_dp == 3
    assert sync.max_spec_reqs_across_dp == 3


def test_mixed_real_dummy_agreement_uses_one_padded_spec_bound():
    work = _Work()
    reduced = _reduced(global_mode=(1, 1), dummy=(0, 1))
    reduced[1] = CUDAGraphMode.FULL.value
    reduced[5] = torch.tensor([2, 3], dtype=torch.int32)
    reduce_patch, group_patch = _patch_reduce(reduced, work)
    manager = SimpleNamespace(
        dispatch=lambda *args, **kwargs: BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.FULL,
            num_tokens=16,
            num_reqs=8,
        )
    )
    desired = BatchExecutionDescriptor(
        cg_mode=CUDAGraphMode.FULL,
        num_tokens=16,
        num_reqs=2,
    )
    with reduce_patch, group_patch:
        real_plan = dp_utils.issue_dp_shape_agreement(
            manager,
            desired,
            16,
            2,
            None,
            2,
            0,
            epoch=7,
            global_mode=1,
            pad_spec_requests=False,
        )
        _, real_sync = dp_utils.resolve_dp_shape_agreement(
            real_plan, manager, desired, 16, 2, None, 0
        )
        dummy_plan = dp_utils.issue_dp_shape_agreement(
            manager,
            desired,
            16,
            2,
            None,
            2,
            1,
            epoch=7,
            global_mode=1,
            pad_spec_requests=True,
        )
        _, dummy_sync = dp_utils.resolve_dp_shape_agreement(
            dummy_plan, manager, desired, 16, 2, None, 1
        )

    assert real_sync is not None and dummy_sync is not None
    assert real_sync.max_spec_reqs_across_dp == 8
    assert dummy_sync.max_spec_reqs_across_dp == 8


def test_real_and_dummy_plans_use_their_own_request_padding_semantics():
    from vllm.models.glm5next.common.kda_dp_plan import get_kda_dp_request_bound
    from vllm.v1.worker.gpu.dp_batch_plan import DPSyncState

    sync = DPSyncState(
        num_tokens_across_dp=torch.tensor([16, 16], dtype=torch.int32),
        uniform_token_count=8,
        eager=False,
        num_reqs=8,
        max_num_reqs_across_dp=3,
        global_mode=1,
        use_plan=True,
    )

    assert get_kda_dp_request_bound(sync, use_padded_requests=False) == 3
    assert get_kda_dp_request_bound(sync, use_padded_requests=True) == 8


def test_immediate_dispatch_does_not_create_an_async_work_wrapper():
    work = _Work()
    reduce_patch, group_patch = _patch_reduce(_reduced(), work)
    with reduce_patch, group_patch:
        _, sync = dp_utils.dispatch_cg_and_sync_dp(
            None,
            num_reqs=2,
            num_tokens=16,
            uniform_token_count=None,
            dp_size=2,
            dp_rank=0,
            need_eager=True,
            plan_epoch=7,
        )

    assert sync is not None and sync.use_plan
    assert work.wait_calls == 0


def test_epoch_mismatch_falls_back_without_changing_dp_sync():
    work = _Work()
    reduce_patch, group_patch = _patch_reduce(_reduced(epoch=(7, 8)), work)
    with reduce_patch, group_patch:
        plan = dp_utils.issue_dp_shape_agreement(
            None, _desc(), 16, 2, None, 2, 0, epoch=7
        )
        desc, sync = dp_utils.resolve_dp_shape_agreement(
            plan, None, _desc(), 16, 2, None, 0
        )

    assert not plan.use_plan
    assert plan.fallback_reason == "epoch_mismatch"
    assert desc.num_tokens == 16
    assert sync is not None


def test_fallback_vote_is_shared_in_the_same_agreement():
    work = _Work()
    reduce_patch, group_patch = _patch_reduce(_reduced(fallback=(0, 1)), work)
    with reduce_patch, group_patch:
        plan = dp_utils.issue_dp_shape_agreement(
            None, _desc(), 16, 2, None, 2, 0, epoch=7
        )
        dp_utils.resolve_dp_shape_agreement(plan, None, _desc(), 16, 2, None, 0)

    assert not plan.use_plan
    assert plan.fallback_reason == "fallback_vote"
