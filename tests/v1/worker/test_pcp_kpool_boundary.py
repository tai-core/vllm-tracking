# SPDX-License-Identifier: Apache-2.0
import torch

from vllm.v1.pcp.kpool_boundary import boundary_token_rows


def test_boundary_rows_only_include_the_existing_incomplete_pool():
    # A: starts at 1, B: starts at 0, C: starts at 7, D: starts at 10.
    positions = torch.tensor([1, 2, 3, 4, 5, 0, 1, 2, 3, 7, 8, 9, 10])
    blocks = torch.tensor([1] * 5 + [2] * 4 + [3] * 3 + [4])
    slots = blocks * 8 + positions % 8
    assert boundary_token_rows(positions, slots, 4, 8).tolist() == [
        [0, 1, 2, -1],
        [9, -1, -1, -1],
        [12, -1, -1, -1],
    ]


def test_decode_and_padding_are_not_processed_as_prefill_boundaries():
    positions = torch.tensor([3, 2, 3, -1])
    slots = torch.tensor([11, 18, 19, -1])
    assert boundary_token_rows(positions, slots, 4, 8, 1).tolist() == [[1, 2, -1, -1]]
    empty = torch.empty(0, dtype=torch.int64)
    assert boundary_token_rows(empty, empty, 4, 8).shape == (0, 4)


def test_complete_pool_cannot_take_a_previous_requests_token():
    from vllm.v1.pcp.kpool_boundary import complete_pool_mask

    slots = torch.tensor([-1, -1, -1, 0, -1, -1, -1, 1])
    positions = torch.arange(8)
    requests = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1])
    assert complete_pool_mask(slots, positions, requests, 4).tolist() == [
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        False,
    ]


def test_batch_chunk_starting_at_703_keeps_the_leading_pool():
    from vllm.v1.pcp.kpool_boundary import complete_pool_mask

    # The preceding batch consumed 703 tokens of a 1024-token request.
    # This batch finishes it and starts 703 tokens of the next request.
    positions = torch.cat((torch.arange(703, 1024), torch.arange(703)))
    requests = torch.tensor([11] * 321 + [13] * 703)
    tails = requests * 8 + positions % 8
    rows = boundary_token_rows(positions, tails, 4, 8)
    assert rows.tolist() == [[0, -1, -1, -1]]
    slots = torch.where(positions % 4 == 3, positions // 4, -1)
    complete = complete_pool_mask(slots, positions, requests, 4)
    assert not complete[0]  # Supplied by the previous chunk's persistent tail.
    assert complete.sum().item() == 80 + 175
    # The next step must complete pool 175 before ordinary local compression.
    positions = torch.arange(703, 1727)
    tails = 13 * 8 + positions % 8
    assert boundary_token_rows(positions, tails, 4, 8).tolist() == [[0, -1, -1, -1]]
