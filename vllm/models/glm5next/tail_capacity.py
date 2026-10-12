# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KPool tail-ring capacity derivation (PROPOSED CONTRACT — not wired).

Separates the tail ring's PHYSICAL capacity ``C`` from the logical
pool size ``R`` so that a rejected MTP draft's ring write cannot
evict an accepted token's entry before the corrected re-run's pool
completion reads it. See ``docs/design/kpool_tail_ring_capacity.md``
for the full contract and the synthetic CPU evidence.
"""

__all__ = ["kpool_tail_ring_capacity"]


def kpool_tail_ring_capacity(
    index_kpool: int,
    num_speculative_tokens: int,
) -> int:
    """Return the physical tail-ring capacity ``C``.

    ``C = R + S`` where ``S = num_speculative_tokens`` is the maximum
    speculation lookahead (the verify emission width is
    ``V = num_speculative_tokens + 1``; the scheduler treats
    ``num_speculative_tokens`` as an upper bound — suffix decoding
    defaults it to the max tree depth and adaptive verification only
    lowers the budget — so ``V`` is a safe bound). No MTP gives
    ``S = 0`` and ``C = R`` — the current behavior exactly.

    The empirical tight minimum is ``R + V - 2`` for ``V >= 3``
    (see the contract doc); the contract keeps one slot of margin.

    Raises:
        ValueError: on non-positive or non-exact-int inputs.
    """
    if type(index_kpool) is not int or index_kpool < 1:
        raise ValueError(
            f"index_kpool must be a positive exact int, got {index_kpool!r}"
        )
    if type(num_speculative_tokens) is not int or num_speculative_tokens < 0:
        raise ValueError(
            "num_speculative_tokens must be a non-negative exact int, got "
            f"{num_speculative_tokens!r}"
        )
    return index_kpool + num_speculative_tokens
