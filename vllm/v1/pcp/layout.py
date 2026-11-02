# SPDX-License-Identifier: Apache-2.0
"""Logical PCP ownership, independent of decode context parallelism.

Block IDs belong to the scheduler's logical pool. A block is stored only on
block_id % size; its local physical ID is block_id // size. Negative slots
are invalid. Physical block zero is valid (unlike SGLang's dummy page).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PCPPageLayout:
    size: int
    page_tokens: int
    pool_tokens: int = 1

    def __post_init__(self) -> None:
        if min(self.size, self.page_tokens, self.pool_tokens) <= 0:
            raise ValueError("PCP size and page/pool dimensions must be positive")
        if self.page_tokens % self.pool_tokens:
            raise ValueError("A cache page must contain complete index pools")

    def locate(self, logical_block: int) -> tuple[int, int]:
        if logical_block < 0:
            raise ValueError("Cannot address an invalid logical block")
        return logical_block % self.size, logical_block // self.size

    def logical_block(self, owner: int, physical_block: int) -> int:
        self._check_rank(owner)
        if physical_block < 0:
            raise ValueError("Physical block must be nonnegative")
        return physical_block * self.size + owner

    def local_capacity(self, logical_capacity: int, rank: int) -> int:
        self._check_rank(rank)
        if logical_capacity < 0:
            raise ValueError("Logical capacity must be nonnegative")
        return max(0, (logical_capacity + self.size - 1 - rank) // self.size)

    def local_slot(self, logical_slot: int, rank: int) -> int:
        self._check_rank(rank)
        if logical_slot < 0:
            return -1
        block, offset = divmod(logical_slot, self.page_tokens)
        owner, physical = self.locate(block)
        return physical * self.page_tokens + offset if owner == rank else -1

    def split_query(self, start: int, length: int) -> tuple[QuerySpan, ...]:
        """Balance complete pools across paired early/late compute spans.

        Storage ownership still uses page_tokens. Several compute ranks may
        produce disjoint rows of one storage page; composition routes each row
        to its unique owner. Only request endpoints may split a pool.
        """
        if start < 0 or length < 0:
            raise ValueError("Query bounds must be nonnegative")
        if length == 0:
            return ()
        first = start // self.pool_tokens
        stop = start + length
        units = (stop + self.pool_tokens - 1) // self.pool_tokens - first
        whole, remainder = divmod(units, self.size)
        rank_units = [whole + (rank < remainder) for rank in range(self.size)]
        chunk_units = [count // 2 for count in rank_units]
        chunk_units += [(count + 1) // 2 for count in reversed(rank_units)]
        spans = []
        offset = first
        for chunk, count in enumerate(chunk_units):
            begin = max(start, offset * self.pool_tokens)
            offset += count
            end = min(stop, offset * self.pool_tokens)
            if begin < end:
                rank = min(chunk, 2 * self.size - 1 - chunk)
                spans.append(QuerySpan(rank, begin, end))
        return tuple(spans)

    def _check_rank(self, rank: int) -> None:
        if not 0 <= rank < self.size:
            raise ValueError("Rank is outside the PCP group")


@dataclass(frozen=True)
class QuerySpan:
    compute_rank: int
    start: int
    stop: int


@dataclass(frozen=True)
class PCPPageTransfer:
    request_id: str
    cache_kind: str
    source_rank: int
    source_block: int
    source_offset: int
    request_start: int
    token_count: int
    decode_dp_rank: int
    version: int = 1


def plan_page_transfers(
    layout: PCPPageLayout,
    request_id: str,
    cache_kind: str,
    logical_blocks: tuple[int, ...],
    num_tokens: int,
    decode_dp_rank: int,
    decode_dp_size: int = 8,
) -> tuple[PCPPageTransfer, ...]:
    """Describe MLA/index bytes in token coordinates, without decode DCP.

    Index transport converts token coordinates to pool rows, retaining the
    incomplete pool separately as tail state. It must not round that pool up
    into a completed compressed index entry.
    """
    if cache_kind not in ("mla", "index"):
        raise ValueError("Only page-backed MLA/index payloads use this mapping")
    if not 0 <= decode_dp_rank < decode_dp_size:
        raise ValueError("Invalid decode data-parallel rank")
    if num_tokens < 0:
        raise ValueError("Token count must be nonnegative")
    count = (num_tokens + layout.page_tokens - 1) // layout.page_tokens
    if len(logical_blocks) != count:
        raise ValueError("Block table must exactly cover the valid token extent")
    if len(set(logical_blocks)) != len(logical_blocks):
        raise ValueError("A request cannot alias two logical page positions")
    result = []
    for position, block in enumerate(logical_blocks):
        owner, physical = layout.locate(block)
        start = position * layout.page_tokens
        valid = min(layout.page_tokens, num_tokens - start)
        if cache_kind == "index":
            valid -= valid % layout.pool_tokens
        if valid:
            result.append(
                PCPPageTransfer(
                    request_id,
                    cache_kind,
                    owner,
                    physical,
                    0,
                    start,
                    valid,
                    decode_dp_rank,
                )
            )
    return tuple(result)


@dataclass(frozen=True)
class KDAHeadTransfer:
    request_id: str
    cache_kind: str
    source_rank: int
    destination_rank: int
    head_start: int
    head_count: int
    request_window: int
    version: int = 1


def plan_kda_transfers(
    num_heads: int,
    prefill_tp_size: int,
    decode_tp_size: int,
    decode_dp_rank: int,
    decode_dp_size: int = 8,
    *,
    request_id: str,
    cache_kind: str,
) -> tuple[KDAHeadTransfer, ...]:
    """Route one request's state to every intersecting decode head shard.

    request_window identifies the request's DP lane inside the KDA TP group;
    destination_rank identifies a head shard, not the MLA request owner.
    The same mapping applies separately to recurrent and convolution states.
    """
    if not request_id or cache_kind not in ("kda_recurrent", "kda_conv"):
        raise ValueError("KDA transfer requires request identity and a state kind")
    if min(num_heads, prefill_tp_size, decode_tp_size) <= 0:
        raise ValueError("Head and group sizes must be positive")
    if num_heads % prefill_tp_size or num_heads % decode_tp_size:
        raise ValueError("Head count must be divisible by both TP sizes")
    if not 0 <= decode_dp_rank < decode_dp_size:
        raise ValueError("Invalid decode data-parallel rank")
    source_width = num_heads // prefill_tp_size
    target_width = num_heads // decode_tp_size
    result = []
    for source in range(prefill_tp_size):
        for target in range(decode_tp_size):
            start = max(source * source_width, target * target_width)
            stop = min((source + 1) * source_width, (target + 1) * target_width)
            if start < stop:
                result.append(
                    KDAHeadTransfer(
                        request_id,
                        cache_kind,
                        source,
                        target,
                        start,
                        stop - start,
                        decode_dp_rank,
                    )
                )
    return tuple(result)


@dataclass(frozen=True)
class PCPTailTransfer:
    request_id: str
    source_rank: int
    source_block: int
    request_start: int
    source_ring_offsets: tuple[int, ...]
    decode_dp_rank: int
    cache_kind: str = "index_tail"
    version: int = 1


def plan_tail_transfer(
    layout: PCPPageLayout,
    request_id: str,
    logical_tail_block: int,
    num_tokens: int,
    ring_tokens: int,
    decode_dp_rank: int,
    decode_dp_size: int = 8,
) -> PCPTailTransfer | None:
    """Transfer only the incomplete pool's K and gate at their ring offsets."""
    if not request_id or num_tokens < 0:
        raise ValueError("Tail transfer requires a request and nonnegative length")
    if ring_tokens < layout.pool_tokens or ring_tokens % layout.pool_tokens:
        raise ValueError("Tail ring must contain an integral number of pools")
    if not 0 <= decode_dp_rank < decode_dp_size:
        raise ValueError("Invalid decode data-parallel rank")
    owner, physical = layout.locate(logical_tail_block)
    tail = num_tokens % layout.pool_tokens
    if tail == 0:
        return None
    start = num_tokens - tail
    return PCPTailTransfer(
        request_id,
        owner,
        physical,
        start,
        tuple(position % ring_tokens for position in range(start, num_tokens)),
        decode_dp_rank,
    )


class PCPTransferLease:
    """Keep source pages pinned until all declared payloads complete.

    The caller owns the actual allocator reference. Cancellation does not
    release it: in-flight reads must complete or acknowledge abort first.
    """

    def __init__(self, payload_ids: frozenset[str]) -> None:
        if not payload_ids:
            raise ValueError("A transfer must declare its required payloads")
        self._required = payload_ids
        self._done: set[str] = set()
        self._cancelled = False

    def complete(self, payload_id: str) -> None:
        if payload_id not in self._required:
            raise ValueError("Unknown transfer payload")
        self._done.add(payload_id)

    def cancel(self) -> None:
        self._cancelled = True

    @property
    def releasable(self) -> bool:
        return self._done == self._required

    @property
    def decode_ready(self) -> bool:
        return self.releasable and not self._cancelled
