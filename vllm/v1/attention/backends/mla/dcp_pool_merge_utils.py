# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hardware-independent DCP candidate packing and stable global top-k.

Two pure-torch functions for the GLM KPool DCP merge, CPU-executable
as-is (no collectives, no Triton/CuteDSL, no GPU-only imports):

1. ``pack_local_pool_candidates`` — the per-rank packing step. Each
   DCP rank's local top-k indices are REQUEST-LOCAL pool ids relative
   to the row's pool-0 column (the contract of
   ``top_k_per_row_prefill``; the device kernel stores
   ``idx - rowStart`` on the strided path and counts from ``rowStart``
   on the vectorized path). The pack resolves each candidate's score
   from the rank's logits table at column ``row_start + local_id`` and
   maps the local id to its GLOBAL pool id
   ``(local_id // J) * (W * J) + rank * J + local_id % J`` (int64
   throughout, matching the CuteDSL pack kernel's arithmetic).

2. ``stable_global_pool_topk`` — the selection step after the
   gather: valid candidates first; among valid, score descending
   with ties broken by global pool id ascending; invalid candidates
   (id == -1 or a NaN score) always sort after every valid candidate.
   Valid ids span the full non-negative int64 range (including
   torch.iinfo(int64).max). Valid +/-inf scores are legal; -inf
   ranks below every finite score but above invalid. Output rows
   shorter than k in valid candidates pad with (id=-1, score=-inf).

3. ``select_global_pool_topk_dcp`` — the thin pack -> all_gather ->
   select adapter binding 1 and 2 together through an injected
   group object (see its docstring for the caller's collective
   ordering responsibilities).

Preconditions (documented, NOT enforced in selection): per-row valid
pool ids are unique (upstream shard split is disjoint and each rank's
local top-k has no duplicates); no duplicate-id merging happens in
that layer.
"""

from typing import Protocol

import torch

__all__ = [
    "derive_decode_row_bounds",
    "pack_local_pool_candidates",
    "select_global_pool_topk_dcp",
    "select_local_pool_candidates",
    "stable_global_pool_topk",
]


class _AllGatherGroup(Protocol):
    """The minimal group interface the adapter needs.

    Matches ``vllm.distributed.GroupCoordinator``'s relevant surface:
    ``world_size``/``rank_in_group`` positive-int attributes and
    ``all_gather(input_, dim=-1) -> torch.Tensor`` (world_size == 1
    returns the input unchanged; world_size > 1 delegates to the
    device communicator). Keeping this a Protocol avoids importing
    any distributed runtime in this module.
    """

    @property
    def world_size(self) -> int: ...

    @property
    def rank_in_group(self) -> int: ...

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor: ...


def pack_local_pool_candidates(
    logits: torch.Tensor,
    local_pool_ids: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    dcp_world_size: int,
    dcp_rank: int,
    pool_interleave: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack one DCP rank's local top-k candidates for the global merge.

    Args:
        logits: fp32 [rows, cols] this rank's pool-score table (row r
            is one query; column c is the score of the pool at column
            offset c, i.e. the rank's pool ``(c // J) * W * J + rank
            * J + c % J``).
        local_pool_ids: int32 or int64 [rows, K] request-local pool
            ids RELATIVE to each row's pool-0 column (the contract of
            ``top_k_per_row_prefill``); negative entries are padding
            and become invalid candidates.
        row_starts: int32 or int64 [rows] the row's pool-0 column
            (absolute, in ``[0, cols]``).
        row_ends: int32 or int64 [rows] the row's absolute exclusive
            causal end column, in ``[row_start, cols]``.
        dcp_world_size / dcp_rank: positive ints, ``0 <= rank < world``.
        pool_interleave: J, a positive int in POOL units (the pool
            group this rank owns contiguously).

    Row bounds: each row must satisfy ``0 <= row_start <= row_end <=
    cols``. A row violating any bound (negative start, end < start,
    end beyond the table) contributes ONLY invalid ``(-inf, -1)``
    candidates — never a clamped read presented as a real score; the
    check is on-device with no host sync and no exception.

    Returns:
        (candidate_scores, global_pool_ids): fp32 [rows, K] and int64
        [rows, K]. An entry is valid iff its local id is a non-negative
        int whose score column ``row_start + local_id`` lies within
        ``[row_start, row_end)`` AND its gathered score is not NaN.
        Valid scores (including legal +/-inf) pass through; invalid
        entries are ``(-inf, -1)``. Inputs are never modified; ids are
        never float-encoded; no host syncs; no per-row Python.

    Overflow safety (two layers): (1) ``local_id`` is bounded by the
    row's AVAILABLE length ``row_end - row_start`` BEFORE any
    addition, so an int64-huge local id can never wrap
    ``row_start + local_id``; (2) a STATIC Python-int bound
    ``(cols - 1) * W + rank * J + (J - 1)`` — provably >= every gid
    this call can emit, since ``((cols-1) // J) * (W * J) + rank * J +
    (cols-1) % J <= (cols-1) * W + rank * J + J - 1`` — is checked
    against int64 max BEFORE any tensor arithmetic. A (W, rank, J)
    combination that is not int64-representable for this table width
    raises ValueError here rather than silently wrapping (including
    the ``rank * J`` term). Practical ranges (e.g. J ~ 2^20 with
    cols ~ 2^32) map far inside the bound; global ids beyond 2^32
    are fully supported and exact.

    Raises:
        ValueError: on wrong ndim/dtype/device/shape, invalid scalar
            parameters, or a (dcp_world_size, dcp_rank,
            pool_interleave) combination whose global-id bound
            overflows int64 for the given table width.
    """
    if type(dcp_world_size) is not int or dcp_world_size < 1:
        raise ValueError(
            f"dcp_world_size must be a positive exact int, got {dcp_world_size!r}"
        )
    if type(dcp_rank) is not int or not 0 <= dcp_rank < dcp_world_size:
        raise ValueError(
            f"dcp_rank must be an exact int in [0, dcp_world_size), got "
            f"{dcp_rank!r} for dcp_world_size {dcp_world_size}"
        )
    if type(pool_interleave) is not int or pool_interleave < 1:
        raise ValueError(
            f"pool_interleave must be a positive exact int, got {pool_interleave!r}"
        )
    if logits.dtype is not torch.float32:
        raise ValueError(f"logits must be fp32, got {logits.dtype}")
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2-D [rows, cols], got {logits.ndim}-D")
    if local_pool_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            f"local_pool_ids must be int32 or int64, got {local_pool_ids.dtype}"
        )
    if local_pool_ids.ndim != 2:
        raise ValueError(
            f"local_pool_ids must be 2-D [rows, K], got {local_pool_ids.ndim}-D"
        )
    rows, cols = logits.shape
    if local_pool_ids.shape[0] != rows:
        raise ValueError(
            "local_pool_ids rows must match logits rows, got "
            f"{local_pool_ids.shape[0]} vs {rows}"
        )
    if row_starts.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"row_starts must be int32 or int64, got {row_starts.dtype}")
    if row_ends.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"row_ends must be int32 or int64, got {row_ends.dtype}")
    if row_starts.ndim != 1 or row_ends.ndim != 1:
        raise ValueError(
            f"row_starts/row_ends must be 1-D [rows], got "
            f"{row_starts.ndim}-D and {row_ends.ndim}-D"
        )
    if row_starts.shape[0] != rows or row_ends.shape[0] != rows:
        raise ValueError(
            "row_starts/row_ends must have one entry per logits row, got "
            f"{row_starts.shape[0]}/{row_ends.shape[0]} vs {rows}"
        )
    dev = logits.device
    if (
        local_pool_ids.device != dev
        or row_starts.device != dev
        or row_ends.device != dev
    ):
        raise ValueError("all inputs must share the logits device")

    # Normalize candidate ids and the row bounds to int64 on the
    # device (no host sync; empty inputs still validate shape above).
    # Static overflow guard, computed in Python exact ints BEFORE any
    # tensor arithmetic: for any valid local id (< cols), the global id
    # ((id // J) * (W * J) + rank * J + id % J) is bounded by
    # (cols - 1) * W + rank * J + (J - 1) <= INT64_MAX. A (W, rank, J)
    # combination whose bound is not int64-representable is REJECTED
    # here rather than silently wrapping. Practical configurations
    # (J up to ~2^20 with cols up to ~2^32) map far inside this bound;
    # global ids beyond 2^32 remain fully supported and exact.
    _int64_max = torch.iinfo(torch.int64).max
    _max_gid = (
        (cols - 1) * dcp_world_size + dcp_rank * pool_interleave + (pool_interleave - 1)
    )
    if (
        _max_gid > _int64_max
        # The mapping's intermediate product W * J must itself be
        # int64-representable (rank * J <= (W-1) * J < W * J is then
        # representable too): without this check W=5, rank=0, J=2**62
        # slips past the _max_gid bound on a narrow table and then
        # wraps inside the tensor multiply.
        or dcp_world_size * pool_interleave > _int64_max
    ):
        raise ValueError(
            f"the global pool id mapping overflows int64 for these "
            f"parameters: bound {_max_gid} (intermediate W*J = "
            f"{dcp_world_size * pool_interleave}) exceeds int64 max with "
            f"dcp_world_size={dcp_world_size}, dcp_rank={dcp_rank}, "
            f"pool_interleave={pool_interleave}; use smaller "
            f"pool_interleave/dcp_world_size"
        )
    ids64 = local_pool_ids.to(torch.int64)
    starts = row_starts.to(torch.int64)
    ends = row_ends.to(torch.int64)
    _, k = ids64.shape

    out_scores = torch.full((rows, k), float("-inf"), dtype=torch.float32, device=dev)
    out_ids = torch.full((rows, k), -1, dtype=torch.int64, device=dev)
    if rows == 0 or k == 0 or cols == 0:
        return out_scores, out_ids

    # Row-bound validity, computed on the device BEFORE any address
    # arithmetic: a row is usable iff 0 <= start <= end <= cols. A row
    # violating any bound contributes NO valid candidate (never a
    # clamped read presented as a real score), and no host sync or
    # exception happens. Sanitize first so the subtraction below
    # cannot wrap even for int64-min/max bounds.
    valid_rows = (starts >= 0) & (ends >= starts) & (ends <= cols)
    safe_starts = torch.where(valid_rows, starts, torch.zeros_like(starts))
    safe_ends = torch.where(valid_rows, ends, torch.zeros_like(ends))
    # Structural candidate validity: the id is in-range iff
    # 0 <= id < row_end - row_start (safe_ends >= safe_starts always).
    nonneg = ids64 >= 0
    available = safe_ends - safe_starts
    in_range = nonneg & (ids64 < available.unsqueeze(-1)) & valid_rows.unsqueeze(-1)
    # The actual score column, only ever computed for in-range ids;
    # safe_local stays bounded by `available` so the addition cannot
    # overflow int64.
    safe_local = torch.where(in_range, ids64, torch.zeros_like(ids64))
    score_col = safe_starts.unsqueeze(-1) + safe_local
    # Defensive gather-side clamp: with the bound above, score_col is
    # already within [0, cols]; clamp ONLY as a gather guard, and mask
    # the result back to invalid so a clamped read is never emitted.
    gather_col = score_col.clamp(0, cols - 1)
    gathered = logits.gather(1, gather_col.to(torch.int64))
    valid = in_range & ~torch.isnan(gathered)

    # Global pool id: (local // J) * (W * J) + rank * J + local % J,
    # int64 throughout. row_starts/row_ends only addressed the score
    # and never enter this mapping.
    group = safe_local // pool_interleave
    global_id = (
        group * (dcp_world_size * pool_interleave)
        + dcp_rank * pool_interleave
        + safe_local % pool_interleave
    )

    out_scores = torch.where(valid, gathered, out_scores)
    out_ids = torch.where(valid, global_id, out_ids)
    return out_scores, out_ids


def stable_global_pool_topk(
    gathered_scores: torch.Tensor,
    gathered_global_pool_ids: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the stable global top-k pool ids per row.

    Args:
        gathered_scores: fp32 [rows, candidates] candidate scores.
        gathered_global_pool_ids: int64 [rows, candidates] global pool
            ids; -1 marks an invalid candidate.
        k: the number of entries to select per row.

    Returns:
        (ids, scores): int64 [rows, k] and fp32 [rows, k]. Entries
        beyond a row's valid candidate count are (id=-1,
        score=-inf). The inputs are never modified; ids are never
        float-encoded at this layer.

    Raises:
        ValueError: on wrong ndim/dtype/device/shape or a non-positive
            or bool k.
    """
    if type(k) is not int or k < 1:
        raise ValueError(f"k must be a positive exact int, got {k!r}")
    if gathered_scores.dtype is not torch.float32:
        raise ValueError(f"gathered_scores must be fp32, got {gathered_scores.dtype}")
    if gathered_global_pool_ids.dtype is not torch.int64:
        raise ValueError(
            "gathered_global_pool_ids must be int64, got "
            f"{gathered_global_pool_ids.dtype}"
        )
    if gathered_scores.device != gathered_global_pool_ids.device:
        raise ValueError(
            "gathered_scores and gathered_global_pool_ids must share a "
            f"device, got {gathered_scores.device} vs "
            f"{gathered_global_pool_ids.device}"
        )
    if gathered_scores.ndim != 2 or gathered_global_pool_ids.ndim != 2:
        raise ValueError(
            "gathered_scores and gathered_global_pool_ids must be 2-D "
            f"[rows, candidates], got {gathered_scores.ndim}-D and "
            f"{gathered_global_pool_ids.ndim}-D"
        )
    if gathered_scores.shape != gathered_global_pool_ids.shape:
        raise ValueError(
            "gathered_scores and gathered_global_pool_ids must share a "
            f"shape, got {tuple(gathered_scores.shape)} vs "
            f"{tuple(gathered_global_pool_ids.shape)}"
        )

    rows, candidates = gathered_scores.shape
    out_ids = torch.full(
        (rows, k), -1, dtype=torch.int64, device=gathered_scores.device
    )
    out_scores = torch.full(
        (rows, k),
        float("-inf"),
        dtype=torch.float32,
        device=gathered_scores.device,
    )
    if rows == 0 or candidates == 0:
        return out_ids, out_scores

    # Validity: a non-negative id AND a non-NaN score. Valid -inf stays
    # valid (it must outrank every invalid candidate).
    valid = (gathered_global_pool_ids >= 0) & ~torch.isnan(gathered_scores)

    # The selection order is a lexicographic (valid-first, score-desc,
    # id-asc) sort. torch has no 64-bit lexsort and ids must not be
    # float-encoded, so build it from STABLE single-key sorts applied
    # least-significant-key first (the standard radix argument):
    #   1) id ascending (int64 exact),
    #   2) score descending,
    #   3) valid first.
    # After step 3, within each row the entries are valid-major; the
    # earlier sorts' relative order survives each later stable pass,
    # yielding exactly (valid, score desc, id asc).
    _, by_id = torch.sort(gathered_global_pool_ids, dim=-1, stable=True)
    ids_id_order = gathered_global_pool_ids.gather(-1, by_id)
    scores_id_order = gathered_scores.gather(-1, by_id)
    # Descending score without mutating inputs: sort the negation
    # ascending. NaN scores only occur on invalid lanes; they were
    # already relegated by validity and their relative position among
    # invalids is irrelevant (invalids are never selected).
    _, by_neg_score = torch.sort(
        torch.where(
            torch.isnan(scores_id_order),
            torch.zeros_like(scores_id_order),
            -scores_id_order,
        ),
        dim=-1,
        stable=True,
    )
    ids_ns_order = ids_id_order.gather(-1, by_neg_score)
    scores_ns_order = scores_id_order.gather(-1, by_neg_score)
    valid_ns_order = valid.gather(-1, by_id).gather(-1, by_neg_score)
    # Valid-major: True (1) sorts after False (0) ascending, so sort
    # DESCENDING on validity to bring valid first.
    _, by_valid = torch.sort(valid_ns_order, dim=-1, descending=True, stable=True)
    ordered_ids = ids_ns_order.gather(-1, by_valid)
    ordered_scores = scores_ns_order.gather(-1, by_valid)
    ordered_valid = valid_ns_order.gather(-1, by_valid)

    take = min(k, candidates)
    sel_ids = ordered_ids[:, :take]
    sel_scores = ordered_scores[:, :take]
    sel_valid = ordered_valid[:, :take]
    # Invalid selections and their scores become the pad pair.
    out_ids[:, :take] = torch.where(sel_valid, sel_ids, torch.full_like(sel_ids, -1))
    out_scores[:, :take] = torch.where(
        sel_valid,
        sel_scores,
        torch.full_like(sel_scores, float("-inf")),
    )
    return out_ids, out_scores


def derive_decode_row_bounds(
    seq_lens: torch.Tensor,
    num_rows: int,
    next_n: int,
    decode_lens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Derive per-query-row (row_start, row_end) for the decode merge.

    ``seq_lens`` is the decode metadata's per-token compressed causal
    bound tensor — the SAME tensor handed to the paged-MQA logits.
    Supported layouts (validated, not just numel-matched):

    - native padded ``(B, next_n)``: the logits rows follow
      ``row = b * next_n + j``; token j's bound must come from the
      per-request-actual construction (the builder's DCP path), and
      rows with ``j >= decode_lens[b]`` are padding — masked to 0.
    - plain/flatten ``(B, 1)`` or the flattened per-token 1-D view:
      one bound per row (``decode_lens[b] == 1`` per row).

    Any other shape is rejected from HOST-visible ndim/shape facts
    only. The indices/varlen layouts — whose flattened numel may
    coincide but whose semantics differ — are rejected by the CALLER
    via decode_metadata.indices before this helper runs; this helper
    deliberately performs NO tensor-value probing (no bool(tensor),
    no .sum().item(), no torch.all on device data), so it never
    syncs. Padded/off-contract rows are masked to 0 ON DEVICE; the
    global max_seq is never substituted.
    """
    if seq_lens.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            f"decode seq_lens must be int32 or int64, got {seq_lens.dtype}"
        )
    if decode_lens.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"decode_lens must be int32 or int64, got {decode_lens.dtype}")
    if type(next_n) is not int or next_n < 1:
        raise ValueError(f"next_n must be a positive int, got {next_n!r}")
    if type(num_rows) is not int or num_rows < 0:
        raise ValueError(f"num_rows must be a non-negative int, got {num_rows!r}")
    if seq_lens.device != decode_lens.device:
        raise ValueError(
            f"decode seq_lens and decode_lens must share a device, got "
            f"{seq_lens.device} vs {decode_lens.device}"
        )
    if decode_lens.dim() != 1:
        raise ValueError(f"decode_lens must be 1-D [B], got {decode_lens.ndim}-D")
    B = decode_lens.shape[0]
    if seq_lens.dim() == 2 and seq_lens.shape[0] == B:
        # Native padded [B, next_n]: the layout is [B, next_n] exactly
        # (a HOST shape fact — no tensor reads); the padded rows are
        # masked ON DEVICE via decode_lens (never trusting that a pad
        # row already carries 0). decode_lens' per-request semantics
        # are guaranteed by the builder (host decode_lens_cpu).
        if seq_lens.shape[1] != next_n:
            raise ValueError(
                f"decode seq_lens layout {tuple(seq_lens.shape)} does not "
                f"match the logits' [B*next_n] rows (next_n={next_n})"
            )
        flat = seq_lens.reshape(-1)
        j_grid = (
            torch.arange(next_n, device=seq_lens.device).unsqueeze(0).expand(B, next_n)
        )
        valid_row = j_grid < decode_lens.unsqueeze(1).to(j_grid.dtype)
        ends = torch.where(
            valid_row.reshape(-1), flat.clamp_min(0), torch.zeros_like(flat)
        )
    elif seq_lens.dim() == 2 and seq_lens.shape[1] == 1 and num_rows == B:
        # Plain decode (B, 1): one row per request, decode_len 1 (a
        # builder-guaranteed contract; an off-contract row is masked
        # to 0 ON DEVICE rather than triggering a host sync).
        flat = seq_lens.reshape(-1)
        ends = torch.where(
            (decode_lens.to(torch.int64) == 1),
            flat.clamp_min(0),
            torch.zeros_like(flat),
        )
    elif seq_lens.dim() == 1 and seq_lens.shape[0] == num_rows:
        # Flattened per-token: each row IS one query token (the
        # flatten path's builder-guaranteed contract: one decode_len
        # per row, B == num_rows). The indices/varlen layouts are
        # rejected by the CALLER via decode_metadata.indices — numel
        # equality alone is NOT a semantic match, and this helper
        # deliberately does no tensor-value probing to distinguish
        # them.
        if num_rows != B:
            raise ValueError(
                f"1-D decode seq_lens requires one decode_len per row "
                f"(B={B} != num_rows={num_rows})"
            )
        flat = seq_lens
        ends = flat.clamp_min(0)
    else:
        raise ValueError(
            f"unsupported decode seq_lens layout {tuple(seq_lens.shape)} "
            f"for {num_rows} logits rows; refusing to guess an alignment"
        )
    if ends.shape[0] != num_rows:
        raise ValueError(
            f"decode bounds have {ends.shape[0]} entries but the logits "
            f"have {num_rows} rows"
        )
    starts = torch.zeros_like(ends)
    return starts, ends


def select_local_pool_candidates(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministically select this rank's local top-k pool ids.

    The DCP merge needs each rank's local candidates under the SAME
    tie discipline as the global merge (score descending, ties by id
    ascending): the native ``top_k_per_row_prefill`` makes no
    documented stability guarantee for equal scores, and a local K
    boundary that drops a SMALLER id in favor of a larger one cannot
    be repaired by the later stable merge (the dropped id never
    reaches it). The same rank-local -> global id mapping is monotone
    in the local id, so selecting ties by LOCAL id ascending is
    exactly the global tie rule restricted to this rank.

    Candidates are the window-relative ids ``column - row_start``;
    validity is the row-bound + non-NaN rule of
    :func:`pack_local_pool_candidates`; ``logits`` is never modified;
    rows shorter than k in valid candidates pad with (-1, -inf). No
    host syncs, no per-row Python.

    Returns:
        (local_pool_ids, scores): int64 [rows, k] and fp32 [rows, k]
        — the inputs to the adapter's pack step.
    """
    if type(k) is not int or k < 1:
        raise ValueError(f"k must be a positive exact int, got {k!r}")
    if logits.dtype is not torch.float32:
        raise ValueError(f"logits must be fp32, got {logits.dtype}")
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2-D [rows, cols], got {logits.ndim}-D")
    if row_starts.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"row_starts must be int32 or int64, got {row_starts.dtype}")
    if row_ends.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"row_ends must be int32 or int64, got {row_ends.dtype}")
    if row_starts.ndim != 1 or row_ends.ndim != 1:
        raise ValueError(
            f"row_starts/row_ends must be 1-D [rows], got "
            f"{row_starts.ndim}-D and {row_ends.ndim}-D"
        )
    rows, cols = logits.shape
    if row_starts.shape[0] != rows or row_ends.shape[0] != rows:
        raise ValueError(
            "row_starts/row_ends must have one entry per logits row, got "
            f"{row_starts.shape[0]}/{row_ends.shape[0]} vs {rows}"
        )
    if row_starts.device != logits.device or row_ends.device != logits.device:
        raise ValueError("row_starts/row_ends must share the logits device")

    starts = row_starts.to(torch.int64)
    ends = row_ends.to(torch.int64)
    valid_rows = (starts >= 0) & (ends >= starts) & (ends <= cols)
    safe_starts = torch.where(valid_rows, starts, torch.zeros_like(starts))
    safe_ends = torch.where(valid_rows, ends, torch.zeros_like(ends))
    available = safe_ends - safe_starts

    # Candidate ids: the FULL window per row, as a [rows, cols]
    # relative-id grid; lanes outside the window are invalid. This is
    # O(rows*cols) memory for the id grid only (int64) — the score
    # tensor is reused by reference.
    ids_grid = torch.arange(cols, dtype=torch.int64, device=logits.device).unsqueeze(
        0
    ).expand(rows, cols) - safe_starts.unsqueeze(-1)
    in_window = (ids_grid >= 0) & (ids_grid < available.unsqueeze(-1))
    valid = in_window & valid_rows.unsqueeze(-1) & ~torch.isnan(logits)

    out_ids = torch.full((rows, k), -1, dtype=torch.int64, device=logits.device)
    out_scores = torch.full(
        (rows, k), float("-inf"), dtype=torch.float32, device=logits.device
    )
    if rows == 0 or cols == 0:
        return out_ids, out_scores

    # The SAME stable three-sort order as stable_global_pool_topk,
    # on (valid-first, score desc, local id asc).
    _, by_id = torch.sort(ids_grid, dim=-1, stable=True)
    ids_id = ids_grid.gather(-1, by_id)
    scores_id = logits.gather(-1, by_id)
    valid_id = valid.gather(-1, by_id)
    _, by_neg = torch.sort(
        torch.where(
            torch.isnan(scores_id),
            torch.zeros_like(scores_id),
            -scores_id,
        ),
        dim=-1,
        stable=True,
    )
    ids_ns = ids_id.gather(-1, by_neg)
    scores_ns = scores_id.gather(-1, by_neg)
    valid_ns = valid_id.gather(-1, by_neg)
    _, by_valid = torch.sort(valid_ns, dim=-1, descending=True, stable=True)
    ordered_ids = ids_ns.gather(-1, by_valid)
    ordered_scores = scores_ns.gather(-1, by_valid)
    ordered_valid = valid_ns.gather(-1, by_valid)

    take = min(k, cols)
    sel_ids = ordered_ids[:, :take]
    sel_scores = ordered_scores[:, :take]
    sel_valid = ordered_valid[:, :take]
    out_ids[:, :take] = torch.where(sel_valid, sel_ids, torch.full_like(sel_ids, -1))
    out_scores[:, :take] = torch.where(
        sel_valid, sel_scores, torch.full_like(sel_scores, float("-inf"))
    )
    return out_ids, out_scores


def select_global_pool_topk_dcp(
    logits: torch.Tensor,
    local_pool_ids: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    pool_interleave: int,
    group: _AllGatherGroup,
    k: int,
    *,
    phase: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack, all-gather, and stably select the global top-k pool ids.

    This rank packs its local candidates with
    :func:`pack_local_pool_candidates` (taking W and rank from
    ``group.world_size`` / ``group.rank_in_group``; the local
    candidate column count K is inferred from ``local_pool_ids`` —
    do not pass W/rank anywhere else), gathers every rank's
    candidates, and selects with :func:`stable_global_pool_topk`.

    Collective order (fixed): scores are all-gathered FIRST, then
    ids — two separate ``group.all_gather(x, dim=1)`` calls keeping
    fp32/int64 dtypes (ids are never funneled through an int32 or
    float buffer). Both collectives are ALWAYS issued for world > 1,
    even when this rank's candidates are all invalid or its logits
    row is empty: participation must not depend on local validity.
    The gathered shapes are validated only AFTER both collectives
    return, so a local structural anomaly never makes this rank skip
    the ids gather its peers are entering (if a collective itself
    raises, the exception propagates — no retry, no fabricated
    success).

    Caller preconditions (this layer cannot verify them locally):
    every rank passes the same ``rows``, the same K, the same
    ``pool_interleave``/``k``, and the SAME query-row order; a rank
    with zero rows may short-circuit to the empty output without any
    collective ONLY under a scheduling guarantee that ALL ranks have
    zero rows this step (misuse on a mixed batch would deadlock the
    peers' collectives). K == 0 is rejected before any collective,
    and ``K >= k`` is enforced as a sufficiency precondition (a
    truncated local top-K cannot guarantee a correct global top-k).
    Full shape/dtype/device/scalar validation runs BEFORE the
    rows == 0 short-circuit — bad parameters on an empty batch are
    still rejected.

    W == 1 never calls the group (``all_gather`` would bypass
    anyway); the selection contract is identical.

    Returns:
        (global_pool_ids, scores): int64 [rows, k] and fp32
        [rows, k] on the logits' device.

    Raises:
        ValueError: on invalid group scalars, K == 0, or a
            post-gather shape/dtype/device mismatch.
    """
    world = getattr(group, "world_size", None)
    rank = getattr(group, "rank_in_group", None)
    if type(world) is not int or world < 1:
        raise ValueError(
            f"group.world_size must be a positive exact int, got {world!r}"
        )
    if type(rank) is not int or not 0 <= rank < world:
        raise ValueError(
            f"group.rank_in_group must be an exact int in [0, {world}), got {rank!r}"
        )
    if local_pool_ids.ndim != 2:
        raise ValueError(
            f"local_pool_ids must be 2-D [rows, K], got {local_pool_ids.ndim}-D"
        )
    rows, local_k = local_pool_ids.shape
    if local_k < 1:
        # K == 0 would make both collectives and the selection
        # degenerate; reject BEFORE any collective so no peer blocks.
        raise ValueError(
            "local_pool_ids must have at least one candidate column, got K=0"
        )
    if type(k) is not int or k < 1:
        raise ValueError(f"k must be a positive exact int, got {k!r}")
    if phase not in ("prefill", "decode", None):
        # The diagnostic phase channel carries ONLY the fixed
        # prefill/decode/None vocabulary — no free-form text.
        raise ValueError(f"phase must be 'prefill', 'decode', or None — got {phase!r}")
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2-D [rows, cols], got {logits.ndim}-D")
    if logits.dtype is not torch.float32:
        raise ValueError(f"logits must be fp32, got {logits.dtype}")
    if local_pool_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            f"local_pool_ids must be int32 or int64, got {local_pool_ids.dtype}"
        )
    if logits.shape[0] != rows:
        raise ValueError(
            f"logits rows must match local_pool_ids rows, got "
            f"{logits.shape[0]} vs {rows}"
        )
    if row_starts.ndim != 1 or row_ends.ndim != 1:
        raise ValueError(
            f"row_starts/row_ends must be 1-D [rows], got "
            f"{row_starts.ndim}-D and {row_ends.ndim}-D"
        )
    if row_starts.shape[0] != rows or row_ends.shape[0] != rows:
        raise ValueError(
            "row_starts/row_ends must have one entry per row, got "
            f"{row_starts.shape[0]}/{row_ends.shape[0]} vs {rows}"
        )
    if (
        row_starts.device != logits.device
        or row_ends.device != logits.device
        or local_pool_ids.device != logits.device
    ):
        raise ValueError("all inputs must share the logits device")
    if local_k < k:
        # Sufficiency: a rank's local top-K is only a valid global
        # top-k candidate set when it already holds k entries — with
        # K < k the local selection itself truncated, so the merged
        # result could miss globally-ranked candidates. Callers must
        # size the local candidate column count >= k (typically
        # k == K exactly).
        raise ValueError(
            f"local candidate columns K={local_k} < k={k}: the local "
            f"top-k is truncated and cannot guarantee a correct global "
            f"top-k; pass K >= k"
        )
    if rows == 0:
        # ONLY valid under the documented all-ranks-empty scheduling
        # guarantee; no collective is issued (see preconditions).
        return (
            torch.empty((0, k), dtype=torch.int64, device=logits.device),
            torch.empty((0, k), dtype=torch.float32, device=logits.device),
        )

    local_scores, local_ids = pack_local_pool_candidates(
        logits,
        local_pool_ids,
        row_starts,
        row_ends,
        world,
        rank,
        pool_interleave,
    )

    # Trial-only pack observation (Boundary B; default OFF — the
    # disabled path is a single None-check with zero tensor access).
    # The pre-gather snapshot NEVER raises and NEVER emits (collective
    # liveness). The pending capture belongs to THIS invocation (the
    # epoch token): if the gathers / structural validation / select /
    # the scalar success observer raise, the except clause ABORTS the
    # pending capture (no receipt, no retained raw refs) and the
    # underlying error propagates unchanged. The receipt is emitted
    # only at the post-success point below, by the owning epoch.
    try:
        if world == 1:
            gathered_scores = local_scores
            gathered_ids = local_ids
        else:
            # Scores first, then ids — both unconditionally; dtype is
            # preserved per tensor (fp32 / int64).
            gathered_scores = group.all_gather(local_scores, dim=1)
            gathered_ids = group.all_gather(local_ids, dim=1)
        # Structural validation only AFTER both collectives completed.
        if not isinstance(gathered_scores, torch.Tensor) or not isinstance(
            gathered_ids, torch.Tensor
        ):
            raise ValueError(
                "group.all_gather must return torch.Tensors, got "
                f"{type(gathered_scores).__name__} and "
                f"{type(gathered_ids).__name__}"
            )
        if gathered_scores.dtype is not torch.float32:
            raise ValueError(
                f"gathered scores must stay fp32, got {gathered_scores.dtype}"
            )
        if gathered_ids.dtype is not torch.int64:
            raise ValueError(f"gathered ids must stay int64, got {gathered_ids.dtype}")
        if gathered_scores.device != logits.device:
            raise ValueError(
                "gathered scores device mismatch: "
                f"{gathered_scores.device} vs {logits.device}"
            )
        if gathered_ids.device != logits.device:
            raise ValueError(
                f"gathered ids device mismatch: {gathered_ids.device} vs "
                f"{logits.device}"
            )
        expected = (rows, world * local_k)
        if gathered_scores.shape != expected:
            raise ValueError(
                f"gathered scores shape {tuple(gathered_scores.shape)} != "
                f"expected {expected} (rows, world*K)"
            )
        if gathered_ids.shape != expected:
            raise ValueError(
                f"gathered ids shape {tuple(gathered_ids.shape)} != "
                f"expected {expected} (rows, world*K)"
            )
        out_ids, out_scores = stable_global_pool_topk(gathered_scores, gathered_ids, k)
    except BaseException:
        # The invocation FAILED (a gather, the structural validation,
        # or the select raised): re-raise the underlying error
        # UNCHANGED (never swallowed, never altered).
        raise
    return out_ids, out_scores
