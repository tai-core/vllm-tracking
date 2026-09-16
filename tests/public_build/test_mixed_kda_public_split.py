# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Mixed-KDA dispatch/state-ownership tests for the PUBLIC glm5next kda.

The REAL ``Glm5NextLinearAttention._forward`` is extracted verbatim from
``vllm/models/glm5next/nvidia/kda.py`` (AST slice, source of truth) and
executed against RECORDING kernel stubs — mixed non-spec batches must
dispatch decode rows to fused_recurrent_kda and prefill rows to
chunk_kda_with_fused_gate with rebased cu_seqlens, suffix-only
gather/scatter, in-place decode output rows, and no overwritten decode
states. Pure prefill / pure decode / spec paths unchanged. CPU-only
synthetic tensors; NO GPU, NO runtime-parity claim.
"""

from __future__ import annotations

import ast
import sys
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
KDA = ROOT / "vllm" / "models" / "glm5next" / "nvidia" / "kda.py"


# ---------------------------------------------------------------------------
# extract the REAL _forward source
# ---------------------------------------------------------------------------

def _extract_forward_src() -> tuple[list[str], str]:
    """Return (signature params, body source) of the REAL _forward."""
    tree = ast.parse(KDA.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Glm5NextLinearAttention":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "_forward":
                    src = ast.get_source_segment(KDA.read_text(), item)
                    lines = src.splitlines()
                    # end of signature = first line ending in ':'
                    sig_end = next(i for i, l in enumerate(lines)
                                   if l.rstrip().endswith(":"))
                    params = [a.arg for a in item.args.args]
                    body = "\n".join(lines[sig_end + 1:])
                    return params, body
    raise AssertionError("Glm5NextLinearAttention._forward not found")


FORWARD_PARAMS, FORWARD_BODY = _extract_forward_src()


# ---------------------------------------------------------------------------
# recording stubs
# ---------------------------------------------------------------------------

class _Calls:
    def __init__(self):
        self.recurrent: list[dict] = []
        self.chunk: list[dict] = []
        self.gathers: list[dict] = []
        self.scatters: list[dict] = []

    def reset(self):
        self.recurrent.clear()
        self.chunk.clear()
        self.gathers.clear()
        self.scatters.clear()


CALLS = _Calls()


def _fake_recurrent(q, k, v, g, beta, **kw):
    out = kw.get("out")
    if out is None:
        out = q.new_zeros(q.shape[0], q.shape[1], q.shape[2], v.shape[-1])
    T = out.shape[1]
    for t in range(T):
        out[0, t] = -(t + 1)  # decode marker: negative row ids
    CALLS.recurrent.append({
        "q_rows": int(q.shape[1]),
        "cu": kw.get("cu_seqlens").tolist(),
        "ssm": kw.get("ssm_state_indices").tolist()
        if kw.get("ssm_state_indices") is not None else None,
        "out_is_alias": kw.get("out") is not None,
        "sigmoid_beta": kw.get("sigmoid_beta"),
        "compute_gate": kw.get("compute_gate"),
        "beta_arg_raw": True,  # caller passes raw beta (no pre-sigmoid)
    })
    return out, None


def _fake_chunk(q, k, v, raw_g, beta, **kw):
    out = q.new_zeros(q.shape[0], q.shape[1], q.shape[2], v.shape[-1])
    T = out.shape[1]
    for t in range(T):
        out[0, t] = (t + 1)  # prefill marker: positive row ids
    CALLS.chunk.append({
        "q_rows": int(q.shape[1]),
        "cu": kw.get("cu_seqlens").tolist(),
        "beta_dtype": str(beta.dtype),
        "beta_presigmoided": True,
        "initial_state_gathered": kw.get("initial_state") is not None,
    })
    return out, None


def _fake_gather(recurrent_state, indices, has_initial):
    CALLS.gathers.append({
        "indices": indices.tolist(),
        "has_initial": None if has_initial is None else has_initial.tolist(),
    })
    n = int(indices.shape[0])
    return torch.zeros(
        (n, recurrent_state.shape[1], recurrent_state.shape[2]),
        dtype=recurrent_state.dtype,
    )


def _fake_scatter(recurrent_state, last_state, indices):
    CALLS.scatters.append({"indices": indices.tolist()})


def _fake_conv_fn(qkv, w, b, **kw):
    return qkv


def _fake_conv_update(qkv, state, w, b, **kw):
    return qkv


# ---------------------------------------------------------------------------
# harness: build the namespace + a fake self, run the real _forward
# ---------------------------------------------------------------------------

def _cast_sigmoid(x):
    return x.float().sigmoid() if x.dtype != torch.float32 else x.sigmoid()


def _make_self(H=2, D=8, conv_first=True, prefix="test.kda"):
    s = types.SimpleNamespace()
    s.local_num_heads = H
    s.head_dim = D
    # qkv row width is 3 * local_projection_size (q|k|v each one wide)
    s.local_projection_size = H * D
    s.A_log = torch.zeros(1)
    s.dt_bias = torch.zeros(1)
    s.kda_safe_gate = True
    s.kda_lower_bound = None
    s.kv_cache = None  # set per-case
    s._conv_state_dim_first = conv_first
    s._merged_conv_weight = None
    s.q_conv1d = types.SimpleNamespace(
        weight=torch.zeros(1), bias=None)
    s.q_conv1d.weight = torch.zeros(3 * H * D, 1, 3)
    s.q_conv1d.bias = torch.zeros(3 * H * D)
    s.k_conv1d = types.SimpleNamespace(weight=torch.zeros(3 * H * D, 1, 3))
    s.v_conv1d = types.SimpleNamespace(weight=torch.zeros(3 * H * D, 1, 3))
    s.prefix = prefix
    return s


class _Meta:
    """GDNAttentionMetadata stand-in (isinstance target in the harness)."""


def _make_meta(num_decodes, prefill_lens, num_actual_tokens,
               decode_tokens=None):
    """GDNAttentionMetadata-shaped object. Non-spec token stream is
    decode-first (decodes then prefills), matching the runner layout the
    split relies on."""
    m = _Meta()
    m.num_actual_tokens = num_actual_tokens
    m.num_decodes = num_decodes
    m.num_prefills = len(prefill_lens)
    m.num_decode_tokens = (
        decode_tokens if decode_tokens is not None else num_decodes)
    m.has_initial_state = torch.zeros(num_decodes + len(prefill_lens),
                                      dtype=torch.bool)
    m.spec_sequence_masks = None
    m.spec_query_start_loc = None
    m.spec_state_indices_tensor = None
    m.spec_token_indx = None
    m.non_spec_token_indx = None
    m.num_accepted_tokens = None
    m.num_spec_decodes = 0
    # decode-first layout: [0, d_tokens, d_tokens+p1, ...]
    # each decode request contributes ONE token; each prefill its length
    qsl = [0]
    for _ in range(num_decodes):
        qsl.append(qsl[-1] + 1)
    for L in prefill_lens:
        qsl.append(qsl[-1] + L)
    assert qsl[-1] == num_actual_tokens, (
        f"qsl {qsl} must cover num_actual_tokens={num_actual_tokens}")
    m.non_spec_query_start_loc = torch.tensor(qsl, dtype=torch.int32)
    n_req = num_decodes + len(prefill_lens)
    m.non_spec_state_indices_tensor = torch.arange(n_req, dtype=torch.int32)
    return m


def _run(self, meta, H=2, D=8):
    ntok = meta.num_actual_tokens
    P = 3 * H * D  # qkv width = 3 * local_projection_size
    qkv = torch.randn(ntok, P)
    g1 = torch.randn(1, ntok, H, D)
    beta = torch.randn(1, ntok, H, 1)
    core = torch.zeros(1, ntok, H, D)
    state = torch.zeros(4, H, D)  # paged cache: 4 slots
    state[3] = float("nan")  # unselected-slot sentinel
    self.kv_cache = (torch.zeros(4, P, 2), state)

    ctx = types.SimpleNamespace(attn_metadata={"test.kda": meta})

    ns = {
        "torch": torch,
        "get_forward_context": lambda: ctx,
        "GDNAttentionMetadata": object,  # isinstance target
        "causal_conv1d_fn": _fake_conv_fn,
        "causal_conv1d_update": _fake_conv_update,
        "fused_recurrent_kda": _fake_recurrent,
        "chunk_kda_with_fused_gate": _fake_chunk,
        "gather_initial_states": _fake_gather,
        "scatter_states": _fake_scatter,
        "_cast_sigmoid": _cast_sigmoid,
    }
    # GDNAttentionMetadata isinstance must pass: use the SAME module-level
    # _Meta that _make_meta constructs instances of
    ns["GDNAttentionMetadata"] = _Meta

    args = ", ".join(FORWARD_PARAMS[1:])  # drop self
    fn_src = f"def _run(self, {args}):\n" + FORWARD_BODY
    exec(compile(fn_src, "<forward>", "exec"), ns)
    ns["_run"](self, qkv, g1, beta, core)
    return core, state


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_mixed_d1_p1_split_dispatch_and_ownership():
    """Mixed D=1 (1 decode token) + P=1 (13-token prefill): recurrent
    handles EXACTLY the decode row; chunk handles the prefill rows with
    rebased cu_seqlens; scatter touches ONLY the prefill slot."""
    CALLS.reset()
    self = _make_self()
    meta = _make_meta(num_decodes=1, prefill_lens=[13],
                      num_actual_tokens=14)
    core, state = _run(self, meta)

    assert len(CALLS.recurrent) == 1, "recurrent must run exactly once"
    r = CALLS.recurrent[0]
    assert r["q_rows"] == 1
    assert r["cu"] == [0, 1]
    assert r["ssm"] == [0]
    assert r["out_is_alias"] is True  # writes into the output buffer
    assert r["sigmoid_beta"] is True and r["compute_gate"] is True

    assert len(CALLS.chunk) == 1, "chunk must run exactly once"
    c = CALLS.chunk[0]
    assert c["q_rows"] == 13
    assert c["cu"] == [0, 13]  # rebased to the suffix token count
    assert c["beta_dtype"] == "torch.float32"  # pre-sigmoided fp32

    # gather/scatter only the prefill suffix slot (index 1)
    assert len(CALLS.gathers) == 1 and CALLS.gathers[0]["indices"] == [1]
    assert len(CALLS.scatters) == 1 and CALLS.scatters[0]["indices"] == [1]

    # output rows: decode row carries the recurrent marker (-1);
    # prefill rows carry the chunk markers (1..13) — in token order
    assert core[0, 0, 0, 0].item() == -1
    for t in range(1, 14):
        assert core[0, t, 0, 0].item() == t

    # the unselected slot sentinel is untouched (no stray scatter)
    assert torch.isnan(state[3]).all()


def test_mixed_d2_p3_multi_request():
    """D=2 + P=3 (prefill lens 3/13/17): recurrent gets both decode rows;
    chunk gets 33 rows; scatter only slots [2,3,4]."""
    CALLS.reset()
    self = _make_self()
    meta = _make_meta(num_decodes=2, prefill_lens=[3, 13, 17],
                      num_actual_tokens=35)
    core, state = _run(self, meta)

    assert len(CALLS.recurrent) == 1
    assert CALLS.recurrent[0]["q_rows"] == 2
    assert CALLS.recurrent[0]["cu"] == [0, 1, 2]
    assert CALLS.recurrent[0]["ssm"] == [0, 1]

    assert len(CALLS.chunk) == 1
    assert CALLS.chunk[0]["q_rows"] == 33
    assert CALLS.chunk[0]["cu"] == [0, 3, 16, 33]
    assert CALLS.scatters[0]["indices"] == [2, 3, 4]
    assert CALLS.gathers[0]["indices"] == [2, 3, 4]

    # token order: rows 0..1 decode markers; rows 2..34 prefill markers
    assert core[0, 0, 0, 0].item() == -1 and core[0, 1, 0, 0].item() == -2
    for t in range(2, 35):
        assert core[0, t, 0, 0].item() == t - 1


def test_pure_prefill_unchanged():
    """Pure prefill (no decodes): the chunk path handles everything —
    same as before the port."""
    CALLS.reset()
    self = _make_self()
    meta = _make_meta(num_decodes=0, prefill_lens=[10],
                      num_actual_tokens=10, decode_tokens=0)
    core, state = _run(self, meta)
    assert len(CALLS.recurrent) == 0
    assert len(CALLS.chunk) == 1
    assert CALLS.chunk[0]["cu"] == [0, 10]
    assert CALLS.scatters[0]["indices"] == [0]


def test_pure_decode_unchanged():
    """Pure decode (no prefills): the recurrent path handles everything —
    same as before the port."""
    CALLS.reset()
    self = _make_self()
    meta = _make_meta(num_decodes=3, prefill_lens=[],
                      num_actual_tokens=3)
    core, state = _run(self, meta)
    assert len(CALLS.chunk) == 0
    assert len(CALLS.recurrent) == 1
    assert CALLS.recurrent[0]["q_rows"] == 3
    assert CALLS.recurrent[0]["ssm"] == [0, 1, 2]


def test_rebased_qsl_uses_slice_token_count_only():
    """The rebased prefill cu_seqlens derive ONLY from the suffix token
    counts (mirror of the GDN builder derivation), not from absolute
    positions: qsl[nd:] - num_decode_tokens."""
    CALLS.reset()
    self = _make_self()
    meta = _make_meta(num_decodes=1, prefill_lens=[5, 7],
                      num_actual_tokens=13)
    _run(self, meta)
    # original qsl = [0,1,6,13]; rebased = [0,5,12]
    assert CALLS.chunk[0]["cu"] == [0, 5, 12]


def test_mixed_batch_does_not_scatter_decode_slots():
    """RED-provable ownership invariant: in a mixed batch the scatter
    indices NEVER include the decode slots (the recurrent kernel owns
    them)."""
    CALLS.reset()
    self = _make_self()
    meta = _make_meta(num_decodes=2, prefill_lens=[4, 6],
                      num_actual_tokens=12)
    _run(self, meta)
    decode_slots = {0, 1}
    scattered = set(CALLS.scatters[0]["indices"])
    assert decode_slots.isdisjoint(scattered), (
        f"decode slots {decode_slots} must never be scattered over; "
        f"scattered={scattered}"
    )
