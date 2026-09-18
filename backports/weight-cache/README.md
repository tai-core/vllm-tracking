# Weight-cache backport provenance

This directory tracks the upstream feature independently of private branch,
rebase, and squash commit identities. `manifest.json` is ordered by original
upstream SHA and PR, with local compatibility and HCU changes recorded separately.
It is a scope/provenance record, **not evidence that model validation has passed**.
The manifest records actual core conflict decisions; pending compatibility and HCU
items must be updated as implementation and acceptance complete.

## Initial backport

1. Start from the recorded fork base and use `git cherry-pick -x` for each original
   upstream commit in manifest order. Preserve original author and PR provenance.
2. Resolve context against this baseline without copying newer whole files. Record
   each intentional semantic deviation and its reason. Keep local compatibility
   commits separate from the original feature commits; keep platform adaptations
   in the HCU plugin where the interface permits.
3. Update provisional notes with actual changed paths, compatibility decisions,
   and acceptance evidence before publishing. Do not infer FP4, EP, multi-node,
   or MTP support from this TP-only scope.
4. A feature PR may be squash merged. Retain this manifest in the merged tree and
   list the original upstream PRs/SHAs in the PR body and squash message. Local
   development commit hashes are not durable feature identifiers.

## Upgrade check

After making the intended official target and original commit objects available
locally, run the checker from a development environment:

```bash
.venv/bin/python tools/check_backports.py \
  --manifest backports/weight-cache/manifest.json \
  --repo . --target-ref upstream/v0.30.0 --json
```

The example ref must be replaced with an actually fetched official target.
The checker uses only read-only Git commands, disables lazy object fetching,
and ignores replacement objects. It does not fetch, check out, edit, or remove
commits. Compare against the intended **upstream target**, not the local backport
branch: a cherry-pick or squash does not make the original commit an ancestor.

| Result | Meaning and required follow-up |
| --- | --- |
| `INCLUDED` | Original SHA is an ancestor of the target. Do not replay it automatically; confirm the target retains the needed behavior and run regression acceptance. |
| `REQUIRED` | Original SHA exists but is not an ancestor in a complete history. The source feature still needs review; an equivalent squash or rewrite may exist. Inspect differences before any replay. |
| `UNKNOWN` | Object is missing, ancestry failed, or shallow history prevents proving exclusion. Fetch/deepen the relevant public history explicitly, then rerun; do not guess. |
| `MANUAL_REVIEW` | Local compatibility/HCU behavior always needs separate review, even when every core SHA is included. |

In particular, an equivalent squash/rewrite can still report `REQUIRED`: this
means the **original source SHA is absent from ancestry**, not permission to
apply the feature again. A patch and semantic review prevents duplicate backports.

Exit status `0` means all upstream ancestry results are known, including
`REQUIRED`; `2` means an error or at least one `UNKNOWN`. This is an inventory
tool, not a compatibility/release gate. Patch equivalence is **NOT_CHECKED**.
No status authorizes dropping local changes or claiming runtime support.

For an upgrade, first use the new upstream implementation, then reapply only
still-needed, reviewed local adapters. Review platform UUID mapping, quant-method
support and post-load reconstruction, socket/protocol lifecycle, model aliases,
loader fallback behavior, memory admission and MTP draft loading independently.
Require cold load, `fallback=false` cache hit, repeated engine restart with the
daemon alive, output comparison and measured memory before deployment. Keep old
launch scripts/evidence archived and a tested rollback source revision available.

## MTP extension: pinned open upstream feature

The MTP extension backports PR [57312](https://github.com/vllm-project/vllm/pull/57312)
at `13a3b1077cd40659f252b18d8fec7603a1ced916` and
`e5d0102f88f019a921888ec22bbe6a36736e88af`. The PR was **OPEN, not merged** when
reviewed on 2026-09-18. Its first commit's actual parent is
`8538017f4b1d01b543345f5dbe1993cb66c80fa9`; a GitHub API base-branch SHA is not
the patch's parent. Reconcile this pinned feature against the eventual upstream
merge instead of assuming the original SHAs will enter main unchanged.

PR [55468](https://github.com/vllm-project/vllm/pull/55468),
`52dd0d7562adb3c2d556ce6fe7ca7c3226e1976b`, supplies the local-rank/master-port
launcher used by this feature. Its upstream structure is retained; hardware
acceptance here remains **single-node TP8/PP1/DP1/DCP1**, not multi-node or EP.
The absent MRV2 `get_pp_safe_draft_load_config` dependency from PR 54416 is not
imported: the compatibility commit uses the new draft-load routing directly
while preserving the existing v0.29 PP behavior. PR 55337's draft-loader fix is
covered by this feature and must not be applied again.

The launcher receives the same MTP speculative config as the engine and starts
two independent TP daemon groups. Each process owns only its selected model
role: target sockets keep their names; the draft uses `*_draft0.sock`, a distinct
cache-key role, and `--weight-cache-draft-master-port`. A draft group receives
the target `VllmConfig` plus the draft `ModelConfig` separately, matching the
engine's constructor contract. This does not build a second full target model,
and `num_speculative_tokens=4` does not mean four draft model copies.

Under `ipc_cache`, default MTP draft loading automatically selects the draft
role and preserves the target's `zero_copy`/`fallback=false` loader policy.
An explicit `draft_load_config` still wins. Do not point an explicit shared
`socket_path` at both roles; use the common `socket_dir` and role-derived names.
EAGLE/EAGLE3 branches and ownership flags are retained from the complete upstream
feature, but HCU acceptance is MTP-only. All owners and consumers must use this
new protocol; an old target-only owner must not be reused.

Existing HCU fixes remain: spawned ranks load general plugins before model
construction, workspace initialization uses the rank's local device, and device
UUID handling remains plugin-owned. GLM MTP's non-Module router references need
their own plugin post-load restoration; target-model support does not prove that
the draft has the same hook coverage.

The upstream launcher starts both groups concurrently. Before cold loading,
measure combined target/draft GPU and host-memory requirements. The draft loader
filters MTP tensor names while iterating the checkpoint; a prefetch strategy can
still read all checkpoint files even though it builds only draft parameters.
This backport does not promise a cheap first draft load or serialize cold loads.

At the source-freeze commit, 13 non-hardware weight-cache contract tests and the
8 maintenance-checker tests pass. Subsequent node 08 acceptance is **PASS** for
the bounded W8A8 TP8/PP1/DP1/DCP1, 64K/C8, MTP k4/local-argmax, AITER-RNE,
8 GiB KV, zero-copy/fallback-false combination. All 16 owner role constructors,
16 engineering requests, observed engine concurrency 8, and selected target
plus complete draft tensor audits passed. Historical RNE strict numeric
failures and two-color-video answer failures remain explicitly recorded.

Node 08 subsequently passed eight more functional checks after a warm restart,
retained identical selected target/all draft tensors, and reclaimed its resources
back to the initial eight-device VRAM state.

Node 10 deployment is **accepted-with-video-limitation**: the new service remains
running by the user's explicit decision. All 16 owner sockets were present;
RNE module mappings and selection logs were verified for all 8 engine workers.
Its independent C8 gate passed 8/8 with
observed peak 8, and six basic text/image/single-color-video cases passed. The
original red/blue-video question and a separately repeated diagnostic request
produced reasoning only; business retries were configured to zero.
A one-sentence reformulation produced a coherent final answer but omitted blue.
Video known-answer checks therefore remain **FAIL**; there is no aggregate
all-requests or video-quality PASS claim. Historical RNE numeric failures remain.
No Torch mHC control was run, so the video behavior is not attributed to RNE,
MTP or the model itself.

Core Draft PR21 is published. GitHub CI did not start because of an account
payment/spending-limit restriction: **NOT_RUN**, not a code-check PASS.
See [the MTP validation record](mtp-validation-20260918.md) for service identity,
timing definitions and evidence limits. The startup KV estimate is not a
full-length C8 capacity test, and model-loading time is not full startup time.

## Focused validation

The checker is stdlib-only and does not import vLLM or need a GPU. Its tests create
temporary Git repositories and exercise the CLI's observable ancestry, failure,
shallow-history, local-adapter and read-only behavior:

```bash
.venv/bin/python -m unittest discover -s tests/tools -p test_check_backports.py -v
```

These tests validate the maintenance tool only; model/cache acceptance is separate.
