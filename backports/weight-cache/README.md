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

## Focused validation

The checker is stdlib-only and does not import vLLM or need a GPU. Its tests create
temporary Git repositories and exercise the CLI's observable ancestry, failure,
shallow-history, local-adapter and read-only behavior:

```bash
.venv/bin/python -m unittest discover -s tests/tools -p test_check_backports.py -v
```

These tests validate the maintenance tool only; model/cache acceptance is separate.
