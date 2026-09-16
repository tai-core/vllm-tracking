#!/usr/bin/env bash
# Source freeze/export for the GLM53 public v0.29 build (audited Git
# archives only — never a whole worktree; no .git/venvs/artifacts).
#
# Produces a docker-build-ready context:
#   <out>/
#     Dockerfile             — from the EXACT frozen Git commit (git show),
#                              never the mutable working tree
#     vllm.tar               — this tree @ the exact commit (git archive)
#     vllm-plugin-das.tar    — bridge @ the exact commit (git archive)
#     MANIFEST.json          — commits, tree hashes, archive + Dockerfile
#                              sha256s, COPY-source completeness check
#
# Usage: bash docker/public-build/export_source.sh <output-dir>
set -euo pipefail

OUT="${1:?usage: export_source.sh <output-dir>}"
VLLM_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
BRIDGE_DIR="${BRIDGE_DIR:-$VLLM_DIR/../v029-bridge}"

VLLM_SHA="$(git -C "$VLLM_DIR" rev-parse HEAD)"
BRIDGE_SHA="$(git -C "$BRIDGE_DIR" rev-parse HEAD)"
VLLM_TREE="$(git -C "$VLLM_DIR" rev-parse HEAD^{tree})"
BRIDGE_TREE="$(git -C "$BRIDGE_DIR" rev-parse HEAD^{tree})"
VLLM_CLEAN="$(git -C "$VLLM_DIR" status --porcelain | wc -l)"
BRIDGE_CLEAN="$(git -C "$BRIDGE_DIR" status --porcelain | wc -l)"
[ "$VLLM_CLEAN" -eq 0 ] || { echo "vllm tree dirty — commit first" >&2; exit 1; }
[ "$BRIDGE_CLEAN" -eq 0 ] || { echo "bridge tree dirty — commit first" >&2; exit 1; }

mkdir -p "$OUT"
# Dockerfile from the FROZEN commit (not the working tree)
git -C "$VLLM_DIR" show "$VLLM_SHA:docker/public-build/Dockerfile.glm53-v029" > "$OUT/Dockerfile"
git -C "$VLLM_DIR" archive --format=tar --prefix=vllm/ -o "$OUT/vllm.tar" "$VLLM_SHA"
git -C "$BRIDGE_DIR" archive --format=tar --prefix=vllm-plugin-das/ -o "$OUT/vllm-plugin-das.tar" "$BRIDGE_SHA"

python3 - "$OUT" "$VLLM_SHA" "$BRIDGE_SHA" "$VLLM_TREE" "$BRIDGE_TREE" <<'PYEOF'
import hashlib
import json
import re
import sys
import tarfile

out, vsha, bsha, vtree, btree = sys.argv[1:6]


def sha256(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


m = {
  "vllm": {
    "role": "official v0.29.0 (98dff2a81d747d1dba01a47f939f48c3526d4206) + minimal GLM PR53906 transplant (5-conflict ledger)",
    "commit": vsha, "tree": vtree,
    "archive": "vllm.tar", "archive_sha256": sha256(f"{out}/vllm.tar"),
    "version_identity": "0.29.0.post1+glm53hcu (truthful; never relabeled)",
  },
  "vllm-plugin-das": {
    "role": "public bridge e1861907 + public-build compatibility-gate adaptation",
    "commit": bsha, "tree": btree,
    "archive": "vllm-plugin-das.tar", "archive_sha256": sha256(f"{out}/vllm-plugin-das.tar"),
    "base_commit": "e1861907949f01b06c4663578c3d0bc88f635953",
  },
  "Dockerfile": {
    "source": "docker/public-build/Dockerfile.glm53-v029 AT the frozen vllm commit (git show; never the working tree)",
    "sha256": sha256(f"{out}/Dockerfile"),
  },
}

# COPY-source completeness: every COPY in the root Dockerfile must exist
# in the exported context (archives keep their vllm//vllm-plugin-das/
# prefixes; the unpack RUN extracts them into the image).
dockerfile = open(f"{out}/Dockerfile").read()
copy_sources = re.findall(r"^COPY (\S+)", dockerfile, re.M)
with tarfile.open(f"{out}/vllm.tar") as t:
    vllm_names = set(t.getnames())
with tarfile.open(f"{out}/vllm-plugin-das.tar") as t:
    bridge_names = set(t.getnames())
ctx_files = {"Dockerfile", "vllm.tar", "vllm-plugin-das.tar",
              "MANIFEST.json"}  # MANIFEST written by this same script
missing = []
for src in copy_sources:
    if src in ctx_files:
        continue  # context-level file (archive/Dockerfile) — present by construction
    # Dockerfile COPYs only context-level files in this design; anything
    # else is a contract violation
    missing.append(src)
if missing:
    raise SystemExit(f"Dockerfile COPY sources not in context: {missing}")
m["copy_source_check"] = {
    "copy_sources": copy_sources,
    "result": "PASS (all COPY sources are context-level archives/Dockerfile)",
}
json.dump(m, open(f"{out}/MANIFEST.json", "w"), indent=1)
print("MANIFEST.json written")
PYEOF
echo "exported: $OUT (vllm $VLLM_SHA / bridge $BRIDGE_SHA; Dockerfile from frozen commit)"
