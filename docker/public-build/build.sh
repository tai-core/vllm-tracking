#!/usr/bin/env bash
# Codex build invocation (reviewed route):
#   1. export audited source archives (git archive; no worktree/.git)
#   2. docker build from the export context
#
# Resource bounds (BW1000 node): MAX_JOBS=32 default (bounded ninja
# parallelism); vLLM extension compile 40-90 min; plugin ~5 min; disk
# ~30 GB. HCU_GFX=936 default target.
#
# BASE_IMAGE override: point at an already-audited LOCAL toolchain image
# identity if the public base lacks das components (first route: DTK
# dependency reuse). The default is the public DTK reference PINNED BY
# DIGEST; it is a toolchain base only — NOT assumed to carry the HCU
# bridge or GLM.
set -euo pipefail

BASE_IMAGE="${BASE_IMAGE:-harbor.sourcefind.cn:5443/hcu/admin/base/custom@sha256:636005656b97927f93e04f03258e8380d650448e9a263e54929e281c47315670}"
MAX_JOBS="${MAX_JOBS:-32}"
HCU_GFX="${HCU_GFX:-936}"
# EXPERIMENTAL macro-header seam input (default empty = off). The
# supervisor supplies the vendor SDK header name at node build time;
# no vendor path/value lives in this script.
HCU_MACRO_HEADER="${HCU_MACRO_HEADER:-}"

VLLM_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
CTX="$(mktemp -d /tmp/glm53-build-ctx.XXXXXX)"

bash "$VLLM_DIR/docker/public-build/export_source.sh" "$CTX"

# The exporter places the frozen-commit Dockerfile at the context root;
# pass it explicitly so a stray file can never shadow it.
docker build \
    -f "$CTX/Dockerfile" \
    --build-arg BASE_IMAGE="$BASE_IMAGE" \
    --build-arg MAX_JOBS="$MAX_JOBS" \
    --build-arg HCU_GFX="$HCU_GFX" \
    --build-arg HCU_MACRO_HEADER="$HCU_MACRO_HEADER" \
    -t glm53-v029-public-build:stage1 \
    "$CTX"

echo "image built: glm53-v029-public-build:stage1"
# Device mapping: the exact BW1000 DCU device interface for container
# launch is NOT invented here — Codex preflight determines the correct
# --device / --gpus / launcher contract from the public HCU runtime
# documentation before running the GPU-gated smoke.
echo "smoke (all gates):      docker run --rm <codex-determined-device-contract> glm53-v029-public-build:stage1"
echo "upstream GLM cpu tests: docker run --rm glm53-v029-public-build:stage1 bash /workspace/vllm/scripts/public-build-smoke/run_upstream_glm_tests.sh"
