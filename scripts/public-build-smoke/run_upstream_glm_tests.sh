#!/usr/bin/env bash
# Codex entrypoint: run the upstream GLM CPU test suites that shipped with
# the transplant (no weights needed). NOTE: on the Mac dev host the local
# pytest run of this suite hit a native teardown crash (exit 139); the
# interrupted local sweep was INCOMPLETE and the crash attribution remains
# UNVERIFIED (host limitation vs source defect). Run here and report the
# actual result; if a native crash reproduces, report it as a concrete
# finding with the torch/triton versions — do not assume either cause.
set -uo pipefail
cd /workspace/vllm
python -m pytest tests/transformers_utils/processors/test_glm5next.py \
    -q -p no:cacheprovider --no-header
