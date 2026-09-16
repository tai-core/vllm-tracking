# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Legacy Docker builder portability contract for the frozen Dockerfile.

- No RUN heredocs (``RUN ... <<``) and no BuildKit-only features
  (--mount, heredoc markers) anywhere.
- Prebuild logic is invoked as plain ``RUN python <script>`` where the
  scripts are TRACKED in the frozen archive at the exact paths the
  Dockerfile references.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = ROOT / "docker" / "public-build" / "Dockerfile.glm53-v029"
SCRIPTS = {
    "dist_state": ROOT / "docker" / "public-build" / "scripts" / "prebuild_dist_state.py",
    "dep_check": ROOT / "docker" / "public-build" / "scripts" / "prebuild_dep_check.py",
}


def test_no_heredoc_or_buildkit_only_syntax():
    src = DOCKERFILE.read_text()
    for line in src.splitlines():
        if line.startswith("RUN"):
            assert "<<" not in line, line
            assert "--mount" not in line, line
    # no COPY --from= or other BuildKit-only flags
    for line in src.splitlines():
        if line.startswith("COPY"):
            assert "--from" not in line, line


def test_prebuild_scripts_tracked_and_invoked():
    src = DOCKERFILE.read_text()
    invocations = re.findall(r"^RUN python (\S+)$", src, re.M)
    assert any(s.endswith("prebuild_dist_state.py") for s in invocations)
    assert any(s.endswith("prebuild_dep_check.py") for s in invocations)
    for script in SCRIPTS.values():
        assert script.exists(), script


def test_prebuild_scripts_reference_paths_match_dockerfile():
    src = DOCKERFILE.read_text()
    # the Dockerfile invokes scripts at their unpacked archive paths
    assert "/workspace/vllm/docker/public-build/scripts/prebuild_dist_state.py" in src
    assert "/workspace/vllm/docker/public-build/scripts/prebuild_dep_check.py" in src
