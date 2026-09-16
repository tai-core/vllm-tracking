# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Narrow build-contract test: export_source.sh produces a complete,
docker-build-ready context from the FROZEN commit.

Run:  bash docker/public-build/export_source.sh <tmpdir>   (test fixture)
      <venv>/python -m pytest tests/public_build/test_export_contract.py

Checks (mirrors the review requirements):
1. context root: Dockerfile + vllm.tar + vllm-plugin-das.tar + MANIFEST
2. MANIFEST sha256s match archives AND the Dockerfile
3. Dockerfile bytes == git show <frozen-commit> (not the working tree);
   frozen commit == HEAD
4. every COPY source in the Dockerfile exists in the context
5. archives: no .git directory component, no venv/artifacts, correct
   prefixes, smoke entrypoints tracked
6. ACTUAL unpack (tar, as the Dockerfile RUN does): image paths resolve,
   CMD target exists after unpack
7. truthful version + --no-deps preserved in the frozen Dockerfile
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tarfile
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXPORT = ROOT / "docker" / "public-build" / "export_source.sh"


@pytest.fixture(scope="module")
def ctx(tmp_path_factory):
    out = tmp_path_factory.mktemp("glm53-ctx")
    r = subprocess.run(["bash", str(EXPORT), str(out)],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    return out


def _sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _has_git_dir(n):
    return any(part == ".git" for part in n.split("/"))


def test_context_root_complete(ctx):
    for f in ("Dockerfile", "vllm.tar", "vllm-plugin-das.tar", "MANIFEST.json"):
        assert (ctx / f).exists(), f


def test_manifest_hashes_match(ctx):
    m = json.loads((ctx / "MANIFEST.json").read_text())
    for k in ("vllm", "vllm-plugin-das"):
        assert _sha256(ctx / m[k]["archive"]) == m[k]["archive_sha256"]
    assert _sha256(ctx / "Dockerfile") == m["Dockerfile"]["sha256"]


def test_dockerfile_from_frozen_commit(ctx):
    m = json.loads((ctx / "MANIFEST.json").read_text())
    r = subprocess.run(
        ["git", "show",
         f"{m['vllm']['commit']}:docker/public-build/Dockerfile.glm53-v029"],
        capture_output=True, cwd=str(ROOT),
    )
    assert r.returncode == 0
    assert hashlib.sha256(r.stdout).hexdigest() == m["Dockerfile"]["sha256"]
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True, cwd=str(ROOT))
    assert head.stdout.strip() == m["vllm"]["commit"]


def test_copy_sources_exist(ctx):
    dockerfile = (ctx / "Dockerfile").read_text()
    for src in re.findall(r"^COPY (\S+)", dockerfile, re.M):
        assert (ctx / src).exists(), src


def test_archives_clean_and_prefixed(ctx):
    for arc, prefix in (("vllm.tar", "vllm/"),
                        ("vllm-plugin-das.tar", "vllm-plugin-das/")):
        with tarfile.open(ctx / arc) as t:
            names = t.getnames()
        assert not any(_has_git_dir(n) for n in names)
        assert all("venv" not in n for n in names)
        assert not any(n.startswith("artifacts/") for n in names)
        assert all(n == prefix.rstrip("/") or n.startswith(prefix) for n in names)
    with tarfile.open(ctx / "vllm.tar") as t:
        names = t.getnames()
    for need in ("vllm/scripts/public-build-smoke/smoke_import_and_kernels.py",
                 "vllm/scripts/public-build-smoke/run_smoke.sh",
                 "vllm/scripts/public-build-smoke/run_upstream_glm_tests.sh"):
        assert need in names, need


def test_actual_unpack_resolves_image_paths(ctx, tmp_path):
    for arc in ("vllm.tar", "vllm-plugin-das.tar"):
        r = subprocess.run(["tar", "-xf", str(ctx / arc), "-C", str(tmp_path)])
        assert r.returncode == 0
    dockerfile = (ctx / "Dockerfile").read_text()
    cmd = re.search(r'CMD \["bash", "([^"]+)"', dockerfile).group(1)
    rel = tmp_path / cmd.replace("/workspace/", "")
    assert rel.exists(), cmd
    assert (tmp_path / "vllm-plugin-das" / "setup.py").exists()


def test_truthful_version_and_no_deps(ctx):
    dockerfile = (ctx / "Dockerfile").read_text()
    assert "0.29.0.post1+glm53hcu" in dockerfile
    assert "--no-deps" in dockerfile
    assert "0.28.1rc0.dev" not in dockerfile


def test_prebuild_scripts_in_archive(ctx):
    with tarfile.open(ctx / "vllm.tar") as t:
        names = t.getnames()
    for need in ("vllm/docker/public-build/scripts/prebuild_dist_state.py",
                 "vllm/docker/public-build/scripts/prebuild_dep_check.py"):
        assert need in names, need


def test_exported_dockerfile_legacy_builder_compatible(ctx):
    src = (ctx / "Dockerfile").read_text()
    for line in src.splitlines():
        if line.startswith("RUN"):
            assert "<<" not in line, line
            assert "--mount" not in line, line
        if line.startswith("COPY"):
            assert "--from" not in line, line
    assert "RUN python /workspace/vllm/docker/public-build/scripts/prebuild_dist_state.py" in src
    assert "RUN python /workspace/vllm/docker/public-build/scripts/prebuild_dep_check.py" in src
