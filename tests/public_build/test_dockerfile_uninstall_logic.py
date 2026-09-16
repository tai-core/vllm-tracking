# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Narrow source test: the prebuild dist-state script's uninstall logic.

The logic now lives in the tracked script
docker/public-build/scripts/prebuild_dist_state.py (legacy Docker
builder compatible — invoked as plain RUN python). Four contract cases
run against the script's main() with controlled importlib.metadata /
subprocess. No vllm import needed.
"""

from __future__ import annotations

import importlib.metadata as md
import subprocess
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "docker" / "public-build" / "scripts" / "prebuild_dist_state.py"


def _ok_dist(name):
    return types.SimpleNamespace(version="0.99", locate_file=lambda n: "/fake")


class UninstallLogicTest(unittest.TestCase):
    def _run(self, dist_fn, run_fn):
        """Run the script's main() with controlled md/subprocess."""
        sys.path.insert(0, str(SCRIPT.parent))
        try:
            import prebuild_dist_state as mod
        finally:
            sys.path.pop(0)
        with mock.patch.object(mod.md, "distribution", side_effect=dist_fn), \
             mock.patch.object(mod.subprocess, "run", side_effect=run_fn):
            try:
                rc = mod.main()
                return rc, None
            except SystemExit as e:
                return None, str(e)

    def test_all_dists_absent_no_removal(self):
        def absent(name):
            raise md.PackageNotFoundError(name)
        rc, _ = self._run(absent, lambda *a, **k: None)
        self.assertEqual(rc, 0)

    def test_uninstall_success(self):
        state = {"gone": False}

        def dist(name):
            if name in ("vllm",) and not state["gone"]:
                return _ok_dist(name)
            raise md.PackageNotFoundError(name)

        def run(*a, **k):
            state["gone"] = True
            return subprocess.CompletedProcess(a, 0, "ok", "")

        rc, _ = self._run(dist, run)
        self.assertEqual(rc, 0)

    def test_nonzero_warning_stderr_still_installed_fails_closed(self):
        def dist(name):
            if name in ("vllm",):
                return _ok_dist(name)
            raise md.PackageNotFoundError(name)

        def run(*a, **k):
            return subprocess.CompletedProcess(a, 3, "", "WARNING: pip problem")

        rc, _ = self._run(dist, run)
        self.assertEqual(rc, 1)  # fail closed; no stderr heuristics

    def test_nonzero_but_verifiably_absent_tolerated(self):
        seq = {"seen": False}

        def dist(name):
            if name in ("vllm", "vllm_hcu") and not seq["seen"]:
                seq["seen"] = True
                return _ok_dist(name)
            raise md.PackageNotFoundError(name)

        def run(*a, **k):
            return subprocess.CompletedProcess(a, 3, "", "WARNING: pip problem")

        rc, _ = self._run(dist, run)
        self.assertEqual(rc, 0)
