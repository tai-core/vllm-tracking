# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

CHECKER = Path(__file__).resolve().parents[2] / "tools/check_backports.py"


class CheckBackportsTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self.git("init", "-q")
        self.git("config", "user.email", "test@example.invalid")
        self.git("config", "user.name", "Backport test")
        self.git("config", "commit.gpgsign", "false")
        self.base = self.commit("base")
        self.tip = self.commit("tip")
        self.manifest = self.root / "manifest.json"

    def git(self, *args, repo=None):
        return subprocess.run(
            ["git", "-C", str(repo or self.repo), *args],
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()

    def commit(self, text):
        (self.repo / "source.txt").write_text(text)
        self.git("add", "source.txt")
        self.git("commit", "-qm", text)
        return self.git("rev-parse", "HEAD")

    def run_checker(self, *shas, target=None, repo=None):
        self.manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "upstream_commits": [{"sha": sha} for sha in shas],
                    "local_adaptations": [{"id": "hcu-restore"}],
                }
            )
        )
        process = subprocess.run(
            [
                sys.executable,
                str(CHECKER),
                "--manifest",
                str(self.manifest),
                "--repo",
                str(repo or self.repo),
                "--target-ref",
                target or self.tip,
                "--json",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        return process.returncode, json.loads(process.stdout)

    def test_ancestor_is_included_but_local_adapter_still_requires_review(self):
        code, report = self.run_checker(self.base)
        self.assertEqual(code, 0)
        self.assertEqual(report["upstream_commits"][0]["status"], "INCLUDED")
        self.assertEqual(report["local_adaptations"][0]["status"], "MANUAL_REVIEW")
        self.assertEqual(report["patch_equivalence"], "NOT_CHECKED")
        self.assertEqual(report["target_sha"], self.tip)

    def test_known_commit_outside_target_history_requires_review(self):
        code, report = self.run_checker(self.tip, target=self.base)
        self.assertEqual(code, 0)
        self.assertEqual(report["upstream_commits"][0]["status"], "REQUIRED")

    def test_identical_tree_rewrite_is_not_claimed_as_patch_equivalence(self):
        self.git("checkout", "-q", "--detach", self.base)
        (self.repo / "source.txt").write_text("tip")
        self.git("add", "source.txt")
        self.git("commit", "-qm", "rewritten commit identity")
        rewritten = self.git("rev-parse", "HEAD")
        self.assertEqual(
            self.git("rev-parse", f"{rewritten}^{{tree}}"),
            self.git("rev-parse", f"{self.tip}^{{tree}}"),
        )
        code, report = self.run_checker(self.tip, target=rewritten)
        self.assertEqual(code, 0)
        self.assertEqual(report["upstream_commits"][0]["status"], "REQUIRED")
        self.assertEqual(report["patch_equivalence"], "NOT_CHECKED")

    def test_missing_commit_is_unknown_and_exits_nonzero(self):
        code, report = self.run_checker("0" * 40)
        self.assertEqual(code, 2)
        self.assertEqual(report["upstream_commits"][0]["status"], "UNKNOWN")

    def test_invalid_target_is_a_structured_error(self):
        code, report = self.run_checker(self.base, target="missing-target")
        self.assertEqual(code, 2)
        self.assertEqual(report["status"], "ERROR")

    def test_shallow_history_does_not_turn_incomplete_ancestry_into_required(self):
        clone = self.root / "shallow"
        self.git("clone", "-q", "--depth=1", self.repo.as_uri(), str(clone))
        self.git("fetch", "-q", "--depth=1", "origin", self.base, repo=clone)
        code, report = self.run_checker(self.tip, self.base, repo=clone)
        self.assertEqual(code, 2)
        self.assertTrue(report["shallow_repository"])
        self.assertEqual(
            [item["status"] for item in report["upstream_commits"]],
            ["INCLUDED", "UNKNOWN"],
        )

    def test_checker_leaves_source_index_refs_and_objects_unchanged(self):
        def snapshot():
            return {
                str(path.relative_to(self.repo)): (
                    path.read_bytes(),
                    path.stat().st_mtime_ns,
                )
                for path in self.repo.rglob("*")
                if path.is_file()
            }

        before = snapshot()
        code, _ = self.run_checker(self.base, self.tip, "0" * 40)
        self.assertEqual(code, 2)
        self.assertEqual(snapshot(), before)

    def test_duplicate_source_identity_is_rejected(self):
        code, report = self.run_checker(self.base, self.base)
        self.assertEqual(code, 2)
        self.assertEqual(report["status"], "ERROR")


if __name__ == "__main__":
    unittest.main()
