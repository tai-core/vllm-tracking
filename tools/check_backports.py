# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Report original upstream commit ancestry without changing the repository."""

import argparse
import json
import os
import re
import subprocess
from pathlib import Path


def git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "--no-replace-objects", "-C", str(repo), *args],
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0", "GIT_NO_LAZY_FETCH": "1"},
        text=True,
        capture_output=True,
        check=False,
    )


def read_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text())
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("manifest must use schema_version 1")
    entries = manifest.get("upstream_commits")
    if not isinstance(entries, list) or not entries:
        raise ValueError("upstream_commits must be a nonempty ordered list")
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("upstream commit entries must be objects")
        sha = entry.get("sha", "")
        if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{40}", sha):
            raise ValueError("upstream commits require full lowercase SHA-1 hashes")
        if sha in seen:
            raise ValueError(f"duplicate upstream commit: {sha}")
        seen.add(sha)
    adaptations = manifest.get("local_adaptations", [])
    if not isinstance(adaptations, list) or any(
        not isinstance(item, dict) or not isinstance(item.get("id"), str)
        for item in adaptations
    ):
        raise ValueError("local_adaptations must be objects with stable string IDs")
    return manifest


def assess(manifest: dict, repo: Path, target_ref: str) -> dict:
    """Resolve an upstream target and report ancestry, not patch equivalence."""
    target = git(
        repo, "rev-parse", "--verify", "--end-of-options", f"{target_ref}^{{commit}}"
    )
    if target.returncode:
        raise ValueError(f"target is not an available commit: {target_ref}")
    target_sha = target.stdout.strip()
    shallow = git(repo, "rev-parse", "--is-shallow-repository")
    if shallow.returncode or shallow.stdout.strip() not in ("true", "false"):
        raise ValueError("cannot determine whether repository history is shallow")
    is_shallow = shallow.stdout.strip() == "true"
    results = []
    for entry in manifest["upstream_commits"]:
        sha = entry["sha"]
        exists = git(repo, "cat-file", "-e", f"{sha}^{{commit}}")
        if exists.returncode:
            status, reason = "UNKNOWN", "Upstream commit object is unavailable."
        else:
            ancestor = git(repo, "merge-base", "--is-ancestor", sha, target_sha)
            if ancestor.returncode == 0:
                status, reason = "INCLUDED", "Original commit is an ancestor of target."
            elif ancestor.returncode != 1:
                status, reason = "UNKNOWN", "Git could not complete the ancestry check."
            elif is_shallow:
                status, reason = "UNKNOWN", "Shallow history cannot prove exclusion."
            else:
                status = "REQUIRED"
                reason = "Original SHA is not a target ancestor; do not auto-apply."
        results.append({**entry, "status": status, "reason": reason})
    adaptations = [
        {
            **item,
            "status": "MANUAL_REVIEW",
            "reason": "Revalidate local behavior separately from upstream ancestry.",
        }
        for item in manifest.get("local_adaptations", [])
    ]
    return {
        "feature": manifest.get("feature"),
        "target_ref": target_ref,
        "target_sha": target_sha,
        "shallow_repository": is_shallow,
        "patch_equivalence": "NOT_CHECKED",
        "upstream_commits": results,
        "local_adaptations": adaptations,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=(
            Path(__file__).resolve().parents[1] / "backports/weight-cache/manifest.json"
        ),
    )
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--target-ref", required=True)
    parser.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON"
    )
    args = parser.parse_args()
    try:
        report = assess(read_manifest(args.manifest), args.repo, args.target_ref)
    except (OSError, ValueError) as exc:
        report = {"status": "ERROR", "error": str(exc)}
        print(json.dumps(report) if args.json else f"ERROR: {exc}")
        return 2
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Target: {report['target_ref']} ({report['target_sha']})")
        for entry in report["upstream_commits"]:
            print(f"{entry['status']:13} {entry['sha']} {entry['reason']}")
        for entry in report["local_adaptations"]:
            print(f"{entry['status']:13} {entry['id']}: {entry['reason']}")
        print("Patch equivalence NOT_CHECKED; this report is not a release gate.")
    return (
        2
        if any(entry["status"] == "UNKNOWN" for entry in report["upstream_commits"])
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
