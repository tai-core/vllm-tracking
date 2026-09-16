# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Prebuild step 1: deliberate preexisting-dist detection + exact removal.

Runs INSIDE the build container (legacy-builder compatible: invoked as
``RUN python /workspace/prebuild_dist_state.py`` -- no heredoc).

Contract:
- Detect vllm / vllm-hcu / vllm_hcu distributions and report each.
- Remove EXACTLY vllm and vllm-hcu (the base image's vllm tree lacks
  GLM and its vllm_hcu dist-info was deleted by the image build --
  both are replaced by this source build).
- torch/triton/compiler/DTK and every other package stay UNTOUCHED.
- A nonzero pip uninstall status is a FAILURE unless every requested
  package is verifiably absent afterwards (pre-deleted dist-info);
  stderr containing "warning" does NOT excuse a nonzero status -- no
  stderr heuristics.
"""

from __future__ import annotations

import importlib.metadata as md
import subprocess
import sys

REMOVE = ("vllm", "vllm-hcu")
DETECT = REMOVE + ("vllm_hcu",)


def main() -> int:
    installed = {}
    for pkg in DETECT:
        try:
            d = md.distribution(pkg)
            installed[pkg] = d.version
            print(f"[dist-state] {pkg}: {d.version} at {d.locate_file('')}")
        except md.PackageNotFoundError:
            print(f"[dist-state] {pkg}: not installed")

    to_remove = [
        p for p in REMOVE if p in installed or p.replace("-", "_") in installed
    ]
    if not to_remove:
        print("[dist-state] neither vllm nor vllm-hcu present -- nothing to remove")
        return 0

    r = subprocess.run(["pip", "uninstall", "-y", *to_remove],
                       capture_output=True, text=True)
    print(r.stdout or r.stderr)
    if r.returncode != 0:
        still = []
        for p in to_remove:
            for name in (p, p.replace("-", "_")):
                try:
                    md.distribution(name)
                    still.append(name)
                except md.PackageNotFoundError:
                    pass
        if still:
            print(f"[dist-state] FAIL: pip uninstall rc={r.returncode} and "
                  f"still installed: {still}", file=sys.stderr)
            return 1
        print("[dist-state] uninstall reported nonzero but packages are "
              "verifiably absent (pre-deleted dist-info) -- continuing")
    return 0


if __name__ == "__main__":
    sys.exit(main())
