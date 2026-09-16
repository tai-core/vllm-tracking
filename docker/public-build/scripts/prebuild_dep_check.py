# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Prebuild step 2: explicit build-dependency validation (fail on missing).

Runs INSIDE the build container (legacy-builder compatible: invoked as
``RUN python /workspace/prebuild_dep_check.py`` -- no heredoc).

NO implicit resolution/upgrade happens anywhere in this build (--no-deps
installs); everything needed must already be in the base. A missing item
is a concrete missing-input finding -- fail loudly, do not mask.
"""

from __future__ import annotations

import importlib
import importlib.metadata as md
import sys

REQUIRED_MODULES = ["torch", "triton", "ninja", "setuptools", "packaging"]
REQUIRED_DISTS = ["torch", "triton", "ninja", "setuptools"]


def main() -> int:
    missing = []
    for m in REQUIRED_MODULES:
        try:
            importlib.import_module(m)
        except Exception as exc:  # noqa: BLE001 -- report the reason verbatim
            missing.append(f"{m}: {type(exc).__name__}")
    for d in REQUIRED_DISTS:
        try:
            md.distribution(d)
        except md.PackageNotFoundError:
            missing.append(f"{d}: distribution missing")

    import torch

    print(f"[deps] torch {torch.__version__} hip={torch.version.hip}")
    if torch.version.hip is None:
        missing.append("torch.version.hip is None -- not a DTK/ROCm-line torch")

    if missing:
        print("[deps] MISSING (concrete missing-input finding, do not mask):",
              file=sys.stderr)
        for m in missing:
            print("  -", m, file=sys.stderr)
        return 1
    print("[deps] all required build dependencies present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
