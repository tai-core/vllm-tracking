# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MUSA: Python 3.10 backfills for the Python 3.11 names vLLM imports."""

import datetime
import sys
import typing

import typing_extensions

_TYPING_NAMES = (
    "LiteralString",
    "Never",
    "NotRequired",
    "Required",
    "Self",
    "TypeVarTuple",
    "Unpack",
    "assert_never",
    "dataclass_transform",
    "reveal_type",
)


def install() -> None:
    if sys.version_info >= (3, 11):
        return
    for name in _TYPING_NAMES:
        if not hasattr(typing, name):
            setattr(typing, name, getattr(typing_extensions, name))
    if not hasattr(datetime, "UTC"):
        datetime.UTC = datetime.timezone.utc  # type: ignore[attr-defined]
