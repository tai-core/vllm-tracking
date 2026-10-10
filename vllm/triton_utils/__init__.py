# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

from vllm.triton_utils.importing import (
    HAS_TRITON,
    TritonLanguagePlaceholder,
    TritonPlaceholder,
)

if TYPE_CHECKING or HAS_TRITON:
    import triton
    import triton.language as tl
    import triton.language.extra.libdevice as tldevice

    try:
        from triton.experimental import gluon
        from triton.experimental.gluon import language as gl
        from triton.language.core import _aggregate as aggregate  # noqa: E501
    except ImportError:
        # The MUSA Torch 2.11 stack uses Triton 3.2, which predates Gluon.
        # Gluon is only consumed by platform-specific AMD kernels, so keep
        # regular Triton available while making unsupported Gluon use fail
        # through the same explicit placeholder as a Triton-less install.
        gluon = TritonLanguagePlaceholder()
        gl = TritonLanguagePlaceholder()
        aggregate = TritonLanguagePlaceholder()
else:
    triton = TritonPlaceholder()
    tl = TritonLanguagePlaceholder()
    tldevice = TritonLanguagePlaceholder()
    gluon = TritonLanguagePlaceholder()
    gl = TritonLanguagePlaceholder()
    aggregate = TritonLanguagePlaceholder()

from vllm.triton_utils.tensor_descriptor import use_tensor_descriptor

LOG2E = 1.4426950408889634
LOGE2 = 0.6931471805599453

__all__ = [
    "HAS_TRITON",
    "triton",
    "tl",
    "tldevice",
    "LOG2E",
    "LOGE2",
    "gluon",
    "gl",
    "aggregate",
    "use_tensor_descriptor",
]
