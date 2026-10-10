"""Compatibility exports for v0.28 MUSA attention backends."""
from vllm.v1.attention.ops.dcp import dcp_a2a_lse_reduce

__all__ = ["dcp_a2a_lse_reduce"]
