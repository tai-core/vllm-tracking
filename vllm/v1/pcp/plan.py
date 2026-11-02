# SPDX-License-Identifier: Apache-2.0
"""Shared routing metadata and result handoff for one PCP forward."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PCPForwardPlan:
    # Keys start with a routing kind, followed by metadata/layout identities.
    # Entries retain their inputs so Python identity reuse cannot alias a plan.
    routes: dict[tuple, Any] = field(default_factory=dict)
    # Layer-specific results are consumed once, never shared as routing metadata.
    pending_current: dict[int, Any] = field(default_factory=dict)


def get_pcp_forward_plan() -> PCPForwardPlan | None:
    from vllm.forward_context import (
        get_forward_context,
        is_forward_context_available,
    )

    if not is_forward_context_available():
        return None
    context = get_forward_context()
    plan = getattr(context, "pcp_plan", None)
    if plan is None:
        plan = PCPForwardPlan()
        context.pcp_plan = plan
    return plan


def get_pcp_plan_cache(plan_key=True):
    """Return forward-local routes, or disable reuse for an unkeyed caller."""
    if plan_key is None:
        return None
    plan = get_pcp_forward_plan()
    return None if plan is None else plan.routes
