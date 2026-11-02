# SPDX-License-Identifier: Apache-2.0
"""Compact page buffers and logical-to-buffer slot mapping."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PCPMaterializedPages:
    logical_pages: torch.Tensor
    data: torch.Tensor

    def remap_slots(self, logical_slots: torch.Tensor, page_rows: int) -> torch.Tensor:
        """Map a known working set to dense slots, preserving padding."""
        pages = torch.div(logical_slots.clamp_min(0), page_rows, rounding_mode="floor")
        indices = torch.searchsorted(self.logical_pages, pages)
        if self.logical_pages.numel() == 0:
            if bool((logical_slots >= 0).any()):
                raise ValueError("Nonempty slots require materialized pages")
            return torch.full_like(logical_slots, -1)
        safe = indices.clamp_max(self.logical_pages.numel() - 1)
        present = (indices < self.logical_pages.numel()) & (
            self.logical_pages[safe] == pages
        )
        if bool(((logical_slots >= 0) & ~present).any()):
            raise ValueError("Slot is outside the materialized working set")
        return torch.where(
            logical_slots >= 0,
            indices * page_rows + logical_slots.remainder(page_rows),
            -1,
        )
