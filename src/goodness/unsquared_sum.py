from __future__ import annotations

import torch

from .base import Goodness


class UnsquaredSumGoodness(Goodness):
    def compute(self, activities: torch.Tensor) -> torch.Tensor:
        return torch.sum(activities, dim=1)
