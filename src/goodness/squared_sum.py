from __future__ import annotations

import torch

from .base import Goodness


class SquaredSumGoodness(Goodness):
    def compute(self, activities: torch.Tensor) -> torch.Tensor:
        return torch.sum(activities * activities, dim=1)
