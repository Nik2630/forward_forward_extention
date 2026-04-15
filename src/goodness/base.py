from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F


class Goodness(ABC):
    """Abstract local objective over pre-normalized activities."""

    @abstractmethod
    def compute(self, activities: torch.Tensor) -> torch.Tensor:
        """Returns one goodness value per sample."""

    def loss(
        self,
        pos_activities: torch.Tensor,
        neg_activities: torch.Tensor,
        threshold: float,
    ) -> torch.Tensor:
        pos_goodness = self.compute(pos_activities)
        neg_goodness = self.compute(neg_activities)

        pos_logits = pos_goodness - threshold
        neg_logits = neg_goodness - threshold

        pos_loss = F.binary_cross_entropy_with_logits(
            pos_logits,
            torch.ones_like(pos_logits),
        )
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_logits,
            torch.zeros_like(neg_logits),
        )
        return 0.5 * (pos_loss + neg_loss)
