from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.optim import Optimizer

from .metrics import MetricsTracker


@dataclass
class TrainStepOutput:
    loss: float
    grad_norm: float
    had_non_finite: bool


class FFTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        threshold: float = 2.0,
        max_grad_norm: float = 5.0,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.threshold = threshold
        self.max_grad_norm = max_grad_norm
        self.metrics = MetricsTracker()

    def train_step(self, x_pos: torch.Tensor, x_neg: torch.Tensor) -> TrainStepOutput:
        self.model.train()
        loss = self.model.local_loss(x_pos, x_neg, threshold=self.threshold)

        if not torch.isfinite(loss):
            raise ValueError("Non-finite loss encountered during FF training")

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        had_non_finite = False
        grad_norm_sq = 0.0
        for param in self.model.parameters():
            if param.grad is None:
                continue
            if not torch.isfinite(param.grad).all():
                had_non_finite = True
            grad_norm_sq += float(torch.sum(param.grad.detach() * param.grad.detach()).cpu().item())

        grad_norm = grad_norm_sq ** 0.5
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        loss_value = float(loss.detach().cpu().item())
        self.metrics.update(train_loss=loss_value, grad_norm=grad_norm)
        return TrainStepOutput(loss=loss_value, grad_norm=grad_norm, had_non_finite=had_non_finite)
