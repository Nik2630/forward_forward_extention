from __future__ import annotations

import torch

from src.goodness.squared_sum import SquaredSumGoodness
from src.models.ff_network import FFNetwork
from src.training.ff_trainer import FFTrainer


def test_train_step_runs() -> None:
    model = FFNetwork(
        input_dim=16,
        hidden_dims=[32, 32],
        goodness_fns=[SquaredSumGoodness(), SquaredSumGoodness()],
        activation_name="relu",
        norm_mode="l2",
        detach_between_layers=True,
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = FFTrainer(model, opt, threshold=1.0)

    x_pos = torch.rand(8, 16)
    x_neg = torch.rand(8, 16)
    out = trainer.train_step(x_pos, x_neg)
    assert out.loss >= 0.0
    assert torch.isfinite(torch.tensor(out.loss))
    assert out.grad_norm >= 0.0
    assert out.had_non_finite is False
