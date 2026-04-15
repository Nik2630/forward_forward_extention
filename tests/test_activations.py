from __future__ import annotations

import torch

from src.models.activations import StudentTNegLogDensity


def test_student_t_activation_finite() -> None:
    act = StudentTNegLogDensity(df=2.0, scale=1.0)
    x = torch.linspace(-20.0, 20.0, steps=1024)
    y = act(x)
    assert torch.isfinite(y).all()
    assert (y >= 0).all()
