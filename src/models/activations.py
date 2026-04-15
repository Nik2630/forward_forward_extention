from __future__ import annotations

import torch
from torch import nn


class StudentTNegLogDensity(nn.Module):
    """Activation based on negative log density under Student-t."""

    def __init__(self, df: float = 2.0, scale: float = 1.0) -> None:
        super().__init__()
        if df <= 0:
            raise ValueError("df must be positive")
        if scale <= 0:
            raise ValueError("scale must be positive")
        self.df = float(df)
        self.scale = float(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z2 = (x / self.scale) ** 2
        return 0.5 * (self.df + 1.0) * torch.log1p(z2 / self.df)


class ReLUSafe(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)


def build_activation(name: str, **kwargs: float) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return ReLUSafe()
    if name in {"student_t_nll", "student-t-nll", "student_t"}:
        return StudentTNegLogDensity(
            df=float(kwargs.get("df", 2.0)),
            scale=float(kwargs.get("scale", 1.0)),
        )
    raise ValueError(f"Unsupported activation: {name}")
