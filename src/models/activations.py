from __future__ import annotations

import torch
import torch.nn.functional as F
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


class LeakyReLUSafe(nn.Module):
    def __init__(self, negative_slope: float = 0.01) -> None:
        super().__init__()
        self.negative_slope = float(negative_slope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(x, negative_slope=self.negative_slope)


class ELUSafe(nn.Module):
    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.elu(x, alpha=self.alpha)


class GELUSafe(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x)


class SiLUSafe(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x)


class SoftplusSafe(nn.Module):
    def __init__(self, beta: float = 1.0, threshold: float = 20.0) -> None:
        super().__init__()
        self.beta = float(beta)
        self.threshold = float(threshold)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x, beta=self.beta, threshold=self.threshold)


class MishSafe(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(F.softplus(x))


class TanhSafe(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)


class SigmoidSafe(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)


class IdentitySafe(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def available_activation_names() -> list[str]:
    return [
        "relu",
        "leaky_relu",
        "elu",
        "gelu",
        "silu",
        "swish",
        "softplus",
        "mish",
        "tanh",
        "sigmoid",
        "identity",
        "student_t_nll",
    ]


def build_activation(name: str, **kwargs: float) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return ReLUSafe()
    if name in {"leaky_relu", "leaky-relu", "lrelu"}:
        return LeakyReLUSafe(negative_slope=float(kwargs.get("negative_slope", 0.01)))
    if name == "elu":
        return ELUSafe(alpha=float(kwargs.get("alpha", 1.0)))
    if name == "gelu":
        return GELUSafe()
    if name in {"silu", "swish"}:
        return SiLUSafe()
    if name == "softplus":
        return SoftplusSafe(
            beta=float(kwargs.get("beta", 1.0)),
            threshold=float(kwargs.get("threshold", 20.0)),
        )
    if name == "mish":
        return MishSafe()
    if name == "tanh":
        return TanhSafe()
    if name == "sigmoid":
        return SigmoidSafe()
    if name in {"identity", "linear"}:
        return IdentitySafe()
    if name in {"student_t_nll", "student-t-nll", "student_t"}:
        return StudentTNegLogDensity(
            df=float(kwargs.get("df", 2.0)),
            scale=float(kwargs.get("scale", 1.0)),
        )
    raise ValueError(f"Unsupported activation: {name}")
