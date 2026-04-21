from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import Goodness
from .squared_sum import SquaredSumGoodness
from .unsquared_sum import UnsquaredSumGoodness


class MeanSquaredGoodness(Goodness):
    def compute(self, activities: torch.Tensor) -> torch.Tensor:
        return torch.mean(activities * activities, dim=1)


class MeanAbsGoodness(Goodness):
    def compute(self, activities: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(activities), dim=1)


class L2NormGoodness(Goodness):
    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = float(eps)

    def compute(self, activities: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.sum(activities * activities, dim=1) + self.eps)


class MaxAbsGoodness(Goodness):
    def compute(self, activities: torch.Tensor) -> torch.Tensor:
        return torch.max(torch.abs(activities), dim=1).values


class LogCoshGoodness(Goodness):
    def compute(self, activities: torch.Tensor) -> torch.Tensor:
        return torch.sum(torch.log(torch.cosh(activities)), dim=1)


class HuberSumGoodness(Goodness):
    def __init__(self, delta: float = 1.0) -> None:
        if delta <= 0:
            raise ValueError("delta must be positive")
        self.delta = float(delta)

    def compute(self, activities: torch.Tensor) -> torch.Tensor:
        zero = torch.zeros_like(activities)
        huber = F.huber_loss(activities, zero, reduction="none", delta=self.delta)
        return torch.sum(huber, dim=1)


class TopKAbsGoodness(Goodness):
    def __init__(self, k: int = 10) -> None:
        if k <= 0:
            raise ValueError("k must be positive")
        self.k = int(k)

    def compute(self, activities: torch.Tensor) -> torch.Tensor:
        k = min(self.k, activities.shape[1])
        values, _ = torch.topk(torch.abs(activities), k=k, dim=1)
        return torch.sum(values, dim=1)


class SoftplusSumGoodness(Goodness):
    def __init__(self, beta: float = 1.0, threshold: float = 20.0) -> None:
        self.beta = float(beta)
        self.threshold = float(threshold)

    def compute(self, activities: torch.Tensor) -> torch.Tensor:
        return torch.sum(F.softplus(activities, beta=self.beta, threshold=self.threshold), dim=1)


class ExponentialMeanGoodness(Goodness):
    def __init__(self, clamp: float = 30.0) -> None:
        self.clamp = float(clamp)

    def compute(self, activities: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(activities, min=-self.clamp, max=self.clamp)
        return torch.mean(torch.exp(x), dim=1)


def available_goodness_names() -> list[str]:
    return [
        "squared_sum",
        "unsquared_sum",
        "mean_squared",
        "mean_abs",
        "l2_norm",
        "max_abs",
        "log_cosh",
        "huber_sum",
        "topk_abs",
        "softplus_sum",
        "exp_mean",
    ]


def build_goodness(name: str, **kwargs: float | int) -> Goodness:
    name = name.lower()
    if name in {"squared_sum", "squared", "ssq"}:
        return SquaredSumGoodness()
    if name in {"unsquared_sum", "unsquared", "sum"}:
        return UnsquaredSumGoodness()
    if name in {"mean_squared", "msq"}:
        return MeanSquaredGoodness()
    if name in {"mean_abs", "l1_mean"}:
        return MeanAbsGoodness()
    if name in {"l2_norm", "norm"}:
        return L2NormGoodness(eps=float(kwargs.get("eps", 1e-8)))
    if name in {"max_abs", "max"}:
        return MaxAbsGoodness()
    if name in {"log_cosh", "logcosh"}:
        return LogCoshGoodness()
    if name in {"huber_sum", "huber"}:
        return HuberSumGoodness(delta=float(kwargs.get("delta", 1.0)))
    if name in {"topk_abs", "topk"}:
        return TopKAbsGoodness(k=int(kwargs.get("k", 10)))
    if name in {"softplus_sum", "softplus"}:
        return SoftplusSumGoodness(
            beta=float(kwargs.get("beta", 1.0)),
            threshold=float(kwargs.get("threshold", 20.0)),
        )
    if name in {"exp_mean", "exponential_mean", "exp"}:
        return ExponentialMeanGoodness(clamp=float(kwargs.get("clamp", 30.0)))
    raise ValueError(f"Unsupported goodness: {name}")
