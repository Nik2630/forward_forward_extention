from __future__ import annotations

import torch

from src.goodness.factory import available_goodness_names, build_goodness


def test_goodness_builder_supports_many_names() -> None:
    x = torch.randn(16, 32)
    names = [
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
    for name in names:
        goodness = build_goodness(name)
        y = goodness.compute(x)
        assert y.shape == (16,)
        assert torch.isfinite(y).all()


def test_goodness_topk_respects_feature_dim() -> None:
    x = torch.tensor([[1.0, -2.0, 0.5]])
    topk = build_goodness("topk_abs", k=10)
    y = topk.compute(x)
    assert torch.allclose(y, torch.tensor([3.5]))


def test_goodness_name_registry_contains_core_entries() -> None:
    names = set(available_goodness_names())
    assert "squared_sum" in names
    assert "unsquared_sum" in names
    assert "huber_sum" in names
