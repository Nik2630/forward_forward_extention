from __future__ import annotations

import torch

from src.goodness.squared_sum import SquaredSumGoodness
from src.goodness.unsquared_sum import UnsquaredSumGoodness


def test_squared_sum_goodness_values() -> None:
    x = torch.tensor([[1.0, 2.0, 3.0], [2.0, 0.0, 1.0]])
    goodness = SquaredSumGoodness().compute(x)
    assert torch.allclose(goodness, torch.tensor([14.0, 5.0]))


def test_unsquared_sum_goodness_values() -> None:
    x = torch.tensor([[1.0, 2.0, 3.0], [2.0, 0.0, 1.0]])
    goodness = UnsquaredSumGoodness().compute(x)
    assert torch.allclose(goodness, torch.tensor([6.0, 3.0]))
