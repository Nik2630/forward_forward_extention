from __future__ import annotations

import torch

from src.goodness.spatial import SpatialBlockGoodness


def test_spatial_goodness_non_overlapping_shape() -> None:
    x = torch.randn(2, 3, 8, 8)
    layer = SpatialBlockGoodness(block_size=4, stride=4, squared=True)
    out = layer(x)
    assert out.shape == (2, 4)


def test_spatial_goodness_overlapping_shape() -> None:
    x = torch.randn(2, 1, 8, 8)
    layer = SpatialBlockGoodness(block_size=4, stride=2, squared=False)
    out = layer(x)
    assert out.shape == (2, 9)
