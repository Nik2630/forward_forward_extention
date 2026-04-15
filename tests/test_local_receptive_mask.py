from __future__ import annotations

import torch

from src.models.layers import build_local_receptive_mask


def test_local_receptive_mask_shape() -> None:
    mask = build_local_receptive_mask(
        in_channels=3,
        height=32,
        width=32,
        grid_size=4,
        receptive_field=11,
        channels_per_location=2,
    )
    assert mask.shape == (4 * 4 * 2, 3 * 32 * 32)


def test_local_receptive_mask_binary() -> None:
    mask = build_local_receptive_mask(
        in_channels=1,
        height=8,
        width=8,
        grid_size=2,
        receptive_field=4,
        channels_per_location=1,
    )
    assert torch.all((mask == 0) | (mask == 1))
    assert torch.all(mask.sum(dim=1) == 16)
