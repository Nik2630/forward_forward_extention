from __future__ import annotations

import torch
from torch import nn


class SpatialBlockGoodness(nn.Module):
    """Computes block-wise goodness for spatial feature maps."""

    def __init__(self, block_size: int = 4, stride: int | None = None, squared: bool = True) -> None:
        super().__init__()
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if stride is not None and stride <= 0:
            raise ValueError("stride must be positive")
        self.block_size = block_size
        self.stride = stride or block_size
        self.squared = squared

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("Expected tensor in BCHW format")
        b, c, h, w = x.shape
        bs = self.block_size
        st = self.stride
        if h < bs or w < bs:
            raise ValueError("Height and width must be at least block_size")

        patches = x.unfold(2, bs, st).unfold(3, bs, st)
        out_h = patches.shape[2]
        out_w = patches.shape[3]
        patches = patches.contiguous().view(b, c, out_h * out_w, bs * bs)
        patches = patches.permute(0, 2, 1, 3).contiguous().view(b, out_h * out_w, c * bs * bs)

        if self.squared:
            return torch.sum(patches * patches, dim=2)
        return torch.sum(patches, dim=2)
