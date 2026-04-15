from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class LayerOutputs:
    pre_norm: torch.Tensor
    post_norm: torch.Tensor


def normalize_activities(x: torch.Tensor, mode: str = "l2", eps: float = 1e-6) -> torch.Tensor:
    if mode == "l2":
        denom = torch.sqrt(torch.sum(x * x, dim=1, keepdim=True) + eps)
        return x / denom
    if mode == "l1":
        denom = torch.sum(x, dim=1, keepdim=True) + eps
        return x / denom
    raise ValueError(f"Unknown normalization mode: {mode}")


class FFLayer(nn.Module):
    """Single FF layer with local activity normalization."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: nn.Module,
        norm_mode: str = "l2",
        detach_output: bool = True,
        connection_mask: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.linear = MaskedLinear(in_features, out_features, connection_mask=connection_mask)
        self.activation = activation
        self.norm_mode = norm_mode
        self.detach_output = detach_output

    def forward(self, x: torch.Tensor) -> LayerOutputs:
        pre = self.activation(self.linear(x))
        post = normalize_activities(pre, mode=self.norm_mode)
        if self.detach_output:
            post = post.detach()
        return LayerOutputs(pre_norm=pre, post_norm=post)


class MaskedLinear(nn.Module):
    """Linear layer with optional fixed binary mask over weights."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        connection_mask: torch.Tensor | None = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        mask_tensor = None
        if connection_mask is not None:
            if connection_mask.shape != (out_features, in_features):
                raise ValueError("connection_mask shape mismatch")
            mask_tensor = connection_mask.float()
        self.register_buffer("mask", mask_tensor)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / fan_in ** 0.5
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        if self.mask is not None:
            weight = weight * self.mask
        return F.linear(x, weight, self.bias)


def build_local_receptive_mask(
    in_channels: int,
    height: int,
    width: int,
    grid_size: int,
    receptive_field: int,
    channels_per_location: int,
) -> torch.Tensor:
    """Builds a binary mask for non-overlapping local receptive fields without weight sharing."""
    if grid_size <= 0 or receptive_field <= 0 or channels_per_location <= 0:
        raise ValueError("grid_size, receptive_field, and channels_per_location must be positive")

    out_features = grid_size * grid_size * channels_per_location
    in_features = in_channels * height * width
    mask = torch.zeros((out_features, in_features), dtype=torch.float32)

    if grid_size == 1:
        stride_h = 1
        stride_w = 1
    else:
        stride_h = max(1, (height - receptive_field) // (grid_size - 1))
        stride_w = max(1, (width - receptive_field) // (grid_size - 1))

    out_idx = 0
    for gy in range(grid_size):
        y0 = min(gy * stride_h, height - receptive_field)
        y1 = y0 + receptive_field
        for gx in range(grid_size):
            x0 = min(gx * stride_w, width - receptive_field)
            x1 = x0 + receptive_field

            pixel_indices = []
            for c in range(in_channels):
                for yy in range(y0, y1):
                    for xx in range(x0, x1):
                        pixel_indices.append(c * height * width + yy * width + xx)

            for _ in range(channels_per_location):
                mask[out_idx, pixel_indices] = 1.0
                out_idx += 1

    return mask
