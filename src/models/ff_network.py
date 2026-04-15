from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import torch
from torch import nn

from ..goodness.base import Goodness
from .activations import build_activation
from .layers import FFLayer


@dataclass
class ForwardTrace:
    pre_norm_activities: List[torch.Tensor]
    final_representation: torch.Tensor


class FFNetwork(nn.Module):
    """Stacked FF layers with per-layer local goodness objectives."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Iterable[int],
        goodness_fns: List[Goodness],
        activation_name: str = "relu",
        activation_kwargs: dict | None = None,
        norm_mode: str = "l2",
        detach_between_layers: bool = True,
        layer_connection_masks: list[torch.Tensor | None] | None = None,
    ) -> None:
        super().__init__()
        hidden_dims = list(hidden_dims)
        if len(goodness_fns) != len(hidden_dims):
            raise ValueError("goodness_fns must match number of hidden layers")

        dims = [input_dim] + hidden_dims
        activation_kwargs = activation_kwargs or {}
        if layer_connection_masks is None:
            layer_connection_masks = [None] * len(hidden_dims)
        if len(layer_connection_masks) != len(hidden_dims):
            raise ValueError("layer_connection_masks must match number of hidden layers")
        layers = []
        for idx in range(len(hidden_dims)):
            layers.append(
                FFLayer(
                    in_features=dims[idx],
                    out_features=dims[idx + 1],
                    activation=build_activation(activation_name, **activation_kwargs),
                    norm_mode=norm_mode,
                    detach_output=detach_between_layers,
                    connection_mask=layer_connection_masks[idx],
                )
            )
        self.layers = nn.ModuleList(layers)
        self.goodness_fns = goodness_fns

    def forward_trace(self, x: torch.Tensor) -> ForwardTrace:
        pre_norm = []
        h = x
        for layer in self.layers:
            layer_out = layer(h)
            pre_norm.append(layer_out.pre_norm)
            h = layer_out.post_norm
        return ForwardTrace(pre_norm_activities=pre_norm, final_representation=h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_trace(x).final_representation

    def local_loss(self, x_pos: torch.Tensor, x_neg: torch.Tensor, threshold: float) -> torch.Tensor:
        pos_trace = self.forward_trace(x_pos)
        neg_trace = self.forward_trace(x_neg)

        losses = []
        for goodness_fn, pos_a, neg_a in zip(
            self.goodness_fns,
            pos_trace.pre_norm_activities,
            neg_trace.pre_norm_activities,
        ):
            losses.append(goodness_fn.loss(pos_a, neg_a, threshold=threshold))
        return torch.mean(torch.stack(losses))

    @torch.no_grad()
    def total_goodness(self, x: torch.Tensor) -> torch.Tensor:
        trace = self.forward_trace(x)
        per_layer = [fn.compute(a) for fn, a in zip(self.goodness_fns, trace.pre_norm_activities)]
        return torch.sum(torch.stack(per_layer, dim=0), dim=0)
