from __future__ import annotations

import torch

from src.models.activations import available_activation_names, build_activation


def test_activation_builder_supports_many_names() -> None:
    x = torch.linspace(-3.0, 3.0, steps=128)
    names = [
        "relu",
        "leaky_relu",
        "elu",
        "gelu",
        "silu",
        "softplus",
        "mish",
        "tanh",
        "sigmoid",
        "identity",
        "student_t_nll",
    ]
    for name in names:
        y = build_activation(name)(x)
        assert y.shape == x.shape
        assert torch.isfinite(y).all()


def test_activation_name_registry_contains_core_entries() -> None:
    names = set(available_activation_names())
    assert "relu" in names
    assert "student_t_nll" in names
    assert "gelu" in names
