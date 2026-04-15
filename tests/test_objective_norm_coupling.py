from __future__ import annotations

import pytest

from experiments.baseline_mnist import resolve_norm_mode


def test_unsquared_requires_l1_norm() -> None:
    with pytest.raises(ValueError):
        resolve_norm_mode("unsquared", "l2")


def test_auto_mode_picks_expected_norm() -> None:
    assert resolve_norm_mode("unsquared", "auto") == "l1"
    assert resolve_norm_mode("squared", "auto") == "l2"
