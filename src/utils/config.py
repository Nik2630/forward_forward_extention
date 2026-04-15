from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    seed: int = 42
    batch_size: int = 128
    epochs: int = 3
    lr: float = 1e-3
    threshold: float = 2.0
