from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class MetricsTracker:
    history: Dict[str, List[float]] = field(default_factory=dict)

    def update(self, **kwargs: float) -> None:
        for key, value in kwargs.items():
            self.history.setdefault(key, []).append(float(value))

    def latest(self) -> Dict[str, float]:
        return {k: v[-1] for k, v in self.history.items() if v}
