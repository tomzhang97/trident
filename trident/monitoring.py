"""Lightweight telemetry helpers for TRIDENT."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, List


def population_stability_index(baseline: Iterable[float], current: Iterable[float], epsilon: float = 1e-6) -> float:
    """Compute a coarse PSI between two discrete distributions."""

    baseline = list(baseline)
    current = list(current)
    if len(baseline) != len(current):
        raise ValueError("baseline and current must have the same length")
    psi = 0.0
    for b, c in zip(baseline, current):
        if b <= 0 or c <= 0:
            continue
        ratio = (c + epsilon) / (b + epsilon)
        psi += (c - b) * math.log(ratio)
    return psi


@dataclass
class CoverageMonitor:
    """Tracks empirical violation rates for calibrated tests."""

    target_alpha: float
    history: List[int] = field(default_factory=list)

    def update(self, success: bool) -> None:
        self.history.append(0 if success else 1)

    @property
    def violation_rate(self) -> float:
        if not self.history:
            return 0.0
        return sum(self.history) / len(self.history)

    def alarm(self, tolerance: float) -> bool:
        return self.violation_rate > tolerance
