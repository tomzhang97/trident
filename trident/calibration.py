"""Lightweight reliability calibration utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple


def _pava(scores: Sequence[float], failures: Sequence[float]) -> List[float]:
    """Perform a simple isotonic regression using pool-adjacent-violators."""

    if len(scores) != len(failures):
        raise ValueError("scores and failures must be the same length")
    # Sort by score ascending to enforce monotonicity
    pairs = sorted(zip(scores, failures), key=lambda x: x[0])
    values = [pair[1] for pair in pairs]
    weights = [1.0] * len(values)
    idx = 0
    while idx < len(values) - 1:
        if values[idx] <= values[idx + 1]:
            idx += 1
            continue
        total_weight = weights[idx] + weights[idx + 1]
        avg = (values[idx] * weights[idx] + values[idx + 1] * weights[idx + 1]) / total_weight
        values[idx] = avg
        weights[idx] = total_weight
        del values[idx + 1]
        del weights[idx + 1]
        idx = max(idx - 1, 0)
    expanded: List[float] = []
    pos = 0
    for weight, value in zip(weights, values):
        expanded.extend([value] * int(weight))
        pos += int(weight)
    # Expand might be shorter due to integer rounding; align with sorted pairs length.
    while len(expanded) < len(pairs):
        expanded.append(values[-1])
    return expanded[: len(pairs)]


@dataclass
class ReliabilityCalibrator:
    """Facet-aware monotonic mapping from scores to p-values."""

    reliability_table_size: int = 20
    tables: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)
    version: str = "dev"

    def fit(
        self,
        scores: Sequence[float],
        sufficiency_labels: Sequence[int],
        bucket_key: str,
    ) -> None:
        """Fit a reliability table for the given bucket."""

        if len(scores) == 0:
            raise ValueError("Cannot fit calibrator on empty data")
        failures = [1.0 - label for label in sufficiency_labels]
        iso_values = _pava(scores, failures)
        sorted_pairs = sorted(zip(scores, iso_values), key=lambda x: x[0])
        stride = max(1, len(sorted_pairs) // self.reliability_table_size)
        table = [sorted_pairs[i] for i in range(0, len(sorted_pairs), stride)]
        if table[-1][0] != sorted_pairs[-1][0]:
            table.append(sorted_pairs[-1])
        self.tables[bucket_key] = table

    def to_pvalue(self, score: float, bucket_key: str) -> float:
        """Map a score to a calibrated p-value for the requested bucket."""

        table = self.tables.get(bucket_key)
        if not table:
            # Default monotone mapping: higher scores imply lower p-value.
            return max(0.0, 1.0 - score)
        for threshold, p_value in table:
            if score <= threshold:
                return min(max(p_value, 0.0), 1.0)
        return min(max(table[-1][1], 0.0), 1.0)

    def merge(self, other: "ReliabilityCalibrator") -> None:
        """Merge calibration tables from another calibrator instance."""

        for key, table in other.tables.items():
            self.tables[key] = table

    @classmethod
    def from_mapping(cls, mapping: Dict[str, List[Tuple[float, float]]], version: str = "dev") -> "ReliabilityCalibrator":
        calibrator = cls()
        calibrator.tables = mapping
        calibrator.version = version
        return calibrator
