"""Enhanced calibration utilities with Mondrian/split support."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Any

import numpy as np
from scipy import stats


def _pava(scores: Sequence[float], failures: Sequence[float]) -> List[float]:
    """Perform isotonic regression using pool-adjacent-violators algorithm."""
    
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
        
        # Violates monotonicity, pool adjacent values
        total_weight = weights[idx] + weights[idx + 1]
        avg = (values[idx] * weights[idx] + values[idx + 1] * weights[idx + 1]) / total_weight
        values[idx] = avg
        weights[idx] = total_weight
        del values[idx + 1]
        del weights[idx + 1]
        idx = max(idx - 1, 0)
    
    # Expand back to original length
    expanded: List[float] = []
    for weight, value in zip(weights, values):
        expanded.extend([value] * int(weight))
    
    # Pad if needed
    while len(expanded) < len(pairs):
        expanded.append(values[-1] if values else 0.0)
    
    return expanded[:len(pairs)]


@dataclass
class MondrianBin:
    """A bin in Mondrian conformal calibration."""
    facet_type: str
    length_bin: str
    score_range: Tuple[float, float]
    samples: List[Tuple[float, bool]] = field(default_factory=list)
    calibration_map: Optional[List[Tuple[float, float]]] = None
    
    def add_sample(self, score: float, is_correct: bool) -> None:
        """Add a calibration sample."""
        self.samples.append((score, is_correct))
    
    def fit_isotonic(self) -> None:
        """Fit isotonic regression on this bin."""
        if not self.samples:
            return
        
        scores = [s for s, _ in self.samples]
        failures = [1.0 - int(correct) for _, correct in self.samples]
        
        iso_values = _pava(scores, failures)
        sorted_pairs = sorted(zip(scores, iso_values), key=lambda x: x[0])
        
        # Create calibration map
        stride = max(1, len(sorted_pairs) // 20)
        self.calibration_map = [sorted_pairs[i] for i in range(0, len(sorted_pairs), stride)]
        if sorted_pairs and self.calibration_map[-1][0] != sorted_pairs[-1][0]:
            self.calibration_map.append(sorted_pairs[-1])
    
    def get_pvalue(self, score: float) -> float:
        """Get calibrated p-value for score."""
        if not self.calibration_map:
            # Uncalibrated fallback
            return max(0.0, 1.0 - score)
        
        for threshold, p_value in self.calibration_map:
            if score <= threshold:
                return min(max(p_value, 0.0), 1.0)
        
        return min(max(self.calibration_map[-1][1], 0.0), 1.0)


@dataclass
class ReliabilityCalibrator:
    """
    Enhanced calibrator with Mondrian/split conformal prediction.
    
    From TRIDENT spec: Uses split/Mondrian + isotonic calibration
    stratified by facet type and length bin.
    """
    
    reliability_table_size: int = 20
    tables: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)
    mondrian_bins: Dict[str, MondrianBin] = field(default_factory=dict)
    version: str = "v1.0"
    use_mondrian: bool = True
    
    # Length bins for Mondrian
    length_bins = [(0, 50, "short"), (50, 150, "medium"), (150, float('inf'), "long")]
    
    # Score bins for additional stratification
    score_bins = [(0.0, 0.3), (0.3, 0.7), (0.7, 1.0)]
    
    def get_bin_key(
        self,
        facet_type: str,
        text_length: int,
        score: Optional[float] = None
    ) -> str:
        """Get Mondrian bin key for given features."""
        # Determine length bin
        length_bin = "medium"
        for min_len, max_len, bin_name in self.length_bins:
            if min_len <= text_length < max_len:
                length_bin = bin_name
                break
        
        # Optionally include score bin
        if score is not None and self.use_mondrian:
            for min_score, max_score in self.score_bins:
                if min_score <= score < max_score:
                    return f"{facet_type}_{length_bin}_{min_score:.1f}_{max_score:.1f}"
        
        return f"{facet_type}_{length_bin}"
    
    def fit(
        self,
        scores: Sequence[float],
        sufficiency_labels: Sequence[int],
        bucket_key: str,
        text_lengths: Optional[Sequence[int]] = None
    ) -> None:
        """
        Fit calibration for a bucket.
        
        If text_lengths provided and use_mondrian=True, uses Mondrian bins.
        Otherwise falls back to simple isotonic calibration.
        """
        if len(scores) == 0:
            raise ValueError("Cannot fit calibrator on empty data")
        
        if self.use_mondrian and text_lengths is not None:
            # Mondrian calibration
            self._fit_mondrian(scores, sufficiency_labels, bucket_key, text_lengths)
        else:
            # Simple isotonic calibration
            self._fit_isotonic(scores, sufficiency_labels, bucket_key)
    
    def _fit_isotonic(
        self,
        scores: Sequence[float],
        sufficiency_labels: Sequence[int],
        bucket_key: str
    ) -> None:
        """Fit simple isotonic regression."""
        failures = [1.0 - label for label in sufficiency_labels]
        iso_values = _pava(scores, failures)
        sorted_pairs = sorted(zip(scores, iso_values), key=lambda x: x[0])
        
        stride = max(1, len(sorted_pairs) // self.reliability_table_size)
        table = [sorted_pairs[i] for i in range(0, len(sorted_pairs), stride)]
        if table[-1][0] != sorted_pairs[-1][0]:
            table.append(sorted_pairs[-1])
        
        self.tables[bucket_key] = table
    
    def _fit_mondrian(
        self,
        scores: Sequence[float],
        sufficiency_labels: Sequence[int],
        facet_type: str,
        text_lengths: Sequence[int]
    ) -> None:
        """Fit Mondrian conformal calibration."""
        # Create bins and add samples
        for score, label, text_len in zip(scores, sufficiency_labels, text_lengths):
            bin_key = self.get_bin_key(facet_type, text_len, score)
            
            if bin_key not in self.mondrian_bins:
                # Determine bins
                length_bin = "medium"
                for min_len, max_len, bin_name in self.length_bins:
                    if min_len <= text_len < max_len:
                        length_bin = bin_name
                        break
                
                score_range = (0.0, 1.0)
                for min_score, max_score in self.score_bins:
                    if min_score <= score < max_score:
                        score_range = (min_score, max_score)
                        break
                
                self.mondrian_bins[bin_key] = MondrianBin(
                    facet_type=facet_type,
                    length_bin=length_bin,
                    score_range=score_range
                )
            
            self.mondrian_bins[bin_key].add_sample(score, bool(label))
        
        # Fit isotonic regression for each bin
        for bin_key in self.mondrian_bins:
            self.mondrian_bins[bin_key].fit_isotonic()
    
    def to_pvalue(
        self,
        score: float,
        bucket_key: str,
        text_length: Optional[int] = None
    ) -> float:
        """
        Map a score to a calibrated p-value.
        
        Uses Mondrian bins if available, otherwise falls back to simple tables.
        """
        # Try Mondrian first if enabled
        if self.use_mondrian and text_length is not None:
            # Parse facet type from bucket_key
            facet_type = bucket_key.split('_')[0] if '_' in bucket_key else bucket_key
            bin_key = self.get_bin_key(facet_type, text_length, score)
            
            if bin_key in self.mondrian_bins:
                return self.mondrian_bins[bin_key].get_pvalue(score)
        
        # Fall back to simple calibration
        table = self.tables.get(bucket_key)
        if not table:
            # Default monotone mapping
            return max(0.0, 1.0 - score)
        
        for threshold, p_value in table:
            if score <= threshold:
                return min(max(p_value, 0.0), 1.0)
        
        return min(max(table[-1][1], 0.0), 1.0)
    
    def merge(self, other: "ReliabilityCalibrator") -> None:
        """Merge calibration tables from another calibrator."""
        for key, table in other.tables.items():
            self.tables[key] = table
        
        for key, bin_obj in other.mondrian_bins.items():
            self.mondrian_bins[key] = bin_obj
    
    def save(self, path: str) -> None:
        """Save calibrator to file."""
        data = {
            'version': self.version,
            'use_mondrian': self.use_mondrian,
            'tables': dict(self.tables),
            'mondrian_bins': {}
        }
        
        # Serialize Mondrian bins
        for key, bin_obj in self.mondrian_bins.items():
            data['mondrian_bins'][key] = {
                'facet_type': bin_obj.facet_type,
                'length_bin': bin_obj.length_bin,
                'score_range': bin_obj.score_range,
                'calibration_map': bin_obj.calibration_map
            }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "ReliabilityCalibrator":
        """Load calibrator from file."""
        with open(path) as f:
            data = json.load(f)
        
        calibrator = cls(
            version=data['version'],
            use_mondrian=data.get('use_mondrian', True)
        )
        calibrator.tables = data['tables']
        
        # Reconstruct Mondrian bins
        for key, bin_data in data.get('mondrian_bins', {}).items():
            bin_obj = MondrianBin(
                facet_type=bin_data['facet_type'],
                length_bin=bin_data['length_bin'],
                score_range=tuple(bin_data['score_range'])
            )
            bin_obj.calibration_map = bin_data['calibration_map']
            calibrator.mondrian_bins[key] = bin_obj
        
        return calibrator
    
    @classmethod
    def from_mapping(
        cls,
        mapping: Dict[str, List[Tuple[float, float]]],
        version: str = "v1.0"
    ) -> "ReliabilityCalibrator":
        """Create calibrator from mapping."""
        calibrator = cls()
        calibrator.tables = mapping
        calibrator.version = version
        return calibrator
    
    def get_version_hash(self) -> str:
        """Get hash of calibration tables for versioning."""
        content = json.dumps({
            'tables': dict(self.tables),
            'mondrian_bins': len(self.mondrian_bins)
        }, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()[:8]


@dataclass 
class CalibrationMonitor:
    """Monitor calibration quality over time."""
    
    target_alpha: float
    window_size: int = 100
    history: List[Tuple[float, bool]] = field(default_factory=list)
    
    def update(self, p_value: float, is_correct: bool) -> None:
        """Update with new observation."""
        self.history.append((p_value, is_correct))
        
        # Keep only recent history
        if len(self.history) > self.window_size * 10:
            self.history = self.history[-self.window_size * 5:]
    
    def get_violation_rate(self, alpha: float) -> float:
        """Get empirical violation rate at given alpha."""
        if not self.history:
            return 0.0
        
        recent = self.history[-self.window_size:] if len(self.history) > self.window_size else self.history
        
        violations = sum(
            1 for p, correct in recent 
            if p <= alpha and not correct
        )
        total = sum(1 for p, _ in recent if p <= alpha)
        
        return violations / max(total, 1)
    
    def is_calibrated(self, tolerance: float = 0.05) -> bool:
        """Check if calibration is within tolerance."""
        violation_rate = self.get_violation_rate(self.target_alpha)
        return abs(violation_rate - self.target_alpha) <= tolerance