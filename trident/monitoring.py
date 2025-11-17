"""Monitoring utilities for drift detection and calibration."""

from __future__ import annotations

import math
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any


@dataclass
class DriftMonitor:
    """Monitor for distribution drift in scores."""
    
    config: Any
    score_history: deque = field(default_factory=lambda: deque(maxlen=1000))
    baseline_distribution: Optional[np.ndarray] = None
    psi_history: List[float] = field(default_factory=list)
    alarm_triggered: bool = False
    
    def check_drift(self, scores: Dict[Tuple[str, str], float]) -> bool:
        """Check if distribution drift has occurred."""
        if not scores:
            return False
        
        # Add scores to history
        current_scores = list(scores.values())
        self.score_history.extend(current_scores)
        
        # Need baseline to compare
        if self.baseline_distribution is None:
            if len(self.score_history) >= 100:
                self.baseline_distribution = self._compute_distribution(
                    list(self.score_history)[:100]
                )
            return False
        
        # Compute PSI
        if len(self.score_history) >= 50:
            recent_scores = list(self.score_history)[-50:]
            recent_dist = self._compute_distribution(recent_scores)
            psi = self._compute_psi(self.baseline_distribution, recent_dist)
            self.psi_history.append(psi)
            
            # Check threshold
            if psi > self.config.psi_threshold:
                self.alarm_triggered = True
                return True
        
        return False
    
    def _compute_distribution(self, scores: List[float], n_bins: int = 10) -> np.ndarray:
        """Compute score distribution."""
        hist, _ = np.histogram(scores, bins=n_bins, range=(0, 1))
        hist = hist + 1e-6  # Avoid zeros
        return hist / hist.sum()
    
    def _compute_psi(self, baseline: np.ndarray, current: np.ndarray) -> float:
        """Compute Population Stability Index."""
        psi = 0.0
        for b, c in zip(baseline, current):
            if b > 0 and c > 0:
                psi += (c - b) * math.log(c / b)
        return psi
    
    def reset_baseline(self) -> None:
        """Reset baseline distribution."""
        if len(self.score_history) >= 100:
            self.baseline_distribution = self._compute_distribution(
                list(self.score_history)[-100:]
            )
        self.alarm_triggered = False


@dataclass
class CalibrationMonitor:
    """Monitor calibration quality."""
    
    target_alpha: float
    window_size: int = 100
    history: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    def update(self, p_value: float, is_correct: bool) -> None:
        """Update with new observation."""
        self.history.append((p_value, is_correct))
    
    def get_violation_rate(self, alpha: float) -> float:
        """Get empirical violation rate at given alpha."""
        if not self.history:
            return 0.0
        
        violations = sum(
            1 for p, correct in self.history 
            if p <= alpha and not correct
        )
        total = sum(1 for p, _ in self.history if p <= alpha)
        
        return violations / max(total, 1)
    
    def is_calibrated(self, tolerance: float = 0.05) -> bool:
        """Check if calibration is within tolerance."""
        violation_rate = self.get_violation_rate(self.target_alpha)
        return abs(violation_rate - self.target_alpha) <= tolerance
    
    def get_calibration_curve(self, n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Get calibration curve."""
        if len(self.history) < n_bins:
            return np.array([]), np.array([])
        
        alphas = np.linspace(0, 1, n_bins + 1)
        expected = []
        observed = []
        
        for i in range(n_bins):
            alpha_low = alphas[i]
            alpha_high = alphas[i + 1]
            
            in_bin = [
                (p, correct) for p, correct in self.history
                if alpha_low <= p < alpha_high
            ]
            
            if in_bin:
                expected.append((alpha_low + alpha_high) / 2)
                observed.append(sum(correct for _, correct in in_bin) / len(in_bin))
        
        return np.array(expected), np.array(observed)


@dataclass
class CoverageMonitor:
    """Monitor facet coverage rates."""
    
    facet_coverage: Dict[str, List[bool]] = field(default_factory=dict)
    passage_efficiency: List[float] = field(default_factory=list)
    
    def update_coverage(
        self,
        facets: List[Any],
        covered_facets: List[str],
        tokens_used: int
    ) -> None:
        """Update coverage statistics."""
        for facet in facets:
            facet_id = facet.facet_id if hasattr(facet, 'facet_id') else str(facet)
            is_covered = facet_id in covered_facets
            
            if facet_id not in self.facet_coverage:
                self.facet_coverage[facet_id] = []
            self.facet_coverage[facet_id].append(is_covered)
        
        # Track efficiency
        coverage_rate = len(covered_facets) / max(len(facets), 1)
        efficiency = coverage_rate / max(tokens_used, 1) * 1000  # Coverage per 1k tokens
        self.passage_efficiency.append(efficiency)
    
    def get_facet_success_rates(self) -> Dict[str, float]:
        """Get success rate for each facet type."""
        success_rates = {}
        for facet_id, coverage_history in self.facet_coverage.items():
            if coverage_history:
                success_rates[facet_id] = sum(coverage_history) / len(coverage_history)
        return success_rates
    
    def get_average_efficiency(self) -> float:
        """Get average passage efficiency."""
        if not self.passage_efficiency:
            return 0.0
        return np.mean(self.passage_efficiency)
    
    def get_coverage_trend(self, window: int = 50) -> List[float]:
        """Get coverage trend over time."""
        all_coverage = []
        
        # Aggregate all facet coverage
        max_len = max(len(v) for v in self.facet_coverage.values()) if self.facet_coverage else 0
        
        for i in range(max_len):
            covered = 0
            total = 0
            for facet_history in self.facet_coverage.values():
                if i < len(facet_history):
                    covered += facet_history[i]
                    total += 1
            
            if total > 0:
                all_coverage.append(covered / total)
        
        # Compute moving average
        if len(all_coverage) < window:
            return all_coverage
        
        trend = []
        for i in range(len(all_coverage) - window + 1):
            window_avg = np.mean(all_coverage[i:i + window])
            trend.append(window_avg)
        
        return trend


class SystemMonitor:
    """Comprehensive system monitoring."""
    
    def __init__(self, config: Any):
        self.config = config
        self.drift_monitor = DriftMonitor(config)
        self.calibration_monitors: Dict[str, CalibrationMonitor] = {}
        self.coverage_monitor = CoverageMonitor()
        self.latency_tracker = LatencyTracker()
        self.memory_tracker = MemoryTracker() if config.telemetry.profile_memory else None
    
    def update(self, pipeline_output: Dict[str, Any]) -> None:
        """Update all monitors with pipeline output."""
        # Update drift monitor
        if 'scores' in pipeline_output:
            self.drift_monitor.check_drift(pipeline_output['scores'])
        
        # Update calibration monitors
        if 'certificates' in pipeline_output:
            for cert in pipeline_output['certificates']:
                facet_type = cert.get('facet_type', 'default')
                if facet_type not in self.calibration_monitors:
                    self.calibration_monitors[facet_type] = CalibrationMonitor(
                        target_alpha=cert.get('alpha_bar', 0.01)
                    )
                
                self.calibration_monitors[facet_type].update(
                    p_value=cert['p_value'],
                    is_correct=cert.get('is_correct', True)  # Would need ground truth
                )
        
        # Update coverage
        if 'facets' in pipeline_output and 'covered_facets' in pipeline_output:
            self.coverage_monitor.update_coverage(
                facets=pipeline_output['facets'],
                covered_facets=pipeline_output['covered_facets'],
                tokens_used=pipeline_output.get('tokens_used', 0)
            )
        
        # Update latency
        if 'latency_ms' in pipeline_output:
            self.latency_tracker.record(pipeline_output['latency_ms'])
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status report."""
        status = {
            'drift': {
                'alarm_triggered': self.drift_monitor.alarm_triggered,
                'latest_psi': self.drift_monitor.psi_history[-1] if self.drift_monitor.psi_history else 0
            },
            'calibration': {
                facet_type: {
                    'is_calibrated': monitor.is_calibrated(),
                    'violation_rate': monitor.get_violation_rate(monitor.target_alpha)
                }
                for facet_type, monitor in self.calibration_monitors.items()
            },
            'coverage': {
                'average_efficiency': self.coverage_monitor.get_average_efficiency(),
                'success_rates': self.coverage_monitor.get_facet_success_rates()
            },
            'latency': self.latency_tracker.get_stats()
        }
        
        if self.memory_tracker:
            status['memory'] = self.memory_tracker.get_stats()
        
        return status


@dataclass
class LatencyTracker:
    """Track latency statistics."""
    
    latencies: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    def record(self, latency_ms: float) -> None:
        """Record a latency measurement."""
        self.latencies.append(latency_ms)
    
    def get_stats(self) -> Dict[str, float]:
        """Get latency statistics."""
        if not self.latencies:
            return {'mean': 0, 'p50': 0, 'p95': 0, 'p99': 0}
        
        latencies = sorted(self.latencies)
        n = len(latencies)
        
        return {
            'mean': np.mean(latencies),
            'p50': latencies[n // 2],
            'p95': latencies[int(n * 0.95)] if n > 20 else latencies[-1],
            'p99': latencies[int(n * 0.99)] if n > 100 else latencies[-1]
        }


class MemoryTracker:
    """Track memory usage."""
    
    def __init__(self):
        try:
            import psutil
            self.psutil = psutil
            self.process = psutil.Process()
        except ImportError:
            self.psutil = None
            self.process = None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        if not self.psutil:
            return {'error': 'psutil not installed'}
        
        mem_info = self.process.memory_info()
        return {
            'rss_mb': mem_info.rss / 1024 / 1024,
            'vms_mb': mem_info.vms / 1024 / 1024,
            'percent': self.process.memory_percent()
        }