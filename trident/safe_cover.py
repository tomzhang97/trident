"""Enhanced Safe-Cover implementation with RC-MCFC algorithm."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any

import numpy as np

from .candidates import Passage
from .facets import Facet
from .config import SafeCoverConfig


@dataclass
class CoverageCertificate:
    """Certificate for facet coverage."""
    facet_id: str
    passage_id: str
    alpha_bar: float
    p_value: float
    timestamp: float = field(default_factory=time.time)
    calibrator_version: str = "v1.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'facet_id': self.facet_id,
            'passage_id': self.passage_id,
            'alpha_bar': self.alpha_bar,
            'p_value': self.p_value,
            'timestamp': self.timestamp,
            'calibrator_version': self.calibrator_version
        }


@dataclass
class SafeCoverResult:
    """Result from Safe-Cover algorithm."""
    selected_passages: List[Passage]
    certificates: List[CoverageCertificate]
    covered_facets: List[str]
    uncovered_facets: List[str]
    dual_lower_bound: float
    abstained: bool
    coverage_map: Dict[str, List[str]]  # passage_id -> covered facet_ids
    total_cost: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'selected_passages': [p.__dict__ for p in self.selected_passages],
            'certificates': [c.to_dict() for c in self.certificates],
            'covered_facets': self.covered_facets,
            'uncovered_facets': self.uncovered_facets,
            'dual_lower_bound': self.dual_lower_bound,
            'abstained': self.abstained,
            'total_cost': self.total_cost
        }


class SafeCoverAlgorithm:
    """Risk-Controlled Min-Cost Facet Cover (RC-MCFC) algorithm."""
    
    def __init__(self, config: SafeCoverConfig, calibrator: Any):
        self.config = config
        self.calibrator = calibrator
        self.fallback_active = False
        self.fallback_scale = config.per_facet_alpha * 0.5  # Conservative fallback
    
    def run(
        self,
        facets: List[Facet],
        passages: List[Passage],
        p_values: Dict[Tuple[str, str], float]
    ) -> SafeCoverResult:
        """
        Run RC-MCFC algorithm with certificates.
        
        This implements the core Safe-Cover algorithm from the TRIDENT spec:
        1. Build fixed coverage sets using Bonferroni thresholds
        2. Run greedy set cover with cost-effectiveness
        3. Generate certificates for covered facets
        4. Check dual lower bound for early abstention
        """
        
        # Step 1: Build coverage sets with Bonferroni correction
        coverage_sets = self._build_coverage_sets(facets, passages, p_values)
        
        # Step 2: Run greedy set cover
        selected, certificates, coverage_map = self._greedy_cover(
            facets, passages, coverage_sets, p_values
        )
        
        # Step 3: Identify covered/uncovered facets
        covered_facets = set()
        for facet_list in coverage_map.values():
            covered_facets.update(facet_list)
        
        uncovered_facets = [f.facet_id for f in facets if f.facet_id not in covered_facets]
        
        # Step 4: Compute dual lower bound
        dual_lb = self._compute_dual_lower_bound(facets, passages, coverage_sets)
        
        # Step 5: Check abstention conditions
        total_cost = sum(p.cost for p in selected)
        abstained = False
        
        # Abstain if we have uncovered facets
        if uncovered_facets or len(selected) == 0:
            abstained = True
        
        # Abstain if dual LB exceeds budget (early abstention)
        if self.config.early_abstain and self.config.token_cap:
            if dual_lb > self.config.token_cap:
                abstained = True
        
        return SafeCoverResult(
            selected_passages=selected,
            certificates=certificates,
            covered_facets=list(covered_facets),
            uncovered_facets=uncovered_facets,
            dual_lower_bound=dual_lb,
            abstained=abstained,
            coverage_map=coverage_map,
            total_cost=total_cost
        )
    
    def _build_coverage_sets(
        self,
        facets: List[Facet],
        passages: List[Passage],
        p_values: Dict[Tuple[str, str], float]
    ) -> Dict[str, Set[str]]:
        """
        Build fixed coverage sets using Bonferroni thresholds.
        
        For each passage, determine which facets it covers based on
        calibrated p-values and Bonferroni-corrected thresholds.
        """
        coverage_sets = {p.pid: set() for p in passages}
        
        for facet in facets:
            # Get facet-specific configuration
            if facet.facet_id in self.config.per_facet_configs:
                facet_config = self.config.per_facet_configs[facet.facet_id]
            else:
                # Use default configuration
                facet_config = type('obj', (object,), {
                    'alpha': self.config.per_facet_alpha,
                    'max_tests': 10
                })()
            
            # Compute Bonferroni threshold
            alpha_bar = facet_config.alpha / max(facet_config.max_tests, 1)

            # alpha_bar = max(alpha_bar, 0.1)  # Floor at 0.1
            
            # Apply fallback if active
            if self.fallback_active:
                alpha_bar *= self.config.per_facet_alpha * self.config.per_facet_configs.get(
                    facet.facet_id, type('obj', (object,), {'fallback_scale': 0.5})()
                ).fallback_scale
            
            # Test coverage for each passage (up to max_tests)
            tests_performed = 0
            for passage in passages:
                if tests_performed >= facet_config.max_tests:
                    break
                
                key = (passage.pid, facet.facet_id)
                if key in p_values:
                    tests_performed += 1
                    
                    if p_values[key] <= alpha_bar:
                        coverage_sets[passage.pid].add(facet.facet_id)
        
        return coverage_sets
    
    def _greedy_cover(
        self,
        facets: List[Facet],
        passages: List[Passage],
        coverage_sets: Dict[str, Set[str]],
        p_values: Dict[Tuple[str, str], float]
    ) -> Tuple[List[Passage], List[CoverageCertificate], Dict[str, List[str]]]:
        """
        Greedy set cover with cost-effectiveness and tie-breaking.
        
        Implements the greedy algorithm that achieves O(log |F|) approximation.
        """
        selected_passages = []
        certificates = []
        coverage_map = {}
        
        uncovered = {f.facet_id for f in facets}
        remaining_passages = {p.pid: p for p in passages}
        
        while uncovered and remaining_passages:
            # Find most cost-effective passage
            best_passage = None
            best_score = (-float('inf'), float('inf'), float('inf'))
            
            for pid, passage in remaining_passages.items():
                # Calculate newly covered facets
                newly_covered = coverage_sets[pid] & uncovered
                
                if not newly_covered:
                    continue
                
                # Cost-effectiveness ratio
                effectiveness = len(newly_covered) / max(passage.cost, 1)
                
                # Mean p-value for tie-breaking
                mean_p = np.mean([
                    p_values.get((pid, fid), 1.0)
                    for fid in newly_covered
                ])
                
                # Score tuple: (effectiveness, -cost, -mean_p)
                score = (effectiveness, -passage.cost, -mean_p)
                
                if score > best_score:
                    best_score = score
                    best_passage = passage
            
            # No more passages can cover uncovered facets
            if best_passage is None:
                break
            
            # Add best passage to selection
            selected_passages.append(best_passage)
            newly_covered = coverage_sets[best_passage.pid] & uncovered
            coverage_map[best_passage.pid] = list(newly_covered)
            
            # Generate certificates for newly covered facets
            for facet_id in newly_covered:
                # Find the facet object
                facet = next(f for f in facets if f.facet_id == facet_id)
                
                # Get configuration
                if facet_id in self.config.per_facet_configs:
                    facet_config = self.config.per_facet_configs[facet_id]
                else:
                    facet_config = type('obj', (object,), {
                        'alpha': self.config.per_facet_alpha,
                        'max_tests': 10
                    })()
                
                alpha_bar = facet_config.alpha / max(facet_config.max_tests, 1)
                
                certificate = CoverageCertificate(
                    facet_id=facet_id,
                    passage_id=best_passage.pid,
                    alpha_bar=alpha_bar,
                    p_value=p_values.get((best_passage.pid, facet_id), 0.0),
                    calibrator_version=self.calibrator.version
                )
                certificates.append(certificate)
            
            # Update uncovered set
            uncovered -= newly_covered
            
            # Remove selected passage from remaining
            del remaining_passages[best_passage.pid]
        
        return selected_passages, certificates, coverage_map
    
    def _compute_dual_lower_bound(
        self,
        facets: List[Facet],
        passages: List[Passage],
        coverage_sets: Dict[str, Set[str]]
    ) -> float:
        """
        Compute dual lower bound using primal-dual relaxation.
        
        This provides a valid lower bound on the optimal cost,
        enabling early abstention when the budget is insufficient.
        """
        # Initialize dual variables
        dual_vars = {f.facet_id: 0.0 for f in facets}
        uncovered = set(f.facet_id for f in facets)
        tight_passages = set()
        
        epsilon = self.config.dual_tolerance
        max_iterations = 100
        
        for _ in range(max_iterations):
            if not uncovered:
                break
            
            # Find minimum slack
            min_slack = float('inf')
            min_passage = None
            
            for passage in passages:
                if passage.pid in tight_passages:
                    continue
                
                # Calculate dual sum for this passage
                dual_sum = sum(
                    dual_vars[fid]
                    for fid in coverage_sets[passage.pid]
                    if fid in dual_vars
                )
                
                slack = passage.cost - dual_sum
                
                if slack < min_slack and coverage_sets[passage.pid] & uncovered:
                    min_slack = slack
                    min_passage = passage
            
            if min_passage is None or min_slack == float('inf'):
                # No feasible solution
                return float('inf')
            
            # Raise dual variables uniformly for uncovered facets
            # that would be covered by min_passage
            facets_to_raise = coverage_sets[min_passage.pid] & uncovered
            if facets_to_raise:
                raise_amount = min_slack / len(facets_to_raise)
                
                for fid in facets_to_raise:
                    dual_vars[fid] += raise_amount
                
                # Mark passage as tight
                tight_passages.add(min_passage.pid)
                
                # Update uncovered
                uncovered -= coverage_sets[min_passage.pid]
        
        # Return sum of dual variables as lower bound
        return sum(dual_vars.values())
    
    def apply_fallback(self) -> None:
        """Apply fallback thresholds when drift is detected."""
        self.fallback_active = True
    
    def reset_fallback(self) -> None:
        """Reset fallback mode."""
        self.fallback_active = False
    
    def get_spurious_facet_prepass(
        self,
        facets: List[Facet],
        passages: List[Passage],
        p_values: Dict[Tuple[str, str], float],
        spurious_alpha: float = 1e-4,
        max_tests_per_facet: int = 3
    ) -> List[Facet]:
        """
        Pre-pass to filter spurious facets with very strict threshold.
        
        This helps remove noisy facets from the mining stage.
        """
        valid_facets = []
        
        for facet in facets:
            # Test up to max_tests_per_facet passages
            tests = 0
            found_support = False
            
            for passage in passages:
                if tests >= max_tests_per_facet:
                    break
                
                key = (passage.pid, facet.facet_id)
                if key in p_values:
                    tests += 1
                    if p_values[key] <= spurious_alpha:
                        found_support = True
                        break
            
            if found_support:
                valid_facets.append(facet)
        
        return valid_facets