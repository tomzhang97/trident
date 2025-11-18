#!/usr/bin/env python3
"""
Simple example demonstrating TRIDENT without requiring large models.

This example uses mock scoring and a simple dataset to show the
core TRIDENT algorithms in action.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from trident.config import TridentConfig, SafeCoverConfig, ParetoConfig
from trident.facets import Facet, FacetType
from trident.candidates import Passage
from trident.calibration import ReliabilityCalibrator
from trident.safe_cover import SafeCoverAlgorithm
from trident.pareto import ParetoKnapsackOptimizer


def create_mock_dataset() -> Tuple[str, List[Facet], List[Passage]]:
    """Create a mock multi-hop question with passages."""
    
    # Question requiring multi-hop reasoning
    question = "Which university did the inventor of the telephone attend?"
    
    # Facets extracted from the question
    facets = [
        Facet(
            facet_id="f1",
            facet_type=FacetType.ENTITY,
            template={"mention": "inventor of the telephone"},
            weight=1.0
        ),
        Facet(
            facet_id="f2",
            facet_type=FacetType.RELATION,
            template={
                "subject": "Alexander Graham Bell",
                "predicate": "invented",
                "object": "telephone"
            },
            weight=1.0
        ),
        Facet(
            facet_id="f3",
            facet_type=FacetType.BRIDGE,
            template={
                "entity1": "Alexander Graham Bell",
                "entity2": "university",
                "relation": "attended"
            },
            weight=1.5  # Higher weight for bridge facet
        )
    ]
    
    # Candidate passages
    passages = [
        Passage(
            pid="p1",
            text="Alexander Graham Bell is credited with inventing the telephone in 1876.",
            cost=20,
            metadata={"source": "wikipedia"}
        ),
        Passage(
            pid="p2",
            text="Bell attended the University of Edinburgh and later taught at Boston University.",
            cost=25,
            metadata={"source": "biography"}
        ),
        Passage(
            pid="p3",
            text="The telephone revolutionized communication in the late 19th century.",
            cost=15,
            metadata={"source": "history"}
        ),
        Passage(
            pid="p4",
            text="Bell was born in Scotland and studied at Edinburgh before moving to Canada.",
            cost=22,
            metadata={"source": "biography"}
        ),
        Passage(
            pid="p5",
            text="Many inventors worked on voice transmission, but Bell received the first patent.",
            cost=18,
            metadata={"source": "patents"}
        )
    ]
    
    return question, facets, passages


def mock_nli_scores() -> Dict[Tuple[str, str], float]:
    """
    Create mock NLI scores for passage-facet pairs.
    
    In practice, these would come from a cross-encoder model.
    """
    scores = {
        # p1 covers f1 and f2 (identifies Bell as inventor)
        ("p1", "f1"): 0.95,  # Strong evidence
        ("p1", "f2"): 0.92,  # Strong evidence
        ("p1", "f3"): 0.10,  # No university info
        
        # p2 covers f3 (Bell's education)
        ("p2", "f1"): 0.20,  # No invention mentioned
        ("p2", "f2"): 0.15,  # No telephone mentioned
        ("p2", "f3"): 0.88,  # Has university info
        
        # p3 has limited coverage
        ("p3", "f1"): 0.30,  # Mentions telephone but not inventor
        ("p3", "f2"): 0.25,
        ("p3", "f3"): 0.05,
        
        # p4 partially covers f3
        ("p4", "f1"): 0.25,
        ("p4", "f2"): 0.20,
        ("p4", "f3"): 0.75,  # Mentions Edinburgh
        
        # p5 has weak coverage
        ("p5", "f1"): 0.40,  # Mentions inventors generically
        ("p5", "f2"): 0.35,  # Mentions Bell got patent
        ("p5", "f3"): 0.10,
    }
    
    # Convert to p-values (1 - score for this mock)
    p_values = {k: 1.0 - v for k, v in scores.items()}
    
    return p_values


def run_safe_cover_demo(
    facets: List[Facet],
    passages: List[Passage],
    p_values: Dict[Tuple[str, str], float]
) -> None:
    """Demonstrate Safe-Cover mode."""
    
    print("\n" + "=" * 60)
    print("SAFE-COVER MODE (RC-MCFC)")
    print("=" * 60)
    
    # Create configuration
    config = SafeCoverConfig(
        per_facet_alpha=0.05,  # 5% FWER per facet
        token_cap=50,  # Token budget
        early_abstain=True
    )
    
    # Create calibrator (using mock calibration)
    calibrator = ReliabilityCalibrator(version="mock_v1")
    
    # Run Safe-Cover algorithm
    algo = SafeCoverAlgorithm(config=config, calibrator=calibrator)
    result = algo.run(facets, passages, p_values)
    
    print(f"\nResults:")
    print(f"  Selected passages: {len(result.selected_passages)}")
    for p in result.selected_passages:
        print(f"    - {p.pid}: {p.text[:60]}... (cost: {p.cost})")
    
    print(f"\n  Covered facets: {result.covered_facets}")
    print(f"  Uncovered facets: {result.uncovered_facets}")
    
    print(f"\n  Certificates issued: {len(result.certificates)}")
    for cert in result.certificates:
        print(f"    - Facet {cert.facet_id} covered by {cert.passage_id}")
        print(f"      α̅={cert.alpha_bar:.3f}, p-value={cert.p_value:.3f}")
    
    print(f"\n  Dual lower bound: {result.dual_lower_bound:.1f}")
    print(f"  Total cost: {result.total_cost}")
    print(f"  Abstained: {result.abstained}")
    
    if result.abstained:
        if result.uncovered_facets:
            print(f"  Reason: Unable to cover facets {result.uncovered_facets}")
        else:
            print(f"  Reason: Budget insufficient (LB={result.dual_lower_bound} > cap={config.token_cap})")


def run_pareto_demo(
    facets: List[Facet],
    passages: List[Passage],
    p_values: Dict[Tuple[str, str], float]
) -> None:
    """Demonstrate Pareto-Knapsack mode."""
    
    print("\n" + "=" * 60)
    print("PARETO-KNAPSACK MODE")
    print("=" * 60)
    
    # Create configuration
    config = ParetoConfig(
        budget=60,  # Token budget
        relaxed_alpha=0.10,  # More relaxed threshold
        weight_default=1.0
    )
    
    # Run Pareto optimizer
    optimizer = ParetoKnapsackOptimizer(config)
    result = optimizer.optimize(facets, passages, p_values)
    
    print(f"\nResults:")
    print(f"  Selected passages: {len(result.selected_passages)}")
    for p in result.selected_passages:
        print(f"    - {p.pid}: {p.text[:60]}... (cost: {p.cost})")
    
    print(f"\n  Covered facets: {result.covered_facets}")
    print(f"  Uncovered facets: {result.uncovered_facets}")
    
    print(f"\n  Achieved utility: {result.achieved_utility:.2f}")
    print(f"  Total cost: {result.total_cost}")
    
    # Show Pareto frontier
    if result.pareto_points:
        print(f"\n  Pareto frontier ({len(result.pareto_points)} points):")
        for utility, cost in result.pareto_points[:5]:
            print(f"    Utility={utility:.2f}, Cost={cost}")


def compare_modes(
    facets: List[Facet],
    passages: List[Passage],
    p_values: Dict[Tuple[str, str], float]
) -> None:
    """Compare Safe-Cover and Pareto modes."""
    
    print("\n" + "=" * 60)
    print("MODE COMPARISON")
    print("=" * 60)
    
    # Run Safe-Cover
    safe_config = SafeCoverConfig(per_facet_alpha=0.05, token_cap=50)
    safe_algo = SafeCoverAlgorithm(
        config=safe_config,
        calibrator=ReliabilityCalibrator()
    )
    safe_result = safe_algo.run(facets, passages, p_values)
    
    # Run Pareto
    pareto_config = ParetoConfig(budget=50, relaxed_alpha=0.10)
    pareto_opt = ParetoKnapsackOptimizer(pareto_config)
    pareto_result = pareto_opt.optimize(facets, passages, p_values)
    
    print("\n  Safe-Cover:")
    print(f"    - Coverage: {len(safe_result.covered_facets)}/{len(facets)} facets")
    print(f"    - Cost: {safe_result.total_cost} tokens")
    print(f"    - Certificates: {len(safe_result.certificates)}")
    print(f"    - Abstained: {safe_result.abstained}")
    
    print("\n  Pareto-Knapsack:")
    print(f"    - Coverage: {len(pareto_result.covered_facets)}/{len(facets)} facets")
    print(f"    - Cost: {pareto_result.total_cost} tokens")
    print(f"    - Utility: {pareto_result.achieved_utility:.2f}")
    print(f"    - No certificates (relaxed mode)")
    
    print("\n  Trade-offs:")
    print("    - Safe-Cover: Provable guarantees, may abstain")
    print("    - Pareto: Always returns answer, no certificates")


def main():
    """Run the demonstration."""
    
    print("=" * 60)
    print("TRIDENT ALGORITHM DEMONSTRATION")
    print("=" * 60)
    
    # Create mock data
    question, facets, passages = create_mock_dataset()
    
    print(f"\nQuestion: {question}")
    print(f"\nFacets extracted: {len(facets)}")
    for facet in facets:
        print(f"  - {facet.facet_id} ({facet.facet_type.value}): {facet.template}")
    
    print(f"\nCandidate passages: {len(passages)}")
    for p in passages:
        print(f"  - {p.pid}: {p.text[:50]}... (cost: {p.cost})")
    
    # Get mock scores
    p_values = mock_nli_scores()
    
    # Run demonstrations
    run_safe_cover_demo(facets, passages, p_values)
    run_pareto_demo(facets, passages, p_values)
    compare_modes(facets, passages, p_values)
    
    print("\n" + "=" * 60)
    print("Demonstration complete!")
    print("\nThis example used mock scores. In practice, you would:")
    print("1. Use a real NLI model for scoring (e.g., DeBERTa)")
    print("2. Load passages from a retrieval corpus")
    print("3. Use a calibrated scorer with real training data")
    print("4. Generate answers with an LLM using selected passages")


if __name__ == "__main__":
    main()