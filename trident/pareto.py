"""Pareto-Knapsack mode utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

from .candidates import Passage
from .facets import Facet

ScoreFn = Callable[[Passage, Facet], float]
BucketFn = Callable[[Passage, Facet], str]


@dataclass
class ParetoResult:
    selected_passages: List[Passage]
    achieved_utility: float
    total_cost: int
    trace: List[Tuple[str, float, int]]


class ParetoKnapsack:
    """Lazy greedy optimizer for relaxed coverage utility."""

    def __init__(
        self,
        budget: int,
        relaxed_alpha: float,
        bucket_fn: Optional[BucketFn] = None,
    ) -> None:
        self.budget = budget
        self.relaxed_alpha = relaxed_alpha
        self.bucket_fn = bucket_fn or (lambda _p, facet: facet.facet_type)

    def run(
        self,
        facets: Iterable[Facet],
        passages: Iterable[Passage],
        score_fn: ScoreFn,
        pvalue_fn: Callable[[float, str], float],
    ) -> ParetoResult:
        facets = list(facets)
        passages = list(passages)
        uncovered: Set[str] = {facet.facet_id for facet in facets}
        covered: Set[str] = set()
        selection: List[Passage] = []
        spent = 0
        trace: List[Tuple[str, float, int]] = []

        def marginal_gain(passage: Passage) -> Tuple[float, float]:
            newly_cover = 0
            for facet in facets:
                if facet.facet_id in covered:
                    continue
                bucket = self.bucket_fn(passage, facet)
                score = score_fn(passage, facet)
                p_value = pvalue_fn(score, bucket)
                if p_value <= self.relaxed_alpha:
                    newly_cover += facet.weight
            return newly_cover, passage.cost

        while spent < self.budget and uncovered:
            best_passage = None
            best_gain = 0.0
            best_cost = 1
            for passage in passages:
                if passage in selection:
                    continue
                gain, cost = marginal_gain(passage)
                if cost == 0:
                    continue
                if gain / cost > best_gain / best_cost:
                    best_passage = passage
                    best_gain = gain
                    best_cost = cost
            if not best_passage or spent + best_passage.cost > self.budget or best_gain <= 0:
                break
            selection.append(best_passage)
            spent += best_passage.cost
            trace.append((best_passage.pid, best_gain, spent))
            for facet in facets:
                if facet.facet_id in covered:
                    continue
                bucket = self.bucket_fn(best_passage, facet)
                score = score_fn(best_passage, facet)
                p_value = pvalue_fn(score, bucket)
                if p_value <= self.relaxed_alpha:
                    covered.add(facet.facet_id)
                    uncovered.discard(facet.facet_id)
        utility = sum(facet.weight for facet in facets if facet.facet_id in covered)
        return ParetoResult(
            selected_passages=selection,
            achieved_utility=utility,
            total_cost=spent,
            trace=trace,
        )
