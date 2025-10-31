"""Risk-controlled Safe-Cover implementation for TRIDENT."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

from .calibration import ReliabilityCalibrator
from .candidates import Passage
from .config import FacetConfig, SafeCoverConfig
from .facets import Facet, ensure_unique_facets

ScoreFn = Callable[[Passage, Facet], float]
BucketFn = Callable[[Passage, Facet], str]


@dataclass
class CoverageCertificate:
    facet_id: str
    passage_id: str
    alpha_bar: float
    p_value: float


@dataclass
class SafeCoverResult:
    selected_passages: List[Passage]
    certificates: List[CoverageCertificate]
    dual_lower_bound: float
    abstained: bool
    uncovered_facets: List[str] = field(default_factory=list)


class SafeCoverAlgorithm:
    """Compute RC-MCFC selections with simple certificates."""

    def __init__(
        self,
        calibrator: ReliabilityCalibrator,
        config: SafeCoverConfig,
        score_fn: ScoreFn,
        bucket_fn: Optional[BucketFn] = None,
    ) -> None:
        self.calibrator = calibrator
        self.config = config
        self.score_fn = score_fn
        self.bucket_fn = bucket_fn or (lambda _p, facet: facet.facet_type)

    def run(self, facets: Iterable[Facet], passages: Iterable[Passage]) -> SafeCoverResult:
        unique_facets = ensure_unique_facets(facets)
        coverage_sets: Dict[str, Set[str]] = {p.pid: set() for p in passages}
        coverage_scores: Dict[Tuple[str, str], float] = {}
        certificates: List[CoverageCertificate] = []

        # Build coverage sets respecting per-facet test budgets.
        for facet in unique_facets:
            config = self.config.per_facet.get(facet.facet_id)
            if not config:
                raise KeyError(f"Missing config for facet {facet.facet_id}")
            alpha_bar = config.alpha / max(config.max_tests, 1)
            tested = 0
            for passage in passages:
                if tested >= config.max_tests:
                    break
                bucket_key = self.bucket_fn(passage, facet)
                score = self.score_fn(passage, facet)
                p_value = self.calibrator.to_pvalue(score, bucket_key)
                coverage_scores[(passage.pid, facet.facet_id)] = p_value
                tested += 1
                if p_value <= alpha_bar:
                    coverage_sets[passage.pid].add(facet.facet_id)
            if tested == 0:
                coverage_scores[("", facet.facet_id)] = 1.0

        uncovered = {facet.facet_id for facet in unique_facets}
        covered: Set[str] = set()
        selected: List[Passage] = []
        remaining_passages = list(passages)

        def greedy_priority(passage: Passage) -> Tuple[float, float, float]:
            new_facets = coverage_sets[passage.pid] - covered
            if not new_facets:
                return (0.0, float(passage.cost), 0.0)
            mean_p = sum(
                coverage_scores[(passage.pid, facet_id)] for facet_id in new_facets
            ) / len(new_facets)
            return (
                len(new_facets) / max(passage.cost, 1),
                -float(passage.cost),
                -mean_p,
            )

        while uncovered and remaining_passages:
            best_passage = max(remaining_passages, key=lambda p: greedy_priority(p))
            priority = greedy_priority(best_passage)
            if priority[0] <= 0:
                break
            selected.append(best_passage)
            newly_covered = coverage_sets[best_passage.pid] & uncovered
            for facet_id in newly_covered:
                config = self.config.per_facet[facet_id]
                alpha_bar = config.alpha / max(config.max_tests, 1)
                certificates.append(
                    CoverageCertificate(
                        facet_id=facet_id,
                        passage_id=best_passage.pid,
                        alpha_bar=alpha_bar,
                        p_value=coverage_scores[(best_passage.pid, facet_id)],
                    )
            )
            uncovered -= newly_covered
            covered |= newly_covered
            remaining_passages = [p for p in remaining_passages if p.pid != best_passage.pid]

        dual_lb = self._lower_bound(unique_facets, coverage_sets, remaining_passages + selected)
        token_cap = self.config.token_cap
        abstained = False
        uncovered_list = sorted(uncovered)
        if uncovered_list:
            abstained = True
        elif token_cap is not None:
            total_cost = sum(p.cost for p in selected)
            if dual_lb - self.config.dual_tolerance > token_cap:
                abstained = True
        return SafeCoverResult(
            selected_passages=selected,
            certificates=certificates,
            dual_lower_bound=dual_lb,
            abstained=abstained,
            uncovered_facets=uncovered_list,
        )

    def _lower_bound(
        self,
        facets: List[Facet],
        coverage_sets: Dict[str, Set[str]],
        passages: List[Passage],
    ) -> float:
        """Compute a conservative lower bound using per-facet cheapest coverage."""

        costs_by_facet: Dict[str, float] = {}
        for facet in facets:
            cheapest = None
            for passage in passages:
                if facet.facet_id in coverage_sets.get(passage.pid, set()):
                    cheapest = passage.cost if cheapest is None else min(cheapest, passage.cost)
            if cheapest is None:
                return float("inf")
            costs_by_facet[facet.facet_id] = cheapest
        if not costs_by_facet:
            return 0.0
        return max(costs_by_facet.values())
