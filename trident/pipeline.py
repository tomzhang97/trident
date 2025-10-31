"""High-level orchestration for TRIDENT pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

from .calibration import ReliabilityCalibrator
from .candidates import Passage
from .config import FacetConfig, ParetoConfig, QueryConfig, SafeCoverConfig
from .facets import Facet, ensure_unique_facets
from .pareto import ParetoKnapsack
from .retrieval import SimpleRetriever
from .safe_cover import SafeCoverAlgorithm, SafeCoverResult


def lexical_score(passage: Passage, facet: Facet) -> float:
    """Simple lexical overlap score between passage and facet template."""

    passage_terms = set(passage.text.lower().split())
    template_terms = set(str(value).lower() for value in facet.template.values())
    if not template_terms:
        return 0.0
    overlap = len(template_terms & passage_terms)
    return overlap / max(len(template_terms), 1)


def lexical_bucket(_passage: Passage, facet: Facet) -> str:
    return facet.facet_type


@dataclass
class PipelineOutputs:
    safe_cover: Optional[SafeCoverResult]
    pareto: Optional[object]
    passages_examined: List[Passage]


class TridentPipeline:
    """Coordinates retrieval, scoring, and selection."""

    def __init__(
        self,
        retriever: SimpleRetriever,
        calibrator: Optional[ReliabilityCalibrator] = None,
    ) -> None:
        self.retriever = retriever
        self.calibrator = calibrator or ReliabilityCalibrator()

    def run(self, config: QueryConfig, top_k: int = 20) -> PipelineOutputs:
        facets = ensure_unique_facets(config.facets)
        retrieval_result = self.retriever.query(config.query, top_k=top_k)
        passages = retrieval_result.passages
        safe_cover_result: Optional[SafeCoverResult] = None
        pareto_result = None

        if config.mode in {"safe_cover", "both"}:
            safe_config = config.safe_cover or self._default_safe_config(facets)
            algo = SafeCoverAlgorithm(
                calibrator=self.calibrator,
                config=safe_config,
                score_fn=lexical_score,
                bucket_fn=lexical_bucket,
            )
            safe_cover_result = algo.run(facets, passages)

        if config.mode in {"pareto", "both"}:
            pareto_conf = config.pareto or ParetoConfig(budget=sum(p.cost for p in passages))
            optimizer = ParetoKnapsack(
                budget=pareto_conf.budget,
                relaxed_alpha=pareto_conf.relaxed_alpha,
                bucket_fn=lexical_bucket,
            )
            pareto_result = optimizer.run(
                facets=facets,
                passages=passages,
                score_fn=lexical_score,
                pvalue_fn=self.calibrator.to_pvalue,
            )

        return PipelineOutputs(
            safe_cover=safe_cover_result,
            pareto=pareto_result,
            passages_examined=passages,
        )

    def _default_safe_config(self, facets: Iterable[Facet]) -> SafeCoverConfig:
        per_facet = {
            facet.facet_id: FacetConfig(alpha=0.01, max_tests=5) for facet in facets
        }
        return SafeCoverConfig(per_facet=per_facet)
