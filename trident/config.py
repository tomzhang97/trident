"""Configuration dataclasses for TRIDENT pipelines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional


@dataclass
class CalibrationConfig:
    """Settings for score calibration."""

    version: str = "dev"
    facet_bins: Optional[Dict[str, List[float]]] = None
    reliability_table_size: int = 20


@dataclass
class FacetConfig:
    """Per-facet statistical controls."""

    alpha: float = 0.01
    max_tests: int = 10
    prefilter_tests: int = 0
    fallback_scale: float = 0.5


@dataclass
class SafeCoverConfig:
    """Settings for the RC-MCFC Safe-Cover algorithm."""

    per_facet: Dict[str, FacetConfig]
    token_cap: Optional[int] = None
    dual_tolerance: float = 1e-6


@dataclass
class ParetoConfig:
    """Settings for Pareto-Knapsack mode."""

    budget: int
    relaxed_alpha: float = 0.05
    weight_default: float = 1.0


@dataclass
class QueryConfig:
    """Bundle of inputs for running the pipeline."""

    query: str
    facets: Iterable["Facet"]
    mode: str = "safe_cover"
    safe_cover: Optional[SafeCoverConfig] = None
    pareto: Optional[ParetoConfig] = None
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    retrieval_params: Dict[str, object] = field(default_factory=dict)
