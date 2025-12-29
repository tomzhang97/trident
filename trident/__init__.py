"""TRIDENT package exposing risk-controlled retrieval pipelines."""

from .config import (
    FacetConfig,
    SafeCoverConfig,
    ParetoConfig,
    CalibrationConfig,
)
from .facets import Facet
from .candidates import Passage

__all__ = [
    "CalibrationConfig",
    "Facet",
    "FacetConfig",
    "ParetoConfig",
    "Passage",
    "SafeCoverConfig",
    "TridentPipeline",
]


def __getattr__(name):  # pragma: no cover - simple lazy import hook
    if name == "TridentPipeline":
        from .pipeline import TridentPipeline

        return TridentPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
