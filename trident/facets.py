"""Facet utilities for TRIDENT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence


@dataclass(frozen=True)
class Facet:
    """Represents a reasoning requirement extracted from a query."""

    facet_id: str
    facet_type: str
    template: Dict[str, Any]
    weight: float = 1.0

    def signature(self) -> str:
        """Return a normalized signature used for caches and logging."""

        tpl_items = sorted(self.template.items())
        tpl_str = ",".join(f"{k}={v}" for k, v in tpl_items)
        return f"{self.facet_type}:{tpl_str}"


def ensure_unique_facets(facets: Iterable[Facet]) -> List[Facet]:
    """Deduplicate facets by identifier while preserving order."""

    seen = set()
    ordered: List[Facet] = []
    for facet in facets:
        if facet.facet_id in seen:
            continue
        ordered.append(facet)
        seen.add(facet.facet_id)
    return ordered


def facet_ids(facets: Sequence[Facet]) -> List[str]:
    """Convenience helper returning the facet identifiers."""

    return [facet.facet_id for facet in facets]
