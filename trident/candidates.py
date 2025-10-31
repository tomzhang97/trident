"""Passage candidate representation and helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional


@dataclass
class Passage:
    """Candidate passage/snippet returned by retrieval."""

    pid: str
    text: str
    cost: int
    metadata: Dict[str, object] = field(default_factory=dict)

    def short_text(self, limit: int = 120) -> str:
        """Return a truncated representation for logging."""

        if len(self.text) <= limit:
            return self.text
        return self.text[: limit - 3] + "..."


def top_k_by_score(scores: Dict[str, float], passages: Dict[str, Passage], k: int) -> List[Passage]:
    """Return the top-k passages according to the provided scores."""

    ordered_ids = sorted(scores, key=scores.get, reverse=True)[:k]
    return [passages[pid] for pid in ordered_ids if pid in passages]


def group_passages_by_field(passages: Iterable[Passage], field_name: str) -> Dict[object, List[Passage]]:
    """Group passages by a metadata field."""

    groups: Dict[object, List[Passage]] = {}
    for passage in passages:
        key = passage.metadata.get(field_name)
        groups.setdefault(key, []).append(passage)
    return groups
