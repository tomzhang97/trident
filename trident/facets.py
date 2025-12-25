#!/usr/bin/env python3
"""
trident/facets.py

Robust facet representation + mining with clean, grammatical NLI hypotheses,
AND improved entity extraction for disjunction/comparison questions like:
"Which magazine was started first Arthur's Magazine or First for Women?"

Fixes:
- No raw question-fragment concatenation into hypotheses
- Disjunction ("A or B") heuristic extraction to avoid missing entities
- Filters junk entities like "first"
- Normalizes relation predicates into English
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

# Optional deps
try:
    import spacy
except Exception:
    spacy = None

try:
    from transformers import pipeline
except Exception:
    pipeline = None


# -------------------------
# Text normalization helpers
# -------------------------

_PUNCT_FIX_RE = re.compile(r"\s+([,.;:?!])")
_MULTI_SPACE_RE = re.compile(r"\s{2,}")
_APOS_S_RE = re.compile(r"\s+'s\b", re.IGNORECASE)
_BAD_CHARS_RE = re.compile(r"[\u200b\u200c\u200d\uFEFF]")  # zero-width chars


def _clean_text(s: str) -> str:
    s = (s or "").strip()
    s = _BAD_CHARS_RE.sub("", s)
    s = _APOS_S_RE.sub("'s", s)
    s = _PUNCT_FIX_RE.sub(r"\1", s)
    s = _MULTI_SPACE_RE.sub(" ", s)
    return s


def _ensure_sentence(s: str) -> str:
    s = _clean_text(s)
    if not s:
        return s
    if s[-1] not in ".?!":
        s += "."
    return s


def _safe_phrase(s: str) -> str:
    s = _clean_text(s)
    s = s.strip("`\"' ")
    return _clean_text(s)


def _normalize_relation_predicate(pred: str) -> str:
    pred = _clean_text(pred).strip()
    if not pred:
        return "is related to"

    pred_lower = pred.lower().replace("-", "_")
    mapping = {
        "related_to": "is related to",
        "is_related_to": "is related to",
        "part_of": "is part of",
        "located_in": "is located in",
        "location": "is located in",
        "born_in": "was born in",
        "birthplace": "was born in",
        "died_in": "died in",
        "death_place": "died in",
        "member_of": "is a member of",
        "membership": "is a member of",
        "capital_of": "is the capital of",
        "capital": "is the capital of",
        "authored": "authored",
        "wrote": "wrote",
        "created": "created",
        "directed": "directed",
        "starred_in": "appeared in",
        "appeared_in": "appeared in",
        "team": "played for",
        "played_for": "played for",
        "temporal_relation": "is associated with",
    }
    if pred_lower in mapping:
        return mapping[pred_lower]

    pred = pred.replace("_", " ").strip()
    if re.match(r"^(is|was|were|are|has|had|have|authored|wrote|created|directed|played|appeared|died|was born)\b",
                pred.lower()):
        return pred
    if len(pred.split()) <= 2:
        return "is related to"
    return pred


def _looks_like_junk_entity(mention: str) -> bool:
    m = _safe_phrase(mention)
    if not m:
        return True

    ml = m.lower()
    bad_tokens = {
        "first", "second", "third", "fourth", "fifth",
        "which", "what", "who", "whom", "whose",
        "when", "where", "why", "how",
        "many", "much",
        "one", "two", "three",
        "this", "that", "these", "those",
        "him", "her", "his", "hers", "its", "their",
        "the", "a", "an",
        "and", "or",
    }
    if ml in bad_tokens:
        return True
    if len(m.split()) == 1 and len(m) <= 3:
        return True
    if re.fullmatch(r"\d+(\.\d+)?", ml):
        return True
    return False


def _dedupe_strings(xs: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for x in xs:
        x = _safe_phrase(x)
        if not x:
            continue
        key = x.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(x)
    return out


# -------------------------
# Facet definitions
# -------------------------

class FacetType(Enum):
    ENTITY = "ENTITY"
    RELATION = "RELATION"
    TEMPORAL = "TEMPORAL"
    NUMERIC = "NUMERIC"
    BRIDGE_HOP1 = "BRIDGE_HOP1"
    BRIDGE_HOP2 = "BRIDGE_HOP2"
    BRIDGE = "BRIDGE"
    COMPARISON = "COMPARISON"
    CAUSAL = "CAUSAL"
    PROCEDURAL = "PROCEDURAL"

    @classmethod
    def core_types(cls) -> list:
        return [cls.ENTITY, cls.RELATION, cls.TEMPORAL, cls.NUMERIC, cls.BRIDGE_HOP1, cls.BRIDGE_HOP2]


@dataclass(frozen=True)
class Facet:
    facet_id: str
    facet_type: FacetType
    template: Dict[str, Any]
    weight: float = 1.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})
        ft = object.__getattribute__(self, "facet_type")
        if isinstance(ft, str):
            object.__setattr__(self, "facet_type", FacetType(ft))

    def to_hypothesis(self) -> str:
        ft = self.facet_type

        if ft == FacetType.ENTITY:
            mention = _safe_phrase(self.template.get("mention", ""))
            if _looks_like_junk_entity(mention):
                return "The passage mentions the relevant entity."
            return _ensure_sentence(f'The passage contains the exact phrase "{mention}".')

        if ft == FacetType.RELATION:
            subj = _safe_phrase(self.template.get("subject", ""))
            obj = _safe_phrase(self.template.get("object", ""))
            pred = _normalize_relation_predicate(self.template.get("predicate", ""))

            if not subj or not obj:
                return "The passage states a relevant relationship between entities."

            pred_clean = _clean_text(pred)
            if not re.match(r"^(is|was|were|are|has|had|have|authored|wrote|created|directed|played|appeared|died|was born)\b",
                            pred_clean.lower()):
                pred_clean = "is related to"
            return _ensure_sentence(f"{subj} {pred_clean} {obj}")

        if ft == FacetType.TEMPORAL:
            event = _safe_phrase(self.template.get("event", ""))
            time = _safe_phrase(self.template.get("time", ""))
            if event and time:
                return _ensure_sentence(f"{event} occurred in {time}")
            if time:
                return _ensure_sentence(f"The event occurred in {time}")
            if event:
                return _ensure_sentence(f"{event} occurred at a specific time")
            return "The passage states a relevant time-related fact."

        if ft == FacetType.NUMERIC:
            entity = _safe_phrase(self.template.get("entity", "")) or _safe_phrase(self.template.get("mention", ""))
            value = _safe_phrase(self.template.get("value", ""))
            unit = _safe_phrase(self.template.get("unit", ""))
            attr = _safe_phrase(self.template.get("attribute", ""))
            if entity and attr and value:
                v = f"{value} {unit}".strip()
                return _ensure_sentence(f"{entity} has {attr} of {v}")
            if entity and value:
                v = f"{value} {unit}".strip()
                return _ensure_sentence(f"{entity} has value {v}")
            if value:
                v = f"{value} {unit}".strip()
                return _ensure_sentence(f"The passage states the value {v}")
            return "The passage states a relevant numeric fact."

        if ft == FacetType.BRIDGE_HOP1:
            e1 = _safe_phrase(self.template.get("entity1", ""))
            eb = _safe_phrase(self.template.get("bridge_entity", ""))
            rel = _normalize_relation_predicate(self.template.get("relation", ""))
            if not e1 or not eb:
                return "The passage supports the first hop of the multi-hop reasoning chain."
            rel_clean = _clean_text(rel)
            if not re.match(r"^(is|was|were|are|has|had|have|authored|wrote|created|directed|played|appeared|died|was born)\b",
                            rel_clean.lower()):
                rel_clean = "is related to"
            return _ensure_sentence(f"{e1} {rel_clean} {eb}")

        if ft == FacetType.BRIDGE_HOP2:
            eb = _safe_phrase(self.template.get("bridge_entity", ""))
            e2 = _safe_phrase(self.template.get("entity2", ""))
            rel = _normalize_relation_predicate(self.template.get("relation", ""))
            if not eb or not e2:
                return "The passage supports the second hop of the multi-hop reasoning chain."
            rel_clean = _clean_text(rel)
            if not re.match(r"^(is|was|were|are|has|had|have|authored|wrote|created|directed|played|appeared|died|was born)\b",
                            rel_clean.lower()):
                rel_clean = "is related to"
            return _ensure_sentence(f"{eb} {rel_clean} {e2}")

        if ft == FacetType.BRIDGE:
            e1 = _safe_phrase(self.template.get("entity1", ""))
            e2 = _safe_phrase(self.template.get("entity2", ""))
            rel = _safe_phrase(self.template.get("relation", ""))
            if e1 and e2 and rel:
                return _ensure_sentence(f"{e1} is connected to {e2} through {rel}")
            return "The passage supports a multi-hop relationship."

        if ft == FacetType.COMPARISON:
            e1 = _safe_phrase(self.template.get("entity1", ""))
            e2 = _safe_phrase(self.template.get("entity2", ""))
            attr = _safe_phrase(self.template.get("attribute", ""))
            if e1 and e2 and attr:
                return _ensure_sentence(f"{e1} and {e2} are compared by {attr}")
            return "The passage supports a relevant comparison."

        if ft == FacetType.CAUSAL:
            cause = _safe_phrase(self.template.get("cause", ""))
            effect = _safe_phrase(self.template.get("effect", ""))
            if cause and effect:
                return _ensure_sentence(f"{cause} causes {effect}")
            return "The passage supports a causal relationship."

        if ft == FacetType.PROCEDURAL:
            steps = self.template.get("steps", [])
            if isinstance(steps, list) and steps:
                steps_clean = [_safe_phrase(x) for x in steps if _safe_phrase(x)]
                if steps_clean:
                    return _ensure_sentence(f"The process involves: {', '.join(steps_clean)}")
            return "The passage supports a relevant procedure."

        return _ensure_sentence(str(self.template))

    def get_keywords(self) -> List[str]:
        keywords: Set[str] = set()

        def add_words(text: str):
            text = _clean_text(text)
            for w in re.findall(r"[A-Za-z0-9']+", text):
                wl = w.lower()
                if len(wl) <= 2:
                    continue
                if wl in {"the", "and", "or", "for", "with", "from", "that", "this", "what", "which", "who"}:
                    continue
                keywords.add(w)

        for v in self.template.values():
            if isinstance(v, str):
                add_words(v)
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, str):
                        add_words(item)

        return sorted(keywords)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "facet_id": self.facet_id,
            "facet_type": self.facet_type.value,
            "template": self.template,
            "weight": self.weight,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Facet":
        return cls(
            facet_id=data["facet_id"],
            facet_type=FacetType(data["facet_type"]),
            template=data["template"],
            weight=data.get("weight", 1.0),
            metadata=data.get("metadata"),
        )


class FacetMiner:
    def __init__(self, config: Any):
        self.config = config

        self.nlp = None
        if spacy is not None:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except Exception:
                self.nlp = None

        self.decomposer = None
        if getattr(config, "use_decomposer", False) and pipeline is not None:
            try:
                self.decomposer = pipeline("text2text-generation", model="google/flan-t5-base")
            except Exception:
                self.decomposer = None

    def extract_facets(
        self,
        query: str,
        supporting_facts: Optional[List[Tuple[str, int]]] = None,
    ) -> List[Facet]:
        query = _clean_text(query)
        facets: List[Facet] = []

        facets.extend(self._extract_entity_facets(query))
        facets.extend(self._extract_relation_facets(query))
        facets.extend(self._extract_temporal_facets(query))
        facets.extend(self._extract_numeric_facets(query))
        facets.extend(self._extract_comparison_facets(query))

        if self._is_multi_hop(query):
            facets.extend(self._extract_bridge_facets(query, supporting_facts))

        return self._deduplicate_facets(facets)

    # -------------------------
    # Entity extraction upgrades
    # -------------------------

    def _extract_or_entities(self, query: str) -> List[str]:
        """
        Heuristic: extract both sides of disjunction questions.
        Handles:
          - "... first A or B?"
          - "... A or B?"
        """
        q = query.strip()

        # Prefer a tighter pattern for "first A or B"
        m = re.search(r"\bfirst\s+(.+?)\s+or\s+(.+?)\s*\?$", q, flags=re.IGNORECASE)
        if not m:
            # Generic "... A or B?"
            m = re.search(r"(.+?)\s+or\s+(.+?)\s*\?$", q, flags=re.IGNORECASE)
        if not m:
            return []

        a, b = m.group(1).strip(), m.group(2).strip()

        # Remove leading boilerplate in a (e.g., "Which magazine was started first")
        a = re.sub(
            r"^(which|what|who|where|when|why|how)\b.*?\b(first|more|less|better|worse|earlier|later)\b",
            "",
            a,
            flags=re.IGNORECASE,
        ).strip()

        # Final cleanup
        a = _safe_phrase(a)
        b = _safe_phrase(b)
        return [x for x in [a, b] if x and not _looks_like_junk_entity(x)]

    def _extract_entity_facets(self, query: str) -> List[Facet]:
        mentions: List[str] = []

        # spaCy entities first
        if self.nlp is not None:
            doc = self.nlp(query)
            for ent in doc.ents:
                mention = (ent.text or "").strip()
                if _looks_like_junk_entity(mention):
                    continue
                mentions.append(mention)

        # Add disjunction entities (critical for Hotpot/2Wiki/MuSiQue comparisons)
        mentions.extend(self._extract_or_entities(query))

        # Fallback regex if still empty
        if not mentions:
            pattern = r"\b[A-Z][A-Za-z0-9']+(?:\s+[A-Z][A-Za-z0-9']+)*\b"
            mentions.extend(re.findall(pattern, query))

        # Dedupe + filter
        mentions = [m for m in _dedupe_strings(mentions) if not _looks_like_junk_entity(m)]

        facets: List[Facet] = []
        for m in mentions:
            facets.append(
                Facet(
                    facet_id=f"entity_{len(facets)}",
                    facet_type=FacetType.ENTITY,
                    template={"mention": _safe_phrase(m)},
                )
            )
        return facets

    # ---- relation ----

    def _extract_relation_facets(self, query: str) -> List[Facet]:
        facets: List[Facet] = []
        relation_patterns = [
            (r"(.+?)\s+(?:is|was|were|are)\s+(?:founded|created|written|directed|produced)\s+by\s+(.+?)(?:\?|$)",
             "created"),
            (r"(.+?)\s+(?:is|was|are|were)\s+the\s+capital\s+of\s+(.+?)(?:\?|$)",
             "capital_of"),
            (r"(.+?)\s+(?:is|was|are|were)\s+(?:located|situated)\s+in\s+(.+?)(?:\?|$)",
             "located_in"),
            (r"(.+?)\s+(?:is|was|are|were)\s+(.+?)\s+of\s+(.+?)(?:\?|$)",
             "related_to"),
        ]

        q = _clean_text(query)
        for pat, rel_type in relation_patterns:
            for m in re.finditer(pat, q, flags=re.IGNORECASE):
                groups = [g.strip() for g in m.groups() if g and g.strip()]
                if len(groups) < 2:
                    continue
                subj = groups[0]
                obj = groups[-1]
                subj = _safe_phrase(subj)
                obj = _safe_phrase(obj)
                if not subj or not obj:
                    continue
                facets.append(
                    Facet(
                        facet_id=f"relation_{len(facets)}",
                        facet_type=FacetType.RELATION,
                        template={"subject": subj, "predicate": rel_type, "object": obj},
                    )
                )
        return facets

    # ---- temporal ----

    def _extract_temporal_facets(self, query: str) -> List[Facet]:
        facets: List[Facet] = []
        temporal_patterns = [
            r"\b(19\d{2}|20\d{2})\b",
            r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",
            r"\b(before|after|during|since|until)\s+(\d{4}|[A-Za-z]+)\b",
        ]
        for pat in temporal_patterns:
            for m in re.finditer(pat, query, flags=re.IGNORECASE):
                time_text = _safe_phrase(m.group(0))
                if not time_text:
                    continue
                event = _safe_phrase(self._extract_temporal_event(query, m.span()))
                facets.append(
                    Facet(
                        facet_id=f"temporal_{len(facets)}",
                        facet_type=FacetType.TEMPORAL,
                        template={"time": time_text, "event": event},
                    )
                )
        return facets

    # ---- numeric ----

    def _extract_numeric_facets(self, query: str) -> List[Facet]:
        facets: List[Facet] = []
        numeric_pattern = r"\b(\d+(?:\.\d+)?)\s*(%|percent|million|billion|thousand|hundred)?\b"
        for m in re.finditer(numeric_pattern, query, flags=re.IGNORECASE):
            value = _safe_phrase(m.group(1))
            unit = _safe_phrase(m.group(2) or "")
            if not value:
                continue
            facets.append(
                Facet(
                    facet_id=f"numeric_{len(facets)}",
                    facet_type=FacetType.NUMERIC,
                    template={"value": value, "unit": unit},
                )
            )
        return facets

    # ---- comparison ----

    def _extract_comparison_facets(self, query: str) -> List[Facet]:
        facets: List[Facet] = []
        comparison_keywords = ["more", "less", "better", "worse", "higher", "lower", "bigger", "smaller", "earlier", "later", "first", "last"]

        ql = query.lower()
        if not any(k in ql for k in comparison_keywords):
            return facets

        # "A ... than B"
        for k in ["more", "less", "better", "worse", "higher", "lower", "bigger", "smaller", "earlier", "later"]:
            pat = rf"(.+?)\s+(?:is|was|are|were)?\s*{re.escape(k)}\s+than\s+(.+?)(?:\?|$)"
            for m in re.finditer(pat, query, flags=re.IGNORECASE):
                e1 = _safe_phrase(m.group(1))
                e2 = _safe_phrase(m.group(2))
                if not e1 or not e2:
                    continue
                facets.append(
                    Facet(
                        facet_id=f"comparison_{len(facets)}",
                        facet_type=FacetType.COMPARISON,
                        template={"entity1": e1, "entity2": e2, "attribute": k},
                    )
                )
        return facets

    # ---- bridge ----

    def _extract_bridge_facets(
        self,
        query: str,
        supporting_facts: Optional[List[Tuple[str, int]]] = None,
    ) -> List[Facet]:
        facets: List[Facet] = []
        if not supporting_facts:
            return facets

        titles: List[str] = []
        seen: Set[str] = set()
        for title, _ in supporting_facts:
            t = _safe_phrase(title)
            if t and t not in seen:
                seen.add(t)
                titles.append(t)

        if len(titles) < 2:
            return facets

        hop_counter = 0
        for i in range(len(titles) - 1):
            e1 = titles[i]
            eb = titles[i + 1]
            e2 = titles[i + 2] if (i + 2) < len(titles) else eb

            r1 = self._infer_relation(query)
            r2 = self._infer_relation(query)

            facets.append(
                Facet(
                    facet_id=f"bridge_hop1_{hop_counter}",
                    facet_type=FacetType.BRIDGE_HOP1,
                    template={"entity1": e1, "relation": r1, "bridge_entity": eb},
                    metadata={"hop_index": 1, "bridge_chain": hop_counter},
                )
            )
            if eb != e2:
                facets.append(
                    Facet(
                        facet_id=f"bridge_hop2_{hop_counter}",
                        facet_type=FacetType.BRIDGE_HOP2,
                        template={"bridge_entity": eb, "relation": r2, "entity2": e2},
                        metadata={"hop_index": 2, "bridge_chain": hop_counter},
                    )
                )
            hop_counter += 1

        return facets

    def _infer_relation(self, query: str) -> str:
        q = query.lower()
        indicators = [
            ("wrote", "wrote"),
            ("authored", "authored"),
            ("created", "created"),
            ("invented", "created"),
            ("directed", "directed"),
            ("produced", "created"),
            ("born in", "was born in"),
            ("died in", "died in"),
            ("located in", "is located in"),
            ("capital of", "is the capital of"),
            ("member of", "is a member of"),
            ("part of", "is part of"),
            ("played for", "played for"),
            ("starred in", "appeared in"),
        ]
        for pat, rel in indicators:
            if pat in q:
                return rel
        return "is related to"

    def _is_multi_hop(self, query: str) -> bool:
        q = query.lower()
        signals = [" and ", " both ", " which ", " that ", " who ", " while ", " whereas "]
        return sum(1 for s in signals if s in q) >= 2 or (" which " in q and " and " in q)

    def _get_entity_context(self, entity: Any, doc: Any) -> str:
        if doc is None:
            return ""
        start = max(0, entity.start - 3)
        end = min(len(doc), entity.end + 3)
        return _clean_text(" ".join(tok.text for tok in doc[start:end]))

    def _extract_temporal_event(self, query: str, time_span: Tuple[int, int]) -> str:
        start = max(0, time_span[0] - 30)
        end = min(len(query), time_span[1] + 30)
        ctx = query[start:end]
        time_text = query[time_span[0]:time_span[1]]
        event = _clean_text(ctx.replace(time_text, "")).strip(" ,;:-")
        if len(event) > 120:
            event = event[:120].rsplit(" ", 1)[0]
        return event

    def _deduplicate_facets(self, facets: List[Facet]) -> List[Facet]:
        seen: Set[str] = set()
        unique: List[Facet] = []
        for f in facets:
            hyp = f.to_hypothesis()
            sig = f"{f.facet_type.value}:{_clean_text(hyp).lower()}"
            if sig in seen:
                continue
            seen.add(sig)
            unique.append(f)
        return unique

    def decompose_query(self, query: str) -> List[str]:
        if self.decomposer is None:
            return [_clean_text(query)]
        prompt = f"Break down this question into simpler sub-questions:\n\n{query}\n"
        try:
            result = self.decomposer(prompt, max_length=200)
            text = result[0]["generated_text"]
            parts = [p.strip("-• \t") for p in text.split("\n") if p.strip()]
            return [_clean_text(p) for p in parts if _clean_text(p)]
        except Exception:
            return [_clean_text(query)]
