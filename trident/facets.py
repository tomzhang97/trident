#!/usr/bin/env python3
"""
trident/facets.py
Update: BRIDGE_HOP1 fallback uses specific assertion to avoid loose entailment.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import spacy
except Exception:
    spacy = None

_PUNCT_FIX_RE = re.compile(r"\s+([,.;:?!])")
_MULTI_SPACE_RE = re.compile(r"\s{2,}")
_APOS_S_RE = re.compile(r"\s+'s\b", re.IGNORECASE)
_BAD_CHARS_RE = re.compile(r"[\u200b\u200c\u200d\uFEFF]")

def _clean_text(s: str) -> str:
    s = (s or "").strip()
    s = _BAD_CHARS_RE.sub("", s)
    s = _APOS_S_RE.sub("'s", s)
    s = _PUNCT_FIX_RE.sub(r"\1", s)
    s = _MULTI_SPACE_RE.sub(" ", s)
    return s

def _ensure_sentence(s: str) -> str:
    s = _clean_text(s)
    if not s: return s
    if s[-1] not in ".?!": s += "."
    return s

def _safe_phrase(s: str) -> str:
    s = _clean_text(s)
    s = s.strip("`\"' ")
    return _clean_text(s)

def _normalize_relation_predicate(pred: str) -> str:
    pred = _clean_text(pred).strip()
    return pred or "is related to"

def _looks_like_junk_entity(mention: str) -> bool:
    m = _safe_phrase(mention)
    if not m: return True
    ml = m.lower()
    bad_tokens = {"first", "second", "third", "which", "what", "who", "when", "where", "one", "two", "this", "that", "the", "a", "an", "and", "or"}
    if ml in bad_tokens: return True
    if len(m.split()) == 1 and len(m) <= 3: return True
    if re.fullmatch(r"\d+(\.\d+)?", ml): return True
    return False

def _dedupe_strings(xs: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for x in xs:
        x = _safe_phrase(x)
        if not x: continue
        key = x.lower()
        if key in seen: continue
        seen.add(key)
        out.append(x)
    return out

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

    def to_hypothesis(self, passage_text: Optional[str] = None) -> str:
        ft = self.facet_type

        if ft == FacetType.ENTITY:
            mention = _safe_phrase(self.template.get("mention", ""))
            if passage_text:
                pat1 = re.search(rf"\b{re.escape(mention)}\s+(?:is|was|are|were)\s+(?:an?|the)\s+([^,.]+)", passage_text, re.IGNORECASE)
                if pat1 and len(pat1.group(1).split()) < 10:
                    return _ensure_sentence(f"{mention} is {pat1.group(1)}")
                pat2 = re.search(rf"\b{re.escape(mention)}\s*[,(]\s*(?:an?|the)\s+([^,)]+)", passage_text, re.IGNORECASE)
                if pat2 and len(pat2.group(1).split()) < 10:
                    return _ensure_sentence(f"{mention} is {pat2.group(1)}")
            return _ensure_sentence(f'The passage identifies "{mention}" unambiguously (e.g. by definition, role, or unique context), not merely mentions the phrase.')

        if ft == FacetType.NUMERIC:
            entity = _safe_phrase(self.template.get("entity", "")) 
            value = _safe_phrase(self.template.get("value", ""))
            unit = _safe_phrase(self.template.get("unit", ""))
            attr = _safe_phrase(self.template.get("attribute", ""))
            val_str = f"{value} {unit}".strip()
            if entity and attr: return _ensure_sentence(f"The passage states that {entity}'s {attr} is {val_str}")
            return _ensure_sentence(f"The passage binds the value {val_str} to a specific property")

        if ft == FacetType.BRIDGE_HOP1:
            e1 = _safe_phrase(self.template.get("entity1", ""))
            eb = _safe_phrase(self.template.get("bridge_entity", ""))
            rel = self.template.get("relation", "is related to")
            
            # Stronger assertion hypothesis
            if rel and rel != "is related to":
                return _ensure_sentence(f"The passage explicitly states that {e1} {rel} {eb}")
            # Even in fallback, use an assertion style
            return _ensure_sentence(f"The passage explicitly mentions a direct factual connection between {e1} and {eb}")

        if ft == FacetType.RELATION:
             return _ensure_sentence(f"{self.template.get('subject')} {self.template.get('predicate')} {self.template.get('object')}")
        
        return _ensure_sentence(str(self.template))

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
            try: self.nlp = spacy.load("en_core_web_sm")
            except Exception: self.nlp = None

    def extract_facets(self, query: str, supporting_facts: Optional[List[Tuple[str, int]]] = None) -> List[Facet]:
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

    # ---- entity ----
    def _extract_or_entities(self, query: str) -> List[str]:
        q = query.strip()
        m = re.search(r"\bfirst\s+(.+?)\s+or\s+(.+?)\s*\?$", q, flags=re.IGNORECASE)
        if not m: m = re.search(r"(.+?)\s+or\s+(.+?)\s*\?$", q, flags=re.IGNORECASE)
        if not m: return []
        a, b = m.group(1).strip(), m.group(2).strip()
        a = re.sub(r"^(which|what|who|where|when|why|how)\b.*?\b(first|more|less|better|worse|earlier|later)\b", "", a, flags=re.IGNORECASE).strip()
        return [x for x in [a, b] if _safe_phrase(x) and not _looks_like_junk_entity(x)]

    def _extract_entity_facets(self, query: str) -> List[Facet]:
        mentions: List[str] = []
        if self.nlp is not None:
            doc = self.nlp(query)
            for ent in doc.ents:
                m = (ent.text or "").strip()
                if not _looks_like_junk_entity(m): mentions.append(m)
        mentions.extend(self._extract_or_entities(query))
        if not mentions:
            mentions.extend(re.findall(r"\b[A-Z][A-Za-z0-9']+(?:\s+[A-Z][A-Za-z0-9']+)*\b", query))
        mentions = [m for m in _dedupe_strings(mentions) if not _looks_like_junk_entity(m)]
        return [Facet(facet_id=f"entity_{i}", facet_type=FacetType.ENTITY, template={"mention": _safe_phrase(m)}) for i, m in enumerate(mentions)]

    # ---- relation ----
    def _extract_relation_facets(self, query: str) -> List[Facet]:
        facets: List[Facet] = []
        patterns = [
            (r"(.+?)\s+(?:is|was|were|are)\s+(?:founded|created|written|directed|produced)\s+by\s+(.+?)(?:\?|$)", "created"),
            (r"(.+?)\s+(?:is|was|are|were)\s+the\s+capital\s+of\s+(.+?)(?:\?|$)", "capital_of"),
            (r"(.+?)\s+(?:is|was|are|were)\s+(?:located|situated)\s+in\s+(.+?)(?:\?|$)", "located_in"),
            (r"(.+?)\s+(?:is|was|are|were)\s+(.+?)\s+of\s+(.+?)(?:\?|$)", "related_to"),
        ]
        q = _clean_text(query)
        for pat, rel_type in patterns:
            for m in re.finditer(pat, q, flags=re.IGNORECASE):
                g = [g.strip() for g in m.groups() if g and g.strip()]
                if len(g) < 2: continue
                facets.append(Facet(facet_id=f"relation_{len(facets)}", facet_type=FacetType.RELATION, template={"subject": g[0], "predicate": rel_type, "object": g[-1]}))
        return facets

    # ---- temporal ----
    def _extract_temporal_facets(self, query: str) -> List[Facet]:
        facets: List[Facet] = []
        patterns = [r"\b(19\d{2}|20\d{2})\b", r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b"]
        for pat in patterns:
            for m in re.finditer(pat, query, flags=re.IGNORECASE):
                time_text = _safe_phrase(m.group(0))
                event = _safe_phrase(self._extract_temporal_event(query, m.span()))
                facets.append(Facet(facet_id=f"temporal_{len(facets)}", facet_type=FacetType.TEMPORAL, template={"time": time_text, "event": event}))
        return facets

    # ---- numeric ----
    def _extract_numeric_facets(self, query: str) -> List[Facet]:
        facets: List[Facet] = []
        numeric_pattern = r"\b(\d+(?:\.\d+)?)\s*(%|percent|million|billion|thousand|hundred)?\b"
        doc = self.nlp(query) if self.nlp else None
        
        for m in re.finditer(numeric_pattern, query, flags=re.IGNORECASE):
            value = _safe_phrase(m.group(1))
            unit = _safe_phrase(m.group(2) or "")
            if not value: continue
            
            entity, attribute = "", ""
            if doc:
                val_start, val_end = m.span(1)
                for token in doc:
                    if token.idx >= val_start and token.idx < val_end:
                        head = token.head
                        if head.pos_ in ["NOUN", "PROPN"]:
                            attribute = head.text
                            for child in head.children:
                                if child.dep_ == "prep" and child.text == "of":
                                    for pobj in child.children:
                                        if pobj.dep_ == "pobj": entity = self._get_compound_noun(pobj)
                        if head.dep_ == "prep" and head.head.pos_ == "VERB":
                            attribute = head.head.text
                            for child in head.head.children:
                                if child.dep_ in ["nsubj", "nsubjpass"]: entity = self._get_compound_noun(child)
            
            if not entity and not attribute:
                q_lower = query.lower()
                if "population" in q_lower: attribute = "population"
                elif "born" in q_lower: attribute = "birth year"
                elif "length" in q_lower: attribute = "length"
                elif "area" in q_lower: attribute = "area"
            
            facets.append(Facet(facet_id=f"numeric_{len(facets)}", facet_type=FacetType.NUMERIC, template={"value": value, "unit": unit, "entity": entity, "attribute": attribute}))
        return facets

    def _get_compound_noun(self, token) -> str:
        phrase = token.text
        lefts = sorted([t for t in token.lefts if t.dep_ in ["compound", "det", "amod"]], key=lambda t: t.i)
        if lefts: phrase = f"{' '.join(t.text for t in lefts)} {phrase}"
        return phrase

    # ---- comparison ----
    def _extract_comparison_facets(self, query: str) -> List[Facet]:
        facets: List[Facet] = []
        for k in ["more", "less", "better", "worse", "higher", "lower", "earlier", "later"]:
            pat = rf"(.+?)\s+(?:is|was|are|were)?\s*{re.escape(k)}\s+than\s+(.+?)(?:\?|$)"
            for m in re.finditer(pat, query, flags=re.IGNORECASE):
                facets.append(Facet(facet_id=f"comparison_{len(facets)}", facet_type=FacetType.COMPARISON, template={"entity1": m.group(1), "entity2": m.group(2), "attribute": k}))
        return facets

    # ---- bridge (UPGRADED) ----
    def _extract_bridge_facets(self, query: str, supporting_facts: Optional[List[Tuple[str, int]]] = None) -> List[Facet]:
        facets: List[Facet] = []
        if not supporting_facts: return facets
        
        titles = _dedupe_strings([t for t, _ in supporting_facts])
        if len(titles) < 2: return facets

        # Infer predicate from query
        q_lower = query.lower()
        relation = "is related to"
        if "guest appearance" in q_lower or "guest appearances" in q_lower:
            relation = "includes guest appearances from"
        elif "feature" in q_lower: # features, featuring
            relation = "features"
        elif "directed" in q_lower:
            relation = "was directed by"
        elif "written" in q_lower:
            relation = "was written by"
        elif "located" in q_lower or "where" in q_lower:
            relation = "is located in"

        hop_counter = 0
        for i in range(len(titles) - 1):
            e1, eb = titles[i], titles[i+1]
            e2 = titles[i+2] if (i + 2) < len(titles) else eb
            
            facets.append(Facet(
                facet_id=f"bridge_hop1_{hop_counter}",
                facet_type=FacetType.BRIDGE_HOP1,
                template={"entity1": e1, "relation": relation, "bridge_entity": eb},
            ))
            if eb != e2:
                facets.append(Facet(
                    facet_id=f"bridge_hop2_{hop_counter}",
                    facet_type=FacetType.BRIDGE_HOP2,
                    template={"bridge_entity": eb, "relation": relation, "entity2": e2},
                ))
            hop_counter += 1
        return facets

    def _is_multi_hop(self, query: str) -> bool:
        q = query.lower()
        return (" and " in q and " which " in q) or sum(1 for s in [" and ", " both ", " which "] if s in q) >= 2

    def _extract_temporal_event(self, query: str, time_span: Tuple[int, int]) -> str:
        start = max(0, time_span[0] - 30)
        end = min(len(query), time_span[1] + 30)
        return query[start:end].replace(query[time_span[0]:time_span[1]], "").strip(" ,;:-")

    def _deduplicate_facets(self, facets: List[Facet]) -> List[Facet]:
        seen: Set[str] = set()
        unique: List[Facet] = []
        for f in facets:
            sig = f"{f.facet_type.value}:{_clean_text(f.to_hypothesis()).lower()}"
            if sig in seen: continue
            seen.add(sig)
            unique.append(f)
        return unique
