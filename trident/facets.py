#!/usr/bin/env python3
"""
trident/facets.py
UPDATES:
- Added generic BRIDGE_HOP type.
- Generates only adjacent hops (no aggregate).
- Works for 2, 3, 4+ title chains.
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
    pred = _clean_text(pred).lower().strip()
    mapping = {
        "capital_of": "is the capital of",
        "located_in": "is located in",
        "created": "was created by",
        "directed": "was directed by",
        "written": "was written by",
        "produced": "was produced by",
        "founded": "was founded by",
        "related_to": "is related to",
        "features": "features"
    }
    for k, v in mapping.items():
        if k in pred: return v
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

# WH-words and generic terms that should not be treated as real RELATION endpoints
_RELATION_WH_WORDS = {"who", "what", "which", "when", "where", "why", "how", "whom", "whose"}
_RELATION_GENERIC = {"the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "is", "are",
                     "was", "were", "be", "been", "being", "have", "has", "had", "do", "does",
                     "did", "will", "would", "could", "should", "may", "might", "must", "shall",
                     "director", "film", "movie", "actor", "actress", "person", "thing", "place",
                     "name", "title", "author", "writer", "book", "song", "album", "band", "group"}

def _clean_relation_endpoint_for_hypothesis(x: str) -> str:
    """
    Clean a RELATION endpoint for hypothesis construction.
    Removes WH-words and generic terms, returns what's left or empty string.
    """
    if not x:
        return ""
    x = _safe_phrase(x)
    # Check if it's a pure WH-word
    if x.lower() in _RELATION_WH_WORDS:
        return ""
    # Extract tokens and filter
    toks = re.findall(r"[A-Za-z0-9]+", x)
    cleaned = [t for t in toks if t.lower() not in _RELATION_WH_WORDS
               and t.lower() not in _RELATION_GENERIC and len(t) > 1]
    if not cleaned:
        return ""
    # Return original if it contains meaningful content, otherwise cleaned version
    return x if len(cleaned) >= len(toks) // 2 else " ".join(cleaned)

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


def _dedupe_entities_by_longest_span(mentions: List[str]) -> List[str]:
    """
    Deduplicate entity mentions by keeping only the longest span.

    If 'Michael' and 'Michael Doeberl' both exist, drop 'Michael'.
    This prevents partial matches from covering wrong passages.

    Example:
        Input: ['Michael', 'Michael Doeberl', 'Edward', 'Edward Dearle']
        Output: ['Michael Doeberl', 'Edward Dearle']
    """
    if not mentions:
        return []

    # Normalize and dedupe exact matches first
    normalized = []
    seen_lower = set()
    for m in mentions:
        m = _safe_phrase(m)
        if not m:
            continue
        m_lower = m.lower()
        if m_lower not in seen_lower:
            seen_lower.add(m_lower)
            normalized.append(m)

    if not normalized:
        return []

    # Sort by length descending so we check longer spans first
    by_length = sorted(normalized, key=lambda x: len(x), reverse=True)

    # Keep only mentions that are not substrings of a longer mention
    result = []
    for m in by_length:
        m_lower = m.lower()
        # Check if this mention is a substring of any already-kept mention
        is_substring = False
        for kept in result:
            kept_lower = kept.lower()
            # Check if m is a proper substring (not equal)
            if m_lower != kept_lower and m_lower in kept_lower:
                is_substring = True
                break
        if not is_substring:
            result.append(m)

    return result


class FacetType(Enum):
    ENTITY = "ENTITY"
    RELATION = "RELATION"
    TEMPORAL = "TEMPORAL"
    NUMERIC = "NUMERIC"
    BRIDGE_HOP = "BRIDGE_HOP" # Generic for all hops
    BRIDGE_HOP1 = "BRIDGE_HOP1" # Legacy/Specific
    BRIDGE_HOP2 = "BRIDGE_HOP2" # Legacy/Specific
    BRIDGE = "BRIDGE"
    COMPARISON = "COMPARISON"
    CAUSAL = "CAUSAL"
    PROCEDURAL = "PROCEDURAL"

    @classmethod
    def core_types(cls) -> list:
        return [cls.ENTITY, cls.RELATION, cls.TEMPORAL, cls.NUMERIC, cls.BRIDGE_HOP]

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
        tpl = self.template

        if ft == FacetType.ENTITY:
            mention = _safe_phrase(tpl.get("mention", ""))
            if passage_text:
                # Try to extract context like "X is a Y" or "X, a Y"
                pat1 = re.search(rf"\b{re.escape(mention)}\s+(?:is|was|are|were)\s+(?:an?|the)\s+([^,.]+)", passage_text, re.IGNORECASE)
                if pat1 and len(pat1.group(1).split()) < 10:
                    return _ensure_sentence(f"{mention} is {pat1.group(1)}")
                pat2 = re.search(rf"\b{re.escape(mention)}\s*[,(]\s*(?:an?|the)\s+([^,)]+)", passage_text, re.IGNORECASE)
                if pat2 and len(pat2.group(1).split()) < 10:
                    return _ensure_sentence(f"{mention} is {pat2.group(1)}")
            # Simple fallback: just check if the passage contains the entity
            # Previous hypothesis was too strict ("identifies unambiguously") for NLI
            return _ensure_sentence(f'The passage mentions "{mention}".')

        if ft == FacetType.NUMERIC:
            entity = _safe_phrase(tpl.get("entity", "")) 
            value = _safe_phrase(tpl.get("value", ""))
            unit = _safe_phrase(tpl.get("unit", ""))
            attr = _safe_phrase(tpl.get("attribute", ""))
            val_str = f"{value} {unit}".strip()
            if entity and attr: return _ensure_sentence(f"The passage states that {entity}'s {attr} is {val_str}")
            return _ensure_sentence(f"The passage binds the value {val_str} to a specific property")

        if ft == FacetType.RELATION:
            s_raw = tpl.get('subject', '')
            o_raw = tpl.get('object', '')
            p = _normalize_relation_predicate(tpl.get('predicate', ''))
            # Clean endpoints - remove WH-words and generic terms
            s = _clean_relation_endpoint_for_hypothesis(s_raw)
            o = _clean_relation_endpoint_for_hypothesis(o_raw)

            # PREDICATE-SPECIFIC HYPOTHESIS TEMPLATES
            # "The passage states something about X" is TOO PERMISSIVE
            # Use predicate-specific templates that require the actual relation
            #
            # CRITICAL: Check facet_text (subject + object + predicate combined)
            # NOT just predicate - because predicate may be generic "is related to"
            # while the actual relation type is in subject/object (e.g., "director of X")
            facet_text_lower = f"{s_raw} {o_raw} {p}".lower()

            # Detect relation kind from combined text (same logic as gate)
            relation_kind = "DEFAULT"
            hypothesis = None

            # Helper: strip relation suffix words from entity to avoid duplication
            # e.g., "film X won" -> "film X" when template adds "won"
            def _strip_relation_suffix(entity: str, suffixes: list) -> str:
                """Strip trailing relation words from entity string."""
                if not entity:
                    return entity
                import re
                # Build pattern: match any suffix word at end (with optional punctuation)
                pattern = r'\s+(?:' + '|'.join(re.escape(w) for w in suffixes) + r')[\s\.\?\!]*$'
                return re.sub(pattern, '', entity, flags=re.IGNORECASE).strip()

            # Director/directed
            if 'direct' in facet_text_lower or 'director' in facet_text_lower:
                relation_kind = "DIRECTOR"
                # Strip director-related words from object to avoid "directed directed"
                o_clean = _strip_relation_suffix(o, ['director', 'directed', 'directs', 'directing',
                                                      'director of', 'directed by'])
                s_clean = _strip_relation_suffix(s, ['director', 'directed', 'directs', 'directing'])
                if o_clean:
                    hypothesis = f"The passage states who directed {o_clean}"
                elif s_clean:
                    hypothesis = f"The passage states that {s_clean} directed a film"

            # Born/birthplace
            elif 'born' in facet_text_lower or 'birth' in facet_text_lower:
                relation_kind = "BORN"
                # Strip birth-related words from object
                o_clean = _strip_relation_suffix(o, ['born', 'birth', 'birthplace', 'born in',
                                                      'was born', 'birthdate', 'birthday'])
                s_clean = _strip_relation_suffix(s, ['born', 'birth', 'birthplace', 'was born'])
                if 'where' in s_raw.lower() or 'place' in facet_text_lower:
                    if o_clean:
                        hypothesis = f"The passage states where {o_clean} was born"
                    elif s_clean:
                        hypothesis = f"The passage states the birthplace of {s_clean}"
                else:
                    if o_clean:
                        hypothesis = f"The passage states when {o_clean} was born"
                    elif s_clean:
                        hypothesis = f"The passage states when {s_clean} was born"

            # Won/award/prize
            elif any(w in facet_text_lower for w in ['won', 'win', 'award', 'prize', 'honor']):
                relation_kind = "AWARD"
                # Strip award-related words from object to avoid "X won won"
                o_clean = _strip_relation_suffix(o, ['won', 'wins', 'winning', 'win', 'award',
                                                      'awards', 'awarded', 'nominated', 'prize',
                                                      'prizes', 'honor', 'honors', 'honoured'])
                s_clean = _strip_relation_suffix(s, ['won', 'wins', 'winning', 'win', 'award'])
                if o_clean:
                    hypothesis = f"The passage states what award {o_clean} won"
                elif s_clean:
                    hypothesis = f"The passage states what {s_clean} won"

            # Created/founded/written
            elif any(w in facet_text_lower for w in ['creat', 'found', 'writ', 'author', 'compose']):
                relation_kind = "CREATED"
                o_clean = _strip_relation_suffix(o, ['created', 'wrote', 'written', 'founded',
                                                      'authored', 'composed', 'created by'])
                s_clean = _strip_relation_suffix(s, ['created', 'wrote', 'written', 'founded'])
                if o_clean:
                    hypothesis = f"The passage states who created {o_clean}"
                elif s_clean:
                    hypothesis = f"The passage states what {s_clean} created"

            # Located/capital
            elif any(w in facet_text_lower for w in ['locat', 'capital', 'situat', 'based']):
                relation_kind = "LOCATION"
                if o:
                    hypothesis = f"The passage states where {o} is located"
                elif s:
                    hypothesis = f"The passage states the location of {s}"

            # Married/spouse
            elif any(w in facet_text_lower for w in ['marr', 'spouse', 'wife', 'husband']):
                relation_kind = "MARRIAGE"
                if o:
                    hypothesis = f"The passage states who {o} married"
                elif s:
                    hypothesis = f"The passage states the spouse of {s}"

            # Debug logging
            import os
            if os.environ.get("TRIDENT_DEBUG_RELATION", "0") == "1":
                print(f"[RELATION HYP] relation_kind={relation_kind} raw_predicate='{p}' "
                      f"facet_text='{facet_text_lower[:60]}...' hypothesis={hypothesis}")

            # Return specific hypothesis if we matched a relation kind
            if hypothesis:
                # Sanity check in debug mode
                if os.environ.get("TRIDENT_DEBUG_RELATION", "0") == "1":
                    hyp_lower = hypothesis.lower()
                    if relation_kind == "DIRECTOR" and "direct" not in hyp_lower:
                        print(f"[RELATION HYP] WARNING: DIRECTOR kind but 'direct' not in hypothesis!")
                    if relation_kind == "AWARD" and not any(w in hyp_lower for w in ["award", "won"]):
                        print(f"[RELATION HYP] WARNING: AWARD kind but no award word in hypothesis!")
                return _ensure_sentence(hypothesis)

            # Default: use predicate in hypothesis (still better than "something about")
            if s and o:
                return _ensure_sentence(f"The passage states that {s} {p} {o}")
            elif o:
                # Use predicate in template to be more specific
                return _ensure_sentence(f"The passage states the {p} of {o}")
            elif s:
                return _ensure_sentence(f"The passage states what {s} {p}")
            else:
                # Both endpoints are garbage - use a generic hypothesis that NLI will likely reject
                return _ensure_sentence(f"The passage states a factual relationship")

        if ft == FacetType.TEMPORAL:
            time = tpl.get('time', '')
            event = tpl.get('event', '')
            if event: return _ensure_sentence(f"The passage states that {event} happened in {time}")
            return _ensure_sentence(f"The passage mentions {time} in relation to a specific event")

        if ft == FacetType.COMPARISON:
            e1 = tpl.get('entity1', '')
            e2 = tpl.get('entity2', '')
            attr = tpl.get('attribute', 'different')
            return _ensure_sentence(f"The passage compares {e1} and {e2} regarding {attr}")

        # BRIDGE HYPOTHESES (Unified)
        if "BRIDGE_HOP" in ft.value:
            e1 = tpl.get("entity1", "")
            eb = tpl.get("bridge_entity", "")
            rel = tpl.get("relation", "")
            if "includes guest" in rel:
                return _ensure_sentence(f"The passage explicitly states that {e1} includes guest appearances from {eb}")
            return _ensure_sentence(f"The passage explicitly mentions a direct factual connection between {e1} and {eb}")

        if ft == FacetType.CAUSAL:
            return _ensure_sentence(f"The passage explains the cause or reason for the event")

        if ft == FacetType.PROCEDURAL:
            return _ensure_sentence(f"The passage describes a procedure, step, or method")

        return _ensure_sentence(str(self.template))

    def get_keywords(self) -> List[str]:
        """Extract keywords from facet template for lexical matching."""
        keywords = []
        tpl = self.template
        ft = self.facet_type

        if ft == FacetType.ENTITY:
            if tpl.get("mention"):
                keywords.append(tpl["mention"])

        elif ft == FacetType.NUMERIC:
            if tpl.get("entity"):
                keywords.append(tpl["entity"])
            if tpl.get("value"):
                keywords.append(str(tpl["value"]))

        elif ft == FacetType.RELATION:
            if tpl.get("subject"):
                keywords.append(tpl["subject"])
            if tpl.get("object"):
                keywords.append(tpl["object"])

        elif ft == FacetType.TEMPORAL:
            if tpl.get("time"):
                keywords.append(tpl["time"])
            if tpl.get("event"):
                keywords.append(tpl["event"])

        elif ft == FacetType.COMPARISON:
            if tpl.get("entity1"):
                keywords.append(tpl["entity1"])
            if tpl.get("entity2"):
                keywords.append(tpl["entity2"])

        elif "BRIDGE_HOP" in ft.value or ft == FacetType.BRIDGE:
            if tpl.get("entity1"):
                keywords.append(tpl["entity1"])
            if tpl.get("bridge_entity"):
                keywords.append(tpl["bridge_entity"])
            elif tpl.get("entity2"):
                keywords.append(tpl["entity2"])

        elif ft == FacetType.CAUSAL:
            # Extract any entities from template
            for key in ["cause", "effect", "entity"]:
                if tpl.get(key):
                    keywords.append(tpl[key])

        elif ft == FacetType.PROCEDURAL:
            # Extract procedure-related terms
            for key in ["action", "step", "method"]:
                if tpl.get(key):
                    keywords.append(tpl[key])

        return [kw for kw in keywords if kw]

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
        facets.extend(self._extract_causal_facets(query))
        
        dataset = str(getattr(self.config, "dataset", "")).lower()
        if "hotpot" not in dataset:
            facets.extend(self._extract_procedural_facets(query))
        
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
        # CRITICAL: Use longest-span dedup to prevent "Michael" when "Michael Doeberl" exists
        # This prevents partial matches from covering wrong passages
        mentions = [m for m in _dedupe_entities_by_longest_span(mentions) if not _looks_like_junk_entity(m)]
        return [Facet(facet_id=f"entity_{i}", facet_type=FacetType.ENTITY, template={"mention": _safe_phrase(m)}) for i, m in enumerate(mentions)]

    # ---- relation (FIXED) ----
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
                g = []
                for grp in m.groups():
                    clean = _safe_phrase(grp)
                    clean = re.sub(r"^(which|what|who|where|when|why|how)\s+\w+\s*", "", clean, flags=re.IGNORECASE).strip()
                    clean = re.sub(r"^(which|what|who|where|when|why|how)\s+(?:of\s+the\s+following\s+)?", "", clean, flags=re.IGNORECASE).strip()
                    g.append(clean)
                
                if len(g) < 2 or not g[0] or not g[-1]: continue
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

    # ---- causal / procedural ----
    def _extract_causal_facets(self, query: str) -> List[Facet]:
        facets: List[Facet] = []
        q = query.lower()
        causal_patterns = [r"\bwhy\b", r"\bwhat\s+(?:was|is|are)\s+the\s+caus(?:e|es)\b", r"\bwhat\s+led\s+to\b", r"\bwhat\s+resulted\s+in\b", r"\breason\s+for\b"]
        if any(re.search(p, q) for p in causal_patterns):
            facets.append(Facet(facet_id="causal_0", facet_type=FacetType.CAUSAL, template={"trigger": "causal"}))
        return facets

    def _extract_procedural_facets(self, query: str) -> List[Facet]:
        facets: List[Facet] = []
        q = query.lower()
        procedural_patterns = [r"\bhow\s+to\b", r"\bwhat\s+are\s+the\s+steps\b", r"\bsteps?\s+to\b", r"\bprocedure\b", r"\bwalk\s+me\s+through\b", r"\bprocess\s+for\b", r"\bmethod\s+for\b"]
        if any(re.search(p, q) for p in procedural_patterns):
            facets.append(Facet(facet_id="procedural_0", facet_type=FacetType.PROCEDURAL, template={"trigger": "procedural"}))
        return facets

    # ---- bridge (GENERIC HOPS) ----
    def _extract_bridge_facets(self, query: str, supporting_facts: Optional[List[Tuple[str, int]]] = None) -> List[Facet]:
        facets: List[Facet] = []
        if not supporting_facts: return facets
        
        titles = _dedupe_strings([t for t, _ in supporting_facts])
        if len(titles) < 2: return facets

        q_lower = query.lower()
        relation = "is related to"
        if "guest appearance" in q_lower: relation = "includes guest appearances from"
        elif "feature" in q_lower: relation = "features"
        elif "directed" in q_lower: relation = "was directed by"
        elif "written" in q_lower: relation = "was written by"
        elif "located" in q_lower: relation = "is located in"

        # Generate adjacent hops for any length (2, 3, 4...)
        # Uses generic BRIDGE_HOP type.
        for i in range(len(titles) - 1):
            e1, e2 = titles[i], titles[i+1]
            facets.append(Facet(
                facet_id=f"bridge_hop_{i+1}",
                facet_type=FacetType.BRIDGE_HOP,
                template={"entity1": e1, "relation": relation, "bridge_entity": e2, "hop": i+1}
            ))
            
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
