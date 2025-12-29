#!/usr/bin/env python3
"""
trident/facets.py
UPDATES:
- Added generic BRIDGE_HOP type.
- Generates only adjacent hops (no aggregate).
- Works for 2, 3, 4+ title chains.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from .relation_schema import RelationRegistry, get_default_registry

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


_RELATION_REGISTRY: RelationRegistry = get_default_registry()


def _normalize_relation_schema(template: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize relation facet schema to a consistent contract.

    The gate/solver rely on deterministic fields, so we materialize
    unified keys while preserving the original template values.
    """
    tpl = dict(template or {})

    subject = tpl.get("subject") or tpl.get("entity1") or tpl.get("entity") or ""
    obj = tpl.get("object") or tpl.get("entity2") or tpl.get("bridge_entity") or ""
    predicate = tpl.get("predicate") or tpl.get("relation") or tpl.get("rel") or ""
    relation_kind = tpl.get("relation_kind") or tpl.get("outer_relation_type") or tpl.get("scoring_relation_kind") or ""
    hop = tpl.get("hop")

    # Preserve legacy hop detection while defaulting to provided value when available
    if hop is None and tpl.get("compositional") and tpl.get("bridge_entity"):
        hop = 2

    # Resolve canonical relation schema using the public allowlist (Wikidata-style)
    if relation_kind:
        tpl.setdefault("raw_relation_kind", relation_kind)
    spec = _RELATION_REGISTRY.lookup(relation_kind) or _RELATION_REGISTRY.lookup(predicate)

    if spec:
        relation_kind = spec.name  # Prefer canonical schema name for downstream gating
        tpl.setdefault("relation_pid", spec.pid)
        tpl["relation_kind"] = spec.name  # Canonicalized name
        tpl.setdefault("relation_label", spec.label)
        tpl.setdefault("relation_schema_source", spec.schema_source)
        tpl.setdefault("relation_schema_version", _RELATION_REGISTRY.version)
        tpl.setdefault("subject_type", spec.subject_type)
        tpl.setdefault("object_type", spec.object_type)
        # Upgrade predicate to a schema alias when weak/empty
        if not predicate or predicate.lower().strip() in {"is related to", "related to", "relation", "relate"}:
            predicate = spec.default_predicate()

    tpl.update({
        "subject": subject,
        "object": obj,
        "predicate": predicate,
        "relation_kind": relation_kind,
        "hop": hop,
    })

    # Upgrade generic predicates to canonical phrases when a relation kind is known.
    # This avoids weak "is related to" predicates that degrade gating and hypotheses.
    canonical_predicate = {
        "DIRECTOR": "directed",
        "BORN": "was born",
        "BIRTHPLACE": "was born",
        "AWARD": "won",
        "CREATED": "created",
        "LOCATION": "is located",
        "MARRIAGE": "married",
        "MOTHER": "is the mother of",
        "FATHER": "is the father of",
        "PARENT": "is the parent of",
        "CHILD": "is the child of",
        "SPOUSE": "is the spouse of",
        "NATIONALITY": "is a citizen of",
    }
    weak_predicates = {"is related to", "related to", "relation", "relate"}
    if relation_kind:
        pred_clean = predicate.lower().strip()
        if not pred_clean or pred_clean in weak_predicates:
            tpl["predicate"] = canonical_predicate.get(str(relation_kind).upper(), tpl.get("predicate", predicate))

    return tpl

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


def instantiate_facets(
    facets: List["Facet"],
    bindings: Dict[str, str]
) -> List["Facet"]:
    """
    Replace symbolic placeholders in facet templates with concrete bound entities.

    This is the REQUIRED facet instantiation step for hop-2 facets in
    two-pass Certified Adaptive Safe-Cover.

    Example:
        bindings = {"DIRECTOR_RESULT": "Xawery Żuławski"}

        Input facet template:
            {"subject": "[DIRECTOR_RESULT]", "predicate": "'s mother", ...}

        Output facet template:
            {"subject": "Xawery Żuławski", "predicate": "'s mother", ...}

    CRITICAL: The output facets are NEW objects. Original facets are not mutated.

    Args:
        facets: List of facets (possibly containing placeholders like [VAR_RESULT])
        bindings: Dict mapping variable names to bound entity values
                  e.g., {"DIRECTOR_RESULT": "Xawery Żuławski"}

    Returns:
        List of new facets with placeholders replaced by concrete values
    """
    import os
    debug = os.environ.get("TRIDENT_DEBUG_INSTANTIATE", "0") == "1"

    if not bindings:
        return list(facets)

    new_facets = []
    for facet in facets:
        # Build new template with placeholders replaced
        old_tpl = facet.template or {}
        new_tpl = dict(old_tpl)  # Shallow copy

        # Track if we made any replacements
        replaced = False

        # Fields that may contain placeholders
        placeholder_fields = ["subject", "object", "hypothesis", "entity1", "entity2", "bridge_entity"]

        for field in placeholder_fields:
            if field not in new_tpl:
                continue
            value = str(new_tpl[field] or "")

            # Replace each binding
            for var_name, entity_value in bindings.items():
                placeholder = f"[{var_name}]"
                if placeholder in value:
                    new_value = value.replace(placeholder, entity_value)
                    new_tpl[field] = new_value
                    replaced = True

                    if debug:
                        print(f"[INSTANTIATE] {facet.facet_id} field={field}: "
                              f"'{value}' -> '{new_value}'")

        # Also update metadata to mark as instantiated
        old_meta = facet.metadata or {}
        new_meta = dict(old_meta)
        if replaced:
            new_meta["instantiated"] = True
            new_meta["bindings_applied"] = list(bindings.keys())

        # ================================================================
        # PARENT NORMALIZATION: Normalize MOTHER/FATHER to PARENT for scoring
        # ================================================================
        # Evidence often says "X is the son of Y" which doesn't contain "mother".
        # NLI fails to entail "mother of X" from "son of Y" reliably.
        # Fix: Score as PARENT (hypothesis: "The passage states a parent of X")
        # and gate anchors include {mother, father, son of, daughter of, parent}.
        # Keep outer_relation_type for reporting/analysis.
        outer_rel = new_tpl.get("outer_relation_type", "")
        if outer_rel in ("MOTHER", "FATHER"):
            new_tpl["scoring_relation_kind"] = "PARENT"
            if debug:
                print(f"[INSTANTIATE] {facet.facet_id} normalized {outer_rel} -> scoring_relation_kind=PARENT")

        # Create new facet with updated template
        # Note: Facet is frozen=True, so we create a new instance
        new_facet = Facet(
            facet_id=facet.facet_id,
            facet_type=facet.facet_type,
            template=new_tpl,
            weight=facet.weight,
            metadata=new_meta,
            required=facet.required,  # Preserve required flag
        )
        new_facets.append(new_facet)

    return new_facets


def is_wh_question(question: str) -> bool:
    """Check if question is a WH-question (who, what, where, when, which, how)."""
    q_lower = question.lower().strip()
    wh_starters = ('who ', 'what ', 'where ', 'when ', 'which ', 'how ', 'whose ')
    return q_lower.startswith(wh_starters) or any(f' {w}' in q_lower[:30] for w in ['who', 'what', 'where', 'when', 'which'])


def mark_required_facets(facets: List["Facet"], question: str) -> List["Facet"]:
    """
    Mark RELATION facets as required for WH-questions.

    For questions like "Who is X?" or "Where was Y born?", the RELATION facet
    MUST be covered for a valid answer. Without it, the answer is unreliable.

    Args:
        facets: List of facets
        question: The original question

    Returns:
        List of facets with required flags set appropriately
    """
    if not is_wh_question(question):
        return facets

    import os
    debug = os.environ.get("TRIDENT_DEBUG_REQUIRED", "0") == "1"

    new_facets = []
    for facet in facets:
        # Mark RELATION facets as required for WH-questions
        if facet.facet_type == FacetType.RELATION:
            if not facet.required:
                # Create new facet with required=True
                new_facet = Facet(
                    facet_id=facet.facet_id,
                    facet_type=facet.facet_type,
                    template=facet.template,
                    weight=facet.weight,
                    metadata=facet.metadata,
                    required=True,
                )
                if debug:
                    print(f"[REQUIRED] Marking RELATION facet as required: {facet.facet_id}")
                new_facets.append(new_facet)
            else:
                new_facets.append(facet)
        else:
            new_facets.append(facet)

    return new_facets

@dataclass(frozen=True)
class Facet:
    facet_id: str
    facet_type: FacetType
    template: Dict[str, Any]
    weight: float = 1.0
    metadata: Dict[str, Any] = None
    required: bool = False  # If True, this facet MUST be covered for certification

    def __post_init__(self):
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})
        ft = object.__getattribute__(self, "facet_type")
        if isinstance(ft, str):
            object.__setattr__(self, "facet_type", FacetType(ft))

        # Normalize relation-like facets to avoid schema drift in gating/solver
        tpl = object.__getattribute__(self, "template")
        if self.facet_type in {FacetType.RELATION, FacetType.BRIDGE_HOP, FacetType.BRIDGE_HOP1, FacetType.BRIDGE_HOP2}:
            tpl = _normalize_relation_schema(tpl)
            object.__setattr__(self, "template", tpl)

            meta = dict(object.__getattribute__(self, "metadata") or {})
            subject_raw = str(tpl.get("subject", ""))
            object_raw = str(tpl.get("object", ""))
            placeholder_re = re.compile(r"\[[A-Z0-9_]+_RESULT\]")

            meta.setdefault("instantiated", not (placeholder_re.search(subject_raw) or placeholder_re.search(object_raw)))
            meta.setdefault("subject_is_entity", bool(subject_raw.strip()) and not placeholder_re.search(subject_raw))
            meta.setdefault("has_two_entity_anchors", bool(subject_raw.strip()) and bool(object_raw.strip()))
            object.__setattr__(self, "metadata", meta)

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
            # CRITICAL FIX: Check for EXPLICIT relation_kind in template FIRST
            # This is needed for compositional hop-2 facets where the subject
            # contains "[DIRECTOR_RESULT]" but the actual relation is MOTHER
            #
            # PARENT NORMALIZATION: Prefer scoring_relation_kind over outer_relation_type.
            # This allows MOTHER/FATHER to be scored as PARENT for better NLI matching.
            explicit_relation_kind = (
                tpl.get('scoring_relation_kind') or
                tpl.get('outer_relation_type') or
                tpl.get('relation_kind')
            )

            # D1 FIX: For hop-2 compositional facets, make hypothesis entity-anchored
            # After instantiation, the subject should be a real entity name (not placeholder)
            is_hop2 = tpl.get('hop') == 2
            is_instantiated = (self.metadata or {}).get('instantiated', False)
            is_placeholder = s_raw.startswith('[') and s_raw.endswith('_RESULT]')

            if is_hop2:
                if is_placeholder:
                    # Subject still has placeholder - not yet instantiated
                    # Use generic "the person" (this facet shouldn't be scored yet)
                    s = "the person"
                elif is_instantiated and s:
                    # D1 FIX: Subject is instantiated - use the actual entity name
                    # This anchors the hypothesis to the specific person for NLI matching
                    # e.g., "The passage states the mother of Andrzej Żuławski"
                    pass  # s is already set correctly from s_raw via _clean_relation_endpoint_for_hypothesis
                # If hop-2 but not instantiated and not placeholder, use s as-is

            # Detect relation kind - explicit first, then keyword fallback
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

            # ================================================================
            # EXPLICIT RELATION KIND (from compositional facets)
            # ================================================================
            if explicit_relation_kind:
                relation_kind = explicit_relation_kind

                if relation_kind == "MOTHER":
                    # "The passage states who is the mother of X" or
                    # "The passage states that X is the son/daughter of Y"
                    hypothesis = f"The passage states the mother of {s}" if s else "The passage identifies a mother-child relationship"

                elif relation_kind == "FATHER":
                    hypothesis = f"The passage states the father of {s}" if s else "The passage identifies a father-child relationship"

                elif relation_kind in ("PARENT", "CHILD"):
                    hypothesis = f"The passage states a parent-child relationship involving {s}" if s else "The passage identifies a parent-child relationship"

                elif relation_kind == "SPOUSE":
                    hypothesis = f"The passage states who {s} married" if s else "The passage identifies a marriage/spouse relationship"

                elif relation_kind == "NATIONALITY":
                    hypothesis = f"The passage states the nationality of {s}" if s else "The passage identifies someone's nationality"

                elif relation_kind == "BIRTHPLACE":
                    hypothesis = f"The passage states where {s} was born" if s else "The passage identifies someone's birthplace"

                elif relation_kind == "AWARD":
                    hypothesis = f"The passage states what award {s} won" if s else "The passage identifies an award"

                # If explicit relation kind matched, return now
                if hypothesis:
                    import os
                    if os.environ.get("TRIDENT_DEBUG_RELATION", "0") == "1":
                        print(f"[RELATION HYP] EXPLICIT relation_kind={relation_kind} hypothesis={hypothesis}")
                    return _ensure_sentence(hypothesis)

            # ================================================================
            # KEYWORD-BASED RELATION KIND (fallback for non-compositional facets)
            # ================================================================
            # CRITICAL: For compositional hop-2 facets, we should have matched above.
            # This section is for regular (non-compositional) RELATION facets.
            facet_text_lower = f"{s_raw} {o_raw} {p}".lower()

            # Skip keyword detection if this is a hop-2 facet (should have been handled above)
            if tpl.get('hop') == 2:
                # Hop-2 without explicit relation_kind - use generic hypothesis
                if bound_entity_var:
                    relation_kind = "HOP2_GENERIC"
                    hypothesis = f"The passage states information about {s}"
                    import os
                    if os.environ.get("TRIDENT_DEBUG_RELATION", "0") == "1":
                        print(f"[RELATION HYP] HOP2 generic: {hypothesis}")
                    return _ensure_sentence(hypothesis)

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
            "required": self.required,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Facet":
        return cls(
            facet_id=data["facet_id"],
            facet_type=FacetType(data["facet_type"]),
            template=data["template"],
            weight=data.get("weight", 1.0),
            metadata=data.get("metadata"),
            required=data.get("required", False),
        )

class FacetMiner:
    """Facet miner for Safe-Cover.

    This version supports **Option B** facet planning:
      - one (optional) LLM call per question to decide (relation_pid/name + required facets)
      - everything else unchanged; Safe-Cover remains gatekeeper
    """

    def __init__(self, config: Any, llm_interface: Optional[Any] = None):
        self.config = config
        # Always define; pipeline may not pass an LLM.
        self.llm = llm_interface

        # Public / normalized relation schema (Wikidata-aligned).
        # Prefer config path, then env, else bundled default registry.
        schema_path = (
            getattr(config, "relation_schema_json_path", None)
            or os.environ.get("TRIDENT_RELATION_SCHEMA_JSON", "")
        )
        self.relation_registry = None
        try:
            if schema_path and os.path.exists(schema_path):
                self.relation_registry = RelationRegistry.from_json(schema_path)
            else:
                self.relation_registry = get_default_registry()
        except Exception:
            # Fail-open to default: schema errors should not crash facet mining.
            self.relation_registry = get_default_registry()
        # Optional spaCy pipeline for entity extraction (set to None by default)
        self.nlp = None

    def extract_facets(self, query: str, supporting_facts: Optional[List[Tuple[str, int]]] = None) -> List[Facet]:
        query = _clean_text(query)
        facets: List[Facet] = []
        facets.extend(self._extract_entity_facets(query))
        facets.extend(self._extract_relation_facets(query, supporting_facts))
        facets.extend(self._extract_temporal_facets(query))
        facets.extend(self._extract_numeric_facets(query))
        facets.extend(self._extract_comparison_facets(query))
        facets.extend(self._extract_causal_facets(query))

        # CRITICAL: Extract compositional (2-hop) facets BEFORE bridge facets
        # e.g., "mother of the director of film X" -> DIRECTOR + MOTHER facets
        facets.extend(self._extract_compositional_facets(query))

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
        nlp = getattr(self, 'nlp', None)
        if nlp is not None:
            doc = nlp(query)
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
    def _extract_relation_facets(
        self,
        query: str,
        supporting_facts: Optional[List[Tuple[str, int]]] = None
    ) -> List[Facet]:
        """Relation facet extraction.

        Option B (LLM-driven):
        - Make **one** LLM call per question to decide the answer relation kind and
          the *required* relation facets for the answer.
        - Fail-closed: if LLM is unavailable or parsing fails, fall back to the
          existing heuristic extractor.

        The returned facets must keep the existing Facet schema so Safe-Cover
        remains the gatekeeper.
        """
        use_llm = bool(getattr(self, "llm", None)) and (
            getattr(self.config, "use_llm_facet_plan", False)
            or os.environ.get("TRIDENT_LLM_FACET_PLAN", "0") == "1"
        )
        if not use_llm:
            return self._extract_relation_facets_heuristic(query)

        # --- LLM facet plan: JSON-only, schema-fixed ---
        prompt = f"""You are planning required relation facets for a QA system.

Question: {query}

Return JSON ONLY with this exact schema:
{{
  "answer_relation_kind": "DIRECTOR|WRITER|STARRING|PRODUCER|LOCATION|BIRTH_DATE|DEATH_DATE|AWARD|NATIONALITY|MOTHER|FATHER|SPOUSE|PARENT|DEFAULT",
  "required_relation_facets": [
    {{
      "relation_kind": "DIRECTOR|WRITER|STARRING|PRODUCER|LOCATION|BIRTH_DATE|DEATH_DATE|AWARD|NATIONALITY|MOTHER|FATHER|SPOUSE|PARENT|DEFAULT",
      "subject": "WHO|WHAT|WHERE|WHEN|<entity string>",
      "predicate": "<short canonical predicate phrase>",
      "object": "<entity string or ?>",
      "required": true
    }}
  ]
}}

Rules:
- Output MUST be a single JSON object and nothing else.
- Do NOT invent entities not mentioned in the question.
- Use subject="WHO" for person answers, "WHAT" for non-person entities, "WHEN" for dates, "WHERE" for locations.
"""

        def _parse_first_json(text: str) -> Optional[Dict[str, Any]]:
            start = text.find("{")
            if start == -1:
                return None
            depth = 0
            for i in range(start, len(text)):
                ch = text[i]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        chunk = text[start:i+1]
                        try:
                            import json as _json
                            return _json.loads(chunk)
                        except Exception:
                            return None
            return None

        try:
            resp = self.llm.generate(prompt, temperature=0.0)
            raw = resp.text if hasattr(resp, "text") else str(resp)
            data = _parse_first_json(raw)
        except Exception:
            data = None

        if not isinstance(data, dict):
            return self._extract_relation_facets_heuristic(query)

        facets: List[Facet] = []
        items = data.get("required_relation_facets") or []
        if not isinstance(items, list) or not items:
            return self._extract_relation_facets_heuristic(query)


        # Canonicalize: fill missing subject/object and apply normalized schema defaults.
        for idx, it in enumerate(items):
            # Normalize relation using public schema (pid/name)
            rk_raw = (it.get("relation_kind") or it.get("relation") or it.get("kind") or "DEFAULT")
            rpid = it.get("relation_pid") or data.get("answer_relation_pid") or ""
            spec = None
            if rpid:
                spec = self.relation_registry.get(rpid) if self.relation_registry else None
                if spec:
                    rk_raw = spec.name
            if not spec and self.relation_registry:
                spec = self.relation_registry.match_from_name(str(rk_raw))

            rk = spec.name if spec else str(rk_raw).upper()

            subj = it.get("subject") or it.get("entity1") or it.get("entity") or ""
            obj = it.get("object") or it.get("entity2") or it.get("bridge_entity") or ""

            pred = (it.get("predicate") or it.get("relation") or "").strip()
            schema_meta: Dict[str, Any]
            if spec:
                if not pred or pred.lower() in {"is related to", "related to", "is related"}:
                    pred = spec.default_predicate
                keywords = sorted(set((spec.keywords or []) + (spec.aliases or [])))
                schema_meta = {
                    "relation_pid": spec.pid,
                    "relation_kind": spec.name,
                    "relation_keywords": keywords,
                    "schema_version": getattr(self.relation_registry, "version", "v1"),
                }
            else:
                schema_meta = {"relation_kind": rk}

            facets.append(
                Facet(
                    facet_id=f"rel_{idx}",
                    facet_type=FacetType.RELATION,
                    template={
                        "subject": subj,
                        "predicate": pred or "is related to",
                        "object": obj,
                        **schema_meta,
                    },
                )
            )

        return facets if facets else self._extract_relation_facets_heuristic(query)
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

    # ---- compositional (2-hop) facets ----
    def _extract_compositional_facets(self, query: str) -> List[Facet]:
        """
        Extract facets for compositional (2-hop) questions.

        Patterns like:
        - "Who is the mother of the director of film X?"
        - "Where was the director of film X born?"
        - "What nationality is the creator of X?"

        These require TWO facets:
        - Hop1: Inner relation (director of X) -> yields bridge entity
        - Hop2: Outer relation (mother of [bridge]) -> yields final answer
        """
        facets: List[Facet] = []
        q = _clean_text(query)
        q_lower = q.lower()

        # Outer relations (the final answer type)
        # Maps pattern -> (relation_type, hypothesis_template)
        outer_relations = {
            'mother': ('MOTHER', 'mother'),
            'father': ('FATHER', 'father'),
            'wife': ('SPOUSE', 'wife'),
            'husband': ('SPOUSE', 'husband'),
            'spouse': ('SPOUSE', 'spouse'),
            'son': ('CHILD', 'son'),
            'daughter': ('CHILD', 'daughter'),
            'child': ('CHILD', 'child'),
            'nationality': ('NATIONALITY', 'nationality'),
            'birthplace': ('BIRTHPLACE', 'birthplace'),
            'born': ('BIRTHPLACE', 'birthplace'),  # "where was X born"
        }

        # Inner relations (the intermediate hop)
        inner_relations = {
            'director': 'DIRECTOR',
            'directed': 'DIRECTOR',
            'creator': 'CREATOR',
            'created': 'CREATOR',
            'founder': 'FOUNDER',
            'founded': 'FOUNDER',
            'author': 'AUTHOR',
            'wrote': 'AUTHOR',
            'written': 'AUTHOR',
            'producer': 'PRODUCER',
            'produced': 'PRODUCER',
            'composer': 'COMPOSER',
            'composed': 'COMPOSER',
            'performer': 'PERFORMER',
            'performed': 'PERFORMER',
            'singer': 'PERFORMER',
            'actor': 'PERFORMER',
            'actress': 'PERFORMER',
            'starring': 'PERFORMER',
        }

        # Pattern: "(outer) of the (inner) of (object)"
        # e.g., "mother of the director of film Polish-Russian War"
        pattern1 = r"(?:who\s+is\s+the\s+|what\s+is\s+the\s+)?(\w+)\s+of\s+(?:the\s+)?(\w+)\s+of\s+(?:(?:the\s+)?(?:film|movie|book|song|album|show|series|novel|play)\s+)?(.+?)(?:\?|$)"
        for m in re.finditer(pattern1, q, flags=re.IGNORECASE):
            outer_word = m.group(1).lower()
            inner_word = m.group(2).lower()
            inner_object = _safe_phrase(m.group(3))

            if outer_word in outer_relations and inner_word in inner_relations:
                outer_type, outer_name = outer_relations[outer_word]
                inner_type = inner_relations[inner_word]

                # Hop1: Inner relation facet (e.g., director of film X)
                hop1_facet = Facet(
                    facet_id=f"comp_hop1_{len(facets)}",
                    facet_type=FacetType.RELATION,
                    template={
                        "subject": "Who",
                        "predicate": f"is the {inner_word} of",
                        "object": inner_object,
                        "hop": 1,
                        "compositional": True,
                        "inner_relation_type": inner_type,
                    },
                    metadata={"compositional_hop": 1, "outer_relation": outer_type}
                )
                facets.append(hop1_facet)

                # Hop2: Outer relation facet (e.g., mother of [hop1_result])
                # Note: subject is a variable reference to hop1 result
                hop2_facet = Facet(
                    facet_id=f"comp_hop2_{len(facets)}",
                    facet_type=FacetType.RELATION,
                    template={
                        "subject": f"[{inner_type}_RESULT]",  # Variable reference
                        "predicate": f"'s {outer_name}",
                        "object": "",
                        "hop": 2,
                        "compositional": True,
                        "outer_relation_type": outer_type,
                        "depends_on_hop": 1,
                    },
                    metadata={"compositional_hop": 2, "depends_on": f"comp_hop1_{len(facets)-1}"}
                )
                facets.append(hop2_facet)

                import os
                if os.environ.get("TRIDENT_DEBUG_COMPOSITIONAL", "0") == "1":
                    print(f"[COMPOSITIONAL] Detected 2-hop pattern:")
                    print(f"  Outer: {outer_type} ({outer_word})")
                    print(f"  Inner: {inner_type} ({inner_word})")
                    print(f"  Object: {inner_object}")

        # Pattern: "Where was the (inner) of (object) born?"
        # e.g., "Where was the director of film X born?"
        pattern2 = r"where\s+(?:was|is)\s+(?:the\s+)?(\w+)\s+of\s+(?:(?:the\s+)?(?:film|movie|book|song|album|show|series|novel|play)\s+)?(.+?)\s+born\s*\?"
        for m in re.finditer(pattern2, q, flags=re.IGNORECASE):
            inner_word = m.group(1).lower()
            inner_object = _safe_phrase(m.group(2))

            if inner_word in inner_relations:
                inner_type = inner_relations[inner_word]

                # Hop1: Inner relation facet
                hop1_facet = Facet(
                    facet_id=f"comp_hop1_{len(facets)}",
                    facet_type=FacetType.RELATION,
                    template={
                        "subject": "Who",
                        "predicate": f"is the {inner_word} of",
                        "object": inner_object,
                        "hop": 1,
                        "compositional": True,
                        "inner_relation_type": inner_type,
                    },
                    metadata={"compositional_hop": 1, "outer_relation": "BIRTHPLACE"}
                )
                facets.append(hop1_facet)

                # Hop2: Birthplace facet
                hop2_facet = Facet(
                    facet_id=f"comp_hop2_{len(facets)}",
                    facet_type=FacetType.RELATION,
                    template={
                        "subject": f"[{inner_type}_RESULT]",
                        "predicate": "was born in",
                        "object": "",
                        "hop": 2,
                        "compositional": True,
                        "outer_relation_type": "BIRTHPLACE",
                        "depends_on_hop": 1,
                    },
                    metadata={"compositional_hop": 2, "depends_on": f"comp_hop1_{len(facets)-1}"}
                )
                facets.append(hop2_facet)

        # Pattern: "What award did the (inner) of (object) win?"
        pattern3 = r"what\s+(?:award|prize)\s+did\s+(?:the\s+)?(\w+)\s+of\s+(?:(?:the\s+)?(?:film|movie|book|song|album|show|series|novel|play)\s+)?(.+?)\s+(?:win|receive)\s*\?"
        for m in re.finditer(pattern3, q, flags=re.IGNORECASE):
            inner_word = m.group(1).lower()
            inner_object = _safe_phrase(m.group(2))

            if inner_word in inner_relations:
                inner_type = inner_relations[inner_word]

                hop1_facet = Facet(
                    facet_id=f"comp_hop1_{len(facets)}",
                    facet_type=FacetType.RELATION,
                    template={
                        "subject": "Who",
                        "predicate": f"is the {inner_word} of",
                        "object": inner_object,
                        "hop": 1,
                        "compositional": True,
                        "inner_relation_type": inner_type,
                    },
                    metadata={"compositional_hop": 1, "outer_relation": "AWARD"}
                )
                facets.append(hop1_facet)

                hop2_facet = Facet(
                    facet_id=f"comp_hop2_{len(facets)}",
                    facet_type=FacetType.RELATION,
                    template={
                        "subject": f"[{inner_type}_RESULT]",
                        "predicate": "won",
                        "object": "",
                        "hop": 2,
                        "compositional": True,
                        "outer_relation_type": "AWARD",
                        "depends_on_hop": 1,
                    },
                    metadata={"compositional_hop": 2, "depends_on": f"comp_hop1_{len(facets)-1}"}
                )
                facets.append(hop2_facet)

        return facets

    # ---- bridge (GENERIC HOPS) ----
    

    # ---- bridge (GENERIC HOPS) ----
    def _extract_bridge_facets(self, query: str, supporting_facts: Optional[List[Tuple[str, int]]] = None) -> List[Facet]:
        """Create generic BRIDGE_HOP facets connecting adjacent supporting facts.

        Uses the **public normalized relation schema** to choose a default predicate
        when possible, but remains safe: these are still *requirements*, and Safe-Cover
        is the gatekeeper.
        """
        facets: List[Facet] = []
        if not supporting_facts:
            return facets

        titles = _dedupe_strings([t for t, _ in supporting_facts])
        if len(titles) < 2:
            return facets

        # Choose a schema relation for the whole question (single decision).
        spec = None
        try:
            spec = self.relation_registry.match_from_question(query) if self.relation_registry else None
        except Exception:
            spec = None

        relation = spec.default_predicate if spec else "is related to"
        schema_meta: Dict[str, Any] = {}
        if spec:
            schema_meta = {
                "relation_pid": spec.pid,
                "relation_kind": spec.name,
                "relation_keywords": sorted(set((spec.keywords or []) + (spec.aliases or []))),
                "schema_version": getattr(self.relation_registry, "version", "v1"),
            }

        for i in range(len(titles) - 1):
            e1, e2 = titles[i], titles[i + 1]
            facets.append(
                Facet(
                    facet_id=f"bridge_hop_{i+1}",
                    facet_type=FacetType.BRIDGE_HOP,
                    template={
                        "entity1": e1,
                        "relation": relation,
                        "bridge_entity": e2,
                        "hop": i + 1,
                        **schema_meta,
                    },
                )
            )

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
