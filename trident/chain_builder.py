"""Multi-hop chain builder for extracting answers from certified passages.
This module builds explicit reasoning chains from certified evidence,
ensuring that the answer is grounded in the actual passages selected.
"""

from __future__ import annotations

import re
import string
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Any

# Entity extraction regex - Title Case spans
ENTITY_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,4})\b")

# =============================================================================
# CERTIFIED-ONLY CONSTANTS
# =============================================================================
# Deny-listed document titles that should never be used for answer extraction
DENY_TITLES = frozenset([
    "disambiguation", "list of", "index of", "outline of", "portal:",
    "category:", "template:", "wikipedia:", "help:", "file:", "mediawiki:"
])

# Month names for date extraction
MONTHS = frozenset([
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec"
])


# =============================================================================
# CERTIFIED-ONLY HELPER FUNCTIONS
# =============================================================================

def _cert_pid(certificates: List[Dict[str, Any]]) -> Set[str]:
    """
    Extract certified passage IDs from certificates.

    Returns:
        Set of passage IDs that have at least one valid certificate.
    """
    return {c.get("passage_id", "") for c in certificates if c.get("passage_id")}


def _norm_ws(s: str) -> str:
    """
    Normalize whitespace in a string.

    Collapses multiple spaces/tabs/newlines into single spaces and strips.
    """
    return " ".join(s.split())


def _normalize_text_unicode(text: str) -> str:
    """
    Normalize text for substring matching across Unicode variants.

    Handles:
    - Unicode normalization (NFKC)
    - Curly quotes -> straight quotes
    - En/em dashes -> hyphens
    - Whitespace normalization
    """
    import unicodedata

    if not text:
        return ""

    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Curly quotes -> straight
    text = text.replace("'", "'").replace("'", "'")
    text = text.replace(""", '"').replace(""", '"')

    # Dashes -> hyphen
    text = text.replace("–", "-").replace("—", "-")

    # Whitespace normalization
    text = _norm_ws(text)

    return text


def _is_deny_title(title: str) -> bool:
    """Check if a title should be denied for answer extraction."""
    if not title:
        return False
    t_lower = title.lower()
    return any(deny in t_lower for deny in DENY_TITLES)


# Relation triggers for different relation kinds (used as fallback when LLM router unavailable)
REL_TRIGGERS = {
    "DIRECTOR": ["directed by", "director", "film directed by", "directed", "filmmaker"],
    "BORN": ["was born", "born in", "birthplace", "native of", "born on"],
    "AWARD": ["won", "award", "nominated", "prize", "received", "honored"],
    "CREATED": ["created", "founded", "wrote", "written by", "author", "composed"],
    "LOCATION": ["located in", "capital of", "situated", "based in", "headquarters"],
    "MARRIAGE": ["married", "spouse", "wife", "husband", "wed"],
    "MOTHER": ["mother", "mom", "son of", "daughter of", "child of"],
    "FATHER": ["father", "dad", "son of", "daughter of", "child of"],
    "PARENT": ["parent", "mother", "father", "son of", "daughter of", "child of"],
    "SPOUSE": ["married", "spouse", "wife", "husband", "wed", "partner"],
    "NATIONALITY": ["nationality", "citizen", "national of", "from"],
    "BIRTHPLACE": ["born in", "birthplace", "native of", "birth place"],
}

# All known relation types for LLM router
KNOWN_RELATION_TYPES = [
    "DIRECTOR", "PRODUCER", "CREATOR", "AUTHOR", "COMPOSER", "PERFORMER",
    "MOTHER", "FATHER", "PARENT", "CHILD", "SPOUSE", "SIBLING",
    "BIRTHPLACE", "BIRTHDATE", "NATIONALITY", "OCCUPATION",
    "AWARD", "LOCATION", "CAPITAL", "HEADQUARTERS",
    "OTHER"
]


def llm_route_relation(
    llm,
    question: str,
    available_types: Optional[List[str]] = None
) -> Tuple[str, float]:
    """
    Use LLM to determine the primary relation type being asked about.

    This replaces keyword-based relation detection with semantic understanding.

    Args:
        llm: LLM interface with generate() method
        question: The question to analyze
        available_types: Optional list of relation types to choose from

    Returns:
        Tuple of (relation_type, confidence)
    """
    import json as json_module
    import os

    debug = os.environ.get("TRIDENT_DEBUG_ROUTER", "0") == "1"

    if available_types is None:
        available_types = KNOWN_RELATION_TYPES

    types_str = ", ".join(available_types)

    prompt = f"""Identify the relation type being asked about in this question.

Question: {question}

Available relation types: {types_str}

Rules:
1. Pick the SINGLE most specific relation type
2. For "who is the mother of X" -> MOTHER (not PARENT)
3. For "where was X born" -> BIRTHPLACE
4. For "who directed X" -> DIRECTOR
5. Return JSON: {{"relation": "TYPE", "confidence": 0.9}}

JSON:"""

    try:
        raw = llm.generate(prompt)
        raw_text = raw.text if hasattr(raw, 'text') else str(raw)
    except Exception as e:
        if debug:
            print(f"[LLM-ROUTER] Error: {e}")
        return keyword_route_relation(question), 0.5

    if debug:
        print(f"[LLM-ROUTER] Raw output: {raw_text[:100]}...")

    # Parse JSON response
    try:
        json_match = re.search(r'\{[^{}]*\}', raw_text, flags=re.DOTALL)
        if json_match:
            out = json_module.loads(json_match.group(0))
        else:
            out = json_module.loads(raw_text.strip())
    except Exception as e:
        if debug:
            print(f"[LLM-ROUTER] Parse error: {e}")
        return keyword_route_relation(question), 0.5

    relation = (out.get("relation") or "").upper().strip()
    confidence = float(out.get("confidence", 0.7))

    # Validate relation type
    if relation not in available_types and relation != "OTHER":
        # Try to find closest match
        for t in available_types:
            if t in relation or relation in t:
                relation = t
                break
        else:
            # Fallback to keyword-based
            if debug:
                print(f"[LLM-ROUTER] Unknown relation '{relation}', falling back to keywords")
            return keyword_route_relation(question), 0.5

    if debug:
        print(f"[LLM-ROUTER] Result: {relation} (confidence={confidence})")

    return relation, confidence


def keyword_route_relation(question: str) -> str:
    """
    Fallback keyword-based relation routing.

    Used when LLM router is unavailable or fails.
    """
    q_lower = question.lower()

    # Check each relation type's triggers
    for relation_type, triggers in REL_TRIGGERS.items():
        for trigger in triggers:
            if trigger in q_lower:
                return relation_type

    return "OTHER"


def route_relation(
    question: str,
    llm: Optional[Any] = None,
    use_llm: bool = True
) -> Tuple[str, float, str]:
    """
    Route question to relation type using LLM (if available) or keywords.

    Args:
        question: The question to analyze
        llm: Optional LLM interface
        use_llm: Whether to use LLM routing (default True)

    Returns:
        Tuple of (relation_type, confidence, method)
        method is "llm" or "keyword"
    """
    if use_llm and llm is not None:
        try:
            relation, confidence = llm_route_relation(llm, question)
            return relation, confidence, "llm"
        except Exception:
            pass

    # Fallback to keyword routing
    relation = keyword_route_relation(question)
    return relation, 0.5, "keyword"


def _looks_like_person(name: str) -> bool:
    """
    General PERSON-ish validator for CSS entity binding.

    This is NOT relation-specific - it's a lightweight type check that accepts
    strings that look like person names and rejects obvious non-names.

    Accepts: "Xawery Żuławski", "John Smith", "Madonna", "李明"
    Rejects: "Polish film", "the movie", "123", "", "directed by someone"
    """
    if not name or len(name) < 2:
        return False

    # Must contain at least one letter (handles Unicode)
    if not re.search(r"[A-Za-zÀ-ÖØ-öø-ÿ\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]", name):
        return False

    # Split into tokens
    toks = [t for t in re.split(r"\s+", name.strip()) if t]
    if not toks:
        return False

    # Single-token names must start with uppercase (or be CJK)
    if len(toks) == 1:
        first_char = toks[0][0]
        # Allow CJK characters
        if re.match(r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]", first_char):
            return True
        # Must start with uppercase for Latin names
        return bool(re.match(r"^[A-ZÀ-ÖØ-Þ]", toks[0]))

    # Multi-token: reject if starts with true lowercase (not uppercase/CJK)
    first_char = name[0]
    if re.match(r"^[a-zà-öø-ÿ]", first_char):
        return False

    # Too many tokens is suspicious (likely a description, not a name)
    if len(toks) > 5:
        return False

    # Reject obvious non-name patterns (case-insensitive for these)
    bad_words = {"film", "movie", "city", "country", "war", "book", "song",
                 "album", "series", "the", "directed", "starring", "featuring"}
    toks_lower = [t.lower().strip(".,()") for t in toks]
    if any(t in bad_words for t in toks_lower):
        return False

    return True


@dataclass
class ChainHop:
    """A single hop in the reasoning chain."""
    passage_id: str
    passage_text: str
    passage_title: Optional[str]
    entities: Set[str]
    relation_kinds: Set[str]
    hop_number: int  # 1 for hop1, 2 for hop2


@dataclass
class ReasoningChain:
    """A complete multi-hop reasoning chain."""
    hop1: ChainHop
    bridge_entity: Optional[str]
    hop2: ChainHop
    score: float

    def get_ordered_passages(self) -> List[Dict[str, Any]]:
        """Return passages in hop order for prompt building."""
        return [
            {
                'pid': self.hop1.passage_id,
                'text': self.hop1.passage_text,
                'title': self.hop1.passage_title,
                'hop': 1,
                'bridge_entity': self.bridge_entity,
            },
            {
                'pid': self.hop2.passage_id,
                'text': self.hop2.passage_text,
                'title': self.hop2.passage_title,
                'hop': 2,
                'bridge_entity': self.bridge_entity,
            }
        ]


def extract_entities(text: str, doc_title: Optional[str] = None) -> Set[str]:
    """Extract entity mentions from text.

    Uses Title Case spans and optionally includes document title.
    """
    ents = set(ENTITY_RE.findall(text))
    if doc_title:
        ents.add(doc_title)
        # Also add normalized version
        ents.add(doc_title.strip())
    return ents


def extract_relation_kinds(text: str) -> Set[str]:
    """Identify which relation kinds are mentioned in the passage."""
    t = text.lower()
    hits = set()
    for kind, triggers in REL_TRIGGERS.items():
        if any(trigger in t for trigger in triggers):
            hits.add(kind)
    return hits


def get_question_entities(question: str, facets: List[Dict[str, Any]]) -> Set[str]:
    """Extract entities from question and facets.

    Uses ENTITY facets if available, otherwise falls back to regex.
    """
    q_ents = set()

    # Extract from ENTITY facets
    for facet in facets:
        if facet.get('facet_type') == 'ENTITY':
            template = facet.get('template', {})
            mention = template.get('mention', '')
            if mention:
                q_ents.add(mention)

    # Also extract from RELATION facets
    for facet in facets:
        if facet.get('facet_type') == 'RELATION':
            template = facet.get('template', {})
            subj = template.get('subject', '')
            obj = template.get('object', '')
            # Clean WH-words
            if subj and subj.lower() not in {'who', 'what', 'where', 'when', 'which'}:
                q_ents.add(subj)
            if obj:
                # Extract entity from object like "director of film Polish-Russian War"
                obj_ents = extract_entities(obj)
                q_ents.update(obj_ents)

    # Fallback: regex on question
    q_ents.update(extract_entities(question))

    return q_ents


def get_required_relation_kinds(facets: List[Dict[str, Any]]) -> Set[str]:
    """Get required relation kinds from facets."""
    required = set()

    for facet in facets:
        if facet.get('facet_type') == 'RELATION':
            template = facet.get('template', {})
            # Check predicate and full facet text
            predicate = template.get('predicate', '').lower()
            facet_text = f"{template.get('subject', '')} {template.get('object', '')} {predicate}".lower()

            for kind, triggers in REL_TRIGGERS.items():
                if any(t in facet_text for t in triggers):
                    required.add(kind)

    return required


def build_chain_from_certified(
    certified_passages: List[Dict[str, Any]],
    question: str,
    facets: List[Dict[str, Any]],
    certificates: Optional[List[Dict[str, Any]]] = None
) -> Optional[ReasoningChain]:
    """Build a multi-hop reasoning chain from certified passages.

    Args:
        certified_passages: Passages that have been certified by Safe-Cover
        question: The original question
        facets: List of facet dicts
        certificates: Optional certificate info for filtering

    Returns:
        ReasoningChain if a valid chain is found, None otherwise
    """
    if len(certified_passages) < 2:
        # Need at least 2 passages for a 2-hop chain
        # If only 1 passage, return it as a single-hop chain (no bridge required)
        if len(certified_passages) == 1:
            p = certified_passages[0]
            ents = extract_entities(p.get('text', ''), p.get('title'))
            kinds = extract_relation_kinds(p.get('text', ''))
            hop = ChainHop(
                passage_id=p.get('pid', ''),
                passage_text=p.get('text', ''),
                passage_title=p.get('title'),
                entities=ents,
                relation_kinds=kinds,
                hop_number=1
            )
            # Single passage chain - explicit None for bridge
            return ReasoningChain(hop1=hop, bridge_entity=None, hop2=hop, score=0.5)
        return None

    # Get question entities and required relation kinds
    q_ents = get_question_entities(question, facets)
    required_kinds = get_required_relation_kinds(facets)

    # Build hop info for each passage
    hops = []
    for p in certified_passages:
        text = p.get('text', '')
        title = p.get('title')
        ents = extract_entities(text, title)
        kinds = extract_relation_kinds(text)
        hops.append(ChainHop(
            passage_id=p.get('pid', ''),
            passage_text=text,
            passage_title=title,
            entities=ents,
            relation_kinds=kinds,
            hop_number=0  # Will be set later
        ))

    # Index entities to passages
    ent2hop = defaultdict(list)
    for i, hop in enumerate(hops):
        for ent in hop.entities:
            ent2hop[ent].append(i)

    # Find bridge entities (appear in >= 2 passages)
    bridges = [ent for ent, idxs in ent2hop.items() if len(set(idxs)) >= 2]

    # Score candidate chains
    best_chain = None
    best_score = -1

    for bridge in bridges:
        passage_indices = list(set(ent2hop[bridge]))

        # Try all pairs of passages containing the bridge
        for i in passage_indices:
            for j in passage_indices:
                if i == j:
                    continue

                hop1 = hops[i]
                hop2 = hops[j]

                # Score this chain
                score = 0.0

                # Hop1 should connect to question entities
                if hop1.entities & q_ents:
                    score += 2.0

                # Hop2 should have required relation kind
                if required_kinds and (hop2.relation_kinds & required_kinds):
                    score += 2.0
                elif not required_kinds:
                    score += 1.0  # No specific requirement

                # Bonus for distinct passages
                score += 1.0

                # Bonus if bridge entity is in the question
                if bridge in q_ents:
                    score += 0.5

                if score > best_score:
                    best_score = score
                    hop1_copy = ChainHop(
                        passage_id=hop1.passage_id,
                        passage_text=hop1.passage_text,
                        passage_title=hop1.passage_title,
                        entities=hop1.entities,
                        relation_kinds=hop1.relation_kinds,
                        hop_number=1
                    )
                    hop2_copy = ChainHop(
                        passage_id=hop2.passage_id,
                        passage_text=hop2.passage_text,
                        passage_title=hop2.passage_title,
                        entities=hop2.entities,
                        relation_kinds=hop2.relation_kinds,
                        hop_number=2
                    )
                    best_chain = ReasoningChain(
                        hop1=hop1_copy,
                        bridge_entity=bridge,
                        hop2=hop2_copy,
                        score=best_score
                    )

    # If no bridge found, fall back to ordering by question entity overlap
    if best_chain is None and len(hops) >= 2:
        # Order by overlap with question entities (descending)
        hops_sorted = sorted(
            enumerate(hops),
            key=lambda x: len(x[1].entities & q_ents),
            reverse=True
        )
        idx1, hop1 = hops_sorted[0]
        idx2, hop2 = hops_sorted[1] if len(hops_sorted) > 1 else hops_sorted[0]

        # Find a common entity as bridge
        common = hop1.entities & hop2.entities
        # Use None if no common bridge exists to avoid prompt hallucinations
        bridge = next(iter(common), None)

        hop1_copy = ChainHop(
            passage_id=hop1.passage_id,
            passage_text=hop1.passage_text,
            passage_title=hop1.passage_title,
            entities=hop1.entities,
            relation_kinds=hop1.relation_kinds,
            hop_number=1
        )
        hop2_copy = ChainHop(
            passage_id=hop2.passage_id,
            passage_text=hop2.passage_text,
            passage_title=hop2.passage_title,
            entities=hop2.entities,
            relation_kinds=hop2.relation_kinds,
            hop_number=2
        )
        best_chain = ReasoningChain(
            hop1=hop1_copy,
            bridge_entity=bridge,
            hop2=hop2_copy,
            score=0.0
        )

    return best_chain


def extract_grounded_answer(
    llm_answer: str,
    hop2_text: str,
    min_overlap_ratio: float = 0.5
) -> Tuple[str, bool]:
    """Extract answer that is grounded in hop2 passage.

    Args:
        llm_answer: Raw answer from LLM
        hop2_text: Text of the hop2 passage
        min_overlap_ratio: Minimum token overlap for grounding

    Returns:
        Tuple of (grounded_answer, is_grounded)
    """
    if not llm_answer or not hop2_text:
        return llm_answer, False

    # Helper for normalization
    def norm(s):
        s = s.lower().strip()
        # Remove punctuation
        s = s.translate(str.maketrans("", "", string.punctuation))
        # Normalize whitespace
        s = " ".join(s.split())
        return s

    # 1. Exact substring check (original)
    answer_lower = llm_answer.lower().strip()
    hop2_lower = hop2_text.lower()

    if answer_lower in hop2_lower:
        return llm_answer.strip(), True

    # 2. Normalized substring check (punctuation agnostic)
    norm_answer = norm(llm_answer)
    norm_hop2 = norm(hop2_text)

    if norm_answer and norm_answer in norm_hop2:
        return llm_answer.strip(), True

    # 3. Sliding window token overlap (fallback)
    # Try to find the best matching substring
    answer_tokens = answer_lower.split()
    if not answer_tokens:
        return llm_answer, False

    # Sliding window to find best match
    hop2_tokens = hop2_lower.split()
    best_match = None
    best_overlap = 0

    for window_size in range(len(answer_tokens), 0, -1):
        for start in range(len(hop2_tokens) - window_size + 1):
            window = hop2_tokens[start:start + window_size]
            overlap = len(set(answer_tokens) & set(window))
            overlap_ratio = overlap / len(answer_tokens)

            if overlap_ratio >= min_overlap_ratio and overlap > best_overlap:
                best_overlap = overlap
                best_match = ' '.join(hop2_text.split()[start:start + window_size])

    if best_match:
        return best_match, True

    # Fall back to original answer with grounding flag
    return llm_answer.strip(), False


def build_single_hop_prompt(question: str, chain: ReasoningChain) -> str:
    """Build a simplified prompt for single-passage questions."""
    prompt_parts = []
    prompt_parts.append("Answer the following question using ONLY the provided evidence passage.")
    prompt_parts.append("")
    prompt_parts.append(f"Question: {question}")
    prompt_parts.append("")
    prompt_parts.append("=== Evidence ===")
    if chain.hop1.passage_title:
        prompt_parts.append(f"[{chain.hop1.passage_title}]")
    prompt_parts.append(chain.hop1.passage_text)
    prompt_parts.append("")
    prompt_parts.append("Instructions:")
    prompt_parts.append("1. The answer must be found in the Evidence passage")
    prompt_parts.append("2. Extract the exact answer span")
    prompt_parts.append("3. If the answer cannot be found, respond with: I cannot answer based on the given context.")
    prompt_parts.append("")
    prompt_parts.append("Answer:")
    return "\n".join(prompt_parts)


def build_chain_prompt(
    question: str,
    chain: ReasoningChain,
    facets: Optional[List[Dict[str, Any]]] = None
) -> str:
    """Build a prompt that uses the reasoning chain structure.

    This prompt explicitly shows the hop structure and asks the LLM
    to extract an answer that is grounded in the evidence.
    """
    # Use simplified prompt for single-hop cases (Hop1 == Hop2)
    if chain.hop1.passage_id == chain.hop2.passage_id:
        return build_single_hop_prompt(question, chain)

    prompt_parts = []
    prompt_parts.append("Answer the following question using ONLY the provided evidence passages.")
    prompt_parts.append("The passages are ordered as a reasoning chain: Hop 1 introduces context, Hop 2 contains the answer.")
    prompt_parts.append("")
    prompt_parts.append(f"Question: {question}")
    prompt_parts.append("")

    # Only include Bridge Entity if it exists (multi-hop with overlap)
    if chain.bridge_entity:
        prompt_parts.append(f"Bridge Entity: {chain.bridge_entity}")
        prompt_parts.append("")
        
    prompt_parts.append("=== HOP 1 (Context) ===")
    if chain.hop1.passage_title:
        prompt_parts.append(f"[{chain.hop1.passage_title}]")
    prompt_parts.append(chain.hop1.passage_text)
    prompt_parts.append("")
    prompt_parts.append("=== HOP 2 (Answer Source) ===")
    if chain.hop2.passage_title:
        prompt_parts.append(f"[{chain.hop2.passage_title}]")
    prompt_parts.append(chain.hop2.passage_text)
    prompt_parts.append("")
    prompt_parts.append("Instructions:")
    prompt_parts.append("1. If the answer is stated verbatim in the evidence, extract the EXACT span (copy it exactly).")
    prompt_parts.append("2. Return ONLY the answer - no explanation, no extra words.")
    prompt_parts.append("3. Preserve accents and special characters exactly as written.")
    prompt_parts.append("4. If the answer is not stated in the evidence, respond exactly: I cannot answer based on the given context.")
    prompt_parts.append("")
    prompt_parts.append("Answer:")

    return "\n".join(prompt_parts)


def get_winning_passages_from_certificates(
    certificates: List[Dict[str, Any]],
    passages: List[Dict[str, Any]],
    facets: List[Dict[str, Any]]
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    """
    Map facet_id -> winning passage, plus facet_type -> [winning passages].

    CRITICAL: Maps by facet_id to preserve all facets (a query can have multiple
    RELATION or ENTITY facets). Also provides a grouped view by facet_type.

    Returns:
        Tuple of:
        - facet_id_map: Dict[facet_id -> {'passage', 'facet', 'p_value', 'passage_id', 'facet_id'}]
        - facet_type_map: Dict[facet_type -> List[winning info dicts]] (stable order by p_value)
    """
    if not certificates:
        return {}, {}

    # Build passage lookup by pid
    pid_to_passage = {p.get('pid', ''): p for p in passages}

    # Build facet lookup by facet_id
    fid_to_facet = {f.get('facet_id', ''): f for f in facets}

    # Primary map: facet_id -> winning info
    facet_id_map: Dict[str, Dict[str, Any]] = {}

    for cert in certificates:
        fid = cert.get('facet_id', '')
        pid = cert.get('passage_id', '')
        p_val = cert.get('p_value', 1.0)

        facet = fid_to_facet.get(fid, {})
        passage = pid_to_passage.get(pid, {})

        if not facet or not passage:
            continue

        ftype = facet.get('facet_type', '')

        # Only keep best certificate per facet_id (in case of duplicates)
        if fid not in facet_id_map or p_val < facet_id_map[fid]['p_value']:
            facet_id_map[fid] = {
                'passage': passage,
                'facet': facet,
                'p_value': p_val,
                'passage_id': pid,
                'facet_id': fid,
                'facet_type': ftype
            }

    # Derived map: facet_type -> [winning info dicts] (sorted by p_value)
    facet_type_map: Dict[str, List[Dict[str, Any]]] = {}
    for info in facet_id_map.values():
        ftype = info['facet_type']
        if ftype not in facet_type_map:
            facet_type_map[ftype] = []
        facet_type_map[ftype].append(info)

    # Sort each list by p_value (best first)
    for ftype in facet_type_map:
        facet_type_map[ftype].sort(key=lambda x: x['p_value'])

    return facet_id_map, facet_type_map


def build_certificate_aware_prompt(
    question: str,
    certificates: List[Dict[str, Any]],
    passages: List[Dict[str, Any]],
    facets: List[Dict[str, Any]]
) -> Tuple[str, Optional[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Build a prompt using the certificate-winning passages.

    CRITICAL: Uses the passages that actually certified each facet,
    not arbitrary chain hops based on entity overlap. Includes ALL
    winning passages (not just one per facet type).

    Returns:
        Tuple of:
        - prompt_string: The constructed prompt
        - relation_winning_info: The best RELATION facet winner (for typed extraction)
        - facet_id_map: Complete facet_id -> winning info map (for debugging/validation)
    """
    facet_id_map, facet_type_map = get_winning_passages_from_certificates(
        certificates, passages, facets
    )

    if not facet_id_map:
        # Fallback: no certificates, can't build certificate-aware prompt
        return "", None, {}

    # Get ALL RELATION-winning passages (sorted by p_value, best first)
    relation_winners = facet_type_map.get('RELATION', [])
    entity_winners = facet_type_map.get('ENTITY', [])

    # Also check for BRIDGE_HOP types
    bridge_winners = []
    for ftype in ['BRIDGE_HOP', 'BRIDGE_HOP1', 'BRIDGE_HOP2']:
        if ftype in facet_type_map:
            bridge_winners.extend(facet_type_map[ftype])

    # The best RELATION winner (lowest p-value) is the primary answer source
    best_relation_info = relation_winners[0] if relation_winners else None

    prompt_parts = []
    prompt_parts.append("Answer the following question using ONLY the provided certified evidence.")
    prompt_parts.append("These passages have been verified to contain the information needed to answer.")
    prompt_parts.append("")
    prompt_parts.append(f"Question: {question}")
    prompt_parts.append("")

    # Include ALL winning passages (deduplicated by pid)
    seen_pids = set()
    passage_num = 1

    # ENTITY passages first (context)
    for info in entity_winners:
        p = info['passage']
        pid = p.get('pid', '')
        if pid and pid not in seen_pids:
            seen_pids.add(pid)
            title = p.get('title', '')
            text = p.get('text', '')
            prompt_parts.append(f"=== Evidence {passage_num} (Context) ===")
            if title:
                prompt_parts.append(f"[{title}]")
            prompt_parts.append(text)
            prompt_parts.append("")
            passage_num += 1

    # BRIDGE passages (if different from ENTITY)
    for info in bridge_winners:
        p = info['passage']
        pid = p.get('pid', '')
        if pid and pid not in seen_pids:
            seen_pids.add(pid)
            title = p.get('title', '')
            text = p.get('text', '')
            prompt_parts.append(f"=== Evidence {passage_num} (Bridge) ===")
            if title:
                prompt_parts.append(f"[{title}]")
            prompt_parts.append(text)
            prompt_parts.append("")
            passage_num += 1

    # RELATION passages (answer sources - MOST IMPORTANT!)
    for i, info in enumerate(relation_winners):
        p = info['passage']
        pid = p.get('pid', '')
        if pid and pid not in seen_pids:
            seen_pids.add(pid)
            title = p.get('title', '')
            text = p.get('text', '')
            label = "Answer Source" if i == 0 else f"Answer Source {i + 1}"
            prompt_parts.append(f"=== Evidence {passage_num} ({label}) ===")
            if title:
                prompt_parts.append(f"[{title}]")
            prompt_parts.append(text)
            prompt_parts.append("")
            passage_num += 1
        elif passage_num == 1 and i == 0:
            # RELATION is the only passage - still show it
            title = p.get('title', '')
            text = p.get('text', '')
            prompt_parts.append("=== Certified Evidence ===")
            if title:
                prompt_parts.append(f"[{title}]")
            prompt_parts.append(text)
            prompt_parts.append("")
            passage_num += 1

    # If no passages were added, bail
    if passage_num == 1:
        return "", None, facet_id_map

    prompt_parts.append("Instructions:")
    prompt_parts.append("1. If the answer is stated verbatim in the evidence, extract the EXACT span (copy it exactly).")
    prompt_parts.append("2. Return ONLY the answer - no explanation, no extra words.")
    prompt_parts.append("3. Preserve accents and special characters exactly as written.")
    prompt_parts.append("4. If the answer is not stated in the evidence, respond exactly: I cannot answer based on the given context.")
    prompt_parts.append("")
    prompt_parts.append("Answer:")

    return "\n".join(prompt_parts), best_relation_info, facet_id_map


def bind_entity_from_hop1_winner(
    relation_type: str,
    passage_text: str
) -> Optional[str]:
    """
    Typed binding of the intermediate entity from hop-1 winner passage.

    This is NOT answer extraction - it's binding a variable for hop-2.
    Much easier because we're extracting a named entity from one known relation.

    Args:
        relation_type: The inner relation type (DIRECTOR, AUTHOR, etc.)
        passage_text: The text of the hop-1 winning passage

    Returns:
        The bound entity name (e.g., "Xawery Żuławski"), or None if not found
    """
    if not passage_text:
        return None

    t = passage_text
    t_lower = t.lower()

    if relation_type == "DIRECTOR":
        # Must have "directed" to extract director name
        if "directed" not in t_lower and "director" not in t_lower:
            return None

        # Pattern: "directed by Name" (most reliable)
        m = re.search(r"(?i)\bdirected\s+by\s+([A-Z][a-zA-ZÀ-ÿ]+(?:\s+[A-Z][a-zA-ZÀ-ÿ]+){0,3})(?:\s*[,\.\(\)]|$|\s+(?:is|was|and|who))", t)
        if m:
            name = m.group(1).strip()
            words = name.split()
            if 1 <= len(words) <= 4:
                return name

        # Pattern: "Name directed" or "Name, who directed"
        m = re.search(r"(?i)([A-Z][a-zA-ZÀ-ÿ]+(?:\s+[A-Z][a-zA-ZÀ-ÿ]+){0,3})\s+(?:,\s*who\s+)?directed\b", t)
        if m:
            name = m.group(1).strip()
            words = name.split()
            if 1 <= len(words) <= 4 and words[0].lower() not in {'the', 'a', 'an', 'film', 'movie'}:
                return name

        # Pattern: "X is an Australian director" -> extract X
        m = re.search(r"(?i)([A-Z][a-zA-ZÀ-ÿ]+(?:\s+[A-Z][a-zA-ZÀ-ÿ]+){0,3})\s+(?:is|was)\s+(?:an?\s+)?(?:\w+\s+)?director", t)
        if m:
            name = m.group(1).strip()
            words = name.split()
            if 1 <= len(words) <= 4:
                return name

    elif relation_type == "AUTHOR":
        if "written" not in t_lower and "wrote" not in t_lower and "author" not in t_lower:
            return None

        m = re.search(r"(?i)\bwritten\s+by\s+([A-Z][a-zA-ZÀ-ÿ]+(?:\s+[A-Z][a-zA-ZÀ-ÿ]+){0,3})(?:\s*[,\.\(\)]|$)", t)
        if m:
            return m.group(1).strip()

        m = re.search(r"(?i)([A-Z][a-zA-ZÀ-ÿ]+(?:\s+[A-Z][a-zA-ZÀ-ÿ]+){0,3})\s+wrote\b", t)
        if m:
            return m.group(1).strip()

    elif relation_type == "CREATOR":
        if "created" not in t_lower and "founder" not in t_lower:
            return None

        m = re.search(r"(?i)\bcreated\s+by\s+([A-Z][a-zA-ZÀ-ÿ]+(?:\s+[A-Z][a-zA-ZÀ-ÿ]+){0,3})(?:\s*[,\.\(\)]|$)", t)
        if m:
            return m.group(1).strip()

        m = re.search(r"(?i)\bfounded\s+by\s+([A-Z][a-zA-ZÀ-ÿ]+(?:\s+[A-Z][a-zA-ZÀ-ÿ]+){0,3})(?:\s*[,\.\(\)]|$)", t)
        if m:
            return m.group(1).strip()

    elif relation_type == "PRODUCER":
        if "produced" not in t_lower and "producer" not in t_lower:
            return None

        m = re.search(r"(?i)\bproduced\s+by\s+([A-Z][a-zA-ZÀ-ÿ]+(?:\s+[A-Z][a-zA-ZÀ-ÿ]+){0,3})(?:\s*[,\.\(\)]|$)", t)
        if m:
            return m.group(1).strip()

    elif relation_type == "COMPOSER":
        if "composed" not in t_lower and "composer" not in t_lower:
            return None

        m = re.search(r"(?i)\bcomposed\s+by\s+([A-Z][a-zA-ZÀ-ÿ]+(?:\s+[A-Z][a-zA-ZÀ-ÿ]+){0,3})(?:\s*[,\.\(\)]|$)", t)
        if m:
            return m.group(1).strip()

    elif relation_type == "PERFORMER":
        if "starred" not in t_lower and "performed" not in t_lower and "starring" not in t_lower:
            return None

        m = re.search(r"(?i)\bstarring\s+([A-Z][a-zA-ZÀ-ÿ]+(?:\s+[A-Z][a-zA-ZÀ-ÿ]+){0,3})(?:\s*[,\.\(\)]|$)", t)
        if m:
            return m.group(1).strip()

    return None


def typed_extract_from_winning_passage(
    question: str,
    relation_info: Dict[str, Any]
) -> Optional[str]:
    """
    Facet-ID–aware typed extraction from the RELATION-winning passage.

    CRITICAL FIXES:
    1. Uses facet template (subject/object/predicate) to constrain extraction
    2. Checks semantic presence before extracting (abort if answer type absent)
    3. Anchors on known entity spans from the facet template
    4. Returns None to allow controlled abstention rather than garbage

    Args:
        question: The original question
        relation_info: Dict with 'passage', 'facet', 'p_value', etc.

    Returns:
        Extracted answer string, or None if extraction not possible
    """
    if not relation_info:
        return None

    passage = relation_info.get('passage', {})
    facet = relation_info.get('facet', {})
    text = passage.get('text', '')

    if not text:
        return None

    # Get facet template - this is the key to facet-ID awareness
    template = facet.get('template', {})
    subject = template.get('subject', '').strip()
    obj = template.get('object', '').strip()
    predicate = template.get('predicate', '').strip()

    # Build context for relation kind detection
    facet_text = f"{subject} {obj} {predicate}".lower()

    # Determine relation kind from facet template
    relation_kind = None
    for kind, triggers in REL_TRIGGERS.items():
        if any(t in facet_text for t in triggers):
            relation_kind = kind
            break

    # Fallback: infer from question (less reliable)
    if not relation_kind:
        q = question.lower()
        if "director" in q or "directed" in q:
            relation_kind = "DIRECTOR"
        elif "born" in q or "birth" in q:
            relation_kind = "BORN"
        elif "award" in q or "won" in q or "prize" in q:
            relation_kind = "AWARD"
        elif "located" in q or "capital" in q:
            relation_kind = "LOCATION"
        elif "married" in q or "spouse" in q or "wife" in q or "husband" in q:
            relation_kind = "MARRIAGE"

    if not relation_kind:
        return None

    t = text
    t_lower = t.lower()
    q = question.lower()

    # ==== SEMANTIC PRESENCE CHECKS ====
    # Abort extraction if the answer type is not present in the passage

    if relation_kind == "DIRECTOR":
        # Must have "directed" or "director" to extract a director name
        if "directed" not in t_lower and "director" not in t_lower:
            return None

        # Extract PERSON name after "directed by" (most reliable pattern)
        m = re.search(r"(?i)\bdirected\s+by\s+([A-Z][a-zA-ZÀ-ÿ]+(?:\s+[A-Z][a-zA-ZÀ-ÿ]+){0,3})(?:\s*[,\.\(\)]|$|\s+(?:is|was|and|who))", t)
        if m:
            name = m.group(1).strip()
            # Validate: should be 2-4 words, not start with common non-name words
            words = name.split()
            if 1 <= len(words) <= 4 and words[0].lower() not in {'the', 'a', 'an', 'this', 'that', 'film', 'movie', 'polish', 'australian'}:
                return name

        # Pattern: "Name directed the film" or "Name, who directed"
        m = re.search(r"(?i)([A-Z][a-zA-ZÀ-ÿ]+(?:\s+[A-Z][a-zA-ZÀ-ÿ]+){0,3})\s+(?:,\s*who\s+)?directed\b", t)
        if m:
            name = m.group(1).strip()
            words = name.split()
            if 1 <= len(words) <= 4 and words[0].lower() not in {'the', 'a', 'an', 'this', 'that', 'film', 'movie'}:
                return name

        # Pattern: "director Name" (less common but valid)
        m = re.search(r"(?i)\bdirector\s+([A-Z][a-zA-ZÀ-ÿ]+(?:\s+[A-Z][a-zA-ZÀ-ÿ]+){0,3})(?:\s*[,\.\(\)]|$)", t)
        if m:
            name = m.group(1).strip()
            words = name.split()
            if 1 <= len(words) <= 4 and words[0].lower() not in {'of', 'the', 'a', 'an', 'is', 'was'}:
                return name

    elif relation_kind == "BORN":
        if "where" in q or "place" in q:
            # Birthplace - must have "born" to extract
            if "born" not in t_lower:
                return None
            # Pattern: "born in Location"
            m = re.search(r"(?i)\bborn\s+in\s+([A-Z][a-zA-ZÀ-ÿ\s\-]+?)(?:\s*[,\.\(\)]|on\s|$)", t)
            if m:
                loc = m.group(1).strip().rstrip(",")
                # Validate: should be 1-4 words, look like a place
                if 1 <= len(loc.split()) <= 4:
                    return loc
        else:
            # Birth date - must have "born" and a year
            if "born" not in t_lower or not re.search(r'\d{4}', t):
                return None
            # Pattern: "born on/in Month Day, Year" or "born on Day Month Year"
            m = re.search(r"(?i)\bborn\s+(?:on\s+)?([A-Z]?[a-zA-Z]+\s+\d{1,2},?\s+\d{4})", t)
            if m:
                return m.group(1).strip()
            m = re.search(r"(?i)\bborn\s+(?:on\s+)?(\d{1,2}\s+[A-Z]?[a-zA-Z]+\s+\d{4})", t)
            if m:
                return m.group(1).strip()

    elif relation_kind == "AWARD":
        # CRITICAL: Must have award-related words to extract an award
        award_indicators = ['award', 'prize', 'medal', 'oscar', 'emmy', 'grammy', 'trophy', 'won', 'awarded', 'received']
        if not any(ind in t_lower for ind in award_indicators):
            return None  # Don't extract - passage doesn't contain award info

        # Pattern: "won the X Award/Prize"
        m = re.search(r"(?i)\bwon\s+(?:the\s+)?([A-Z][a-zA-ZÀ-ÿ\s\-]+?(?:Award|Prize|Medal|Oscar|Emmy|Grammy|Trophy))", t)
        if m:
            return m.group(1).strip()
        # Pattern: "received the X Award"
        m = re.search(r"(?i)\breceived\s+(?:the\s+)?([A-Z][a-zA-ZÀ-ÿ\s\-]+?(?:Award|Prize|Medal))", t)
        if m:
            return m.group(1).strip()
        # Pattern: "awarded the X"
        m = re.search(r"(?i)\bawarded\s+(?:the\s+)?([A-Z][a-zA-ZÀ-ÿ\s\-]+?(?:Award|Prize|Medal))", t)
        if m:
            return m.group(1).strip()

    elif relation_kind == "LOCATION":
        # Must have location-related words
        loc_indicators = ['located', 'capital', 'headquarters', 'based', 'situated']
        if not any(ind in t_lower for ind in loc_indicators):
            return None

        m = re.search(r"(?i)\blocated\s+in\s+([A-Z][a-zA-ZÀ-ÿ\s\-\,]+?)(?:\s*[,\.\(\)]|$)", t)
        if m:
            return m.group(1).strip().rstrip(",")
        m = re.search(r"(?i)\bcapital\s+(?:city\s+)?(?:is|of)\s+([A-Z][a-zA-ZÀ-ÿ\s\-]+?)(?:\s*[,\.\(\)]|$)", t)
        if m:
            return m.group(1).strip()
        m = re.search(r"(?i)\bheadquarters\s+(?:is|are)\s+(?:in\s+)?([A-Z][a-zA-ZÀ-ÿ\s\-\,]+?)(?:\s*[,\.\(\)]|$)", t)
        if m:
            return m.group(1).strip().rstrip(",")

    elif relation_kind == "MARRIAGE":
        # Must have marriage-related words
        if "married" not in t_lower and "spouse" not in t_lower and "wife" not in t_lower and "husband" not in t_lower:
            return None

        m = re.search(r"(?i)\bmarried\s+(?:to\s+)?([A-Z][a-zA-ZÀ-ÿ]+(?:\s+[A-Z][a-zA-ZÀ-ÿ]+){0,3})(?:\s*[,\.\(\)]|in\s|$)", t)
        if m:
            return m.group(1).strip()
        m = re.search(r"(?i)\bspouse\s+(?:is\s+)?([A-Z][a-zA-ZÀ-ÿ]+(?:\s+[A-Z][a-zA-ZÀ-ÿ]+){0,3})(?:\s*[,\.\(\)]|$)", t)
        if m:
            return m.group(1).strip()

    # No valid extraction possible - return None for controlled abstention
    return None


def regex_extract_answer_from_hop2(
    question: str,
    facets: List[Dict[str, Any]],
    hop2_text: str
) -> Optional[str]:
    """
    DEPRECATED: Use typed_extract_from_winning_passage instead.

    This function extracts from arbitrary hop2_text, which may NOT be
    the passage that certified the RELATION facet. It can produce
    garbage like "of film and TV" from unrelated passages.

    Kept for backwards compatibility but should be replaced.
    """
    if not hop2_text:
        return None

    t = hop2_text
    q = question.lower()

    # Check if we have RELATION facets
    has_relation = any(f.get("facet_type") == "RELATION" for f in facets)

    if has_relation:
        # DIRECTOR: "directed by X" or "X directed"
        if "director" in q or "directed" in q or "direct" in q:
            # Pattern: "directed by Name"
            m = re.search(r"(?i)\bdirected\s+by\s+([A-Z][a-zA-ZÀ-ÿ\s\-\.]+?)(?:\s*[,\.\(\)]|$)", t)
            if m:
                return m.group(1).strip()
            # Pattern: "director Name"
            m = re.search(r"(?i)\bdirector\s+([A-Z][a-zA-ZÀ-ÿ\s\-\.]+?)(?:\s*[,\.\(\)]|$)", t)
            if m:
                return m.group(1).strip()
            # Pattern: "film by Name"
            m = re.search(r"(?i)\bfilm\s+by\s+([A-Z][a-zA-ZÀ-ÿ\s\-\.]+?)(?:\s*[,\.\(\)]|$)", t)
            if m:
                return m.group(1).strip()

        # BORN: "born in X" or "was born on X"
        if "born" in q or "birth" in q:
            # Birthplace
            if "where" in q or "place" in q:
                m = re.search(r"(?i)\bborn\s+in\s+([A-Z][a-zA-ZÀ-ÿ\s\-\,]+?)(?:\s*[,\.\(\)]|on\s|$)", t)
                if m:
                    return m.group(1).strip().rstrip(",")
            # Birth date
            else:
                m = re.search(r"(?i)\bborn\s+(?:on\s+)?([A-Z]?[a-zA-Z0-9\s\,]+\d{4})", t)
                if m:
                    return m.group(1).strip()

        # AWARD: "won X" or "received X award"
        if "award" in q or "won" in q or "prize" in q:
            m = re.search(r"(?i)\bwon\s+(?:the\s+)?([A-Z][a-zA-ZÀ-ÿ\s\-]+?(?:Award|Prize|Medal|Oscar|Emmy|Grammy))", t)
            if m:
                return m.group(1).strip()
            m = re.search(r"(?i)\breceived\s+(?:the\s+)?([A-Z][a-zA-ZÀ-ÿ\s\-]+?(?:Award|Prize|Medal))", t)
            if m:
                return m.group(1).strip()

        # LOCATION: "located in X" or "capital of X is Y"
        if "located" in q or "capital" in q or "headquarters" in q:
            m = re.search(r"(?i)\blocated\s+in\s+([A-Z][a-zA-ZÀ-ÿ\s\-\,]+?)(?:\s*[,\.\(\)]|$)", t)
            if m:
                return m.group(1).strip().rstrip(",")
            m = re.search(r"(?i)\bcapital\s+(?:is|of)\s+([A-Z][a-zA-ZÀ-ÿ\s\-]+?)(?:\s*[,\.\(\)]|$)", t)
            if m:
                return m.group(1).strip()
            m = re.search(r"(?i)\bheadquarters\s+(?:is|are)\s+(?:in\s+)?([A-Z][a-zA-ZÀ-ÿ\s\-\,]+?)(?:\s*[,\.\(\)]|$)", t)
            if m:
                return m.group(1).strip().rstrip(",")

        # MARRIAGE: "married X" or "spouse X"
        if "married" in q or "spouse" in q or "wife" in q or "husband" in q:
            m = re.search(r"(?i)\bmarried\s+(?:to\s+)?([A-Z][a-zA-ZÀ-ÿ\s\-\.]+?)(?:\s*[,\.\(\)]|in\s|$)", t)
            if m:
                return m.group(1).strip()

    return None


# ==============================================================================
# CERTIFIED SPAN SELECTION (CSS)
# ==============================================================================
# General, relation-agnostic answer extraction that requires exact substring
# grounding in certified evidence. No per-relation regex patterns.
# ==============================================================================

@dataclass
class CSSResult:
    """Result from Certified Span Selection."""
    abstain: bool
    answer: str
    passage_id: str
    reason: str  # OK, NO_SPAN, SPAN_NOT_VERBATIM, PID_NOT_IN_WINNERS, PARSE_ERROR
    confidence: float = 0.0


def _find_exact_substring(haystack: str, needle: str) -> Optional[Tuple[int, int]]:
    """Find exact substring match (case-sensitive first, then case-insensitive)."""
    if not needle or not haystack:
        return None

    # Try case-sensitive first
    i = haystack.find(needle)
    if i >= 0:
        return (i, i + len(needle))

    # Try case-insensitive
    i = haystack.lower().find(needle.lower())
    if i >= 0:
        return (i, i + len(needle))

    return None


def _find_fuzzy_substring(haystack: str, needle: str, max_distance: int = 2) -> Optional[Tuple[int, int, str]]:
    """
    Find fuzzy substring match allowing for minor differences.

    Returns (start, end, matched_text) or None.
    Used as fallback when exact match fails due to punctuation/whitespace differences.
    """
    if not needle or not haystack:
        return None

    # Normalize both strings
    def normalize(s: str) -> str:
        s = s.lower().strip()
        s = re.sub(r'[^\w\s]', '', s)  # Remove punctuation
        s = ' '.join(s.split())  # Normalize whitespace
        return s

    needle_norm = normalize(needle)
    if not needle_norm:
        return None

    # Sliding window search
    words = haystack.split()
    needle_words = needle_norm.split()

    if not needle_words:
        return None

    for i in range(len(words) - len(needle_words) + 1):
        window = words[i:i + len(needle_words)]
        window_norm = [normalize(w) for w in window]

        # Check if normalized words match
        if window_norm == needle_words:
            # Find the actual span in the original text
            start_word = ' '.join(words[:i])
            matched_text = ' '.join(window)
            start_pos = len(start_word) + (1 if start_word else 0)
            return (start_pos, start_pos + len(matched_text), matched_text)

    return None


def certified_span_select(
    llm,
    question: str,
    winner_passages: List[Dict[str, Any]],
    max_chars_per_passage: int = 800
) -> CSSResult:
    """
    Certified Span Selection: Extract answer as exact substring from evidence.

    This is relation-agnostic and requires verbatim grounding.

    Args:
        llm: LLM interface with generate() method
        question: The question to answer
        winner_passages: List of certificate-winning passages [{pid, text, title}, ...]
        max_chars_per_passage: Max chars to include per passage (for context limits)

    Returns:
        CSSResult with answer span and verification status
    """
    import json as json_module
    import os

    debug = os.environ.get("TRIDENT_DEBUG_CSS", "0") == "1"

    if not winner_passages:
        return CSSResult(abstain=True, answer="", passage_id="", reason="NO_PASSAGES")

    # Build compact evidence
    evidence_parts = []
    for p in winner_passages:
        pid = p.get("pid", "")
        title = p.get("title", "")
        text = (p.get("text") or "")[:max_chars_per_passage]
        evidence_parts.append(f"[{pid}] {title}\n{text}")

    evidence_text = "\n\n".join(evidence_parts)

    prompt = f"""You must answer ONLY using the Evidence below.
Return a JSON object with keys:
  "pid": string, the evidence passage id you copied from
  "answer": string, an EXACT substring copied verbatim from that passage
  "confidence": number between 0 and 1

Rules:
1) The answer MUST be copied exactly from the evidence text (verbatim).
2) Copy the shortest complete answer span - usually a name, place, date, or short phrase.
3) If you cannot find an exact answer substring, output: {{"pid": "", "answer": "", "confidence": 0}}

Question: {question}

Evidence:
{evidence_text}

JSON:"""

    try:
        raw = llm.generate(prompt)
        raw_text = raw.text if hasattr(raw, 'text') else str(raw)
    except Exception as e:
        if debug:
            print(f"[CSS] LLM generation failed: {e}")
        return CSSResult(abstain=True, answer="", passage_id="", reason="LLM_ERROR")

    if debug:
        print(f"[CSS] Raw LLM output: {raw_text[:200]}...")

    # Parse JSON response
    try:
        # Try to find JSON object in response
        json_match = re.search(r'\{[^{}]*\}', raw_text, flags=re.DOTALL)
        if json_match:
            out = json_module.loads(json_match.group(0))
        else:
            out = json_module.loads(raw_text.strip())
    except Exception as e:
        if debug:
            print(f"[CSS] JSON parse failed: {e}")
        return CSSResult(abstain=True, answer="", passage_id="", reason="PARSE_ERROR")

    pid = (out.get("pid") or "").strip()
    answer = (out.get("answer") or "").strip()
    confidence = float(out.get("confidence", 0))

    if debug:
        print(f"[CSS] Parsed: pid={pid}, answer={answer[:50]}..., conf={confidence}")

    # Check for empty response (LLM abstention)
    if not pid or not answer:
        return CSSResult(abstain=True, answer="", passage_id=pid, reason="NO_SPAN", confidence=confidence)

    # Verify passage ID is in winners
    passage_map = {p.get("pid", ""): p for p in winner_passages}
    if pid not in passage_map:
        if debug:
            print(f"[CSS] PID {pid} not in winners: {list(passage_map.keys())}")
        return CSSResult(abstain=True, answer=answer, passage_id=pid, reason="PID_NOT_IN_WINNERS", confidence=confidence)

    # Verify exact grounding
    passage_text = passage_map[pid].get("text", "")

    # Try exact match first
    match = _find_exact_substring(passage_text, answer)
    if match is not None:
        if debug:
            print(f"[CSS] Exact match found at {match}")
        return CSSResult(abstain=False, answer=answer, passage_id=pid, reason="OK", confidence=confidence)

    # Try fuzzy match as fallback
    fuzzy = _find_fuzzy_substring(passage_text, answer)
    if fuzzy is not None:
        start, end, matched_text = fuzzy
        if debug:
            print(f"[CSS] Fuzzy match: '{answer}' -> '{matched_text}'")
        # Return the actual text from passage (properly grounded)
        return CSSResult(abstain=False, answer=matched_text, passage_id=pid, reason="OK_FUZZY", confidence=confidence)

    if debug:
        print(f"[CSS] No match found for '{answer}' in passage")
    return CSSResult(abstain=True, answer=answer, passage_id=pid, reason="SPAN_NOT_VERBATIM", confidence=confidence)


# ==============================================================================
# CONSTRAINED CANDIDATE SELECTION
# ==============================================================================
# Instead of free-form extraction, build a candidate set and have LLM pick by
# index. This ensures verbatim answers and enables support-based ranking.
# ==============================================================================

@dataclass
class QuestionType:
    """Detected question type for constraining candidates."""
    category: str  # "either_or", "who", "what", "where", "when", "how_many", "yes_no", "other"
    expected_type: str  # "PERSON", "PLACE", "DATE", "NUMBER", "BINARY", "ENTITY", "OTHER"
    options: Optional[List[str]] = None  # For either/or questions


def detect_question_type(question: str) -> QuestionType:
    """
    Detect question type to constrain candidate extraction.

    Returns:
        QuestionType with category, expected_type, and options (for either/or)
    """
    q = question.strip()
    q_lower = q.lower()

    # Either/or questions: "X or Y?" pattern
    # Match patterns like "was it A or B", "is this X or Y", "A or B?"
    either_or_match = re.search(
        r'\b(is|was|are|were|did|does|do|can|could|would|should|will)\s+.{1,50}?\s+(\w+(?:\s+\w+){0,3})\s+or\s+(\w+(?:\s+\w+){0,3})\s*\??$',
        q_lower
    )
    if either_or_match:
        opt1 = either_or_match.group(2).strip()
        opt2 = either_or_match.group(3).strip("?. ")
        return QuestionType(
            category="either_or",
            expected_type="BINARY_CHOICE",
            options=[opt1, opt2]
        )

    # Simple "A or B?" at end
    simple_or = re.search(r'(\w+(?:\s+\w+){0,4})\s+or\s+(\w+(?:\s+\w+){0,4})\s*\??$', q_lower)
    if simple_or:
        opt1 = simple_or.group(1).strip()
        opt2 = simple_or.group(2).strip("?. ")
        # Make sure these look like answer options (not "this or that")
        if opt1.lower() not in {'this', 'that', 'it', 'he', 'she', 'they'}:
            return QuestionType(
                category="either_or",
                expected_type="BINARY_CHOICE",
                options=[opt1, opt2]
            )

    # Yes/No questions
    if q_lower.startswith(('is ', 'are ', 'was ', 'were ', 'do ', 'does ', 'did ',
                           'can ', 'could ', 'should ', 'would ', 'has ', 'have ', 'had ')):
        return QuestionType(category="yes_no", expected_type="BINARY", options=["yes", "no"])

    # Who questions -> PERSON
    if q_lower.startswith('who ') or ' who ' in q_lower[:30]:
        return QuestionType(category="who", expected_type="PERSON")

    # Where questions -> PLACE
    if q_lower.startswith('where ') or ' where ' in q_lower[:30]:
        return QuestionType(category="where", expected_type="PLACE")

    # When questions -> DATE
    if q_lower.startswith('when ') or ' when ' in q_lower[:30]:
        return QuestionType(category="when", expected_type="DATE")

    # How many/much -> NUMBER
    if q_lower.startswith('how many ') or q_lower.startswith('how much '):
        return QuestionType(category="how_many", expected_type="NUMBER")

    # What questions -> ENTITY (generic)
    if q_lower.startswith('what ') or ' what ' in q_lower[:30]:
        # Check for more specific "what [type]" patterns
        what_person = re.match(r'what\s+(person|actor|actress|director|author|singer|player)', q_lower)
        if what_person:
            return QuestionType(category="what", expected_type="PERSON")
        what_place = re.match(r'what\s+(city|country|place|location|state|region)', q_lower)
        if what_place:
            return QuestionType(category="what", expected_type="PLACE")
        what_date = re.match(r'what\s+(year|date|time|day|month)', q_lower)
        if what_date:
            return QuestionType(category="what", expected_type="DATE")
        return QuestionType(category="what", expected_type="ENTITY")

    # Default: other
    return QuestionType(category="other", expected_type="OTHER")


def _normalize_for_dedup(s: str) -> str:
    """
    E2 FIX: Normalize string for substring deduplication.

    Handles punctuation variations so "D'Arcy Coulson" and "Darcy Coulson" compare correctly.
    """
    import re
    # Replace apostrophes, quotes, hyphens, periods with space
    s = re.sub(r"[\s''\.\-]+", " ", s.strip().lower())
    # Collapse multiple spaces
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def extract_candidates(
    winner_passages: List[Dict[str, Any]],
    question_type: QuestionType,
    max_candidates: int = 10
) -> List[Tuple[str, float, str]]:
    """
    Extract candidate answers from winner passages based on question type.

    Returns:
        List of (candidate_text, support_score, source_pid) tuples,
        sorted by support_score descending.
    """
    candidates: Dict[str, Tuple[float, str]] = {}  # text -> (score, pid)

    # For either/or questions, use the provided options
    if question_type.category == "either_or" and question_type.options:
        for opt in question_type.options:
            # Find which passage(s) support this option
            support = 0.0
            source_pid = ""
            for p in winner_passages:
                text_lower = (p.get("text") or "").lower()
                if opt.lower() in text_lower:
                    support += 1.0
                    if not source_pid:
                        source_pid = p.get("pid", "")
            if support > 0 or True:  # Always include options for either/or
                candidates[opt] = (support, source_pid)
        # Return sorted by support
        return sorted(
            [(text, score, pid) for text, (score, pid) in candidates.items()],
            key=lambda x: x[1],
            reverse=True
        )[:max_candidates]

    # For yes/no questions
    if question_type.category == "yes_no":
        # Count evidence for yes vs no
        yes_support = 0.0
        no_support = 0.0
        yes_pid = ""
        no_pid = ""

        for p in winner_passages:
            text = (p.get("text") or "").lower()
            pid = p.get("pid", "")
            # Look for affirmative/negative language
            if any(w in text for w in ['is a', 'was a', 'are the', 'were the', 'did ', 'does ', 'has ', 'had ']):
                yes_support += 0.5
                if not yes_pid:
                    yes_pid = pid
            if any(w in text for w in ['not ', "n't ", 'never ', 'no ', 'none ']):
                no_support += 0.5
                if not no_pid:
                    no_pid = pid

        return [
            ("yes", yes_support, yes_pid),
            ("no", no_support, no_pid)
        ]

    # For PERSON questions
    if question_type.expected_type == "PERSON":
        # E1 FIX: Punctuation-aware pattern for apostrophes, initials, etc.
        # Handles: D'Arcy, O'Connor, J.K. Rowling
        # Includes common name prefixes: de, van, von, la, el, al, bin, ibn
        person_pattern = re.compile(
            r"\b([A-Z][a-zA-ZÀ-ÿ''\.­-]+(?:\s+(?:de|van|von|la|el|al|bin|ibn)?[A-Z][a-zA-ZÀ-ÿ''\.­-]+){0,5})\b"
        )
        for p in winner_passages:
            text = p.get("text") or ""
            pid = p.get("pid", "")
            for match in person_pattern.finditer(text):
                name = match.group(1).strip()
                # Validate: looks like a person name
                if _looks_like_person(name):
                    # Compute support: count occurrences across all passages
                    support = sum(
                        1 for pp in winner_passages
                        if name.lower() in (pp.get("text") or "").lower()
                    )
                    # MULTI-TOKEN BONUS: Prefer longer spans (more specific)
                    # Add 0.5 per token to favor multi-token names over single tokens
                    token_count = len(name.split())
                    length_bonus = 0.5 * (token_count - 1)
                    adjusted_support = support + length_bonus
                    if name not in candidates or adjusted_support > candidates[name][0]:
                        candidates[name] = (adjusted_support, pid)

        # SUBSTRING DEDUPLICATION: Remove single-token candidates that are
        # substrings of higher-scoring multi-token candidates
        # E2 FIX: Use normalization so punctuation variants compare correctly
        if len(candidates) > 1:
            to_remove = set()
            sorted_cands = sorted(candidates.items(), key=lambda x: x[1][0], reverse=True)
            for i, (name1, _) in enumerate(sorted_cands):
                name1_norm = _normalize_for_dedup(name1)
                for name2, _ in sorted_cands[i+1:]:
                    name2_norm = _normalize_for_dedup(name2)
                    # If shorter name is a substring of longer name, mark for removal
                    if len(name2_norm) < len(name1_norm) and name2_norm in name1_norm:
                        to_remove.add(name2)
            for name in to_remove:
                if name in candidates:
                    del candidates[name]

    # For PLACE questions
    elif question_type.expected_type == "PLACE":
        # Extract capitalized spans that look like places
        place_pattern = re.compile(
            r'\b([A-Z][a-zA-ZÀ-ÿ]+(?:\s+[A-Z][a-zA-ZÀ-ÿ]+){0,3})\b'
        )
        for p in winner_passages:
            text = p.get("text") or ""
            pid = p.get("pid", "")
            for match in place_pattern.finditer(text):
                place = match.group(1).strip()
                # Filter out obvious non-places
                if len(place) >= 2 and place.lower() not in {
                    'the', 'a', 'an', 'he', 'she', 'they', 'it', 'his', 'her',
                    'their', 'was', 'were', 'is', 'are', 'has', 'had', 'have'
                }:
                    support = sum(
                        1 for pp in winner_passages
                        if place.lower() in (pp.get("text") or "").lower()
                    )
                    if place not in candidates or support > candidates[place][0]:
                        candidates[place] = (support, pid)

    # For DATE questions
    elif question_type.expected_type == "DATE":
        # Extract date patterns
        date_patterns = [
            # Year only: 1990, 2024
            re.compile(r'\b(1[0-9]{3}|20[0-2][0-9])\b'),
            # Month Day, Year: January 1, 2020
            re.compile(r'\b([A-Z][a-z]+\s+\d{1,2},?\s+\d{4})\b'),
            # Day Month Year: 1 January 2020
            re.compile(r'\b(\d{1,2}\s+[A-Z][a-z]+\s+\d{4})\b'),
            # Month Year: January 2020
            re.compile(r'\b([A-Z][a-z]+\s+\d{4})\b'),
        ]
        for p in winner_passages:
            text = p.get("text") or ""
            pid = p.get("pid", "")
            for pattern in date_patterns:
                for match in pattern.finditer(text):
                    date_str = match.group(1).strip()
                    support = sum(
                        1 for pp in winner_passages
                        if date_str in (pp.get("text") or "")
                    )
                    if date_str not in candidates or support > candidates[date_str][0]:
                        candidates[date_str] = (support, pid)

    # For NUMBER questions
    elif question_type.expected_type == "NUMBER":
        number_pattern = re.compile(r'\b(\d+(?:,\d{3})*(?:\.\d+)?)\b')
        for p in winner_passages:
            text = p.get("text") or ""
            pid = p.get("pid", "")
            for match in number_pattern.finditer(text):
                num = match.group(1).strip()
                support = sum(
                    1 for pp in winner_passages
                    if num in (pp.get("text") or "")
                )
                if num not in candidates or support > candidates[num][0]:
                    candidates[num] = (support, pid)

    # For ENTITY or OTHER: extract all capitalized spans
    else:
        # E1 FIX: Punctuation-aware pattern for apostrophes, initials, etc.
        # Handles: D'Arcy, O'Connor, J.K. Rowling, C.S. Lewis
        # Increased max tokens from 4 to 6 for longer entity names
        entity_pattern = re.compile(
            r"\b([A-Z][a-zA-ZÀ-ÿ''\.­-]+(?:\s+[A-Z][a-zA-ZÀ-ÿ''\.­-]+){0,6})\b"
        )
        # Common stop-words to filter out
        stop_words = {'The', 'A', 'An', 'He', 'She', 'They', 'It', 'His', 'Her',
                      'Their', 'Was', 'Were', 'Is', 'Are', 'Has', 'Had', 'Have',
                      'This', 'That', 'These', 'Those', 'In', 'On', 'At', 'By'}
        for p in winner_passages:
            text = p.get("text") or ""
            pid = p.get("pid", "")
            for match in entity_pattern.finditer(text):
                entity = match.group(1).strip()
                # Skip single-token stop-words
                if entity in stop_words:
                    continue
                if len(entity) >= 2:
                    support = sum(
                        1 for pp in winner_passages
                        if entity.lower() in (pp.get("text") or "").lower()
                    )
                    # MULTI-TOKEN BONUS: Prefer longer spans (more specific)
                    token_count = len(entity.split())
                    length_bonus = 0.5 * (token_count - 1)
                    adjusted_support = support + length_bonus
                    if entity not in candidates or adjusted_support > candidates[entity][0]:
                        candidates[entity] = (adjusted_support, pid)

        # SUBSTRING DEDUPLICATION: Remove single-token candidates that are
        # substrings of higher-scoring multi-token candidates
        # E2 FIX: Use normalization so punctuation variants compare correctly
        if len(candidates) > 1:
            to_remove = set()
            sorted_cands = sorted(candidates.items(), key=lambda x: x[1][0], reverse=True)
            for i, (name1, _) in enumerate(sorted_cands):
                name1_norm = _normalize_for_dedup(name1)
                for name2, _ in sorted_cands[i+1:]:
                    name2_norm = _normalize_for_dedup(name2)
                    # If shorter name is a substring of longer name, mark for removal
                    if len(name2_norm) < len(name1_norm) and name2_norm in name1_norm:
                        to_remove.add(name2)
            for name in to_remove:
                if name in candidates:
                    del candidates[name]

    # Sort by support score and return top candidates
    sorted_candidates = sorted(
        [(text, score, pid) for text, (score, pid) in candidates.items()],
        key=lambda x: x[1],
        reverse=True
    )

    # E3 FIX: Check for low-quality candidates and return empty to trigger fallback
    # For PERSON/ENTITY types, if all candidates are single-token, it's likely noise
    if question_type.expected_type in ("PERSON", "ENTITY"):
        if sorted_candidates:
            max_token_count = max(len(c[0].split()) for c in sorted_candidates)
            if max_token_count < 2:
                # All candidates are single tokens - likely low quality
                # Return empty to trigger CSS fallback instead of picking noise
                import os
                debug = os.environ.get("TRIDENT_DEBUG_CONSTRAINED", "0") == "1"
                if debug:
                    print(f"[EXTRACT] Low-quality candidates: all single-token, returning empty")
                    print(f"[EXTRACT] Would have returned: {[c[0] for c in sorted_candidates[:5]]}")
                return []

    return sorted_candidates[:max_candidates]


def compute_support_score(
    candidate: str,
    passages: List[Dict[str, Any]]
) -> float:
    """
    Compute deterministic support score for a candidate.

    Score = count of passages mentioning candidate, weighted by passage rank.
    """
    if not candidate or not passages:
        return 0.0

    candidate_lower = candidate.lower()
    score = 0.0

    for i, p in enumerate(passages):
        text_lower = (p.get("text") or "").lower()
        title_lower = (p.get("title") or "").lower()

        # Check if candidate appears
        if candidate_lower in text_lower or candidate_lower in title_lower:
            # Weight by rank (earlier passages = higher weight)
            weight = 1.0 / (1 + i * 0.1)
            score += weight

    return score


@dataclass
class ConstrainedSelectionResult:
    """Result from constrained candidate selection."""
    answer: str
    candidate_index: int
    confidence: float
    passage_id: str
    candidates: List[str]
    support_scores: List[float]
    reason: str  # OK, NO_CANDIDATES, LLM_ERROR, INVALID_INDEX


def llm_pick_candidate(
    llm,
    question: str,
    passages: List[Dict[str, Any]],
    candidates: List[Tuple[str, float, str]],
    max_chars_per_passage: int = 600
) -> ConstrainedSelectionResult:
    """
    Ask LLM to pick ONE candidate by index from a constrained set.

    CRITICAL: LLM can ONLY pick from the provided candidates by index.
    This ensures the answer is always a verbatim substring from evidence.

    Args:
        llm: LLM interface with generate() method
        question: The question to answer
        passages: Winner passages for context
        candidates: List of (text, support_score, source_pid) tuples
        max_chars_per_passage: Max chars per passage in prompt

    Returns:
        ConstrainedSelectionResult with selected answer and metadata
    """
    import json as json_module
    import os

    debug = os.environ.get("TRIDENT_DEBUG_CONSTRAINED", "0") == "1"

    if not candidates:
        return ConstrainedSelectionResult(
            answer="",
            candidate_index=-1,
            confidence=0.0,
            passage_id="",
            candidates=[],
            support_scores=[],
            reason="NO_CANDIDATES"
        )

    # Build evidence text
    evidence_parts = []
    for i, p in enumerate(passages[:5]):  # Limit to 5 passages
        pid = p.get("pid", f"P{i}")
        title = p.get("title", "")
        text = (p.get("text") or "")[:max_chars_per_passage]
        evidence_parts.append(f"[{pid}] {title}\n{text}")
    evidence_text = "\n\n".join(evidence_parts)

    # Build candidate list with indices
    candidate_list = []
    candidate_texts = []
    support_scores = []
    for i, (text, score, pid) in enumerate(candidates):
        candidate_list.append(f"{i}: {text}")
        candidate_texts.append(text)
        support_scores.append(score)
    candidates_text = "\n".join(candidate_list)

    # Prompt: LLM must pick by index ONLY
    prompt = f"""Pick the ONE candidate that best answers the question.

RULES:
1. You MUST pick from the numbered candidates below
2. Return ONLY a JSON object with "index" (integer) and "confidence" (0-1)
3. If no candidate answers the question, return {{"index": -1, "confidence": 0}}

Question: {question}

Candidates:
{candidates_text}

Evidence:
{evidence_text}

JSON:"""

    if debug:
        print(f"[CONSTRAINED] Prompt:\n{prompt[:500]}...")
        print(f"[CONSTRAINED] {len(candidates)} candidates")

    try:
        raw = llm.generate(prompt)
        raw_text = raw.text if hasattr(raw, 'text') else str(raw)
    except Exception as e:
        if debug:
            print(f"[CONSTRAINED] LLM error: {e}")
        return ConstrainedSelectionResult(
            answer="",
            candidate_index=-1,
            confidence=0.0,
            passage_id="",
            candidates=candidate_texts,
            support_scores=support_scores,
            reason="LLM_ERROR"
        )

    if debug:
        print(f"[CONSTRAINED] Raw output: {raw_text[:200]}...")

    # Parse JSON response
    try:
        json_match = re.search(r'\{[^{}]*\}', raw_text, flags=re.DOTALL)
        if json_match:
            out = json_module.loads(json_match.group(0))
        else:
            out = json_module.loads(raw_text.strip())
    except Exception as e:
        if debug:
            print(f"[CONSTRAINED] JSON parse error: {e}")
        # Try to extract just the index
        index_match = re.search(r'\b(\d+)\b', raw_text)
        if index_match:
            out = {"index": int(index_match.group(1)), "confidence": 0.5}
        else:
            return ConstrainedSelectionResult(
                answer="",
                candidate_index=-1,
                confidence=0.0,
                passage_id="",
                candidates=candidate_texts,
                support_scores=support_scores,
                reason="PARSE_ERROR"
            )

    index = out.get("index", -1)
    confidence = float(out.get("confidence", 0.5))

    # Validate index
    if not isinstance(index, int) or index < 0 or index >= len(candidates):
        if debug:
            print(f"[CONSTRAINED] Invalid index: {index}")
        # Fallback: pick the candidate with highest support score
        if candidates:
            best_idx = 0
            best_score = candidates[0][1]
            for i, (_, score, _) in enumerate(candidates):
                if score > best_score:
                    best_score = score
                    best_idx = i
            return ConstrainedSelectionResult(
                answer=candidates[best_idx][0],
                candidate_index=best_idx,
                confidence=0.3,  # Low confidence for fallback
                passage_id=candidates[best_idx][2],
                candidates=candidate_texts,
                support_scores=support_scores,
                reason="FALLBACK_SUPPORT"
            )
        return ConstrainedSelectionResult(
            answer="",
            candidate_index=-1,
            confidence=0.0,
            passage_id="",
            candidates=candidate_texts,
            support_scores=support_scores,
            reason="INVALID_INDEX"
        )

    # Valid selection
    selected = candidates[index]
    if debug:
        print(f"[CONSTRAINED] Selected index {index}: '{selected[0]}'")

    return ConstrainedSelectionResult(
        answer=selected[0],
        candidate_index=index,
        confidence=confidence,
        passage_id=selected[2],
        candidates=candidate_texts,
        support_scores=support_scores,
        reason="OK"
    )


def get_answer_facet_passages(
    question: str,
    certificates: List[Dict[str, Any]],
    all_passages: List[Dict[str, Any]],
    facets: Optional[List[Dict[str, Any]]] = None
) -> Tuple[List[Dict[str, Any]], List[str], str]:
    """
    Get passages that won the answer-determining facet(s).

    CERTIFIED-ONLY INVARIANT:
    - Only returns passages that have certificates (in _cert_pid set)
    - Filters out passages with deny-listed titles
    - Orders by p-value (best first)

    For WH-questions, the answer facet is typically RELATION/TEMPORAL/NUMERIC.
    ENTITY facets just confirm the question subject exists, not the answer.

    Args:
        question: The question
        certificates: List of certificate dicts with passage_id, facet_id, etc.
        all_passages: All passages
        facets: Optional list of facet dicts for filtering

    Returns:
        Tuple of (answer_passages, answer_facet_ids, selection_reason)
    """
    import os
    debug = os.environ.get("TRIDENT_DEBUG_CONSTRAINED", "0") == "1"

    if not certificates:
        return [], [], "NO_CERTIFICATES"

    # CERTIFIED-ONLY: Get the set of certified passage IDs
    certified_pids = _cert_pid(certificates)

    # Build passage lookup (only certified passages)
    pid_to_passage = {}
    for p in all_passages:
        pid = p.get("pid")
        if pid and pid in certified_pids:
            # Filter out deny-listed titles
            title = p.get("title", "")
            if not _is_deny_title(title):
                pid_to_passage[pid] = p

    if debug:
        print(f"[ANSWER-FACET] certified_pids={len(certified_pids)}, valid_passages={len(pid_to_passage)}")

    # Build facet_id -> certificates mapping
    facet_to_certs: Dict[str, List[Dict[str, Any]]] = {}
    for cert in certificates:
        fid = cert.get("facet_id", "")
        if fid:
            if fid not in facet_to_certs:
                facet_to_certs[fid] = []
            facet_to_certs[fid].append(cert)

    # Detect question type to determine answer facet priority
    qtype = detect_question_type(question)

    if debug:
        print(f"[ANSWER-FACET] qtype={qtype.category} expected={qtype.expected_type}")
        print(f"[ANSWER-FACET] facet_ids with certs: {list(facet_to_certs.keys())}")

    # Build facet type lookup if facets provided
    facet_types: Dict[str, str] = {}
    if facets:
        for f in facets:
            fid = f.get("facet_id", "")
            ftype = f.get("facet_type", "")
            if fid and ftype:
                facet_types[fid] = ftype

    # Priority order for answer facets based on question type
    # WH-questions need RELATION/TEMPORAL/NUMERIC facets for the actual answer
    # ENTITY facets just confirm the question subject exists
    priority_facet_types = []

    if qtype.category == "when":
        priority_facet_types = ["TEMPORAL", "RELATION", "NUMERIC"]
    elif qtype.category == "where":
        priority_facet_types = ["RELATION", "TEMPORAL"]
    elif qtype.category in ("who", "what"):
        priority_facet_types = ["RELATION", "TEMPORAL", "NUMERIC"]
    elif qtype.category == "how_many":
        priority_facet_types = ["NUMERIC", "RELATION", "TEMPORAL"]
    else:
        # Default: prefer RELATION over ENTITY
        priority_facet_types = ["RELATION", "TEMPORAL", "NUMERIC", "ENTITY"]

    # Find answer facets in priority order
    answer_facet_ids = []
    for ptype in priority_facet_types:
        for fid in facet_to_certs:
            if facet_types.get(fid) == ptype and fid not in answer_facet_ids:
                answer_facet_ids.append(fid)

    # If no priority facets, use all non-ENTITY facets first, then ENTITY
    if not answer_facet_ids:
        non_entity = [fid for fid in facet_to_certs if facet_types.get(fid) != "ENTITY"]
        entity = [fid for fid in facet_to_certs if facet_types.get(fid) == "ENTITY"]
        answer_facet_ids = non_entity + entity

    # If still empty, use all facets
    if not answer_facet_ids:
        answer_facet_ids = list(facet_to_certs.keys())

    if debug:
        print(f"[ANSWER-FACET] priority_types={priority_facet_types}")
        print(f"[ANSWER-FACET] answer_facet_ids={answer_facet_ids}")

    # Get passages that won answer facets (CERTIFIED-ONLY: must be in pid_to_passage)
    answer_passages = []
    seen_pids = set()

    # Get all certs for answer facets, sorted by p-value (best first)
    answer_certs = []
    for fid in answer_facet_ids:
        answer_certs.extend(facet_to_certs.get(fid, []))

    answer_certs.sort(key=lambda c: c.get("p_value", 1.0))

    for cert in answer_certs:
        pid = cert.get("passage_id")
        # CERTIFIED-ONLY: Only include if in pid_to_passage (certified and not deny-listed)
        if pid and pid not in seen_pids and pid in pid_to_passage:
            seen_pids.add(pid)
            answer_passages.append(pid_to_passage[pid])

    selection_reason = f"FACET_TARGETED:{','.join(answer_facet_ids[:3])}"

    if debug:
        print(f"[ANSWER-FACET] answer_passages: {len(answer_passages)}")
        for ap in answer_passages[:3]:
            print(f"  - {ap.get('pid', '')[:12]}... ({len(ap.get('text', ''))} chars)")

    return answer_passages, answer_facet_ids, selection_reason


def constrained_span_select(
    llm,
    question: str,
    winner_passages: List[Dict[str, Any]],
    max_candidates: int = 10,
    max_chars_per_passage: int = 600,
    certificates: Optional[List[Dict[str, Any]]] = None
) -> ConstrainedSelectionResult:
    """
    Full constrained selection pipeline: detect type -> extract candidates -> LLM pick.

    CERTIFIED-ONLY INVARIANT:
    - If certificates provided, only use passages in _cert_pid(certificates)
    - Candidates are FROZEN at extraction time (immutable list)
    - LLM can only select by index from frozen candidates
    - Final answer is verified to be verbatim substring before return

    Args:
        llm: LLM interface
        question: The question to answer
        winner_passages: Certificate-winning passages
        max_candidates: Max candidates to extract
        max_chars_per_passage: Max chars per passage in prompt
        certificates: Optional certificates for certified-only enforcement

    Returns:
        ConstrainedSelectionResult with answer and metadata
    """
    import os
    debug = os.environ.get("TRIDENT_DEBUG_CONSTRAINED", "0") == "1"

    # CERTIFIED-ONLY: Filter passages if certificates provided
    if certificates:
        certified_pids = _cert_pid(certificates)
        winner_passages = [
            p for p in winner_passages
            if p.get("pid") in certified_pids and not _is_deny_title(p.get("title", ""))
        ]
        if debug:
            print(f"[CONSTRAINED] Filtered to {len(winner_passages)} certified passages")

    if not winner_passages:
        return ConstrainedSelectionResult(
            answer="",
            candidate_index=-1,
            confidence=0.0,
            passage_id="",
            candidates=[],
            support_scores=[],
            reason="NO_CERTIFIED_PASSAGES"
        )

    # Step 1: Detect question type
    qtype = detect_question_type(question)
    if debug:
        print(f"[CONSTRAINED] Question type: {qtype.category}, expected: {qtype.expected_type}")
        if qtype.options:
            print(f"[CONSTRAINED] Options: {qtype.options}")

    # Step 2: Extract candidates (FROZEN at this point - immutable)
    candidates = extract_candidates(winner_passages, qtype, max_candidates)

    # FREEZE: Convert to tuple to ensure immutability
    frozen_candidates = tuple(candidates)

    if debug:
        print(f"[CONSTRAINED] Extracted {len(frozen_candidates)} candidates (frozen)")
        for i, (text, score, pid) in enumerate(frozen_candidates[:5]):
            pid_str = pid[:12] if pid else "?"
            print(f"  {i}: '{text}' (support={score}, pid={pid_str}...)")

    if not frozen_candidates:
        return ConstrainedSelectionResult(
            answer="",
            candidate_index=-1,
            confidence=0.0,
            passage_id="",
            candidates=[],
            support_scores=[],
            reason="NO_CANDIDATES"
        )

    # Step 3: LLM picks from frozen candidates
    result = llm_pick_candidate(
        llm=llm,
        question=question,
        passages=winner_passages,
        candidates=list(frozen_candidates),  # Pass as list but from frozen source
        max_chars_per_passage=max_chars_per_passage
    )

    # Step 4: VERIFY the answer is verbatim in source passage
    if result.answer and result.passage_id:
        pid_to_passage = {p.get("pid", ""): p for p in winner_passages}
        source_passage = pid_to_passage.get(result.passage_id)
        if source_passage:
            source_text = source_passage.get("text", "")
            # Normalize for matching
            source_norm = _normalize_text_unicode(source_text)
            answer_norm = _normalize_text_unicode(result.answer)
            if answer_norm and answer_norm not in source_norm:
                if debug:
                    print(f"[CONSTRAINED] VERIFY FAILED: '{result.answer}' not in passage")
                # Return with VERIFY_FAILED reason but keep answer for debugging
                return ConstrainedSelectionResult(
                    answer=result.answer,
                    candidate_index=result.candidate_index,
                    confidence=0.0,  # Zero confidence on verify failure
                    passage_id=result.passage_id,
                    candidates=result.candidates,
                    support_scores=result.support_scores,
                    reason="VERIFY_FAILED"
                )

    return result


def bind_entity_via_css(
    llm,
    inner_question: str,
    hop1_passages: List[Dict[str, Any]],
    max_chars: int = 600,
    certificates: Optional[List[Dict[str, Any]]] = None
) -> Optional[str]:
    """
    CHOICE-ONLY entity binding using Certified Span Selection.

    CERTIFIED-ONLY INVARIANT:
    - First extracts PERSON candidates from passages
    - LLM picks by index from frozen candidates (not free-form)
    - Final answer is verified to be verbatim substring

    Example:
        inner_question: "Who is the director of Polish-Russian War (film)?"
        Candidates: ["Xawery Żuławski", "John Smith", ...]
        LLM picks: index 0
        Returns: "Xawery Żuławski" (verbatim from passage)

    Args:
        llm: LLM interface
        inner_question: The hop-1 question asking for the bridge entity
        hop1_passages: Certificate-winning passages from hop-1
        max_chars: Max chars per passage
        certificates: Optional certificates for certified-only enforcement

    Returns:
        The bound entity string, or None if binding failed
    """
    import json as json_module
    import os

    debug = os.environ.get("TRIDENT_DEBUG_CSS", "0") == "1"

    if not hop1_passages:
        return None

    # CERTIFIED-ONLY: Filter passages if certificates provided
    if certificates:
        certified_pids = _cert_pid(certificates)
        hop1_passages = [
            p for p in hop1_passages
            if p.get("pid") in certified_pids and not _is_deny_title(p.get("title", ""))
        ]
        if debug:
            print(f"[CSS-BIND] Filtered to {len(hop1_passages)} certified passages")

    if not hop1_passages:
        return None

    # Step 1: Extract PERSON candidates from passages (FROZEN)
    # Use the PERSON pattern from extract_candidates
    person_pattern = re.compile(
        r"\b([A-Z][a-zA-ZÀ-ÿ''\.­-]+(?:\s+(?:de|van|von|la|el|al|bin|ibn)?[A-Z][a-zA-ZÀ-ÿ''\.­-]+){0,5})\b"
    )

    candidates: Dict[str, Tuple[float, str]] = {}  # text -> (score, pid)

    for p in hop1_passages:
        text = p.get("text") or ""
        pid = p.get("pid", "")

        for match in person_pattern.finditer(text):
            name = match.group(1).strip()

            # Validate: looks like a person name
            if not _looks_like_person(name):
                continue

            # Compute support score
            support = sum(
                1 for pp in hop1_passages
                if name.lower() in (pp.get("text") or "").lower()
            )

            # Multi-token bonus
            token_count = len(name.split())
            length_bonus = 0.5 * (token_count - 1)
            adjusted_support = support + length_bonus

            if name not in candidates or adjusted_support > candidates[name][0]:
                candidates[name] = (adjusted_support, pid)

    # Deduplicate: remove single-token candidates that are substrings of multi-token
    if len(candidates) > 1:
        to_remove = set()
        sorted_cands = sorted(candidates.items(), key=lambda x: x[1][0], reverse=True)
        for i, (name1, _) in enumerate(sorted_cands):
            name1_norm = _normalize_for_dedup(name1)
            for name2, _ in sorted_cands[i+1:]:
                name2_norm = _normalize_for_dedup(name2)
                if len(name2_norm) < len(name1_norm) and name2_norm in name1_norm:
                    to_remove.add(name2)
        for name in to_remove:
            if name in candidates:
                del candidates[name]

    # Sort by support score
    sorted_candidates = sorted(
        [(text, score, pid) for text, (score, pid) in candidates.items()],
        key=lambda x: x[1],
        reverse=True
    )[:10]  # Max 10 candidates

    # FREEZE candidates
    frozen_candidates = tuple(sorted_candidates)

    if debug:
        print(f"[CSS-BIND] Extracted {len(frozen_candidates)} PERSON candidates (frozen)")
        for i, (text, score, pid) in enumerate(frozen_candidates[:5]):
            print(f"  {i}: '{text}' (support={score})")

    if not frozen_candidates:
        if debug:
            print(f"[CSS-BIND] No PERSON candidates found")
        return None

    # Step 2: LLM picks by index from frozen candidates
    # Build compact evidence
    evidence_parts = []
    for p in hop1_passages[:3]:  # Limit to 3 passages
        pid = p.get("pid", "")
        title = p.get("title", "")
        text = (p.get("text") or "")[:max_chars]
        evidence_parts.append(f"[{pid}] {title}\n{text}")

    evidence_text = "\n\n".join(evidence_parts)

    # Build numbered candidate list
    candidate_list = []
    for i, (text, score, pid) in enumerate(frozen_candidates):
        candidate_list.append(f"{i}: {text}")
    candidates_text = "\n".join(candidate_list)

    # CHOICE-ONLY prompt: LLM must pick by index
    prompt = f"""Pick the ONE person's name that answers the question.

RULES:
1. You MUST pick from the numbered candidates below BY INDEX
2. Return ONLY a JSON object with "index" (integer) and "confidence" (0-1)
3. If no candidate answers the question, return {{"index": -1, "confidence": 0}}

Question: {inner_question}

Candidates:
{candidates_text}

Evidence:
{evidence_text}

JSON:"""

    try:
        raw = llm.generate(prompt)
        raw_text = raw.text if hasattr(raw, 'text') else str(raw)
    except Exception as e:
        if debug:
            print(f"[CSS-BIND] LLM generation failed: {e}")
        return None

    if debug:
        print(f"[CSS-BIND] Raw output: {raw_text[:200]}...")

    # Parse JSON response
    try:
        json_match = re.search(r'\{[^{}]*\}', raw_text, flags=re.DOTALL)
        if json_match:
            out = json_module.loads(json_match.group(0))
        else:
            out = json_module.loads(raw_text.strip())
    except Exception as e:
        if debug:
            print(f"[CSS-BIND] JSON parse error: {e}")
        # Try to extract just the index
        index_match = re.search(r'\b(\d+)\b', raw_text)
        if index_match:
            out = {"index": int(index_match.group(1)), "confidence": 0.5}
        else:
            return None

    index = out.get("index", -1)

    # Validate index
    if not isinstance(index, int) or index < 0 or index >= len(frozen_candidates):
        if debug:
            print(f"[CSS-BIND] Invalid index: {index}")
        # Fallback: pick highest-scoring candidate
        if frozen_candidates:
            answer = frozen_candidates[0][0]
            if debug:
                print(f"[CSS-BIND] Fallback to highest-scoring: '{answer}'")
            return answer
        return None

    # Get selected candidate
    selected_text, selected_score, selected_pid = frozen_candidates[index]

    if debug:
        print(f"[CSS-BIND] Selected index {index}: '{selected_text}'")

    # Step 3: Verify it's verbatim in passage
    passage_map = {p.get("pid", ""): p for p in hop1_passages}
    verified = False

    if selected_pid and selected_pid in passage_map:
        passage_text = passage_map[selected_pid].get("text", "")
        if _find_exact_substring(passage_text, selected_text) is not None:
            verified = True
    else:
        # Try all passages
        for p in hop1_passages:
            if _find_exact_substring(p.get("text", ""), selected_text) is not None:
                verified = True
                break

    if not verified:
        if debug:
            print(f"[CSS-BIND] VERIFY FAILED: '{selected_text}' not found verbatim")
        return None

    if debug:
        print(f"[CSS-BIND] Accepted: '{selected_text}'")

    return selected_text


def build_inner_question_from_facet(facet: Dict[str, Any]) -> str:
    """
    Build the inner question from a hop-1 facet for entity binding.

    Example facet template:
        {"subject": "Who", "predicate": "is the director of", "object": "Polish-Russian War"}

    Returns:
        "Who is the director of Polish-Russian War?"
    """
    template = facet.get("template", {})
    subject = template.get("subject", "Who")
    predicate = template.get("predicate", "")
    obj = template.get("object", "")

    # Build natural question
    question = f"{subject} {predicate} {obj}".strip()

    # Ensure it ends with question mark
    if not question.endswith("?"):
        question += "?"

    return question


def get_winner_passages_only(
    certificates: List[Dict[str, Any]],
    passages: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Extract only the certificate-winning passages (deduplicated).

    CERTIFIED-ONLY INVARIANT:
    - Only includes passages that have valid certificates
    - Filters out passages with deny-listed titles
    - Orders by certificate p-value (best first)

    Args:
        certificates: List of certificates with passage_id
        passages: List of all selected passages

    Returns:
        List of unique winning passages (ordered by certificate p-value)
    """
    if not certificates:
        return []

    # CERTIFIED-ONLY: Get set of certified passage IDs
    certified_pids = _cert_pid(certificates)

    # Build passage lookup (only certified passages, excluding deny-listed)
    pid_to_passage = {}
    for p in passages:
        pid = p.get("pid", "")
        if pid and pid in certified_pids:
            title = p.get("title", "")
            if not _is_deny_title(title):
                pid_to_passage[pid] = p

    # Sort certificates by p-value (best first)
    sorted_certs = sorted(certificates, key=lambda c: c.get("p_value", 1.0))

    # Collect unique winning passages
    seen_pids = set()
    winners = []

    for cert in sorted_certs:
        pid = cert.get("passage_id", "")
        # CERTIFIED-ONLY: Must be in pid_to_passage (certified and not deny-listed)
        if pid and pid not in seen_pids and pid in pid_to_passage:
            seen_pids.add(pid)
            winners.append(pid_to_passage[pid])

    return winners
