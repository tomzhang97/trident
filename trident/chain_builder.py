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

# Relation triggers for different relation kinds
REL_TRIGGERS = {
    "DIRECTOR": ["directed by", "director", "film directed by", "directed", "filmmaker"],
    "BORN": ["was born", "born in", "birthplace", "native of", "born on"],
    "AWARD": ["won", "award", "nominated", "prize", "received", "honored"],
    "CREATED": ["created", "founded", "wrote", "written by", "author", "composed"],
    "LOCATION": ["located in", "capital of", "situated", "based in", "headquarters"],
    "MARRIAGE": ["married", "spouse", "wife", "husband", "wed"],
}


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
