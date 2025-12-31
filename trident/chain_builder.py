"""Multi-hop chain builder for extracting answers from certified passages.

This module builds explicit reasoning chains from certified evidence,
ensuring that the answer is grounded in the actual passages selected.
"""

from __future__ import annotations

import json
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
    text = text.replace(""", '"').replace(""", '"')
    text = text.replace("'", "'").replace("'", "'")

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
class QuestionType:
    """Detected question type for constraining candidates."""
    category: str  # "either_or", "who", "what", "where", "when", "how_many", "yes_no", "other"
    expected_type: str  # "PERSON", "PLACE", "DATE", "NUMBER", "BINARY", "ENTITY", "OTHER"
    options: Optional[List[str]] = None  # For either/or questions


@dataclass
class ConstrainedSelectionResult:
    """Result from constrained candidate selection."""
    answer: str
    candidate_index: int
    confidence: float
    passage_id: str
    candidates: List[str]
    support_scores: List[float]
    reason: str  # OK, NO_PASSAGES, NO_VERIFIED_SPAN, etc.
    tokens_used: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


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

    # Normalized unicode fallback (handles dash/quote variants)
    hay_norm = _normalize_text_unicode(haystack)
    needle_norm = _normalize_text_unicode(needle)
    if needle_norm:
        j = hay_norm.find(needle_norm)
        if j >= 0:
            return (j, j + len(needle_norm))

    return None


def _extract_first_json_object(text: str) -> str:
    """Extract the first JSON object from text, ignoring trailing chatter."""
    stripped = text.strip()
    start = stripped.find("{")
    if start == -1:
        raise ValueError("No JSON object found in text")

    # reject code fences before the JSON
    if "```" in stripped[:start]:
        raise ValueError("Code fences found before JSON")

    depth = 0
    end_idx = -1
    for idx in range(start, len(stripped)):
        ch = stripped[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end_idx = idx
                break

    if depth != 0 or end_idx == -1:
        raise ValueError("Unbalanced JSON braces in text")

    return stripped[start:end_idx + 1]


def strict_json_call(llm, prompt: str, max_new_tokens: int = 160, temperature: float = 0.0):
    """Run an LLM call that must return exactly one JSON object.

    This helper centralizes the JSON-only contract so downstream callers do not
    attempt their own permissive parsing. It extracts the first balanced JSON
    object and ignores any trailing text, while still rejecting code fences or
    unbalanced braces to avoid ambiguous parses.

    Returns:
        tuple(parsed_json, raw_output)
    """

    guard = "Return ONLY JSON. No extra keys. No commentary."
    wrapped_prompt = f"{guard}\n{prompt.strip()}\n{guard}"
    raw = llm.generate(
        wrapped_prompt,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )
    raw_text = raw.text if hasattr(raw, "text") else str(raw)
    parsed = json.loads(_extract_first_json_object(raw_text))
    return parsed, raw


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
    Normalize string for substring deduplication.

    Handles punctuation variations so "D'Arcy Coulson" and "Darcy Coulson" compare correctly.
    """
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
            if source_pid:
                candidates[opt] = (support, source_pid)
        # Return sorted by support
        return sorted(
            [(text, score, pid) for text, (score, pid) in candidates.items() if pid],
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

        if not yes_pid and winner_passages:
            yes_pid = winner_passages[0].get("pid", "")
        if not no_pid and winner_passages:
            no_pid = winner_passages[0].get("pid", "")

        return [
            ("yes", yes_support, yes_pid),
            ("no", no_support, no_pid)
        ]

    # For PERSON questions
    if question_type.expected_type == "PERSON":
        # Punctuation-aware pattern for apostrophes, initials, etc.
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
                    token_count = len(name.split())
                    length_bonus = 0.5 * (token_count - 1)
                    adjusted_support = support + length_bonus
                    if name not in candidates or adjusted_support > candidates[name][0]:
                        candidates[name] = (adjusted_support, pid)

        # SUBSTRING DEDUPLICATION: Remove single-token candidates that are
        # substrings of higher-scoring multi-token candidates
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
        date_patterns = [
            re.compile(r'\b(1[0-9]{3}|20[0-2][0-9])\b'),  # Year only
            re.compile(r'\b([A-Z][a-z]+\s+\d{1,2},?\s+\d{4})\b'),  # Month Day, Year
            re.compile(r'\b(\d{1,2}\s+[A-Z][a-z]+\s+\d{4})\b'),  # Day Month Year
            re.compile(r'\b([A-Z][a-z]+\s+\d{4})\b'),  # Month Year
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
        entity_pattern = re.compile(
            r"\b([A-Z][a-zA-ZÀ-ÿ''\.­-]+(?:\s+[A-Z][a-zA-ZÀ-ÿ''\.­-]+){0,6})\b"
        )
        stop_words = {'The', 'A', 'An', 'He', 'She', 'They', 'It', 'His', 'Her',
                      'Their', 'Was', 'Were', 'Is', 'Are', 'Has', 'Had', 'Have',
                      'This', 'That', 'These', 'Those', 'In', 'On', 'At', 'By'}
        for p in winner_passages:
            text = p.get("text") or ""
            pid = p.get("pid", "")
            for match in entity_pattern.finditer(text):
                entity = match.group(1).strip()
                if entity in stop_words:
                    continue
                if len(entity) >= 2:
                    support = sum(
                        1 for pp in winner_passages
                        if entity.lower() in (pp.get("text") or "").lower()
                    )
                    token_count = len(entity.split())
                    length_bonus = 0.5 * (token_count - 1)
                    adjusted_support = support + length_bonus
                    if entity not in candidates or adjusted_support > candidates[entity][0]:
                        candidates[entity] = (adjusted_support, pid)

        # SUBSTRING DEDUPLICATION
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

    # Sort by support score and return top candidates
    sorted_candidates = sorted(
        [
            (text, score, pid)
            for text, (score, pid) in candidates.items()
            if pid
        ],
        key=lambda x: x[1],
        reverse=True
    )

    # Check for low-quality candidates and return empty to trigger fallback
    if question_type.expected_type in ("PERSON", "ENTITY"):
        if sorted_candidates:
            max_token_count = max(len(c[0].split()) for c in sorted_candidates)
            if max_token_count < 2:
                import os
                debug = os.environ.get("TRIDENT_DEBUG_CONSTRAINED", "0") == "1"
                if debug:
                    print(f"[EXTRACT] Low-quality candidates: all single-token, returning empty")
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


def get_answer_facet_passages(
    question: str,
    certificates: List[Dict[str, Any]],
    all_passages: List[Dict[str, Any]],
    facets: Optional[List[Dict[str, Any]]] = None,
    max_answer_facets: int = 3,
    max_answer_passages: int = 5,
) -> Tuple[List[Dict[str, Any]], List[str], str]:
    """
    Select passages corresponding to the *answer-determining* facets.

    Certified-only, fail-closed properties:
    - Only passages whose PID appears in certificates are eligible.
    - For each facet, only the best (lowest p-value) certificate is used to fetch evidence.
    """
    import os

    debug = os.environ.get("TRIDENT_DEBUG_CONSTRAINED", "0") == "1"

    if not certificates:
        return [], [], "NO_CERTIFICATES"

    # Certified-only PID scope
    certified_pids = _cert_pid(certificates)

    # Build passage lookup from certified-only pool
    pid_to_passage = {
        p.get("pid"): p
        for p in all_passages
        if p.get("pid") in certified_pids
    }

    # Best cert per facet (lowest p-value)
    facet_to_certs: Dict[str, List[Dict[str, Any]]] = {}
    for cert in certificates:
        fid = cert.get("facet_id") or ""
        pid = cert.get("passage_id") or cert.get("pid") or ""
        if fid and pid and pid in pid_to_passage:
            facet_to_certs.setdefault(fid, []).append(cert)

    if not facet_to_certs:
        return [], [], "NO_CERT_MAPPABLE_PASSAGES"

    best_cert_by_fid: Dict[str, Dict[str, Any]] = {}
    for fid, certs in facet_to_certs.items():
        best_cert_by_fid[fid] = min(certs, key=lambda c: float(c.get("p_value", 1.0)))

    certified_facet_ids = list(best_cert_by_fid.keys())

    # Choose answer facets deterministically
    selected_facet_ids = [
        fid for fid, _ in sorted(
            ((fid, cert.get("p_value", 1.0)) for fid, cert in best_cert_by_fid.items()),
            key=lambda x: float(x[1])
        )
    ][:max_answer_facets]
    selection_reason = "TOP_PVALUE"

    if not selected_facet_ids:
        selected_facet_ids = certified_facet_ids[:max_answer_facets]
        selection_reason = f"HEURISTIC:{','.join(selected_facet_ids)}"

    # Fetch winning passages from best certs (strict provenance)
    target_certs = [best_cert_by_fid[fid] for fid in selected_facet_ids if fid in best_cert_by_fid]
    target_certs.sort(key=lambda c: float(c.get("p_value", 1.0)))

    answer_passages: List[Dict[str, Any]] = []
    seen = set()
    for cert in target_certs:
        pid = cert.get("passage_id") or cert.get("pid") or ""
        if not pid or pid in seen:
            continue
        if pid in pid_to_passage:
            seen.add(pid)
            answer_passages.append(pid_to_passage[pid])
        if len(answer_passages) >= max_answer_passages:
            break

    if debug:
        apids = [p.get("pid") for p in answer_passages]
        print(f"[ANSWER-FACET] reason={selection_reason} facets={selected_facet_ids} pids={apids}")

    return answer_passages, selected_facet_ids, selection_reason


def constrained_span_select(
    llm,
    question: str,
    winner_passages: List[Dict[str, Any]],
    max_candidates: int = 10,
    max_chars_per_passage: int = 600,
    certificates: Optional[List[Dict[str, Any]]] = None,
) -> ConstrainedSelectionResult:
    """
    FIX A: Frozen span extraction with robust JSON parsing.

    Primary: Extract answer as exact substring with offsets (certified-only).
    Secondary: Candidate-index selection (legacy) but still certified-only if certificates provided.

    Fail-closed: returns empty answer if nothing can be machine-verified.
    """
    import os

    debug = os.environ.get("TRIDENT_DEBUG_CONSTRAINED", "0") == "1"

    if not winner_passages:
        return ConstrainedSelectionResult(
            answer="",
            candidate_index=-1,
            confidence=0.0,
            passage_id="",
            candidates=[],
            support_scores=[],
            reason="NO_PASSAGES"
        )

    if certificates:
        certified_pids = _cert_pid(certificates)
        winner_passages = [p for p in winner_passages if p.get("pid") in certified_pids]
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

    # Frozen span extraction prompt
    snippet_map: Dict[str, str] = {}
    evidence_parts = []
    for p in winner_passages:
        pid = p.get("pid")
        if not pid:
            continue
        snip = (p.get("text") or "")[:max_chars_per_passage]
        snippet_map[pid] = snip
        evidence_parts.append(f"[{pid}] {snip}")

    evidence_text = "\n\n".join(evidence_parts)

    span_prompt = f"""Extract the answer as an EXACT substring from the evidence.

Question: {question}

Evidence:
{evidence_text}

Rules:
1. The answer_span must be copied VERBATIM from the evidence.
2. Provide start_char and end_char offsets relative to the snippet shown for that pid.
3. Return JSON in ONE of these formats:
   Format 1 (preferred): {{ "pid": "...", "start_char": 0, "end_char": 0, "answer_span": "...", "confidence": 0.0 }}
   Format 2 (alternate): {{ "pid": "...", "answer": "...", "confidence": 0.0 }}
4. If not found, return {{ "answer_span": null }}

JSON:"""

    def _norm_ws(s: str) -> str:
        return " ".join((s or "").split())

    try:
        resp = llm.generate(span_prompt, temperature=0.0)
        raw = resp.text if hasattr(resp, "text") else str(resp)

        # FIX A: Parse only the first JSON object (ignore trailing chatter / code fences)
        try:
            data = json.loads(_extract_first_json_object(raw))
        except Exception as parse_error:
            if debug:
                print(f"[FROZEN-SPAN] JSON parse failed: {parse_error}")
                print(f"[FROZEN-SPAN] Raw output: {raw[:200]}")
            return ConstrainedSelectionResult(
                answer="",
                candidate_index=-1,
                confidence=0.0,
                passage_id="",
                candidates=[],
                support_scores=[],
                reason="PARSE_ERROR"
            )

        # FIX A: Accept either schema
        # Schema 1: {pid, start_char, end_char, answer_span, confidence}
        # Schema 2: {pid, answer, confidence}

        span = data.get("answer_span") or data.get("answer")
        pid = data.get("pid") or ""

        # Normalize pid: "[context_4]" → "context_4"
        if pid.startswith("[") and pid.endswith("]"):
            pid = pid[1:-1]

        start_char = data.get("start_char")
        end_char = data.get("end_char")
        conf = float(data.get("confidence", 0.0))

        if span and pid in snippet_map:
            snippet = snippet_map[pid]

            # Try to verify with offsets if provided
            match = False
            if isinstance(start_char, int) and isinstance(end_char, int):
                if 0 <= start_char < end_char <= len(snippet):
                    sliced = snippet[start_char:end_char]
                    if _norm_ws(sliced) == _norm_ws(span):
                        match = True

            # Deterministic repair: exact find in original snippet if offsets wrong
            if not match:
                idx = snippet.find(span)
                if idx != -1:
                    start_char, end_char = idx, idx + len(span)
                    if _norm_ws(snippet[start_char:end_char]) == _norm_ws(span):
                        match = True

            if match:
                # Type sanity checks (lightweight)
                q_type = detect_question_type(question)
                ok = True
                if getattr(q_type, "expected_type", None) == "NUMBER":
                    ok = bool(re.search(r"\d", span)) and len(span) <= 50
                elif getattr(q_type, "expected_type", None) == "DATE":
                    ok = bool(re.search(r"\d{4}", span) or re.search(r"\d", span))
                elif getattr(q_type, "category", None) == "yes_no":
                    ok = _norm_ws(span).lower() in {"yes", "no"}

                if ok:
                    if debug:
                        print(f"[FROZEN-SPAN] pid={pid} span='{span}' [{start_char}:{end_char}] conf={conf:.2f}")
                    return ConstrainedSelectionResult(
                        answer=span,
                        candidate_index=0,
                        confidence=conf,
                        passage_id=pid,
                        candidates=[span],
                        support_scores=[1.0],
                        reason="OK_FROZEN_SPAN"
                    )
    except Exception as ex:
        if debug:
            print(f"[FROZEN-SPAN] Failed: {ex}")

    return ConstrainedSelectionResult(
        answer="",
        candidate_index=-1,
        confidence=0.0,
        passage_id="",
        candidates=[],
        support_scores=[],
        reason="NO_VERIFIED_SPAN"
    )


def bind_entity_via_css(
    llm,
    inner_question: str,
    hop1_passages: List[Dict[str, Any]],
    max_chars: int = 600,
    min_confidence: float = 0.55,
    certificates: Optional[List[Dict[str, Any]]] = None,
) -> Optional[str]:
    """
    FIX A: Canonical entity binder with robust JSON parsing.

    Accepts either:
    - {index, confidence}
    - {pid, answer, confidence}

    Certified-only option:
    - If certificates provided, restrict hop1_passages to certified PIDs.
    Fail-closed:
    - If parsing fails, index out of bounds, or confidence < threshold => None.
    """
    import os

    debug = os.environ.get("TRIDENT_DEBUG_CSS", "0") == "1"

    if not hop1_passages:
        return None

    if certificates:
        certified_pids = _cert_pid(certificates)
        hop1_passages = [p for p in hop1_passages if p.get("pid") in certified_pids]
        if not hop1_passages:
            return None

    # Candidate enumeration
    q_type = detect_question_type(inner_question)
    raw_candidates = extract_candidates(hop1_passages, q_type, max_candidates=20)

    # Build normalized -> (display_text, source_pid)
    cand_map: Dict[str, Tuple[str, str]] = {}
    for text, score, pid in raw_candidates:
        if not text or not pid:
            continue
        toks = text.split()
        ok = False
        if len(toks) >= 2:
            ok = True
        elif len(toks) == 1:
            t = toks[0]
            if len(t) >= 4 and t.lower() not in {"this", "that", "film", "movie", "city"}:
                ok = True
        if not ok:
            continue
        norm = _normalize_text_unicode(text)
        if norm and norm not in cand_map:
            cand_map[norm] = (text, pid)

    cand_items = list(cand_map.values())
    cand_items.sort(key=lambda x: (-(len(x[0].split())), x[0].lower(), x[1]))
    candidate_list = cand_items[:8]

    if not candidate_list:
        if debug:
            print("[CSS-BIND] No candidates after filtering.")
        return None

    # Build evidence snippets
    relevant_pids = {pid for _, pid in candidate_list}
    snippet_map: Dict[str, str] = {}
    evidence_parts = []
    for p in hop1_passages:
        pid = p.get("pid")
        if pid in relevant_pids:
            snip = (p.get("text") or "")[:max_chars]
            snippet_map[pid] = snip
            evidence_parts.append(f"[{pid}] {snip}")

    candidates_formatted = "\n".join([f"{i}: {c[0]}" for i, c in enumerate(candidate_list)])
    evidence_text = "\n".join(evidence_parts)

    prompt = f"""Identify the entity that answers the question.
Select ONE candidate from the list by index.

Question: {inner_question}

Candidates:
{candidates_formatted}

Evidence:
{evidence_text}

Rules:
1. Return JSON in ONE of these formats:
   Format 1 (preferred): {{ "index": int, "confidence": float }}
   Format 2 (alternate): {{ "pid": "...", "answer": "...", "confidence": float }}
2. If none match, return {{ "index": -1 }}

JSON:"""

    try:
        resp = llm.generate(prompt, temperature=0.0)
        raw = resp.text if hasattr(resp, "text") else str(resp)

        # FIX A: Robust JSON extraction
        def _first_json_obj(text: str) -> Optional[Dict[str, Any]]:
            """Parse the *first* JSON object from model output."""
            cleaned = re.sub(r"```[\s\S]*?```", " ", text).strip()
            s0 = cleaned.find("{")
            if s0 == -1:
                return None
            try:
                dec = json.JSONDecoder()
                obj, _ = dec.raw_decode(cleaned[s0:])
                return obj if isinstance(obj, dict) else None
            except Exception:
                m = re.search(r"\{[\s\S]*?\}", cleaned)
                if not m:
                    return None
                try:
                    obj = json.loads(m.group(0))
                    return obj if isinstance(obj, dict) else None
                except Exception:
                    return None

        data = _first_json_obj(raw)
        if not data:
            if debug:
                print(f"[CSS-BIND] Failed: could not parse JSON from: {raw[:120]!r}")
            return None

        idx = int(data.get("index", -1)) if "index" in data and str(data.get("index")).strip() != "" else -1
        conf = float(data.get("confidence", 0.0) or 0.0)

        # FIX A: Accept alternate schema: {"pid": "...", "answer": "..."}
        if idx < 0:
            pid_raw = str(data.get("pid", "") or "").strip().strip("[]")
            ans_raw = str(data.get("answer", "") or "").strip()

            # FIX A: Map pid or answer back to a candidate index
            if pid_raw:
                for j, (_txt, _pid) in enumerate(candidate_list):
                    if (_pid or "") == pid_raw:
                        idx = j
                        break
            if idx < 0 and ans_raw:
                norm_ans = _normalize_text_unicode(ans_raw)
                for j, (_txt, _pid) in enumerate(candidate_list):
                    if _normalize_text_unicode(_txt) == norm_ans:
                        idx = j
                        break

        if conf < min_confidence:
            return None
        if not (0 <= idx < len(candidate_list)):
            return None

        chosen_text, source_pid = candidate_list[idx]
        snippet = snippet_map.get(source_pid, "")

        # Provenance check (normalization-consistent containment)
        if _normalize_text_unicode(chosen_text) in _normalize_text_unicode(snippet):
            if debug:
                print(f"[CSS-BIND] idx={idx} conf={conf:.2f} -> {chosen_text} (pid={source_pid})")
            return chosen_text

    except Exception as ex:
        if debug:
            print(f"[CSS-BIND] Failed: {ex}")

    return None


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

    question = f"{subject} {predicate} {obj}".strip()

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
    """
    if not certificates:
        return []

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
        if pid and pid not in seen_pids and pid in pid_to_passage:
            seen_pids.add(pid)
            winners.append(pid_to_passage[pid])

    return winners
