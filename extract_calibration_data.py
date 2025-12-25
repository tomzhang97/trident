#!/usr/bin/env python3
"""
extract_calibration_data.py

Unified calibration-data extractor for multi-hop QA datasets (HotpotQA / 2Wiki / MuSiQue).

What this script does (robustly, across datasets):
1) Load dataset examples
2) Mine facets from question (FacetMiner)
3) For each (facet, passage), compute NLI score
4) Assign a *facet-conditioned* label:
   - label=1 if the passage contains gold support evidence AND that gold evidence satisfies the facet
   - label=0 otherwise
5) Write JSONL calibration records

Why this fixes your "facets still seem off" issue:
- Old labeling used title-level supporting_facts -> many false positives.
- New labeling uses *supporting sentences* (when available) and checks whether the facet is actually satisfied.

Supports:
- HotpotQA: uses `supporting_facts` (title, sent_idx) and `context` (title, [sentences])
- 2WikiMultiHopQA: supports common formats (see DatasetAdapter2Wiki)
- MuSiQue: supports common formats (see DatasetAdapterMuSiQue)

If your local 2wiki / musique json differs, you only need to tweak the adapter methods.

Usage examples:
    # Hotpot
    python extract_calibration_data.py \
        --dataset hotpot \
        --data_path hotpotqa/data/hotpot_train_v1.json \
        --output_path calibration_hotpot.jsonl \
        --num_samples 500 \
        --device cuda:0

    # 2Wiki
    python extract_calibration_data.py \
        --dataset 2wiki \
        --data_path 2wiki/train.json \
        --output_path calibration_2wiki.jsonl \
        --num_samples 500 \
        --device cuda:0

    # MuSiQue
    python extract_calibration_data.py \
        --dataset musique \
        --data_path musique/train.json \
        --output_path calibration_musique.jsonl \
        --num_samples 500 \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from tqdm import tqdm

# Add trident to path
sys.path.insert(0, str(Path(__file__).parent))

from trident.facets import FacetMiner, FacetType, Facet
from trident.nli_scorer import NLIScorer
from trident.candidates import Passage
from trident.config import TridentConfig


# -------------------------
# Normalization utilities
# -------------------------

import unicodedata

_WS_RE = re.compile(r"\s+")


def norm(s: str) -> str:
    """
    Normalize text for robust exact-phrase matching.
    - NFKC unicode normalization
    - Normalize curly quotes and apostrophes
    - Collapse whitespace
    - Lowercase
    """
    s = unicodedata.normalize("NFKC", s or "")
    # Normalize quotes
    s = s.replace("'", "'").replace("'", "'")
    s = s.replace(""", '"').replace(""", '"')
    s = _WS_RE.sub(" ", s).strip().lower()
    return s


def contains_exact_phrase(passage: str, phrase: str) -> bool:
    """
    Check if passage contains the exact phrase (after normalization).
    Used for ENTITY facets.
    """
    p = norm(passage)
    ph = norm(phrase)
    return ph != "" and ph in p


def contains_value(passage: str, value: str) -> bool:
    """
    Check if passage contains the value string (after normalization).
    Used for NUMERIC facets (simple containment).
    """
    p = norm(passage)
    v = norm(str(value))
    return v != "" and v in p


def contains_number_token(passage: str, num: str) -> bool:
    """
    Check if passage contains the number as a token (not part of larger number).
    Example: 12 matches "12" but not "2012" or "120".
    """
    p = norm(passage)
    num_norm = re.escape(norm(num))
    return re.search(rf"(?<!\d){num_norm}(?!\d)", p) is not None


def facet_to_query_text(facet: Any) -> str:
    """
    Always store the *actual NLI hypothesis string*.
    """
    if hasattr(facet, "to_hypothesis") and callable(getattr(facet, "to_hypothesis")):
        try:
            hyp = facet.to_hypothesis()
            if isinstance(hyp, str) and hyp.strip():
                return hyp
        except Exception:
            pass
    # fallback
    return str(facet)


def facet_type_str(facet: Any) -> str:
    ft = getattr(facet, "facet_type", None)
    if ft is None:
        return "UNKNOWN"
    return getattr(ft, "value", None) or str(ft)


# -------------------------
# Facet satisfaction (for labels)
# -------------------------

def facet_satisfied_in_text(facet: Facet, text: str) -> bool:
    """
    Conservative satisfiability check for calibration labels.
    Uses structured facet.template where possible.

    Important: this is NOT your runtime verifier, it's only for making labels less noisy.
    The goal is: reduce false positives dramatically and make labels match hypothesis semantics.

    CRITICAL: Labels must match what the hypothesis actually asks!
    - ENTITY hypothesis: "The passage contains the exact phrase X" → label based on exact phrase match
    - NUMERIC hypothesis: "The passage states the value X" → label based on value containment
    """
    if not text:
        return False

    ft = facet.facet_type
    tpl = facet.template or {}

    # ENTITY: require exact phrase match (matches hypothesis: "contains the exact phrase")
    if ft == FacetType.ENTITY:
        mention = str(tpl.get("mention", ""))
        return contains_exact_phrase(text, mention)

    # RELATION: require both endpoints appear (predicate matching is too brittle)
    if ft == FacetType.RELATION:
        subj = str(tpl.get("subject", ""))
        obj = str(tpl.get("object", ""))
        if not subj or not obj:
            return False
        return contains_exact_phrase(text, subj) and contains_exact_phrase(text, obj)

    # BRIDGE hops: require the endpoints appear
    if ft == FacetType.BRIDGE_HOP1:
        e1 = str(tpl.get("entity1", ""))
        eb = str(tpl.get("bridge_entity", ""))
        if not e1 or not eb:
            return False
        return contains_exact_phrase(text, e1) and contains_exact_phrase(text, eb)

    if ft == FacetType.BRIDGE_HOP2:
        eb = str(tpl.get("bridge_entity", ""))
        e2 = str(tpl.get("entity2", ""))
        if not eb or not e2:
            return False
        return contains_exact_phrase(text, eb) and contains_exact_phrase(text, e2)

    if ft == FacetType.BRIDGE:
        e1 = str(tpl.get("entity1", ""))
        e2 = str(tpl.get("entity2", ""))
        if not e1 or not e2:
            return False
        return contains_exact_phrase(text, e1) and contains_exact_phrase(text, e2)

    # TEMPORAL: if time appears, good enough
    if ft == FacetType.TEMPORAL:
        time = str(tpl.get("time", ""))
        event = str(tpl.get("event", ""))
        return (contains_value(text, time) if time else False) or (contains_value(text, event) if event else False)

    # NUMERIC: require value containment (matches hypothesis: "states the value")
    if ft == FacetType.NUMERIC:
        val = str(tpl.get("value", ""))
        unit = str(tpl.get("unit", ""))
        # Try full "value unit" first, then just value
        if val and unit:
            full_value = f"{val} {unit}".strip()
            if contains_value(text, full_value):
                return True
        if val:
            return contains_value(text, val)
        return False

    # COMPARISON/CAUSAL/PROCEDURAL: too brittle for labeling; default to false to avoid noise
    return False


# -------------------------
# Example canonicalization (dataset adapters)
# -------------------------

@dataclass
class QAExample:
    qid: str
    question: str
    # Each context passage is (title, sentences[])
    context: List[Tuple[str, List[str]]]
    # Gold supports: title -> set(sentence indices). If unknown, empty.
    supporting: Dict[str, Set[int]]


class DatasetAdapter(ABC):
    @abstractmethod
    def load(self, path: str, limit: Optional[int]) -> List[Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def to_example(self, raw: Dict[str, Any]) -> Optional[QAExample]:
        raise NotImplementedError


class DatasetAdapterHotpot(DatasetAdapter):
    def load(self, path: str, limit: Optional[int]) -> List[Dict[str, Any]]:
        with open(path) as f:
            data = json.load(f)
        return data[:limit] if limit else data

    def to_example(self, raw: Dict[str, Any]) -> Optional[QAExample]:
        qid = raw.get("_id") or raw.get("id") or raw.get("qid")
        question = raw.get("question", "")
        context = raw.get("context", [])
        supporting_facts = raw.get("supporting_facts", [])

        if not qid or not question or not context or not supporting_facts:
            return None

        # Hotpot context is list of [title, [sentences]]
        ctx: List[Tuple[str, List[str]]] = []
        for item in context:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                continue
            title, sents = item[0], item[1]
            if isinstance(title, str) and isinstance(sents, list):
                ctx.append((title, [str(x) for x in sents]))

        if not ctx:
            return None

        sup: DefaultDict[str, Set[int]] = defaultdict(set)
        for t, idx in supporting_facts:
            if isinstance(t, str):
                try:
                    sup[t].add(int(idx))
                except Exception:
                    continue

        return QAExample(qid=str(qid), question=str(question), context=ctx, supporting=dict(sup))


class DatasetAdapter2Wiki(DatasetAdapter):
    """
    2WikiMultiHopQA has multiple variants in the wild.
    We support common patterns:

    - raw['id'] or raw['_id']
    - question: raw['question']
    - context can be:
        a) raw['context'] as list of [title, [sentences]]  (same as Hotpot)
        b) raw['context'] as list of dicts with 'title' and 'sentences'/'text'
        c) raw['paragraphs'] or raw['docs'] etc.

    - supporting facts can be:
        a) raw['supporting_facts'] as (title, sent_idx) like Hotpot
        b) raw['supporting_facts'] as list of titles only (no indices) -> we treat as title-level only
        c) raw['evidence'] or raw['supporting'] etc (best-effort)
    """
    def load(self, path: str, limit: Optional[int]) -> List[Dict[str, Any]]:
        with open(path) as f:
            data = json.load(f)
        # sometimes JSONL
        if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
            data = data["data"]
        return data[:limit] if limit else data

    def _parse_context(self, raw: Dict[str, Any]) -> List[Tuple[str, List[str]]]:
        if "context" in raw:
            c = raw["context"]
            # list of [title, [sents]]
            if isinstance(c, list) and c and isinstance(c[0], (list, tuple)):
                out = []
                for item in c:
                    if isinstance(item, (list, tuple)) and len(item) == 2:
                        title, sents = item[0], item[1]
                        if isinstance(title, str) and isinstance(sents, list):
                            out.append((title, [str(x) for x in sents]))
                return out

            # list of dicts
            if isinstance(c, list) and c and isinstance(c[0], dict):
                out = []
                for d in c:
                    title = d.get("title") or d.get("id") or d.get("doc_title")
                    if not isinstance(title, str):
                        continue
                    if "sentences" in d and isinstance(d["sentences"], list):
                        sents = [str(x) for x in d["sentences"]]
                    elif "text" in d:
                        # text may be string or list
                        if isinstance(d["text"], list):
                            sents = [str(x) for x in d["text"]]
                        else:
                            sents = [str(d["text"])]
                    else:
                        continue
                    out.append((title, sents))
                return out

        # fallback: paragraphs/docs
        for key in ("paragraphs", "docs", "documents", "passages"):
            if key in raw and isinstance(raw[key], list):
                out = []
                for d in raw[key]:
                    if isinstance(d, dict):
                        title = d.get("title") or d.get("id") or d.get("doc_title")
                        if not isinstance(title, str):
                            continue
                        if "sentences" in d and isinstance(d["sentences"], list):
                            sents = [str(x) for x in d["sentences"]]
                        elif "text" in d:
                            if isinstance(d["text"], list):
                                sents = [str(x) for x in d["text"]]
                            else:
                                sents = [str(d["text"])]
                        else:
                            continue
                        out.append((title, sents))
                return out

        return []

    def _parse_supporting(self, raw: Dict[str, Any]) -> Dict[str, Set[int]]:
        sup: DefaultDict[str, Set[int]] = defaultdict(set)

        # Hotpot-style
        sf = raw.get("supporting_facts")
        if isinstance(sf, list) and sf and isinstance(sf[0], (list, tuple)) and len(sf[0]) >= 1:
            # could be (title, idx) or just titles
            for item in sf:
                if isinstance(item, (list, tuple)) and len(item) >= 1:
                    title = item[0]
                    if not isinstance(title, str):
                        continue
                    if len(item) >= 2:
                        try:
                            sup[title].add(int(item[1]))
                        except Exception:
                            # title-only evidence
                            pass
                    else:
                        pass
            return dict(sup)

        # Alternative: evidence list of dicts
        for key in ("evidence", "supporting", "gold_evidence"):
            ev = raw.get(key)
            if isinstance(ev, list):
                for e in ev:
                    if isinstance(e, dict):
                        title = e.get("title")
                        if isinstance(title, str):
                            idxs = e.get("sent_idx") or e.get("sentences") or e.get("idxs")
                            if isinstance(idxs, list):
                                for i in idxs:
                                    try:
                                        sup[title].add(int(i))
                                    except Exception:
                                        continue
                            elif isinstance(idxs, int):
                                sup[title].add(int(idxs))
                return dict(sup)

        return dict(sup)

    def to_example(self, raw: Dict[str, Any]) -> Optional[QAExample]:
        qid = raw.get("_id") or raw.get("id") or raw.get("qid")
        question = raw.get("question", "")
        if not qid or not question:
            return None

        ctx = self._parse_context(raw)
        if not ctx:
            return None

        sup = self._parse_supporting(raw)
        # We allow empty supporting (some 2wiki formats might not carry it),
        # but then labels will be all-0 (still useful for distribution check, not for calibration).
        return QAExample(qid=str(qid), question=str(question), context=ctx, supporting=sup)


class DatasetAdapterMuSiQue(DatasetAdapter):
    """
    MuSiQue also varies by release.
    Common patterns:
    - id: raw['id'] or raw['_id']
    - question: raw['question']
    - context: raw['context'] as list of paragraphs, each with 'title'/'sentences'/'text'
      OR sometimes it's already [title, [sentences]] like Hotpot.
    - gold supports: may appear as:
        - raw['supporting_facts'] Hotpot-style
        - raw['supporting_paragraphs'] titles or indices
        - raw['evidence'] etc.

    We implement best-effort parsing.
    """
    def load(self, path: str, limit: Optional[int]) -> List[Dict[str, Any]]:
        # MuSiQue is often JSONL
        p = Path(path)
        if p.suffix.lower() in {".jsonl", ".jsonlines"}:
            data = []
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data.append(json.loads(line))
                    if limit and len(data) >= limit:
                        break
            return data

        with open(path) as f:
            data = json.load(f)

        # sometimes wrapped
        if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
            data = data["data"]

        return data[:limit] if limit else data

    def _parse_context(self, raw: Dict[str, Any]) -> List[Tuple[str, List[str]]]:
        # Hotpot-like
        c = raw.get("context")
        if isinstance(c, list) and c:
            if isinstance(c[0], (list, tuple)) and len(c[0]) == 2:
                out = []
                for item in c:
                    if isinstance(item, (list, tuple)) and len(item) == 2:
                        title, sents = item[0], item[1]
                        if isinstance(title, str) and isinstance(sents, list):
                            out.append((title, [str(x) for x in sents]))
                return out

            if isinstance(c[0], dict):
                out = []
                for d in c:
                    title = d.get("title") or d.get("id") or d.get("doc_title") or "passage"
                    if not isinstance(title, str):
                        title = "passage"
                    if "sentences" in d and isinstance(d["sentences"], list):
                        sents = [str(x) for x in d["sentences"]]
                    elif "text" in d:
                        if isinstance(d["text"], list):
                            sents = [str(x) for x in d["text"]]
                        else:
                            sents = [str(d["text"])]
                    else:
                        continue
                    out.append((title, sents))
                return out

        # fallback keys
        for key in ("paragraphs", "passages", "docs", "documents"):
            if key in raw and isinstance(raw[key], list):
                out = []
                for d in raw[key]:
                    if not isinstance(d, dict):
                        continue
                    title = d.get("title") or d.get("id") or d.get("doc_title") or "passage"
                    if not isinstance(title, str):
                        title = "passage"
                    if "sentences" in d and isinstance(d["sentences"], list):
                        sents = [str(x) for x in d["sentences"]]
                    elif "text" in d:
                        if isinstance(d["text"], list):
                            sents = [str(x) for x in d["text"]]
                        else:
                            sents = [str(d["text"])]
                    else:
                        continue
                    out.append((title, sents))
                return out

        return []

    def _parse_supporting(self, raw: Dict[str, Any]) -> Dict[str, Set[int]]:
        sup: DefaultDict[str, Set[int]] = defaultdict(set)

        # Hotpot-style
        sf = raw.get("supporting_facts")
        if isinstance(sf, list):
            for item in sf:
                if isinstance(item, (list, tuple)) and len(item) >= 1:
                    title = item[0]
                    if not isinstance(title, str):
                        continue
                    if len(item) >= 2:
                        try:
                            sup[title].add(int(item[1]))
                        except Exception:
                            pass
            return dict(sup)

        # title-only supporting paragraphs
        sp = raw.get("supporting_paragraphs") or raw.get("supporting_titles")
        if isinstance(sp, list):
            for t in sp:
                if isinstance(t, str):
                    # title-level only; no sent indices
                    sup[t]  # create empty set
            return dict(sup)

        # evidence dicts
        ev = raw.get("evidence")
        if isinstance(ev, list):
            for e in ev:
                if isinstance(e, dict):
                    title = e.get("title")
                    if isinstance(title, str):
                        idxs = e.get("sent_idx") or e.get("idxs")
                        if isinstance(idxs, list):
                            for i in idxs:
                                try:
                                    sup[title].add(int(i))
                                except Exception:
                                    continue
                        elif isinstance(idxs, int):
                            sup[title].add(int(idxs))
            return dict(sup)

        return dict(sup)

    def to_example(self, raw: Dict[str, Any]) -> Optional[QAExample]:
        qid = raw.get("_id") or raw.get("id") or raw.get("qid")
        question = raw.get("question", "")
        if not qid or not question:
            return None

        ctx = self._parse_context(raw)
        if not ctx:
            return None

        sup = self._parse_supporting(raw)
        return QAExample(qid=str(qid), question=str(question), context=ctx, supporting=sup)


def get_adapter(dataset: str) -> DatasetAdapter:
    d = dataset.lower().strip()
    if d in {"hotpot", "hotpotqa"}:
        return DatasetAdapterHotpot()
    if d in {"2wiki", "2wikimultihopqa", "wiki"}:
        return DatasetAdapter2Wiki()
    if d in {"musique", "msq"}:
        return DatasetAdapterMuSiQue()
    raise ValueError(f"Unknown dataset '{dataset}'. Choose from: hotpot, 2wiki, musique.")


# -------------------------
# Gold support extraction
# -------------------------

def is_valid_hypothesis(hypothesis: str) -> bool:
    """
    Filter out garbage hypotheses that will cause NLI to behave randomly.

    Invalid patterns:
    - Contains "?" (question mark) - not a declarative statement
    - Contains "occurred in after being" / "occurred in after who" - garbled templates
    - Too short or nonsensical
    """
    if not hypothesis or len(hypothesis.strip()) < 10:
        return False

    # Question marks indicate malformed hypotheses
    if "?" in hypothesis:
        return False

    # Known garbage patterns from template bugs
    garbage_patterns = [
        "occurred in after being",
        "occurred in after who",
        "occurred in after what",
        "is related to … what",
        "is related to ... what",
    ]

    hyp_lower = hypothesis.lower()
    for pattern in garbage_patterns:
        if pattern in hyp_lower:
            return False

    return True


def get_lexical_match(facet: Facet, passage_text: str) -> Optional[bool]:
    """
    Perform lexical gate check for facets that make lexical claims.

    Returns:
        True if lexical condition is satisfied
        False if lexical condition is NOT satisfied (NLI should be forced low)
        None if facet doesn't make a lexical claim (use NLI normally)

    For ENTITY facets ("contains exact phrase"), we check if phrase exists.
    For NUMERIC facets ("states the value"), we check if value exists.

    This prevents NLI from hallucinating entailment when the phrase/value isn't there.
    """
    ft = facet.facet_type
    tpl = facet.template or {}

    # ENTITY: "The passage contains the exact phrase X"
    if ft == FacetType.ENTITY:
        mention = str(tpl.get("mention", ""))
        if mention:
            return contains_exact_phrase(passage_text, mention)
        return None

    # NUMERIC: "The passage states the value X"
    if ft == FacetType.NUMERIC:
        val = str(tpl.get("value", ""))
        unit = str(tpl.get("unit", ""))
        if val and unit:
            full_value = f"{val} {unit}".strip()
            if contains_value(passage_text, full_value):
                return True
        if val:
            return contains_value(passage_text, val)
        return None

    # For other facet types, don't apply lexical gate
    return None


def supporting_text_for_title(title: str, sentences: List[str], supporting: Dict[str, Set[int]]) -> str:
    idxs = sorted(i for i in supporting.get(title, set()) if 0 <= i < len(sentences))
    if not idxs:
        return ""
    return " ".join(sentences[i] for i in idxs)


# -------------------------
# Extraction
# -------------------------

def extract_calibration_samples(
    raw_data: List[Dict[str, Any]],
    adapter: DatasetAdapter,
    facet_miner: FacetMiner,
    nli_scorer: NLIScorer,
) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []

    for raw in tqdm(raw_data, desc="Extracting calibration samples"):
        ex = adapter.to_example(raw)
        if ex is None:
            continue

        # Mine facets (supporting is passed only if your miner uses it)
        supporting_facts_for_miner: Optional[List[Tuple[str, int]]] = None
        if ex.supporting:
            # flatten (title, idx) for miner bridge heuristics
            supporting_facts_for_miner = [(t, i) for t, idxs in ex.supporting.items() for i in idxs] or None

        facets = facet_miner.extract_facets(ex.question, supporting_facts_for_miner)
        if not facets:
            continue

        # For each passage
        for i, (title, sentences) in enumerate(ex.context):
            passage_text = " ".join(sentences)
            passage_id = f"context_{i}"

            # Title has supporting evidence only if we have indices (non-empty set)
            has_supporting = bool(ex.supporting.get(title, set()))
            sup_text = supporting_text_for_title(title, sentences, ex.supporting) if has_supporting else ""

            passage = Passage(
                pid=passage_id,
                text=passage_text,
                cost=len(passage_text.split()),
            )

            for facet in facets:
                # Generate hypothesis and validate it
                hypothesis = facet_to_query_text(facet)

                # FILTER: Skip garbage hypotheses
                if not is_valid_hypothesis(hypothesis):
                    continue

                # Get NLI score
                details = nli_scorer.score_with_details(passage, facet)
                score = details.final_score
                entail_prob = details.entailment_score
                neutral_prob = details.neutral_score
                contra_prob = details.contradiction_score

                # LEXICAL GATE: For lexical hypotheses, force low score if lexical match fails
                lexical_match = get_lexical_match(facet, passage_text)
                if lexical_match is False:
                    # Phrase/value doesn't exist → force NLI to very low confidence
                    # Set probs to neutral=1, others=0, and very low score
                    score = 0.0
                    entail_prob = 0.0
                    neutral_prob = 1.0
                    contra_prob = 0.0

                # Facet-conditioned label:
                # positive only if (a) passage has gold support sentences AND (b) those sentences satisfy the facet
                label = 1 if (has_supporting and facet_satisfied_in_text(facet, sup_text)) else 0

                sample = {
                    "id": f"{ex.qid}_{passage_id}_{facet.facet_id}",
                    "query_id": ex.qid,
                    "hypothesis": hypothesis,  # Top-level for easy access
                    "score": float(score),
                    "probs": {
                        "entail": float(entail_prob),
                        "neutral": float(neutral_prob),
                        "contra": float(contra_prob),
                   },
                    "label": int(label),
                    "metadata": {
                        "facet_type": facet_type_str(facet),
                        "text_length": len(passage_text),
                        "retriever_score": 1.0,  # oracle context regime
                        "passage_id": passage_id,
                        "facet_id": facet.facet_id,
                        "question": ex.question,
                        "facet_query": hypothesis,  # Duplicate for backwards compat
                        "lexical_match": lexical_match,  # True/False/None for diagnostics
                        "lexical_gate_applied": lexical_match is False,  # Boolean flag
                        # Debug helpers (remove if you want smaller files)
                        "title": title,
                        "supporting_title": bool(has_supporting),
                        "supporting_text": sup_text,
                        "passage_preview": passage_text[:200],
                    },
                }
                samples.append(sample)

    return samples


# -------------------------
# Main
# -------------------------

def main():
    parser = argparse.ArgumentParser(description="Extract facet-conditioned NLI calibration data")
    parser.add_argument("--dataset", type=str, required=True, help="hotpot | 2wiki | musique")
    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset file (json or jsonl)")
    parser.add_argument("--output_path", type=str, default="calibration_data.jsonl", help="Output JSONL path")
    parser.add_argument("--num_samples", type=int, default=500, help="Number of raw examples to process")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for NLI scorer")

    args = parser.parse_args()

    print("Initializing NLI scorer and facet miner...")
    config = TridentConfig()
    facet_miner = FacetMiner(config)
    nli_scorer = NLIScorer(config.nli, device=args.device)

    adapter = get_adapter(args.dataset)

    print(f"Loading {args.dataset} data from {args.data_path}...")
    raw_data = adapter.load(args.data_path, limit=args.num_samples)
    print(f"Loaded {len(raw_data)} raw examples")

    print("Extracting calibration samples...")
    samples = extract_calibration_samples(raw_data, adapter, facet_miner, nli_scorer)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    print(f"\n✅ Extracted {len(samples)} calibration samples")
    print(f"📁 Saved to: {output_path}")

    if len(samples) == 0:
        print("\n⚠️ No samples produced. Most likely: adapter parsing mismatch or missing supporting evidence fields.")
        return

    pos = sum(1 for s in samples if s["label"] == 1)
    neg = len(samples) - pos
    avg = sum(float(s["score"]) for s in samples) / len(samples)

    print(f"\n📊 Statistics:")
    print(f"  Positive samples: {pos} ({pos/len(samples)*100:.1f}%)")
    print(f"  Negative samples: {neg} ({neg/len(samples)*100:.1f}%)")
    print(f"  Avg NLI score: {avg:.3f}")

    # Per-facet-type breakdown
    from collections import defaultdict
    type_stats: DefaultDict[str, Dict[str, int]] = defaultdict(lambda: {"pos": 0, "neg": 0})
    for s in samples:
        ft = s["metadata"].get("facet_type", "UNKNOWN")
        if s["label"] == 1:
            type_stats[ft]["pos"] += 1
        else:
            type_stats[ft]["neg"] += 1

    print(f"\n📊 Per-facet-type breakdown:")
    for ft in sorted(type_stats.keys()):
        p = type_stats[ft]["pos"]
        n = type_stats[ft]["neg"]
        total = p + n
        print(f"  {ft:15s}: {p:4d} pos, {n:4d} neg ({p/total*100:5.1f}% positive)")

    # Lexical gate stats
    lexical_gate_count = sum(1 for s in samples if s["metadata"].get("lexical_gate_applied", False))
    print(f"\n🚪 Lexical gate triggered: {lexical_gate_count} times ({lexical_gate_count/len(samples)*100:.1f}%)")

    print("\n⚠️  IMPORTANT: Facet type selection for calibration")
    print("  - ENTITY/NUMERIC: Safe to use (lexical, labels match hypothesis)")
    print("  - RELATION/TEMPORAL: Use cautiously - labels are Hotpot supporting facts,")
    print("    NOT true entailment. High entail + label=0 is often correct behavior.")
    print("  - Recommendation: Start with ENTITY/NUMERIC only for cleanest calibration.")

    print("\n🔧 Next step: Train calibrator")
    print("  python train_calibration.py \\")
    print(f"    --data_path {output_path} \\")
    print("    --output_path calibrator.json \\")
    print("    --use_mondrian")


if __name__ == "__main__":
    main()
