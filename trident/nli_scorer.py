"""NLI/Cross-encoder scoring module.
UPDATES:
- Short token fallback: If phrase is "US" (no long tokens), falls back to exact match.
- Robust normalization syntax.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from collections import OrderedDict
import hashlib
import os
import re
import unicodedata

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .facets import Facet, FacetType
from .candidates import Passage
from .config import NLIConfig

@dataclass
class NLIScore:
    entailment_score: float
    contradiction_score: float
    neutral_score: float
    final_score: float

def _sha1(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()

def _normalize_label(s: str) -> str:
    s = (s or "").strip().lower()
    if "entail" in s: return "entailment"
    if "contra" in s: return "contradiction"
    if "neutral" in s: return "neutral"
    return s

_WS_RE = re.compile(r"\s+")

def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    # Robust normalization: standard quotes
    s = (s.replace("’", "'").replace("‘", "'")
          .replace("“", '"').replace("”", '"'))
    s = s.replace(",", "") 
    s = _WS_RE.sub(" ", s).strip().lower()
    return s

def _contains_exact_phrase(passage: str, phrase: str) -> bool:
    return _norm(phrase) in _norm(passage) and _norm(phrase) != ""

def _token_in_text(tok: str, text: str) -> bool:
    return bool(re.search(rf"(?<!\w){re.escape(tok)}(?!\w)", text))

_STOP_WORDS = {"the", "and", "of", "in", "to", "a", "an", "for", "or", "is", "are", "was", "were"}

def _contains_tokens(passage: str, phrase: str) -> bool:
    """
    Looser check: do significant tokens from phrase appear in passage?
    Hyphen/punct robust: "Polish-Russian" -> ["polish", "russian"].
    """
    import math

    if not phrase:
        return False

    p_norm = _norm(passage)
    ph_norm = _norm(phrase)

    # Split phrase into tokens robustly (handles hyphen/underscore/punct)
    raw = re.split(r"[\s\-_]+", ph_norm)
    tokens = [t for t in raw if len(t) > 2 and t not in _STOP_WORDS]

    # If no significant tokens remain, fall back to exact-phrase match
    if not tokens:
        return _contains_exact_phrase(passage, phrase)

    hits = sum(1 for t in tokens if _token_in_text(t, p_norm))
    need = max(1, math.ceil(len(tokens) / 2))
    return hits >= need

def _contains_value(passage: str, value: str) -> bool:
    val_str = _norm(str(value))
    if not val_str: return False
    pas_str = _norm(passage)
    esc_val = re.escape(val_str)
    pattern = rf"(?<![\w.]){esc_val}(?![\w.])"
    return bool(re.search(pattern, pas_str))

# WH-words and generic terms that should not be treated as real entity endpoints
_WH_WORDS = {"who", "what", "which", "when", "where", "why", "how", "whom", "whose"}
_GENERIC_TERMS = {"the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "is", "are",
                  "was", "were", "be", "been", "being", "have", "has", "had", "do", "does",
                  "did", "will", "would", "could", "should", "may", "might", "must", "shall",
                  "director", "film", "movie", "actor", "actress", "person", "thing", "place",
                  "name", "title", "author", "writer", "book", "song", "album", "band", "group"}

def _clean_relation_endpoint(x: str) -> str:
    """
    Clean a RELATION endpoint by removing WH-words and generic terms.
    Returns empty string if no meaningful content remains.
    """
    if not x:
        return ""
    # Extract alphanumeric tokens
    toks = re.findall(r"[A-Za-z0-9]+", x.lower())
    # Filter out WH-words and generic terms
    cleaned = [t for t in toks if t not in _WH_WORDS and t not in _GENERIC_TERMS and len(t) > 1]
    return " ".join(cleaned).strip()

def _check_lexical_gate(facet: Facet, passage_text: str) -> Optional[bool]:
    ft = facet.facet_type
    tpl = facet.template or {}
    
    if ft == FacetType.ENTITY:
        return _contains_exact_phrase(passage_text, str(tpl.get("mention", "")))
    if ft == FacetType.NUMERIC:
        return _contains_value(passage_text, str(tpl.get("value", "")))
    
    # Generic BRIDGE_HOP
    if "BRIDGE_HOP" in ft.value:
        e1 = str(tpl.get("entity1", "") or tpl.get("entity", "") or "")
        e2 = str(tpl.get("bridge_entity", "") or tpl.get("entity2", "") or "")
        
        if not e1 or not e2:
            return None 
            
        # Exact match on start node, token match (or safe exact) on end node
        return _contains_exact_phrase(passage_text, e1) and _contains_tokens(passage_text, e2)

    if ft == FacetType.RELATION:
        s_raw = str(tpl.get("subject", ""))
        o_raw = str(tpl.get("object", ""))
        # Clean endpoints - remove WH-words and generic terms
        s = _clean_relation_endpoint(s_raw)
        o = _clean_relation_endpoint(o_raw)
        debug_rel = os.environ.get("TRIDENT_DEBUG_RELATION", "0") == "1"

        # If both endpoints are empty after cleaning, can't gate reliably - let NLI decide
        if not s and not o:
            if debug_rel:
                print(f"[RELATION GATE] subj_raw='{s_raw}' obj_raw='{o_raw}' -> BOTH EMPTY after cleaning, gate=True (let NLI decide)")
            return True  # Don't gate, let NLI score it

        s_match = _contains_tokens(passage_text, s) if s else False
        o_match = _contains_tokens(passage_text, o) if o else False
        # Require only ONE of subject OR object (multi-hop QA often splits entities across passages)
        result = s_match or o_match
        if debug_rel:
            print(f"[RELATION GATE] subj_raw='{s_raw}' subj_clean='{s}' obj_raw='{o_raw}' obj_clean='{o}' s_match={s_match} o_match={o_match} -> gate={result}")
        return result

    if ft == FacetType.COMPARISON:
        e1 = str(tpl.get("entity1", ""))
        e2 = str(tpl.get("entity2", ""))
        return _contains_tokens(passage_text, e1) or _contains_tokens(passage_text, e2)

    if ft == FacetType.TEMPORAL:
        return _contains_exact_phrase(passage_text, str(tpl.get("time", "")))
        
    return None

class NLIScorer:
    def __init__(self, config: NLIConfig, device: str = "cuda:0"):
        self.config = config
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(config.model_name).to(device)
        self.model.eval()
        self.cache_size = int(getattr(config, "cache_size", 10000))
        self.use_cache = bool(getattr(config, "use_cache", True))
        self.cache = OrderedDict()
        self.entail_idx, self.neutral_idx, self.contra_idx = self._infer_label_indices()

    def _infer_label_indices(self) -> Tuple[int, int, int]:
        num_labels = int(self.model.config.num_labels)
        if num_labels == 2: return 1, 0, 0
        
        id2label = getattr(self.model.config, "id2label", None) or {}
        label2id = getattr(self.model.config, "label2id", None) or {}
        idx_map = {}
        
        if isinstance(id2label, dict):
            for idx, lab in id2label.items(): idx_map[_normalize_label(str(lab))] = int(idx)
        if len(idx_map) < 3 and isinstance(label2id, dict):
            for lab, idx in label2id.items(): idx_map[_normalize_label(str(lab))] = int(idx)
            
        return idx_map.get("entailment", 2), idx_map.get("neutral", 1), idx_map.get("contradiction", 0)

    def _cache_key(self, passage_text: str, hypothesis: str) -> Tuple[str, str]:
        return (_sha1(passage_text), _sha1(hypothesis))

    def _entity_lexical_score(self, passage_text: str, facet: Facet) -> float:
        """
        Score ENTITY facet using lexical containment (not NLI).

        NLI with "The passage mentions X" is too permissive - any passage
        containing the string gets high entailment even if irrelevant.

        Returns:
            0.0 if entity not found
            0.5-1.0 based on presence and prominence (position, frequency)
        """
        mention = facet.template.get("mention", "") if facet.template else ""
        if not mention:
            return 0.0

        # Normalize for matching
        text_lower = passage_text.lower()
        mention_lower = mention.lower()

        # Check for exact phrase match
        if mention_lower not in text_lower:
            # Try with hyphen/space normalization
            import re
            mention_norm = re.sub(r"[-_]+", " ", mention_lower)
            text_norm = re.sub(r"[-_]+", " ", text_lower)
            if mention_norm not in text_norm:
                return 0.0

        # Entity is present - compute score based on prominence
        # Base score: 0.7 for presence
        score = 0.7

        # Bonus for early mention (first 20% of passage) - up to +0.15
        first_pos = text_lower.find(mention_lower)
        if first_pos >= 0:
            rel_pos = first_pos / max(len(text_lower), 1)
            if rel_pos < 0.2:
                score += 0.15 * (1 - rel_pos / 0.2)

        # Bonus for multiple mentions - up to +0.15
        count = text_lower.count(mention_lower)
        if count > 1:
            score += min(0.15, 0.05 * (count - 1))

        return min(1.0, score)

    def score(self, passage: Passage, facet: Facet) -> float:
        premise = passage.text
        debug_rel = os.environ.get("TRIDENT_DEBUG_RELATION", "0") == "1"
        debug_entity = os.environ.get("TRIDENT_DEBUG_ENTITY", "0") == "1"

        # ENTITY: Use lexical containment, not NLI
        # NLI with "The passage mentions X" is too permissive
        if facet.facet_type == FacetType.ENTITY:
            score = self._entity_lexical_score(premise, facet)
            if debug_entity:
                mention = facet.template.get("mention", "") if facet.template else ""
                print(f"[ENTITY LEXICAL] mention={mention!r} score={score:.4f}")
            return score

        hypothesis = facet.to_hypothesis(passage.text)

        if debug_rel and facet.facet_type == FacetType.RELATION:
            print(f"[RELATION HYP] {hypothesis}")

        lexical_match = _check_lexical_gate(facet, premise)
        if lexical_match is False:
            if debug_rel and facet.facet_type == FacetType.RELATION:
                print(f"[RELATION SCORE] GATED OUT -> score=0.0")
            return 0.0

        if self.use_cache:
            key = self._cache_key(premise, hypothesis)
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]

        inputs = self.tokenizer(premise, hypothesis, truncation="only_first", max_length=self.config.max_length, padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            probs = F.softmax(self.model(**inputs).logits[0], dim=-1)

        s = float(probs[self.entail_idx].item()) - 0.5 * float(probs[self.contra_idx].item() if probs.numel() > 2 else 0.0)

        if debug_rel and facet.facet_type == FacetType.RELATION:
            entail = float(probs[self.entail_idx].item())
            contra = float(probs[self.contra_idx].item()) if probs.numel() > 2 else 0.0
            print(f"[RELATION SCORE] NLI entail={entail:.4f} contra={contra:.4f} -> score={s:.4f}")

        if self.use_cache:
            if len(self.cache) >= self.cache_size: self.cache.popitem(last=False)
            self.cache[key] = s
        return s

    def batch_score(self, pairs: List[Tuple[Passage, Facet]]) -> List[float]:
        if not pairs: return []
        scores = []
        bs = int(getattr(self.config, "batch_size", 32))
        debug_rel = os.environ.get("TRIDENT_DEBUG_RELATION", "0") == "1"
        debug_entity = os.environ.get("TRIDENT_DEBUG_ENTITY", "0") == "1"

        for i in range(0, len(pairs), bs):
            batch = pairs[i:i + bs]
            to_run_indices = []
            to_run_inputs = []
            batch_data = []
            facet_types = []  # Track facet types for debug logging

            results_map = {}

            for idx, (p, f) in enumerate(batch):
                facet_types.append(f.facet_type)

                # ENTITY: Use lexical containment, not NLI
                if f.facet_type == FacetType.ENTITY:
                    s = self._entity_lexical_score(p.text, f)
                    if debug_entity:
                        mention = f.template.get("mention", "") if f.template else ""
                        print(f"[ENTITY BATCH] idx={idx} mention={mention!r} score={s:.4f}")
                    results_map[idx] = s
                    batch_data.append(None)  # Placeholder
                    continue

                hyp = f.to_hypothesis(p.text)
                key = self._cache_key(p.text, hyp)
                batch_data.append(key)

                lexical_match = _check_lexical_gate(f, p.text)
                if lexical_match is False:
                    if debug_rel and f.facet_type == FacetType.RELATION:
                        print(f"[RELATION BATCH] idx={idx} GATED OUT -> score=0.0")
                    results_map[idx] = 0.0
                    continue

                if self.use_cache and key in self.cache:
                    self.cache.move_to_end(key)
                    results_map[idx] = self.cache[key]
                else:
                    to_run_indices.append(idx)
                    to_run_inputs.append((p.text, hyp))

            if to_run_inputs:
                premises = [x[0] for x in to_run_inputs]
                hyps = [x[1] for x in to_run_inputs]
                
                inputs = self.tokenizer(premises, hyps, truncation="only_first", max_length=self.config.max_length, padding=True, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    probs = F.softmax(self.model(**inputs).logits, dim=-1)
                
                for k, idx in enumerate(to_run_indices):
                    entail = float(probs[k, self.entail_idx].item())
                    contra = float(probs[k, self.contra_idx].item()) if probs.size(-1) > 2 else 0.0
                    s = entail - 0.5 * contra
                    results_map[idx] = s

                    if debug_rel and facet_types[idx] == FacetType.RELATION:
                        print(f"[RELATION BATCH] idx={idx} NLI entail={entail:.4f} contra={contra:.4f} -> score={s:.4f}")

                    if self.use_cache:
                        key = batch_data[idx]
                        if len(self.cache) >= self.cache_size: self.cache.popitem(last=False)
                        self.cache[key] = s

            for idx in range(len(batch)):
                scores.append(results_map[idx])

        return scores

    def score_with_details(self, passage: Passage, facet: Facet) -> NLIScore:
        premise = passage.text

        # ENTITY: Use lexical containment, not NLI
        if facet.facet_type == FacetType.ENTITY:
            score = self._entity_lexical_score(premise, facet)
            # Return synthetic NLIScore with lexical score as both entail and final
            return NLIScore(score, 0.0, 1.0 - score, score)

        hypothesis = facet.to_hypothesis(passage.text)

        lexical_match = _check_lexical_gate(facet, premise)
        if lexical_match is False:
            return NLIScore(0.0, 0.0, 1.0, 0.0)

        inputs = self.tokenizer(premise, hypothesis, truncation="only_first", max_length=self.config.max_length, padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            probs = F.softmax(self.model(**inputs).logits[0], dim=-1)

        entail = float(probs[self.entail_idx].item())
        neutral = float(probs[self.neutral_idx].item()) if probs.numel() > 2 else float(probs[0].item())
        contra = float(probs[self.contra_idx].item()) if probs.numel() > 2 else 0.0
        final = entail - 0.5 * contra

        return NLIScore(entail, contra, neutral, final)

    def clear_cache(self) -> None:
        self.cache.clear()

    def get_cache_stats(self) -> Dict[str, Any]:
        return {"cache_size": len(self.cache), "max_cache_size": self.cache_size}
