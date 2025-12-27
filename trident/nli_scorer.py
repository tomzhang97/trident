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

def _contains_tokens(passage: str, phrase: str) -> bool:
    """
    Looser check: Do significant tokens from phrase appear in passage?
    Safe Fallback: If no significant tokens (e.g. "US"), require exact match.
    """
    p_norm = _norm(passage)
    ph_norm = _norm(phrase)
    tokens = [t for t in ph_norm.split() if len(t) > 2 and t not in {"the", "and", "of", "in"}]
    
    # FIX: Don't let short entities bypass the gate
    if not tokens: 
        return _contains_exact_phrase(passage, phrase)
        
    hits = sum(1 for t in tokens if _token_in_text(t, p_norm))
    return hits >= (len(tokens) / 2)

def _contains_value(passage: str, value: str) -> bool:
    val_str = _norm(str(value))
    if not val_str: return False
    pas_str = _norm(passage)
    esc_val = re.escape(val_str)
    pattern = rf"(?<![\w.]){esc_val}(?![\w.])"
    return bool(re.search(pattern, pas_str))

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
        s = str(tpl.get("subject", ""))
        o = str(tpl.get("object", ""))
        s_match = _contains_tokens(passage_text, s) if s else False
        o_match = _contains_tokens(passage_text, o) if o else False
        # FIX: Require only ONE of subject OR object (like COMPARISON)
        # Multi-hop QA often has subject in one passage, object in another
        result = s_match or o_match
        if os.environ.get("TRIDENT_DEBUG_RELATION", "0") == "1":
            print(f"[RELATION GATE] subj='{s}' obj='{o}' s_match={s_match} o_match={o_match} -> gate={result}")
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

    def score(self, passage: Passage, facet: Facet) -> float:
        premise = passage.text
        hypothesis = facet.to_hypothesis(passage.text)
        debug_rel = os.environ.get("TRIDENT_DEBUG_RELATION", "0") == "1"

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

        for i in range(0, len(pairs), bs):
            batch = pairs[i:i + bs]
            to_run_indices = []
            to_run_inputs = []
            batch_data = []
            facet_types = []  # Track facet types for debug logging

            results_map = {}

            for idx, (p, f) in enumerate(batch):
                hyp = f.to_hypothesis(p.text)
                key = self._cache_key(p.text, hyp)
                batch_data.append((key))
                facet_types.append(f.facet_type)

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
