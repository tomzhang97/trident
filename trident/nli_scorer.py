"""NLI/Cross-encoder scoring module.
Updates:
1. Numeric matching uses strict boundaries (Issue B).
2. score_with_details matches runtime truncation settings (Issue C).
3. BRIDGE_HOP1 lexical gate added.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from collections import OrderedDict
import hashlib
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
    s = s.replace("'", "'").replace("'", "'").replace(""", '"').replace(""", '"')
    s = _WS_RE.sub(" ", s).strip().lower()
    return s

def _contains_exact_phrase(passage: str, phrase: str) -> bool:
    return _norm(phrase) in _norm(passage) and _norm(phrase) != ""

def _contains_value(passage: str, value: str) -> bool:
    """
    Strict numeric matching.
    Prevents '10' matching '10th', '2010', or '10.5'.
    Regex: (?<![\w.])value(?![\w.])
    """
    val_str = _norm(str(value))
    if not val_str: return False
    pas_str = _norm(passage)
    
    # Escape value for regex
    esc_val = re.escape(val_str)
    # Lookbehind: Not preceded by word char or dot
    # Lookahead: Not followed by word char or dot
    pattern = rf"(?<![\w.]){esc_val}(?![\w.])"
    
    return bool(re.search(pattern, pas_str))

def _check_lexical_gate(facet: Facet, passage_text: str) -> Optional[bool]:
    ft = facet.facet_type
    tpl = facet.template or {}
    
    if ft == FacetType.ENTITY:
        mention = str(tpl.get("mention", ""))
        return _contains_exact_phrase(passage_text, mention) if mention else None
        
    if ft == FacetType.NUMERIC:
        val = str(tpl.get("value", ""))
        return _contains_value(passage_text, val) if val else None

    # NEW: Bridge Gate (Require both entities)
    if ft == FacetType.BRIDGE_HOP1:
        e1 = str(tpl.get("entity1", ""))
        be = str(tpl.get("bridge_entity", ""))
        if e1 and be:
            return _contains_exact_phrase(passage_text, e1) and _contains_exact_phrase(passage_text, be)
        return None
        
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

        lexical_match = _check_lexical_gate(facet, premise)
        if lexical_match is False: return 0.0

        if self.use_cache:
            key = self._cache_key(premise, hypothesis)
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]

        # Standard runtime settings
        inputs = self.tokenizer(premise, hypothesis, truncation="only_first", max_length=self.config.max_length, padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            probs = F.softmax(self.model(**inputs).logits[0], dim=-1)

        s = float(probs[self.entail_idx].item()) - 0.5 * float(probs[self.contra_idx].item() if probs.numel() > 2 else 0.0)
        
        if self.use_cache:
            if len(self.cache) >= self.cache_size: self.cache.popitem(last=False)
            self.cache[key] = s
        return s

    def batch_score(self, pairs: List[Tuple[Passage, Facet]]) -> List[float]:
        if not pairs: return []
        scores = []
        bs = int(getattr(self.config, "batch_size", 32))

        for i in range(0, len(pairs), bs):
            batch = pairs[i:i + bs]
            to_run_indices = []
            to_run_inputs = [] 
            batch_data = [] 
            
            results_map = {} 

            for idx, (p, f) in enumerate(batch):
                hyp = f.to_hypothesis(p.text)
                key = self._cache_key(p.text, hyp)
                batch_data.append((key))

                lexical_match = _check_lexical_gate(f, p.text)
                if lexical_match is False:
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
                    
                    if self.use_cache:
                        key = batch_data[idx]
                        if len(self.cache) >= self.cache_size: self.cache.popitem(last=False)
                        self.cache[key] = s

            for idx in range(len(batch)):
                scores.append(results_map[idx])

        return scores

    def score_with_details(self, passage: Passage, facet: Facet) -> NLIScore:
        """
        FIXED (Issue C): Now matches runtime truncation settings.
        Old: truncation=True, padding="max_length"
        New: truncation="only_first", padding=True
        """
        premise = passage.text
        hypothesis = facet.to_hypothesis(passage.text)

        lexical_match = _check_lexical_gate(facet, premise)
        if lexical_match is False:
            return NLIScore(0.0, 0.0, 1.0, 0.0)

        inputs = self.tokenizer(
            premise, hypothesis, 
            truncation="only_first",  # Matched to runtime
            max_length=self.config.max_length, 
            padding=True,             # Matched to runtime
            return_tensors="pt"
        ).to(self.device)
        
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
