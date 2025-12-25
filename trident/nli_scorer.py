"""NLI/Cross-encoder scoring module for facet sufficiency testing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from collections import OrderedDict
import hashlib

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .facets import Facet
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
    # common variants
    if "entail" in s:
        return "entailment"
    if "contra" in s:
        return "contradiction"
    if "neutral" in s:
        return "neutral"
    return s


class NLIScorer:
    """Cross-encoder/NLI model for scoring passage-facet pairs."""

    def __init__(self, config: NLIConfig, device: str = "cuda:0"):
        self.config = config
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name
        ).to(device)
        self.model.eval()

        # Cache with LRU eviction
        self.cache_size = int(getattr(config, "cache_size", 10000))
        self.use_cache = bool(getattr(config, "use_cache", True))
        self.cache: "OrderedDict[Tuple[str, str], float]" = OrderedDict()

        # Determine label indices from model config (authoritative)
        self.entail_idx, self.neutral_idx, self.contra_idx = self._infer_label_indices()

    def _infer_label_indices(self) -> Tuple[int, int, int]:
        """
        Infer indices for (entailment, neutral, contradiction) from HF config.

        Works for:
        - 3-way MNLI (id2label = {0:'CONTRADICTION',1:'NEUTRAL',2:'ENTAILMENT'} etc.)
        - 2-way models (treat positive class as entailment)
        """
        num_labels = int(self.model.config.num_labels)

        # 2-way: assume label 1 is "entailment/positive"
        if num_labels == 2:
            return 1, 0, 0

        # 3-way: inspect id2label/label2id
        id2label = getattr(self.model.config, "id2label", None) or {}
        label2id = getattr(self.model.config, "label2id", None) or {}

        # Prefer id2label if available
        idx_map: Dict[str, int] = {}
        if isinstance(id2label, dict) and len(id2label) == num_labels:
            for idx, lab in id2label.items():
                idx_map[_normalize_label(str(lab))] = int(idx)

        # If id2label not usable, try label2id
        if len(idx_map) < 3 and isinstance(label2id, dict):
            for lab, idx in label2id.items():
                idx_map[_normalize_label(str(lab))] = int(idx)

        # Fallback guesses if still missing
        entail = idx_map.get("entailment", None)
        neutral = idx_map.get("neutral", None)
        contra = idx_map.get("contradiction", None)

        if entail is None or neutral is None or contra is None:
            # Very conservative fallback: assume common MNLI ordering
            # (contradiction, neutral, entailment)
            contra = 0
            neutral = 1
            entail = 2

        return int(entail), int(neutral), int(contra)

    def _cache_key(self, passage_text: str, hypothesis: str) -> Tuple[str, str]:
        # Cache by content hash, not ids
        return (_sha1(passage_text), _sha1(hypothesis))

    def score(self, passage: Passage, facet: Facet) -> float:
        premise = passage.text
        hypothesis = facet.to_hypothesis()

        if self.use_cache:
            key = self._cache_key(premise, hypothesis)
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]

        inputs = self.tokenizer(
            premise,
            hypothesis,
            truncation="only_first",
            max_length=self.config.max_length,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits[0]
            probs = F.softmax(logits, dim=-1)

        entailment_score = float(probs[self.entail_idx].item())
        neutral_score = float(probs[self.neutral_idx].item()) if probs.numel() > 2 else float(probs[0].item())
        contradiction_score = float(probs[self.contra_idx].item()) if probs.numel() > 2 else 0.0

        # Your original combination (keep it stable)
        sufficiency_score = entailment_score - 0.5 * contradiction_score

        if self.use_cache:
            if len(self.cache) >= self.cache_size:
                self.cache.popitem(last=False)
            self.cache[key] = float(sufficiency_score)

        return float(sufficiency_score)

    def batch_score(self, pairs: List[Tuple[Passage, Facet]]) -> List[float]:
        if not pairs:
            return []

        scores: List[float] = []
        bs = int(getattr(self.config, "batch_size", 32))

        for i in range(0, len(pairs), bs):
            batch = pairs[i:i + bs]

            to_run: List[Tuple[Passage, Facet]] = []
            cached: Dict[Tuple[str, str], float] = {}

            if self.use_cache:
                for p, f in batch:
                    hyp = f.to_hypothesis()
                    key = self._cache_key(p.text, hyp)
                    if key in self.cache:
                        self.cache.move_to_end(key)
                        cached[key] = self.cache[key]
                    else:
                        to_run.append((p, f))
            else:
                to_run = batch

            if to_run:
                premises = [p.text for p, _ in to_run]
                hyps = [f.to_hypothesis() for _, f in to_run]

                inputs = self.tokenizer(
                    premises,
                    hyps,
                    truncation="only_first",
                    max_length=self.config.max_length,
                    padding=True,
                    return_tensors="pt"
                ).to(self.device)

                with torch.no_grad():
                    probs = F.softmax(self.model(**inputs).logits, dim=-1)

                for j, (p, f) in enumerate(to_run):
                    entail = float(probs[j, self.entail_idx].item())
                    contra = float(probs[j, self.contra_idx].item()) if probs.size(-1) > 2 else 0.0
                    s = entail - 0.5 * contra

                    if self.use_cache:
                        key = self._cache_key(p.text, f.to_hypothesis())
                        if len(self.cache) >= self.cache_size:
                            self.cache.popitem(last=False)
                        self.cache[key] = float(s)

            # emit in original order
            for p, f in batch:
                if self.use_cache:
                    key = self._cache_key(p.text, f.to_hypothesis())
                    if key in self.cache:
                        self.cache.move_to_end(key)
                        scores.append(float(self.cache[key]))
                    else:
                        scores.append(float(self.score(p, f)))
                else:
                    scores.append(float(self.score(p, f)))

        return scores

    def score_with_details(self, passage: Passage, facet: Facet) -> NLIScore:
        premise = passage.text
        hypothesis = facet.to_hypothesis()

        inputs = self.tokenizer(
            premise,
            hypothesis,
            truncation=True,
            max_length=self.config.max_length,
            padding="max_length",
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits[0]
            probs = F.softmax(logits, dim=-1)

        entailment_score = float(probs[self.entail_idx].item())
        neutral_score = float(probs[self.neutral_idx].item()) if probs.numel() > 2 else float(probs[0].item())
        contradiction_score = float(probs[self.contra_idx].item()) if probs.numel() > 2 else 0.0
        sufficiency_score = entailment_score - 0.5 * contradiction_score

        return NLIScore(
            entailment_score=entailment_score,
            contradiction_score=contradiction_score,
            neutral_score=neutral_score,
            final_score=float(sufficiency_score),
        )

    def clear_cache(self) -> None:
        self.cache.clear()

    def get_cache_stats(self) -> Dict[str, Any]:
        return {
            "cache_size": len(self.cache),
            "max_cache_size": self.cache_size,
            "use_cache": self.use_cache,
            "label_indices": {
                "entailment": self.entail_idx,
                "neutral": self.neutral_idx,
                "contradiction": self.contra_idx,
            },
            "id2label": getattr(self.model.config, "id2label", None),
        }
