#!/usr/bin/env python3
"""
extract_calibration_data.py
Updates:
- BRIDGE_HOP1 label logic: Now defaults to 1 if 'has_sup' (Title match), 
  allowing Sigma script to do the strict checking later.
- Keeps 'passage_full' for Sigma checks.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from trident.facets import FacetMiner, FacetType, Facet
from trident.nli_scorer import NLIScorer
from trident.candidates import Passage
from trident.config import TridentConfig
import unicodedata

_WS_RE = re.compile(r"\s+")

def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = s.replace("'", "'").replace("'", "'").replace(""", '"').replace(""", '"')
    s = _WS_RE.sub(" ", s).strip().lower()
    return s

def contains_exact_phrase(passage: str, phrase: str) -> bool:
    return norm(phrase) in norm(passage) and norm(phrase) != ""

def contains_value(passage: str, value: str) -> bool:
    """Strict numeric matching: (?<![\w.])value(?![\w.])"""
    val_str = norm(str(value))
    if not val_str: return False
    pas_str = norm(passage)
    esc_val = re.escape(val_str)
    pattern = rf"(?<![\w.]){esc_val}(?![\w.])"
    return bool(re.search(pattern, pas_str))

def facet_satisfied_in_text(facet: Facet, text: str) -> bool:
    if not text: return False
    ft = facet.facet_type
    tpl = facet.template or {}
    if ft == FacetType.ENTITY:
        return contains_exact_phrase(text, str(tpl.get("mention", "")))
    if ft == FacetType.NUMERIC:
        val = str(tpl.get("value", ""))
        return contains_value(text, val)
    if ft == FacetType.RELATION:
        return contains_exact_phrase(text, str(tpl.get("subject", ""))) and \
               contains_exact_phrase(text, str(tpl.get("object", "")))
    return False

@dataclass
class QAExample:
    qid: str
    question: str
    context: List[Tuple[str, List[str]]]
    supporting: Dict[str, Set[int]]

class DatasetAdapterHotpot:
    def load(self, path: str, limit: int = None):
        with open(path) as f: data = json.load(f)
        return data[:limit] if limit else data

    def to_example(self, raw):
        qid = raw.get("_id") or raw.get("id")
        q = raw.get("question")
        ctx = raw.get("context", [])
        sup = raw.get("supporting_facts", [])
        if not qid or not q or not ctx: return None
        
        parsed_ctx = []
        for item in ctx:
            if len(item) == 2: parsed_ctx.append((item[0], item[1]))
            
        sup_dict = defaultdict(set)
        for t, i in sup: sup_dict[t].add(i)
        
        return QAExample(qid, q, parsed_ctx, dict(sup_dict))

def extract_calibration_samples(raw_data, adapter, miner, scorer):
    samples = []
    
    for raw in tqdm(raw_data, desc="Extracting"):
        ex = adapter.to_example(raw)
        if not ex: continue
        
        sup_list = [(t, i) for t, idxs in ex.supporting.items() for i in idxs]
        facets = miner.extract_facets(ex.question, sup_list)
        
        for i, (title, sentences) in enumerate(ex.context):
            passage_text = " ".join(sentences)
            passage_id = f"context_{i}"
            
            has_sup = bool(ex.supporting.get(title))
            sup_idxs = sorted(i for i in ex.supporting.get(title, set()) if i < len(sentences))
            sup_text = " ".join(sentences[j] for j in sup_idxs) if sup_idxs else ""

            passage = Passage(pid=passage_id, text=passage_text, cost=0)

            for facet in facets:
                hyp = facet.to_hypothesis(passage_text)
                if "?" in hyp or len(hyp) < 10: continue

                # Capture lexical_match status
                lexical_match = None
                if facet.facet_type == FacetType.ENTITY:
                    lexical_match = contains_exact_phrase(passage_text, facet.template.get("mention", ""))
                elif facet.facet_type == FacetType.NUMERIC:
                    lexical_match = contains_value(passage_text, facet.template.get("value", ""))
                elif facet.facet_type == FacetType.BRIDGE_HOP1:
                    e1 = facet.template.get("entity1", "")
                    be = facet.template.get("bridge_entity", "")
                    lexical_match = contains_exact_phrase(passage_text, e1) and contains_exact_phrase(passage_text, be)

                details = scorer.score_with_details(passage, facet)
                
                # --- FIX: Support Label Logic ---
                if facet.facet_type == FacetType.BRIDGE_HOP1:
                    # Option A: Trust Title Support. 
                    # We defer stricter checking to the Sigma conversion step.
                    label = 1 if has_sup else 0
                else:
                    # Default: Title Support AND content verification in sup_text
                    label = 1 if (has_sup and facet_satisfied_in_text(facet, sup_text)) else 0

                samples.append({
                    "id": f"{ex.qid}_{passage_id}_{facet.facet_id}",
                    "query_id": ex.qid,
                    "hypothesis": hyp,
                    "score": details.final_score,
                    "probs": {
                        "entail": details.entailment_score,
                        "neutral": details.neutral_score,
                        "contra": details.contradiction_score,
                    },
                    "label": label,
                    "metadata": {
                        "facet_type": facet.facet_type.value,
                        "facet_template": facet.template,
                        "passage_preview": passage_text[:200],
                        "passage_full": passage_text, # CRITICAL for Sigma
                        "lexical_match": lexical_match,
                        "supporting_text": sup_text,
                        "question": ex.question,
                        "title": title
                    }
                })
    return samples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_path", default="calibration_data.jsonl")
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    config = TridentConfig()
    miner = FacetMiner(config)
    scorer = NLIScorer(config.nli, device=args.device)
    adapter = DatasetAdapterHotpot()
    
    raw = adapter.load(args.data_path, args.num_samples)
    samples = extract_calibration_samples(raw, adapter, miner, scorer)
    
    with open(args.output_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
            
    print(f"Extracted {len(samples)} samples to {args.output_path}")

if __name__ == "__main__":
    main()