#!/usr/bin/env python3
"""
extract_calibration_data.py
UPDATES:
- Added DatasetAdapter2Wiki (Standard 2WikiMultihop support).
- Added DatasetAdapterMusique (JSONL support).
- Switch logic in main() handles hotpot, musique, and 2wiki.
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
from trident.nli_scorer import NLIScorer, _check_lexical_gate
from trident.candidates import Passage
from trident.config import TridentConfig
import unicodedata

_WS_RE = re.compile(r"\s+")

def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = s.replace("'", "'").replace("'", "'").replace(""", '"').replace(""", '"')
    s = s.replace(",", "")
    s = _WS_RE.sub(" ", s).strip().lower()
    return s

@dataclass
class QAExample:
    qid: str
    question: str
    context: List[Tuple[str, List[str]]]
    supporting: Dict[str, Set[int]]

# --- ADAPTERS ---

class DatasetAdapterHotpot:
    def load(self, path: str, limit: int = None):
        with open(path) as f: 
            data = json.load(f)
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

class DatasetAdapter2Wiki:
    """
    Adapter for 2WikiMultihop.
    Format is nearly identical to HotpotQA (JSON list), but usually uses '_id'.
    """
    def load(self, path: str, limit: int = None):
        with open(path) as f: 
            data = json.load(f)
        return data[:limit] if limit else data

    def to_example(self, raw):
        # 2Wiki uses '_id' typically
        qid = raw.get("_id") or raw.get("id")
        q = raw.get("question")
        ctx = raw.get("context", [])
        sup = raw.get("supporting_facts", [])
        
        if not qid or not q or not ctx: return None
        
        # Parse Context: [[Title, [Sentences]], ...]
        parsed_ctx = []
        for item in ctx:
            # item is [title, sentences]
            if len(item) >= 2: 
                parsed_ctx.append((item[0], item[1]))
            
        # Parse Support: [[Title, sent_index], ...]
        sup_dict = defaultdict(set)
        for item in sup:
            if len(item) >= 2:
                t, i = item[0], item[1]
                sup_dict[t].add(i)
        
        return QAExample(qid, q, parsed_ctx, dict(sup_dict))

class DatasetAdapterMusique:
    """Adapter for MuSiQue (JSONL format)."""
    def load(self, path: str, limit: int = None):
        data = []
        with open(path) as f:
            for i, line in enumerate(f):
                if limit and i >= limit: break
                if line.strip():
                    data.append(json.loads(line))
        return data

    def to_example(self, raw):
        qid = raw.get("id")
        q = raw.get("question")
        if not qid or not q: return None
        
        parsed_ctx = []
        sup_dict = defaultdict(set)
        
        # MuSiQue structure: [{"title": "...", "paragraph_text": "...", "is_supporting": bool}]
        for p in raw.get("paragraphs", []):
            title = p.get("title", "")
            text = p.get("paragraph_text", "")
            if not title or not text: continue
            
            # Treat entire paragraph as one block (index 0)
            parsed_ctx.append((title, [text]))
            
            if p.get("is_supporting", False):
                sup_dict[title].add(0) 
                
        return QAExample(qid, q, parsed_ctx, dict(sup_dict))

# --- EXTRACTION LOGIC ---

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

                lexical_match = _check_lexical_gate(facet, passage_text)
                details = scorer.score_with_details(passage, facet)
                
                support_label = 1 if has_sup else 0

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
                    "label": support_label,         
                    "support_label": support_label, 
                    "metadata": {
                        "facet_type": facet.facet_type.value,
                        "facet_template": facet.template,
                        "passage_preview": passage_text[:200],
                        "passage_full": passage_text,
                        "lexical_match": lexical_match,
                        "supporting_text": sup_text,
                        "question": ex.question,
                        "title": title
                    }
                })
    return samples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="hotpot, musique, or 2wiki")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_path", default="calibration_data.jsonl")
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    config = TridentConfig()
    config.dataset = args.dataset
    
    miner = FacetMiner(config)
    scorer = NLIScorer(config.nli, device=args.device)
    
    # Adapter Selection
    ds = args.dataset.lower()
    if "hotpot" in ds:
        adapter = DatasetAdapterHotpot()
    elif "musique" in ds:
        adapter = DatasetAdapterMusique()
    elif "2wiki" in ds:
        adapter = DatasetAdapter2Wiki()
    else:
        print(f"Warning: Unknown dataset '{args.dataset}', assuming Hotpot format.")
        adapter = DatasetAdapterHotpot()
    
    print(f"Loading {ds} from {args.data_path}...")
    raw = adapter.load(args.data_path, args.num_samples)
    samples = extract_calibration_samples(raw, adapter, miner, scorer)
    
    with open(args.output_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
            
    print(f"Extracted {len(samples)} samples to {args.output_path}")

if __name__ == "__main__":
    main()