#!/usr/bin/env python3
"""
Train calibration model.
UPDATES:
- Stop skipping lex_failures (calibration safety).
- Added overall positive rate sanity check.
- Canonical filtering.
"""

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

# Add trident to path
sys.path.insert(0, str(Path(__file__).parent))
from trident.calibration import ReliabilityCalibrator, SelectionConditionalCalibrator

def canonical_ft(ft: str) -> str:
    return "BRIDGE_HOP" if "BRIDGE_HOP" in ft else ft

def load_data(path: str, requested_types: list = None):
    data = []
    lex_fail_count = 0
    missing_score = 0
    
    # Pre-canonicalize requests
    allowed_types = None
    if requested_types:
        allowed_types = set()
        for t in requested_types:
            allowed_types.add(t)
            allowed_types.add(canonical_ft(t))
    
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            meta = r.get("metadata", {})
            raw_ft = meta.get("facet_type", "UNKNOWN")
            
            ft = canonical_ft(raw_ft)
            
            if ft == "UNKNOWN": continue
            if allowed_types and ft not in allowed_types: continue

            # METRIC ONLY: Do not skip lex failures
            lex_match = meta.get("lexical_match")
            if lex_match is False:
                lex_fail_count += 1
            
            # SCORE SAFETY
            score = r.get("score")
            if score is None:
                if "probs" in r and "entail" in r["probs"]:
                    score = float(r["probs"]["entail"])
                else:
                    missing_score += 1
                    continue
            else:
                score = float(score)

            data.append({
                "score": score,
                "label": int(r["label"]),
                "facet_type": ft,
                "text_length": int(meta.get("text_length", 0))
            })
            
    print(f"Loaded {len(data)} samples.")
    print(f"Info: Included {lex_fail_count} samples where lexical_match=False.")
    
    # SANITY CHECK
    if data:
        n_pos = sum(d["label"] for d in data)
        print(f"Sanity Check: Overall positive rate: {n_pos/len(data)*100:.2f}% ({n_pos}/{len(data)})")
    
    if missing_score > 0:
        print(f"Skipped {missing_score} items due to missing scores.")
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_path", default="calibrator.pkl")
    parser.add_argument("--use_mondrian", action="store_true")
    args = parser.parse_args()

    data = load_data(args.data_path)
    
    # n_min=25 is safer for sparse types
    cal = SelectionConditionalCalibrator(n_min=25, use_mondrian=args.use_mondrian)
    
    counts = defaultdict(lambda: [0, 0])
    for d in data:
        cal.add_calibration_sample(d["score"], bool(d["label"]), d["facet_type"], d["text_length"])
        counts[d["facet_type"]][0] += 1
        counts[d["facet_type"]][1] += d["label"]

    print("\nTraining Conformal Calibrator...")
    for ft, (n, pos) in counts.items():
        print(f"  {ft:<15}: {n} samples ({pos/n*100:.1f}% pos)")

    cal.finalize()
    
    wrapper = ReliabilityCalibrator(use_mondrian=args.use_mondrian)
    wrapper.set_conformal_calibrator(cal)
    
    out_path = Path(args.output_path)
    if out_path.suffix != '.pkl': out_path = out_path.with_suffix('.pkl')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, 'wb') as f:
        pickle.dump(wrapper, f)
    
    wrapper.save(out_path.with_suffix(".json").as_posix())
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()