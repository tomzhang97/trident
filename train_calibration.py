#!/usr/bin/env python3
"""
Train calibration model (Mondrian Conformal Prediction).
FIXES:
- Uses SelectionConditionalCalibrator for RANK-BASED p-values (Fixes saturation).
- Filters lexical_match=False for BRIDGE as well (Gate failure = Untestable).
- Skips degenerate buckets.
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

def load_data(path: str, facet_types: list = None):
    data = []
    skipped_lex = 0
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            meta = r.get("metadata", {})
            ft = meta.get("facet_type", "UNKNOWN")

            if facet_types and ft not in facet_types:
                continue

            # --- SMART LEXICAL FILTER ---
            # Filter lexical_match=False for ENTITY, NUMERIC, and BRIDGE.
            # If the gate fails (lexical_match=False), the item is "untestable"
            # and should not be part of the negative pool.
            lex_match = meta.get("lexical_match")
            if ft in ["ENTITY", "NUMERIC", "BRIDGE_HOP1"]:
                if lex_match is False:
                    skipped_lex += 1
                    continue
            
            if "probs" in r and "entail" in r["probs"]:
                score = float(r["probs"]["entail"])
            else:
                score = float(r.get("score", 0.0))
                
            data.append({
                "score": float(score),
                "label": int(r["label"]),
                "facet_type": str(ft),
                "text_length": int(meta.get("text_length", 0))
            })
    print(f"Skipped {skipped_lex} items due to lexical gate failure.")
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_path", default="calibrator.pkl")
    parser.add_argument("--use_mondrian", action="store_true")
    parser.add_argument("--facet_types", type=str, help="Comma-sep list of types")
    args = parser.parse_args()

    fts = args.facet_types.split(",") if args.facet_types else None
    
    print(f"Loading data from {args.data_path}...")
    data = load_data(args.data_path, fts)
    print(f"Loaded {len(data)} calibration samples.")
    
    if len(data) == 0:
        print("Error: No data loaded.")
        sys.exit(1)

    # Use Rank-Based Conformal Calibrator
    # n_min=50 enforces merging small bins to ensure robust p-values
    conformal_cal = SelectionConditionalCalibrator(n_min=50, use_mondrian=args.use_mondrian)
    
    # Accumulate samples
    counts = defaultdict(lambda: [0, 0]) # [total, pos]
    
    for d in data:
        ft = d["facet_type"]
        conformal_cal.add_calibration_sample(
            score=d["score"],
            is_sufficient=bool(d["label"]),
            facet_type=ft,
            text_length=d["text_length"]
        )
        counts[ft][0] += 1
        counts[ft][1] += d["label"]

    print("\nTraining Conformal Calibrator...")
    for ft, (n, npos) in counts.items():
        if npos == 0 or npos == n:
            print(f"  {ft:<15}: {n} samples ({npos} pos) -> WARNING: degenerate")
        else:
            print(f"  {ft:<15}: {n} samples ({npos/n*100:.1f}% positive)")

    # Finalize (sorts negative pools, merges small bins)
    conformal_cal.finalize()

    # Wrap in ReliabilityCalibrator for compatibility
    cal = ReliabilityCalibrator(use_mondrian=args.use_mondrian)
    cal.set_conformal_calibrator(conformal_cal)
    
    # Save
    out_path = Path(args.output_path)
    if out_path.suffix != '.pkl': out_path = out_path.with_suffix('.pkl')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, 'wb') as f:
        pickle.dump(cal, f)
    print(f"\nSaved Calibrator (Pickle) to: {out_path}")

    # JSON check
    json_path = out_path.with_suffix('.json')
    try:
        # Save the INNER conformal calibrator to JSON for inspection
        conformal_cal.save(str(json_path))
        print(f"Saved Calibrator (JSON)   to: {json_path}")
    except Exception as e:
        print(f"⚠️  JSON Save Warning: {e}")

if __name__ == "__main__":
    main()