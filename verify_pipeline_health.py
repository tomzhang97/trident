#!/usr/bin/env python3
"""
verify_pipeline_health.py
Diagnose retention rates from Gold Support -> Sigma Label.
"""

import argparse
import collections
import json
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to sigma-labeled jsonl file")
    args = parser.parse_args()

    print(f"Checking Pipeline Health in {args.path}...")
    
    total_counts = collections.Counter()
    gold_counts = collections.Counter()
    sigma_counts = collections.Counter()
    
    try:
        with open(args.path) as f:
            for line in f:
                r = json.loads(line)
                # Use canonical type if available, else raw
                meta = r.get("metadata", {})
                ft = meta.get("facet_type_canonical") or meta.get("facet_type", "UNKNOWN")
                
                # Check labels
                support_lab = int(r.get("support_label", 0))
                sigma_lab = int(r.get("label", 0))
                
                total_counts[ft] += 1
                if support_lab == 1:
                    gold_counts[ft] += 1
                if sigma_lab == 1:
                    sigma_counts[ft] += 1
                    
    except FileNotFoundError:
        print(f"Error: File {args.path} not found.")
        sys.exit(1)
        
    print("\n{:<15} {:<8} {:<8} {:<8} {:<10}".format("Facet Type", "Total", "Gold", "Sigma", "Retention"))
    print("-" * 55)
    
    # Sort by Facet Type
    for ft in sorted(total_counts.keys()):
        tot = total_counts[ft]
        gold = gold_counts[ft]
        sig = sigma_counts[ft]
        
        # Retention: Sigma / Gold (percentage of gold-supported items we kept)
        ret = (sig / gold * 100.0) if gold > 0 else 0.0
        
        print(f"{ft:<15} {tot:<8} {gold:<8} {sig:<8} {ret:>5.1f}%")
        
    print("-" * 55)
    print("Diagnostics Guide:")
    print("1. If Gold is 0 -> Extraction not finding supporting facts (FacetMiner coverage issue).")
    print("2. If Sigma is 0 but Gold > 0 -> Strict Sigma checks killing everything (Check heuristics).")
    print("3. If Total is low -> FacetMiner not generating this type of facet.")

if __name__ == "__main__":
    main()