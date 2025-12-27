#!/usr/bin/env python3
"""
Detailed diagnostic script for Σ calibration data.
Correctly handles 'sigma' semantics (lex=True, label=0 is EXPECTED).
"""

import json
import sys
from collections import defaultdict

def analyze_lexical_mismatches(samples):
    print("\n=== LEXICAL MATCH vs LABEL ANALYSIS ===")
    
    # Check semantics
    label_semantics = samples[0].get("metadata", {}).get("label_semantics", "support")
    print(f"Detected label semantics: {label_semantics}")
    
    if label_semantics == "sigma":
        print("✅ SIGMA MODE: lex=True + label=0 is EXPECTED (Critical Negatives)")
    
    bugs = []      # lex=False, label=1
    crit_negs = [] # lex=True, label=0
    
    for s in samples:
        ft = s["metadata"].get("facet_type")
        if ft not in ["ENTITY", "NUMERIC"]: continue
        
        lex = s["metadata"].get("lexical_match")
        lab = s["label"]
        
        if lex is False and lab == 1:
            bugs.append(s)
        elif lex is True and lab == 0:
            crit_negs.append(s)
            
    print(f"\nResults:")
    print(f"  BUGS (lex=False, label=1): {len(bugs)} (Should be 0)")
    print(f"  CRITICAL NEGATIVES (lex=True, label=0): {len(crit_negs)} (Should be HIGH for robust training)")
    
    if label_semantics != "sigma" and len(crit_negs) > 100:
        print("⚠️  Warning: High mismatches but not in Sigma mode? Check labels.")

def main():
    path = sys.argv[1]
    samples = []
    with open(path) as f:
        for line in f: samples.append(json.loads(line))
        
    analyze_lexical_mismatches(samples)
    
    # Stats
    pos = sum(s["label"] for s in samples)
    print(f"\nTotal: {len(samples)}, Positive: {pos} ({pos/len(samples)*100:.1f}%)")

if __name__ == "__main__":
    main()
