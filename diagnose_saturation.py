import json
import numpy as np
import sys
import pickle  # <--- Added missing import
from collections import Counter
from pathlib import Path

# Adjust path if needed to find trident
sys.path.insert(0, str(Path.cwd()))

try:
    from trident.calibration import ReliabilityCalibrator
except ImportError:
    print("Could not import ReliabilityCalibrator. Ensure you are in the trident root.")
    sys.exit(1)

def main():
    # 1. Load the Calibrator
    cal_path = "calibrator.pkl"
    try:
        with open(cal_path, "rb") as f:
            cal = pickle.load(f)
        print(f"Loaded calibrator from {cal_path}")
    except Exception as e:
        print(f"Pickle load failed: {e}")
        try:
            cal = ReliabilityCalibrator.load("calibrator.json")
            print("Loaded calibrator from JSON.")
        except:
            print("Could not load calibrator. Exiting.")
            sys.exit(1)

    p1_count = 0
    total = 0
    by_ft = Counter()
    by_ft_p1 = Counter()
    examples = []

    print("\nAnalyzing p-value saturation...")
    with open("calibration_hotpot_sigma.jsonl") as f:
        for line in f:
            r = json.loads(line)
            m = r.get("metadata", {})
            
            # Skip failures of lexical gate (untreated / score 0)
            if m.get("lexical_match") is False:
                continue
                
            ft = m.get("facet_type", "UNK")
            
            # Get score (prefer probs.entail)
            if "probs" in r and "entail" in r["probs"]:
                score = float(r["probs"]["entail"])
            else:
                score = float(r.get("score", 0.0))
                
            length = int(m.get("text_length", 100))
            
            # Compute P-value
            try:
                p = float(cal.to_pvalue(score, ft, length))
            except Exception:
                p = 1.0 # Default fallback
                
            total += 1
            by_ft[ft] += 1
            
            # Check for saturation (p ~ 1.0)
            if p >= 0.9999:
                p1_count += 1
                by_ft_p1[ft] += 1
                # Save examples of "High Score but P=1.0" (Scary saturation)
                if len(examples) < 5 and score > 0.8:
                    examples.append((ft, score, length, m.get("passage_preview","")[:80]))

    print(f"\nSaturation Report:")
    print(f"  Total analyzed: {total}")
    print(f"  p >= 0.9999:    {p1_count} ({p1_count/total*100:.1f}%)")
    
    print("\nBreakdown by Type (Saturated / Total):")
    for ft, count in by_ft.items():
        sat = by_ft_p1[ft]
        print(f"  {ft:<15}: {sat}/{count} ({sat/count*100:.1f}%)")

    if examples:
        print("\n⚠️  High-Score Items that are Saturated (p=1.0):")
        print("(If this list is empty, your saturation is just low-scoring negatives, which is good)")
        for ft, s, l, txt in examples:
            print(f"  [{ft}] Score={s:.3f} Len={l} | {txt}...")
    else:
        print("\n✅ No high-score items are saturated. The p=1.0 cases are correctly low-scoring.")

if __name__ == "__main__":
    main()