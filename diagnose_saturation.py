import json
import numpy as np
import sys
from collections import Counter
# Adjust path if needed
sys.path.insert(0, ".") 
try:
    from trident.calibration import ReliabilityCalibrator
except ImportError:
    print("Could not import ReliabilityCalibrator. Ensure you are in the trident root.")
    sys.exit(1)

def main():
    try:
        with open("calibrator.pkl", "rb") as f:
            cal = pickle.load(f)
        print("Loaded calibrator from pickle.")
    except Exception as e:
        print(f"Pickle load failed: {e}")
        # Fallback if you saved as JSON
        try:
            cal = ReliabilityCalibrator.load("calibrator.json")
            print("Loaded calibrator from JSON.")
        except:
            print("Could not load calibrator.")
            sys.exit(1)

    p1_count = 0
    total = 0
    by_ft = Counter()
    by_ft_p1 = Counter()
    examples = []

    print("Analyzing p-value saturation...")
    with open("calibration_hotpot_sigma.jsonl") as f:
        for line in f:
            r = json.loads(line)
            m = r.get("metadata", {})
            
            # Skip failures of lexical gate (score is forced to 0 anyway)
            if m.get("lexical_match") is False:
                continue
                
            ft = m.get("facet_type", "UNK")
            # Get score
            score = float(r.get("probs", {}).get("entail", r.get("score", 0.0)))
            length = int(m.get("text_length", 100))
            
            # Compute P-value
            try:
                p = float(cal.to_pvalue(score, ft, length))
            except:
                p = 1.0 # Default fallback
                
            total += 1
            by_ft[ft] += 1
            
            if p >= 0.9999:
                p1_count += 1
                by_ft_p1[ft] += 1
                if len(examples) < 5 and score > 0.5:
                    examples.append((ft, score, length, m.get("passage_preview","")[:100]))

    print(f"\nSaturation Report:")
    print(f"  Total analyzed: {total}")
    print(f"  p >= 0.9999:    {p1_count} ({p1_count/total*100:.1f}%)")
    
    print("\nBreakdown by Type (Saturated / Total):")
    for ft, count in by_ft.items():
        sat = by_ft_p1[ft]
        print(f"  {ft:<15}: {sat}/{count} ({sat/count*100:.1f}%)")

    print("\nSample Saturated Items (High Score but p=1.0?):")
    for ft, s, l, txt in examples:
        print(f"  [{ft}] Score={s:.3f} Len={l} | {txt}...")

if __name__ == "__main__":
    main()