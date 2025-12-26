import json
import numpy as np
import sys

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "calibration_hotpot_sigma.jsonl"
    print(f"Checking separability in {path}...")
    
    pos_scores, neg_scores = [], []
    
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r["metadata"]["facet_type"] != "ENTITY": continue
            
            s = r["score"]
            if r["label"] == 1:
                pos_scores.append(s)
            else:
                neg_scores.append(s)
                
    p = np.array(pos_scores)
    n = np.array(neg_scores)
    
    print(f"\nENTITY Stats:")
    print(f"  Pos: {len(p)}, Neg: {len(n)}")
    print(f"  Pos Mean: {p.mean():.3f} (std: {p.std():.3f})")
    print(f"  Neg Mean: {n.mean():.3f} (std: {n.std():.3f})")
    
    # Overlap check
    p5 = np.percentile(p, 5) if len(p) else 0
    n95 = np.percentile(n, 95) if len(n) else 0
    print(f"  Neg 95th: {n95:.3f}")
    print(f"  Pos 5th:  {p5:.3f}")
    
    if p5 > n95:
        print("✅ Excellent separability (Gap exists)")
    else:
        print("⚠️  Overlap detected (Expected with difficult negatives)")

if __name__ == "__main__":
    main()
