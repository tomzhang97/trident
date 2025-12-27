import json
import numpy as np
import sys

def main():
    print("Analyzing Discriminative Power (ENTITY)...")
    scores, y = [], []
    
    with open("calibration_hotpot_sigma.jsonl") as f:
        for line in f:
            r = json.loads(line)
            m = r.get("metadata", {})
            
            if m.get("facet_type") != "ENTITY": 
                continue
            if m.get("lexical_match") is False:
                continue
                
            s = float(r.get("probs", {}).get("entail", r.get("score", 0.0)))
            scores.append(s)
            y.append(int(r["label"]))

    scores = np.array(scores)
    y = np.array(y)
    
    pos_mask = (y == 1)
    neg_mask = (y == 0)
    
    print(f"Data: {len(scores)} samples ({pos_mask.sum()} Pos, {neg_mask.sum()} Neg)")
    
    print("\nThreshold Analysis:")
    print(f"  {'Thr':<5} {'TPR':<6} {'FPR':<6} {'Selected%':<10}")
    print("-" * 35)
    
    for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        pred = scores >= t
        
        tp = (pred & pos_mask).sum()
        fp = (pred & neg_mask).sum()
        
        tpr = tp / max(1, pos_mask.sum())
        fpr = fp / max(1, neg_mask.sum())
        sel = pred.mean()
        
        print(f"  {t:<5.1f} {tpr:<6.3f} {fpr:<6.3f} {sel:<10.1%}")

    # Percentiles
    print("\nScore Distributions:")
    if pos_mask.sum() > 0:
        p_scores = scores[pos_mask]
        print(f"  Pos: p50={np.median(p_scores):.3f}, p90={np.percentile(p_scores, 90):.3f}")
    if neg_mask.sum() > 0:
        n_scores = scores[neg_mask]
        print(f"  Neg: p50={np.median(n_scores):.3f}, p90={np.percentile(n_scores, 90):.3f}")

if __name__ == "__main__":
    main()