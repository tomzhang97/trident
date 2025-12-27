import json
import collections
import sys

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "calibration_hotpot_sigma.jsonl"
    c = collections.Counter()
    
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            ft = r["metadata"].get("facet_type", "UNKNOWN")
            lab = int(r["label"])
            c[(ft, lab)] += 1
            
    print("\nClass Counts per Type:")
    types = sorted(list(set(k[0] for k in c.keys())))
    for t in types:
        pos = c[(t, 1)]
        neg = c[(t, 0)]
        print(f"  {t:<15} Pos: {pos:>4}  Neg: {neg:>5}  Rate: {pos/(pos+neg)*100:.1f}%")
        
    num_pos = c[("NUMERIC", 1)]
    if num_pos < 20:
        print("\n⚠️  NUMERIC still has < 20 positives. Consider merging buckets.")
    else:
        print(f"\n✅ NUMERIC has {num_pos} positives (Healthy).")

if __name__ == "__main__":
    main()