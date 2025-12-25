#!/usr/bin/env python3
"""
Detailed diagnostic script for calibration data quality.

Checks for:
1. Label-hypothesis mismatches (lexical gate failures)
2. High-confidence negatives (potential labeling issues)
3. Per-facet-type quality metrics
"""

import json
import sys
from collections import defaultdict
from pathlib import Path


def load_samples(path: str):
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples


def analyze_lexical_mismatches(samples):
    """Check for cases where lexical_match disagrees with label."""
    print("\n" + "=" * 80)
    print("LEXICAL MATCH vs LABEL ANALYSIS")
    print("=" * 80)

    lexical_types = {"ENTITY", "NUMERIC"}

    mismatches = []
    for s in samples:
        ft = s["metadata"].get("facet_type", "")
        if ft not in lexical_types:
            continue

        label = int(s["label"])
        lex = s["metadata"].get("lexical_match")

        # For lexical facets, lexical_match should align with label
        # If lex=False, label should be 0 (phrase doesn't exist)
        # If lex=True, label should be 1 (phrase exists in supporting text)

        if lex is False and label == 1:
            mismatches.append({
                "type": "lex=False but label=1",
                "sample": s,
                "severity": "HIGH"
            })
        elif lex is True and label == 0:
            mismatches.append({
                "type": "lex=True but label=0",
                "sample": s,
                "severity": "MEDIUM"  # Could be valid if passage isn't supporting
            })

    print(f"\nFound {len(mismatches)} potential mismatches in lexical facets")

    # Show a few examples
    for i, m in enumerate(mismatches[:5]):
        s = m["sample"]
        print(f"\n#{i+1} [{m['severity']}] {m['type']}")
        print(f"  Type: {s['metadata'].get('facet_type')}")
        print(f"  Hypothesis: {s.get('hypothesis')}")
        print(f"  Label: {s['label']}, Lexical match: {s['metadata'].get('lexical_match')}")
        print(f"  Supporting text: {s['metadata'].get('supporting_text', '')[:100]}")
        print(f"  Passage: {s['metadata'].get('passage_preview', '')[:100]}")

    return mismatches


def analyze_high_confidence_negatives(samples, threshold=0.7):
    """Find high NLI score but label=0 cases."""
    print("\n" + "=" * 80)
    print(f"HIGH CONFIDENCE NEGATIVES (score > {threshold}, label=0)")
    print("=" * 80)

    by_type = defaultdict(list)

    for s in samples:
        if s["label"] == 0 and s["score"] > threshold:
            ft = s["metadata"].get("facet_type", "UNKNOWN")
            by_type[ft].append(s)

    for ft in sorted(by_type.keys()):
        samples_ft = by_type[ft]
        print(f"\n{ft}: {len(samples_ft)} high-conf negatives")

        # Show top 3
        samples_ft.sort(key=lambda x: x["score"], reverse=True)
        for i, s in enumerate(samples_ft[:3]):
            print(f"\n  #{i+1} score={s['score']:.4f}")
            print(f"    Hypothesis: {s.get('hypothesis')}")
            print(f"    Lexical match: {s['metadata'].get('lexical_match')}")
            print(f"    Passage: {s['metadata'].get('passage_preview', '')[:120]}")

            if ft in {"ENTITY", "NUMERIC"}:
                print(f"    ⚠️  POTENTIAL BUG - lexical facet should have low score if label=0")
            else:
                print(f"    ℹ️  Expected - {ft} labels are Hotpot supporting facts, not entailment")

    return by_type


def analyze_per_type_quality(samples):
    """Per-facet-type calibration quality metrics."""
    print("\n" + "=" * 80)
    print("PER-FACET-TYPE QUALITY METRICS")
    print("=" * 80)

    stats = defaultdict(lambda: {
        "total": 0,
        "pos": 0,
        "neg": 0,
        "avg_score_pos": [],
        "avg_score_neg": [],
        "lexical_gate_triggered": 0
    })

    for s in samples:
        ft = s["metadata"].get("facet_type", "UNKNOWN")
        stats[ft]["total"] += 1

        if s["label"] == 1:
            stats[ft]["pos"] += 1
            stats[ft]["avg_score_pos"].append(s["score"])
        else:
            stats[ft]["neg"] += 1
            stats[ft]["avg_score_neg"].append(s["score"])

        if s["metadata"].get("lexical_gate_applied", False):
            stats[ft]["lexical_gate_triggered"] += 1

    print(f"\n{'Type':<15} {'Total':>6} {'Pos':>5} {'Neg':>5} {'Pos%':>6} {'Score+':>7} {'Score-':>7} {'Gate':>5}")
    print("-" * 80)

    for ft in sorted(stats.keys()):
        st = stats[ft]
        pos_pct = st["pos"] / st["total"] * 100 if st["total"] > 0 else 0
        avg_pos = sum(st["avg_score_pos"]) / len(st["avg_score_pos"]) if st["avg_score_pos"] else 0
        avg_neg = sum(st["avg_score_neg"]) / len(st["avg_score_neg"]) if st["avg_score_neg"] else 0

        print(f"{ft:<15} {st['total']:>6} {st['pos']:>5} {st['neg']:>5} {pos_pct:>5.1f}% "
              f"{avg_pos:>7.3f} {avg_neg:>7.3f} {st['lexical_gate_triggered']:>5}")

    print("\nRecommendations:")
    for ft in sorted(stats.keys()):
        st = stats[ft]
        if st["pos"] < 20:
            print(f"  ⚠️  {ft}: Only {st['pos']} positives - may be too sparse for Mondrian calibration")

        if ft in {"ENTITY", "NUMERIC"}:
            avg_neg = sum(st["avg_score_neg"]) / len(st["avg_score_neg"]) if st["avg_score_neg"] else 0
            if avg_neg > 0.5:
                print(f"  ⚠️  {ft}: High avg negative score ({avg_neg:.3f}) - check labeling logic")
            else:
                print(f"  ✅ {ft}: Safe for calibration (lexical, avg neg score={avg_neg:.3f})")
        else:
            print(f"  ⚠️  {ft}: Use cautiously - labels are supporting facts, not entailment")

    return stats


def inspect_one_bad_numeric(samples):
    """Deep dive into one problematic NUMERIC sample."""
    print("\n" + "=" * 80)
    print("DEEP DIVE: One problematic NUMERIC sample")
    print("=" * 80)

    for s in samples:
        if (s["metadata"].get("facet_type") == "NUMERIC" and
            s["label"] == 0 and
            s["score"] > 0.8):

            print("\nFull metadata:")
            print(json.dumps(s["metadata"], indent=2))
            print(f"\nHypothesis: {s.get('hypothesis')}")
            print(f"Score: {s['score']:.4f}")
            print(f"Label: {s['label']}")
            print(f"Lexical match: {s['metadata'].get('lexical_match')}")
            print(f"\nProbs: entail={s['probs']['entail']:.4f}, "
                  f"neutral={s['probs']['neutral']:.4f}, "
                  f"contra={s['probs']['contra']:.4f}")
            return

    print("\nNo problematic NUMERIC samples found (score > 0.8, label=0)")


def main():
    if len(sys.argv) < 2:
        print("Usage: python diagnose_calibration_detailed.py <calibration_data.jsonl>")
        sys.exit(1)

    path = sys.argv[1]
    if not Path(path).exists():
        print(f"Error: {path} not found")
        sys.exit(1)

    print(f"Loading samples from {path}...")
    samples = load_samples(path)
    print(f"Loaded {len(samples)} samples")

    # Run analyses
    analyze_lexical_mismatches(samples)
    analyze_high_confidence_negatives(samples)
    analyze_per_type_quality(samples)
    inspect_one_bad_numeric(samples)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Next steps:
1. If you see many "lex=False but label=1" → labeling bug, check facet_satisfied_in_text()
2. If you see many "lex=True but label=0" → passage contains phrase but isn't supporting fact (expected)
3. For ENTITY/NUMERIC with high avg negative scores → check lexical gate is working
4. For RELATION/TEMPORAL → high-conf negatives are expected with Hotpot labels
5. Use only facet types with 20+ positives and clean separation for calibration
""")


if __name__ == "__main__":
    main()
