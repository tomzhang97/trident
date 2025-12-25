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

    # Check if we're using entailment labels or supporting-fact labels
    has_support_label = any("support_label" in s for s in samples[:10])
    if has_support_label:
        print("\n✅ Detected entailment-based labels (support_label field exists)")
        print("   Using 'label' for analysis (should match lexical_match)")
        label_semantics = "entailment"
    else:
        print("\n⚠️  Using Hotpot supporting-fact labels")
        print("   'High mismatches' are EXPECTED - they mean:")
        print("   - lex=True + label=0: phrase exists but not a supporting fact")
        print("   - lex=False + label=1: BUG in facet_satisfied_in_text()")
        print("\n   💡 Run convert_to_entail_labels.py to fix this for calibration!")
        label_semantics = "support"

    mismatches_false_pos = []  # lex=False but label=1 (ALWAYS a bug)
    mismatches_true_neg = []   # lex=True but label=0 (expected for support labels)

    for s in samples:
        ft = s["metadata"].get("facet_type", "")
        if ft not in lexical_types:
            continue

        label = int(s["label"])
        lex = s["metadata"].get("lexical_match")

        if lex is False and label == 1:
            mismatches_false_pos.append(s)
        elif lex is True and label == 0:
            mismatches_true_neg.append(s)

    print(f"\n📊 Found:")
    print(f"   lex=False + label=1: {len(mismatches_false_pos)} (BUGS - phrase doesn't exist!)")
    print(f"   lex=True + label=0:  {len(mismatches_true_neg)} ", end="")
    if label_semantics == "support":
        print("(EXPECTED - phrase exists but not supporting)")
    else:
        print("(UNEXPECTED - check labeling logic)")

    # Show examples of BUGS (lex=False + label=1)
    if mismatches_false_pos:
        print(f"\n⚠️  CRITICAL BUGS (lex=False but label=1):")
        for i, s in enumerate(mismatches_false_pos[:3]):
            print(f"\n#{i+1} Type: {s['metadata'].get('facet_type')}")
            print(f"  Hypothesis: {s.get('hypothesis')}")
            print(f"  Label: {s['label']}, Lexical match: {s['metadata'].get('lexical_match')}")
            print(f"  Supporting text: {s['metadata'].get('supporting_text', '')[:100]}...")
            print(f"  → BUG: facet_satisfied_in_text() returned True but phrase doesn't exist!")

    # Show examples of lex=True + label=0 (interpretation depends on label semantics)
    if mismatches_true_neg and label_semantics == "entailment":
        print(f"\n⚠️  Unexpected (lex=True but label=0) with entailment labels:")
        for i, s in enumerate(mismatches_true_neg[:3]):
            print(f"\n#{i+1} Type: {s['metadata'].get('facet_type')}")
            print(f"  Hypothesis: {s.get('hypothesis')}")
            print(f"  Label: {s['label']}, Lexical match: {s['metadata'].get('lexical_match')}")
            print(f"  Passage: {s['metadata'].get('passage_preview', '')[:100]}...")

    return mismatches_false_pos, mismatches_true_neg


def analyze_high_confidence_negatives(samples, threshold=0.7):
    """Find high NLI score but label=0 cases."""
    print("\n" + "=" * 80)
    print(f"HIGH CONFIDENCE NEGATIVES (score > {threshold}, label=0)")
    print("=" * 80)

    # Detect label semantics
    has_support_label = any("support_label" in s for s in samples[:10])
    label_semantics = "entailment" if has_support_label else "support"

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
                lex = s['metadata'].get('lexical_match')
                if label_semantics == "entailment":
                    if lex is False:
                        print(f"    ✅ OK - lexical gate should force score=0 (check gate is working)")
                    else:
                        print(f"    ⚠️  BUG - lex=True but label=0 with entailment labels")
                else:  # support labels
                    if lex is False:
                        print(f"    ✅ OK - lexical gate forces low score")
                    else:
                        print(f"    ℹ️  Expected - phrase exists but not supporting fact")
            else:
                if label_semantics == "entailment":
                    print(f"    ⚠️  Consider excluding {ft} from calibration (non-lexical)")
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

    # Detect label semantics for summary
    has_support_label = any("support_label" in samples for samples in [samples] if samples)
    label_semantics = "entailment" if has_support_label else "support"

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if label_semantics == "support":
        print("""
⚠️  CRITICAL: You're using Hotpot supporting-fact labels for calibration!

Problem: Labels mean "is this a supporting fact?" but hypotheses ask "does passage entail facet?"
Result: Many lex=True + label=0 cases (phrase exists but not supporting)

✅ SOLUTION: Convert to entailment labels

Run this command:
  python convert_to_entail_labels.py calibration_hotpot_fixed.jsonl calibration_hotpot_entail.jsonl

This will:
  - Keep only ENTITY/NUMERIC facets (lexical, clean)
  - Set label = int(lexical_match) for true entailment
  - Preserve original as support_label
  - Fix the label semantics mismatch

Then re-run diagnostics on the new file.

Next steps:
1. Fix "lex=False but label=1" bugs (if any) in facet_satisfied_in_text()
2. Convert to entailment labels (command above)
3. Re-run: python diagnose_calibration_detailed.py calibration_hotpot_entail.jsonl
4. Train calibrator on entailment labels
""")
    else:
        print("""
✅ Using entailment-based labels (good for calibration!)

Next steps:
1. Fix any "lex=False but label=1" bugs (labeling errors)
2. Check "lex=True but label=0" cases (should be rare/zero with entailment labels)
3. Ensure facet types have 20+ positives for Mondrian calibration
4. Train calibrator:

   # ENTITY only (safest):
   python train_calibration.py \\
     --data_path <this_file> \\
     --output_path calibrator_entity.json \\
     --use_mondrian \\
     --facet_types ENTITY

   # ENTITY+NUMERIC (if NUMERIC has 20+ positives):
   python train_calibration.py \\
     --data_path <this_file> \\
     --output_path calibrator.json \\
     --use_mondrian
""")


if __name__ == "__main__":
    main()
