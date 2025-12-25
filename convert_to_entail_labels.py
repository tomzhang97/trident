#!/usr/bin/env python3
"""
Convert Hotpot-based labels to entailment-based labels for calibration.

Problem: Current labels are "is this a Hotpot supporting fact?"
         But hypotheses ask "does this passage entail the facet?"

Solution: For ENTITY/NUMERIC facets, use lexical_match as the true label.

Usage:
  python convert_to_entail_labels.py calibration_hotpot_fixed.jsonl calibration_hotpot_entail.jsonl
"""

import json
import sys
from pathlib import Path


def convert_to_entail_labels(input_path: str, output_path: str, keep_types=None):
    """
    Convert supporting-fact labels to entailment labels.

    For ENTITY/NUMERIC facets:
      - Preserve original label as support_label
      - Set label = int(bool(lexical_match))

    This gives us true entailment labels for calibration.
    """
    if keep_types is None:
        keep_types = {"ENTITY", "NUMERIC"}  # Start with lexical facets only

    n_in = 0
    n_out = 0
    n_skipped_type = 0
    n_skipped_no_lex = 0

    stats = {
        "support_pos": 0,
        "support_neg": 0,
        "entail_pos": 0,
        "entail_neg": 0,
        "flipped_to_pos": 0,  # support=0, entail=1
        "flipped_to_neg": 0,  # support=1, entail=0
    }

    with open(input_path) as f, open(output_path, "w") as g:
        for line in f:
            n_in += 1
            r = json.loads(line)

            ft = r.get("metadata", {}).get("facet_type")
            if ft not in keep_types:
                n_skipped_type += 1
                continue

            lex = r.get("metadata", {}).get("lexical_match")
            if lex is None:
                n_skipped_no_lex += 1
                continue

            # Track original label
            orig_label = int(r["label"])
            if orig_label == 1:
                stats["support_pos"] += 1
            else:
                stats["support_neg"] += 1

            # Preserve original label as support_label
            r["support_label"] = orig_label

            # Set entailment label based on lexical match
            entail_label = int(bool(lex))
            r["label"] = entail_label

            if entail_label == 1:
                stats["entail_pos"] += 1
            else:
                stats["entail_neg"] += 1

            # Track flips
            if orig_label == 0 and entail_label == 1:
                stats["flipped_to_pos"] += 1
            elif orig_label == 1 and entail_label == 0:
                stats["flipped_to_neg"] += 1

            g.write(json.dumps(r, ensure_ascii=False) + "\n")
            n_out += 1

    return n_in, n_out, n_skipped_type, n_skipped_no_lex, stats


def main():
    if len(sys.argv) < 2:
        print("Usage: python convert_to_entail_labels.py <input.jsonl> [output.jsonl] [facet_types]")
        print("\nExamples:")
        print("  # ENTITY+NUMERIC (default):")
        print("  python convert_to_entail_labels.py cal.jsonl cal_entail.jsonl")
        print("\n  # ENTITY only:")
        print("  python convert_to_entail_labels.py cal.jsonl cal_entity.jsonl ENTITY")
        print("\n  # Multiple types:")
        print("  python convert_to_entail_labels.py cal.jsonl cal_custom.jsonl ENTITY,NUMERIC,RELATION")
        sys.exit(1)

    input_path = sys.argv[1]
    if not Path(input_path).exists():
        print(f"Error: {input_path} not found")
        sys.exit(1)

    # Default output path
    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        output_path = input_path.replace(".jsonl", "_entail.jsonl")

    # Parse facet types
    if len(sys.argv) >= 4:
        keep_types = set(sys.argv[3].split(","))
    else:
        keep_types = {"ENTITY", "NUMERIC"}

    print(f"Converting {input_path} → {output_path}")
    print(f"Keeping facet types: {', '.join(sorted(keep_types))}")
    print()

    n_in, n_out, n_skip_type, n_skip_lex, stats = convert_to_entail_labels(
        input_path, output_path, keep_types
    )

    print(f"✅ Conversion complete!")
    print(f"   Input samples:  {n_in}")
    print(f"   Output samples: {n_out}")
    print(f"   Skipped (wrong facet type): {n_skip_type}")
    print(f"   Skipped (no lexical_match): {n_skip_lex}")
    print()

    print(f"📊 Label changes:")
    print(f"   Support labels: {stats['support_pos']} pos, {stats['support_neg']} neg "
          f"({stats['support_pos']/(stats['support_pos']+stats['support_neg'])*100:.1f}% positive)")
    print(f"   Entail labels:  {stats['entail_pos']} pos, {stats['entail_neg']} neg "
          f"({stats['entail_pos']/(stats['entail_pos']+stats['entail_neg'])*100:.1f}% positive)")
    print()
    print(f"   Flipped support=0 → entail=1: {stats['flipped_to_pos']} "
          f"(phrase exists but not supporting)")
    print(f"   Flipped support=1 → entail=0: {stats['flipped_to_neg']} "
          f"(supporting but phrase doesn't exist - BUG!)")
    print()

    if stats['flipped_to_neg'] > 0:
        print(f"⚠️  WARNING: {stats['flipped_to_neg']} cases where support=1 but lexical_match=False")
        print(f"   This suggests a bug in facet_satisfied_in_text() - it marked label=1")
        print(f"   but the phrase/value doesn't actually exist in supporting text!")

    if stats['entail_pos'] < 20:
        print(f"⚠️  WARNING: Only {stats['entail_pos']} positive samples")
        print(f"   This may be too sparse for Mondrian calibration.")
        print(f"   Consider:")
        print(f"   - Using non-Mondrian calibration")
        print(f"   - Merging facet types into one bucket")
        print(f"   - Extracting more calibration data")

    print(f"\n🔧 Next step: Train calibrator")
    print(f"   python train_calibration.py \\")
    print(f"     --data_path {output_path} \\")
    print(f"     --output_path calibrator.json \\")
    print(f"     --use_mondrian")


if __name__ == "__main__":
    main()
