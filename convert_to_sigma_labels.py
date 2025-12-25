#!/usr/bin/env python3
"""
Convert to Σ(p,f) labels: "passage sufficiently supports facet under schema"

This is NOT simple lexical entailment. Σ(p,f) requires:
- ENTITY: Unambiguous entity identification (not just name mention)
- NUMERIC/TEMPORAL: Value bound to correct subject/property (not stray numbers)
- RELATION/BRIDGE: Explicit relational predicate (not topical overlap)

Usage:
  python convert_to_sigma_labels.py calibration_hotpot_fixed.jsonl calibration_hotpot_sigma.jsonl
"""

import json
import re
import sys
from pathlib import Path
from typing import Optional, Dict, Any


# -------------------------
# Schema-specific checks for Σ(p,f)
# -------------------------

def check_entity_sigma(passage: str, supporting_text: str, mention: str, lexical_match: bool) -> bool:
    """
    ENTITY Σ(p,f): Passage unambiguously identifies the entity.

    Not just "name appears", but proper identification:
    - Mention present (lexical_match=True)
    - Local context indicates intended entity (apposition, definition, role, title)
    - Optional: disambiguation signals
    - Reject: list-like co-mentions

    Args:
        passage: Full passage text
        supporting_text: Gold supporting sentences (if available)
        mention: Entity mention from facet
        lexical_match: Whether mention appears in text

    Returns:
        True if passage sufficiently supports entity identification
    """
    if not lexical_match:
        return False

    # Use supporting text if available, else passage
    text = supporting_text if supporting_text else passage
    text_lower = text.lower()
    mention_lower = mention.lower()

    # Find mention position
    if mention_lower not in text_lower:
        return False

    # Check for identification patterns near mention
    # Look for: apposition, role, definition, descriptive context

    # Pattern 1: Apposition "X (Y)" or "X, Y,"
    # e.g., "Leonid Levin (Russian: ...)" or "Leonid Levin, a computer scientist,"
    if re.search(rf"\b{re.escape(mention_lower)}\s*[\(,]", text_lower):
        return True

    # Pattern 2: "is a/an/the" definition
    # e.g., "Leonid Levin is a Soviet-American computer scientist"
    if re.search(rf"\b{re.escape(mention_lower)}\s+is\s+(a|an|the)\s+\w+", text_lower):
        return True

    # Pattern 3: Possessive or role
    # e.g., "Oberoi's hotel", "director X"
    if re.search(rf"\b{re.escape(mention_lower)}('s|s')\s+\w+", text_lower):
        return True

    # Pattern 4: First mention with full name/description
    # If this is the only mention and it's substantial (not in a list)
    mentions = text_lower.count(mention_lower)
    if mentions == 1 and len(text) > len(mention) * 3:
        # Check it's not in a list-like context
        # Reject: "..., X, Y, Z, ..." or "X and Y"
        context_window = 50
        pos = text_lower.find(mention_lower)
        start = max(0, pos - context_window)
        end = min(len(text), pos + len(mention) + context_window)
        context = text_lower[start:end]

        # Count commas and "and" around mention
        comma_count = context.count(',')
        and_count = context.count(' and ')

        if comma_count <= 1 and and_count <= 1:
            return True

    # Default: be conservative
    # If none of the identification patterns match, reject
    # (better to have false negatives than false positives for calibration)
    return False


def check_numeric_sigma(passage: str, supporting_text: str, value: str, unit: str,
                       entity: str, attribute: str, lexical_match: bool) -> bool:
    """
    NUMERIC/TEMPORAL Σ(p,f): Value bound to correct subject/property.

    Not just "number appears", but proper binding:
    - Subject anchor present (entity/topic)
    - Property cue present ("started", "born", "population", "founded", etc.)
    - Value appears in compatible pattern near cue/subject
    - Reject: stray numbers like "Bathurst 12 Hour" for "track length 12"

    Args:
        passage: Full passage text
        supporting_text: Gold supporting sentences
        value: Numeric value from facet
        unit: Unit (if any)
        entity: Entity/subject the value binds to
        attribute: Attribute/property name
        lexical_match: Whether value appears in text

    Returns:
        True if value is properly bound to subject/property
    """
    if not lexical_match:
        return False

    text = supporting_text if supporting_text else passage
    text_lower = text.lower()
    value_lower = value.lower()

    # Find value position
    if value_lower not in text_lower:
        return False

    # Property cue words for different attribute types
    property_cues = {
        'temporal': ['in', 'on', 'during', 'from', 'to', 'since', 'until', 'born', 'died',
                    'founded', 'established', 'started', 'began', 'ended', 'released',
                    'published', 'premiered', 'launched'],
        'count': ['population', 'residents', 'people', 'inhabitants', 'members', 'employees',
                 'students', 'total', 'number of'],
        'length': ['length', 'long', 'distance', 'km', 'miles', 'meters', 'feet'],
        'size': ['area', 'size', 'square', 'acres', 'hectares'],
        'duration': ['hour', 'hours', 'minutes', 'days', 'weeks', 'months', 'years', 'duration'],
        'age': ['age', 'old', 'years old'],
        'rank': ['ranked', 'position', 'place', 'largest', 'smallest', 'first', 'second', 'third'],
    }

    # Check if value appears in a compound phrase (red flag for stray number)
    # e.g., "Bathurst 12 Hour", "Route 66", "Apollo 11"
    value_pos = text_lower.find(value_lower)
    context_before = text_lower[max(0, value_pos - 20):value_pos]
    context_after = text_lower[value_pos + len(value_lower):value_pos + len(value_lower) + 20]

    # Red flag: value is part of a proper name
    # Pattern: "Word Number Word" with no punctuation
    if re.search(r'\b\w+\s*$', context_before) and re.search(r'^\s*\w+\b', context_after):
        # Check if there's a property cue nearby
        window = 100
        window_start = max(0, value_pos - window)
        window_end = min(len(text), value_pos + len(value) + window)
        window_text = text_lower[window_start:window_end]

        # Look for property cues
        has_property_cue = False
        for cue_type, cues in property_cues.items():
            for cue in cues:
                if cue in window_text:
                    has_property_cue = True
                    break
            if has_property_cue:
                break

        # If it's part of a name and no property cue nearby, reject
        if not has_property_cue:
            return False

    # Positive signals: value near property cues
    window = 80
    window_start = max(0, value_pos - window)
    window_end = min(len(text), value_pos + len(value) + window)
    window_text = text_lower[window_start:window_end]

    # Check for property cues in window
    for cue_type, cues in property_cues.items():
        for cue in cues:
            if cue in window_text:
                return True

    # Check for entity/subject near value
    if entity:
        entity_lower = entity.lower()
        if entity_lower in window_text:
            return True

    # Check for explicit binding patterns
    # "X has Y of Z", "X is Z", "X: Z"
    if re.search(rf"\b(has|is|was|were|of|:\s*){re.escape(value_lower)}", text_lower):
        return True

    # Default: reject (conservative)
    return False


def check_relation_sigma(passage: str, supporting_text: str,
                        subject: str, object_: str, predicate: str,
                        lexical_match_subj: bool, lexical_match_obj: bool) -> bool:
    """
    RELATION Σ(p,f): Explicit relational predicate, not just topical overlap.

    Args:
        passage: Full passage text
        supporting_text: Gold supporting sentences
        subject: Subject entity
        object_: Object entity
        predicate: Relation predicate
        lexical_match_subj: Whether subject appears
        lexical_match_obj: Whether object appears

    Returns:
        True if relation is explicitly stated
    """
    if not (lexical_match_subj and lexical_match_obj):
        return False

    text = supporting_text if supporting_text else passage
    text_lower = text.lower()

    # For now, use conservative heuristic:
    # Both entities present + supporting_text non-empty = likely relational
    # This could be made more sophisticated with dependency parsing

    # Check entities are reasonably close (within 100 chars)
    subj_pos = text_lower.find(subject.lower())
    obj_pos = text_lower.find(object_.lower())

    if subj_pos >= 0 and obj_pos >= 0:
        distance = abs(subj_pos - obj_pos)
        if distance < 100:
            return True

    return False


def compute_sigma_label(item: Dict[str, Any]) -> int:
    """
    Compute Σ(p,f) label: does passage sufficiently support facet under schema?

    Σ = 1 iff (support_label == 1) AND (schema_ok == True)

    Args:
        item: Calibration sample with metadata

    Returns:
        0 or 1
    """
    meta = item.get("metadata", {})
    facet_type = meta.get("facet_type", "")

    # Get supporting fact label (from Hotpot)
    # Use original label as proxy for "is this a supporting fact?"
    support_label = int(item.get("label", 0))

    # Get lexical match
    lexical_match = meta.get("lexical_match")

    # Get text
    passage = meta.get("passage_preview", "")
    supporting_text = meta.get("supporting_text", "")

    # Schema-specific checks
    schema_ok = False

    if facet_type == "ENTITY":
        # Extract entity mention (from hypothesis or facet_query)
        hypothesis = item.get("hypothesis", "")
        # Pattern: "contains the exact phrase "X""
        match = re.search(r'exact phrase "([^"]+)"', hypothesis)
        if match:
            mention = match.group(1)
            schema_ok = check_entity_sigma(passage, supporting_text, mention, lexical_match)

    elif facet_type == "NUMERIC":
        # Extract value, unit, entity, attribute from hypothesis
        hypothesis = item.get("hypothesis", "")

        # Pattern 1: "states the value X"
        match = re.search(r'states the value (.+)\.?$', hypothesis)
        if match:
            value_str = match.group(1).strip()
            # Parse "12" or "12 km" or "1952"
            parts = value_str.split()
            value = parts[0]
            unit = " ".join(parts[1:]) if len(parts) > 1 else ""

            # For now, use simple check (could be enhanced with entity/attribute extraction)
            schema_ok = check_numeric_sigma(passage, supporting_text, value, unit,
                                           entity="", attribute="", lexical_match=lexical_match)

    elif facet_type == "RELATION":
        # Extract subject, object from hypothesis
        hypothesis = item.get("hypothesis", "")
        # This is complex - for now, be conservative
        # Require both support_label=1 and lexical_match
        schema_ok = (support_label == 1 and lexical_match is True)

    elif facet_type in ["TEMPORAL", "BRIDGE", "BRIDGE_HOP1", "BRIDGE_HOP2"]:
        # For now, use support_label as proxy
        schema_ok = (support_label == 1)

    else:
        # Unknown type: default to support_label
        schema_ok = (support_label == 1)

    # Σ label: support AND schema
    sigma_label = 1 if (support_label == 1 and schema_ok) else 0

    return sigma_label


def convert_to_sigma_labels(input_path: str, output_path: str, keep_types=None):
    """
    Convert to Σ(p,f) labels with schema-specific checks.

    Args:
        input_path: Input JSONL with lexical_match and support labels
        output_path: Output JSONL with sigma labels
        keep_types: Facet types to keep (default: ENTITY, NUMERIC)
    """
    if keep_types is None:
        keep_types = {"ENTITY", "NUMERIC"}  # Start with lexical types

    n_in = 0
    n_out = 0
    n_skipped_type = 0

    stats = {
        "support_pos": 0,
        "support_neg": 0,
        "sigma_pos": 0,
        "sigma_neg": 0,
        "support_yes_sigma_no": 0,  # Support=1 but Σ=0 (failed schema check)
        "lex_true_sigma_neg": 0,    # lex=True but Σ=0 (critical negatives!)
    }

    with open(input_path) as f, open(output_path, "w") as g:
        for line in f:
            n_in += 1
            r = json.loads(line)

            ft = r.get("metadata", {}).get("facet_type")
            if ft not in keep_types:
                n_skipped_type += 1
                continue

            # Original label (support label)
            orig_label = int(r["label"])
            if orig_label == 1:
                stats["support_pos"] += 1
            else:
                stats["support_neg"] += 1

            # Compute Σ label
            sigma_label = compute_sigma_label(r)

            # Track statistics
            if sigma_label == 1:
                stats["sigma_pos"] += 1
            else:
                stats["sigma_neg"] += 1

            if orig_label == 1 and sigma_label == 0:
                stats["support_yes_sigma_no"] += 1

            lex = r.get("metadata", {}).get("lexical_match")
            if lex is True and sigma_label == 0:
                stats["lex_true_sigma_neg"] += 1

            # Store labels
            r["support_label"] = orig_label
            r["label"] = sigma_label  # Overwrite with Σ label

            g.write(json.dumps(r, ensure_ascii=False) + "\n")
            n_out += 1

    return n_in, n_out, n_skipped_type, stats


def main():
    if len(sys.argv) < 2:
        print("Usage: python convert_to_sigma_labels.py <input.jsonl> [output.jsonl] [facet_types]")
        print("\nExamples:")
        print("  # ENTITY+NUMERIC (default):")
        print("  python convert_to_sigma_labels.py cal.jsonl cal_sigma.jsonl")
        print("\n  # ENTITY only:")
        print("  python convert_to_sigma_labels.py cal.jsonl cal_sigma_entity.jsonl ENTITY")
        sys.exit(1)

    input_path = sys.argv[1]
    if not Path(input_path).exists():
        print(f"Error: {input_path} not found")
        sys.exit(1)

    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        output_path = input_path.replace(".jsonl", "_sigma.jsonl")

    if len(sys.argv) >= 4:
        keep_types = set(sys.argv[3].split(","))
    else:
        keep_types = {"ENTITY", "NUMERIC"}

    print(f"Converting {input_path} → {output_path}")
    print(f"Keeping facet types: {', '.join(sorted(keep_types))}")
    print(f"\nApplying schema-specific Σ(p,f) checks:")
    print(f"  - ENTITY: Unambiguous identification (not just mention)")
    print(f"  - NUMERIC: Value bound to property (not stray numbers)")
    print()

    n_in, n_out, n_skip_type, stats = convert_to_sigma_labels(
        input_path, output_path, keep_types
    )

    print(f"✅ Conversion complete!")
    print(f"   Input samples:  {n_in}")
    print(f"   Output samples: {n_out}")
    print(f"   Skipped (wrong type): {n_skip_type}")
    print()

    print(f"📊 Label statistics:")
    print(f"   Support labels: {stats['support_pos']} pos, {stats['support_neg']} neg "
          f"({stats['support_pos']/(stats['support_pos']+stats['support_neg'])*100:.1f}% positive)")
    print(f"   Σ labels:       {stats['sigma_pos']} pos, {stats['sigma_neg']} neg "
          f"({stats['sigma_pos']/(stats['sigma_pos']+stats['sigma_neg'])*100:.1f}% positive)")
    print()

    print(f"🔍 Schema filtering:")
    print(f"   Support=1 but Σ=0: {stats['support_yes_sigma_no']} "
          f"(failed schema check - wrong binding)")
    print(f"   lex=True but Σ=0:  {stats['lex_true_sigma_neg']} "
          f"(CRITICAL NEGATIVES for calibration!)")
    print()

    if stats['lex_true_sigma_neg'] < 10:
        print(f"⚠️  WARNING: Very few lex=True but Σ=0 negatives ({stats['lex_true_sigma_neg']})")
        print(f"   Schema checks may be too permissive. Consider strengthening.")

    if stats['sigma_pos'] < 20:
        print(f"⚠️  WARNING: Only {stats['sigma_pos']} Σ=1 positives")
        print(f"   May be too sparse for Mondrian calibration.")

    expected_pos_rate = stats['sigma_pos'] / (stats['sigma_pos'] + stats['sigma_neg'])
    if expected_pos_rate > 0.95:
        print(f"⚠️  WARNING: Σ positive rate too high ({expected_pos_rate*100:.1f}%)")
        print(f"   Schema checks may be too weak. Should have more negatives.")
    elif expected_pos_rate > 0.3 and expected_pos_rate < 0.8:
        print(f"✅ Good Σ positive rate ({expected_pos_rate*100:.1f}%) - balanced distribution")

    print(f"\n🔧 Next step: Train calibrator on Σ labels")
    print(f"   python train_calibration.py \\")
    print(f"     --data_path {output_path} \\")
    print(f"     --output_path calibrator.pkl \\")
    print(f"     --use_mondrian")


if __name__ == "__main__":
    main()
