#!/usr/bin/env python3
"""
Diagnostic: Check facet hypothesis quality and NLI scores.

This script helps diagnose why NLI scores are so low.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from trident.config import TridentConfig
from trident.facets import FacetMiner

# Load one example from calibration data
cal_path = Path("calibration_data.jsonl")
if cal_path.exists():
    with open(cal_path) as f:
        samples = [json.loads(line) for line in f][:20]  # First 20 samples

    print("=" * 60)
    print("CALIBRATION DATA DIAGNOSTIC")
    print("=" * 60)

    # Group by label
    positive = [s for s in samples if s['label'] == 1]
    negative = [s for s in samples if s['label'] == 0]

    print(f"\n📊 Score Distribution:")
    print(f"  Positive samples (supporting): {len(positive)}")
    if positive:
        pos_scores = [s['score'] for s in positive]
        print(f"    Min: {min(pos_scores):.3f}")
        print(f"    Max: {max(pos_scores):.3f}")
        print(f"    Avg: {sum(pos_scores)/len(pos_scores):.3f}")

    print(f"\n  Negative samples (non-supporting): {len(negative)}")
    if negative:
        neg_scores = [s['score'] for s in negative]
        print(f"    Min: {min(neg_scores):.3f}")
        print(f"    Max: {max(neg_scores):.3f}")
        print(f"    Avg: {sum(neg_scores)/len(neg_scores):.3f}")

    print(f"\n🔍 Sample Facet Queries:")
    for i, s in enumerate(samples[:5]):
        print(f"\n  [{i+1}] Label: {s['label']}, Score: {s['score']:.3f}")
        print(f"      Question: {s['metadata']['question']}")
        print(f"      Facet: {s['metadata']['facet_query'][:100]}...")

    # Check for score-label correlation
    print(f"\n⚠️  Issues Detected:")
    bad_positive = [s for s in positive if s['score'] < 0.3]
    good_negative = [s for s in negative if s['score'] > 0.7]

    if bad_positive:
        print(f"  - {len(bad_positive)} supporting passages have score < 0.3")
        print(f"    This suggests facet hypotheses don't match passage content well")

    if good_negative:
        print(f"  - {len(good_negative)} non-supporting passages have score > 0.7")
        print(f"    This is concerning - NLI model is giving high scores to wrong passages")

else:
    print("❌ calibration_data.jsonl not found")
    print("   Run extract_calibration_data.py first")

# Test facet mining
print(f"\n" + "=" * 60)
print("FACET HYPOTHESIS FORMAT TEST")
print("=" * 60)

config = TridentConfig()
facet_miner = FacetMiner(config)

test_question = "Were Scott Derrickson and Ed Wood of the same nationality?"
test_supporting_facts = [
    ("Scott Derrickson", 0),
    ("Ed Wood", 0)
]

facets = facet_miner.extract_facets(test_question, test_supporting_facts)

print(f"\nQuestion: {test_question}")
print(f"Facets extracted: {len(facets)}")
for i, facet in enumerate(facets[:3]):
    print(f"\n[{i+1}] Facet ID: {facet.facet_id}")
    print(f"    Type: {facet.facet_type.value}")
    print(f"    Hypothesis: {facet.to_hypothesis()}")
    print(f"    Template: {facet.template}")

print("\n" + "=" * 60)
print("RECOMMENDATION")
print("=" * 60)
print("""
If you see:
1. Supporting passages with scores < 0.3: Facet hypotheses are poorly formed
2. Non-supporting passages with scores > 0.7: NLI model confusion
3. Average positive score < 0.5: Need better hypothesis generation

Solutions:
- Extract from training set instead of dev set (more examples, better distribution)
- Increase num_samples to 500+ for more robust calibration
- Consider adjusting facet hypothesis templates
""")
