#!/usr/bin/env python3
"""
Quick test to verify Σ(p,f) hypothesis templates.
Shows before/after comparison.
"""

# Import facets.py directly without going through __init__.py
import sys
import importlib.util

spec = importlib.util.spec_from_file_location("facets", "/home/user/trident/trident/facets.py")
facets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(facets)

Facet = facets.Facet
FacetType = facets.FacetType

def test_entity_hypothesis():
    """Test ENTITY hypothesis asks Σ question."""
    facet = Facet(
        facet_id="test_entity_1",
        facet_type=FacetType.ENTITY,
        template={"mention": "Leonid Levin"},
        weight=1.0
    )

    hyp = facet.to_hypothesis()
    print("ENTITY Hypothesis:")
    print(f"  {hyp}")
    print()

    # Check it contains Σ language
    assert "identifies" in hyp.lower(), "Should use 'identifies'"
    assert "unambiguously" in hyp.lower(), "Should ask for unambiguous identification"
    assert "not merely" in hyp.lower(), "Should distinguish from mere mention"
    assert "Leonid Levin" in hyp, "Should contain entity name"

    # Check it DOESN'T use lexical language
    assert "contains" not in hyp.lower() or "identifies" in hyp.lower(), "Should not be purely lexical"

    print("✅ ENTITY hypothesis asks Σ question (schema-bound sufficiency)")
    return hyp


def test_numeric_hypothesis():
    """Test NUMERIC hypothesis asks Σ question about property binding."""
    facet = Facet(
        facet_id="test_numeric_1",
        facet_type=FacetType.NUMERIC,
        template={
            "entity": "Bathurst 12 Hour",
            "attribute": "length",
            "value": "12",
            "unit": "hours"
        },
        weight=1.0
    )

    hyp = facet.to_hypothesis()
    print("NUMERIC Hypothesis (with entity + attribute + value):")
    print(f"  {hyp}")
    print()

    # Check it contains property binding language
    assert "states that" in hyp.lower(), "Should use 'states that'"
    assert "'s" in hyp or "has" in hyp, "Should show property binding"
    assert "Bathurst 12 Hour" in hyp, "Should contain entity"
    assert "12" in hyp, "Should contain value"

    print("✅ NUMERIC hypothesis asks Σ question (property binding)")

    # Test fallback case (value only)
    facet2 = Facet(
        facet_id="test_numeric_2",
        facet_type=FacetType.NUMERIC,
        template={"value": "42"},
        weight=1.0
    )

    hyp2 = facet2.to_hypothesis()
    print("NUMERIC Hypothesis (value only, fallback):")
    print(f"  {hyp2}")
    print()

    # Even fallback should suggest binding
    assert "binds" in hyp2.lower() or "property" in hyp2.lower(), "Fallback should suggest property binding"

    print("✅ NUMERIC fallback hypothesis suggests property binding")
    return hyp, hyp2


def main():
    print("=" * 80)
    print("Testing Σ(p,f) Hypothesis Templates")
    print("=" * 80)
    print()

    print("OLD templates (lexical):")
    print("  ENTITY: 'The passage contains the exact phrase \"{mention}\".'")
    print("  NUMERIC: 'The passage states the value {value}.'")
    print()

    print("NEW templates (Σ schema-bound):")
    print("  ENTITY: 'The passage identifies \"{mention}\" unambiguously (not merely a mention).'")
    print("  NUMERIC: 'The passage states that {entity}'s {attr} is {value}.'")
    print()
    print("=" * 80)
    print()

    try:
        entity_hyp = test_entity_hypothesis()
        print()

        numeric_hyp, numeric_fallback = test_numeric_hypothesis()
        print()

        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print()
        print("✅ All hypothesis templates updated successfully!")
        print()
        print("Hypotheses now ask Σ(p,f) questions:")
        print("  - ENTITY: Unambiguous identification (not mere mention)")
        print("  - NUMERIC: Property binding (not mere value containment)")
        print()
        print("This should fix Blocker B (TPR=0 / hypothesis-verifier mismatch).")
        print()
        print("Next steps:")
        print("  1. Re-extract calibration data (will use new hypotheses)")
        print("  2. Convert to Σ labels with convert_to_sigma_labels.py")
        print("  3. Train calibrator and check TPR > 0")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()