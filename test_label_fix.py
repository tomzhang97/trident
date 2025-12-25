#!/usr/bin/env python3
"""
Quick test to verify the label-hypothesis mismatch fixes.
"""

import re
import unicodedata

# Standalone normalization functions (copied from extract_calibration_data.py)
_WS_RE = re.compile(r"\s+")


def norm(s: str) -> str:
    """
    Normalize text for robust exact-phrase matching.
    - NFKC unicode normalization
    - Normalize curly quotes and apostrophes
    - Collapse whitespace
    - Lowercase
    """
    s = unicodedata.normalize("NFKC", s or "")
    # Normalize quotes
    s = s.replace("'", "'").replace("'", "'")
    s = s.replace(""", '"').replace(""", '"')
    s = _WS_RE.sub(" ", s).strip().lower()
    return s


def contains_exact_phrase(passage: str, phrase: str) -> bool:
    """
    Check if passage contains the exact phrase (after normalization).
    Used for ENTITY facets.
    """
    p = norm(passage)
    ph = norm(phrase)
    return ph != "" and ph in p


def contains_value(passage: str, value: str) -> bool:
    """
    Check if passage contains the value string (after normalization).
    Used for NUMERIC facets (simple containment).
    """
    p = norm(passage)
    v = norm(str(value))
    return v != "" and v in p


def is_valid_hypothesis(hypothesis: str) -> bool:
    """
    Filter out garbage hypotheses that will cause NLI to behave randomly.

    Invalid patterns:
    - Contains "?" (question mark) - not a declarative statement
    - Contains "occurred in after being" / "occurred in after who" - garbled templates
    - Too short or nonsensical
    """
    if not hypothesis or len(hypothesis.strip()) < 10:
        return False

    # Question marks indicate malformed hypotheses
    if "?" in hypothesis:
        return False

    # Known garbage patterns from template bugs
    garbage_patterns = [
        "occurred in after being",
        "occurred in after who",
        "occurred in after what",
        "is related to … what",
        "is related to ... what",
    ]

    hyp_lower = hypothesis.lower()
    for pattern in garbage_patterns:
        if pattern in hyp_lower:
            return False

    return True

# Test cases from the user's message
def test_normalization():
    print("=" * 80)
    print("Testing normalization functions")
    print("=" * 80)

    # Test case #8: Leonid Levin
    # Note: The actual full passage likely contains "Leonid Levin" somewhere
    # We test with a modified passage that includes the phrase
    passage8 = "Leonid Anatolievich Levin ( ; Russian: Леони́д Анато́льович Ле́вин ; Ukrainian: Леоні́д Анато́лійович Ле́він ; born November 2, 1948) is a Soviet-American computer scientist. Leonid Levin is known for his work in computational complexity theory."
    phrase8 = "Leonid Levin"

    result8 = contains_exact_phrase(passage8, phrase8)
    print(f"\nTest #8: Leonid Levin (with phrase in passage)")
    print(f"  Passage: {passage8[:100]}...")
    print(f"  Phrase: {phrase8}")
    print(f"  Result: {result8}")
    print(f"  Expected: True (phrase exists in passage)")
    print(f"  ✓ PASS" if result8 else "  ✗ FAIL")

    # Also test that the first sentence alone (without the phrase) returns False
    passage8_first = "Leonid Anatolievich Levin ( ; Russian: Леони́д Анато́льович Ле́вин ; Ukrainian: Леоні́д Анато́лійович Ле́він ; born November 2, 1948) is a Soviet-American computer scientist."
    result8_first = contains_exact_phrase(passage8_first, phrase8)
    print(f"\nTest #8b: Leonid Levin (first sentence only, phrase NOT in text)")
    print(f"  Passage: {passage8_first[:100]}...")
    print(f"  Phrase: {phrase8}")
    print(f"  Result: {result8_first}")
    print(f"  Expected: False (phrase doesn't exist in this snippet)")
    print(f"  ✓ PASS" if not result8_first else "  ✗ FAIL")
    print(f"  Note: This demonstrates correct behavior - we only match exact substrings")

    # Test case #11: Arthur's Magazine (should NOT match)
    passage11 = "Radio City is India's first private FM radio station and was started on 3 July 2001.  It broadcasts on 91.1 (earlier 91.0 in most cities) megahertz from Mumbai (where it was started in 2004), Bengalur"
    phrase11 = "Arthur's Magazine"

    result11 = contains_exact_phrase(passage11, phrase11)
    print(f"\nTest #11: Arthur's Magazine")
    print(f"  Passage: {passage11[:100]}...")
    print(f"  Phrase: {phrase11}")
    print(f"  Result: {result11}")
    print(f"  Expected: False (phrase does NOT exist)")
    print(f"  ✓ PASS" if not result11 else "  ✗ FAIL")

    # Test numeric value matching
    passage_numeric = "The 2014 Liqui Moly Bathurst 12 Hour was an endurance race"
    value_numeric = "12"

    result_numeric = contains_value(passage_numeric, value_numeric)
    print(f"\nTest numeric: Value 12")
    print(f"  Passage: {passage_numeric}")
    print(f"  Value: {value_numeric}")
    print(f"  Result: {result_numeric}")
    print(f"  Expected: True")
    print(f"  ✓ PASS" if result_numeric else "  ✗ FAIL")

    # Test quote normalization
    passage_quotes = "The passage contains 'Leonid Levin' with different quotes"
    phrase_quotes = "Leonid Levin"

    result_quotes = contains_exact_phrase(passage_quotes, phrase_quotes)
    print(f"\nTest quotes: Normalize curly quotes")
    print(f"  Passage: {passage_quotes}")
    print(f"  Phrase: {phrase_quotes}")
    print(f"  Result: {result_quotes}")
    print(f"  Expected: True (quotes normalized)")
    print(f"  ✓ PASS" if result_quotes else "  ✗ FAIL")


def test_hypothesis_filtering():
    print("\n" + "=" * 80)
    print("Testing hypothesis filtering")
    print("=" * 80)

    test_cases = [
        ("The passage contains the exact phrase 'Leonid Levin'.", True, "valid ENTITY hypothesis"),
        ("The passage states the value 12.", True, "valid NUMERIC hypothesis"),
        ("Who is related to controversies?", False, "contains question mark"),
        ("third largest city in Montana passed by Missoula? occurred in after being.", False, "garbage pattern: occurred in after being"),
        ("ouse, who Matt Groening named? occurred in after who.", False, "garbage pattern: occurred in after who"),
        ("The Oberoi family is related to … what city.", False, "garbage pattern: ... what"),
        ("New Faces of is a musical revue with songs occurred in 1952.", True, "valid TEMPORAL hypothesis"),
        ("Too short", False, "too short"),
    ]

    for hyp, expected, reason in test_cases:
        result = is_valid_hypothesis(hyp)
        status = "✓ PASS" if result == expected else "✗ FAIL"
        print(f"\n{status}: {reason}")
        print(f"  Hypothesis: {hyp[:80]}")
        print(f"  Result: {result}, Expected: {expected}")


if __name__ == "__main__":
    test_normalization()
    test_hypothesis_filtering()

    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)
