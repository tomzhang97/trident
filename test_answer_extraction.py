#!/usr/bin/env python3
"""Test script to verify answer extraction improvements."""

import re
import string
import sys
from pathlib import Path


def _normalize_for_filter(text: str) -> str:
    """Helper function to normalize text for filtering."""
    t = text.lower().strip()
    t = t.translate(str.maketrans("", "", string.punctuation))
    t = " ".join(t.split())
    return t


def _looks_like_meta(text: str) -> bool:
    """Filter out lines that look like meta / system chatter, not the actual answer."""
    norm = _normalize_for_filter(text)
    if not norm:
        return True

    # Pure labels like "Answer", "Final answer"
    bad_exact = {
        "answer",
        "final answer",
        "short answer",
        "prediction",
        "explanation",
        "reasoning",
        "i cannot answer based on the given context",
        "cannot answer",
        "no answer",
    }
    if norm in bad_exact:
        return True

    # Filler/meta phrases that often appear at the start
    filler_prefixes = [
        "ok",
        "okay",
        "sure",
        "let us think",
        "lets think",
        "let s think",
        "let's think",
        "first",
        "second",
        "step",
        "therefore",
        "thus",
        "so the answer",
    ]
    for pref in filler_prefixes:
        if norm.startswith(pref + " "):
            return True

    bad_substrings = [
        "step ",
        "document ",
        "context ",
        "evidence ",
        "reasoning:",
        "reasoning :",
        "analysis:",
        "analysis :",
        "the given context",
        "given context",
        "isnt",
        "is not",
        "should be",
        "would be",
    ]
    return any(bad in norm for bad in bad_substrings)


def extract_final_answer(raw_text: str, question: str, dataset: str = "hotpotqa") -> str:
    """
    Extract the final answer from raw LLM output using a dataset-specific method.

    Args:
        raw_text: The raw text output from the LLM.
        question: The original question (optional).
        dataset: Dataset name ('hotpotqa' or 'musique') for dataset-specific extraction.

    Returns:
        The extracted answer string.
    """
    if dataset.lower() == "musique":
        return _extract_musique_answer(raw_text)
    else:
        # Original HotPotQA extraction - DO NOT MODIFY
        return _extract_hotpotqa_answer(raw_text)


def _extract_hotpotqa_answer(raw_text: str) -> str:
    """
    Extract answer for HotPotQA dataset.

    NOTE: This is the ORIGINAL extraction logic - do not modify.
    """
    answer = raw_text.strip()

    # First, try to extract from "Final answer:" format (case-insensitive)
    match = re.search(r'(?:^|\n)\s*final\s+answer\s*:\s*(.+?)(?:\n|$)', answer, re.IGNORECASE | re.MULTILINE)
    if match:
        answer = match.group(1).strip()
        answer = re.sub(r'\s*final\s+answer\s*:.*$', '', answer, flags=re.IGNORECASE).strip()
        return answer

    # Fall back to removing common prefixes
    prefixes_to_remove = [
        "Answer:", "A:", "Response:", "The answer is:",
        "Based on the context,", "According to the documents,"
    ]

    for prefix in prefixes_to_remove:
        if answer.lower().startswith(prefix.lower()):
            answer = answer[len(prefix):].strip()
            break

    # Take first sentence if answer is very long
    if len(answer) > 500 and '.' in answer:
        answer = answer.split('.')[0] + '.'

    return answer


def _extract_musique_answer(raw_text: str) -> str:
    """
    Extract answer for MuSiQue dataset.

    Universal extraction for multi-hop QA - similar to standard approaches.
    """
    text = raw_text.strip()

    if not text:
        return ""

    # Primary: Look for "Final answer:" pattern
    match = re.search(r'(?i)final\s+answer\s*[:\-]\s*(.+?)(?:\n|$)', text)
    if match:
        answer = match.group(1).strip()
        answer = answer.rstrip('.,;:')
        return answer

    # Secondary: Look for "Answer:" or "The answer is" patterns
    other_patterns = [
        r"(?mi)^answer\s*[:\-]\s*(.+?)(?:\n|$)",
        r"(?mi)the\s+answer\s+is\s+(.+?)(?:\.|,|\n|$)",
    ]

    for pat in other_patterns:
        m = re.search(pat, text)
        if m:
            answer = m.group(1).strip()
            answer = answer.rstrip('.,;:')
            if answer and len(answer) > 1:
                return answer

    # Fallback: remove common prefixes
    prefixes_to_remove = [
        "Answer:", "A:", "Response:", "The answer is:",
        "Based on the context,", "According to the documents,"
    ]

    for prefix in prefixes_to_remove:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()
            break

    # Take first sentence if very long
    if len(text) > 500 and '.' in text:
        text = text.split('.')[0]

    return text.strip().rstrip('.,;:')


def test_extraction():
    """Test various extraction scenarios for HotPotQA and MuSiQue."""

    # HotPotQA test cases (raw_output, question, expected_answer, description)
    # NOTE: These tests match the ORIGINAL HotPotQA extraction behavior
    hotpotqa_test_cases = [
        (
            "Final answer: no",
            "Do the drinks Gibson and Zurracapote both contain gin?",
            "no",
            "Simple yes/no with Final answer:"
        ),
        (
            "Final answer: Beijing",
            "In which city is the ambassador based?",
            "Beijing",
            "Simple factual answer with Final answer:"
        ),
        (
            "Reasoning...\nFinal answer: no",
            "Do the drinks Gibson and Zurracapote both contain gin?",
            "no",
            "With reasoning before final answer"
        ),
        (
            "Final answer: Tim Burton",
            "Who directed the movie?",
            "Tim Burton",
            "Person name answer"
        ),
        (
            "Final answer: yes",
            "Are they the same?",
            "yes",
            "Yes/No answer"
        ),
        (
            "Answer: Fairfax County",
            "What county is it in?",
            "Fairfax County",
            "Answer prefix format"
        ),
        (
            "The answer is: Paris",
            "What is the capital?",
            "Paris",
            "The answer is format"
        ),
    ]

    # MuSiQue test cases (raw_output, question, expected_answer, description)
    musique_test_cases = [
        (
            "Let me trace through this. The author of 'The Great Gatsby' is F. Scott Fitzgerald. "
            "He was born in Saint Paul, Minnesota. Final answer: Saint Paul, Minnesota",
            "Where was the author of 'The Great Gatsby' born?",
            "Saint Paul, Minnesota",
            "Multi-hop entity answer"
        ),
        (
            "Step 1: Identify the director of Inception - Christopher Nolan. "
            "Step 2: Find his nationality - British-American. Final answer: British-American",
            "What is the nationality of the director of Inception?",
            "British-American",
            "Multi-hop with steps"
        ),
        (
            "Based on the context, the capital of France is Paris. The mayor of Paris is Anne Hidalgo. "
            "Final answer: Anne Hidalgo",
            "Who is the mayor of the capital of France?",
            "Anne Hidalgo",
            "Compositional question answer"
        ),
        (
            "The information needed to answer this question is not available in the context. "
            "Final answer: unanswerable",
            "What is the population of the city where Einstein was born?",
            "unanswerable",
            "Unanswerable question"
        ),
        (
            "The band was formed in 1960. Their first album came out in 1963. "
            "Thus, the answer is 1963.",
            "When was the first album released by the band formed in 1960?",
            "1963",
            "Year answer with 'thus' pattern"
        ),
        (
            "Looking at the paragraphs:\n"
            "- The movie was released in 2010\n"
            "- The director is David Fincher\n"
            "Final answer: David Fincher",
            "Who directed the movie released in 2010?",
            "David Fincher",
            "Person name with bullet points"
        ),
        (
            "After combining information from multiple sources, I found that the company was founded in Seattle. "
            "The answer is Seattle",
            "Where was the company founded?",
            "Seattle",
            "City answer without Final answer: prefix"
        ),
        (
            "The Nobel Prize winner in Physics 1921 was Albert Einstein. He developed the theory at ETH Zurich. "
            "Final answer: ETH Zurich.",
            "Where did the 1921 Nobel Prize winner in Physics develop his famous theory?",
            "ETH Zurich",
            "Institution answer with trailing period"
        ),
    ]

    print("Testing HotPotQA answer extraction...")
    print("=" * 80)

    passed = 0
    failed = 0

    for i, (raw_output, question, expected, description) in enumerate(hotpotqa_test_cases, 1):
        extracted = extract_final_answer(raw_output, question, dataset="hotpotqa")

        if expected == "":
            success = extracted == "" or len(extracted) < 3
        else:
            success = extracted.lower().strip() == expected.lower().strip()

        status = "PASS" if success else "FAIL"

        print(f"\nTest {i} [HotPotQA]: {description}")
        print(f"Question: {question}")
        print(f"Raw output: {raw_output[:80]}...")
        print(f"Expected: '{expected}'")
        print(f"Extracted: '{extracted}'")
        print(f"Status: {status}")

        if success:
            passed += 1
        else:
            failed += 1

    print("\n" + "=" * 80)
    print("Testing MuSiQue answer extraction...")
    print("=" * 80)

    for i, (raw_output, question, expected, description) in enumerate(musique_test_cases, 1):
        extracted = extract_final_answer(raw_output, question, dataset="musique")

        if expected == "":
            success = extracted == "" or len(extracted) < 3
        else:
            success = extracted.lower().strip() == expected.lower().strip()

        status = "PASS" if success else "FAIL"

        print(f"\nTest {i} [MuSiQue]: {description}")
        print(f"Question: {question}")
        print(f"Raw output: {raw_output[:80]}...")
        print(f"Expected: '{expected}'")
        print(f"Extracted: '{extracted}'")
        print(f"Status: {status}")

        if success:
            passed += 1
        else:
            failed += 1

    total_tests = len(hotpotqa_test_cases) + len(musique_test_cases)
    print("\n" + "=" * 80)
    print(f"Results: {passed}/{total_tests} passed, {failed} failed")

    return failed == 0


if __name__ == "__main__":
    success = test_extraction()
    sys.exit(0 if success else 1)