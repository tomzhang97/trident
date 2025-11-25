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


def extract_final_answer(raw_text: str, question: str) -> str:
    """
    Extract the final answer from raw LLM output using a more robust method.

    Args:
        raw_text: The raw text output from the LLM.
        question: The original question, used for context (e.g., Yes/No detection).

    Returns:
        The extracted answer string.
    """
    text = raw_text.strip()

    if not text:
        return ""

    # 1) First priority: Look for "Final answer:" pattern (case-insensitive)
    # This is the most explicit indicator from our prompt
    final_answer_patterns = [
        r"(?i)final\s+answer\s*[:\-]\s*(.+?)(?:\n|$)",  # "Final answer: <answer>"
        r"(?i)final\s+answer\s*[:\-]\s*(.+)",  # Fallback without newline requirement
    ]

    for pat in final_answer_patterns:
        m = re.search(pat, text)
        if m:
            cand = m.group(1).strip()
            # Clean up common trailing artifacts
            cand = re.sub(r'\s*(Human:|Assistant:|Question:).*$', '', cand, flags=re.IGNORECASE)
            # If empty or too short after cleanup, skip
            if not cand or len(cand) < 1:
                continue
            if not _looks_like_meta(cand):
                return cand

    # 2) Yes/No shortcut for binary questions
    q_lower = question.strip().lower()
    is_yesno_question = q_lower.startswith((
        "is ", "are ", "was ", "were ",
        "do ", "does ", "did ",
        "can ", "could ", "should ", "would ",
        "has ", "have ", "had "
    ))

    if is_yesno_question:
        t_lower = text.lower()
        # Look for explicit yes/no patterns
        # Count occurrences and use the last one mentioned
        yes_matches = list(re.finditer(r'\byes\b', t_lower))
        no_matches = list(re.finditer(r'\bno\b', t_lower))

        if yes_matches or no_matches:
            # Get the position of the last yes/no
            last_yes_pos = yes_matches[-1].start() if yes_matches else -1
            last_no_pos = no_matches[-1].start() if no_matches else -1

            # Return the one that appears last in the text
            if last_no_pos > last_yes_pos:
                return "no"
            elif last_yes_pos > last_no_pos:
                return "yes"

    # 3) Look for other answer patterns
    other_patterns = [
        r"(?mi)^answer\s*[:\-]\s*(.+?)(?:\n|$)",
        r"(?mi)the\s+answer\s+is\s+(.+?)(?:\.|$)",  # Require space after "is" to avoid matching "isn't"
        r"(?mi)therefore[,]?\s+(?:the\s+answer\s+is\s+)?(.+?)(?:\.|$)",
    ]

    for pat in other_patterns:
        m = re.search(pat, text)
        if m:
            cand = m.group(1).strip()
            # Remove trailing punctuation and artifacts
            cand = re.sub(r'[.,;]+$', '', cand)
            # Clean up Human:/Assistant: artifacts
            cand = re.sub(r'\s*(Human:|Assistant:|Question:).*$', '', cand, flags=re.IGNORECASE)
            cand = re.sub(r'(Human:|Assistant:|Question:).*$', '', cand, flags=re.IGNORECASE)
            if cand and len(cand) > 1 and not _looks_like_meta(cand):
                return cand

    # 4) Fallback: scan from the BOTTOM for a non-meta short line
    #    This avoids grabbing opening filler like "Okay,".
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for line in reversed(lines):
        # Skip lines with colons (usually labels)
        if ":" in line:
            continue
        if _looks_like_meta(line):
            continue
        # Look for substantive content (not too short, not too long)
        if 2 <= len(line) <= 100:
            # Clean up
            line = re.sub(r'^(so|thus|therefore|hence)[,\s]+', '', line, flags=re.IGNORECASE)
            # Clean artifacts
            line = re.sub(r'\s*(Human:|Assistant:|Question:).*$', '', line, flags=re.IGNORECASE)
            line = re.sub(r'(Human:|Assistant:|Question:).*$', '', line, flags=re.IGNORECASE)
            if len(line) > 1 and not _looks_like_meta(line):
                return line

    # 5) Final fallback: last sentence or phrase
    # Split by sentence-ending punctuation
    sentences = re.split(r'[.!?]\s+', text)
    if sentences:
        last_sentence = sentences[-1].strip()
        # Clean artifacts
        last_sentence = re.sub(r'\s*(Human:|Assistant:|Question:).*$', '', last_sentence, flags=re.IGNORECASE)
        last_sentence = re.sub(r'(Human:|Assistant:|Question:).*$', '', last_sentence, flags=re.IGNORECASE)
        # Remove trailing periods
        last_sentence = last_sentence.rstrip('.')
        # Skip if it's just a fragment (starts with punctuation, is very short, or looks like meta)
        if last_sentence.startswith(("'", '"', '-', 'n\'t')):
            return ""
        # Check if it's substantial (more than 4 chars and not just punctuation/fragments)
        if last_sentence and len(last_sentence) > 4 and not _looks_like_meta(last_sentence):
            # Make sure it has at least one letter
            if any(c.isalpha() for c in last_sentence):
                return last_sentence

    # 6) If all else fails, return empty rather than garbage
    return ""


def test_extraction():
    """Test various extraction scenarios."""

    test_cases = [
        # (raw_output, question, expected_answer, description)
        (
            "Based on the context, Gibson contains gin and Zurracapote does not. Final answer: no",
            "Do the drinks Gibson and Zurracapote both contain gin?",
            "no",
            "Simple yes/no with Final answer:"
        ),
        (
            "Let me analyze the documents. The Moroccan ambassador is based in Beijing. Final answer: Beijing",
            "In which city is the ambassador based?",
            "Beijing",
            "Simple factual answer with Final answer:"
        ),
        (
            "Step 1: Find info about Gibson. Step 2: Find info about Zurracapote. Final answer: no",
            "Do the drinks Gibson and Zurracapote both contain gin?",
            "no",
            "With step-by-step reasoning"
        ),
        (
            "the given context. Final answer:",
            "Some question?",
            "",
            "Incomplete answer (empty after Final answer:)"
        ),
        (
            "Galleria. Therefore, the answer isn't",
            "What county is it in?",
            "",
            "Incomplete mid-sentence (should be filtered as meta)"
        ),
        (
            "So the answer should be",
            "What is the answer?",
            "",
            "Meta content (should be filtered)"
        ),
        (
            "Looking at the context, the answer is Fairfax County.",
            "What county is it in?",
            "Fairfax County",
            "Natural language answer"
        ),
        (
            "The movie was directed by Tim Burton. Final answer: Tim Burton",
            "Who directed the movie?",
            "Tim Burton",
            "Person name answer"
        ),
        (
            "Is this a yes/no question? Yes, the answer is yes.",
            "Are they the same?",
            "yes",
            "Yes/No question detection"
        ),
        (
            "After checking both sources, I found that no, they are not the same.",
            "Are they the same?",
            "no",
            "Yes/No question with 'no' in sentence"
        ),
        (
            "IT products and services.Human: What is the name",
            "What products does it provide?",
            "IT products and services",
            "Cut off with Human: artifact"
        ),
    ]

    print("Testing answer extraction...")
    print("=" * 80)

    passed = 0
    failed = 0

    for i, (raw_output, question, expected, description) in enumerate(test_cases, 1):
        extracted = extract_final_answer(raw_output, question)

        # For empty expected answers, we're checking if extraction produces empty or filters out meta
        if expected == "":
            success = extracted == "" or len(extracted) < 3
        else:
            success = extracted.lower().strip() == expected.lower().strip()

        status = "✓ PASS" if success else "✗ FAIL"

        print(f"\nTest {i}: {description}")
        print(f"Question: {question}")
        print(f"Raw output: {raw_output[:100]}...")
        print(f"Expected: '{expected}'")
        print(f"Extracted: '{extracted}'")
        print(f"Status: {status}")

        if success:
            passed += 1
        else:
            failed += 1

    print("\n" + "=" * 80)
    print(f"Results: {passed}/{len(test_cases)} passed, {failed} failed")

    return failed == 0


if __name__ == "__main__":
    success = test_extraction()
    sys.exit(0 if success else 1)