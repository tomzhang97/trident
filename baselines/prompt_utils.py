"""Standardized prompt formatting and answer extraction utilities.

This module ensures all baseline adapters use the same prompt format and answer
extraction logic as Trident for fair comparison.
"""

import re
from typing import List, Dict, Any, Optional


def build_trident_style_prompt(
    question: str,
    passages: List[Dict[str, Any]],
    facets: Optional[List[str]] = None
) -> str:
    """
    Build a Trident-style multi-hop prompt.

    This matches the format in trident/llm_interface.py:build_multi_hop_prompt
    to ensure fair comparison across all baselines.

    Args:
        question: The question to answer
        passages: List of passage dicts with 'text' or 'content' key
        facets: Optional reasoning requirements (not typically used by baselines)

    Returns:
        Formatted prompt string
    """
    # Build context block
    context_parts = []
    for i, passage in enumerate(passages, 1):
        text = passage.get('text', passage.get('content', ''))
        context_parts.append(f"[{i}] {text}")

    context = "\n\n".join(context_parts)

    # Build requirements block if facets provided
    if facets:
        requirements = "\n".join(f"- {f}" for f in facets)
        requirements_block = f"Reasoning requirements:\n{requirements}\n\n"
    else:
        requirements_block = ""

    # Build full prompt (matches Trident's format exactly)
    prompt = (
        f"Context:\n"
        f"{context}\n\n"
        f"{requirements_block}"
        "You are answering a multi-hop question from the HotpotQA dataset.\n"
        "1. Use ONLY the information in the context above.\n"
        "2. First, think briefly and, if helpful, reason in a few short steps.\n"
        "3. Then, on the LAST line of your response, output the answer in the form:\n"
        "   Final answer: <short answer>\n"
        "   - If the question is yes/no, use exactly 'yes' or 'no' as the short answer.\n"
        "   - Otherwise, use a short phrase or name, not a full sentence.\n"
        "4. Do NOT add anything after the 'Final answer:' line.\n"
        "If you truly cannot answer based on the context, use:\n"
        "   Final answer: I cannot answer based on the given context.\n\n"
        f"Question: {question}\n"
        "Now reason and then give the final answer.\n"
    )

    return prompt


KETRAG_SYSTEM_PROMPT = """
---Role---

You are a helpful assistant responding to questions about data in the tables and supplementary materials provided.


---Goal---

Answer the user's question directly by extracting correct information from the data tables provided. The answer will be either a word, a phrase or a short sentence; the answer is supposed to be as short as possible.

If the answer can not be inferred from the data provided, say "Insufficient information." Do not make anything up.

For example, suppose the question is: "What country does the political movement started at the Christmas Meeting of 1888 seek sovereignty from?", your answer should be: "Denmark".

Do not include information where the supporting evidence for it is not provided in the data tables.

---Data tables---

{context_data}

---Goal---

Answer the user's question directly by extracting correct information from the data tables provided. The answer will be either a word, a phrase or a short sentence; the answer is supposed to be as short as possible.

If the answer can not be inferred from the data provided, say "Insufficient information." Do not make anything up.

For example, suppose the question is: "What country does the political movement started at the Christmas Meeting of 1888 seek sovereignty from?", your answer should be: "Denmark".

Do not include information where the supporting evidence for it is not provided in the data tables.
"""


def build_ketrag_original_prompt(question: str, raw_context: str) -> List[Dict[str, str]]:
    """Reproduce the official KET-RAG chat prompt structure.

    The official pipeline sends a system prompt (``LOCAL_SEARCH_EXACT_SYSTEM_PROMPT``)
    with the retrieved tables and a separate user turn containing only the
    question. We mirror that layout so answers follow the exact KET-RAG
    formatting rather than the simplified Question/Answer header.
    """

    system_prompt = KETRAG_SYSTEM_PROMPT.format(context_data=raw_context)

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]


def extract_ketrag_original_answer(generated_text: str) -> str:
    """Lightweight extraction for the original KET-RAG-style generations."""

    answer = generated_text.strip()

    # Remove a leading "Answer:" prefix if the model echoes it
    if answer.lower().startswith("answer:"):
        answer = answer[len("answer:"):].strip()

    return answer


def extract_trident_style_answer(generated_text: str) -> str:
    """
    Extract answer from generated text using Trident's extraction logic.

    This matches the logic in trident/llm_interface.py:extract_answer
    to ensure consistent answer extraction across all baselines.

    Args:
        generated_text: Raw LLM output

    Returns:
        Extracted answer text
    """
    answer = generated_text.strip()

    # First, try to extract from "Final answer:" format (case-insensitive)
    # This is the format requested by build_trident_style_prompt
    match = re.search(
        r'(?:^|\n)\s*final\s+answer\s*:\s*(.+?)(?:\n|$)',
        answer,
        re.IGNORECASE | re.MULTILINE
    )
    if match:
        # Extract the answer portion after "Final answer:"
        answer = match.group(1).strip()
        # Remove any trailing "Final answer:" artifacts
        answer = re.sub(r'\s*final\s+answer\s*:.*$', '', answer, flags=re.IGNORECASE).strip()
    else:
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
