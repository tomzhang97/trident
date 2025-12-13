"""Debug utility to inspect KET-RAG contexts and prompts for a single question."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.prompt_utils import build_ketrag_original_prompt, build_trident_style_prompt


QUESTION_ID = "5a8b57f25542995d1e6f1371"
QUESTION_TEXT = "Were Scott Derrickson and Ed Wood of the same nationality?"
DEFAULT_CONTEXT_FILE = Path("data/ketrag_debug/context_keyword_0.5.json")


def load_contexts(context_path: Path) -> Dict[str, str]:
    with context_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {entry["id"]: entry["context"] for entry in data}


def parse_ketrag_context(context_str: str) -> List[Dict[str, str]]:
    passages: List[Dict[str, str]] = []

    if "-----Entities and Relationships-----" in context_str:
        parts = context_str.split("-----Text source that may be relevant-----")

        if len(parts) > 0:
            graph_section = parts[0].replace("-----Entities and Relationships-----", "").strip()
            if graph_section and graph_section != "N/A":
                passages.append({"text": f"Knowledge Graph:\n{graph_section}"})

        if len(parts) > 1:
            text_section = parts[1].strip()
            lines = text_section.split("\n")
            for line in lines:
                if "|" in line and not line.startswith("id|"):
                    _, text = line.split("|", 1)
                    passages.append({"text": text.strip()})

    if not passages:
        passages.append({"text": context_str})

    return passages


def main() -> None:
    context_by_qid = load_contexts(DEFAULT_CONTEXT_FILE)
    ketrag_context = context_by_qid[QUESTION_ID]

    print("Context first 1000 chars:\n", ketrag_context[:1000], "\n", sep="")
    woodson_index = ketrag_context.find("Woodson")
    print("First 'Woodson' mention index:", woodson_index)

    ketrag_messages = build_ketrag_original_prompt(QUESTION_TEXT, ketrag_context)
    print("Original prompt messages (roles and first 500 chars):")
    for msg in ketrag_messages:
        print(msg["role"], ":", msg["content"][:500], "\n---\n")

    system_len = len(ketrag_messages[0]["content"])
    user_len = len(ketrag_messages[1]["content"])
    print(f"System content length: {system_len}\nUser content length: {user_len}\n")

    passages = parse_ketrag_context(ketrag_context)
    trident_prompt = build_trident_style_prompt(QUESTION_TEXT, passages)
    print("Trident-style prompt first 1000 chars:\n", trident_prompt[:1000], sep="")


if __name__ == "__main__":
    main()