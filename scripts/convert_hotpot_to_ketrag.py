#!/usr/bin/env python3
"""Convert HotpotQA JSONL to KET-RAG format.

HotpotQA format (JSONL):
{
    "_id": "question_id",
    "question": "...",
    "answer": "...",
    "context": [["title", ["sent1", "sent2", ...]], ...],
    "supporting_facts": [["title", sent_idx], ...],
    "type": "bridge" | "comparison"
}

KET-RAG expected format (directory structure):
<output_dir>/
    input/
        documents.txt  # One document per line (concatenated context)
    qa-pairs/
        qa-pairs.json  # [{"id": "...", "question": "...", "answer": "..."}, ...]
    settings.yaml      # GraphRAG settings

This script creates the necessary directory structure and files.
"""

import json
import sys
from pathlib import Path


def convert_hotpot_to_ketrag(hotpot_jsonl: str, output_dir: str):
    """Convert HotpotQA JSONL to KET-RAG format."""

    output_path = Path(output_dir)
    input_dir = output_path / "input"
    qa_dir = output_path / "qa-pairs"

    # Create directories
    input_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)

    # Load HotpotQA data (supports both JSONL and JSON array formats)
    print(f"Loading HotpotQA data from: {hotpot_jsonl}")
    data = []

    with open(hotpot_jsonl, 'r') as f:
        # Peek at first character to determine format
        first_char = f.read(1)
        f.seek(0)

        if first_char == '[':
            # JSON array format
            print("  Detected JSON array format")
            data = json.load(f)
        else:
            # JSONL format (one JSON object per line)
            print("  Detected JSONL format")
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data.append(json.loads(line))

    print(f"Loaded {len(data)} questions")

    # Convert to KET-RAG format
    qa_pairs = []
    documents = []

    for example in data:
        qid = example.get('_id', 'unknown')
        question = example.get('question', '')
        answer = example.get('answer', '')
        context = example.get('context', [])

        # Create QA pair
        qa_pairs.append({
            "id": qid,
            "question": question,
            "answer": answer
        })

        # Create document from context
        # Each context entry is [title, [sent1, sent2, ...]]
        doc_text = ""
        for title, sentences in context:
            if isinstance(sentences, list):
                text = " ".join(sentences)
            else:
                text = sentences
            doc_text += f"{title}. {text} "

        documents.append(doc_text.strip())

    # Write QA pairs
    qa_file = qa_dir / "qa-pairs.json"
    print(f"Writing QA pairs to: {qa_file}")
    with open(qa_file, 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, indent=2)

    # Write documents (one per line)
    doc_file = input_dir / "documents.txt"
    print(f"Writing {len(documents)} documents to: {doc_file}")
    with open(doc_file, 'w', encoding='utf-8') as f:
        for doc in documents:
            # Clean newlines from doc text
            doc_clean = doc.replace('\n', ' ').replace('\r', ' ')
            f.write(doc_clean + '\n')

    # Create basic settings.yaml
    settings_file = output_path / "settings.yaml"
    if not settings_file.exists():
        print(f"Creating settings.yaml at: {settings_file}")
        settings_content = """encoding_model: cl100k_base
skip_workflows: []
llm:
  api_key: ${GRAPHRAG_API_KEY}
  type: openai_chat
  model: gpt-4o-mini
  model_supports_json: true
  max_tokens: 4000
  request_timeout: 180.0
  api_base: null
  api_version: null
  organization: null
  deployment_name: null
  tokens_per_minute: 50000
  requests_per_minute: 500
  max_retries: 10
  max_retry_wait: 10.0
  sleep_on_rate_limit_recommendation: true
  concurrent_requests: 25

parallelization:
  stagger: 0.3
  num_threads: 50

async_mode: threaded

embeddings:
  async_mode: threaded
  llm:
    api_key: ${GRAPHRAG_API_KEY}
    type: openai_embedding
    model: text-embedding-3-small
    api_base: null
    api_version: null
    organization: null
    deployment_name: null
    tokens_per_minute: 150000
    requests_per_minute: 1000
    max_retries: 10
    max_retry_wait: 10.0
    sleep_on_rate_limit_recommendation: true
    concurrent_requests: 25
  parallelization:
    stagger: 0.3
    num_threads: 50

chunks:
  size: 300
  overlap: 100
  group_by_columns: [id]

input:
  type: file
  file_type: text
  base_dir: "input"
  file_encoding: utf-8
  file_pattern: ".*\\\\.txt$"

cache:
  type: file
  base_dir: "cache"

storage:
  type: file
  base_dir: "output"

reporting:
  type: file
  base_dir: "output"

entity_extraction:
  prompt: "prompts/entity_extraction.txt"
  entity_types: [organization,person,geo,event]
  max_gleanings: 0

summarize_descriptions:
  prompt: "prompts/summarize_descriptions.txt"
  max_length: 500

claim_extraction:
  enabled: false

community_reports:
  prompt: "prompts/community_report.txt"
  max_length: 2000
  max_input_length: 8000

cluster_graph:
  max_cluster_size: 10

embed_graph:
  enabled: false

umap:
  enabled: false

snapshots:
  graphml: false
  raw_entities: false
  top_level_nodes: false

local_search:
  text_unit_prop: 0.5
  community_prop: 0.1
  conversation_history_max_turns: 5
  top_k_entities: 10
  top_k_relationships: 10
  max_tokens: 12000
"""
        with open(settings_file, 'w') as f:
            f.write(settings_content)

    print("\n" + "="*60)
    print("Conversion complete!")
    print("="*60)
    print(f"Output directory: {output_path}")
    print(f"  - QA pairs: {qa_file}")
    print(f"  - Documents: {doc_file}")
    print(f"  - Settings: {settings_file}")
    print("\nNext steps:")
    print("  1. cd KET-RAG")
    print(f"  2. poetry run graphrag index --root {output_dir}/")
    print(f"  3. poetry run python indexing_sket/create_context.py {output_dir}/ keyword 0.5")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_hotpot_to_ketrag.py <hotpot.jsonl> <output_dir>")
        print("")
        print("Example:")
        print("  python scripts/convert_hotpot_to_ketrag.py \\")
        print("    data/hotpotqa_dev_shards/shard_0.jsonl \\")
        print("    KET-RAG/ragtest-hotpot")
        sys.exit(1)

    hotpot_jsonl = sys.argv[1]
    output_dir = sys.argv[2]

    convert_hotpot_to_ketrag(hotpot_jsonl, output_dir)
