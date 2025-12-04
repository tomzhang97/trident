# KET-RAG Baseline Implementations

This document explains the two KET-RAG baseline implementations in TRIDENT and when to use each.

## Overview

TRIDENT provides **two** KET-RAG baseline implementations:

1. **KET-RAG (Reimpl)** - `ketrag_reimpl` - In-framework reimplementation
2. **KET-RAG (Official)** - `ketrag_official` - Faithful wrapper around official code

## Why Two Implementations?

### The Problem with "Faithful Wrappers"

The original KET-RAG adapter claimed to be a "faithful wrapper" but was actually a reimplementation. It:
- Never called official KET-RAG code
- Implemented skeleton + keyword logic from scratch
- Did per-query indexing (not GraphRAG's global indexing)
- Used different prompts, thresholds, and text processing

This is **useful for studying the idea**, but **not faithful to the original paper**.

### The Solution: Two Separate Adapters

We now provide both approaches explicitly:

| Aspect | KET-RAG (Reimpl) | KET-RAG (Official) |
|--------|------------------|-------------------|
| **Code source** | Reimplemented in TRIDENT | Uses official KET-RAG repo |
| **Indexing** | Per-query (in answer()) | Offline (precomputed) |
| **Retrieval** | Custom implementation | Official KET-RAG pipeline |
| **Generation** | Standardized (Trident) | Standardized (Trident) |
| **Dependencies** | graphrag, networkx, sklearn | Official KET-RAG repo |
| **Use case** | Studying idea, ablations | Faithful paper comparison |
| **Setup time** | None (runs immediately) | Requires indexing (~30min) |

## KET-RAG (Reimpl) - In-Framework Reimplementation

### What It Does

Implements the KET-RAG **approach** (skeleton + keyword RAG) entirely within TRIDENT:

1. **SkeletonRAG**:
   - Computes chunk importance via TF-IDF + PageRank
   - Selects top 30% of chunks
   - Extracts entities/relations via LLM prompts
   - Builds knowledge graph skeleton

2. **KeywordRAG**:
   - Extracts top keywords from each chunk
   - Builds keyword-chunk bipartite graph
   - Retrieves chunks by keyword overlap

3. **Dual-channel retrieval**:
   - Combines skeleton KG facts + keyword chunks
   - Formats into Trident prompt
   - Generates answer with specified LLM

### When to Use

✅ **Use `ketrag_reimpl` when:**
- You want to quickly test KET-RAG's approach
- You're doing ablations (changing hyperparameters)
- You need it to run without external dependencies
- You're studying the skeleton + keyword idea

❌ **Don't use `ketrag_reimpl` for:**
- Claiming faithful comparison to the KET-RAG paper
- Benchmark results you'll publish
- Exact replication of paper results

### Usage

```bash
# With OpenAI
python experiments/eval_full_baselines.py \
  --data_path data/hotpotqa_dev_shards/shard_0.jsonl \
  --baselines ketrag_reimpl \
  --ketrag_model gpt-4o-mini

# With local LLM
python experiments/eval_full_baselines.py \
  --data_path data/hotpotqa_dev_shards/shard_0.jsonl \
  --baselines ketrag_reimpl \
  --use_local_llm \
  --local_llm_model Qwen/Qwen2.5-7B-Instruct \
  --local_llm_device cuda:0
```

### Implementation Details

**File**: `baselines/full_ketrag_reimpl_adapter.py`

**Key hyperparameters**:
- `skeleton_ratio=0.3` - Use top 30% of chunks for skeleton
- `max_skeleton_triples=10` - Max KG triples to include
- `max_keyword_chunks=5` - Max chunks from keyword retrieval
- PageRank threshold: `0.2` for edge creation

**Differences from official**:
- Custom entity extraction prompt (not official prompt templates)
- Per-query indexing (not GraphRAG's global pipeline)
- Simplified keyword extraction (not official tokenization)
- Different text preprocessing

## KET-RAG (Official) - Faithful Wrapper

### What It Does

Uses the **official KET-RAG repository** for all retrieval logic:

1. **Offline indexing** (manual step):
   - Run official `graphrag index` to create entities/relations
   - Run official `create_context.py` to retrieve per-question contexts
   - Saves contexts to JSON file

2. **Online generation** (adapter):
   - Loads precomputed contexts from JSON
   - Formats into Trident prompt
   - Generates answer with specified LLM

### When to Use

✅ **Use `ketrag_official` when:**
- You need faithful comparison to the KET-RAG paper
- You're publishing benchmark results
- You want exact replication of paper retrieval
- You're comparing retrieval methods (not implementation details)

❌ **Don't use `ketrag_official` for:**
- Quick experiments (indexing takes 30+ minutes)
- Ablating retrieval hyperparameters (requires re-indexing)
- Environments without the KET-RAG repo

### Usage

**Step 1**: Precompute contexts using official KET-RAG

```bash
# Quick way: Use our helper script
./scripts/prepare_ketrag_contexts.sh \
  data/hotpotqa_dev_shards/shard_0.jsonl \
  ragtest-hotpot

# Manual way:
# 1. Convert HotpotQA to KET-RAG format
python scripts/convert_hotpot_to_ketrag.py \
  data/hotpotqa_dev_shards/shard_0.jsonl \
  KET-RAG/ragtest-hotpot

# 2. Run GraphRAG indexing (takes 10-30 min)
cd KET-RAG
export GRAPHRAG_API_KEY=sk-...
poetry run graphrag index --root ragtest-hotpot/

# 3. Create contexts
poetry run python indexing_sket/create_context.py \
  ragtest-hotpot/ keyword 0.5
cd ..
```

This creates: `KET-RAG/ragtest-hotpot/output/ragtest-hotpot-keyword-0.5.json`

**Step 2**: Run evaluation with precomputed contexts

```bash
python experiments/eval_full_baselines.py \
  --data_path data/hotpotqa_dev_shards/shard_0.jsonl \
  --baselines ketrag_official \
  --ketrag_context_file KET-RAG/ragtest-hotpot/output/ragtest-hotpot-keyword-0.5.json \
  --ketrag_model gpt-4o-mini \
  --ketrag_prompt_style trident \
  --ketrag_compare_original_prompt
```

Use `--ketrag_prompt_style original` if you want the model to answer directly
over the raw KET-RAG context string instead of the standardized Trident prompt.
Add `--ketrag_compare_original_prompt` to emit both generations (primary +
alternate) in the stats so you can see whether the keyword 0.5 retrieval or the
prompting is responsible for poor scores.

### Implementation Details

**File**: `baselines/full_ketrag_official_adapter.py`

**Context format** (from official KET-RAG):
```json
[
  {
    "id": "question_id",
    "context": "-----Entities and Relationships-----\n<graph>\n-----Text source that may be relevant-----\nid|text\nchunk_1|...\nchunk_2|..."
  }
]
```

**What's official**:
- ✅ All retrieval (skeleton + keyword)
- ✅ All hyperparameters (thresholds, top-k, etc.)
- ✅ All indexing (GraphRAG pipeline)
- ✅ All text preprocessing

**What's standardized** (for fair comparison):
- Prompt format (Trident's multi-hop format)
- LLM model (user-specified)
- Answer extraction (Trident's logic)
- Token/latency measurement

## Comparison Table

| Feature | Reimpl | Official |
|---------|--------|----------|
| **Retrieval fidelity** | Approximate | Exact |
| **Setup time** | 0 min | 30+ min |
| **Per-query cost** | Higher (indexing) | Lower (precomputed) |
| **Ablations** | Easy | Hard (re-index) |
| **Dependencies** | Lightweight | Full KET-RAG repo |
| **Paper claims** | ❌ Not faithful | ✅ Faithful |
| **Quick experiments** | ✅ Yes | ❌ No |
| **Benchmark results** | ❌ Don't publish | ✅ Publishable |

## Recommendations

### For Development & Exploration
Use **`ketrag_reimpl`**:
```bash
python experiments/eval_full_baselines.py \
  --baselines ketrag_reimpl selfrag vanillarag \
  --max_samples 50
```

### For Paper Comparison & Benchmarks
Use **`ketrag_official`**:
```bash
# 1. Precompute (once)
./scripts/prepare_ketrag_contexts.sh data/hotpotqa_dev.jsonl ragtest-hotpot

# 2. Evaluate (many times with different models/prompts)
python experiments/eval_full_baselines.py \
  --baselines ketrag_official \
  --ketrag_context_file KET-RAG/ragtest-hotpot/output/ragtest-hotpot-keyword-0.5.json
```

### For Ablations
Use **`ketrag_reimpl`** and modify hyperparameters:
```python
from baselines.full_ketrag_reimpl_adapter import FullKETRAGReimplAdapter

adapter = FullKETRAGReimplAdapter(
    skeleton_ratio=0.5,  # Try different ratios
    max_skeleton_triples=20,  # Try more triples
    max_keyword_chunks=10,  # Try more chunks
)
```

## Transparency in Papers

When reporting results, be explicit:

❌ **Misleading**:
> "We compare against KET-RAG [Citation]"
>
> *Using `ketrag_reimpl`*

✅ **Honest**:
> "We compare against:
> - KET-RAG (official): Uses precomputed contexts from the original implementation
> - KET-RAG (reimpl): Our in-framework reimplementation of the skeleton + keyword approach"

## Summary

- **`ketrag_reimpl`**: Fast, flexible, good for studying the idea
- **`ketrag_official`**: Slow, faithful, good for benchmark comparison

Both standardize generation (prompt + model) for fair comparison to TRIDENT, but only `ketrag_official` preserves the original retrieval logic.
