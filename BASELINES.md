# Baseline Systems for TRIDENT Comparison

This document describes the baseline systems implemented for fair comparison with TRIDENT, including design decisions, improvements, and usage guidelines.

## Overview

We implement three baseline systems for comparison:

1. **Self-RAG-style baseline**: Retrieval-gated question answering with optional critic
2. **GraphRAG-style baseline**: Graph-based retrieval and reasoning
3. **KET-RAG-style baseline**: Cost-efficient multi-granular indexing with skeleton KG and keyword-chunk bipartite graph

All baselines follow simplified but valid implementations of their respective papers, with careful attention to fairness and consistency with TRIDENT.

---

## Self-RAG Baseline

### Design

The Self-RAG baseline implements:

1. **Retrieval gate**: LLM decides whether to retrieve documents (`RETRIEVE` vs `NO_RETRIEVE`) with robust parsing
2. **Context-aware answer generation**: Answers using retrieved documents (max_new_tokens=16, matching TRIDENT)
3. **Optional critic**: Verifies answer support (`SUPPORTS`, `CONTRADICTS`, `INSUFFICIENT`)
4. **Fallback generation**: If critic finds issues, generates answer without context (max_new_tokens=16)

### Key Parameters

```python
SelfRAGSystem(
    llm=instrumented_llm,
    retriever=retriever,
    k=8,                         # Number of documents to retrieve
    use_critic=False,            # Whether to use critic verification
    allow_oracle_context=False   # Whether to accept pre-provided context
)
```

### Evaluation Regimes

#### 1. Retrieval-only mode (recommended for TRIDENT comparison)

```bash
python eval_compare_baselines.py \
    --systems self_rag \
    --common_k 8 \
    --data_path data/hotpotqa_dev.json \
    --output_dir results/selfrag_retrieval
```

- Sets `allow_oracle_context=False` (default)
- Self-RAG **must** use retrieval gate and retriever
- If dataset provides `context`, an error is raised
- Directly comparable to TRIDENT's retrieval behavior

#### 2. Oracle-context mode (for ablation studies)

```bash
python eval_compare_baselines.py \
    --systems self_rag \
    --common_k 8 \
    --selfrag_allow_oracle_context \
    --data_path data/hotpotqa_dev.json \
    --output_dir results/selfrag_oracle
```

- Self-RAG uses provided `context` from dataset
- Bypasses retrieval gate
- Should be labeled as "Self-RAG over oracle context" in paper

#### 3. With critic enabled

```bash
python eval_compare_baselines.py \
    --systems self_rag \
    --common_k 8 \
    --selfrag_use_critic \
    --data_path data/hotpotqa_dev.json \
    --output_dir results/selfrag_critic
```

- Adds verification step after initial answer
- If critic returns `CONTRADICTS` or `INSUFFICIENT`, generates fallback answer
- Increases token usage but may improve accuracy

### Improvements from Initial Version

1. ✅ **Enforced retrieval regime**: Can now prevent oracle context bypass
2. ✅ **Functional critic**: Actually triggers fallback generation on issues
3. ✅ **Harmonized k**: Uses same retrieval count as other baselines
4. ✅ **Clear naming**: Distinguishes retrieval vs oracle-context modes
5. ✅ **Robust gate parsing**: Regex-based parsing handles malformed LLM outputs
6. ✅ **Unified answer budget**: Uses max_new_tokens=16 matching TRIDENT

---

## GraphRAG Baseline

### Design

The GraphRAG baseline implements:

1. **Document retrieval**: Retrieve top-k documents
2. **Graph construction**: Build simple graph from documents
3. **Seed selection**: LLM selects relevant seed nodes
4. **Subgraph expansion**: BFS-based multi-hop expansion
5. **Community summarization**: Summarize subgraphs (max_new_tokens=128)
6. **Answer generation**: Answer from summaries (max_new_tokens=16, matching TRIDENT)

### Key Parameters

```python
GraphRAGSystem(
    llm=instrumented_llm,
    retriever=retriever,
    k=8,              # Number of documents to retrieve
    topk_nodes=20,    # Candidate nodes for seed selection
    max_seeds=10,     # Maximum seed nodes to select
    max_hops=2        # Maximum hops for BFS expansion
)
```

### Graph Construction

The baseline builds a simple graph structure:

- **Nodes**: Each retrieved document becomes a node
- **Edges**: Sequential documents are connected (can be extended)
- **Expansion**: True BFS-based hop expansion respecting `max_hops`

### Evaluation

```bash
python eval_compare_baselines.py \
    --systems graphrag \
    --common_k 8 \
    --data_path data/hotpotqa_dev.json \
    --output_dir results/graphrag
```

### Improvements from Initial Version

1. ✅ **Real BFS expansion**: Actually uses `max_hops` parameter with proper graph traversal
2. ✅ **Seed ID validation**: Validates LLM-generated node IDs and falls back to top candidates
3. ✅ **Harmonized k**: Uses same retrieval count as other baselines
4. ✅ **Robust to LLM errors**: Won't crash on malformed seed outputs
5. ✅ **Unified answer budget**: Uses max_new_tokens=16 matching TRIDENT

### Honest Limitations (to mention in paper)

- Simple chain-based graph structure (not a pre-built global knowledge graph)
- Per-query graph construction (not persistent across queries)
- Should be called "Graph-style RAG baseline" or "Local GraphRAG-lite"

---

## KET-RAG Baseline

### Design

**Note:** This is a simplified KET-RAG-style baseline reimplemented for fair comparison with TRIDENT. It captures the core dual-channel architecture but is not the official KET-RAG implementation.

The KET-RAG baseline implements a cost-efficient multi-granular indexing framework:

1. **SkeletonRAG**: Selects key chunks via vocabulary-based importance scoring, then extracts KG skeleton using LLM
2. **KeywordRAG**: Builds keyword-chunk bipartite graph for lightweight retrieval
3. **Dual-channel retrieval**: Combines entity/knowledge channel and keyword channel
4. **Answer generation**: Uses both skeleton facts and keyword-matched chunks (max_new_tokens=16)

### Key Parameters

```python
KETRAGSystem(
    llm=instrumented_llm,
    retriever=retriever,
    k=8,                         # Number of documents to retrieve
    skeleton_ratio=0.3,          # Ratio of chunks for skeleton KG (30%)
    max_skeleton_triples=10,     # Max triples from skeleton
    max_keyword_chunks=5         # Max chunks from keyword index
)
```

### How It Works

#### Indexing Phase (Per-Query)

1. **Document Retrieval**: Retrieve top-k documents (k=8, harmonized with other baselines)
2. **Chunk Importance**: Compute importance scores using vocabulary richness (not true PageRank)
3. **Skeleton Construction**:
   - Select top 30% of chunks by importance
   - Extract entities and relationships using LLM
   - Build knowledge graph skeleton from key chunks only
4. **Keyword Index**:
   - Extract keywords from all chunks (stopword filtering)
   - Build lightweight keyword-chunk bipartite graph

#### Query Phase

1. **Entity Channel**:
   - Extract query entities
   - Retrieve relevant KG triples from skeleton
2. **Keyword Channel**:
   - Match query keywords against index
   - Retrieve top-matching chunks
3. **Dual-Channel Generation**:
   - Combine skeleton facts and keyword chunks
   - Generate answer using both contexts (max_new_tokens=16, matching TRIDENT)

### Evaluation

```bash
python eval_complete_runnable.py --worker \
    --systems ketrag \
    --common_k 8 \
    --data_path data/hotpotqa_dev.json \
    --output_dir results/ketrag
```

### Improvements from Original Paper

Our implementation provides a simplified but valid KET-RAG baseline:

1. ✅ **Dual-channel retrieval**: Entity/KG + keyword channels
2. ✅ **Skeleton KG construction**: From key chunks using LLM extraction
3. ✅ **Keyword-chunk graph**: Lightweight bipartite graph
4. ✅ **Harmonized k**: Uses same retrieval count as other baselines
5. ✅ **Unified answer budget**: Uses max_new_tokens=16 matching TRIDENT

### Honest Limitations (to mention in paper)

- **Not the official implementation**: This is a reimplementation for fair baseline comparison
- **Simplified importance scoring**: Vocabulary richness heuristic, not true PageRank with graph structure
- **Per-query construction**: Builds indices from retrieved documents, not persistent global indices
- **Simple keyword extraction**: Basic stopword filtering and frequency, not sophisticated NLP
- **LLM entity extraction**: Uses LLM for KG construction (adds to cost but enables flexibility)
- **Must be labeled as**: "KET-RAG-style baseline", "Simplified KET-RAG", or "KETRAG-lite"

### Key Advantages

- **Cost-efficient**: Skeleton KG only from key chunks (30%), not all chunks
- **Multi-granular**: Combines structured (KG) and unstructured (keywords) indexing
- **Balanced**: Better cost-quality tradeoff than full Graph-RAG

### Citation

Based on: [KET-RAG: A Cost-Efficient Multi-Granular Indexing Framework for Graph-RAG](https://arxiv.org/abs/2502.09304)

---

## Fairness Checklist for Paper Comparison

To ensure baselines are not "strawman" and reviewers accept them:

### 1. Same LLM Configuration ✅

All systems use identical:
- Model name and version
- Temperature (0.0 for deterministic comparison)
- `max_new_tokens` (16 for final answers, matching TRIDENT)
- Random seed

**Note:** Intermediate steps (GraphRAG summaries, entity extraction, etc.) may use different token budgets, but all final answer generation uses exactly 16 tokens for fair comparison.

### 2. Same Retrieval Settings ✅

All systems use:
- Same retriever (DenseRetriever with Contriever)
- Same index (built from same corpus)
- Same `k` for first-stage retrieval (controlled by `--common_k`)

### 3. Same Evaluation Regime ✅

**Choose one regime and apply to all systems:**

#### Option A: Retrieval regime (recommended)
- All systems: `context=None` (or used for indexing only in KET-RAG)
- TRIDENT, Self-RAG, GraphRAG, KET-RAG all use retrievers
- Most realistic for real-world comparison

#### Option B: Oracle-context regime
- All systems: `context=gold_paragraphs`
- No one uses retriever (or uses oracle docs for building indices)
- Tests evidence selection only
- **Must** label clearly in paper (e.g., "over oracle context")

### 4. Honest Naming in Paper ✅

Examples of good naming:

- ✅ "Self-RAG-style retrieval baseline"
- ✅ "Graph-style RAG baseline (local per-query graphs)"
- ✅ "Simplified Self-RAG (gate + one critic call, no iteration)"
- ❌ "Full Self-RAG implementation"
- ❌ "GraphRAG with global knowledge graph"

### 5. Token Accounting ✅

All systems properly track:
- All LLM calls (including gates, critics, summaries)
- Prompt tokens + completion tokens
- Consistent through `InstrumentedLLM` wrapper

---

## Command-Line Interface

### Full Comparison

Compare all systems with consistent parameters:

```bash
python eval_compare_baselines.py \
    --systems all \
    --common_k 8 \
    --data_path data/hotpotqa_dev_subset.json \
    --output_dir results/comparison \
    --model meta-llama/Llama-2-7b-hf \
    --device 0 \
    --seed 42
```

### Individual Systems

```bash
# Self-RAG only
python eval_compare_baselines.py \
    --systems self_rag \
    --common_k 8 \
    --selfrag_use_critic \
    --data_path data/hotpotqa_dev.json \
    --output_dir results/selfrag

# GraphRAG only
python eval_compare_baselines.py \
    --systems graphrag \
    --common_k 8 \
    --data_path data/hotpotqa_dev.json \
    --output_dir results/graphrag

# TRIDENT variants
python eval_compare_baselines.py \
    --systems trident_pareto,trident_safe_cover \
    --budget_tokens 2000 \
    --data_path data/hotpotqa_dev.json \
    --output_dir results/trident
```

### Key Arguments

- `--common_k`: Retrieval k for all baselines (default: 8)
- `--selfrag_use_critic`: Enable Self-RAG critic
- `--selfrag_allow_oracle_context`: Allow oracle context in Self-RAG
- `--budget_tokens`: Token budget for TRIDENT (default: 2000)
- `--max_examples`: Limit evaluation examples (0 for all)

---

## Output Files

Each run produces:

1. `{system}_results.json`: Detailed per-query results
2. `token_usage_summary.json`: Aggregate statistics
3. Console output: Formatted comparison table

Example summary metrics:
- Average/median tokens used
- Average/median latency
- Exact Match (EM) and F1 scores
- Abstention rate

---

## Common Pitfalls to Avoid

### ❌ Don't: Use different k values without documenting

```python
# Bad: different k for each system
selfrag = SelfRAGSystem(k=8)
graphrag = GraphRAGSystem(k=20)  # Different k!
```

### ✅ Do: Use harmonized k

```python
# Good: same k for all
common_k = 8
selfrag = SelfRAGSystem(k=common_k)
graphrag = GraphRAGSystem(k=common_k)
```

### ❌ Don't: Mix retrieval and oracle-context regimes

```python
# Bad: inconsistent context usage
selfrag.answer(question, context=None)  # Retrieval
graphrag.answer(question, context=gold)  # Oracle
```

### ✅ Do: Use same regime for all

```python
# Good: all retrieval
selfrag.answer(question, context=None)
graphrag.answer(question, context=None)

# Or all oracle-context (with clear labeling)
selfrag.answer(question, context=gold)
graphrag.answer(question, context=gold)
```

---

## Citation Guidance

When describing these baselines in your paper:

### Self-RAG

> We implement a simplified Self-RAG baseline following Asai et al. (2023), with retrieval gating and single-step critic verification. Unlike the full Self-RAG system, we do not perform iterative refinement. For fair comparison, we use the same retriever and k=8 documents as TRIDENT.

### GraphRAG

> We implement a graph-style RAG baseline inspired by GraphRAG (Edge et al., 2024), building per-query graphs from retrieved documents with BFS-based expansion (max_hops=2). Unlike GraphRAG's global persistent knowledge graphs, our baseline constructs graphs dynamically per query. We use k=8 for retrieval, matching TRIDENT and Self-RAG.

### KET-RAG

> We implement a simplified KET-RAG-style baseline inspired by the dual-channel architecture of KET-RAG ([paper citation]), combining SkeletonRAG (knowledge graph skeleton from key chunks) and KeywordRAG (keyword-chunk bipartite graph). Our implementation uses vocabulary-based importance scoring to select key chunks (top 30%), LLM-based entity extraction for the skeleton, and simple keyword matching for the bipartite graph. Unlike the original KET-RAG with persistent indices, our baseline constructs per-query indices from retrieved documents. We use k=8 for retrieval and max_new_tokens=16 for answer generation, matching other baselines.

---

## Questions?

For implementation details, see:
- `baselines/self_rag_system.py`
- `baselines/graphrag_system.py`
- `baselines/ketrag_system.py`
- `eval_compare_baselines.py`

For TRIDENT implementation:
- `trident/pipeline.py`
- `trident/config.py`
