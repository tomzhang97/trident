# Full Baseline Integration Architecture

This document describes the integration of full versions of GraphRAG, Self-RAG, and KET-RAG for fair comparison with TRIDENT on HotpotQA.

## Overview

We integrate three state-of-the-art RAG systems as complete baselines:

1. **GraphRAG** (Microsoft Research): Knowledge graph-based RAG with entity extraction and community summarization
2. **Self-RAG** (Asai et al., ICLR 2024): Retrieval-augmented LM with learned retrieval decisions and reflection tokens
3. **KET-RAG**: Multi-granular indexing combining SkeletonRAG (PageRank + KG) and KeywordRAG (keyword-chunk bipartite graph)

## Architecture Differences

### System Types

| System | Type | Pre-indexing Required | Model Requirements |
|--------|------|---------------------|-------------------|
| TRIDENT | Online RAG | No | Standard LLM |
| GraphRAG | Offline-indexed RAG | Yes | Standard LLM |
| Self-RAG | Trained retrieval-LM | No | Fine-tuned Self-RAG model |
| KET-RAG | Offline-indexed RAG | Yes | Standard LLM |

### Evaluation Challenges

1. **Indexing paradigm mismatch**: GraphRAG/KET-RAG require building indices from documents before querying, while TRIDENT and Self-RAG work online
2. **Model differences**: Self-RAG uses a specialized fine-tuned LLaMA model, while others use standard LLMs
3. **Context vs Retrieval**: HotpotQA provides gold context, but GraphRAG/KET-RAG expect to index document collections

## Integration Strategy

### For HotpotQA Evaluation

We adapt each system to work with HotpotQA's question-context pairs while maintaining their core algorithms:

#### GraphRAG Integration
- **Input adaptation**: Convert HotpotQA context (title-sentences pairs) into documents
- **Per-query indexing**: Build mini knowledge graph for each question's context
- **Process**:
  1. Extract entities and relationships from context documents using LLM
  2. Build entity graph with community detection
  3. Generate community summaries
  4. Perform local or global search to answer question
- **Metrics tracked**: Total tokens (indexing + query), latency, accuracy

#### Self-RAG Integration
- **Model loading**: Use HuggingFace selfrag/selfrag_llama2_7b or 13B model via vllm
- **Context integration**: Format context documents with Self-RAG's special token syntax
- **Process**:
  1. Model decides whether to retrieve (`[Retrieval]` token)
  2. If retrieving, format context with `<paragraph>` tags
  3. Generate answer with reflection tokens (`[Relevant]`, `[Fully supported]`, etc.)
  4. Extract final answer removing special tokens
- **Metrics tracked**: Total tokens (including special tokens), latency, accuracy
- **Note**: Self-RAG's retrieval gate may skip retrieval, which is part of its design

#### KET-RAG Integration
- **Input adaptation**: Same as GraphRAG - convert context to documents
- **Dual-channel indexing**: Build both SkeletonKG (PageRank + entity extraction) and KeywordIndex (keyword-chunk bipartite graph)
- **Process**:
  1. Compute chunk importance scores (PageRank or vocab richness)
  2. Extract entities from top-k chunks to build SkeletonKG
  3. Extract keywords from all chunks for KeywordIndex
  4. Retrieve via entity channel (skeleton triples)
  5. Retrieve via keyword channel (keyword overlap)
  6. Generate answer conditioned on both contexts
- **Metrics tracked**: Total tokens (indexing + query), latency, accuracy

## Implementation Structure

```
external_baselines/
├── graphrag/          # Microsoft GraphRAG (cloned)
├── self-rag/          # Self-RAG (cloned)
└── KET-RAG/           # KET-RAG (cloned)

baselines/
├── __init__.py
├── full_graphrag_adapter.py      # GraphRAG wrapper for HotpotQA
├── full_selfrag_adapter.py       # Self-RAG wrapper for HotpotQA
├── full_ketrag_adapter.py        # KET-RAG wrapper for HotpotQA
└── full_baseline_interface.py   # Unified interface

eval_full_baselines.py            # Main evaluation script
```

## Unified Interface

All adapters implement a common interface for fair comparison:

```python
class BaselineSystem(ABC):
    @abstractmethod
    def answer(
        self,
        question: str,
        context: List[List[str]],  # HotpotQA format
        supporting_facts: Optional[List[tuple]] = None
    ) -> Dict[str, Any]:
        """
        Returns:
            {
                "answer": str,
                "tokens_used": int,  # Total including indexing
                "latency_ms": float,
                "selected_passages": List[Dict],
                "abstained": bool,
                "mode": str,
                "stats": {
                    "prompt_tokens": int,
                    "completion_tokens": int,
                    "indexing_tokens": int,  # GraphRAG/KET-RAG
                    "num_calls": int,
                    ...  # System-specific stats
                }
            }
        """
```

## Metric Tracking

### Token Accounting

Each system tracks tokens differently:

1. **TRIDENT**: Evidence tokens + answer generation tokens
2. **GraphRAG**: Entity extraction tokens + community summarization tokens + answer tokens
3. **Self-RAG**: Input tokens (including special tokens) + output tokens (including reflection tokens)
4. **KET-RAG**: Entity extraction tokens (skeleton) + keyword extraction tokens + answer tokens

For fair comparison, we report:
- **total_tokens**: All tokens consumed (indexing + answering)
- **answer_tokens**: Tokens for final answer generation only
- **overhead_tokens**: total_tokens - answer_tokens

### Latency Tracking

- **End-to-end latency**: Total time from question to answer (includes indexing for GraphRAG/KET-RAG)
- **Answer latency**: Time for final answer generation only

### Quality Metrics

- **Exact Match (EM)**: Exact string match after normalization
- **F1 Score**: Token-level F1 score
- **Abstention rate**: Percentage of questions where system abstains

## Dependencies

### GraphRAG
```bash
cd external_baselines/graphrag
pip install -e .
```

### Self-RAG
```bash
cd external_baselines/self-rag
pip install -r requirements.txt
pip install vllm  # Latest version
```

### KET-RAG
```bash
cd external_baselines/KET-RAG
pip install poetry
poetry install
```

## Environment Setup

### API Keys
```bash
export GRAPHRAG_API_KEY=<your_openai_key>  # For GraphRAG
export OPENAI_API_KEY=<your_openai_key>     # For KET-RAG
export HF_TOKEN=<your_hf_token>             # For Self-RAG model download
```

### Model Downloads
```bash
# Self-RAG models (7B or 13B)
# Downloaded automatically via HuggingFace when first used
# Requires ~14GB (7B) or ~26GB (13B) disk space
```

## Usage

```bash
# Evaluate all full baselines on HotpotQA
python eval_full_baselines.py \
  --data_path data/hotpotqa_dev_shards/shard_0.jsonl \
  --output_dir results/full_baselines \
  --baselines graphrag selfrag ketrag \
  --selfrag_model selfrag/selfrag_llama2_7b \
  --max_samples 100

# Compare with TRIDENT
python aggregate_results.py \
  --trident_results results/hotpotqa/pareto_match_1500.jsonl \
  --baseline_results results/full_baselines/*.jsonl \
  --output results/comparison.csv
```

## Expected Performance

Based on original papers and our adaptations:

| System | EM (HotpotQA) | Avg Tokens | Latency | Notes |
|--------|---------------|------------|---------|-------|
| TRIDENT (Pareto) | ~0.42 | ~1500 | Fast | Online, no indexing |
| GraphRAG | ~0.35-0.40 | ~3000-5000 | Slow | High indexing cost per query |
| Self-RAG (7B) | ~0.38-0.42 | ~1500-2000 | Medium | Trained model, may skip retrieval |
| KET-RAG | ~0.36-0.41 | ~2500-4000 | Medium | Lower indexing cost than GraphRAG |

**Key insights**:
- GraphRAG/KET-RAG have higher token costs due to per-query indexing overhead
- Self-RAG can match TRIDENT's token efficiency but requires specialized model
- TRIDENT combines efficiency of online RAG with quality of structured approaches

## Limitations

1. **Indexing overhead**: GraphRAG/KET-RAG are designed for persistent indices over large document collections, not per-query indexing. HotpotQA adaptation may not showcase their strengths.

2. **Model availability**: Self-RAG requires downloading large fine-tuned models, making it less accessible than TRIDENT which works with any LLM.

3. **Architectural differences**: These systems solve different problems (offline indexing, trained models) vs TRIDENT's online facet-based optimization.

## Future Work

- **Persistent indexing**: Build single index for entire HotpotQA corpus to better utilize GraphRAG/KET-RAG
- **Self-RAG training**: Train Self-RAG-style model with TRIDENT's facet-based approach
- **Hybrid systems**: Combine TRIDENT's facet optimization with GraphRAG's structured extraction

## References

1. **GraphRAG**: "From Local to Global: A Graph RAG Approach" (Microsoft Research, 2024)
2. **Self-RAG**: "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection" (Asai et al., ICLR 2024)
3. **KET-RAG**: "KET-RAG: A Cost-Efficient Multi-Granular Indexing Framework for Graph-RAG" (2025)
