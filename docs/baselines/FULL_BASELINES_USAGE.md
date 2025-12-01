# Full Baselines Usage Guide

Complete guide for setting up and evaluating full versions of GraphRAG, Self-RAG, and KET-RAG as baselines for TRIDENT comparison.

## Quick Start

### 1. Installation

```bash
# Make sure you're in the trident directory
cd /path/to/trident

# Run installation script
./install_full_baselines.sh
```

> Tip: Keep your active conda/virtualenv loaded when running the script so all
> baseline dependencies land in the same environment as TRIDENT. The script now
> forces Poetry (for KET-RAG) to reuse the current environment instead of
> creating its own venv.

### 2. Set Environment Variables

```bash
# For GraphRAG and KET-RAG (OpenAI API)
export GRAPHRAG_API_KEY="sk-..."
# OR
export OPENAI_API_KEY="sk-..."

# For Self-RAG (HuggingFace model download)
export HF_TOKEN="hf_..."  # Optional, only needed for private models
```

### 3. Run Evaluation

```bash
# Evaluate all baselines on a small sample (for testing)
python eval_full_baselines.py \
    --data_path data/hotpotqa_dev_shards/shard_0.jsonl \
    --max_samples 10 \
    --baselines all \
    --output_dir results/full_baselines_test

# Evaluate specific baselines
python eval_full_baselines.py \
    --data_path data/hotpotqa_dev_shards/shard_0.jsonl \
    --baselines selfrag graphrag \
    --max_samples 50

# Full evaluation (all baselines, entire dataset)
python eval_full_baselines.py \
    --data_path data/hotpotqa_dev_shards/shard_0.jsonl \
    --baselines all \
    --output_dir results/full_baselines_complete
```

## Understanding Metric Separation

### Why We Separate Indexing from Query Costs

To ensure **fair comparison** with original baseline papers, we report two sets of metrics:

**1. Query-Only Metrics (PRIMARY)**
- **Purpose**: Matches how original papers report performance
- **Assumption**: Index is pre-built offline (one-time cost)
- **Includes**: Only online inference costs (answer generation)
- **Use for**: Fair comparison with published baseline results

**2. Total Metrics (Reference)**
- **Purpose**: Shows full cost for HotpotQA per-query adaptation
- **Includes**: Indexing costs + query costs
- **Use for**: Understanding full system overhead in dynamic context scenarios

### Why This Matters

**Original Design**:
- GraphRAG/KET-RAG are designed for **persistent indices** over large static corpora
- Index is built **once offline**, then reused for many queries
- Papers report **query-only** costs assuming pre-built index

**HotpotQA Adaptation**:
- HotpotQA provides **dynamic contexts per question** (10-20 paragraphs each)
- Requires **per-query indexing** to handle different contexts
- Mixing indexing and query costs would **unfairly penalize** these systems

### Example Breakdown

**GraphRAG on HotpotQA:**
- Indexing (offline simulation): ~2500 tokens (entity extraction + community detection)
- Query (online): ~800 tokens (answer generation from graph)
- **Primary metric reported**: 800 tokens (query-only)
- **Reference metric**: 3300 tokens (total)

**Self-RAG on HotpotQA:**
- Indexing: 0 tokens (uses pre-trained model, no indexing)
- Query: ~1650 tokens (generation with reflection tokens)
- **Primary metric reported**: 1650 tokens (same as total)

### How to Interpret Results

When comparing results:
- **For fairness**: Compare query-only metrics across all systems
- **For HotpotQA cost**: Use total metrics (includes indexing overhead)
- **For production**: Consider whether index would be pre-built (query-only) or rebuilt per-query (total)

## Detailed Installation

### Prerequisites

- Python 3.10+
- CUDA (optional, for vllm/Self-RAG GPU acceleration)
- At least 16GB RAM (32GB recommended for Self-RAG 13B)
- OpenAI API key (for GraphRAG and KET-RAG)

### Manual Installation

If the automated script fails, install each baseline manually:

#### GraphRAG

```bash
cd external_baselines/graphrag
pip install -e .
```

Dependencies: fnllm, openai, networkx, pandas, lancedb, nltk, etc.

#### Self-RAG

```bash
cd external_baselines/self-rag
pip install -r requirements.txt
pip install vllm  # Latest version recommended
```

Dependencies: vllm, transformers, flash-attn (requires CUDA), datasets, etc.

**Note**: Self-RAG requires vllm which has specific CUDA requirements. If you encounter issues:
- Check vllm installation docs: https://vllm.readthedocs.io/en/latest/
- For CPU-only: Self-RAG will be slow, not recommended

#### KET-RAG

```bash
cd external_baselines/KET-RAG
pip install poetry  # If not already installed
poetry install
```

Dependencies: Similar to GraphRAG (it's based on GraphRAG v0.4.1)

### Verify Installation

```bash
python -c "from baselines import FullGraphRAGAdapter, FullSelfRAGAdapter, FullKETRAGAdapter; print('✓ All adapters available')"
```

## Usage Details

### Evaluation Script Options

```bash
python eval_full_baselines.py --help
```

**Key arguments:**

- `--data_path`: Path to HotpotQA JSONL file (required)
- `--output_dir`: Directory for results (default: results/full_baselines)
- `--baselines`: Which baselines to run (choices: graphrag, selfrag, ketrag, all)
- `--max_samples`: Limit number of samples (for testing)
- `--graphrag_model`: Model for GraphRAG/KET-RAG (default: gpt-4o-mini)
- `--selfrag_model`: Self-RAG model (default: selfrag/selfrag_llama2_7b)
- `--selfrag_max_tokens`: Max tokens for Self-RAG (default: 100)

### Example Workflows

#### 1. Quick Test (10 samples, all baselines)

```bash
python eval_full_baselines.py \
    --data_path data/hotpotqa_dev_shards/shard_0.jsonl \
    --max_samples 10 \
    --baselines all
```

#### 2. Compare Self-RAG 7B vs 13B

```bash
# 7B model
python eval_full_baselines.py \
    --data_path data/hotpotqa_dev_shards/shard_0.jsonl \
    --baselines selfrag \
    --selfrag_model selfrag/selfrag_llama2_7b \
    --output_dir results/selfrag_7b

# 13B model
python eval_full_baselines.py \
    --data_path data/hotpotqa_dev_shards/shard_0.jsonl \
    --baselines selfrag \
    --selfrag_model selfrag/selfrag_llama2_13b \
    --output_dir results/selfrag_13b
```

#### 3. Test Different LLMs for GraphRAG

```bash
# GPT-4o-mini (fast, cheap)
python eval_full_baselines.py \
    --data_path data/hotpotqa_dev_shards/shard_0.jsonl \
    --baselines graphrag \
    --graphrag_model gpt-4o-mini \
    --max_samples 50

# GPT-4o (more capable, expensive)
python eval_full_baselines.py \
    --data_path data/hotpotqa_dev_shards/shard_0.jsonl \
    --baselines graphrag \
    --graphrag_model gpt-4o \
    --max_samples 50
```

#### 4. Full Evaluation on Complete Dataset

```bash
# WARNING: This will be expensive (many LLM calls)
# Estimate: ~$10-50 depending on model and dataset size

python eval_full_baselines.py \
    --data_path data/hotpotqa_dev_shards/shard_0.jsonl \
    --baselines all \
    --graphrag_model gpt-4o-mini \
    --selfrag_model selfrag/selfrag_llama2_7b \
    --output_dir results/full_eval_complete
```

## Output Files

### Per-Baseline Results

Each baseline generates:

1. **`{baseline}_results.jsonl`**: Individual results (one per question)
   ```json
   {
     "question_id": "5a8b57f25542995d1e6f1371",
     "question": "What is...",
     "answer": "correct answer",
     "prediction": "predicted answer",
     "em": 1.0,
     "f1": 1.0,
     "tokens_used": 1500,
     "latency_ms": 2500.0,
     "abstained": false,
     "mode": "selfrag",
     "stats": {...}
   }
   ```

2. **`{baseline}_summary.json`**: Aggregate statistics
   ```json
   {
     "baseline": "selfrag",
     "num_examples": 100,
     "num_processed": 98,
     "num_abstained": 2,
     "abstention_rate": 0.02,
     "avg_em": 0.42,
     "avg_f1": 0.51,
     "avg_tokens": 1523.5,
     "median_tokens": 1489.0,
     "avg_latency_ms": 2341.2,
     "median_latency_ms": 2187.5
   }
   ```

### Combined Summary

**`combined_summary.json`**: All baselines comparison
```json
{
  "graphrag": {...},
  "selfrag": {...},
  "ketrag": {...}
}
```

## Comparing with TRIDENT

### 1. Run TRIDENT Evaluation

```bash
python eval_complete_runnable.py \
    --data_path data/hotpotqa_dev_shards/shard_0.jsonl \
    --config_family pareto_match_1500 \
    --output_dir results/trident \
    --mode pareto
```

### 2. Run Full Baselines

```bash
python eval_full_baselines.py \
    --data_path data/hotpotqa_dev_shards/shard_0.jsonl \
    --baselines all \
    --output_dir results/full_baselines
```

### 3. Compare Results

```bash
python aggregate_results.py \
    --trident_results results/trident/pareto_match_1500.jsonl \
    --baseline_results results/full_baselines/*_results.jsonl \
    --output results/comparison.csv
```

Example comparison output:

| System | EM | F1 | Avg Tokens | Avg Latency (ms) | Abstention Rate |
|--------|----|----|------------|------------------|-----------------|
| TRIDENT (Pareto) | 0.42 | 0.53 | 1500 | 1200 | 0.01 |
| GraphRAG | 0.38 | 0.49 | 3500 | 4500 | 0.00 |
| Self-RAG (7B) | 0.41 | 0.52 | 1650 | 2300 | 0.02 |
| KET-RAG | 0.39 | 0.50 | 2800 | 3800 | 0.00 |

## System-Specific Notes

### GraphRAG

**Strengths:**
- Structured knowledge graph representation
- Good for multi-hop reasoning over entities

**Limitations:**
- High indexing cost due to per-query adaptation (not designed for this)
- Slow due to multi-stage pipeline (entity extraction → graph building → summarization → answer)

**Token breakdown (Separated Metrics):**
- **Indexing (Offline)**:
  - Entity extraction: ~1000-2000 tokens
  - Community summarization: ~500-1000 tokens
  - **Subtotal: ~2000-3000 tokens**
- **Query (Online)**:
  - Answer generation with graph context: ~500-800 tokens
  - **Subtotal: ~500-800 tokens**
- **Total (HotpotQA per-query): ~2500-4000 tokens**

**Note**: Query-only costs (~500-800) are reported as primary metrics to match original paper.

### Self-RAG

**Strengths:**
- Learned retrieval decisions (can skip retrieval when not needed)
- Reflection tokens provide interpretability
- Competitive performance with standard LLMs

**Limitations:**
- Requires fine-tuned model (not available for all model sizes)
- Model-specific (can't easily swap LLM)
- Requires GPU for reasonable performance

**Token breakdown (Separated Metrics):**
- **Indexing (Offline)**: 0 tokens (uses pre-trained model, no per-query indexing)
- **Query (Online)**:
  - Input (question + context): ~1000-1500 tokens
  - Output (answer + reflection tokens): ~50-150 tokens
  - **Subtotal: ~1100-1700 tokens**
- **Total: ~1100-1700 tokens per question**

**Note**: Self-RAG has zero indexing costs, so query-only = total costs.

### KET-RAG

**Strengths:**
- Multi-granular indexing (skeleton + keywords)
- Lower indexing cost than full GraphRAG
- Dual-channel retrieval combines structured and keyword-based

**Limitations:**
- Still requires per-query indexing for HotpotQA
- More complex than simple retrieval

**Token breakdown (Separated Metrics):**
- **Indexing (Offline)**:
  - Skeleton extraction (PageRank + entity extraction on top 30%): ~800-1500 tokens
  - Keyword indexing: ~0 tokens (rule-based)
  - **Subtotal: ~800-1500 tokens**
- **Query (Online)**:
  - Dual-channel retrieval + answer generation: ~700-1000 tokens
  - **Subtotal: ~700-1000 tokens**
- **Total (HotpotQA per-query): ~1500-2500 tokens**

**Note**: Query-only costs (~700-1000) are reported as primary metrics to match original paper.

## Troubleshooting

### GraphRAG/KET-RAG Errors

**Error: "API key required"**
```bash
export GRAPHRAG_API_KEY="sk-..."
# OR
export OPENAI_API_KEY="sk-..."
```

**Error: "Module 'graphrag' not found"**
```bash
cd external_baselines/graphrag
pip install -e .
```

### Self-RAG Errors

**Error: "vllm not available"**
```bash
pip install vllm
# If CUDA issues, see: https://vllm.readthedocs.io/en/latest/getting_started/installation.html
```

**Error: "Model download failed"**
- Check internet connection
- Verify HuggingFace access (some models require acceptance of terms)
- Set HF_TOKEN if needed: `export HF_TOKEN="hf_..."`

**Error: "CUDA out of memory"**
- Use 7B model instead of 13B
- Reduce batch size in vllm (not exposed in current API, would need modification)

### General Issues

**Slow performance:**
- GraphRAG/KET-RAG: Use cheaper model (gpt-4o-mini instead of gpt-4)
- Self-RAG: Ensure GPU is being used (check with `nvidia-smi`)

**High costs:**
- Use `--max_samples` for testing before full evaluation
- Use gpt-4o-mini for GraphRAG/KET-RAG (much cheaper than gpt-4)

## Cost Estimates

Approximate costs for 100 questions:

| Baseline | LLM | Cost per Question | Total (100 questions) |
|----------|-----|-------------------|----------------------|
| GraphRAG | gpt-4o-mini | $0.05-0.10 | $5-10 |
| GraphRAG | gpt-4o | $0.30-0.50 | $30-50 |
| Self-RAG | 7B (local) | $0 | $0 (GPU compute) |
| Self-RAG | 13B (local) | $0 | $0 (GPU compute) |
| KET-RAG | gpt-4o-mini | $0.03-0.08 | $3-8 |
| KET-RAG | gpt-4o | $0.20-0.40 | $20-40 |

**TRIDENT**: Comparable to Self-RAG when using same LLM backend (no indexing overhead)

## Best Practices

1. **Start small**: Test with `--max_samples 10` before full evaluation
2. **Use cheap models for testing**: gpt-4o-mini for GraphRAG/KET-RAG
3. **Monitor costs**: OpenAI API usage dashboard
4. **Save results frequently**: Script saves after each question
5. **Use same dataset**: Ensure fair comparison with TRIDENT

## Citation

If you use these baselines in your research, please cite the original papers:

**GraphRAG:**
```bibtex
@article{edge2024local,
  title={From Local to Global: A Graph RAG Approach to Query-Focused Summarization},
  author={Edge, Darren and Trinh, Ha and Cheng, Newman and others},
  journal={arXiv preprint arXiv:2404.16130},
  year={2024}
}
```

**Self-RAG:**
```bibtex
@inproceedings{asai2024selfrag,
  title={Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection},
  author={Asai, Akari and Wu, Zeqiu and Wang, Yizhong and Sil, Avirup and Hajishirzi, Hannaneh},
  booktitle={ICLR},
  year={2024}
}
```

**KET-RAG:**
```bibtex
@article{ketrag2025,
  title={KET-RAG: A Cost-Efficient Multi-Granular Indexing Framework for Graph-RAG},
  author={...},
  journal={arXiv preprint arXiv:2502.09304},
  year={2025}
}
```
