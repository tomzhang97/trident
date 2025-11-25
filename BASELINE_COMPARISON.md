# Baseline Comparison Guide

This document explains how to compare TRIDENT with Self-RAG and GraphRAG baselines for token usage and performance evaluation.

## Overview

The baseline comparison system allows you to evaluate:
- **TRIDENT-Pareto**: Pareto-Knapsack mode
- **TRIDENT-SafeCover**: Safe-Cover mode with certificates
- **Self-RAG**: Self-retrieval augmented generation
- **GraphRAG**: Graph-based retrieval augmented generation

All systems use the same LLM, retrieval setup, and dataset for fair comparison.

## Quick Start

### 1. Run All Baselines

```bash
python eval_compare_baselines.py \
    --data_path data/hotpot_dev_sample.json \
    --output_dir runs/comparison \
    --model meta-llama/Llama-2-7b-hf \
    --device 0 \
    --systems all
```

### 2. Run Specific Systems

```bash
python eval_compare_baselines.py \
    --data_path data/hotpot_dev_sample.json \
    --output_dir runs/comparison \
    --model meta-llama/Llama-2-7b-hf \
    --device 0 \
    --systems trident_pareto,self_rag
```

### 3. Run with Custom Configuration

```bash
python eval_compare_baselines.py \
    --data_path data/hotpot_dev_sample.json \
    --output_dir runs/comparison \
    --config configs/my_config.json \
    --systems all
```

## Command-Line Options

### Required Arguments

- `--data_path`: Path to evaluation data (JSON format)
- `--output_dir`: Directory for saving results

### System Selection

- `--systems`: Comma-separated list of systems to run
  - Options: `trident_pareto`, `trident_safe_cover`, `self_rag`, `graphrag`, or `all`
  - Default: `all`

### Model Configuration

- `--model`: LLM model name (default: `meta-llama/Llama-2-7b-hf`)
- `--device`: CUDA device ID (default: 0, use -1 for CPU)
- `--load_in_8bit`: Enable 8-bit quantization

### Pipeline Configuration

- `--budget_tokens`: Token budget for TRIDENT (default: 2000)
- `--seed`: Random seed for reproducibility (default: 42)

### Baseline-Specific Options

- `--selfrag_use_critic`: Enable critic/verification step in Self-RAG

### Evaluation Options

- `--max_examples`: Maximum examples to evaluate (default: 0 for all)
- `--config`: Path to custom configuration file

## Configuration File

You can create a JSON configuration file to customize all parameters:

```json
{
  "mode": "pareto",
  "safe_cover": {
    "per_facet_alpha": 0.01,
    "token_cap": 2000
  },
  "pareto": {
    "budget": 2000,
    "relaxed_alpha": 0.5,
    "use_vqc": true,
    "use_bwk": true
  },
  "llm": {
    "model_name": "meta-llama/Llama-2-7b-hf",
    "temperature": 0.0,
    "max_new_tokens": 256
  },
  "baselines": {
    "selfrag_k": 8,
    "selfrag_use_critic": false,
    "graphrag_k": 20,
    "graphrag_max_seeds": 10
  }
}
```

## Output Files

The comparison script generates the following files in `output_dir`:

1. **`{system}_results.json`**: Detailed per-example results for each system
   - Contains: query_id, question, prediction, ground_truth, tokens_used, latency_ms, EM, F1, stats

2. **`token_usage_summary.json`**: Summary statistics for all systems
   - Contains: avg_tokens, median_tokens, avg_latency_ms, avg_em, avg_f1, abstention_rate

3. **Console output**: Formatted comparison table

## Example Output

```
====================================================================================================
TOKEN USAGE COMPARISON
====================================================================================================
System                    Avg Tokens   Med Tokens   Avg Latency     EM       F1
----------------------------------------------------------------------------------------------------
trident_pareto            3764.2       3621.0       21869.4         0.654    0.721
trident_safe_cover        2891.5       2756.0       18234.7         0.612    0.689
self_rag                  1245.8       1198.0       8432.1          0.589    0.651
graphrag                  4521.3       4387.0       28765.2         0.671    0.738
====================================================================================================
```

## Using the Original Eval Script

The original `eval_complete_runnable.py` script continues to work as before for running TRIDENT only:

```bash
python eval_complete_runnable.py \
    --worker \
    --data_path data/hotpot_dev_sample.json \
    --output_dir runs/trident_only \
    --mode pareto \
    --budget_tokens 2000
```

## Baseline System Details

### Self-RAG

Self-RAG implements a retrieval-gated approach:
1. **Retrieval Decision**: LLM decides whether to retrieve documents
2. **Answer Generation**: Uses retrieved documents (if any) to answer
3. **Optional Critic**: Verifies answer support (with `--selfrag_use_critic`)

**Key Parameters**:
- `selfrag_k`: Number of documents to retrieve (default: 8)
- `selfrag_use_critic`: Enable verification step (default: false)

### GraphRAG

GraphRAG uses graph-based retrieval and summarization:
1. **Graph Construction**: Builds simple graph from retrieved documents
2. **Node Selection**: LLM selects relevant nodes
3. **Subgraph Expansion**: Expands around selected nodes
4. **Community Summarization**: LLM summarizes each subgraph
5. **Answer Generation**: Uses summaries to answer

**Key Parameters**:
- `graphrag_k`: Initial documents to retrieve (default: 20)
- `graphrag_topk_nodes`: Candidate nodes to consider (default: 20)
- `graphrag_max_seeds`: Maximum seed nodes (default: 10)
- `graphrag_max_hops`: Subgraph expansion hops (default: 2)

### TRIDENT

TRIDENT provides two modes for comparison:
- **Pareto Mode**: Optimizes utility vs. token budget tradeoff
- **Safe-Cover Mode**: Provides statistical certificates for facet coverage

Both modes use facet decomposition, two-stage scoring, and optional VQC/BwK.

## Programmatic Usage

You can also use the baseline systems programmatically:

```python
from trident.llm_interface import LLMInterface
from trident.llm_instrumentation import InstrumentedLLM
from baselines import SelfRAGSystem, GraphRAGSystem

# Initialize LLM
llm = LLMInterface(model_name="meta-llama/Llama-2-7b-hf")
instrumented_llm = InstrumentedLLM(llm)

# Initialize Self-RAG
self_rag = SelfRAGSystem(
    llm=instrumented_llm,
    retriever=my_retriever,
    k=8,
    use_critic=False
)

# Run inference
result = self_rag.answer(
    question="What is the capital of France?",
    context=provided_context  # optional
)

print(f"Answer: {result['answer']}")
print(f"Tokens used: {result['tokens_used']}")
print(f"Latency: {result['latency_ms']:.2f}ms")
```

## Notes

- All systems use the same LLM and retrieval backend for fair comparison
- Token counting is consistent across all systems using the instrumentation layer
- TRIDENT's token counts include all LLM calls (facet decomposition, VQC, answer generation)
- Baselines' token counts include all their respective LLM calls
- For datasets with provided context (like HotpotQA), retrieval is skipped and context is used directly

## Troubleshooting

### Import Errors

If you get import errors, make sure you're running from the trident root directory:

```bash
cd /path/to/trident
python eval_compare_baselines.py ...
```

### CUDA Out of Memory

Try these solutions:
- Use 8-bit quantization: `--load_in_8bit`
- Reduce batch size in config
- Use a smaller model
- Use CPU: `--device -1`

### Retriever Not Found

Make sure your data file includes context or you have a corpus configured. For HotpotQA-style datasets with provided context, the script will use that directly.
