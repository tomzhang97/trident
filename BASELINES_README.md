# TRIDENT Baseline Comparison Guide

This guide explains how to run and compare TRIDENT with baseline systems (Self-RAG and GraphRAG) for multi-hop question answering.

## Overview

The codebase now supports running multiple retrieval-augmented generation systems:

1. **TRIDENT (Safe-Cover mode)**: Uses certified passage selection with statistical guarantees
2. **TRIDENT (Pareto mode)**: Budget-constrained utility maximization with optional VQC and BwK
3. **Self-RAG**: Baseline with retrieval gating and optional critic
4. **GraphRAG**: Graph-based retrieval with community summarization

All systems are instrumented to track token usage and latency for fair comparison.

## Quick Start

### Running Individual Systems

#### 1. TRIDENT (Pareto Mode)
```bash
python eval_complete_runnable.py \
  --worker \
  --data_path data/hotpot_dev_sample.json \
  --output_dir runs/trident_pareto \
  --mode pareto \
  --config configs/trident_pareto.json \
  --model meta-llama/Llama-2-7b-hf \
  --device 0
```

#### 2. TRIDENT (Safe-Cover Mode)
```bash
python eval_complete_runnable.py \
  --worker \
  --data_path data/hotpot_dev_sample.json \
  --output_dir runs/trident_safe_cover \
  --mode safe_cover \
  --config configs/trident_safe_cover.json \
  --model meta-llama/Llama-2-7b-hf \
  --device 0
```

#### 3. Self-RAG
```bash
python eval_complete_runnable.py \
  --worker \
  --data_path data/hotpot_dev_sample.json \
  --output_dir runs/self_rag \
  --mode self_rag \
  --config configs/self_rag.json \
  --model meta-llama/Llama-2-7b-hf \
  --device 0 \
  --selfrag_k 8 \
  --selfrag_use_critic
```

#### 4. GraphRAG
```bash
python eval_complete_runnable.py \
  --worker \
  --data_path data/hotpot_dev_sample.json \
  --output_dir runs/graphrag \
  --mode graphrag \
  --config configs/graphrag.json \
  --model meta-llama/Llama-2-7b-hf \
  --device 0 \
  --graphrag_k 20 \
  --graphrag_max_seeds 10
```

### Running All Baselines for Comparison

Use the `eval_compare_baselines.py` script to run all systems and generate a comparison table:

```bash
python eval_compare_baselines.py \
  --data_path data/hotpot_dev_sample.json \
  --output_dir runs/comparison \
  --systems all \
  --model meta-llama/Llama-2-7b-hf \
  --device 0 \
  --max_examples 100
```

Or specify specific systems:
```bash
python eval_compare_baselines.py \
  --data_path data/hotpot_dev_sample.json \
  --output_dir runs/comparison \
  --systems trident_pareto,self_rag,graphrag \
  --model meta-llama/Llama-2-7b-hf \
  --device 0
```

## Command-Line Options

### Core Arguments
- `--data_path`: Path to evaluation data (JSON format)
- `--output_dir`: Directory for results
- `--mode`: System mode (safe_cover, pareto, both, self_rag, graphrag)
- `--model`: LLM model name (default: meta-llama/Llama-2-7b-hf)
- `--device`: CUDA device ID (-1 for CPU)
- `--config`: Path to configuration file (optional)

### TRIDENT-Specific Options
- `--budget_tokens`: Token budget for TRIDENT (default: 2000)
- `--temperature`: Sampling temperature (default: 0.0)
- `--max_new_tokens`: Maximum tokens to generate (default: 256)

### Self-RAG Options
- `--selfrag_k`: Number of documents to retrieve (default: 8)
- `--selfrag_use_critic`: Enable critic/verification step

### GraphRAG Options
- `--graphrag_k`: Number of documents to retrieve (default: 20)
- `--graphrag_topk_nodes`: Top-k candidate nodes (default: 20)
- `--graphrag_max_seeds`: Maximum seed nodes to select (default: 10)

## Configuration Files

Pre-configured JSON files are available in `configs/`:

- `trident_pareto.json` - TRIDENT with Pareto-Knapsack optimization
- `trident_safe_cover.json` - TRIDENT with Safe-Cover certificates
- `self_rag.json` - Self-RAG baseline
- `graphrag.json` - GraphRAG baseline
- `baseline_comparison.json` - Default settings for all systems

### Example Configuration Structure

```json
{
  "mode": "pareto",
  "pareto": {
    "budget": 2000,
    "relaxed_alpha": 0.5,
    "use_vqc": true,
    "use_bwk": true,
    "max_vqc_iterations": 3
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

## Output Format

### Individual System Results

Each system produces a JSON file with detailed results:

```json
{
  "config": {...},
  "results": [
    {
      "query_id": "5a8b57f25542995d1e6f1371",
      "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
      "prediction": "yes",
      "ground_truth": ["yes"],
      "tokens_used": 1234,
      "latency_ms": 4567.8,
      "abstained": false,
      "mode": "pareto",
      "metrics": {...}
    },
    ...
  ],
  "summary": {
    "num_examples": 100,
    "avg_em": 0.67,
    "avg_f1": 0.74,
    "avg_tokens": 1834.5,
    "avg_latency_ms": 5234.2
  }
}
```

### Comparison Summary

When using `eval_compare_baselines.py`, a comparison table is generated:

```
================================================================================
TOKEN USAGE COMPARISON
================================================================================
System                    Avg Tokens   Med Tokens   Avg Latency     EM       F1
--------------------------------------------------------------------------------
trident_pareto            1834.5       1756.0       5234.2          0.670    0.742
trident_safe_cover        1456.2       1398.0       4123.5          0.645    0.718
self_rag                  892.3        854.0        2345.6          0.612    0.689
graphrag                  2134.7       2089.0       6789.1          0.634    0.701
================================================================================
```

## Token Usage Tracking

All systems use the same instrumentation to ensure fair comparison:

### InstrumentedLLM Wrapper
- Tracks prompt tokens, completion tokens, and total tokens
- Measures latency for each LLM call
- Accumulates statistics across all calls for a single query

### QueryStats Object
Each query accumulates:
- `total_tokens`: Total tokens used (prompt + completion)
- `total_prompt_tokens`: Total prompt tokens
- `total_completion_tokens`: Total completion tokens
- `num_calls`: Number of LLM API calls
- `latency_ms`: Total latency in milliseconds

## Baseline System Details

### Self-RAG
Self-RAG implements a retrieve-then-read pipeline with:
1. **Retrieval Gate**: LLM decides whether to retrieve documents
2. **Answer Generation**: Generates answer based on retrieved documents
3. **Optional Critic**: Verifies answer support (if `--selfrag_use_critic` is enabled)

**Prompts**:
- Retrieval decision prompt
- Answer generation prompt
- Critic/verification prompt

### GraphRAG
GraphRAG implements graph-based retrieval with:
1. **Graph Construction**: Builds simple graph from retrieved documents
2. **Node Selection**: LLM selects relevant seed nodes
3. **Subgraph Expansion**: Expands to neighboring nodes
4. **Community Summarization**: LLM summarizes each subgraph
5. **Answer Generation**: Generates answer from summaries

**Prompts**:
- Node selection prompt
- Community summary prompt
- Answer generation prompt

## Extending with New Baselines

To add a new baseline system:

1. **Create System Class** in `baselines/`:
```python
class MyRAGSystem:
    def __init__(self, llm: InstrumentedLLM, retriever, **kwargs):
        self.llm = llm
        self.retriever = retriever

    def answer(self, question: str, context=None, **kwargs) -> Dict[str, Any]:
        qstats = QueryStats()

        # Your RAG logic here
        # Use timed_llm_call() for each LLM call

        return {
            "answer": answer,
            "tokens_used": qstats.total_tokens,
            "latency_ms": qstats.latency_ms,
            "abstained": False,
            "mode": "my_rag",
            "stats": {...}
        }
```

2. **Update Configuration** in `trident/config.py`:
```python
@dataclass
class BaselineConfig:
    # Add your baseline settings
    myrag_param1: int = 10
    myrag_param2: bool = False
```

3. **Add to eval_complete_runnable.py**:
```python
elif mode == "my_rag":
    return MyRAGSystem(
        llm=self.instrumented_llm,
        retriever=self.retriever,
        param1=self.config.baselines.myrag_param1,
        param2=self.config.baselines.myrag_param2
    )
```

4. **Update argparse** to accept the new mode.

## Data Format

The evaluation scripts expect JSON data in the following format:

```json
[
  {
    "_id": "5a8b57f25542995d1e6f1371",
    "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
    "answer": "yes",
    "context": [
      ["Scott Derrickson", ["Scott Derrickson (born July 16, 1966) is an American director..."]],
      ["Ed Wood", ["Edward Davis Wood Jr. (October 10, 1924 – December 10, 1978) was an American filmmaker..."]]
    ],
    "supporting_facts": [
      ["Scott Derrickson", 0],
      ["Ed Wood", 0]
    ]
  },
  ...
]
```

**Required fields**:
- `_id` or `id`: Unique query identifier
- `question`: The question text
- `answer`: Ground truth answer (string or list of strings)

**Optional fields**:
- `context`: Pre-provided context passages (for datasets like HotpotQA)
- `supporting_facts`: Annotated supporting facts for facet mining

## Troubleshooting

### ImportError: No module named 'transformers'
```bash
pip install transformers torch sentence-transformers
```

### CUDA Out of Memory
- Reduce `--batch_size` or use `--load_in_8bit`
- Reduce `--budget_tokens` for TRIDENT
- Reduce `--selfrag_k` or `--graphrag_k` for baselines

### Model Download Issues
Ensure you have access to the model:
```bash
huggingface-cli login
```

### Slow Execution
- Use `--load_in_8bit` for faster inference
- Reduce `--max_examples` for quick tests
- Consider using vLLM for batched serving

## Citation

If you use these baseline implementations, please cite the original papers:

**TRIDENT**:
```
[Your TRIDENT paper citation]
```

**Self-RAG**:
```
@inproceedings{asai2023selfrag,
  title={Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection},
  author={Asai, Akari and others},
  booktitle={ICLR},
  year={2024}
}
```

**GraphRAG**:
```
@article{edge2024graphrag,
  title={From Local to Global: A Graph RAG Approach to Query-Focused Summarization},
  author={Edge, Darren and others},
  journal={arXiv preprint arXiv:2404.16130},
  year={2024}
}
```

## Contact

For issues or questions about the baseline implementations, please open an issue on GitHub.
