# TRIDENT CLI Commands Reference

This document lists all available CLI commands for running TRIDENT experiments, separated by what currently exists in the code vs. what is planned.

---

## Currently Available Commands

### 1. Basic TRIDENT Pipeline (`trident/cli.py`)

```bash
# Run TRIDENT on a dataset
python -m trident.cli dataset \
    --dataset hotpot_qa \
    --split validation \
    --limit 10 \
    --mode safe_cover

# Options:
#   --dataset: Dataset name (hotpot_qa, natural_questions, trivia_qa)
#   --split: Dataset split (default: validation)
#   --limit: Maximum number of queries (default: 10)
#   --mode: Operating mode (safe_cover, pareto, both)
```

### 2. Full Baseline Evaluation (`experiments/eval_full_baselines.py`)

```bash
# Run Self-RAG on HotpotQA
python experiments/eval_full_baselines.py \
    --data_path data/hotpotqa_dev.json \
    --output_dir results/selfrag_hotpotqa \
    --baselines selfrag \
    --max_samples 100

# Run Self-RAG on 2WikiMultiHopQA (auto-detected from path)
python experiments/eval_full_baselines.py \
    --data_path data/2wikimultihop_dev.json \
    --output_dir results/selfrag_2wiki \
    --baselines selfrag

# Run Self-RAG on MuSiQue
python experiments/eval_full_baselines.py \
    --data_path data/musique_ans_v1.0_dev.jsonl \
    --output_dir results/selfrag_musique \
    --baselines selfrag

# Explicit dataset specification
python experiments/eval_full_baselines.py \
    --data_path data/my_data.json \
    --output_dir results/baselines \
    --dataset 2wiki \
    --baselines selfrag

# Run multiple baselines
python experiments/eval_full_baselines.py \
    --data_path data/hotpotqa_dev.json \
    --output_dir results/all_baselines \
    --baselines selfrag ketrag_reimpl vanillarag hipporag

# Self-RAG with custom parameters
python experiments/eval_full_baselines.py \
    --data_path data/hotpotqa_dev.json \
    --output_dir results/selfrag_custom \
    --baselines selfrag \
    --selfrag_model selfrag/selfrag_llama2_7b \
    --selfrag_max_tokens 100 \
    --selfrag_mode adaptive_retrieval \
    --selfrag_ndocs 10 \
    --selfrag_gpu_memory_utilization 0.5 \
    --local_llm_device cuda:0

# KET-RAG official (requires precomputed contexts)
python experiments/eval_full_baselines.py \
    --data_path data/hotpotqa_dev.json \
    --output_dir results/ketrag_official \
    --baselines ketrag_official \
    --ketrag_context_file KET-RAG/ragtest-hotpot/output/ragtest-hotpot-keyword-0.5.json

# GraphRAG (requires precomputed index)
python experiments/eval_full_baselines.py \
    --data_path data/hotpotqa_dev.json \
    --output_dir results/graphrag \
    --baselines graphrag \
    --graphrag_index_dir graphrag/output/

# Options:
#   --baselines: selfrag, ketrag_reimpl, ketrag_official, vanillarag, hipporag, graphrag, all
#   --dataset: hotpotqa, musique, 2wiki (auto-detected if not specified)
#   --max_samples: Limit number of samples for testing
#   --use_local_llm: Use local HuggingFace LLM instead of OpenAI
#   --local_llm_model: HuggingFace model name (default: Qwen/Qwen2.5-7B-Instruct)
#   --local_llm_device: Device (cuda:0, cuda:1, cpu)
```

### 3. Complete Experiment Runner (`experiments/eval_complete_runnable.py`)

```bash
# TRIDENT Safe-Cover mode
python experiments/eval_complete_runnable.py \
    --data_path data/hotpotqa_dev.json \
    --output_dir results/trident_safecover \
    --mode safe_cover \
    --model meta-llama/Llama-2-7b-hf \
    --budget_tokens 2000 \
    --device 0

# TRIDENT Pareto mode
python experiments/eval_complete_runnable.py \
    --data_path data/hotpotqa_dev.json \
    --output_dir results/trident_pareto \
    --mode pareto \
    --model meta-llama/Llama-2-7b-hf \
    --budget_tokens 1500 \
    --device 0

# With config family
python experiments/eval_complete_runnable.py \
    --data_path data/hotpotqa_dev.json \
    --output_dir results/trident_pareto_cheap \
    --config_family pareto_cheap_1500 \
    --device 0

# Self-RAG via complete runner
python experiments/eval_complete_runnable.py \
    --data_path data/hotpotqa_dev.json \
    --output_dir results/selfrag_complete \
    --mode self_rag \
    --config_family selfrag_base \
    --device 0

# Multi-GPU parallel mode
python experiments/eval_complete_runnable.py \
    --data_path data/hotpotqa_dev.json \
    --output_dir results/parallel \
    --mode safe_cover \
    --num_gpus 4 \
    --shard_size 100

# 2WikiMultiHopQA (auto-detected)
python experiments/eval_complete_runnable.py \
    --data_path data/2wikimultihop_dev.json \
    --output_dir results/2wiki_trident \
    --mode safe_cover \
    --device 0

# Options:
#   --mode: safe_cover, pareto, both, self_rag, graphrag, ketrag
#   --config_family: Named config (pareto_cheap_1500, safe_cover_equal_2500, selfrag_base, etc.)
#   --model: HuggingFace model name
#   --budget_tokens: Token budget (legacy, prefer --config_family)
#   --num_gpus: Number of GPUs for parallel processing
#   --shard_size: Examples per shard for multi-GPU mode
#   --limit: Limit number of examples
#   --dataset: Dataset name (auto-detected from path)
```

---

## Configuration Families Available

### Pareto Mode Configs
```
pareto_cheap_1500    # budget=1500, max_units=8, no VQC/BwK
pareto_cheap_2000    # budget=2000, max_units=10, no VQC/BwK
pareto_mid_2500      # budget=2500, max_units=12, with VQC/BwK
pareto_match_400     # Low-cost variants for ablation
pareto_match_500
pareto_match_650
pareto_match_800
pareto_match_1000
pareto_match_1100
pareto_match_1300
pareto_match_1500
```

### Safe-Cover Mode Configs
```
safe_cover_loose_4000   # α=0.05, max_tokens=4000, max_units=16
safe_cover_equal_2500   # α=0.02, max_tokens=2500, max_units=12
safe_cover_equal_2000   # α=0.02, max_tokens=2000, max_units=10
```

### Self-RAG Configs
```
selfrag_base     # k=8, no critic, no oracle context
selfrag_strong   # k=16, with critic, no oracle context
selfrag_oracle   # k=8, no critic, with gold context (upper bound)
```

---

## Dataset Paths and Formats

### HotpotQA
```bash
# JSON format (list of examples)
data/hotpotqa_dev.json
data/hotpotqa_dev_shards/shard_0.json

# JSONL format (one example per line)
data/hotpotqa_dev.jsonl
```

### 2WikiMultiHopQA
```bash
# Auto-detected from path containing "2wiki" or "wikimultihop"
data/2wikimultihop_dev.json
data/2wiki_dev.json
data/wikimultihop_dev.json

# Or explicit dataset specification
--dataset 2wiki
```

### MuSiQue
```bash
# JSONL format (auto-detected from path containing "musique")
data/musique_ans_v1.0_dev.jsonl
musique/data/musique_ans_v1.0_dev.jsonl

# Or explicit dataset specification
--dataset musique
```

---

## Output Files

### Baseline Evaluation Output
```
results/
├── {baseline}_results.jsonl    # Per-example results
├── {baseline}_summary.json     # Aggregate metrics
└── combined_summary.json       # Cross-baseline comparison
```

### TRIDENT Evaluation Output
```
results/
├── results.json               # All results with config
├── shards/                    # For multi-GPU mode
│   └── shard_X_Y.json
└── results/                   # Per-shard results
    └── shard_X_Y/
        └── results.json
```

---

## Environment Variables

```bash
# Required for OpenAI-based baselines (KET-RAG, VanillaRAG, HippoRAG)
export OPENAI_API_KEY="your-api-key"

# Required for Self-RAG model download
export HF_TOKEN="your-huggingface-token"

# Optional: HuggingFace cache directory
export HF_CACHE_DIR="/path/to/cache"

# Optional: CUDA device selection for multi-process
export CUDA_VISIBLE_DEVICES="0,1,2,3"
```

---

## Quick Start Examples

### Run Self-RAG on all supported datasets

```bash
# HotpotQA
python experiments/eval_full_baselines.py \
    --data_path data/hotpotqa_dev.json \
    --output_dir results/selfrag/hotpotqa \
    --baselines selfrag \
    --max_samples 500

# 2WikiMultiHopQA
python experiments/eval_full_baselines.py \
    --data_path data/2wikimultihop_dev.json \
    --output_dir results/selfrag/2wiki \
    --baselines selfrag \
    --max_samples 500

# MuSiQue
python experiments/eval_full_baselines.py \
    --data_path data/musique_ans_v1.0_dev.jsonl \
    --output_dir results/selfrag/musique \
    --baselines selfrag \
    --max_samples 500
```

### Compare TRIDENT vs Baselines

```bash
# TRIDENT Safe-Cover
python experiments/eval_complete_runnable.py \
    --data_path data/hotpotqa_dev.json \
    --output_dir results/comparison/trident \
    --mode safe_cover \
    --config_family safe_cover_equal_2000

# Self-RAG baseline
python experiments/eval_full_baselines.py \
    --data_path data/hotpotqa_dev.json \
    --output_dir results/comparison/selfrag \
    --baselines selfrag

# VanillaRAG baseline
python experiments/eval_full_baselines.py \
    --data_path data/hotpotqa_dev.json \
    --output_dir results/comparison/vanillarag \
    --baselines vanillarag
```
