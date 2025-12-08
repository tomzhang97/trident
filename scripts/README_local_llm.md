# Using Local LLMs with KET-RAG and GraphRAG

This guide explains how to create context files for KET-RAG and indexes for GraphRAG using local LLMs instead of OpenAI.

## Prerequisites

1. **vLLM Server** (for LLM inference):
```bash
pip install vllm
```

2. **Sentence Transformers** (for local embeddings):
```bash
pip install sentence-transformers
```

## Step 1: Start a vLLM Server

Start a vLLM server with your preferred model:

```bash
# For Llama-3-8B
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --port 8000 \
    --gpu-memory-utilization 0.9

# For Qwen2.5-7B
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port 8000 \
    --gpu-memory-utilization 0.9
```

The server will be available at `http://localhost:8000/v1`.

## Step 2: Create KET-RAG Context Files

**Prerequisites**: You need pre-built GraphRAG indexes (parquet files) in the KET-RAG format.

```bash
# Create context using keyword-based retrieval
python scripts/create_ketrag_context_local.py \
    --root_path external_baselines/KET-RAG/ragtest-hotpot \
    --strategy keyword \
    --budget 0.5 \
    --embedding_model sentence-transformers/all-MiniLM-L6-v2 \
    --device cuda:0

# Create context using text similarity
python scripts/create_ketrag_context_local.py \
    --root_path external_baselines/KET-RAG/ragtest-hotpot \
    --strategy text \
    --budget 0.5 \
    --embedding_model sentence-transformers/all-MiniLM-L6-v2 \
    --device cuda:0
```

### Arguments:
- `--root_path`: Path to KET-RAG data directory (contains `output/` with parquet files)
- `--strategy`: Context building strategy (`keyword`, `text`, or `none`)
- `--budget`: Budget for graph context (0.0-1.0, where 0.5 = 50% graph, 50% text)
- `--embedding_model`: HuggingFace embedding model name
- `--device`: Device for embeddings (`cuda:0`, `cuda:1`, `cpu`)
- `--qa_file`: Optional custom QA pairs file
- `--output`: Optional custom output path

## Step 3: Create GraphRAG Index

```bash
# Generate settings and run indexing
python scripts/create_graphrag_index_local.py \
    --input_dir /path/to/documents \
    --output_dir /path/to/graphrag_index \
    --llm_api_base http://localhost:8000/v1 \
    --llm_model meta-llama/Meta-Llama-3-8B-Instruct \
    --use_local_embeddings \
    --embedding_model sentence-transformers/all-MiniLM-L6-v2

# Or just generate settings.yaml without running
python scripts/create_graphrag_index_local.py \
    --input_dir /path/to/documents \
    --output_dir /path/to/graphrag_index \
    --llm_api_base http://localhost:8000/v1 \
    --llm_model meta-llama/Meta-Llama-3-8B-Instruct \
    --settings_only
```

### Arguments:
- `--input_dir`: Directory containing input documents (.txt files)
- `--output_dir`: Output directory for the index
- `--llm_api_base`: vLLM server URL (e.g., `http://localhost:8000/v1`)
- `--llm_model`: Model name used in vLLM server
- `--use_local_embeddings`: Use sentence-transformers instead of API
- `--embedding_model`: HuggingFace embedding model
- `--chunk_size`: Document chunk size (default: 1200)
- `--max_tokens`: Max tokens for LLM (default: 4000)
- `--settings_only`: Only generate settings.yaml

## Step 4: Run Experiments with Local LLM

After creating contexts/indexes, run the evaluation:

```bash
# KET-RAG evaluation
python experiments/eval_full_baselines.py \
    --baselines ketrag_official \
    --ketrag_context_file external_baselines/KET-RAG/ragtest-hotpot/output/ragtest-hotpot-keyword-0.5.json \
    --data_path runs/_shards/validation_0_99.json \
    --use_local_llm \
    --local_llm_model meta-llama/Meta-Llama-3-8B-Instruct \
    --local_llm_device cuda:0 \
    --output_dir results/full_baselines

# GraphRAG evaluation
python experiments/eval_full_baselines.py \
    --baselines graphrag \
    --graphrag_context_file /path/to/graphrag_contexts.json \
    --data_path runs/_shards/validation_0_99.json \
    --use_local_llm \
    --local_llm_model meta-llama/Meta-Llama-3-8B-Instruct \
    --local_llm_device cuda:0 \
    --output_dir results/full_baselines
```

## Supported Embedding Models

Any sentence-transformers model can be used:
- `sentence-transformers/all-MiniLM-L6-v2` (384 dim, fast)
- `sentence-transformers/all-mpnet-base-v2` (768 dim, better quality)
- `BAAI/bge-small-en-v1.5` (384 dim)
- `BAAI/bge-base-en-v1.5` (768 dim)
- `BAAI/bge-large-en-v1.5` (1024 dim)

## Supported LLM Models

Any model supported by vLLM:
- `meta-llama/Meta-Llama-3-8B-Instruct`
- `meta-llama/Meta-Llama-3-70B-Instruct`
- `Qwen/Qwen2.5-7B-Instruct`
- `Qwen/Qwen2.5-14B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`

## Troubleshooting

### vLLM Server Issues
- Ensure the model fits in GPU memory
- Use `--gpu-memory-utilization 0.8` to reduce memory usage
- Use `--tensor-parallel-size 2` for multi-GPU

### Embedding Dimension Mismatch
- Ensure the embedding model matches what was used during indexing
- Check that parquet files contain the expected embedding dimensions

### Out of Memory
- Use `--load_in_8bit` or `--load_in_4bit` for local LLM evaluation
- Reduce batch sizes in the scripts
