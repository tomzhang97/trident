# Using Local LLMs with KET-RAG and GraphRAG

This guide explains how to set up and run KET-RAG and GraphRAG with local LLMs.

## Prerequisites

1. **vLLM Server** (for LLM inference):
```bash
pip install vllm
```

2. **Sentence Transformers** (for local embeddings):
```bash
pip install sentence-transformers
```

3. **GraphRAG** (included in KET-RAG folder):
```bash
pip install poetry
cd KET-RAG && poetry install
```

---

## KET-RAG Complete Workflow

Following the original KET-RAG README structure:

### Step 1: Start vLLM Server

```bash
# For Llama-3-8B
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --port 8000 \
    --gpu-memory-utilization 0.9

# For embedding server (optional, can use sentence-transformers instead)
python -m vllm.entrypoints.openai.api_server \
    --model BAAI/bge-base-en-v1.5 \
    --port 8001 \
    --gpu-memory-utilization 0.3
```

### Step 2: Initialize the Project

**Option A: Using setup script (recommended)**
```bash
python scripts/setup_ketrag_local.py \
    --root_path ragtest-musique \
    --llm_api_base http://localhost:8000/v1 \
    --llm_model meta-llama/Meta-Llama-3-8B-Instruct \
    --use_local_embeddings \
    --local_embedding_model sentence-transformers/all-MiniLM-L6-v2
```

**Option B: Manual setup**
```bash
# Initialize project structure
python -m graphrag init --root ragtest-musique/

# Then edit settings.yaml to use local LLM (see Settings section below)
```

### Step 3: Prepare Your Data

Place your input documents in `ragtest-musique/input/`:
```
ragtest-musique/
├── input/
│   ├── doc1.txt
│   ├── doc2.txt
│   └── ...
├── qa-pairs/
│   └── qa-pairs.json  # Format: [{"id": "...", "question": "...", "answer": "..."}]
└── settings.yaml
```

### Step 4: Tune Prompts (Optional)

**Note**: This step is optional. Default prompts work well for most use cases.

```bash
# Use the standalone script (recommended - avoids typer CLI issues)
python scripts/prompt_tune_ketrag.py ragtest-musique/

# Or with options
python scripts/prompt_tune_ketrag.py ragtest-musique/ \
    --domain "multi-hop question answering" \
    --selection-method random \
    --limit 15
```

### Step 5: Build the Index

```bash
# Make sure you're using KET-RAG's graphrag (not the newer version)
cd external_baselines/KET-RAG
python -m graphrag index --root ../../ragtest-musique/

# Or set PYTHONPATH explicitly
PYTHONPATH=external_baselines/KET-RAG python -m graphrag index --root ragtest-musique/
```

This creates parquet files in `ragtest-musique/output/`.

**Note**: If you have both KET-RAG and the newer graphrag installed, ensure you're using the correct version. KET-RAG uses graphrag 0.4.1.

### Step 6: Generate Context

**Using local embeddings:**
```bash
python scripts/create_ketrag_context_local.py \
    --root_path ragtest-musique \
    --strategy keyword \
    --budget 0.5 \
    --embedding_model sentence-transformers/all-MiniLM-L6-v2 \
    --device cuda:0
```

**Using original script (requires OpenAI API):**
```bash
export GRAPHRAG_API_KEY=your_key
python KET-RAG/indexing_sket/create_context.py ragtest-musique keyword 0.5
```

#### Context Parameters:
- **strategy**: `keyword` (recommended), `text`, or `none`
- **budget** (θ): 0.0-1.0 (0.5 = 50% graph context, 50% text context)

### Step 7: Generate Answers

**Using local LLM (HuggingFace):**
```bash
python scripts/llm_answer_local.py ragtest-musique \
    --use_local_llm \
    --local_llm_model meta-llama/Meta-Llama-3-8B-Instruct \
    --device cuda:0
```

**Using vLLM server:**
```bash
python scripts/llm_answer_local.py ragtest-musique \
    --api_base http://localhost:8000/v1 \
    --model meta-llama/Meta-Llama-3-8B-Instruct
```

**Using OpenAI API (original):**
```bash
export GRAPHRAG_API_KEY=your_key
python KET-RAG/indexing_sket/llm_answer.py ragtest-musique
```

### Step 8: Run Evaluation

```bash
python experiments/eval_full_baselines.py \
    --baselines ketrag_official \
    --ketrag_context_file ragtest-musique/output/ragtest-musique-keyword-0.5.json \
    --data_path runs/_shards/validation_0_99.json \
    --use_local_llm \
    --local_llm_model meta-llama/Meta-Llama-3-8B-Instruct \
    --local_llm_device cuda:0 \
    --output_dir results/full_baselines
```

---

## GraphRAG Complete Workflow

### Step 1: Start vLLM Server

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --port 8000
```

### Step 2: Setup and Index

```bash
python scripts/create_graphrag_index_local.py \
    --input_dir /path/to/documents \
    --output_dir /path/to/graphrag_index \
    --llm_api_base http://localhost:8000/v1 \
    --llm_model meta-llama/Meta-Llama-3-8B-Instruct \
    --use_local_embeddings \
    --embedding_model sentence-transformers/all-MiniLM-L6-v2
```

### Step 3: Run Evaluation

```bash
python experiments/eval_full_baselines.py \
    --baselines graphrag \
    --graphrag_context_file /path/to/graphrag_contexts.json \
    --data_path runs/_shards/validation_0_99.json \
    --use_local_llm \
    --local_llm_model meta-llama/Meta-Llama-3-8B-Instruct \
    --output_dir results/full_baselines
```

---

## Settings.yaml for Local LLM

Example `settings.yaml` for local LLM with vLLM server:

```yaml
models:
  default_chat:
    type: openai_chat
    model_provider: openai
    auth_type: api_key
    api_key: EMPTY
    model: meta-llama/Meta-Llama-3-8B-Instruct
    api_base: http://localhost:8000/v1
    model_supports_json: true
    concurrent_requests: 25
    async_mode: threaded
    max_retries: 10

  default_embedding:
    type: sentence_transformers_embedding
    model: sentence-transformers/all-MiniLM-L6-v2
    # Or use API:
    # type: openai_embedding
    # api_base: http://localhost:8001/v1
    # model: BAAI/bge-base-en-v1.5
    # api_key: EMPTY

input:
  storage:
    type: file
    base_dir: "input"
  file_type: text

chunks:
  size: 1200
  overlap: 100

output:
  type: file
  base_dir: "output"
```

---

## Supported Models

### LLM Models (via vLLM)
- `meta-llama/Meta-Llama-3-8B-Instruct`
- `meta-llama/Meta-Llama-3-70B-Instruct`
- `Qwen/Qwen2.5-7B-Instruct`
- `Qwen/Qwen2.5-14B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`

### Embedding Models (sentence-transformers)
- `sentence-transformers/all-MiniLM-L6-v2` (384 dim, fast)
- `sentence-transformers/all-mpnet-base-v2` (768 dim, better quality)
- `BAAI/bge-small-en-v1.5` (384 dim)
- `BAAI/bge-base-en-v1.5` (768 dim)
- `BAAI/bge-large-en-v1.5` (1024 dim)

---

## Troubleshooting

### vLLM Server Issues
- **Out of memory**: Use `--gpu-memory-utilization 0.8` or smaller model
- **Multi-GPU**: Use `--tensor-parallel-size 2`
- **Connection refused**: Ensure server is running and port is correct

### Embedding Dimension Mismatch
- Ensure embedding model matches what was used during indexing
- Check parquet files for expected dimensions

### Index Build Failures
- Check `ragtest-*/reports/` for error logs
- Ensure prompts exist in `prompts/` directory
- Verify API connection with curl:
  ```bash
  curl http://localhost:8000/v1/models
  ```

### Answer Generation Errors
- Reduce `--max_concurrent` if hitting rate limits
- Check context files exist in `output/` directory
- Verify QA pairs format in `qa-pairs/qa-pairs.json`

---

## File Structure

After running the complete workflow:

```
ragtest-musique/
├── input/                          # Input documents
│   └── *.txt
├── output/                         # Index outputs
│   ├── create_final_entities.parquet
│   ├── create_final_relationships.parquet
│   ├── create_final_community_reports.parquet
│   ├── create_final_text_units.parquet
│   ├── split_text_units.parquet
│   ├── keyword_index.parquet
│   ├── ragtest-musique-keyword-0.5.json    # Context file
│   └── answer-ragtest-musique-keyword-0.5.json  # Answers
├── cache/                          # LLM cache
├── reports/                        # Build reports
├── prompts/                        # Prompt templates
├── qa-pairs/
│   └── qa-pairs.json              # QA pairs for evaluation
├── settings.yaml                   # Configuration
└── .env                           # API keys
```
