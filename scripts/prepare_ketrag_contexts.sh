#!/bin/bash
# Script to prepare KET-RAG contexts for HotpotQA evaluation
#
# This script runs the official KET-RAG pipeline to precompute contexts
# for use with the faithful KET-RAG adapter (FullKETRAGAdapter).
#
# Usage:
#   ./scripts/prepare_ketrag_contexts.sh <hotpot_data.json> <output_dir> [local_llm_url]
#
# Examples:
#   # With OpenAI (default)
#   ./scripts/prepare_ketrag_contexts.sh data/hotpotqa_dev.json ragtest-hotpot
#
#   # With local LLM (requires vLLM server)
#   ./scripts/prepare_ketrag_contexts.sh data/hotpotqa_dev.json ragtest-hotpot http://localhost:8000/v1
#
# Local LLM setup:
#   1. Start vLLM server with OpenAI-compatible API:
#      python -m vllm.entrypoints.openai.api_server \
#        --model Qwen/Qwen2.5-7B-Instruct \
#        --port 8000
#
#   2. Run this script with the server URL

set -e

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <hotpot_data.json> <output_dir> [local_llm_url]"
    echo ""
    echo "Examples:"
    echo "  # With OpenAI (default)"
    echo "  $0 data/hotpotqa_dev.json ragtest-hotpot"
    echo ""
    echo "  # With local LLM (requires vLLM server)"
    echo "  $0 data/hotpotqa_dev.json ragtest-hotpot http://localhost:8000/v1"
    exit 1
fi

HOTPOT_DATA="$1"
OUTPUT_DIR="$2"
LOCAL_LLM_URL="${3:-}"  # Optional

# Get absolute paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
KETRAG_DIR="$PROJECT_ROOT/external_baselines/KET-RAG"
KETRAG_OUTPUT_DIR="$KETRAG_DIR/$OUTPUT_DIR"

echo "=================================================="
echo "Preparing KET-RAG contexts for official adapter"
echo "=================================================="
echo "Input: $HOTPOT_DATA"
echo "Output directory: $OUTPUT_DIR"
echo "KET-RAG location: $KETRAG_DIR"

if [ -n "$LOCAL_LLM_URL" ]; then
    echo "LLM mode: Local (via vLLM server at $LOCAL_LLM_URL)"
else
    echo "LLM mode: OpenAI API"
fi
echo ""

# Check KET-RAG directory exists
if [ ! -d "$KETRAG_DIR" ]; then
    echo "ERROR: KET-RAG directory not found at $KETRAG_DIR"
    echo "Please ensure the KET-RAG repository is cloned in the project root"
    exit 1
fi

# Check poetry is available
if ! command -v poetry &> /dev/null; then
    echo "ERROR: poetry not found. Please install poetry:"
    echo "  curl -sSL https://install.python-poetry.org | python3 -"
    exit 1
fi

# Ensure KET-RAG is a valid Poetry project
if [ ! -f "$KETRAG_DIR/pyproject.toml" ]; then
    echo "ERROR: pyproject.toml not found in $KETRAG_DIR"
    echo "Please ensure the KET-RAG repository is complete"
    exit 1
fi

# Install dependencies if the virtual environment is not set up
echo "[Setup] Installing KET-RAG dependencies via poetry (if needed)..."
cd "$KETRAG_DIR"
poetry install --no-interaction --no-root >/dev/null
# Ensure NLTK stopwords are available for keyword indexing
poetry run python - <<'PY'
import nltk

# Download quietly; will no-op if already present
nltk.download('stopwords', quiet=True)
PY
cd "$PROJECT_ROOT"

# Step 1: Convert HotpotQA to KET-RAG format
echo "[Step 1/4] Converting HotpotQA data to KET-RAG format..."
cd "$PROJECT_ROOT"
python scripts/convert_hotpot_to_ketrag.py "$HOTPOT_DATA" "$KETRAG_OUTPUT_DIR"

# Update settings.yaml for local LLM if specified
if [ -n "$LOCAL_LLM_URL" ]; then
    echo ""
    echo "[Configuring for local LLM...]"
    SETTINGS_FILE="$KETRAG_OUTPUT_DIR/settings.yaml"

    # Backup original settings
    cp "$SETTINGS_FILE" "$SETTINGS_FILE.backup"

    # Update LLM config to use local server
    cat > "$SETTINGS_FILE" << 'EOF'
encoding_model: cl100k_base
skip_workflows: []

llm:
  api_key: "local-key"  # Dummy key for local server
  type: openai_chat
  model: local-model
  model_supports_json: true
  max_tokens: 4000
  request_timeout: 180.0
  api_base: LOCAL_LLM_URL_PLACEHOLDER
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
    api_key: "local-key"
    type: openai_embedding
    model: local-embedding
    api_base: LOCAL_LLM_URL_PLACEHOLDER
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
  file_pattern: ".*\\.txt$"

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
EOF

    # Replace placeholder with actual URL
    sed -i "s|LOCAL_LLM_URL_PLACEHOLDER|$LOCAL_LLM_URL|g" "$SETTINGS_FILE"

    echo "  ✓ Updated settings.yaml for local LLM"
    echo "  Note: Make sure your vLLM server supports embedding model or use OpenAI for embeddings"
fi

# Step 2: Initialize GraphRAG project
echo ""
echo "[Step 2/4] Initializing GraphRAG project..."
cd "$KETRAG_DIR"

# Check if already initialized
if [ -f "$OUTPUT_DIR/prompts/entity_extraction.txt" ]; then
    echo "  Project already initialized, skipping..."
else
    poetry run graphrag init --root "$OUTPUT_DIR/"
fi

# Step 3: Run GraphRAG indexing (this may take a while)
echo ""
echo "[Step 3/4] Running GraphRAG indexing (this may take 10-30 minutes)..."
if [ -n "$LOCAL_LLM_URL" ]; then
    echo "  Using local LLM at: $LOCAL_LLM_URL"
else
    echo "  Using OpenAI API"
    echo "  Set GRAPHRAG_API_KEY environment variable if not already set"
    if [ -z "$GRAPHRAG_API_KEY" ]; then
        echo "  WARNING: GRAPHRAG_API_KEY not set!"
    fi
fi

cd "$KETRAG_DIR"
# Work around a Typer/Click flag parsing bug by calling the indexing function directly
set +e
poetry run python - <<PY
from pathlib import Path

from graphrag.cli.index import index_cli
from graphrag.index.emit.types import TableEmitterType
from graphrag.logging import ReporterType

index_cli(
    root_dir=Path("$OUTPUT_DIR"),
    verbose=False,
    resume=None,
    memprofile=False,
    cache=True,
    reporter=ReporterType.RICH,
    config_filepath=None,
    emit=[TableEmitterType.Parquet],
    dry_run=False,
    skip_validation=False,
    output_dir=None,
)
PY
INDEX_EXIT_CODE=$?
set -e

LOG_FILE="$KETRAG_OUTPUT_DIR/logs/indexing-engine.log"
if [ $INDEX_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "[Error] GraphRAG indexing failed (exit code: $INDEX_EXIT_CODE)"
    if [ -f "$LOG_FILE" ]; then
        echo "Showing last 40 lines of the indexing log: $LOG_FILE"
        echo "--------------------------------------------------"
        tail -n 40 "$LOG_FILE"
        echo "--------------------------------------------------"
    else
        echo "No indexing log found at $LOG_FILE"
    fi
    exit $INDEX_EXIT_CODE
fi

# Step 4: Create contexts with keyword strategy
echo ""
echo "[Step 4/4] Creating contexts with keyword strategy..."
cd "$KETRAG_DIR"
poetry run python indexing_sket/create_context.py "$OUTPUT_DIR/" keyword 0.5

echo ""
echo "=================================================="
echo "KET-RAG context preparation complete!"
echo "=================================================="
echo ""
echo "Context file created at:"
echo "  $KETRAG_OUTPUT_DIR/output/${OUTPUT_DIR}-keyword-0.5.json"
echo ""
echo "You can now run the official KET-RAG adapter:"
echo "  cd $PROJECT_ROOT"
echo "  python experiments/eval_full_baselines.py \\"
echo "    --data_path $HOTPOT_DATA \\"
echo "    --baselines ketrag_official \\"
echo "    --ketrag_context_file $KETRAG_OUTPUT_DIR/output/${OUTPUT_DIR}-keyword-0.5.json"
echo ""
