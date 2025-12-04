#!/bin/bash
# Script to prepare KET-RAG contexts for HotpotQA evaluation
#
# This script runs the official KET-RAG pipeline to precompute contexts
# for use with the faithful KET-RAG adapter (FullKETRAGAdapter).
#
# Usage:
#   ./scripts/prepare_ketrag_contexts.sh <hotpot_data.jsonl> <output_dir>
#
# Example:
#   ./scripts/prepare_ketrag_contexts.sh data/hotpotqa_dev_shards/shard_0.jsonl ragtest-hotpot
#
# This will create:
#   KET-RAG/ragtest-hotpot/qa-pairs/qa-pairs.json
#   KET-RAG/ragtest-hotpot/output/ragtest-hotpot-keyword-0.5.json
#
# The output JSON can then be used with:
#   python eval_full_baselines.py \
#     --baselines ketrag_official \
#     --ketrag_context_file KET-RAG/ragtest-hotpot/output/ragtest-hotpot-keyword-0.5.json

set -e

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <hotpot_data.jsonl> <output_dir>"
    echo ""
    echo "Example:"
    echo "  $0 data/hotpotqa_dev_shards/shard_0.jsonl ragtest-hotpot"
    exit 1
fi

HOTPOT_DATA="$1"
OUTPUT_DIR="$2"
KETRAG_ROOT="KET-RAG/${OUTPUT_DIR}"

echo "=================================================="
echo "Preparing KET-RAG contexts for official adapter"
echo "=================================================="
echo "Input: $HOTPOT_DATA"
echo "Output: $KETRAG_ROOT"
echo ""

# Step 1: Convert HotpotQA JSONL to KET-RAG format
echo "[Step 1/4] Converting HotpotQA data to KET-RAG format..."
python scripts/convert_hotpot_to_ketrag.py "$HOTPOT_DATA" "$KETRAG_ROOT"

# Step 2: Initialize GraphRAG project
echo "[Step 2/4] Initializing GraphRAG project..."
cd KET-RAG
poetry run graphrag init --root "${OUTPUT_DIR}/"

# Step 3: Run GraphRAG indexing (this may take a while)
echo "[Step 3/4] Running GraphRAG indexing (this may take 10-30 minutes)..."
echo "  Note: This uses OpenAI API and will consume tokens"
echo "  Set GRAPHRAG_API_KEY environment variable if not already set"
poetry run graphrag index --root "${OUTPUT_DIR}/"

# Step 4: Create contexts with keyword strategy
echo "[Step 4/4] Creating contexts with keyword strategy..."
poetry run python indexing_sket/create_context.py "${OUTPUT_DIR}/" keyword 0.5

cd ..

echo ""
echo "=================================================="
echo "KET-RAG context preparation complete!"
echo "=================================================="
echo ""
echo "Context file created at:"
echo "  ${KETRAG_ROOT}/output/${OUTPUT_DIR}-keyword-0.5.json"
echo ""
echo "You can now run the official KET-RAG adapter:"
echo "  python experiments/eval_full_baselines.py \\"
echo "    --data_path $HOTPOT_DATA \\"
echo "    --baselines ketrag_official \\"
echo "    --ketrag_context_file ${KETRAG_ROOT}/output/${OUTPUT_DIR}-keyword-0.5.json"
echo ""
