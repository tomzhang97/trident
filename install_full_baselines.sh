#!/bin/bash
# Installation script for full baseline systems

set -e  # Exit on error

echo "=========================================="
echo "Installing Full Baseline Systems"
echo "=========================================="
echo ""

# Check if external_baselines directory exists
if [ ! -d "external_baselines" ]; then
    echo "ERROR: external_baselines directory not found!"
    echo "Please ensure you've cloned the baseline repositories."
    exit 1
fi

# Function to install a baseline
install_baseline() {
    local name=$1
    local dir=$2
    local install_cmd=$3

    echo ""
    echo "------------------------------------------"
    echo "Installing $name"
    echo "------------------------------------------"

    if [ ! -d "$dir" ]; then
        echo "ERROR: $dir not found. Please clone the repository first."
        return 1
    fi

    cd "$dir"
    echo "Installing from: $(pwd)"

    # Run installation command
    eval "$install_cmd"

    cd - > /dev/null
    echo "✓ $name installed successfully"
}

# Install GraphRAG
install_baseline "GraphRAG" \
    "external_baselines/graphrag" \
    "pip install -e . --quiet"

# Install Self-RAG dependencies
install_baseline "Self-RAG" \
    "external_baselines/self-rag" \
    "pip install -r requirements.txt --quiet"

# Install KET-RAG
echo ""
echo "------------------------------------------"
echo "Installing KET-RAG"
echo "------------------------------------------"
cd external_baselines/KET-RAG

# Check if poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "Poetry not found. Installing poetry..."
    pip install poetry --quiet
fi

echo "Installing KET-RAG with poetry (using current environment)..."
# Force Poetry to reuse the active environment so dependencies stay compatible
# with the current (e.g., conda) environment used by TRIDENT.
export POETRY_VIRTUALENVS_CREATE=${POETRY_VIRTUALENVS_CREATE:-false}
poetry install --quiet --no-root

cd - > /dev/null
echo "✓ KET-RAG installed successfully"

# Install scikit-learn for KET-RAG PageRank
echo ""
echo "------------------------------------------"
echo "Installing scikit-learn for KET-RAG"
echo "------------------------------------------"
pip install scikit-learn --quiet
echo "✓ scikit-learn installed successfully"

# Install vllm for Self-RAG (if not already installed)
echo ""
echo "------------------------------------------"
echo "Installing vllm for Self-RAG"
echo "------------------------------------------"
pip install vllm --quiet || echo "Warning: vllm installation may require CUDA. You may need to install manually."

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Set environment variables:"
echo "   export GRAPHRAG_API_KEY=<your_openai_key>  # For GraphRAG and KET-RAG"
echo "   export HF_TOKEN=<your_hf_token>             # For Self-RAG model download"
echo ""
echo "2. Test installation:"
echo "   python -c 'from baselines import FullGraphRAGAdapter, FullSelfRAGAdapter, FullKETRAGAdapter; print(\"✓ All adapters imported successfully\")'"
echo ""
echo "3. Run evaluation:"
echo "   python eval_full_baselines.py --data_path data/hotpotqa_dev_shards/shard_0.jsonl --max_samples 10"
echo ""
