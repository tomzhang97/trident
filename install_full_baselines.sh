#!/bin/bash
# Installation script for full baseline systems

set -e  # Exit on error

echo "=========================================="
echo "Installing Full Baseline Systems"
echo "=========================================="
echo ""

# Support both legacy external_baselines/ layout and repositories at repo root
if [ -d "external_baselines" ]; then
    BASELINE_ROOT="external_baselines"
else
    BASELINE_ROOT="."
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
    "$BASELINE_ROOT/graphrag" \
    "pip install -e . --quiet"

# Install Self-RAG dependencies
# Flash-attn requires torch to be installed FIRST (needs it at build time)
echo ""
echo "------------------------------------------"
echo "Installing Self-RAG dependencies (Step 1: torch)"
echo "------------------------------------------"

# Always ensure torch is installed before flash-attn
pip install torch --quiet
echo "✓ torch installed/verified"

echo ""
echo "------------------------------------------"
echo "Installing Self-RAG dependencies (Step 2: requirements without flash-attn)"
echo "------------------------------------------"

# Install dependencies excluding flash-attn first
if [ -d "$BASELINE_ROOT/self-rag" ]; then
    cd "$BASELINE_ROOT/self-rag"
    echo "Installing from: $(pwd)"

    # Install all requirements except flash-attn
    grep -v "flash-attn" requirements.txt > /tmp/selfrag_requirements_no_flash.txt
    pip install -r /tmp/selfrag_requirements_no_flash.txt --quiet

    echo "✓ Self-RAG core dependencies installed"

    # Try to install flash-attn separately (optional)
    echo ""
    echo "Installing flash-attn (this may take a while and requires CUDA)..."
    if pip install flash-attn>=2.3.6 --quiet 2>/dev/null; then
        echo "✓ flash-attn installed successfully"
    else
        echo "⚠ Warning: flash-attn installation failed. This is optional for inference."
        echo "  If you need flash-attn, please install it manually with CUDA support."
    fi

    cd - > /dev/null
else
    echo "ERROR: $BASELINE_ROOT/self-rag not found. Please clone the repository first."
fi

# Install KET-RAG
echo ""
echo "------------------------------------------"
echo "Installing KET-RAG"
echo "------------------------------------------"
cd "$BASELINE_ROOT/KET-RAG"

# Check if poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "Poetry not found. Installing poetry..."
    pip install poetry --quiet
fi

echo "Installing KET-RAG with poetry..."
poetry install --quiet

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
