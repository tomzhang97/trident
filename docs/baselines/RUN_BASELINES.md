# Running Baseline Evaluations

## Quick Reference Commands

### GraphRAG Baseline

```bash
python eval_complete_runnable.py --worker \
  --data_path runs/_shards/validation_0_99.json \
  --output_dir runs/graphrag \
  --mode graphrag \
  --config configs/graphrag.json \
  --device 2 \
  --model Meta-Llama-3-8B-Instruct
```

### Self-RAG Baseline

```bash
python eval_complete_runnable.py --worker \
  --data_path runs/_shards/validation_0_99.json \
  --output_dir runs/selfrag \
  --mode self_rag \
  --config configs/self_rag.json \
  --device 2 \
  --model Meta-Llama-3-8B-Instruct
```

### Self-RAG with Critic

```bash
python eval_complete_runnable.py --worker \
  --data_path runs/_shards/validation_0_99.json \
  --output_dir runs/selfrag_critic \
  --mode self_rag \
  --config configs/self_rag.json \
  --device 2 \
  --model Meta-Llama-3-8B-Instruct \
  --selfrag_use_critic
```

### TRIDENT Pareto Mode

```bash
python eval_complete_runnable.py --worker \
  --data_path runs/_shards/validation_0_99.json \
  --output_dir runs/trident_pareto \
  --mode pareto \
  --config configs/trident_pareto.json \
  --device 2 \
  --model Meta-Llama-3-8B-Instruct \
  --budget_tokens 2000
```

### TRIDENT Safe Cover Mode

```bash
python eval_complete_runnable.py --worker \
  --data_path runs/_shards/validation_0_99.json \
  --output_dir runs/trident_safe_cover \
  --mode safe_cover \
  --config configs/trident_safe_cover.json \
  --device 2 \
  --model Meta-Llama-3-8B-Instruct \
  --budget_tokens 2000
```

---

## Key Command-Line Arguments

### Required Arguments

- `--worker`: Run as worker (required flag)
- `--data_path`: Path to data shard JSON file
- `--output_dir`: Directory for output files
- `--mode`: System mode (`graphrag`, `self_rag`, `pareto`, `safe_cover`, `both`)

### Model Configuration

- `--model`: LLM model name (e.g., `Meta-Llama-3-8B-Instruct`)
- `--device`: CUDA device number (e.g., `2`) or `-1` for CPU
- `--temperature`: Sampling temperature (default: `0.0`)
- `--max_new_tokens`: Max tokens to generate (default: `512`)
- `--load_in_8bit`: Load model in 8-bit mode (flag)

### Retrieval Configuration

- `--common_k`: Common k for all baselines (default: `8`)
- `--retrieval_method`: `dense`, `hybrid`, or `sparse` (default: `dense`)
- `--encoder_model`: Encoder model (default: `facebook/contriever`)
- `--top_k`: Top-k passages to retrieve (default: `100`)
- `--corpus_path`: Path to corpus for retrieval

### GraphRAG-Specific Arguments

- `--graphrag_k`: Number of documents (defaults to `--common_k`)
- `--graphrag_max_seeds`: Max seed nodes (default: `10`)
- `--graphrag_topk_nodes`: Top-k candidate nodes (default: `20`)
- `--graphrag_max_hops`: Max BFS hops (default: `2`)

### Self-RAG-Specific Arguments

- `--selfrag_k`: Number of documents (defaults to `--common_k`)
- `--selfrag_use_critic`: Enable critic verification (flag)
- `--selfrag_allow_oracle_context`: Allow oracle context mode (flag)

### TRIDENT-Specific Arguments

- `--budget_tokens`: Token budget (default: `2000`)

### Other Arguments

- `--config`: Path to config JSON file (optional)
- `--dataset`: Dataset name (default: `hotpotqa`)
- `--seed`: Random seed (default: `42`)

---

## Configuration Files

### Using Config Files

You can specify a config file with `--config`:

```bash
python eval_complete_runnable.py --worker \
  --data_path runs/_shards/validation_0_99.json \
  --output_dir runs/graphrag \
  --mode graphrag \
  --config configs/baseline_comparison.json \
  --device 2
```

### Available Configs

- `configs/baseline_comparison.json`: Harmonized config for all baselines
- `configs/graphrag.json`: GraphRAG-specific config
- `configs/self_rag.json`: Self-RAG-specific config
- `configs/trident_pareto.json`: TRIDENT Pareto mode
- `configs/trident_safe_cover.json`: TRIDENT Safe Cover mode

---

## Fair Comparison Setup

### Same k for all baselines

```bash
# GraphRAG with k=8
python eval_complete_runnable.py --worker \
  --data_path runs/_shards/validation_0_99.json \
  --output_dir runs/graphrag_k8 \
  --mode graphrag \
  --common_k 8 \
  --device 2 \
  --model Meta-Llama-3-8B-Instruct

# Self-RAG with k=8
python eval_complete_runnable.py --worker \
  --data_path runs/_shards/validation_0_99.json \
  --output_dir runs/selfrag_k8 \
  --mode self_rag \
  --common_k 8 \
  --device 2 \
  --model Meta-Llama-3-8B-Instruct
```

### Using Common Config

Create a shared config file and use it for all runs:

```bash
# All systems use configs/baseline_comparison.json with common_k=8
for mode in graphrag self_rag pareto safe_cover; do
  python eval_complete_runnable.py --worker \
    --data_path runs/_shards/validation_0_99.json \
    --output_dir runs/${mode} \
    --mode ${mode} \
    --config configs/baseline_comparison.json \
    --device 2 \
    --model Meta-Llama-3-8B-Instruct
done
```

---

## Running Multiple Shards

If you have multiple data shards, run them separately:

```bash
# Process shards 0-99
python eval_complete_runnable.py --worker \
  --data_path runs/_shards/validation_0_99.json \
  --output_dir runs/graphrag \
  --mode graphrag \
  --device 2 \
  --model Meta-Llama-3-8B-Instruct

# Process shards 100-199
python eval_complete_runnable.py --worker \
  --data_path runs/_shards/validation_100_199.json \
  --output_dir runs/graphrag \
  --mode graphrag \
  --device 2 \
  --model Meta-Llama-3-8B-Instruct
```

---

## Output Files

Each run produces:

1. **`{mode}_results.json`**: Detailed per-query results with:
   - Question, answer, ground truth
   - Tokens used, latency
   - EM and F1 scores
   - System-specific stats

2. **Logs**: In `{output_dir}/logs/`

3. **Aggregated metrics**: Can be computed from results JSON

---

## Troubleshooting

### Error: ModuleNotFoundError: No module named 'trident'

**Problem**: Running from wrong directory or incorrect script path.

**Solution**: Run from the repository root:
```bash
cd /path/to/trident
python eval_complete_runnable.py --worker ...
```

**NOT:**
```bash
python experiments/eval_complete_runnable.py --worker ...  # Wrong!
```

### Error: GraphRAGSystem.answer() got an unexpected keyword argument 'supporting_facts'

**Problem**: Old baseline code without updated interface.

**Solution**: Pull latest changes:
```bash
git pull origin claude/strengthen-baselines-0133AVGR6VmM1Xny6rXttn2A
```

### Error: ValueError: Self-RAG in retrieval mode does not accept oracle context

**Problem**: Dataset provides context but Self-RAG is in retrieval-only mode.

**Solution**: Either:
1. Enable oracle context: `--selfrag_allow_oracle_context`
2. Or modify evaluation to pass `context=None`

### Low GPU Memory

**Problem**: OOM errors with large models.

**Solutions**:
1. Use 8-bit quantization: `--load_in_8bit`
2. Reduce batch sizes in config
3. Use smaller model
4. Use different GPU: `--device 3`

---

## Performance Tips

### Speed up evaluation

1. **Use multiple GPUs in parallel:**
```bash
# GPU 0: GraphRAG
CUDA_VISIBLE_DEVICES=0 python eval_complete_runnable.py --worker \
  --mode graphrag --device 0 ... &

# GPU 1: Self-RAG
CUDA_VISIBLE_DEVICES=1 python eval_complete_runnable.py --worker \
  --mode self_rag --device 0 ... &

wait
```

2. **Process shards in parallel:**
```bash
for shard in runs/_shards/*.json; do
  python eval_complete_runnable.py --worker \
    --data_path "$shard" \
    --output_dir runs/graphrag \
    --mode graphrag \
    --device 2 &
done
wait
```

3. **Use quantization:**
```bash
--load_in_8bit  # Reduces memory, slightly slower inference
```

---

## Verification

After running, verify results:

```bash
# Check output exists
ls runs/graphrag/graphrag_results.json

# Count queries processed
jq 'length' runs/graphrag/graphrag_results.json

# View average metrics
jq '[.[] | .em] | add/length' runs/graphrag/graphrag_results.json  # Avg EM
jq '[.[] | .tokens_used] | add/length' runs/graphrag/graphrag_results.json  # Avg tokens
```

---

## Complete Example: Full Comparison

```bash
#!/bin/bash

MODEL="Meta-Llama-3-8B-Instruct"
DATA="runs/_shards/validation_0_99.json"
DEVICE=2
COMMON_K=8

# GraphRAG
python eval_complete_runnable.py --worker \
  --data_path "$DATA" \
  --output_dir runs/graphrag \
  --mode graphrag \
  --common_k $COMMON_K \
  --device $DEVICE \
  --model "$MODEL"

# Self-RAG
python eval_complete_runnable.py --worker \
  --data_path "$DATA" \
  --output_dir runs/selfrag \
  --mode self_rag \
  --common_k $COMMON_K \
  --device $DEVICE \
  --model "$MODEL"

# Self-RAG with Critic
python eval_complete_runnable.py --worker \
  --data_path "$DATA" \
  --output_dir runs/selfrag_critic \
  --mode self_rag \
  --common_k $COMMON_K \
  --selfrag_use_critic \
  --device $DEVICE \
  --model "$MODEL"

# TRIDENT Pareto
python eval_complete_runnable.py --worker \
  --data_path "$DATA" \
  --output_dir runs/trident_pareto \
  --mode pareto \
  --budget_tokens 2000 \
  --device $DEVICE \
  --model "$MODEL"

# TRIDENT Safe Cover
python eval_complete_runnable.py --worker \
  --data_path "$DATA" \
  --output_dir runs/trident_safe_cover \
  --mode safe_cover \
  --budget_tokens 2000 \
  --device $DEVICE \
  --model "$MODEL"

echo "All evaluations complete!"
```

Save as `run_all_baselines.sh` and execute:
```bash
chmod +x run_all_baselines.sh
./run_all_baselines.sh
```
