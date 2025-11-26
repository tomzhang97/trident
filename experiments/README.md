# How to Run Baseline Evaluations

## Important: Run from Repository Root

The evaluation script is located at the **root** of the repository, not in an `experiments/` subdirectory.

### ❌ Wrong (will fail):
```bash
python experiments/eval_complete_runnable.py --worker ...
```

### ✅ Correct:
```bash
cd /path/to/trident
python eval_complete_runnable.py --worker ...
```

---

## Quick Start Commands

Make sure you're in the repository root first:
```bash
cd ~/trident-main  # or wherever your repo is
pwd  # Should show /path/to/trident
```

### GraphRAG
```bash
python eval_complete_runnable.py --worker \
  --data_path runs/_shards/validation_0_99.json \
  --output_dir runs/graphrag \
  --mode graphrag \
  --config configs/graphrag.json \
  --device 2 \
  --model Meta-Llama-3-8B-Instruct
```

### Self-RAG
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

---

## With HF_ENDPOINT (for mirror access)

If you need to use a Hugging Face mirror:

```bash
HF_ENDPOINT=https://hf-mirror.com python eval_complete_runnable.py --worker \
  --data_path runs/_shards/validation_0_99.json \
  --output_dir runs/graphrag \
  --mode graphrag \
  --config configs/graphrag.json \
  --device 2 \
  --model Meta-Llama-3-8B-Instruct
```

---

## Full Comparison Run

```bash
#!/bin/bash
# Save as run_all.sh in the repo root

MODEL="Meta-Llama-3-8B-Instruct"
DATA="runs/_shards/validation_0_99.json"
DEVICE=2

# GraphRAG
python eval_complete_runnable.py --worker \
  --data_path "$DATA" \
  --output_dir runs/graphrag \
  --mode graphrag \
  --config configs/graphrag.json \
  --device $DEVICE \
  --model "$MODEL"

# Self-RAG
python eval_complete_runnable.py --worker \
  --data_path "$DATA" \
  --output_dir runs/selfrag \
  --mode self_rag \
  --config configs/self_rag.json \
  --device $DEVICE \
  --model "$MODEL"

# Self-RAG with Critic
python eval_complete_runnable.py --worker \
  --data_path "$DATA" \
  --output_dir runs/selfrag_critic \
  --mode self_rag \
  --config configs/self_rag.json \
  --device $DEVICE \
  --model "$MODEL" \
  --selfrag_use_critic

# TRIDENT Pareto
python eval_complete_runnable.py --worker \
  --data_path "$DATA" \
  --output_dir runs/trident_pareto \
  --mode pareto \
  --config configs/trident_pareto.json \
  --device $DEVICE \
  --model "$MODEL" \
  --budget_tokens 2000

# TRIDENT Safe Cover
python eval_complete_runnable.py --worker \
  --data_path "$DATA" \
  --output_dir runs/trident_safe_cover \
  --mode safe_cover \
  --config configs/trident_safe_cover.json \
  --device $DEVICE \
  --model "$MODEL" \
  --budget_tokens 2000

echo "All evaluations complete!"
```

Then run:
```bash
chmod +x run_all.sh
./run_all.sh
```

---

## Troubleshooting

### Error: No module named 'trident'

**Cause**: Running from wrong directory or using wrong path.

**Solution**:
```bash
cd ~/trident-main  # Go to repo root
python eval_complete_runnable.py --worker ...  # Run from root
```

**NOT:**
```bash
python experiments/eval_complete_runnable.py  # Wrong!
```

### Error: BaselineConfig got unexpected keyword argument 'common_k'

**Cause**: Old version of config.py.

**Solution**: Pull latest changes:
```bash
git pull origin claude/strengthen-baselines-0133AVGR6VmM1Xny6rXttn2A
```

### Error: GraphRAGSystem.answer() got unexpected keyword argument 'supporting_facts'

**Cause**: Old baseline code.

**Solution**: Pull latest changes:
```bash
git pull origin claude/strengthen-baselines-0133AVGR6VmM1Xny6rXttn2A
```

---

## Key Parameters

All commands support these arguments:

- `--common_k 8`: Shared retrieval k (default: 8)
- `--device 2`: GPU device number
- `--model Meta-Llama-3-8B-Instruct`: Model name
- `--config configs/xxx.json`: Config file
- `--selfrag_use_critic`: Enable Self-RAG critic
- `--selfrag_allow_oracle_context`: Allow oracle context
- `--graphrag_k 8`: GraphRAG retrieval k
- `--graphrag_max_hops 2`: Max BFS hops
- `--budget_tokens 2000`: TRIDENT token budget

See `RUN_BASELINES.md` for complete documentation.
