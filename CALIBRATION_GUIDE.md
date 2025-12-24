# Safe-Cover Calibration Guide

This guide explains how to generate calibration data and use it with Safe-Cover.

## Why Calibration is Required

Safe-Cover uses **conformal prediction** to provide statistical coverage guarantees. This requires:
- A calibration dataset of (facet, passage, label) tuples
- NLI scores for each (facet, passage) pair
- Binary labels: 1 = passage is sufficient, 0 = insufficient

Without calibration data, Safe-Cover **cannot compute valid p-values** and will abstain on all queries.

## Quick Start (3 Steps)

### Step 1: Extract Calibration Data from HotpotQA

Use the training set to create calibration samples:

```bash
python extract_calibration_data.py \
  --data_path hotpotqa/data/hotpot_train_v1.json \
  --output_path calibration_data.jsonl \
  --num_samples 500 \
  --device cuda:0
```

**What this does:**
- Mines facets from 500 training questions
- Scores all (facet, passage) pairs with NLI
- Labels passages using `supporting_facts` as ground truth
- Outputs JSONL with format: `{id, score, label, metadata}`

**Expected output:**
```
✅ Extracted 15000+ calibration samples
📁 Saved to: calibration_data.jsonl
📊 Statistics:
  Positive samples: 3000 (20%)
  Negative samples: 12000 (80%)
```

### Step 2: Train Calibrator

Build the calibration model:

```bash
python train_calibration.py \
  --data_path calibration_data.jsonl \
  --output_path calibrator.json \
  --use_mondrian \
  --calibration_n_min 50
```

**Parameters:**
- `--use_mondrian`: Enable Mondrian binning (recommended)
- `--calibration_n_min`: Minimum negatives per bin before merging (default: 50)

**Output:**
```
✅ Calibration complete
📁 Saved to: calibrator.json
📊 Bin statistics: 54 bins, avg 280 samples/bin
```

### Step 3: Run Safe-Cover with Calibration

```bash
CUDA_VISIBLE_DEVICES=1 python experiments/eval_complete_runnable.py \
  --data_path hotpotqa/data/hotpot_dev_distractor_v1.json \
  --num_gpus 1 \
  --model Meta-Llama-3-8B-Instruct \
  --mode safe_cover \
  --config_family safe_cover_equal_2000 \
  --calibration_path calibrator.json \
  --output_dir results/hotpot/llama3/safe_cover_calibrated
```

**Key parameter:**
- `--calibration_path calibrator.json` ← This loads the trained calibrator

## Verification

Check that calibration is working:

1. **Look for loading message:**
   ```
   📊 Loading calibration from: calibrator.json
   ✅ Calibrator loaded successfully
   ```

2. **Check results:**
   ```json
   "certificate_audit": {
     "certificate_rate": 0.85,  // Should be > 0
     "total_certificates": 250   // Should be > 0
   }
   ```

3. **Abstention rate:**
   - Without calibration: **100%** abstention
   - With calibration: Should be **< 20%** for alpha=0.02

## Troubleshooting

### Still 100% Abstention

**Problem:** Alpha is too strict for your calibration distribution

**Solutions:**
1. Use relaxed config for testing:
   ```bash
   --config_family safe_cover_2000_relaxed  # alpha=0.10
   ```

2. Extract more calibration data (increase `--num_samples`)

3. Check calibration quality:
   ```python
   import json
   with open('calibrator.json') as f:
       cal = json.load(f)
   print(f"Bins: {len(cal['tables'])}")
   ```

### Low Certificate Rate

**Problem:** Not enough calibration samples per bin

**Solutions:**
1. Increase `--num_samples` in extraction (try 1000-2000)
2. Decrease `--calibration_n_min` in training (try 30)
3. Disable Mondrian for simpler calibration:
   ```bash
   --use_mondrian false
   ```

## Advanced: Custom Calibration Data

If you have your own labeled dataset:

```jsonl
{"id": "sample_1", "score": 0.85, "label": 1, "metadata": {"facet_type": "ENTITY", "text_length": 120, "retriever_score": 0.9}}
{"id": "sample_2", "score": 0.45, "label": 0, "metadata": {"facet_type": "RELATION", "text_length": 85, "retriever_score": 0.7}}
```

**Required fields:**
- `score`: NLI/relevance score (0-1)
- `label`: 1 if passage is sufficient for facet, 0 otherwise
- `metadata.facet_type`: ENTITY, RELATION, TEMPORAL, NUMERIC, BRIDGE_HOP1, BRIDGE_HOP2
- `metadata.text_length`: Passage length in characters
- `metadata.retriever_score`: Retrieval score (0-1)

## Calibration for Different Datasets

### MuSiQue

Use MuSiQue training data:

```bash
python extract_calibration_data.py \
  --data_path musique/data/musique_full_v1.0_train.jsonl \
  --output_path musique_calibration.jsonl \
  --num_samples 500 \
  --device cuda:0

python train_calibration.py \
  --data_path musique_calibration.jsonl \
  --output_path musique_calibrator.json \
  --use_mondrian

# Run with MuSiQue calibrator
python experiments/eval_complete_runnable.py \
  --data_path musique/data/musique_ans_v1.0_dev.jsonl \
  --dataset musique \
  --mode safe_cover \
  --config_family safe_cover_2000_no_mondrian \
  --calibration_path musique_calibrator.json \
  --output_dir results/musique/safe_cover
```

## Ablation Studies with Calibration

All ablation configs work with calibration:

```bash
# 1. Pareto 500 (No Rerank) - doesn't need calibration
python experiments/eval_complete_runnable.py \
  --config_family pareto_match_500_no_rerank \
  --mode pareto \
  ...

# 2. Safe 2000 (NLI 0.8)
python experiments/eval_complete_runnable.py \
  --config_family safe_cover_2000_nli08 \
  --calibration_path calibrator.json \
  ...

# 3. Safe 2000 (Mondrian Off)
python experiments/eval_complete_runnable.py \
  --config_family safe_cover_2000_no_mondrian \
  --calibration_path calibrator.json \
  ...
```

## Files Reference

| File | Purpose |
|------|---------|
| `extract_calibration_data.py` | Extract calibration samples from HotpotQA |
| `train_calibration.py` | Train calibration model |
| `quick_calibrate.py` | Generate synthetic data (testing only) |
| `CALIBRATION_GUIDE.md` | This guide |

## Summary

**Minimum viable workflow:**
```bash
# 1. Extract (10-15 min with NLI on GPU)
python extract_calibration_data.py --data_path hotpotqa/data/hotpot_train_v1.json --output_path cal.jsonl --num_samples 500

# 2. Train (< 1 min)
python train_calibration.py --data_path cal.jsonl --output_path cal.json --use_mondrian

# 3. Run Safe-Cover (pass --calibration_path)
python experiments/eval_complete_runnable.py ... --calibration_path cal.json
```

**Without calibration:** Safe-Cover will not work (100% abstention).

**With calibration:** Safe-Cover provides coverage guarantees with certificates.
