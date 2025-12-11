# MuSiQue Dataset Experiments

This directory contains the MuSiQue dataset and tools for running TRIDENT experiments on it.

## Dataset Overview

MuSiQue (Multi-hop Questions via Single-hop Question composition) is a multi-hop question answering dataset that requires reasoning over multiple documents. The dataset includes:

- **2-hop, 3-hop, and 4-hop questions** requiring multi-step reasoning
- **Answerable and unanswerable versions** for evaluating answerability detection
- **Supporting fact annotations** for evaluating evidence selection

## Data Files

Located in `musique/data/`:

| File | Description | Size |
|------|-------------|------|
| `musique_ans_v1.0_dev.jsonl` | Answerable questions (dev) | ~30MB |
| `musique_ans_v1.0_test.jsonl` | Answerable questions (test) | ~28MB |
| `musique_full_v1.0_dev.jsonl` | Full dataset with unanswerable (dev) | ~59MB |
| `musique_full_v1.0_test.jsonl` | Full dataset with unanswerable (test) | ~55MB |
| `dev_test_singlehop_questions_v1.0.json` | Single-hop decomposition questions | ~1MB |

## Quick Start

### 1. View Dataset Statistics

```bash
python musique/data_loader.py --stats_only --split ans_dev
```

### 2. Run a Quick Experiment

```bash
# Run TRIDENT Safe-Cover on 50 examples for testing
python experiments/run_musique_experiment.py \
    --mode safe_cover \
    --split ans_dev \
    --limit 50 \
    --model meta-llama/Llama-2-7b-hf
```

### 3. Run Full Experiment

```bash
# Run on full answerable dev set
python experiments/run_musique_experiment.py \
    --mode safe_cover \
    --split ans_dev \
    --config configs/musique.json
```

### 4. Evaluate Results

```bash
# Evaluate a completed experiment
python experiments/evaluate_musique.py \
    --results_dir runs/musique/safe_cover_ans_dev_xxx/
```

## Available Modes

| Mode | Description |
|------|-------------|
| `safe_cover` | TRIDENT Safe-Cover (risk-controlled retrieval) |
| `pareto` | TRIDENT Pareto-Knapsack (quality-cost optimization) |
| `self_rag` | Self-RAG baseline |
| `graphrag` | GraphRAG baseline |
| `ketrag` | KET-RAG baseline |

## Experiment Workflow

### Step 1: Prepare Data

The data loader automatically converts MuSiQue format to TRIDENT's standard format:

```python
from musique.data_loader import MuSiQueDataLoader

loader = MuSiQueDataLoader()
examples = loader.load_and_convert('ans_dev', limit=100)

# Create shards for parallel processing
shard_paths = loader.save_shards(
    examples,
    output_dir='runs/musique_shards',
    shard_size=50
)
```

### Step 2: Run Experiments

Using the experiment runner:

```bash
python experiments/run_musique_experiment.py \
    --mode safe_cover \
    --split ans_dev \
    --model meta-llama/Llama-2-7b-hf \
    --num_gpus 4
```

Or manually with the evaluation script:

```bash
python experiments/eval_complete_runnable.py \
    --worker \
    --data_path runs/musique_shards/ans_dev_0_99.json \
    --output_dir results/musique/shard_0 \
    --mode safe_cover \
    --config configs/musique.json \
    --dataset musique
```

### Step 3: Evaluate Results

```bash
# Run evaluation with official MuSiQue metrics
python experiments/evaluate_musique.py \
    --results_dir runs/musique/safe_cover_ans_dev_xxx/ \
    --split ans_dev
```

## Evaluation Metrics

### Official MuSiQue Metrics

- **answer_f1**: F1 score on predicted answers
- **answer_em**: Exact match on predicted answers
- **support_f1**: F1 score on supporting paragraph selection
- **group_answer_sufficiency_f1**: Joint answer + answerability F1 (full dataset)
- **group_support_sufficiency_f1**: Joint support + answerability F1 (full dataset)

### TRIDENT Internal Metrics

- **exact_match**: Normalized exact match
- **f1_score**: Token-level F1
- **abstention_rate**: Fraction of abstained queries
- **avg_tokens**: Average tokens used per query
- **avg_latency_ms**: Average processing time

## Data Format

### Input Format (MuSiQue)

```json
{
    "id": "2hop__460946_294723",
    "paragraphs": [
        {
            "idx": 0,
            "title": "...",
            "paragraph_text": "...",
            "is_supporting": false
        }
    ],
    "question": "Who is the spouse of the Green performer?",
    "question_decomposition": [...],
    "answer": "Miquette Giraudy",
    "answer_aliases": [],
    "answerable": true
}
```

### TRIDENT Standard Format

```json
{
    "_id": "2hop__460946_294723",
    "question": "Who is the spouse of the Green performer?",
    "answer": "Miquette Giraudy",
    "answer_aliases": [],
    "context": [["Title", ["Sentence 1.", "Sentence 2."]], ...],
    "supporting_facts": [["Title", 0], ["Title", 1]],
    "type": "2hop",
    "answerable": true
}
```

### Prediction Format

```json
{
    "id": "2hop__460946_294723",
    "predicted_answer": "Miquette Giraudy",
    "predicted_support_idxs": [5, 10],
    "predicted_answerable": true
}
```

## Configuration

The default MuSiQue configuration is in `configs/musique.json`:

```json
{
    "mode": "safe_cover",
    "safe_cover": {
        "token_cap": 2500,
        "per_facet_alpha": 0.01,
        "coverage_threshold": 0.15
    },
    "evaluation": {
        "dataset": "musique",
        "metrics": ["em", "f1", "support_em", "support_f1", "faithfulness"]
    }
}
```

## Tips

1. **Start small**: Use `--limit 50` for initial testing
2. **Use GPU**: Set `--device 0` for GPU acceleration
3. **Parallel processing**: Use `--num_gpus 4` for multi-GPU setups
4. **Full evaluation**: Use `full_dev` split to evaluate answerability detection
5. **Compare baselines**: Run the same data with different `--mode` options

## Files

```
musique/
├── data/                           # Dataset files
│   ├── musique_ans_v1.0_dev.jsonl
│   ├── musique_ans_v1.0_test.jsonl
│   ├── musique_full_v1.0_dev.jsonl
│   ├── musique_full_v1.0_test.jsonl
│   └── dev_test_singlehop_questions_v1.0.json
├── data_loader.py                  # Data loading and conversion
├── evaluate_v1.0.py               # MuSiQue evaluator (self-contained, supports TRIDENT)
└── README.md                      # This file

experiments/
├── run_musique_experiment.py      # End-to-end experiment runner
└── evaluate_musique.py            # Evaluation wrapper for TRIDENT results

configs/
└── musique.json                   # MuSiQue experiment configuration
```

## References

- [MuSiQue Paper](https://arxiv.org/abs/2108.00573)
- [MuSiQue Dataset](https://github.com/stonybrooknlp/musique)
