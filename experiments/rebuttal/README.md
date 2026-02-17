# Rebuttal Experiment Suite

Experiment-only rebuttal plan aligned to three reviewers (22Vu, Kn2g, AXWh),
optimized for max score lift per GPU-hour.

## Quick Start

```bash
# Run the full suite (recommended order):
python -m experiments.rebuttal.run_all \
    --hotpot_path data/hotpotqa_dev.json \
    --musique_path data/musique_dev.jsonl \
    --results_dir runs/paper_results \
    --calibration_dir data/calibration \
    --output_root runs/rebuttal \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --device 0

# Minimal "score-moving" subset (E1 + E2 + E1.5 + E1.6):
python -m experiments.rebuttal.run_all --minimal \
    --hotpot_path data/hotpotqa_dev.json \
    --output_root runs/rebuttal \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --device 0

# Dry run (show commands without executing):
python -m experiments.rebuttal.run_all --dry_run \
    --hotpot_path data/hotpotqa_dev.json
```

## Experiment Slate

### Tier 0 -- Trust Fixes (no GPU)

| ID    | Script                     | Purpose                       |
|-------|----------------------------|-------------------------------|
| E0.1  | `e0_1_sanity_check.py`     | Recompute metrics, flag typos |
| E0.2  | `e0_2_calibration_protocol.py` | Document calibration pool |

### Tier 1 -- Core Runs (highest ROI)

| ID   | Script                      | Purpose                          | Reviewer |
|------|-----------------------------|----------------------------------|----------|
| E1   | `e1_latency_breakdown.py`   | Stage-level latency breakdown    | 22Vu/Kn2g|
| E2   | `e2_facet_robustness.py`    | Facet miner robustness ablation  | Kn2g/AXWh|
| E3   | `e3_verifier_k_sweep.py`    | Verifier shortlist K sweep       | 22Vu/Kn2g|

### Tier 1.5 -- Reviewer-Specific Blockers

| ID    | Script                          | Purpose                       | Reviewer |
|-------|---------------------------------|-------------------------------|----------|
| E1.5  | `e1_5_backbone_recalibration.py`| Backbone calibration transfer | 22Vu     |
| E1.6  | `e1_6_retrieval_sensitivity.py` | Retriever-agnostic gains      | AXWh     |

### Tier 2 -- Accountability Evidence

| ID    | Script                      | Purpose                           | Reviewer |
|-------|-----------------------------|-----------------------------------|----------|
| E1.7  | `e1_7_safe_cover_curve.py`  | Safe-Cover alpha operating curve  | Kn2g     |
| E5    | `e5_unsupported_answer.py`  | Faithfulness proxy                | All      |

## Execution Order (recommended)

1. E0.1 + E0.2 (no GPU, fast)
2. E1 (latency breakdown)
3. E2 (facet robustness)
4. E3 (verifier K sweep)
5. E1.5 (backbone recalibration)
6. E1.6 (retrieval sensitivity)
7. E1.7 (Safe-Cover curve)
8. E5 (unsupported answer rate)

If time-constrained, the **score-moving subset** is: **E1 + E2 + E1.5 + E1.6**.

## Output Format

Every experiment emits a JSON report to `<output_root>/<experiment_id>/`:

```json
{
  "metadata": {
    "experiment_id": "e1_latency_hotpotqa",
    "dataset": "hotpotqa",
    "split": "dev",
    "budget": 500,
    "mode": "safe_cover,pareto",
    "backbone": "meta-llama/Meta-Llama-3-8B-Instruct",
    "seed": 42,
    "timestamp": "2025-01-15T10:30:00"
  },
  "metrics": { ... },
  "compute": { ... },
  "per_query": [ ... ],
  "summary_table": "| Method | EM | F1 | ... |"
}
```

## Running Individual Experiments

Each experiment can be run standalone:

```bash
# E0.1: Sanity check
python -m experiments.rebuttal.e0_1_sanity_check \
    --results_dir runs/paper_results --output_dir runs/rebuttal/e0_1

# E1: Latency breakdown
python -m experiments.rebuttal.e1_latency_breakdown \
    --data_path data/hotpotqa_dev.json --budget 500 --limit 200 \
    --model meta-llama/Meta-Llama-3-8B-Instruct --device 0

# E2: Facet robustness (3 seeds)
python -m experiments.rebuttal.e2_facet_robustness \
    --data_path data/hotpotqa_dev.json --budget 500 --limit 500 \
    --seeds 42 123 456 --device 0

# E3: Verifier K sweep
python -m experiments.rebuttal.e3_verifier_k_sweep \
    --data_path data/hotpotqa_dev.json --budget 500 --limit 500 \
    --ks 8 16 32 --device 0

# E1.5: Backbone recalibration
python -m experiments.rebuttal.e1_5_backbone_recalibration \
    --data_path data/hotpotqa_dev.json --limit 200 \
    --backbones meta-llama/Meta-Llama-3-8B-Instruct Qwen/Qwen3-8B \
    --device 0

# E1.6: Retrieval sensitivity
python -m experiments.rebuttal.e1_6_retrieval_sensitivity \
    --data_path data/hotpotqa_dev.json --budget 500 --limit 500 \
    --retrievers dense bm25 --top_ns 20 50 100 --device 0

# E1.7: Safe-Cover operating curve
python -m experiments.rebuttal.e1_7_safe_cover_curve \
    --data_path data/musique_dev.jsonl --dataset musique \
    --limit 200 --alphas 0.1 0.05 0.01 --device 0

# E5: Unsupported answer rate
python -m experiments.rebuttal.e5_unsupported_answer \
    --data_path data/hotpotqa_dev.json --budget 500 --limit 500 --device 0
```

## Expected Rebuttal Sentences

Each experiment is designed to produce one key rebuttal sentence:

- **E1**: "Safe-Cover overhead is dominated by verification (X% time; Y pairs; Z batches) and is batch-parallelizable."
- **E2**: "Even with corrupted facets, Pareto maintains dF1 over top-k; degradation is graceful."
- **E3**: "Reducing K cuts verifier pairs by X% and latency by Y ms with only dF1 = -Z (smooth knob)."
- **E1.5**: "Per-backbone calibration reduces cross-architecture discrepancy: ECE improves [x->y]."
- **E1.6**: "Across retriever types and pool sizes, TRIDENT-Pareto consistently improves dF1."
- **E1.7**: "Safe-Cover behaves as intended: lowering alpha increases abstention while improving answered-subset accuracy."
- **E5**: "TRIDENT reduces unsupported-answer rate vs top-k by [delta] (conservative lower bound)."
