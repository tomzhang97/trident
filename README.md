# TRIDENT v1

A complete implementation of the **TRIDENT v1** specification for **Risk-Controlled, Budget-Aware RAG**. This system implements the **RC-MCFC** (Risk-Controlled Min-Cost Facet Cover) algorithm with provable guarantees, along with a **Pareto-Knapsack** mode for high-throughput serving.

## Features

- **Safe-Cover Mode**: Provides per-facet coverage certificates with FWER control and early abstention
- **Pareto-Knapsack Mode**: Maximizes submodular utility under token budgets with VQC and BwK integration
- **Two-stage Scoring**: Efficient facet-aware shortlisting with CE/NLI verification
- **Mondrian Calibration**: Split/Mondrian conformal prediction with isotonic regression
- **Drift Monitoring**: PSI-based distribution shift detection with automatic fallback
- **Multi-hop Support**: Facet mining for entities, relations, temporal, numeric, and bridge requirements
- **Local LLM Support**: Works with HuggingFace models, vLLM, and quantization

## Installation

### Quick Setup

```bash
chmod +x setup.sh
./setup.sh
```

### Manual Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -U pip
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Create directories
mkdir -p runs/_shards results configs .cache/{datasets,models}
```

## Usage

### 1. Prepare Data Shards

```bash
python experiments/prepare_shards.py \
  --dataset hotpotqa \
  --split validation \
  --shard_size 100 \
  --limit 1000 \
  --generate_commands \
  --num_gpus 4 \
  --budget_tokens 2000 \
  --mode safe_cover
```

This creates:
- Data shards in `runs/_shards/`
- A manifest file with shard information
- Shell scripts for running evaluation

### 2. Run Evaluation

**Single shard:**
```bash
python experiments/eval_complete_runnable.py \
  --worker \
  --data_path runs/_shards/validation_0_99.json \
  --output_dir results/hotpotqa_test/validation_0_99 \
  --device 0 \
  --budget_tokens 2000 \
  --mode safe_cover \
  --model meta-llama/Llama-2-7b-hf \
  --temperature 0.0 \
  --seed 42
```

**Parallel execution:**
```bash
# Run all shards in parallel
./runs/_shards/run_parallel.sh
```

**With custom configuration:**
```bash
python experiments/eval_complete_runnable.py \
  --worker \
  --data_path runs/_shards/validation_0_99.json \
  --output_dir results/custom_exp \
  --config configs/my_config.json \
  --device 0
```

### 3. Aggregate Results

```bash
python experiments/aggregate_results.py \
  --results_dir results/hotpotqa_validation
```

This generates:
- `evaluation_report.txt`: Human-readable report
- `aggregated_metrics.json`: Full metrics in JSON
- `errors.csv`: Failed queries analysis
- `pareto_data.csv`: Pareto frontier data (if applicable)

## Command-Line Arguments

### eval_complete_runnable.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_path` | Required | Path to data shard JSON |
| `--output_dir` | Required | Output directory for results |
| `--model` | meta-llama/Llama-2-7b-hf | LLM model name |
| `--device` | 0 | CUDA device (-1 for CPU) |
| `--temperature` | 0.0 | Sampling temperature |
| `--max_new_tokens` | 256 | Max tokens to generate |
| `--load_in_8bit` | False | 8-bit quantization |
| `--mode` | safe_cover | safe_cover/pareto/both |
| `--budget_tokens` | 2000 | Token budget |
| `--retrieval_method` | dense | dense/sparse/hybrid |
| `--encoder_model` | facebook/contriever | Retrieval encoder |
| `--corpus_path` | None | Path to retrieval corpus |
| `--top_k` | 100 | Top-k passages to retrieve |

## Configuration

Edit `configs/default.json` to customize:

```json
{
  "mode": "safe_cover",
  "safe_cover": {
    "per_facet_alpha": 0.01,
    "token_cap": 2000,
    "early_abstain": true,
    "monitor_drift": true,
    "psi_threshold": 0.5
  },
  "pareto": {
    "budget": 2000,
    "relaxed_alpha": 0.05,
    "use_vqc": true,
    "use_bwk": true,
    "max_vqc_iterations": 3
  },
  "llm": {
    "model_name": "meta-llama/Llama-2-7b-hf",
    "temperature": 0.0,
    "max_new_tokens": 256
  },
  "retrieval": {
    "method": "dense",
    "encoder_model": "facebook/contriever",
    "top_k": 100
  }
}
```

## Benchmarks Supported

- **HotpotQA**: Multi-hop reasoning
- **MuSiQue**: Multi-hop with diverse reasoning
- **2WikiMultihopQA**: Two-hop questions
- **Natural Questions**: Open-domain QA
- **TriviaQA**: Reading comprehension
- **SQuAD/SQuAD v2**: Extractive QA

## Key Components

### RC-MCFC (Safe-Cover Mode)
- Fixed coverage sets via Bonferroni correction
- Greedy set cover with **O(log |F|)** approximation
- Dual lower bounds for early abstention
- Per-facet **FWER ≤ α** certificates

### Pareto-Knapsack Mode
- Lazy greedy for submodular maximization
- **VQC** for targeted query rewriting
- **BwK** for consumption-aware exploration
- Pareto frontier computation

### Calibration
- **Mondrian bins**: Stratified by facet type and text length
- **Isotonic regression**: Monotonic score calibration
- **Online recalibration**: Drift detection and adaptation

## Module Structure

```
trident/
├── safe_cover.py        # RC-MCFC algorithm with certificates
├── pareto.py           # Pareto-Knapsack optimizer
├── pipeline.py         # Main TRIDENT orchestrator
├── facets.py           # Facet mining and representation
├── candidates.py       # Passage candidate utilities
├── nli_scorer.py       # Cross-encoder/NLI scoring
├── retrieval.py        # Dense/sparse/hybrid retrieval
├── calibration.py      # Mondrian/split calibration
├── vqc.py              # Verifier-driven Query Compiler
├── bwk.py              # Bandits with Knapsacks controller
├── monitoring.py       # Drift and quality monitoring
├── evaluation.py       # Benchmark evaluation utilities
├── llm_interface.py    # Local LLM support
├── config.py           # Configuration dataclasses
└── logging_utils.py    # Telemetry and logging

experiments/
├── eval_complete_runnable.py  # Main evaluation script
├── prepare_shards.py          # Data sharding utility
└── aggregate_results.py       # Results aggregation
```

## Performance Tips

1. **GPU Memory**: Use `--load_in_8bit` for large models
2. **Batching**: Increase NLI batch size for throughput
3. **Caching**: Results are cached automatically
4. **Parallelism**: Use multiple GPUs with sharding
5. **vLLM**: Install vLLM for high-throughput serving

## Theory & Guarantees

1. **RC-MCFC Approximation**: O(log |F|) approximation to min-cost cover
2. **Certificate Semantics**: Per-facet FWER ≤ α_f under calibration
3. **Dual Lower Bound**: LB_dual ≤ OPT for early abstention
4. **Pareto Optimality**: (1-1/e) approximation for monotone submodular
5. **BwK Regret**: Sublinear under bounded rewards

## Troubleshooting

**CUDA out of memory:**
- Use `--load_in_8bit` or `--load_in_4bit`
- Reduce `--top_k` passages
- Decrease NLI batch size

**Slow evaluation:**
- Enable vLLM for faster inference
- Use hybrid retrieval instead of dense
- Increase number of workers

**Poor quality:**
- Tune `per_facet_alpha` (lower = stricter)
- Increase `budget_tokens`
- Enable VQC in Pareto mode

## Acknowledgments

Based on the TRIDENT specification for provably efficient retrieval-augmented generation.
