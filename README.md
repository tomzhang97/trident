# TRIDENT

A compact Python implementation of the TRIDENT v2.1 specification. The code offers
risk-controlled Safe-Cover and Pareto-Knapsack selection modes with simple lexical
scoring so that experiments can run end-to-end on multi-hop QA benchmarks.

## Installation

Create a virtual environment and install the lightweight dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install scikit-learn datasets
```

`scikit-learn` is optional—if absent the retriever falls back to a bag-of-words
overlap heuristic.

## Quickstart

Run the dataset driver to evaluate a small slice of a benchmark:

```bash
python -m trident.cli dataset --dataset hotpot_qa --split validation --limit 50 --mode safe_cover
```

Use `--mode pareto` to benchmark the relaxed Pareto-Knapsack selector or
`--mode both` to run both strategies.

## Package Overview

See [`docs/STRUCTURE.md`](docs/STRUCTURE.md) for the module layout. Core modules:

* `trident/safe_cover.py` – RC-MCFC Safe-Cover algorithm with certificates.
* `trident/pareto.py` – Pareto-Knapsack greedy optimizer.
* `trident/pipeline.py` – Retrieval → scoring → selection orchestration.
* `trident/evaluation.py` – Dataset loader and evaluation metrics.

The implementations prefer clarity over micro-optimisation (奥卡姆剃刀原则 + KISS原则).
