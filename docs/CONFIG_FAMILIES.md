# TRIDENT Config Families

This document describes the concrete configuration families for running cost-quality frontier experiments with TRIDENT.

## Overview

The config families provide named presets that control:
1. **Evidence token budgets** - How many tokens to use for context
2. **Unit limits** - Maximum number of passages/units to select
3. **Risk parameters** - Statistical control levels (for Safe-Cover)
4. **Abstention behavior** - Whether to abstain when constraints cannot be satisfied

## Available Config Families

### 1. Pareto: "Cheap TRIDENT" Configs

**Goal:** Match or slightly exceed Self-RAG token usage (~1500-2000) with significantly higher EM/F1.

| Config name          | Target tokens | Max units | VQC | BwK | Purpose                                  |
| -------------------- | ------------: | --------: | :-: | :-: | ---------------------------------------- |
| `pareto_cheap_1500`  |          1500 |         8 |  ✗  |  ✗  | Aggressive budget match with Self-RAG    |
| `pareto_cheap_2000`  |          2000 |        10 |  ✗  |  ✗  | Slightly more generous but still "cheap" |
| `pareto_mid_2500`    |          2500 |        12 |  ✓  |  ✓  | Equal-budget vs Safe-Cover candidate     |

**Usage:**
```python
from trident.config import TridentConfig
from trident.config_families import PARETO_CHEAP_1500

config = TridentConfig(
    mode="pareto",
    pareto=PARETO_CHEAP_1500,
    # ... other settings
)
```

### 2. Safe-Cover: Strict Risk Control

**Goal:** Demonstrate risk-controlled coverage with different budget regimes.

| Config name              | α_f (facet risk) | Max tokens | Abstain on infeasible | Purpose                                 |
| ------------------------ | ---------------: | ---------: | :-------------------: | --------------------------------------- |
| `safe_cover_loose_4000`  |             0.05 |       4000 |          ✗            | Higher budget + stricter risk           |
| `safe_cover_equal_2500`  |             0.02 |       2500 |          ✓            | Same budget as pareto_mid, strict risk  |
| `safe_cover_equal_2000`  |             0.02 |       2000 |          ✓            | Same budget as pareto_cheap_2000        |

**Usage:**
```python
from trident.config import TridentConfig
from trident.config_families import SAFE_COVER_EQUAL_2500

config = TridentConfig(
    mode="safe_cover",
    safe_cover=SAFE_COVER_EQUAL_2500,
    # ... other settings
)
```

**Key differences:**
- **Loose budget regime** (`safe_cover_loose_4000`): Accepts higher token cost to enforce stricter risk control. Falls back to Pareto if infeasible.
- **Equal budget regime** (`safe_cover_equal_*`): Same budget as Pareto configs, but with strict risk control. Will abstain if constraints cannot be satisfied.

### 3. Self-RAG: Fair Baseline Comparison

**Goal:** Fair comparison with TRIDENT at different capacity levels.

| Config name      | k  | Critic | Oracle context | Purpose                           |
| ---------------- | -: | :----: | :------------: | --------------------------------- |
| `selfrag_base`   |  8 |   ✗    |       ✗        | Standard Self-RAG (≈1450 tokens)  |
| `selfrag_strong` | 16 |   ✓    |       ✗        | Stronger baseline to test limits  |
| `selfrag_oracle` |  8 |   ✗    |       ✓        | Upper bound (uses gold context)   |

**Usage:**
```python
from trident.config import TridentConfig
from trident.config_families import SELFRAG_BASE

config = TridentConfig(
    mode="pareto",  # Not used for Self-RAG
    baselines=SELFRAG_BASE,
    # ... other settings
)
```

## Quick Start

### Command-Line Usage

```bash
# List all available configs
python examples/run_config_families.py --list

# Create and view a config
python examples/run_config_families.py --config pareto_cheap_1500

# Save config to file
python examples/run_config_families.py --config safe_cover_equal_2500 --output my_config.json
```

### Programmatic Usage

```python
from trident.config_families import get_config, get_selfrag_config

# Get a TRIDENT config
pareto_cfg = get_config("pareto_cheap_1500")
safe_cover_cfg = get_config("safe_cover_equal_2500")

# Get a Self-RAG config
selfrag_cfg = get_selfrag_config("selfrag_base")

# Use in your experiment
from trident.pipeline import TridentPipeline
from trident.config import TridentConfig

config = TridentConfig(mode="pareto", pareto=pareto_cfg)
pipeline = TridentPipeline(config, llm, retriever)
```

## Key Metrics to Track

When running experiments with these configs, make sure to log:

### For all configs:
- `avg_em` - Exact match accuracy
- `avg_f1` - Token-level F1 score
- `avg_tokens_total` - Total tokens used (evidence + answer)
- `avg_evidence_tokens` - Evidence tokens only
- `avg_latency_ms` - End-to-end latency

### For Safe-Cover configs:
- `avg_num_units` - Number of passages selected
- `avg_num_violated_facets` - Number of facets not covered
- `abstention_rate` - Fraction of queries where system abstained

### For Self-RAG configs:
- `avg_num_docs_retrieved` - Number of documents retrieved
- `retrieval_gate_distribution` - Fraction of RETRIEVE vs NO_RETRIEVE decisions
- `critic_label_distribution` - Distribution of critic labels (if using critic)

## Recommended Experiment Sets

### Minimal Set (Quick Validation)
1. `pareto_cheap_1500`
2. `pareto_mid_2500`
3. `safe_cover_equal_2500`
4. `selfrag_base`

### Full Set (Complete Frontier)
1. `pareto_cheap_1500`
2. `pareto_cheap_2000`
3. `pareto_mid_2500`
4. `safe_cover_loose_4000`
5. `safe_cover_equal_2500`
6. `safe_cover_equal_2000`
7. `selfrag_base`
8. `selfrag_strong`

## Implementation Details

### Budget Enforcement

The configs control budget through:

1. **`max_evidence_tokens`**: Hard limit on total evidence tokens
   - Used by both Pareto and Safe-Cover modes
   - Algorithm stops when this limit would be exceeded

2. **`max_units`**: Maximum number of passages to select
   - Prevents selecting too many small passages
   - Typical values: 8-16

3. **`stop_on_budget`**: Whether to strictly enforce budget
   - `True`: Skip passages that would exceed budget
   - `False`: Allow slight budget overruns

### Abstention Logic (Safe-Cover only)

```python
# Safe-Cover checks two conditions:
is_infeasible = (
    len(uncovered_facets) > 0 or  # Has uncovered facets
    dual_lower_bound > max_evidence_tokens  # Budget insufficient
)

# Abstain based on config:
if is_infeasible and abstain_on_infeasible:
    answer = "ABSTAINED"
```

**When `abstain_on_infeasible=True`:**
- System abstains if it cannot satisfy all coverage constraints under budget
- Shows true Safe-Cover behavior (strict risk control)

**When `abstain_on_infeasible=False`:**
- System returns best-effort solution even if some facets are uncovered
- May fall back to Pareto if `fallback_to_pareto=True`

### Evidence Token Tracking

The pipeline now tracks evidence tokens separately:

```python
result = pipeline.process_query(query, mode="pareto")

# Access metrics
total_tokens = result.tokens_used  # Evidence + answer generation
evidence_tokens = result.metrics.get("evidence_tokens", 0)  # Evidence only
answer_tokens = total_tokens - evidence_tokens  # Answer generation only
```

This allows fair comparisons where answer budget is held constant.

## Extending Config Families

To add a new config preset:

```python
# In trident/config_families.py

PARETO_CUSTOM = ParetoConfig(
    budget=3000,
    max_evidence_tokens=3000,
    max_units=14,
    stop_on_budget=True,
    relaxed_alpha=0.25,  # Adjust risk level
    use_vqc=True,
    use_bwk=True,
)

# Register it
PARETO_CONFIGS["pareto_custom"] = PARETO_CUSTOM
```

## Troubleshooting

### "Config not found" error
Make sure you're using the exact config name from the registry:
```python
from trident.config_families import ALL_CONFIGS
print(list(ALL_CONFIGS.keys()))
```

### Budget not being enforced
Check that `stop_on_budget=True` in your config. Also verify that `max_evidence_tokens` is set.

### Safe-Cover always abstaining
Try:
1. Reducing `per_facet_alpha` (less strict)
2. Increasing `max_evidence_tokens` (more budget)
3. Setting `abstain_on_infeasible=False` (accept partial solutions)
4. Setting `fallback_to_pareto=True` (use Pareto as fallback)

### Self-RAG using too many tokens
The config controls `k` (number of documents), but actual token usage depends on document lengths. Monitor `avg_evidence_tokens` to see actual usage.

## References

- [TRIDENT Paper](link-to-paper)
- [Self-RAG Paper](https://arxiv.org/abs/2310.11511)
- [Main README](README.md)
