# TRIDENT v1 — A Risk-Controlled, Budget-Aware RAG Framework

This document is a **review-ready, self-contained specification** of TRIDENT v1. TRIDENT unifies **budget-aware evidence selection** with **verifier-grounded calibration**, and exposes two serving regimes: a **certificate-bearing Safe-Cover mode** designed for auditable reliability, and a **Pareto mode** designed for throughput and cost–quality tradeoffs. The system is implemented with **dense retrieval + reranking**, **NLI-based verification**, **Mondrian conformal calibration**, and **end-to-end telemetry** for reproducible evaluation.

---

## 0) Executive Summary

- **Goal.** Improve answer quality and faithfulness under hard **token and latency** constraints by selecting **small evidence sets** that are **verifier-supported** rather than relying on naive top-k retrieval.
- **Core idea.** Convert verifier scores into **calibrated p-values** for "facet sufficiency," then use those p-values as the decision layer for (i) **certificate-bearing coverage** (Safe-Cover) or (ii) **budgeted utility maximization** (Pareto).
- **Serving regimes.**
    1. **Safe-Cover (certified):** attempts to cover mined facets using **fixed ex-ante thresholds** and logs **auditable certificates**; supports abstention when infeasible.
    2. **Pareto (uncertified):** optimizes a quality–cost tradeoff under explicit budgets (`budget`, `max_evidence_tokens`, `max_units`) and reports **Pareto curves**.
- **Engineering realism (aligned to code).**
    - Retrieval: dense `facebook/contriever` with top-k candidate pool (default: 100) + cross-encoder reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`, rerank_top_k=20).
    - Verifier: NLI `microsoft/deberta-v2-xlarge-mnli` with thresholding (default: `score_threshold=0.9` for pre-screen only) and LRU caching (10,000 entries).
    - Calibration: **conformal** (v1.0) with **Mondrian bins** (6 facet types × 3 length buckets × 3 retriever score buckets = 54 bins); `n_min=50` controls bin feasibility.
    - Telemetry: latency/tokens, cache stats, GPU utilization, per-stage traces, and per-facet certificate logs.

---

## 1) Problem Setup & Notation

### 1.1 Facet decomposition

Given a query q, a **facet miner** produces a set of reasoning requirements:

```
F(q) = {f₁, ..., fₘ}, f ∈ {ENTITY, RELATION, TEMPORAL, NUMERIC, BRIDGE_HOP1, BRIDGE_HOP2}
```

Each facet is represented as a **canonical template** (e.g., a typed relation or a bridge hop constraint). Pareto mode may attach optional weights `wf`; Safe-Cover treats facets as mandatory.

### 1.2 Candidate passages and costs

A retriever produces a candidate pool P with per-passage costs `c(p)` measured in **evidence/prompt tokens** (and optionally latency estimates). TRIDENT constrains selection by:

- **Hard budget** (Pareto): `pareto.budget`, plus caps `max_evidence_tokens` and `max_units`.
- **Hard cap** (Safe-Cover): `safe_cover.token_cap` and `stop_on_budget`.

### 1.3 Facet sufficiency event

For each facet–passage pair (f, p), define the sufficiency event:

```
Σ(p, f) = "p contains enough information to satisfy facet f."
```

This is the event the verifier and calibration are trying to make auditable; it is **not** defined in terms of downstream EM/F1.

**Operational Σ(p,f) Definition (Span-Level):**
- **ENTITY:** Mention of entity e (canonical or alias) with correct type is present in passage.
- **RELATION:** Span includes e₁, e₂ and predicates the typed relation (explicit or implicit).
- **TEMPORAL:** Event + normalized time; ISO-8601 within ±δ tolerance.
- **NUMERIC:** Quantity + value; unit-normalized within relative ε tolerance.
- **BRIDGE_HOP1:** (e₁, r₁, e_bridge) is asserted in the passage.
- **BRIDGE_HOP2:** (e_bridge, r₂, e₂) is asserted in the passage.

### 1.4 Verifier scores → calibrated p-values

A verifier (NLI) produces a score `s(p, f)`. A conformal calibrator (with Mondrian binning) maps `s(p, f)` to a p-value `π(p, f)` estimating the risk of **false sufficiency** (i.e., `Pr[¬Σ(p, f)]`) under the bin's exchangeability assumptions.

**Important implementation constraint:** deterministic conformal p-values have a minimum achievable value `1/(n_b + 1)` for bin size `n_b`; Safe-Cover must handle infeasible thresholds (see §3).

---

## 2) Safe-Cover Mode — Certificate-Bearing Coverage

Safe-Cover is the "auditable" regime: it aims to cover all mined facets when possible and produce a trace that a reviewer can follow.

### 2.1 Alpha semantics (authoritative, code-aligned)

The code exposes `safe_cover.per_facet_alpha` (legacy name). We interpret it as the **query-level** FWER target:

```python
# trident/config.py - SafeCoverConfig
# NOTE: per_facet_alpha is the QUERY-LEVEL FWER target (α_query).
# The name is legacy; it does NOT mean each facet gets this budget.

alpha_query := safe_cover.per_facet_alpha  # (legacy name; alias alpha_query exposed)
alpha_f := alpha_query / |F(q)|             # per-facet allocation
alpha_bar_f := alpha_f / T_f                # per-test threshold
```

These three values are logged in every certificate. Mis-tuning `per_facet_alpha` as a true "per-facet alpha" will inflate FWER.

### 2.2 Fixed coverage sets

For each passage p, define its covered facets:

```
C(p) = {f ∈ F : π(p, f) ≤ ᾱf}
```

Coverage sets are **fixed ex-ante** because `ᾱf` and `Tf` are predeclared. This is what prevents "moving thresholds" and makes the selection trace stable and reviewable.

### 2.3 Discrete feasibility policy for deterministic p-values (must-have)

Safe-Cover enforces a **discrete feasibility guard**. For any facet f and active bin b:

**If `alpha_bar_f < 1/(|N_b|+1)` then apply, in order:**
1. Switch `pvalue_mode="randomized"` for facet f; else
2. Merge bins per `merge_order = retriever_score → length → facet_type` until feasible; else
3. Abstain with `reason="pvalue_infeasible_small_bin"`

We log the branch taken and the final `bin_size`.

```python
# trident/config.py - SafeCoverConfig
feasibility_auto_switch_to_randomized: bool = True  # Auto-switch to randomized when infeasible
feasibility_auto_merge_bins: bool = True  # Auto-merge bins when infeasible
```

This prevents "Safe-Cover fails everywhere" artifacts that look like algorithmic weakness but are actually calibration arithmetic.

### 2.4 Objective and selection

Safe-Cover solves a min-cost cover:

```
min_{S ⊆ P} Σ_{p ∈ S} c(p)  s.t.  ⋃_{p ∈ S} C(p) ⊇ F
```

Implementation uses **cost-effective greedy** with deterministic tie-breaks (including a stable passage id fallback) and produces:

- the selected set S,
- per-facet certificates for each covered facet,
- and an abstention reason if infeasible or budget-exhausted.

### 2.5 Dual lower bound inside greedy (iteration-level)

At **each greedy iteration**, compute (or incrementally update) `LB_dual` on the **residual uncovered facets** and compare against `B_ctx - cost_ctx` (remaining budget).

If `LB_dual > B_ctx - cost_ctx`, abstain with `INFEASIBILITY_PROVEN`, logging `(LB_dual, B_remaining)`.

### 2.6 Abstention reasons

```python
class AbstentionReason(Enum):
    NONE = "none"
    NO_COVERING_PASSAGES = "no_covering_passages"  # ∃f ∈ F_uncov with no p : f ∈ C(p)
    INFEASIBILITY_PROVEN = "infeasibility_proven"  # LB_dual > B_ctx - cost_ctx
    BUDGET_EXHAUSTED = "budget_exhausted"  # No candidates fit within remaining budget
    PVALUE_INFEASIBLE_SMALL_BIN = "pvalue_infeasible_small_bin"  # α_bar_f < 1/(n_b+1) cannot resolve
```

### 2.7 Distributional constraint (Safe-Cover)

**`use_vqc=False`, `use_bwk=False` in Safe-Cover mode.**

VQC (Verifier-driven Query Compiler) and BwK (Bandits with Knapsacks) are **DISABLED** in Safe-Cover mode because they change the candidate distribution and invalidate selection-conditional calibration.

They may be used in Pareto mode only. The fallback path (Safe-Cover → Pareto) may use VQC/BwK, but the resulting answer is explicitly marked as uncertified (`fallback_from='safe_cover'`).

---

## 3) Calibration, Monitoring, and Validity Scope

TRIDENT does **not** claim distribution-free validity under arbitrary adaptivity. Instead it makes the validity conditions explicit and auditable.

### 3.1 BuildCalibrator (selection-conditional replay)

Replay the deployed retrieval stack on a labeled corpus:

1. For each query q in D_cal:
   - a) Retrieve with the exact retriever + index snapshot
   - b) Rerank/shortlist with the exact shortlister, T_f, and tie-breaks
   - c) Score every shortlisted (p, f) with the deployed verifier
   - d) Place negative scores (Σ=0) into the Mondrian bin b(p, f)

2. Enforce `n_min` via the fixed merge order (score → length → type); store `|N_b|` and sorted scores

**Versions recorded:** `retriever_hash`, `index_snapshot_id`, `shortlister_hash`, `verifier_hash`, `bin_spec_hash`, `calibration_corpus_hash`

Certificates are **invalid** if any version differs.

### 3.2 Calibration configuration (code-aligned)

```python
# trident/calibration.py
@dataclass
class CalibrationConfig:
    method: str = "conformal"      # "conformal", "isotonic", "platt", "beta"
    version: str = "v1.0"
    use_mondrian: bool = True
    n_min: int = 50                # Minimum per-bin negatives before merging
    pvalue_mode: str = "deterministic"  # "deterministic" or "randomized"
    label_noise_epsilon: float = 0.0
```

### 3.3 Mondrian binning strategy (54-bin grid)

```python
facet_types = ["ENTITY", "RELATION", "TEMPORAL", "NUMERIC", "BRIDGE_HOP1", "BRIDGE_HOP2"]
length_buckets = [(0, 50, "short"), (50, 150, "medium"), (150, 10000, "long")]
retriever_score_buckets = [(0.0, 0.33, "low"), (0.33, 0.67, "medium"), (0.67, 1.0, "high")]
# Bin key format: "{FACET_TYPE}_{length_bin}_{retriever_bin}"
# Example: "ENTITY_short_high", "RELATION_medium_low"
```

### 3.4 Conformal p-value computation

**Deterministic Mode** (Conservative):
```
π(p, f) = (1 + #{s' ∈ N_b : s' ≥ s(p, f)}) / (1 + n)
```
Property: Super-uniform (Pr[π ≤ t | ¬Σ] ≤ t + 1/(n+1))

**Randomized Mode** (Exact):
```
π_rand(p, f) = (1 + #{s' > s} + U · #{s' = s}) / (1 + n), U ~ Unif(0,1)
```
Property: Exact super-uniformity (Pr[π_rand ≤ t | ¬Σ] = t)

**Bin Merging Strategy:**
1. If `n_negatives < n_min` (50), merge to coarser bin
2. **Order of Merging**: retriever_score → length → facet_type
3. Recursively merge until all bins have ≥ n_min negatives

### 3.5 Monitoring is future-episode only

Monitoring (PSI/KL drift, violation buffers) is permitted to **trigger alarms, threshold shrinkage, or recalibration for future episodes**, but it must not change thresholds **mid-episode** in Safe-Cover unless you implement an explicit round-wise α-spending design.

```python
# Monitoring config (SafeCoverConfig)
monitor_drift: bool = False
psi_threshold: float = 0.25
kl_threshold: float = 0.5
violation_multiplier: float = 2.0
threshold_shrink_factor: float = 0.7
recalibration_buffer_size: int = 1000
```

---

## 4) Pareto Mode — Budgeted Utility (No Certificates)

Pareto mode is the throughput regime: it trades hard guarantees for performance and delivers cost–quality curves.

### 4.1 Budget knobs (matches code)

```python
# trident/config.py
@dataclass
class ParetoConfig:
    budget: int = 2000                   # Total token budget
    relaxed_alpha: float = 0.3           # Relaxed threshold for coverage
    weight_default: float = 1.0          # Default facet weight
    stop_on_budget: bool = True
    max_evidence_tokens: Optional[int] = None
    max_units: Optional[int] = None
    use_vqc: bool = True                 # Verifier-driven Query Compiler
    use_bwk: bool = True                 # Bandits with Knapsacks
    max_vqc_iterations: int = 3
    bwk_exploration_bonus: float = 0.1
```

Pareto reports empirical performance and cost curves; it does not emit coverage certificates.

### 4.2 Optional modules (explicitly scoped)

**VQC (Verifier-driven Query Compiler):**
- Triggers when: `use_vqc=True` AND uncovered_facets AND cost < 80% budget
- Generates query rewrites for uncovered facets
- Max iterations: `max_vqc_iterations` (default: 3)

**BwK (Bandits with Knapsacks):**
- Arms: `add_evidence`, `hop_expand`, `re_read`, `vqc_rewrite`
- Uses consumption-aware UCB
- Exploration bonus: `bwk_exploration_bonus` (default: 0.1)

These modules are **Pareto-only**. They must NOT be used with Safe-Cover unless explicitly redesigned for multi-round α-spending.

---

## 5) Certificate Schema (Explicit)

Per covered facet, the certificate contains:

```python
@dataclass
class CoverageCertificate:
    facet_id: str           # Unique identifier for the facet
    facet_type: str         # ENTITY, RELATION, TEMPORAL, NUMERIC, BRIDGE_HOP1, BRIDGE_HOP2
    passage_id: str         # ID of the passage covering this facet
    p_value: float          # Calibrated p-value for this (passage, facet) pair
    threshold: float        # ᾱ_f used for this coverage decision
    alpha_facet: float      # α_f (per-facet error budget = α_query / |F(q)|)
    alpha_query: float      # Query-level FWER target
    t_f: int               # Max tests per facet (Bonferroni budget T_f)
    bin: str               # Calibration bin key used
    bin_size: int          # Number of negatives in the calibration bin
    pvalue_mode: str       # "deterministic" or "randomized"
    calibrator_version: str
    retriever_version: str
    shortlister_version: str
    verifier_version: str
    timestamp: float       # Unix timestamp
```

Certificates are **INVALID** if any version differs from calibration time.

---

## 6) Runtime System & SLO Controls (Implementation View)

### 6.1 Retrieval + reranking (current stack)

```python
# trident/retrieval.py
@dataclass
class RetrievalConfig:
    method: str = "dense"               # dense, sparse, hybrid
    encoder_model: str = "facebook/contriever"
    top_k: int = 100                    # Initial retrieval pool
    rerank_top_k: int = 20              # Final reranked results
    use_reranker: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
```

**Retrieval methods:**
- **Dense**: Contriever embeddings with cosine similarity
- **Sparse**: BM25 (k1=1.2, b=0.75)
- **Hybrid**: Reciprocal rank fusion (alpha=0.5)

### 6.2 Verification and caching

```python
# trident/nli_scorer.py
@dataclass
class NLIConfig:
    model_name: str = "microsoft/deberta-v2-xlarge-mnli"
    batch_size: int = 32
    max_length: int = 512
    use_cache: bool = True
    cache_size: int = 10000
    # NOTE: score_threshold is for PRE-SCREEN only (computational pruning).
    # Coverage decisions use calibrated p-values, NOT this threshold.
    score_threshold: float = 0.9
```

**Scoring formula:**
```
sufficiency_score = entailment_score - 0.5 * contradiction_score
```

**Cache:** LRU cache keyed by `(passage_id, facet_id, model_version)` to stabilize cost and enable fair comparisons.

### 6.3 Token/latency accounting and generation budget

TRIDENT reports decomposed usage:

- evidence tokens (selected passages),
- overhead prompt tokens (templates/system),
- verifier tokens (if any),
- generation tokens,
- and end-to-end latency percentiles.

**Generation budget (B_gen):**
`B_gen` is enforced at decode time (`max_new_tokens`). It is NOT part of the covering LP and does not affect dual-LB soundness. We log `hit_max_new_tokens: {true|false}`.

---

## 7) Theory — Claims and Their Scope

1. **Greedy approximation for set cover** holds for Safe-Cover **because** C(p) sets are fixed ex-ante.
2. **Risk control semantics** apply to the calibrated sufficiency tests **under the stated calibration contract** (selection-conditional, version-matched).
3. **Dual-LB abstention** is sound when computed on the residual uncovered facets and compared against remaining budget at each iteration.
4. Pareto is optimization-first and reports empirical Pareto curves; no formal certificates are claimed there.
5. **Holm adjustment is reporting-only.** If mentioned, it is **post-hoc reporting** and does **NOT** alter coverage sets or greedy selection.

---

## 8) Evaluation Protocol (Aligned to code)

### 8.1 Supported datasets

```python
SUPPORTED_DATASETS = {
    'hotpot_qa': 'hotpot_qa',           # HotpotQA (fullwiki)
    'musique': 'musique',               # MuSiQue
    '2wiki': '2WikiMultiHopQA',         # 2WikiMultiHopQA
    'natural_questions': 'natural_questions',
    'trivia_qa': 'trivia_qa',
}
```

### 8.2 Metrics computed

```python
@dataclass
class EvaluationMetrics:
    exact_match: float          # Exact match accuracy
    f1_score: float            # Token-level F1
    support_em: float          # Supporting facts EM
    support_f1: float          # Supporting facts F1
    faithfulness: float        # Answer faithful to passages
    abstention_rate: float     # Fraction of abstained queries
    avg_tokens_used: float     # Average token consumption
    avg_latency_ms: float      # Average latency in milliseconds
    coverage_rate: float       # Average facet coverage
```

### 8.3 Baselines

```python
BASELINES = {
    'selfrag': FullSelfRAGAdapter,
    'graphrag': FullGraphRAGAdapter,
    'ketrag_reimpl': FullKETRAGReimplAdapter,
    'ketrag_official': FullKETRAGAdapter,
    'vanillarag': FullVanillaRAGAdapter,
    'hipporag': FullHippoRAGAdapter,
}
```

### 8.4 Evaluation/ablation additions (for reviewer trust)

- **Miner recall estimate `R_miner`** on a labeled subset, reported next to EM/F1
- **Bin feasibility rate** (fraction of facets that needed randomization/merge)
- **Anti-conservativeness check:** simulate with oracle label buffer and report empirical FWER vs target
- **Pareto reproducibility:** publish YAML configs & seeds for each Pareto point; show per-stage latency breakdown

---

## 9) Facet Pre-Pruning (§11.3) — Out of Scope for Guarantees

Facet pre-pruning drops hypotheses **after** seeing verifier evidence, which changes the multiple-testing family.

**Policy (code-aligned):**
```python
# trident/config.py - SafeCoverConfig
# Facet Pre-Pruning (OFF by default - Section 11.3)
# NOTE: Pre-pruning drops hypotheses after seeing verifier evidence, which
# changes the multiple-testing family. It is OFF by default and out-of-scope
# for FWER guarantees. If enabled, treat it as part of the miner (pre-verification).
enable_spurious_facet_prepass: bool = False  # OFF by default
spurious_alpha: float = 1e-4
spurious_max_tests: int = 3
```

If you keep it enabled, treat it as **part of the miner** (pre-verification) or require it to be monotone and external.

---

## 10) Config ↔ Math Mapping Table

| Math Symbol | Config Field | Default | Description |
|-------------|--------------|---------|-------------|
| α_query | `safe_cover.per_facet_alpha` | 0.05 | Query-level FWER target |
| α_f | Computed: `alpha_query / \|F(q)\|` | - | Per-facet allocation |
| ᾱ_f | Computed: `alpha_f / T_f` | - | Per-test threshold |
| T_f | `per_facet_configs[f].max_tests` | 10 | Max tests per facet |
| B_ctx | `safe_cover.token_cap` | 2000 | Context budget cap |
| N_min | `calibration.n_min` | 50 | Min negatives per bin |
| ε | `calibration.label_noise_epsilon` | 0.0 | Label noise rate |
| B_pareto | `pareto.budget` | 2000 | Pareto mode budget |
| α_relaxed | `pareto.relaxed_alpha` | 0.3 | Relaxed coverage threshold |

---

## 11) Appendix-Level Details

### 11.1 Tie-breaking in greedy

Priority is a tuple:
```
(|C(p) \ Covered| / c(p), -c(p), -mean_{f ∈ C(p) \ Covered} π(p,f), pid)
```
(i.e., coverage-per-token, then cheaper, then smaller mean p-value, then ascending passage ID for determinism across shards).

### 11.2 Named configuration families

```python
# Pareto configs
PARETO_CHEAP_1500 = ParetoConfig(budget=1500, max_units=8, use_vqc=False, use_bwk=False)
PARETO_CHEAP_2000 = ParetoConfig(budget=2000, max_units=10, use_vqc=False, use_bwk=False)
PARETO_MID_2500 = ParetoConfig(budget=2500, max_units=12, use_vqc=True, use_bwk=True)

# Safe-Cover configs
SAFE_COVER_LOOSE_4000 = SafeCoverConfig(per_facet_alpha=0.05, max_evidence_tokens=4000, max_units=16)
SAFE_COVER_EQUAL_2500 = SafeCoverConfig(per_facet_alpha=0.02, max_evidence_tokens=2500, max_units=12)
SAFE_COVER_EQUAL_2000 = SafeCoverConfig(per_facet_alpha=0.02, max_evidence_tokens=2000, max_units=10)
```

---

## 12) Component Summary Table

| Component | Default Model/Parameter | Key Configurable | Range/Options |
|-----------|------------------------|------------------|----------------|
| **Retrieval** | facebook/contriever + ms-marco-MiniLM | top_k=100, rerank_top_k=20 | dense/sparse/hybrid |
| **NLI/Verifier** | microsoft/deberta-v2-xlarge-mnli | batch_size=32, cache_size=10000 | Pre-screen threshold=0.9 |
| **Calibration** | Mondrian conformal (54 bins) | pvalue_mode, n_min=50 | deterministic/randomized |
| **Safe-Cover** | FWER α=0.05, token_cap=2000 | per_facet_alpha, token_cap | No VQC/BwK |
| **Pareto** | Lazy greedy, budget=2000 | relaxed_alpha=0.3, use_vqc/bwk | 400-2500 tokens |
| **VQC** | Query rewriting | max_vqc_iterations=3 | Pareto-only |
| **BwK** | Contextual bandits UCB | exploration_bonus=0.1 | 4 arms, Pareto-only |
| **Baselines** | Self-RAG 7B, GraphRAG, KET-RAG | ndocs, k values | Standardized interface |

---

## 13) Why this is Future-Proof

- **Certificates, not trends.** As models/retrievers change, the **certificate pathway** (Safe-Cover) remains valid with re-calibration and monitoring.
- **Strict separation of regimes.** New components (e.g., reasoning LLMs, vector DBs) can plug into **Pareto mode** without touching the proofs in Safe-Cover.
- **Bounded risk + bounded cost.** Early-abstain and α-spending make it suitable for **compliance-sensitive** domains (legal/healthcare) where **fail-safe** behavior is critical.
- **Composable.** VQC, BwK, graph hops are **optional modules** with documented contracts and scopes.
