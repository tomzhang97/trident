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
    - Verifier: NLI `microsoft/deberta-v2-xlarge-mnli` with thresholding (default: `score_threshold=0.9`) and LRU caching (10,000 entries).
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

### 1.4 Verifier scores → calibrated p-values

A verifier (NLI) produces a score `s(p, f)`. A conformal calibrator (with Mondrian binning) maps `s(p, f)` to a p-value `π(p, f)` estimating the risk of **false sufficiency** (i.e., `Pr[¬Σ(p, f)]`) under the bin's exchangeability assumptions.

**Important implementation constraint:** deterministic conformal p-values have a minimum achievable value `1/(n_b + 1)` for bin size `n_b`; Safe-Cover must handle infeasible thresholds (see §3).

---

## 2) Safe-Cover Mode — Certificate-Bearing Coverage

Safe-Cover is the "auditable" regime: it aims to cover all mined facets when possible and produce a trace that a reviewer can follow.

### 2.1 α semantics (code-aligned)

The code exposes the following configuration in `SafeCoverConfig`:

```python
# trident/config.py
@dataclass
class SafeCoverConfig:
    per_facet_alpha: float = 0.05      # Query-level FWER target (α_query)
    token_cap: Optional[int] = 2000    # Hard context budget cap (B_ctx)
    max_evidence_tokens: Optional[int] = None
    max_units: Optional[int] = None
    early_abstain: bool = True
    abstain_on_infeasible: bool = False
    stop_on_budget: bool = True
    dual_tolerance: float = 1e-6
    coverage_threshold: float = 0.15
    fallback_to_pareto: bool = True
    pvalue_mode: str = "deterministic"  # or "randomized"
    monitor_drift: bool = False
    psi_threshold: float = 0.25
```

The interpretation is:
- **Query-level target:** `α_query := safe_cover.per_facet_alpha`
- **Per-facet allocation:** `αf := α_query / |F(q)|`
- **Per-test threshold:** `ᾱf := αf / Tf`, where `Tf` is the **maximum number of verifier probes** allowed for that facet.

This mapping is logged in certificates to prevent silent misconfiguration.

### 2.2 Fixed coverage sets

For each passage p, define its covered facets:

```
C(p) = {f ∈ F : π(p, f) ≤ ᾱf}
```

Coverage sets are **fixed ex-ante** because `ᾱf` and `Tf` are predeclared. This is what prevents "moving thresholds" and makes the selection trace stable and reviewable.

### 2.3 Feasibility guard for deterministic p-values (must-have)

Safe-Cover must enforce a **discrete feasibility guard**:

- If deterministic p-values are used, and `ᾱf < 1/(n_b + 1)` in any bin actually used for facet f, then certification is **impossible** for that facet under that bin.
- The system must apply a policy: switch that facet to randomized p-values, or merge bins until feasible, or abstain with a clear reason (logged).

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

### 2.5 Early abstention with a dual lower bound

Safe-Cover computes a **dual lower bound** `LB_dual` on the remaining uncovered facets and compares it against the **remaining** budget each iteration. If `LB_dual > B_remaining`, the system abstains with an infeasibility certificate and logs the bound trace.

**Abstention reasons** (from code):
- `NO_COVERING_PASSAGES` - No passage covers an uncovered facet
- `INFEASIBILITY_PROVEN` - Dual lower bound exceeds budget
- `BUDGET_EXHAUSTED` - No candidates fit within remaining budget

---

## 3) Calibration, Monitoring, and Validity Scope

TRIDENT does **not** claim distribution-free validity under arbitrary adaptivity. Instead it makes the validity conditions explicit and auditable.

### 3.1 Calibration configuration (code-aligned)

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

### 3.2 Mondrian binning strategy (54-bin grid)

```python
facet_types = ["ENTITY", "RELATION", "TEMPORAL", "NUMERIC", "BRIDGE_HOP1", "BRIDGE_HOP2"]
length_buckets = [(0, 50, "short"), (50, 150, "medium"), (150, 10000, "long")]
retriever_score_buckets = [(0.0, 0.33, "low"), (0.33, 0.67, "medium"), (0.67, 1.0, "high")]
# Bin key format: "{FACET_TYPE}_{length_bin}_{retriever_bin}"
# Example: "ENTITY_short_high", "RELATION_medium_low"
```

### 3.3 Conformal p-value computation

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

### 3.4 Selection-conditional calibration (versioned)

Calibration must be **selection-conditional** on the deployed retrieval/shortlisting policy. TRIDENT therefore treats the following as part of the certificate contract and logs versions/hashes:

- retriever encoder + index snapshot,
- reranker model/version,
- verifier model/version,
- bin specification + `n_min` policy,
- calibration corpus identifier.

Certificates are **invalid** if these versions differ at test time.

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

### 4.2 Named configuration families (from code)

```python
# trident/config_families.py
PARETO_CHEAP_1500 = ParetoConfig(budget=1500, max_units=8, use_vqc=False, use_bwk=False)
PARETO_CHEAP_2000 = ParetoConfig(budget=2000, max_units=10, use_vqc=False, use_bwk=False)
PARETO_MID_2500 = ParetoConfig(budget=2500, max_units=12, use_vqc=True, use_bwk=True)

SAFE_COVER_LOOSE_4000 = SafeCoverConfig(per_facet_alpha=0.05, max_evidence_tokens=4000, max_units=16)
SAFE_COVER_EQUAL_2500 = SafeCoverConfig(per_facet_alpha=0.02, max_evidence_tokens=2500, max_units=12)
SAFE_COVER_EQUAL_2000 = SafeCoverConfig(per_facet_alpha=0.02, max_evidence_tokens=2000, max_units=10)
```

### 4.3 Optional modules (explicitly scoped)

**VQC (Verifier-driven Query Compiler):**
- Triggers when: `use_vqc=True` AND uncovered_facets AND cost < 80% budget
- Generates query rewrites for uncovered facets
- Max iterations: `max_vqc_iterations` (default: 3)

**BwK (Bandits with Knapsacks):**
- Arms: `add_evidence`, `hop_expand`, `re_read`, `vqc_rewrite`
- Uses consumption-aware UCB
- Exploration bonus: `bwk_exploration_bonus` (default: 0.1)

These modules are **Pareto-only** unless explicitly redesigned for Safe-Cover with multi-round α-spending.

---

## 5) Runtime System & SLO Controls (Implementation View)

### 5.1 Retrieval + reranking (current stack)

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

### 5.2 Verification and caching

```python
# trident/nli_scorer.py
@dataclass
class NLIConfig:
    model_name: str = "microsoft/deberta-v2-xlarge-mnli"
    batch_size: int = 32
    max_length: int = 512
    use_cache: bool = True
    cache_size: int = 10000
    score_threshold: float = 0.9
```

**Scoring formula:**
```
sufficiency_score = entailment_score - 0.5 * contradiction_score
```

**Cache:** LRU cache keyed by `(passage_id, facet_id, model_version)` to stabilize cost and enable fair comparisons.

### 5.3 Token/latency accounting (must be explicit)

TRIDENT reports decomposed usage:

- evidence tokens (selected passages),
- overhead prompt tokens (templates/system),
- verifier tokens (if any),
- generation tokens,
- and end-to-end latency percentiles.

This avoids the common confusion where "token cap" is mistaken for total prompt+system tokens.

---

## 6) Interfaces (Reproducibility and Audits)

### 6.1 Inputs (conceptual)

- Query + mined facets
- Mode = {safe_cover, pareto, both}
- Budgets and caps
- Calibrator/version identifiers

### 6.2 Outputs (from code)

```python
# trident/pipeline.py
class PipelineOutput:
    answer: str
    selected_passages: List[Dict]  # pid, text, cost, covered_facets
    certificates: Optional[List]   # For Safe-Cover only
    abstained: bool
    tokens_used: int
    latency_ms: float
    metrics: Dict[str, float]
    mode: str                      # "safe_cover" or "pareto"
    facets: List[Dict]
    trace: Dict                    # Telemetry
```

---

## 7) Theory — Claims and Their Scope

1. **Greedy approximation for set cover** holds for Safe-Cover **because** C(p) sets are fixed ex-ante.
2. **Risk control semantics** apply to the calibrated sufficiency tests **under the stated calibration contract** (selection-conditional, version-matched).
3. **Dual-LB abstention** is sound when computed on the residual uncovered facets and compared against remaining budget.
4. Pareto is optimization-first and reports empirical Pareto curves; no formal certificates are claimed there.

---

## 8) Evaluation Protocol (Aligned to code)

### 8.1 Supported datasets

```python
# Loaded via HuggingFace or custom loaders
# Auto-detection from data_path
SUPPORTED_DATASETS = {
    'hotpot_qa': 'hotpot_qa',           # HotpotQA (fullwiki)
    'musique': 'musique',               # MuSiQue
    '2wiki': '2WikiMultiHopQA',         # 2WikiMultiHopQA (NEW)
    'natural_questions': 'natural_questions',
    'trivia_qa': 'trivia_qa',
}
```

### 8.2 Metrics computed

```python
# trident/evaluation.py
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

### 8.3 Baselines (from code)

```python
# baselines/full_baseline_interface.py
BASELINES = {
    'selfrag': FullSelfRAGAdapter,      # Self-RAG with reflection tokens
    'vanillarag': FullVanillaRAGAdapter,       # Simple dense retrieval
    'hipporag': FullHippoRAGAdapter,           # HippoRAG
}
```

### 8.4 Primary evaluation benchmarks

- **Benchmarks:** HotpotQA, 2WikiMultiHopQA, MuSiQue
- **Models:** Meta-Llama-3-8B-Instruct and Qwen3-8B (configurable)
- **Baselines:** VanillaRAG, Self-RAG, HippoRAG
- **Metrics:** EM/F1 + abstention rate; cost (token decomposition) + latency (mean/median/p95)
- **Safe-Cover audit:** certificate validity checks, infeasibility abstention analysis
- **Pareto curves:** quality vs evidence tokens / end-to-end latency across budgets

---

## 9) Practical Considerations & Risks (Reality-checked)

- **Facet miner recall limits what Safe-Cover can certify.** Report miner recall estimate.
- **Calibration bins can be too small.** Enforce feasibility guard; log bin sizes (n_min=50).
- **Token caps must be unambiguous.** Separate evidence cap from total prompt tokens.
- **Determinism matters.** Stable tie-breaks and version logging are required for paper-grade reproducibility.

---

## 10) What's New (Positioning, kept honest)

- **Not just "verify then answer."** TRIDENT makes verification **calibrated and auditable**, and separates certified vs throughput regimes cleanly.
- **Not just "graph/memory RAG."** TRIDENT's organizing principle is **budgeted, verifier-grounded selection**, compatible with many retrieval backends.
- **Operational accountability.** Versioned calibration + certificate logs + dual-bound abstention make it suited to settings where "why did you trust this evidence?" matters.

---

## 11) Appendix-Level Details

### 11.1 Tie-breaking in greedy

Priority is a tuple:
```
(|C(p) \ Covered| / c(p), -c(p), -mean_{f ∈ C(p) \ Covered} π(p,f))
```
(i.e., coverage-per-token, then cheaper, then smaller mean p-value).

### 11.2 Facet definitions

- **ENTITY:** canonical mention resolution present.
- **RELATION:** typed predicate with subject/object roles.
- **TEMPORAL/NUMERIC:** compatible units and normalization.
- **BRIDGE:** existence of a 2-hop chain (E₁ → E' → E₂) with type-consistent schema and temporal compatibility.

### 11.3 Spurious facet pre-pass

Allocate `α̃f << ᾱf` (e.g., 1e-4) with `Tf_pre ≤ 3`; drop facets failing all pre-tests; log drops.

---

## 12) Component Summary Table

| Component | Default Model/Parameter | Key Configurable | Range/Options |
|-----------|------------------------|------------------|----------------|
| **Retrieval** | facebook/contriever + ms-marco-MiniLM | top_k=100, rerank_top_k=20 | dense/sparse/hybrid |
| **NLI/Verifier** | microsoft/deberta-v2-xlarge-mnli | batch_size=32, cache_size=10000 | score_threshold=0.9 |
| **Calibration** | Mondrian conformal (54 bins) | pvalue_mode, n_min=50 | deterministic/randomized |
| **Safe-Cover** | FWER α=0.05, token_cap=2000 | per_facet_alpha, token_cap | 50-4000 tokens |
| **Pareto** | Lazy greedy, budget=2000 | relaxed_alpha=0.3, use_vqc/bwk | 400-2500 tokens |
| **VQC** | Query rewriting | max_vqc_iterations=3 | Pareto-only |
| **BwK** | Contextual bandits UCB | exploration_bonus=0.1 | 4 arms |
| **Baselines** | Self-RAG 7B | ndocs, k values | Standardized interface |

---

## 13) Why this is Future-Proof

- **Certificates, not trends.** As models/retrievers change, the **certificate pathway** (Safe-Cover) remains valid with re-calibration and monitoring.
- **Strict separation of regimes.** New components (e.g., reasoning LLMs, vector DBs) can plug into **Pareto mode** without touching the proofs in Safe-Cover.
- **Bounded risk + bounded cost.** Early-abstain and α-spending make it suitable for **compliance-sensitive** domains (legal/healthcare) where **fail-safe** behavior is critical.
- **Composable.** VQC, BwK, graph hops are **optional modules** with documented contracts and scopes.
