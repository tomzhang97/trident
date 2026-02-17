"""Shared reporting utilities for rebuttal experiments.

Every experiment emits a single JSON with:
  - metadata: dataset, split, budget, mode, backbone, seed, timestamp
  - metrics: EM/F1, abstention, evidence_tokens, latency stats
  - compute: total_pairs_scored, num_batches (where relevant)

And prints a table sorted by delta-F1 vs baseline.
"""

from __future__ import annotations

import json
import os
import re
import string
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Standard JSON envelope
# ---------------------------------------------------------------------------

@dataclass
class ExperimentMetadata:
    experiment_id: str
    dataset: str
    split: str = "dev"
    budget: int = 500
    mode: str = "pareto"
    backbone: str = ""
    seed: int = 42
    limit: Optional[int] = None
    timestamp: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")


@dataclass
class ExperimentReport:
    """Standard report envelope for every rebuttal experiment."""

    metadata: ExperimentMetadata
    metrics: Dict[str, Any] = field(default_factory=dict)
    compute: Dict[str, Any] = field(default_factory=dict)
    per_query: List[Dict[str, Any]] = field(default_factory=list)
    summary_table: str = ""

    # ------------------------------------------------------------------
    def save(self, output_dir: str) -> str:
        """Write report JSON and return path."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        fname = f"{self.metadata.experiment_id}.json"
        path = out / fname
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=_json_default)
        return str(path)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": asdict(self.metadata),
            "metrics": self.metrics,
            "compute": self.compute,
            "per_query": self.per_query,
            "summary_table": self.summary_table,
        }


def _json_default(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, set):
        return sorted(obj)
    return str(obj)


# ---------------------------------------------------------------------------
# Metric helpers (stand-alone, no model loading)
# ---------------------------------------------------------------------------

def normalize_answer(text: str) -> str:
    """Normalize for EM/F1 (official 2WikiMultiHop order)."""
    text = re.sub(r"\b(a|an|the)\b", " ", text, flags=re.IGNORECASE)
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred: str, gold: str) -> float:
    pred_norm = normalize_answer(pred)
    gold_norm = normalize_answer(gold)
    special = {"yes", "no", "noanswer"}
    if pred_norm in special or gold_norm in special:
        return float(pred_norm == gold_norm)
    pt = pred_norm.split()
    gt = gold_norm.split()
    if not pt or not gt:
        return 0.0
    common = Counter(pt) & Counter(gt)
    n = sum(common.values())
    if n == 0:
        return 0.0
    prec = n / len(pt)
    rec = n / len(gt)
    return 2 * prec * rec / (prec + rec)


def compute_em_f1(
    predictions: Sequence[str],
    references: Sequence[str],
) -> Tuple[float, float]:
    """Return (mean_em, mean_f1) over a batch."""
    ems = [exact_match(p, r) for p, r in zip(predictions, references)]
    f1s = [f1_score(p, r) for p, r in zip(predictions, references)]
    return float(np.mean(ems)), float(np.mean(f1s))


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: Sequence[float],
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Return (mean, ci_lower, ci_upper)."""
    rng = np.random.RandomState(seed)
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return 0.0, 0.0, 0.0
    means = [float(rng.choice(arr, size=len(arr), replace=True).mean()) for _ in range(n_bootstrap)]
    alpha = 1 - ci
    lo = float(np.percentile(means, alpha / 2 * 100))
    hi = float(np.percentile(means, (1 - alpha / 2) * 100))
    return float(arr.mean()), lo, hi


# ---------------------------------------------------------------------------
# Latency percentile helpers
# ---------------------------------------------------------------------------

def latency_percentiles(
    latencies_ms: Sequence[float],
) -> Dict[str, float]:
    arr = np.asarray(latencies_ms, dtype=float)
    if len(arr) == 0:
        return {"p50": 0.0, "p95": 0.0, "mean": 0.0}
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "mean": float(arr.mean()),
    }


# ---------------------------------------------------------------------------
# Delta-F1 comparison table
# ---------------------------------------------------------------------------

def delta_f1_table(
    rows: List[Dict[str, Any]],
    baseline_key: str = "top_k",
) -> str:
    """Build a markdown table sorted by delta-F1 vs *baseline_key* row.

    Each row dict must contain at least: {"label", "em", "f1"}.
    Optional keys: "abstention_rate", "avg_evidence_tokens", "latency_p50".
    """
    baseline_f1 = 0.0
    for r in rows:
        if r.get("label") == baseline_key:
            baseline_f1 = r["f1"]
            break

    for r in rows:
        r["delta_f1"] = r["f1"] - baseline_f1

    rows_sorted = sorted(rows, key=lambda r: r["delta_f1"], reverse=True)

    hdr = "| Method | EM | F1 | dF1 | Abstain | EvTok | Lat p50 |"
    sep = "|--------|----|----|-----|---------|-------|---------|"
    lines = [hdr, sep]
    for r in rows_sorted:
        lines.append(
            f"| {r['label']} "
            f"| {r['em']:.3f} "
            f"| {r['f1']:.3f} "
            f"| {r['delta_f1']:+.3f} "
            f"| {r.get('abstention_rate', 0):.3f} "
            f"| {r.get('avg_evidence_tokens', 0):.0f} "
            f"| {r.get('latency_p50', 0):.0f} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stage-level timing context manager
# ---------------------------------------------------------------------------

class StageTimer:
    """Accumulates wall-clock time per named stage."""

    def __init__(self):
        self.stages: Dict[str, float] = {}
        self._stack: List[Tuple[str, float]] = []

    def start(self, name: str):
        self._stack.append((name, time.perf_counter()))

    def stop(self):
        if not self._stack:
            return
        name, t0 = self._stack.pop()
        elapsed = (time.perf_counter() - t0) * 1000  # ms
        self.stages[name] = self.stages.get(name, 0.0) + elapsed

    def as_dict(self) -> Dict[str, float]:
        return {k: round(v, 2) for k, v in self.stages.items()}
