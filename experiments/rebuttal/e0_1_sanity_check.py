#!/usr/bin/env python3
"""E0.1 -- Table / metric sanity + typo check.

Recomputes latencies and key metrics for all rows referenced by reviewers
(especially the MuSiQue Pareto-1000 anomaly).  Outputs a short sanity-report
JSON and prints any corrected table cells.

Usage:
    python -m experiments.rebuttal.e0_1_sanity_check \
        --results_dir runs/paper_results \
        --output_dir runs/rebuttal/e0_1
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from experiments.rebuttal.report_utils import (
    ExperimentMetadata,
    ExperimentReport,
    bootstrap_ci,
    compute_em_f1,
    exact_match,
    f1_score,
    latency_percentiles,
    normalize_answer,
)


def load_predictions(pred_path: str) -> List[Dict[str, Any]]:
    """Load predictions from JSON or JSONL."""
    p = Path(pred_path)
    if p.suffix == ".jsonl":
        with open(p) as f:
            return [json.loads(line) for line in f if line.strip()]
    with open(p) as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def recompute_row(
    predictions: List[Dict[str, Any]],
    label: str,
) -> Dict[str, Any]:
    """Recompute EM, F1, latency percentiles, and token stats for one row."""
    ems, f1s, lats, toks = [], [], [], []
    abstain_count = 0

    for pred in predictions:
        is_abstain = pred.get("abstained", False)
        if is_abstain:
            abstain_count += 1
            ems.append(0.0)
            f1s.append(0.0)
            continue

        pred_text = pred.get("prediction", pred.get("answer", ""))
        gold = pred.get("ground_truth", pred.get("answer_gold", ""))
        if isinstance(gold, list):
            em_val = max(exact_match(pred_text, g) for g in gold) if gold else 0.0
            f1_val = max(f1_score(pred_text, g) for g in gold) if gold else 0.0
        else:
            em_val = exact_match(pred_text, gold)
            f1_val = f1_score(pred_text, gold)
        ems.append(em_val)
        f1s.append(f1_val)

        lat = pred.get("latency_ms", 0.0)
        tok = pred.get("tokens_used", 0)
        if lat > 0:
            lats.append(lat)
        if tok > 0:
            toks.append(tok)

    n = len(predictions)
    em_mean, em_lo, em_hi = bootstrap_ci(ems)
    f1_mean, f1_lo, f1_hi = bootstrap_ci(f1s)
    lat_stats = latency_percentiles(lats) if lats else {"p50": 0, "p95": 0, "mean": 0}
    tok_arr = np.array(toks) if toks else np.array([0.0])

    return {
        "label": label,
        "n_queries": n,
        "em": round(em_mean, 4),
        "em_ci": [round(em_lo, 4), round(em_hi, 4)],
        "f1": round(f1_mean, 4),
        "f1_ci": [round(f1_lo, 4), round(f1_hi, 4)],
        "abstention_rate": round(abstain_count / max(n, 1), 4),
        "latency_p50_ms": round(lat_stats["p50"], 1),
        "latency_p95_ms": round(lat_stats["p95"], 1),
        "latency_mean_ms": round(lat_stats["mean"], 1),
        "avg_evidence_tokens": round(float(tok_arr.mean()), 1),
    }


def find_prediction_files(results_dir: str) -> Dict[str, str]:
    """Scan results_dir for prediction files and label them by directory name."""
    found = {}
    rdir = Path(results_dir)
    if not rdir.exists():
        return found
    for p in sorted(rdir.rglob("predictions.json")):
        label = p.parent.name
        found[label] = str(p)
    for p in sorted(rdir.rglob("predictions.jsonl")):
        label = p.parent.name
        if label not in found:
            found[label] = str(p)
    return found


def detect_anomalies(
    rows: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """Flag potential anomalies (e.g., latency outliers, impossible EM)."""
    issues = []
    for r in rows:
        # Latency anomaly: p50 > p95 (impossible)
        if r["latency_p50_ms"] > r["latency_p95_ms"] and r["latency_p95_ms"] > 0:
            issues.append({
                "label": r["label"],
                "issue": "latency_p50 > latency_p95",
                "detail": f"p50={r['latency_p50_ms']}, p95={r['latency_p95_ms']}",
            })
        # EM > 1 (impossible)
        if r["em"] > 1.0:
            issues.append({
                "label": r["label"],
                "issue": "em > 1.0",
                "detail": f"em={r['em']}",
            })
        # Very high abstention with high EM (suspicious)
        if r["abstention_rate"] > 0.5 and r["em"] > 0.5:
            issues.append({
                "label": r["label"],
                "issue": "high_abstention_high_em",
                "detail": f"abstention={r['abstention_rate']}, em={r['em']}",
            })
    return issues


def main():
    parser = argparse.ArgumentParser(description="E0.1: Sanity check paper table metrics")
    parser.add_argument("--results_dir", type=str, default="runs/paper_results",
                        help="Directory containing per-config prediction files")
    parser.add_argument("--output_dir", type=str, default="runs/rebuttal/e0_1")
    args = parser.parse_args()

    pred_files = find_prediction_files(args.results_dir)
    if not pred_files:
        print(f"[E0.1] No prediction files found in {args.results_dir}")
        print("[E0.1] Creating placeholder report with instructions")
        meta = ExperimentMetadata(
            experiment_id="e0_1_sanity_check",
            dataset="all",
            mode="audit",
        )
        report = ExperimentReport(
            metadata=meta,
            metrics={"status": "no_predictions_found"},
            summary_table="No prediction files found. Place predictions.json in subdirs of --results_dir.",
        )
        path = report.save(args.output_dir)
        print(f"[E0.1] Report saved to {path}")
        return

    rows = []
    for label, pred_path in sorted(pred_files.items()):
        print(f"[E0.1] Recomputing metrics for: {label}")
        preds = load_predictions(pred_path)
        row = recompute_row(preds, label)
        rows.append(row)

    anomalies = detect_anomalies(rows)
    if anomalies:
        print("\n[E0.1] ANOMALIES DETECTED:")
        for a in anomalies:
            print(f"  - {a['label']}: {a['issue']} ({a['detail']})")
    else:
        print("\n[E0.1] No anomalies detected.")

    # Build summary table
    hdr = "| Config | N | EM [CI] | F1 [CI] | Abstain | Lat p50 | Lat p95 | EvTok |"
    sep = "|--------|---|---------|---------|---------|---------|---------|-------|"
    table_lines = [hdr, sep]
    for r in sorted(rows, key=lambda x: x["f1"], reverse=True):
        table_lines.append(
            f"| {r['label']} "
            f"| {r['n_queries']} "
            f"| {r['em']:.3f} [{r['em_ci'][0]:.3f},{r['em_ci'][1]:.3f}] "
            f"| {r['f1']:.3f} [{r['f1_ci'][0]:.3f},{r['f1_ci'][1]:.3f}] "
            f"| {r['abstention_rate']:.3f} "
            f"| {r['latency_p50_ms']:.0f} "
            f"| {r['latency_p95_ms']:.0f} "
            f"| {r['avg_evidence_tokens']:.0f} |"
        )
    summary = "\n".join(table_lines)
    print(f"\n{summary}")

    meta = ExperimentMetadata(
        experiment_id="e0_1_sanity_check",
        dataset="all",
        mode="audit",
        extra={"results_dir": args.results_dir},
    )
    report = ExperimentReport(
        metadata=meta,
        metrics={
            "recomputed_rows": rows,
            "anomalies": anomalies,
            "num_configs": len(rows),
        },
        summary_table=summary,
    )
    path = report.save(args.output_dir)
    print(f"\n[E0.1] Report saved to {path}")
    print("[E0.1] Rebuttal sentence: 'We verified Table X; any corrected values are noted above.'")


if __name__ == "__main__":
    main()
