#!/usr/bin/env python3
"""Orchestrator for all rebuttal experiments.

Runs experiments in the recommended order for maximum rebuttal impact:
  1. E0.1 (sanity) + E0.2 (calibration protocol)  -- no GPU
  2. E1  latency breakdown                         -- 1 GPU, fast
  3. E2  facet robustness                           -- 1 GPU, medium
  4. E3  verifier K sweep                           -- 1 GPU, medium
  5. E1.5 backbone recalibration                    -- 1 GPU, fast
  6. E1.6 retrieval sensitivity                     -- 1 GPU, medium
  7. E1.7 Safe-Cover curve                          -- 1 GPU, fast
  8. E5  unsupported answer rate                    -- 1 GPU, medium

The "score moving" minimal subset is: E1 + E2 + E1.5 + E1.6.

Usage:
    # Run everything:
    python -m experiments.rebuttal.run_all \
        --hotpot_path data/hotpotqa_dev.json \
        --musique_path data/musique_dev.jsonl \
        --results_dir runs/paper_results \
        --calibration_dir data/calibration \
        --output_root runs/rebuttal \
        --model meta-llama/Meta-Llama-3-8B-Instruct \
        --device 0

    # Run minimal "score moving" subset:
    python -m experiments.rebuttal.run_all \
        --hotpot_path data/hotpotqa_dev.json \
        --output_root runs/rebuttal \
        --model meta-llama/Meta-Llama-3-8B-Instruct \
        --device 0 --minimal

    # Run only specific tiers:
    python -m experiments.rebuttal.run_all \
        --hotpot_path data/hotpotqa_dev.json \
        --output_root runs/rebuttal \
        --tiers 0 1
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


EXPERIMENT_ORDER = [
    # (id, module, tier, description, required_data)
    ("e0_1", "experiments.rebuttal.e0_1_sanity_check", 0,
     "Table/metric sanity check", "results_dir"),
    ("e0_2", "experiments.rebuttal.e0_2_calibration_protocol", 0,
     "Calibration pool protocol", "calibration_dir"),
    ("e1", "experiments.rebuttal.e1_latency_breakdown", 1,
     "Latency breakdown (Hotpot)", "hotpot_path"),
    ("e2", "experiments.rebuttal.e2_facet_robustness", 1,
     "Facet robustness ablation", "hotpot_path"),
    ("e3", "experiments.rebuttal.e3_verifier_k_sweep", 1,
     "Verifier K sweep", "hotpot_path"),
    ("e1_5", "experiments.rebuttal.e1_5_backbone_recalibration", 1.5,
     "Backbone recalibration", "hotpot_path"),
    ("e1_6", "experiments.rebuttal.e1_6_retrieval_sensitivity", 1.5,
     "Retrieval sensitivity", "hotpot_path"),
    ("e1_7", "experiments.rebuttal.e1_7_safe_cover_curve", 2,
     "Safe-Cover operating curve", "musique_path"),
    ("e5", "experiments.rebuttal.e5_unsupported_answer", 2,
     "Unsupported answer rate", "hotpot_path"),
]

MINIMAL_SUBSET = {"e1", "e2", "e1_5", "e1_6"}


def build_cmd(
    exp_id: str,
    module: str,
    args: argparse.Namespace,
) -> Optional[List[str]]:
    """Build the command line for one experiment."""
    output_dir = os.path.join(args.output_root, exp_id)
    cmd = [sys.executable, "-m", module, "--output_dir", output_dir]

    if exp_id == "e0_1":
        if not args.results_dir:
            return None
        cmd += ["--results_dir", args.results_dir]

    elif exp_id == "e0_2":
        if not args.calibration_dir:
            return None
        cmd += ["--calibration_dir", args.calibration_dir]

    elif exp_id == "e1":
        if not args.hotpot_path:
            return None
        cmd += [
            "--data_path", args.hotpot_path,
            "--dataset", "hotpotqa",
            "--budget", str(args.budget),
            "--limit", "200",
            "--model", args.model,
            "--device", str(args.device),
        ]
        if args.load_in_8bit:
            cmd.append("--load_in_8bit")

    elif exp_id == "e2":
        if not args.hotpot_path:
            return None
        limit = "200" if args.fast else "500"
        cmd += [
            "--data_path", args.hotpot_path,
            "--dataset", "hotpotqa",
            "--budget", str(args.budget),
            "--limit", limit,
            "--model", args.model,
            "--device", str(args.device),
            "--seeds", "42", "123", "456",
        ]
        if args.load_in_8bit:
            cmd.append("--load_in_8bit")

    elif exp_id == "e3":
        if not args.hotpot_path:
            return None
        limit = "200" if args.fast else "500"
        cmd += [
            "--data_path", args.hotpot_path,
            "--dataset", "hotpotqa",
            "--budgets", str(args.budget),
            "--limit", limit,
            "--model", args.model,
            "--device", str(args.device),
            "--ks", "8", "16", "32",
        ]
        if args.load_in_8bit:
            cmd.append("--load_in_8bit")

    elif exp_id == "e1_5":
        if not args.hotpot_path:
            return None
        cmd += [
            "--data_path", args.hotpot_path,
            "--dataset", "hotpotqa",
            "--budget", str(args.budget),
            "--limit", "200",
            "--model", args.model,  # will use --backbones for both
            "--device", str(args.device),
        ]
        if args.backbones:
            cmd += ["--backbones"] + args.backbones
        if args.shared_calibration_path:
            cmd += ["--shared_calibration_path", args.shared_calibration_path]
        if args.per_backbone_calibration_dir:
            cmd += ["--per_backbone_calibration_dir", args.per_backbone_calibration_dir]
        if args.load_in_8bit:
            cmd.append("--load_in_8bit")

    elif exp_id == "e1_6":
        if not args.hotpot_path:
            return None
        limit = "200" if args.fast else "500"
        cmd += [
            "--data_path", args.hotpot_path,
            "--dataset", "hotpotqa",
            "--budget", str(args.budget),
            "--limit", limit,
            "--model", args.model,
            "--device", str(args.device),
            "--top_ns", "20", "50", "100",
            "--retrievers", "dense", "bm25",
        ]
        if args.load_in_8bit:
            cmd.append("--load_in_8bit")

    elif exp_id == "e1_7":
        data_path = args.musique_path or args.hotpot_path
        dataset = "musique" if args.musique_path else "hotpotqa"
        if not data_path:
            return None
        cmd += [
            "--data_path", data_path,
            "--dataset", dataset,
            "--budget", str(args.budget),
            "--limit", "200",
            "--model", args.model,
            "--device", str(args.device),
            "--alphas", "0.1", "0.05", "0.01",
        ]
        if args.load_in_8bit:
            cmd.append("--load_in_8bit")

    elif exp_id == "e5":
        if not args.hotpot_path:
            return None
        limit = "200" if args.fast else "500"
        cmd += [
            "--data_path", args.hotpot_path,
            "--dataset", "hotpotqa",
            "--budget", str(args.budget),
            "--limit", limit,
            "--model", args.model,
            "--device", str(args.device),
        ]
        if args.load_in_8bit:
            cmd.append("--load_in_8bit")

    return cmd


def run_experiment(
    exp_id: str,
    module: str,
    tier: float,
    description: str,
    args: argparse.Namespace,
) -> dict:
    """Run one experiment and return status."""
    cmd = build_cmd(exp_id, module, args)
    if cmd is None:
        return {"exp_id": exp_id, "status": "skipped", "reason": "missing data path"}

    print(f"\n{'='*70}")
    print(f"  [{exp_id}] Tier {tier}: {description}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*70}\n")

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(Path(__file__).resolve().parent.parent.parent),
            capture_output=not args.verbose,
            text=True,
            timeout=args.timeout,
        )
        elapsed = time.time() - t0

        if result.returncode != 0:
            print(f"  [{exp_id}] FAILED (exit code {result.returncode})")
            if not args.verbose and result.stderr:
                print(f"  stderr (last 500 chars): {result.stderr[-500:]}")
            return {
                "exp_id": exp_id, "status": "failed",
                "returncode": result.returncode,
                "elapsed_s": round(elapsed, 1),
                "stderr_tail": result.stderr[-500:] if result.stderr else "",
            }

        print(f"  [{exp_id}] PASSED ({elapsed:.0f}s)")
        return {"exp_id": exp_id, "status": "passed", "elapsed_s": round(elapsed, 1)}

    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        print(f"  [{exp_id}] TIMEOUT ({args.timeout}s)")
        return {"exp_id": exp_id, "status": "timeout", "elapsed_s": round(elapsed, 1)}
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  [{exp_id}] ERROR: {e}")
        return {"exp_id": exp_id, "status": "error", "error": str(e), "elapsed_s": round(elapsed, 1)}


# ---------------------------------------------------------------------------
# Result aggregation
# ---------------------------------------------------------------------------

def _find_report_json(exp_dir: str) -> Optional[str]:
    """Find the experiment report JSON inside an experiment output directory."""
    exp_path = Path(exp_dir)
    if not exp_path.is_dir():
        return None
    jsons = list(exp_path.glob("*.json"))
    if not jsons:
        return None
    # Prefer the one that isn't run_summary.json
    for j in jsons:
        if j.name != "run_summary.json":
            return str(j)
    return str(jsons[0])


def _extract_headline(exp_id: str, report: Dict[str, Any]) -> Dict[str, Any]:
    """Extract headline metrics from one experiment's report JSON.

    Returns a dict with standardised keys:
        experiment, description, headline_metric, headline_value,
        detail (dict of additional key-value pairs for the table).
    """
    metrics = report.get("metrics", {})
    meta = report.get("metadata", {})

    if exp_id == "e0_1":
        # Sanity check: report anomaly count
        status = metrics.get("status", "")
        if status == "no_predictions_found":
            return {"experiment": "E0.1", "description": "Sanity check",
                    "headline_metric": "status", "headline_value": "no data",
                    "detail": {}}
        # Count configs checked
        n_configs = len([k for k in metrics if k not in ("status",)])
        anomalies = report.get("compute", {}).get("anomalies", [])
        return {"experiment": "E0.1", "description": "Sanity check",
                "headline_metric": "anomalies",
                "headline_value": len(anomalies) if isinstance(anomalies, list) else 0,
                "detail": {"configs_checked": n_configs}}

    elif exp_id == "e0_2":
        # Calibration protocol: documentation artefact
        paragraph = metrics.get("paragraph_summary", "")
        has_artefacts = bool(metrics.get("protocol_artifacts", {}))
        return {"experiment": "E0.2", "description": "Calibration protocol",
                "headline_metric": "artefacts_present",
                "headline_value": has_artefacts,
                "detail": {}}

    elif exp_id == "e1":
        # Latency breakdown: safe_cover vs pareto F1 & p50 latency
        detail = {}
        for mode in ("safe_cover", "pareto"):
            m = metrics.get(mode, {})
            if m:
                detail[f"{mode}_F1"] = m.get("f1", "")
                detail[f"{mode}_EM"] = m.get("em", "")
                lat = m.get("latency", {})
                detail[f"{mode}_lat_p50"] = lat.get("p50", "")
                detail[f"{mode}_pairs"] = m.get("verification", {}).get(
                    "total_pairs_scored_mean", "")
        sc = metrics.get("safe_cover", {})
        return {"experiment": "E1", "description": "Latency breakdown",
                "headline_metric": "safe_cover_F1",
                "headline_value": sc.get("f1", ""),
                "detail": detail}

    elif exp_id == "e2":
        # Facet robustness: delta-F1 vs top-k for baseline condition
        detail = {}
        for cond, res in metrics.items():
            detail[f"{cond}_F1"] = res.get("trident_f1_mean", "")
            detail[f"{cond}_dF1_vs_topk"] = res.get("delta_f1_vs_topk_mean", "")
        baseline_f1 = metrics.get("baseline", {}).get("trident_f1_mean", "")
        return {"experiment": "E2", "description": "Facet robustness",
                "headline_metric": "baseline_F1",
                "headline_value": baseline_f1,
                "detail": detail}

    elif exp_id == "e3":
        # Verifier K sweep: collect per-K results
        detail = {}
        best_f1 = 0
        for key, res in metrics.items():
            f1 = res.get("f1", 0)
            detail[f"{key}_F1"] = f1
            detail[f"{key}_lat_p50"] = res.get("latency_p50", "")
            if isinstance(f1, (int, float)) and f1 > best_f1:
                best_f1 = f1
        return {"experiment": "E3", "description": "Verifier K sweep",
                "headline_metric": "best_F1",
                "headline_value": best_f1,
                "detail": detail}

    elif exp_id == "e1_5":
        # Backbone recalibration: per-backbone delta-F1 (per_bb - shared)
        detail = {}
        for key, res in metrics.items():
            if isinstance(res, dict):
                detail[f"{key}_F1"] = res.get("f1", "")
                detail[f"{key}_ECE"] = res.get("verifier_ece", "")
                detail[f"{key}_AUC"] = res.get("verifier_auc", "")
        return {"experiment": "E1.5", "description": "Backbone recalibration",
                "headline_metric": "configs_tested",
                "headline_value": len(metrics),
                "detail": detail}

    elif exp_id == "e1_6":
        # Retrieval sensitivity: per-retriever/N results
        detail = {}
        for key, res in metrics.items():
            if isinstance(res, dict):
                detail[f"{key}_F1"] = res.get("f1", res.get("trident_f1", ""))
                detail[f"{key}_dF1"] = res.get("delta_f1_vs_topk", "")
        return {"experiment": "E1.6", "description": "Retrieval sensitivity",
                "headline_metric": "configs_tested",
                "headline_value": len(metrics),
                "detail": detail}

    elif exp_id == "e1_7":
        # Safe-Cover curve: per-alpha abstention/F1 tradeoff
        detail = {}
        for key, res in metrics.items():
            if isinstance(res, dict):
                detail[f"{key}_abstain"] = res.get("abstention_rate", "")
                detail[f"{key}_F1"] = res.get("overall_f1", res.get("answered_f1", ""))
        return {"experiment": "E1.7", "description": "Safe-Cover curve",
                "headline_metric": "alphas_tested",
                "headline_value": len(metrics),
                "detail": detail}

    elif exp_id == "e5":
        # Unsupported answer rate
        trident = metrics.get("trident_pareto", {})
        topk = metrics.get("top_k", {})
        delta = (topk.get("unsupported_rate", 0) or 0) - (trident.get("unsupported_rate", 0) or 0)
        detail = {
            "trident_unsup": trident.get("unsupported_rate", ""),
            "topk_unsup": topk.get("unsupported_rate", ""),
            "trident_F1": trident.get("answered_f1", ""),
            "topk_F1": topk.get("answered_f1", ""),
        }
        return {"experiment": "E5", "description": "Unsupported answer rate",
                "headline_metric": "delta_unsup (topk-trident)",
                "headline_value": round(delta, 4) if isinstance(delta, float) else delta,
                "detail": detail}

    # Fallback
    return {"experiment": exp_id, "description": "",
            "headline_metric": "raw_keys",
            "headline_value": list(metrics.keys())[:5],
            "detail": {}}


def aggregate_results(output_root: str, statuses: List[Dict]) -> str:
    """Read all per-experiment reports and produce a unified aggregate report.

    Returns the path to the aggregate JSON.
    """
    root = Path(output_root)
    headlines = []
    all_metrics = {}

    for s in statuses:
        exp_id = s["exp_id"]
        exp_dir = root / exp_id
        report_path = _find_report_json(str(exp_dir))

        if report_path is None:
            headlines.append({
                "experiment": exp_id,
                "description": "",
                "headline_metric": "status",
                "headline_value": s.get("status", "unknown"),
                "detail": {},
            })
            continue

        with open(report_path) as f:
            report = json.load(f)

        headline = _extract_headline(exp_id, report)
        headline["status"] = s.get("status", "unknown")
        headline["elapsed_s"] = s.get("elapsed_s", 0)
        headlines.append(headline)
        all_metrics[exp_id] = {
            "report_path": report_path,
            "metadata": report.get("metadata", {}),
            "metrics": report.get("metrics", {}),
            "compute": report.get("compute", {}),
            "summary_table": report.get("summary_table", ""),
        }

    # Build unified markdown table
    table_lines = []
    table_lines.append("")
    table_lines.append("=" * 78)
    table_lines.append("  AGGREGATE REBUTTAL RESULTS")
    table_lines.append("=" * 78)
    table_lines.append("")
    table_lines.append(
        f"| {'Exp':<6} | {'Description':<25} | {'Status':<7} "
        f"| {'Headline Metric':<25} | {'Value':<12} | {'Time':>6} |"
    )
    table_lines.append(
        f"|{'-'*8}|{'-'*27}|{'-'*9}"
        f"|{'-'*27}|{'-'*14}|{'-'*8}|"
    )

    for h in headlines:
        val = h["headline_value"]
        if isinstance(val, float):
            val_str = f"{val:.4f}"
        else:
            val_str = str(val)[:12]
        elapsed = h.get("elapsed_s", 0)
        table_lines.append(
            f"| {h['experiment']:<6} | {h['description']:<25} "
            f"| {h.get('status', '?'):<7} "
            f"| {h['headline_metric']:<25} | {val_str:<12} "
            f"| {elapsed:>5.0f}s |"
        )

    table_lines.append("")

    # Per-experiment detail sections
    for h in headlines:
        if not h.get("detail"):
            continue
        table_lines.append(f"  [{h['experiment']}] {h['description']} -- detail:")
        for k, v in h["detail"].items():
            if isinstance(v, float):
                table_lines.append(f"    {k}: {v:.4f}")
            else:
                table_lines.append(f"    {k}: {v}")
        table_lines.append("")

    aggregate_table = "\n".join(table_lines)
    print(aggregate_table)

    # Save aggregate JSON
    aggregate_path = root / "aggregate_results.json"
    with open(aggregate_path, "w") as f:
        json.dump({
            "headlines": headlines,
            "per_experiment": all_metrics,
            "aggregate_table": aggregate_table,
        }, f, indent=2, default=str)

    print(f"  Aggregate results saved to {aggregate_path}")
    return str(aggregate_path)


def main():
    parser = argparse.ArgumentParser(
        description="Run all rebuttal experiments in recommended order"
    )
    # Data paths
    parser.add_argument("--hotpot_path", type=str, default="",
                        help="Path to HotpotQA dev data (JSON)")
    parser.add_argument("--musique_path", type=str, default="",
                        help="Path to MuSiQue dev data (JSONL)")
    parser.add_argument("--results_dir", type=str, default="",
                        help="Dir with paper results for E0.1 sanity check")
    parser.add_argument("--calibration_dir", type=str, default="",
                        help="Dir with calibrator JSONs for E0.2")

    # Output
    parser.add_argument("--output_root", type=str, default="runs/rebuttal")

    # Model
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--backbones", nargs="+", default=[])
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--budget", type=int, default=500)

    # Calibration for E1.5
    parser.add_argument("--shared_calibration_path", type=str, default="")
    parser.add_argument("--per_backbone_calibration_dir", type=str, default="")

    # Experiment selection
    parser.add_argument("--minimal", action="store_true",
                        help="Run only the score-moving subset (E1+E2+E1.5+E1.6)")
    parser.add_argument("--tiers", type=float, nargs="+", default=None,
                        help="Only run specific tiers (e.g., --tiers 0 1)")
    parser.add_argument("--only", nargs="+", default=None,
                        help="Only run specific experiment IDs (e.g., --only e1 e2)")
    parser.add_argument("--skip", nargs="+", default=[],
                        help="Skip specific experiment IDs")

    # Execution
    parser.add_argument("--fast", action="store_true",
                        help="Use smaller limits for faster iteration")
    parser.add_argument("--verbose", action="store_true",
                        help="Show experiment stdout/stderr in real-time")
    parser.add_argument("--timeout", type=int, default=7200,
                        help="Per-experiment timeout in seconds (default 2h)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without running")

    args = parser.parse_args()

    # Filter experiments
    experiments = EXPERIMENT_ORDER
    if args.minimal:
        experiments = [(eid, m, t, d, rd) for eid, m, t, d, rd in experiments
                       if eid in MINIMAL_SUBSET]
    if args.tiers is not None:
        experiments = [(eid, m, t, d, rd) for eid, m, t, d, rd in experiments
                       if t in args.tiers]
    if args.only:
        experiments = [(eid, m, t, d, rd) for eid, m, t, d, rd in experiments
                       if eid in args.only]
    experiments = [(eid, m, t, d, rd) for eid, m, t, d, rd in experiments
                   if eid not in args.skip]

    if not experiments:
        print("No experiments selected. Check --minimal, --tiers, --only, --skip flags.")
        return

    print(f"Rebuttal experiment suite: {len(experiments)} experiments selected")
    print(f"Output root: {args.output_root}")
    print(f"Order: {' -> '.join(eid for eid, *_ in experiments)}")

    if args.dry_run:
        print("\n[DRY RUN] Commands that would be executed:")
        for exp_id, module, tier, desc, _ in experiments:
            cmd = build_cmd(exp_id, module, args)
            if cmd:
                print(f"\n  [{exp_id}] {desc}")
                print(f"  {' '.join(cmd)}")
            else:
                print(f"\n  [{exp_id}] SKIP (missing data)")
        return

    # Run experiments
    statuses = []
    total_t0 = time.time()
    for exp_id, module, tier, desc, _ in experiments:
        status = run_experiment(exp_id, module, tier, desc, args)
        statuses.append(status)

        # Stop on critical failure (tier 0) unless --continue-on-error
        if status["status"] == "failed" and tier == 0:
            print(f"\n[WARN] Tier 0 experiment {exp_id} failed. Continuing with remaining experiments.")

    total_elapsed = time.time() - total_t0

    # Summary
    print(f"\n{'='*70}")
    print(f"  REBUTTAL EXPERIMENT SUMMARY")
    print(f"{'='*70}")
    passed = sum(1 for s in statuses if s["status"] == "passed")
    failed = sum(1 for s in statuses if s["status"] == "failed")
    skipped = sum(1 for s in statuses if s["status"] == "skipped")
    print(f"  Total: {len(statuses)} | Passed: {passed} | Failed: {failed} | Skipped: {skipped}")
    print(f"  Total time: {total_elapsed:.0f}s")

    for s in statuses:
        icon = {"passed": "OK", "failed": "FAIL", "skipped": "SKIP",
                "timeout": "TIME", "error": "ERR"}.get(s["status"], "?")
        elapsed = s.get("elapsed_s", 0)
        print(f"  [{icon}] {s['exp_id']:8s} ({elapsed:.0f}s)")

    # Save run summary
    summary_path = Path(args.output_root) / "run_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump({
            "statuses": statuses,
            "total_elapsed_s": round(total_elapsed, 1),
            "args": {k: str(v) for k, v in vars(args).items()},
        }, f, indent=2)
    print(f"\n  Run summary saved to {summary_path}")

    # Aggregate results from all experiment reports
    passed_or_done = [s for s in statuses if s["status"] != "skipped"]
    if passed_or_done:
        print(f"\n{'='*70}")
        print(f"  AGGREGATING RESULTS")
        print(f"{'='*70}")
        aggregate_results(args.output_root, statuses)


if __name__ == "__main__":
    main()
