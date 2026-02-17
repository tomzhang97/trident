#!/usr/bin/env python3
"""Orchestrator for all rebuttal experiments.

Runs experiments concurrently across available GPUs:
  - CPU experiments (E0.1, E0.2) run first in parallel.
  - GPU experiments are batched across available GPUs: each experiment
    is pinned to one GPU and multiple experiments run simultaneously.
  - With 5 GPUs and 7 GPU experiments, this takes ~2 batches instead
    of 7 sequential runs.

The "score moving" minimal subset is: E1 + E2 + E1.5 + E1.6.

Usage:
    # Run everything on GPUs 2-6 (concurrent):
    python -m experiments.rebuttal.run_all \
        --hotpot_path data/hotpotqa_dev.json \
        --musique_path data/musique_dev.jsonl \
        --results_dir runs/paper_results \
        --calibration_dir data/calibration \
        --output_root runs/rebuttal \
        --model meta-llama/Meta-Llama-3-8B-Instruct \
        --gpu_ids 2,3,4,5,6

    # Run minimal subset on GPUs 2-5:
    python -m experiments.rebuttal.run_all \
        --hotpot_path data/hotpotqa_dev.json \
        --output_root runs/rebuttal \
        --model meta-llama/Meta-Llama-3-8B-Instruct \
        --gpu_ids 2,3,4,5 --minimal

    # Single-GPU sequential (old behavior):
    python -m experiments.rebuttal.run_all \
        --hotpot_path data/hotpotqa_dev.json \
        --output_root runs/rebuttal \
        --model meta-llama/Meta-Llama-3-8B-Instruct \
        --device 0

    # Run only specific experiments:
    python -m experiments.rebuttal.run_all \
        --hotpot_path data/hotpotqa_dev.json \
        --output_root runs/rebuttal \
        --gpu_ids 2,3 --only e1 e2
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


EXPERIMENT_ORDER = [
    # (id, module, tier, description, required_data, needs_gpu)
    ("e0_1", "experiments.rebuttal.e0_1_sanity_check", 0,
     "Table/metric sanity check", "results_dir", False),
    ("e0_2", "experiments.rebuttal.e0_2_calibration_protocol", 0,
     "Calibration pool protocol", "calibration_dir", False),
    ("e1", "experiments.rebuttal.e1_latency_breakdown", 1,
     "Latency breakdown (Hotpot)", "hotpot_path", True),
    ("e2", "experiments.rebuttal.e2_facet_robustness", 1,
     "Facet robustness ablation", "hotpot_path", True),
    ("e3", "experiments.rebuttal.e3_verifier_k_sweep", 1,
     "Verifier K sweep", "hotpot_path", True),
    ("e1_5", "experiments.rebuttal.e1_5_backbone_recalibration", 1.5,
     "Backbone recalibration", "hotpot_path", True),
    ("e1_6", "experiments.rebuttal.e1_6_retrieval_sensitivity", 1.5,
     "Retrieval sensitivity", "hotpot_path", True),
    ("e1_7", "experiments.rebuttal.e1_7_safe_cover_curve", 2,
     "Safe-Cover operating curve", "musique_path", True),
    ("e5", "experiments.rebuttal.e5_unsupported_answer", 2,
     "Unsupported answer rate", "hotpot_path", True),
]

MINIMAL_SUBSET = {"e1", "e2", "e1_5", "e1_6"}

# Experiments that have many arms and benefit from intra-experiment
# multi-GPU parallelism (when more GPUs are available than experiments).
MULTI_ARM_EXPERIMENTS = {"e2", "e1_5", "e1_6"}


def build_cmd(
    exp_id: str,
    module: str,
    args: argparse.Namespace,
    *,
    gpu_id: Optional[int] = None,
) -> Optional[List[str]]:
    """Build the command line for one experiment.

    Args:
        gpu_id: If provided, the experiment is pinned via CUDA_VISIBLE_DEVICES
                in _launch_experiment, so --device is always 0.
    """
    output_dir = os.path.join(args.output_root, exp_id)
    # When gpu_id is set, _launch_experiment pins CUDA_VISIBLE_DEVICES to that
    # single physical GPU, so the script must use cuda:0.
    device = 0 if gpu_id is not None else args.device
    cmd = [sys.executable, "-u", "-m", module, "--output_dir", output_dir]

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
            "--device", str(device),
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
            "--device", str(device),
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
            "--device", str(device),
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
            "--model", args.model,
            "--device", str(device),
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
            "--device", str(device),
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
            "--device", str(device),
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
            "--device", str(device),
        ]
        if args.load_in_8bit:
            cmd.append("--load_in_8bit")

    return cmd


# ---------------------------------------------------------------------------
# Experiment execution
# ---------------------------------------------------------------------------

def _launch_experiment(
    exp_id: str,
    module: str,
    tier: float,
    description: str,
    args: argparse.Namespace,
    gpu_id: Optional[int] = None,
) -> Tuple[Optional[subprocess.Popen], dict]:
    """Launch an experiment as a non-blocking subprocess.

    Returns (process, status_stub).  If cmd is None (missing data),
    returns (None, skipped_status).
    """
    cmd = build_cmd(exp_id, module, args, gpu_id=gpu_id)
    if cmd is None:
        return None, {"exp_id": exp_id, "status": "skipped",
                       "reason": "missing data path", "gpu_id": gpu_id}

    gpu_label = f"GPU {gpu_id}" if gpu_id is not None else "CPU"
    print(f"  [{exp_id}] {description} -> {gpu_label}")
    if args.verbose:
        print(f"    CMD: {' '.join(cmd)}")

    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Write stdout/stderr to per-experiment log files
    log_dir = Path(args.output_root) / exp_id
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = open(log_dir / "stdout.log", "w")
    stderr_log = open(log_dir / "stderr.log", "w")

    proc = subprocess.Popen(
        cmd,
        cwd=str(Path(__file__).resolve().parent.parent.parent),
        stdout=stdout_log if not args.verbose else None,
        stderr=stderr_log if not args.verbose else None,
        text=True,
        env=env,
    )

    return proc, {
        "exp_id": exp_id, "status": "running", "gpu_id": gpu_id,
        "_t0": time.time(), "_proc": proc, "_description": description,
        "_tier": tier, "_stdout_log": stdout_log, "_stderr_log": stderr_log,
    }


def _wait_for_batch(running: List[dict], timeout: int) -> List[dict]:
    """Wait for all running experiments to finish.  Returns final statuses."""
    statuses = []
    for info in running:
        proc = info.get("_proc")
        if proc is None:
            # Already resolved (skipped)
            statuses.append(info)
            continue

        exp_id = info["exp_id"]
        gpu_id = info.get("gpu_id", "?")
        try:
            proc.wait(timeout=timeout)
            elapsed = time.time() - info["_t0"]

            if proc.returncode != 0:
                # Read last 500 chars of stderr for diagnostics
                stderr_tail = ""
                stderr_log_path = Path(info["_stderr_log"].name)
                info["_stderr_log"].close()
                info["_stdout_log"].close()
                if stderr_log_path.exists():
                    text = stderr_log_path.read_text()
                    stderr_tail = text[-500:]
                print(f"  [{exp_id}] FAILED (GPU {gpu_id}, exit {proc.returncode}, {elapsed:.0f}s)")
                if stderr_tail:
                    print(f"    stderr: ...{stderr_tail}")
                statuses.append({
                    "exp_id": exp_id, "status": "failed",
                    "returncode": proc.returncode,
                    "elapsed_s": round(elapsed, 1),
                    "gpu_id": gpu_id,
                    "stderr_tail": stderr_tail,
                })
            else:
                info["_stderr_log"].close()
                info["_stdout_log"].close()
                print(f"  [{exp_id}] PASSED (GPU {gpu_id}, {elapsed:.0f}s)")
                statuses.append({
                    "exp_id": exp_id, "status": "passed",
                    "elapsed_s": round(elapsed, 1), "gpu_id": gpu_id,
                })

        except subprocess.TimeoutExpired:
            proc.kill()
            info["_stderr_log"].close()
            info["_stdout_log"].close()
            elapsed = time.time() - info["_t0"]
            print(f"  [{exp_id}] TIMEOUT (GPU {gpu_id}, {timeout}s)")
            statuses.append({
                "exp_id": exp_id, "status": "timeout",
                "elapsed_s": round(elapsed, 1), "gpu_id": gpu_id,
            })
        except Exception as e:
            info["_stderr_log"].close()
            info["_stdout_log"].close()
            elapsed = time.time() - info["_t0"]
            print(f"  [{exp_id}] ERROR (GPU {gpu_id}): {e}")
            statuses.append({
                "exp_id": exp_id, "status": "error",
                "error": str(e), "elapsed_s": round(elapsed, 1),
                "gpu_id": gpu_id,
            })

    return statuses


def run_experiment_sequential(
    exp_id: str,
    module: str,
    tier: float,
    description: str,
    args: argparse.Namespace,
) -> dict:
    """Run one experiment synchronously (single-GPU fallback)."""
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
    # Prefer the one that isn't run_summary.json or stdout/stderr logs
    for j in jsons:
        if j.name not in ("run_summary.json",):
            return str(j)
    return str(jsons[0])


def _extract_headline(exp_id: str, report: Dict[str, Any]) -> Dict[str, Any]:
    """Extract headline metrics from one experiment's report JSON."""
    metrics = report.get("metrics", {})

    if exp_id == "e0_1":
        status = metrics.get("status", "")
        if status == "no_predictions_found":
            return {"experiment": "E0.1", "description": "Sanity check",
                    "headline_metric": "status", "headline_value": "no data",
                    "detail": {}}
        n_configs = len([k for k in metrics if k not in ("status",)])
        anomalies = report.get("compute", {}).get("anomalies", [])
        return {"experiment": "E0.1", "description": "Sanity check",
                "headline_metric": "anomalies",
                "headline_value": len(anomalies) if isinstance(anomalies, list) else 0,
                "detail": {"configs_checked": n_configs}}

    elif exp_id == "e0_2":
        has_artefacts = bool(metrics.get("protocol_artifacts", {}))
        return {"experiment": "E0.2", "description": "Calibration protocol",
                "headline_metric": "artefacts_present",
                "headline_value": has_artefacts, "detail": {}}

    elif exp_id == "e1":
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
                "headline_value": sc.get("f1", ""), "detail": detail}

    elif exp_id == "e2":
        detail = {}
        for cond, res in metrics.items():
            if isinstance(res, dict):
                detail[f"{cond}_F1"] = res.get("trident_f1_mean", "")
                detail[f"{cond}_dF1_vs_topk"] = res.get("delta_f1_vs_topk_mean", "")
        baseline_f1 = metrics.get("baseline", {}).get("trident_f1_mean", "")
        return {"experiment": "E2", "description": "Facet robustness",
                "headline_metric": "baseline_F1",
                "headline_value": baseline_f1, "detail": detail}

    elif exp_id == "e3":
        detail = {}
        best_f1 = 0
        for key, res in metrics.items():
            if isinstance(res, dict):
                f1 = res.get("f1", 0)
                detail[f"{key}_F1"] = f1
                detail[f"{key}_lat_p50"] = res.get("latency_p50", "")
                if isinstance(f1, (int, float)) and f1 > best_f1:
                    best_f1 = f1
        return {"experiment": "E3", "description": "Verifier K sweep",
                "headline_metric": "best_F1",
                "headline_value": best_f1, "detail": detail}

    elif exp_id == "e1_5":
        detail = {}
        for key, res in metrics.items():
            if isinstance(res, dict):
                detail[f"{key}_F1"] = res.get("f1", "")
                detail[f"{key}_ECE"] = res.get("verifier_ece", "")
                detail[f"{key}_AUC"] = res.get("verifier_auc", "")
        return {"experiment": "E1.5", "description": "Backbone recalibration",
                "headline_metric": "configs_tested",
                "headline_value": len(metrics), "detail": detail}

    elif exp_id == "e1_6":
        detail = {}
        for key, res in metrics.items():
            if isinstance(res, dict):
                detail[f"{key}_F1"] = res.get("f1", res.get("trident_f1", ""))
                detail[f"{key}_dF1"] = res.get("delta_f1_vs_topk", "")
        return {"experiment": "E1.6", "description": "Retrieval sensitivity",
                "headline_metric": "configs_tested",
                "headline_value": len(metrics), "detail": detail}

    elif exp_id == "e1_7":
        detail = {}
        for key, res in metrics.items():
            if isinstance(res, dict):
                detail[f"{key}_abstain"] = res.get("abstention_rate", "")
                detail[f"{key}_F1"] = res.get("overall_f1", res.get("answered_f1", ""))
        return {"experiment": "E1.7", "description": "Safe-Cover curve",
                "headline_metric": "alphas_tested",
                "headline_value": len(metrics), "detail": detail}

    elif exp_id == "e5":
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

    return {"experiment": exp_id, "description": "",
            "headline_metric": "raw_keys",
            "headline_value": list(metrics.keys())[:5], "detail": {}}


def aggregate_results(output_root: str, statuses: List[Dict]) -> str:
    """Read all per-experiment reports and produce a unified aggregate report."""
    root = Path(output_root)
    headlines = []
    all_metrics = {}

    for s in statuses:
        exp_id = s["exp_id"]
        exp_dir = root / exp_id
        report_path = _find_report_json(str(exp_dir))

        if report_path is None:
            headlines.append({
                "experiment": exp_id, "description": "",
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
        headline["gpu_id"] = s.get("gpu_id", "")
        headlines.append(headline)
        all_metrics[exp_id] = {
            "report_path": report_path,
            "metadata": report.get("metadata", {}),
            "metrics": report.get("metrics", {}),
            "compute": report.get("compute", {}),
            "summary_table": report.get("summary_table", ""),
        }

    # Build unified table
    table_lines = ["", "=" * 80, "  AGGREGATE REBUTTAL RESULTS", "=" * 80, ""]
    table_lines.append(
        f"| {'Exp':<6} | {'Description':<25} | {'Status':<7} "
        f"| {'Headline Metric':<25} | {'Value':<12} | {'GPU':>3} | {'Time':>6} |"
    )
    table_lines.append(
        f"|{'-'*8}|{'-'*27}|{'-'*9}"
        f"|{'-'*27}|{'-'*14}|{'-'*5}|{'-'*8}|"
    )

    for h in headlines:
        val = h["headline_value"]
        val_str = f"{val:.4f}" if isinstance(val, float) else str(val)[:12]
        elapsed = h.get("elapsed_s", 0)
        gpu = h.get("gpu_id", "")
        table_lines.append(
            f"| {h['experiment']:<6} | {h['description']:<25} "
            f"| {h.get('status', '?'):<7} "
            f"| {h['headline_metric']:<25} | {val_str:<12} "
            f"| {str(gpu):>3} | {elapsed:>5.0f}s |"
        )

    table_lines.append("")

    for h in headlines:
        if not h.get("detail"):
            continue
        table_lines.append(f"  [{h['experiment']}] {h['description']} -- detail:")
        for k, v in h["detail"].items():
            table_lines.append(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")
        table_lines.append("")

    aggregate_table = "\n".join(table_lines)
    print(aggregate_table)

    aggregate_path = root / "aggregate_results.json"
    with open(aggregate_path, "w") as f:
        json.dump({
            "headlines": headlines,
            "per_experiment": all_metrics,
            "aggregate_table": aggregate_table,
        }, f, indent=2, default=str)

    print(f"  Aggregate results saved to {aggregate_path}")
    return str(aggregate_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run all rebuttal experiments (concurrent across GPUs)"
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

    # Model / GPU
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--backbones", nargs="+", default=[])
    parser.add_argument("--device", type=int, default=0,
                        help="GPU id for single-GPU mode (when --gpu_ids not set)")
    parser.add_argument("--gpu_ids", type=str, default="",
                        help="Comma-separated GPU ids for concurrent execution "
                             "(e.g. '2,3,4,5,6'). Enables experiment-level parallelism.")
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
                        help="Print commands and GPU assignments without running")

    args = parser.parse_args()

    # Parse GPU ids: --gpu_ids flag > CUDA_VISIBLE_DEVICES env > empty (sequential)
    if args.gpu_ids:
        gpu_ids = [int(x) for x in args.gpu_ids.split(",") if x.strip()]
    else:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        gpu_ids = [int(x.strip()) for x in visible.split(",") if x.strip()] if visible else []
    concurrent = len(gpu_ids) > 0

    # Filter experiments
    experiments = list(EXPERIMENT_ORDER)
    if args.minimal:
        experiments = [e for e in experiments if e[0] in MINIMAL_SUBSET]
    if args.tiers is not None:
        experiments = [e for e in experiments if e[2] in args.tiers]
    if args.only:
        experiments = [e for e in experiments if e[0] in args.only]
    experiments = [e for e in experiments if e[0] not in args.skip]

    if not experiments:
        print("No experiments selected. Check --minimal, --tiers, --only, --skip flags.")
        return

    # Separate CPU and GPU experiments
    cpu_experiments = [e for e in experiments if not e[5]]
    gpu_experiments = [e for e in experiments if e[5]]

    n_gpus = len(gpu_ids) if concurrent else 1
    mode_str = f"concurrent ({n_gpus} GPUs: {','.join(str(g) for g in gpu_ids)})" if concurrent else f"sequential (GPU {args.device})"
    print(f"Rebuttal experiment suite: {len(experiments)} experiments")
    print(f"  Mode: {mode_str}")
    print(f"  CPU experiments: {len(cpu_experiments)} ({', '.join(e[0] for e in cpu_experiments) or 'none'})")
    print(f"  GPU experiments: {len(gpu_experiments)} ({', '.join(e[0] for e in gpu_experiments) or 'none'})")
    print(f"  Output root: {args.output_root}")

    # --- Dry run: show assignments ----------------------------------------
    if args.dry_run:
        print("\n[DRY RUN] Execution plan:")
        if cpu_experiments:
            print("\n  Batch 0 (CPU, concurrent):")
            for e in cpu_experiments:
                cmd = build_cmd(e[0], e[1], args)
                status = "SKIP (missing data)" if cmd is None else ' '.join(cmd)
                print(f"    [{e[0]}] {e[3]}")
                print(f"      {status}")

        # Batch GPU experiments
        batches = _batch_experiments(gpu_experiments, gpu_ids if concurrent else [args.device])
        for batch_idx, batch in enumerate(batches, 1):
            print(f"\n  Batch {batch_idx} (GPU, concurrent):")
            for exp_tuple, gid in batch:
                cmd = build_cmd(exp_tuple[0], exp_tuple[1], args, gpu_id=gid)
                status = "SKIP (missing data)" if cmd is None else ' '.join(cmd)
                print(f"    [{exp_tuple[0]}] {exp_tuple[3]} -> GPU {gid}")
                print(f"      {status}")
        return

    # --- Execute ----------------------------------------------------------
    all_statuses = []
    total_t0 = time.time()

    # Phase 1: CPU experiments (concurrent)
    if cpu_experiments:
        print(f"\n{'='*70}")
        print(f"  PHASE 1: CPU experiments ({len(cpu_experiments)})")
        print(f"{'='*70}")
        running = []
        for exp_tuple in cpu_experiments:
            proc, info = _launch_experiment(
                exp_tuple[0], exp_tuple[1], exp_tuple[2], exp_tuple[3], args,
            )
            if proc is None:
                all_statuses.append(info)
            else:
                running.append(info)
        if running:
            batch_statuses = _wait_for_batch(running, args.timeout)
            all_statuses.extend(batch_statuses)

    # Phase 2: GPU experiments
    if gpu_experiments:
        if concurrent:
            # Concurrent: batch across available GPUs
            batches = _batch_experiments(gpu_experiments, gpu_ids)
            for batch_idx, batch in enumerate(batches, 1):
                print(f"\n{'='*70}")
                print(f"  PHASE 2, BATCH {batch_idx}/{len(batches)}: "
                      f"{len(batch)} experiments across GPUs")
                print(f"{'='*70}")
                running = []
                for exp_tuple, gid in batch:
                    proc, info = _launch_experiment(
                        exp_tuple[0], exp_tuple[1], exp_tuple[2], exp_tuple[3],
                        args, gpu_id=gid,
                    )
                    if proc is None:
                        all_statuses.append(info)
                    else:
                        running.append(info)
                if running:
                    batch_statuses = _wait_for_batch(running, args.timeout)
                    all_statuses.extend(batch_statuses)
        else:
            # Sequential: one at a time on single GPU
            print(f"\n{'='*70}")
            print(f"  PHASE 2: GPU experiments (sequential, GPU {args.device})")
            print(f"{'='*70}")
            for exp_tuple in gpu_experiments:
                status = run_experiment_sequential(
                    exp_tuple[0], exp_tuple[1], exp_tuple[2], exp_tuple[3], args,
                )
                all_statuses.append(status)

    total_elapsed = time.time() - total_t0

    # --- Summary ----------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  REBUTTAL EXPERIMENT SUMMARY")
    print(f"{'='*70}")
    passed = sum(1 for s in all_statuses if s["status"] == "passed")
    failed = sum(1 for s in all_statuses if s["status"] == "failed")
    skipped = sum(1 for s in all_statuses if s["status"] == "skipped")
    print(f"  Total: {len(all_statuses)} | Passed: {passed} | Failed: {failed} | Skipped: {skipped}")
    print(f"  Wall time: {total_elapsed:.0f}s")

    for s in all_statuses:
        icon = {"passed": "OK", "failed": "FAIL", "skipped": "SKIP",
                "timeout": "TIME", "error": "ERR"}.get(s["status"], "?")
        elapsed = s.get("elapsed_s", 0)
        gpu = s.get("gpu_id", "")
        gpu_str = f" GPU {gpu}" if gpu != "" else ""
        print(f"  [{icon}] {s['exp_id']:8s} ({elapsed:.0f}s{gpu_str})")

    # Save run summary
    summary_path = Path(args.output_root) / "run_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    # Strip non-serializable fields
    clean_statuses = [
        {k: v for k, v in s.items() if not k.startswith("_")}
        for s in all_statuses
    ]
    with open(summary_path, "w") as f:
        json.dump({
            "statuses": clean_statuses,
            "total_elapsed_s": round(total_elapsed, 1),
            "gpu_ids": gpu_ids,
            "concurrent": concurrent,
            "args": {k: str(v) for k, v in vars(args).items()},
        }, f, indent=2)
    print(f"\n  Run summary saved to {summary_path}")

    # Aggregate results
    passed_or_done = [s for s in all_statuses if s["status"] != "skipped"]
    if passed_or_done:
        print(f"\n{'='*70}")
        print(f"  AGGREGATING RESULTS")
        print(f"{'='*70}")
        aggregate_results(args.output_root, all_statuses)


def _batch_experiments(
    experiments: List[Tuple],
    gpu_ids: List[int],
) -> List[List[Tuple]]:
    """Assign experiments to GPUs in round-robin batches.

    Returns list of batches, where each batch is a list of
    (experiment_tuple, gpu_id) pairs that can run concurrently.
    """
    n_gpus = len(gpu_ids)
    if n_gpus == 0:
        return [[(e, 0)] for e in experiments]

    batches = []
    for i in range(0, len(experiments), n_gpus):
        batch = []
        for j, exp in enumerate(experiments[i:i + n_gpus]):
            batch.append((exp, gpu_ids[j % n_gpus]))
        batches.append(batch)
    return batches


if __name__ == "__main__":
    main()
