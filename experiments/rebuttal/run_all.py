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
from typing import List, Optional


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

    # Save summary
    summary_path = Path(args.output_root) / "run_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump({
            "statuses": statuses,
            "total_elapsed_s": round(total_elapsed, 1),
            "args": {k: str(v) for k, v in vars(args).items()},
        }, f, indent=2)
    print(f"\n  Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
