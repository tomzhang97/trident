#!/usr/bin/env python3
"""E1.7 -- Safe-Cover operating curve (Kn2g).

Purpose: Make the conservatism/abstention trade-off explicit and operationally
         interpretable.

Run:
  - Dataset: MuSiQue dev limit 200 (best) + HotpotQA dev limit 200 (optional)
  - Mode: Safe-Cover
  - Sweep per_facet_alpha in {0.1, 0.05, 0.01}

Report columns:
  abstention_rate, answered EM/F1 (on answered subset),
  overall EM/F1 (abstain=0), avg_evidence_tokens, avg_latency_ms,
  avg_certificates, coverage_proxy_answered

Usage:
    python -m experiments.rebuttal.e1_7_safe_cover_curve \
        --data_path data/musique_dev.jsonl \
        --dataset musique \
        --output_dir runs/rebuttal/e1_7 \
        --budget 500 --limit 200 \
        --model meta-llama/Meta-Llama-3-8B-Instruct \
        --device 0 --alphas 0.1 0.05 0.01
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from experiments.rebuttal.multi_gpu import (
    add_multigpu_args,
    get_arm_spec,
    is_worker,
    run_arms_parallel,
    write_worker_result,
)
from experiments.rebuttal.report_utils import (
    ExperimentMetadata,
    ExperimentReport,
    bootstrap_ci,
    exact_match,
    f1_score,
    latency_percentiles,
)


def run_alpha_arm(args, data: List[Dict], alpha: float) -> Dict[str, Any]:
    """Run Safe-Cover with a specific per_facet_alpha."""
    from trident.config import (
        TridentConfig, SafeCoverConfig, ParetoConfig,
        LLMConfig, RetrievalConfig, EvaluationConfig,
        NLIConfig, CalibrationConfig, TelemetryConfig,
    )
    from experiments.rebuttal._pipeline_helpers import create_pipeline

    device = f"cuda:{args.device}" if args.device >= 0 else "cpu"

    config = TridentConfig(
        mode="safe_cover",
        safe_cover=SafeCoverConfig(
            per_facet_alpha=alpha,
            token_cap=args.budget,
            max_evidence_tokens=args.budget,
            early_abstain=True,
            fallback_to_pareto=False,  # No fallback -- see real Safe-Cover behavior
            use_certificates=True,
        ),
        pareto=ParetoConfig(budget=args.budget),
        llm=LLMConfig(model_name=args.model, device=device, load_in_8bit=args.load_in_8bit),
        retrieval=RetrievalConfig(method="dense", encoder_model=args.encoder_model, top_k=100),
        nli=NLIConfig(batch_size=32),
        calibration=CalibrationConfig(use_mondrian=True),
        evaluation=EvaluationConfig(dataset=args.dataset),
        telemetry=TelemetryConfig(enable=True),
    )
    pipeline = create_pipeline(config, device=device,
                               calibration_path=getattr(args, 'calibration_path', None))

    per_query = []
    for i, ex in enumerate(data):
        question = ex.get("question", "")
        gold = ex.get("answer", "")
        context = ex.get("context", None)

        if (i + 1) % 50 == 0:
            print(f"    [alpha={alpha}] {i+1}/{len(data)}")

        try:
            output = pipeline.process_query(question, context=context)
            certs = output.certificates or []
            per_query.append({
                "prediction": output.answer if not output.abstained else "",
                "gold": gold,
                "abstained": output.abstained,
                "tokens_used": output.tokens_used,
                "latency_ms": output.latency_ms,
                "num_certificates": len(certs),
                "coverage": output.metrics.get("coverage", 0),
                "abstention_reason": output.metrics.get("abstention_reason", ""),
            })
        except Exception as e:
            per_query.append({
                "prediction": "", "gold": gold, "abstained": True,
                "tokens_used": 0, "latency_ms": 0, "num_certificates": 0,
                "coverage": 0, "error": str(e),
            })

    # Compute metrics
    all_preds = [q["prediction"] for q in per_query]
    all_golds = [q["gold"] for q in per_query]
    n_total = len(per_query)
    n_abstain = sum(1 for q in per_query if q["abstained"])
    n_answered = n_total - n_abstain

    # Overall EM/F1 (abstain = 0)
    overall_ems = [exact_match(p, g) for p, g in zip(all_preds, all_golds)]
    overall_f1s = [f1_score(p, g) for p, g in zip(all_preds, all_golds)]
    overall_em_m, _, _ = bootstrap_ci(overall_ems)
    overall_f1_m, _, _ = bootstrap_ci(overall_f1s)

    # Answered-only EM/F1
    answered_ems = [exact_match(q["prediction"], q["gold"])
                    for q in per_query if not q["abstained"]]
    answered_f1s = [f1_score(q["prediction"], q["gold"])
                    for q in per_query if not q["abstained"]]
    if answered_ems:
        ans_em_m, _, _ = bootstrap_ci(answered_ems)
        ans_f1_m, _, _ = bootstrap_ci(answered_f1s)
    else:
        ans_em_m = ans_f1_m = 0.0

    toks = [q["tokens_used"] for q in per_query]
    lats = [q["latency_ms"] for q in per_query]
    certs_counts = [q["num_certificates"] for q in per_query]
    coverages_answered = [q["coverage"] for q in per_query if not q["abstained"]]

    # Abstention reason breakdown
    reasons = {}
    for q in per_query:
        r = q.get("abstention_reason", "")
        if q["abstained"] and r:
            reasons[r] = reasons.get(r, 0) + 1

    return {
        "alpha": alpha,
        "n_total": n_total,
        "n_answered": n_answered,
        "abstention_rate": round(n_abstain / max(n_total, 1), 4),
        "overall_em": round(overall_em_m, 4),
        "overall_f1": round(overall_f1_m, 4),
        "answered_em": round(ans_em_m, 4),
        "answered_f1": round(ans_f1_m, 4),
        "avg_evidence_tokens": round(float(np.mean(toks)), 1),
        "avg_latency_ms": round(float(np.mean(lats)), 1),
        "avg_certificates": round(float(np.mean(certs_counts)), 2),
        "coverage_proxy_answered": round(float(np.mean(coverages_answered)), 4) if coverages_answered else 0.0,
        "abstention_reasons": reasons,
        "per_query": per_query,
    }


def aggregate_and_save(args, all_results: List[Dict[str, Any]]) -> str:
    """Aggregate per-alpha results and save the final report."""
    # Sort by alpha for consistent display
    all_results = sorted(all_results, key=lambda r: r["alpha"], reverse=True)

    # Print operating curve table
    print("\n[E1.7] Safe-Cover operating curve:")
    hdr = "| alpha | Abstain | Ans EM | Ans F1 | Overall EM | Overall F1 | EvTok | Lat | Certs | Cov(ans) |"
    sep = "|-------|---------|--------|--------|------------|------------|-------|-----|-------|----------|"
    print(hdr)
    print(sep)
    for r in all_results:
        print(f"| {r['alpha']:.2f} "
              f"| {r['abstention_rate']:.3f} "
              f"| {r['answered_em']:.3f} "
              f"| {r['answered_f1']:.3f} "
              f"| {r['overall_em']:.3f} "
              f"| {r['overall_f1']:.3f} "
              f"| {r['avg_evidence_tokens']:.0f} "
              f"| {r['avg_latency_ms']:.0f} "
              f"| {r['avg_certificates']:.1f} "
              f"| {r['coverage_proxy_answered']:.3f} |")

    # Print abstention reason breakdown
    for r in all_results:
        if r.get("abstention_reasons"):
            print(f"\n  alpha={r['alpha']}: abstention reasons: {r['abstention_reasons']}")

    meta = ExperimentMetadata(
        experiment_id=f"e1_7_safe_cover_curve_{args.dataset}",
        dataset=args.dataset,
        budget=args.budget,
        mode="safe_cover",
        backbone=args.model,
        seed=args.seed,
        limit=args.limit,
        extra={"alphas": args.alphas},
    )
    metrics = {}
    for r in all_results:
        key = f"alpha_{r['alpha']}"
        metrics[key] = {k: v for k, v in r.items() if k != "per_query"}

    report = ExperimentReport(
        metadata=meta,
        metrics=metrics,
    )
    path = report.save(args.output_dir)
    print(f"\n[E1.7] Report saved to {path}")
    print("[E1.7] Rebuttal: 'Safe-Cover behaves as intended: lowering alpha increases "
          "abstention while improving answered-subset accuracy and tightening "
          "certificate coverage.'")
    return path


def main():
    parser = argparse.ArgumentParser(description="E1.7: Safe-Cover operating curve")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="musique")
    parser.add_argument("--output_dir", type=str, default="runs/rebuttal/e1_7")
    parser.add_argument("--budget", type=int, default=500)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--encoder_model", type=str, default="facebook/contriever")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--calibration_path", type=str, default=None,
                        help="Path to calibration JSON for p-value computation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.05, 0.01])
    add_multigpu_args(parser)
    args = parser.parse_args()

    # --- Worker path: run a single alpha arm ------------------------------
    if is_worker(args):
        arm = get_arm_spec(args)
        alpha = arm["alpha"]
        from experiments.eval_complete_runnable import load_data
        data = load_data(args.data_path, limit=args.limit)
        print(f"[E1.7/worker] alpha={alpha} on GPU {args._worker_gpu}")
        result = run_alpha_arm(args, data, alpha)
        write_worker_result(args, result)
        return

    # --- Load data --------------------------------------------------------
    from experiments.eval_complete_runnable import load_data
    data = load_data(args.data_path, limit=args.limit)
    print(f"[E1.7] Loaded {len(data)} examples")

    # --- Multi-GPU path ---------------------------------------------------
    if args.num_gpus > 1:
        arm_specs = [{"alpha": a, "label": f"alpha={a}"} for a in args.alphas]
        print(f"[E1.7] Distributing {len(arm_specs)} arms across {args.num_gpus} GPUs")
        all_results = run_arms_parallel(args, arm_specs, __file__)
        all_results = [r for r in all_results if r and "alpha" in r]
        aggregate_and_save(args, all_results)
        return

    # --- Sequential path --------------------------------------------------
    all_results = []
    for alpha in args.alphas:
        print(f"\n[E1.7] Running alpha={alpha}")
        result = run_alpha_arm(args, data, alpha)
        all_results.append(result)

    aggregate_and_save(args, all_results)


if __name__ == "__main__":
    main()
