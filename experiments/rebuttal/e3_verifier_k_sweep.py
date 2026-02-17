#!/usr/bin/env python3
"""E3 -- Verifier compute / shortlist K sweep (tightened).

Purpose: Answer 22Vu/Kn2g "cost trade-offs; tightening threshold spikes
         abstentions."

Sweep rerank_top_k: K in {8, 16, 32}.  Run TRIDENT-Pareto at B=500 (and
optionally B=1000).

Report columns:
  EM/F1, avg_evidence_tokens, p50/p95 latency, total_pairs_scored,
  delta_pairs_pct vs Kmax, abstention_rate

Usage:
    python -m experiments.rebuttal.e3_verifier_k_sweep \
        --data_path data/hotpotqa_dev.json \
        --dataset hotpotqa \
        --output_dir runs/rebuttal/e3 \
        --config_family pareto_match_500_alpha06 --limit 500 \
        --model meta-llama/Meta-Llama-3-8B-Instruct \
        --device 0
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
    delta_f1_table,
    exact_match,
    f1_score,
    latency_percentiles,
)


def run_k_sweep_arm(args, data, rerank_top_k: int) -> Dict[str, Any]:
    """Run TRIDENT-Pareto with a specific rerank_top_k."""
    from experiments.rebuttal._pipeline_helpers import build_config, create_pipeline

    device = f"cuda:{args.device}" if args.device >= 0 else "cpu"

    config = build_config(args, retrieval_overrides=dict(rerank_top_k=rerank_top_k),
                          telemetry_overrides=dict(track_latency=True))
    pipeline = create_pipeline(config, device=device,
                               calibration_path=getattr(args, 'calibration_path', None))

    per_query = []
    for i, ex in enumerate(data):
        question = ex.get("question", "")
        gold = ex.get("answer", "")
        context = ex.get("context", None)

        if (i + 1) % 50 == 0:
            print(f"    [K={rerank_top_k}] {i+1}/{len(data)}")

        try:
            output = pipeline.process_query(question, context=context)
            trace = output.trace or {}
            scoring_info = trace.get("scoring", {})
            per_query.append({
                "prediction": output.answer if not output.abstained else "",
                "gold": gold,
                "abstained": output.abstained,
                "tokens_used": output.tokens_used,
                "latency_ms": output.latency_ms,
                "total_pairs_scored": scoring_info.get("num_scores", 0),
            })
        except Exception as e:
            per_query.append({
                "prediction": "", "gold": gold, "abstained": True,
                "tokens_used": 0, "latency_ms": 0, "total_pairs_scored": 0,
                "error": str(e),
            })

    # Aggregate
    preds = [q["prediction"] for q in per_query]
    golds = [q["gold"] for q in per_query]
    ems = [exact_match(p, g) for p, g in zip(preds, golds)]
    f1s = [f1_score(p, g) for p, g in zip(preds, golds)]
    lats = [q["latency_ms"] for q in per_query]
    toks = [q["tokens_used"] for q in per_query]
    pairs = [q["total_pairs_scored"] for q in per_query]
    abstain = sum(1 for q in per_query if q["abstained"])

    em_m, em_lo, em_hi = bootstrap_ci(ems)
    f1_m, f1_lo, f1_hi = bootstrap_ci(f1s)
    lat = latency_percentiles(lats)

    budget = config.pareto.budget
    return {
        "rerank_top_k": rerank_top_k,
        "budget": budget,
        "n_queries": len(data),
        "em": round(em_m, 4),
        "em_ci": [round(em_lo, 4), round(em_hi, 4)],
        "f1": round(f1_m, 4),
        "f1_ci": [round(f1_lo, 4), round(f1_hi, 4)],
        "abstention_rate": round(abstain / max(len(data), 1), 4),
        "avg_evidence_tokens": round(float(np.mean(toks)), 1),
        "latency_p50": round(lat["p50"], 1),
        "latency_p95": round(lat["p95"], 1),
        "latency_mean": round(lat["mean"], 1),
        "total_pairs_scored_mean": round(float(np.mean(pairs)), 1),
        "total_pairs_scored_p50": round(float(np.percentile(pairs, 50)), 1),
        "per_query": per_query,
    }


def aggregate_and_save(args, all_results: List[Dict[str, Any]]) -> str:
    """Aggregate per-arm results and save the final report."""
    # Compute delta_pairs_pct vs Kmax
    k_max_pairs = max(
        (r["total_pairs_scored_mean"] for r in all_results), default=1
    )
    for r in all_results:
        r["delta_pairs_pct"] = round(
            (r["total_pairs_scored_mean"] - k_max_pairs) / max(k_max_pairs, 1) * 100, 1
        )

    # Build table
    budget = all_results[0]["budget"] if all_results else 0
    table_rows = []
    for r in all_results:
        table_rows.append({
            "label": f"K={r['rerank_top_k']}_B={r['budget']}",
            "em": r["em"],
            "f1": r["f1"],
            "abstention_rate": r["abstention_rate"],
            "avg_evidence_tokens": r["avg_evidence_tokens"],
            "latency_p50": r["latency_p50"],
        })

    table = delta_f1_table(table_rows, baseline_key=f"K={max(args.ks)}_B={budget}")
    print(f"\n{table}")

    # Print sweep summary
    print("\n[E3] Sweep summary:")
    hdr = "| K | B | EM | F1 | Pairs | dPairs% | Lat p50 | Lat p95 | Abstain |"
    sep = "|---|---|----|----|-------|---------|---------|---------|---------|"
    print(hdr)
    print(sep)
    for r in sorted(all_results, key=lambda x: (x["budget"], x["rerank_top_k"])):
        print(f"| {r['rerank_top_k']} | {r['budget']} "
              f"| {r['em']:.3f} | {r['f1']:.3f} "
              f"| {r['total_pairs_scored_mean']:.0f} "
              f"| {r.get('delta_pairs_pct', 0):+.1f}% "
              f"| {r['latency_p50']:.0f} | {r['latency_p95']:.0f} "
              f"| {r['abstention_rate']:.3f} |")

    meta = ExperimentMetadata(
        experiment_id=f"e3_k_sweep_{args.dataset}",
        dataset=args.dataset,
        mode="pareto",
        backbone=args.model,
        seed=args.seed,
        limit=args.limit,
        extra={"config_family": args.config_family, "ks": args.ks},
    )
    metrics = {f"K={r['rerank_top_k']}_B={r['budget']}": {k: v for k, v in r.items() if k != "per_query"}
               for r in all_results}

    report = ExperimentReport(
        metadata=meta,
        metrics=metrics,
        summary_table=table,
    )
    path = report.save(args.output_dir)
    print(f"\n[E3] Report saved to {path}")
    print("[E3] Rebuttal: 'Reducing K cuts verifier pairs by X% and latency by Y ms "
          "with only dF1 = -Z (smooth knob).'")
    return path


def main():
    parser = argparse.ArgumentParser(description="E3: Verifier K sweep")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="hotpotqa")
    parser.add_argument("--output_dir", type=str, default="runs/rebuttal/e3")
    parser.add_argument("--config_family", type=str, required=True,
                        help="Config family name (e.g. pareto_match_500_alpha06)")
    parser.add_argument("--ks", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--encoder_model", type=str, default="facebook/contriever")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--calibration_path", type=str, default=None,
                        help="Path to calibration JSON for p-value computation")
    parser.add_argument("--seed", type=int, default=42)
    add_multigpu_args(parser)
    args = parser.parse_args()

    # --- Worker path: run a single K arm ------------------------------------
    if is_worker(args):
        arm = get_arm_spec(args)
        k = arm["k"]
        from experiments.eval_complete_runnable import load_data
        data = load_data(args.data_path, limit=args.limit)
        print(f"[E3/worker] K={k} on GPU {args._worker_gpu}")
        result = run_k_sweep_arm(args, data, k)
        write_worker_result(args, result)
        return

    # --- Load data --------------------------------------------------------
    from experiments.eval_complete_runnable import load_data
    data = load_data(args.data_path, limit=args.limit)
    print(f"[E3] Loaded {len(data)} examples")

    # --- Multi-GPU path ---------------------------------------------------
    if args.num_gpus > 1:
        arm_specs = []
        for k in args.ks:
            arm_specs.append({
                "k": k,
                "label": f"K={k}",
            })
        print(f"[E3] Distributing {len(arm_specs)} arms across {args.num_gpus} GPUs")
        all_results = run_arms_parallel(args, arm_specs, __file__)
        # Filter out failed arms
        all_results = [r for r in all_results if r and "rerank_top_k" in r]
        aggregate_and_save(args, all_results)
        return

    # --- Sequential path --------------------------------------------------
    all_results = []
    for k in args.ks:
        print(f"\n[E3] Running K={k}")
        result = run_k_sweep_arm(args, data, k)
        all_results.append(result)

    aggregate_and_save(args, all_results)


if __name__ == "__main__":
    main()
