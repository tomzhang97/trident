#!/usr/bin/env python3
"""E1 -- Latency breakdown (tightened).

Purpose: Answer 22Vu/Kn2g "Safe-Cover overhead" and make parallelizability
         credible.

Run modes: safe_cover and pareto at B=500 on HotpotQA dev (limit 200) and
           optionally MuSiQue dev (limit 200).

Instrumentation:
  - Per-stage wall_ms breakdown: facet_mining, retrieval, rerank,
    verification, calibration, selection, generation
  - verification.items (total_pairs_scored)
  - verification.num_calls (num_batches)

Report columns:
  p50/p95 latency, stage breakdown, total_pairs_scored, num_batches,
  avg_evidence_tokens

Usage:
    python -m experiments.rebuttal.e1_latency_breakdown \
        --data_path data/hotpotqa_dev.json \
        --dataset hotpotqa \
        --output_dir runs/rebuttal/e1 \
        --budget 500 --limit 200 \
        --model meta-llama/Meta-Llama-3-8B-Instruct \
        --device 0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    StageTimer,
    bootstrap_ci,
    compute_em_f1,
    delta_f1_table,
    exact_match,
    f1_score,
    latency_percentiles,
)


def build_config(args, mode: str):
    """Build a TridentConfig for the given mode."""
    from trident.config import (
        TridentConfig, SafeCoverConfig, ParetoConfig,
        LLMConfig, RetrievalConfig, EvaluationConfig,
        NLIConfig, CalibrationConfig, TelemetryConfig,
    )
    device = f"cuda:{args.device}" if args.device >= 0 else "cpu"

    llm_cfg = LLMConfig(
        model_name=args.model,
        device=device,
        load_in_8bit=args.load_in_8bit,
        temperature=0.0,
    )
    retrieval_cfg = RetrievalConfig(
        method="dense",
        encoder_model=args.encoder_model,
        top_k=100,
    )
    nli_cfg = NLIConfig(batch_size=32)
    cal_cfg = CalibrationConfig(use_mondrian=True)
    eval_cfg = EvaluationConfig(dataset=args.dataset)
    tel_cfg = TelemetryConfig(enable=True, track_latency=True)

    if mode == "safe_cover":
        sc_cfg = SafeCoverConfig(
            per_facet_alpha=0.05,
            token_cap=args.budget,
            max_evidence_tokens=args.budget,
            early_abstain=True,
            fallback_to_pareto=False,
            use_certificates=True,
        )
        p_cfg = ParetoConfig(budget=args.budget)
        return TridentConfig(
            mode="safe_cover", safe_cover=sc_cfg, pareto=p_cfg,
            llm=llm_cfg, retrieval=retrieval_cfg, nli=nli_cfg,
            calibration=cal_cfg, evaluation=eval_cfg, telemetry=tel_cfg,
        )
    else:
        p_cfg = ParetoConfig(
            budget=args.budget,
            max_evidence_tokens=args.budget,
            max_units=8,
            stop_on_budget=True,
            use_vqc=False,
            use_bwk=False,
        )
        sc_cfg = SafeCoverConfig()
        return TridentConfig(
            mode="pareto", safe_cover=sc_cfg, pareto=p_cfg,
            llm=llm_cfg, retrieval=retrieval_cfg, nli=nli_cfg,
            calibration=cal_cfg, evaluation=eval_cfg, telemetry=tel_cfg,
        )


def run_with_instrumentation(
    pipeline,
    query: str,
    context: Any,
    mode: str,
) -> Dict[str, Any]:
    """Run a single query and collect stage-level timing."""
    timer = StageTimer()

    # Overall start
    t0 = time.perf_counter()

    # We use the pipeline's process_query and then extract trace info
    output = pipeline.process_query(query, context=context, mode=mode)
    total_ms = (time.perf_counter() - t0) * 1000

    # Extract trace from pipeline telemetry
    trace = output.trace if output.trace else {}
    events = trace.get("events", [])

    # Build stage timing from trace events
    stage_times = {}
    for ev in events:
        name = ev.get("name", "")
        duration = ev.get("duration_ms", 0)
        if name and duration > 0:
            stage_times[name] = stage_times.get(name, 0) + duration

    # Verification compute stats from NLI scorer cache
    nli_data = trace.get("scoring", {})
    total_pairs = nli_data.get("num_scores", 0)
    # Estimate batches: NLI batch size is typically 32
    num_batches = max(1, (total_pairs + 31) // 32) if total_pairs > 0 else 0

    return {
        "answer": output.answer,
        "abstained": output.abstained,
        "tokens_used": output.tokens_used,
        "latency_ms": total_ms,
        "pipeline_latency_ms": output.latency_ms,
        "stage_times_ms": stage_times,
        "total_pairs_scored": total_pairs,
        "num_batches": num_batches,
        "num_facets": len(output.facets) if output.facets else 0,
        "num_passages_selected": len(output.selected_passages),
        "certificates": len(output.certificates) if output.certificates else 0,
        "mode": output.mode,
    }


def run_experiment(args, mode: str, data: List[Dict]) -> Dict[str, Any]:
    """Run the full experiment for one mode."""
    from trident.pipeline import TridentPipeline

    config = build_config(args, mode)
    pipeline = TridentPipeline(config)

    per_query = []
    predictions = []
    references = []

    for i, ex in enumerate(data):
        qid = ex.get("_id", str(i))
        question = ex.get("question", "")
        gold = ex.get("answer", "")
        context = ex.get("context", None)

        print(f"  [{mode}] Query {i+1}/{len(data)}: {question[:60]}...")
        result = run_with_instrumentation(pipeline, question, context, mode)
        result["query_id"] = qid
        result["ground_truth"] = gold

        predictions.append(result["answer"] if not result["abstained"] else "")
        references.append(gold)
        per_query.append(result)

    # Aggregate metrics
    ems = [exact_match(p, r) for p, r in zip(predictions, references)]
    f1s = [f1_score(p, r) for p, r in zip(predictions, references)]
    lats = [q["latency_ms"] for q in per_query]
    toks = [q["tokens_used"] for q in per_query]
    pairs = [q["total_pairs_scored"] for q in per_query]
    batches = [q["num_batches"] for q in per_query]
    abstain_count = sum(1 for q in per_query if q["abstained"])

    em_mean, em_lo, em_hi = bootstrap_ci(ems)
    f1_mean, f1_lo, f1_hi = bootstrap_ci(f1s)
    lat_stats = latency_percentiles(lats)

    # Aggregate stage breakdown
    all_stages: Dict[str, List[float]] = {}
    for q in per_query:
        for stage, ms in q["stage_times_ms"].items():
            all_stages.setdefault(stage, []).append(ms)
    stage_summary = {}
    for stage, times in all_stages.items():
        arr = np.array(times)
        stage_summary[stage] = {
            "p50": round(float(np.percentile(arr, 50)), 1),
            "p95": round(float(np.percentile(arr, 95)), 1),
            "mean": round(float(arr.mean()), 1),
            "pct_of_total": round(float(arr.sum()) / max(sum(lats), 1) * 100, 1),
        }

    return {
        "mode": mode,
        "n_queries": len(data),
        "em": round(em_mean, 4),
        "em_ci": [round(em_lo, 4), round(em_hi, 4)],
        "f1": round(f1_mean, 4),
        "f1_ci": [round(f1_lo, 4), round(f1_hi, 4)],
        "abstention_rate": round(abstain_count / max(len(data), 1), 4),
        "avg_evidence_tokens": round(float(np.mean(toks)), 1),
        "latency": lat_stats,
        "stage_breakdown": stage_summary,
        "verification": {
            "total_pairs_scored_mean": round(float(np.mean(pairs)), 1),
            "total_pairs_scored_p50": round(float(np.percentile(pairs, 50)), 1),
            "num_batches_mean": round(float(np.mean(batches)), 1),
        },
        "per_query": per_query,
    }


def aggregate_and_save(args, results_by_mode: Dict[str, Any]) -> str:
    """Aggregate per-mode results and save the final report. Returns report path."""
    # Build comparison table
    rows = []
    for mode, res in results_by_mode.items():
        rows.append({
            "label": f"TRIDENT-{mode}",
            "em": res["em"],
            "f1": res["f1"],
            "abstention_rate": res["abstention_rate"],
            "avg_evidence_tokens": res["avg_evidence_tokens"],
            "latency_p50": res["latency"]["p50"],
        })
    table = delta_f1_table(rows, baseline_key="TRIDENT-pareto")
    print(f"\n{table}")

    # Print stage breakdown
    for mode, res in results_by_mode.items():
        print(f"\n[E1] Stage breakdown ({mode}):")
        for stage, stats in res["stage_breakdown"].items():
            print(f"  {stage}: p50={stats['p50']:.0f}ms, p95={stats['p95']:.0f}ms, "
                  f"{stats['pct_of_total']:.1f}% of total")
        vf = res["verification"]
        print(f"  Verification: {vf['total_pairs_scored_mean']:.0f} pairs/query, "
              f"{vf['num_batches_mean']:.0f} batches/query")

    # Save report
    modes_list = list(results_by_mode.keys())
    meta = ExperimentMetadata(
        experiment_id=f"e1_latency_{args.dataset}",
        dataset=args.dataset,
        budget=args.budget,
        mode=",".join(modes_list),
        backbone=args.model,
        seed=args.seed,
        limit=args.limit,
    )
    # Strip per_query from top-level metrics to keep report compact
    metrics = {}
    for mode, res in results_by_mode.items():
        m = {k: v for k, v in res.items() if k != "per_query"}
        metrics[mode] = m

    all_per_query = []
    for mode, res in results_by_mode.items():
        for q in res.get("per_query", []):
            q["_mode"] = mode
            all_per_query.append(q)

    report = ExperimentReport(
        metadata=meta,
        metrics=metrics,
        per_query=all_per_query,
        summary_table=table,
    )
    path = report.save(args.output_dir)
    print(f"\n[E1] Report saved to {path}")
    print("[E1] Rebuttal sentence: 'Safe-Cover overhead is dominated by verification "
          "(X% time; Y pairs; Z batches) and is batch-parallelizable.'")
    return path


def main():
    parser = argparse.ArgumentParser(description="E1: Latency breakdown")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="hotpotqa")
    parser.add_argument("--output_dir", type=str, default="runs/rebuttal/e1")
    parser.add_argument("--budget", type=int, default=500)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--encoder_model", type=str, default="facebook/contriever")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--modes", nargs="+", default=["safe_cover", "pareto"],
                        help="Modes to run")
    add_multigpu_args(parser)
    args = parser.parse_args()

    # --- Worker path: run a single mode and write result ------------------
    if is_worker(args):
        arm = get_arm_spec(args)
        mode = arm["mode"]
        from experiments.eval_complete_runnable import load_data
        data = load_data(args.data_path, limit=args.limit)
        print(f"[E1/worker] Running mode={mode} on GPU {args._worker_gpu}")
        result = run_experiment(args, mode, data)
        result["_mode"] = mode
        write_worker_result(args, result)
        return

    # --- Load data --------------------------------------------------------
    from experiments.eval_complete_runnable import load_data
    data = load_data(args.data_path, limit=args.limit)
    print(f"[E1] Loaded {len(data)} examples from {args.data_path}")

    # --- Multi-GPU path: distribute modes across GPUs ---------------------
    if args.num_gpus > 1:
        arm_specs = [{"mode": m, "label": f"mode={m}"} for m in args.modes]
        print(f"[E1] Distributing {len(arm_specs)} arms across {args.num_gpus} GPUs")
        arm_results = run_arms_parallel(args, arm_specs, __file__)
        results_by_mode = {}
        for r in arm_results:
            if r and "_mode" in r:
                results_by_mode[r["_mode"]] = r
            elif r and "mode" in r:
                results_by_mode[r["mode"]] = r
        aggregate_and_save(args, results_by_mode)
        return

    # --- Sequential path --------------------------------------------------
    results_by_mode = {}
    for mode in args.modes:
        print(f"\n[E1] Running mode: {mode}")
        result = run_experiment(args, mode, data)
        results_by_mode[mode] = result

    aggregate_and_save(args, results_by_mode)


if __name__ == "__main__":
    main()
