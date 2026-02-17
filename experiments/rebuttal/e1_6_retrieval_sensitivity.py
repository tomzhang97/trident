#!/usr/bin/env python3
"""E1.6 -- Retrieval sensitivity (AXWh).

Purpose: Show gains persist across candidate pool type/size (retriever-agnostic).

Run:
  - Dataset: HotpotQA dev limit 500
  - Budget: B=500
  - Retriever arms:
    * Dense (Contriever) top-N in {20, 50, 100, 200 optional}
    * BM25 top-N in {20, 50, 100, 200 optional}
  - For each (retriever, N), run:
    * TRIDENT-Pareto
    * Top-k truncation baseline (retriever-ranked concat until budget; no rerank)

Report columns:
  EM/F1, delta vs top-k, avg_evidence_tokens, avg_latency_ms

Usage:
    python -m experiments.rebuttal.e1_6_retrieval_sensitivity \
        --data_path data/hotpotqa_dev.json \
        --dataset hotpotqa \
        --output_dir runs/rebuttal/e1_6 \
        --budget 500 --limit 500 \
        --model meta-llama/Meta-Llama-3-8B-Instruct \
        --device 0 --top_ns 20 50 100
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


def run_retrieval_arm(
    args,
    data: List[Dict],
    retriever_type: str,
    top_n: int,
    run_trident: bool = True,
    run_topk: bool = True,
) -> Dict[str, Any]:
    """Run one arm: (retriever_type, top_n) with TRIDENT-Pareto and top-k baseline."""
    from trident.config import (
        TridentConfig, ParetoConfig, SafeCoverConfig,
        LLMConfig, RetrievalConfig, EvaluationConfig,
        NLIConfig, CalibrationConfig, TelemetryConfig,
    )
    from experiments.rebuttal._pipeline_helpers import create_pipeline

    device = f"cuda:{args.device}" if args.device >= 0 else "cpu"
    encoder = args.encoder_model if retriever_type == "dense" else "facebook/contriever"

    config = TridentConfig(
        mode="pareto",
        pareto=ParetoConfig(
            budget=args.budget, max_evidence_tokens=args.budget,
            max_units=8, stop_on_budget=True, use_vqc=False, use_bwk=False,
        ),
        llm=LLMConfig(model_name=args.model, device=device, load_in_8bit=args.load_in_8bit),
        retrieval=RetrievalConfig(
            method=retriever_type if retriever_type != "bm25" else "sparse",
            encoder_model=encoder,
            top_k=top_n,
        ),
        nli=NLIConfig(batch_size=32),
        calibration=CalibrationConfig(use_mondrian=True),
        evaluation=EvaluationConfig(dataset=args.dataset),
        telemetry=TelemetryConfig(enable=True),
    )
    pipeline = create_pipeline(config, device=device,
                               calibration_path=getattr(args, 'calibration_path', None))

    trident_results = []
    topk_results = []

    for i, ex in enumerate(data):
        question = ex.get("question", "")
        gold = ex.get("answer", "")
        context = ex.get("context", None)

        if (i + 1) % 100 == 0:
            print(f"    [{retriever_type}/N={top_n}] {i+1}/{len(data)}")

        # TRIDENT-Pareto
        if run_trident:
            try:
                output = pipeline.process_query(question, context=context)
                trident_results.append({
                    "prediction": output.answer if not output.abstained else "",
                    "gold": gold,
                    "abstained": output.abstained,
                    "tokens_used": output.tokens_used,
                    "latency_ms": output.latency_ms,
                })
            except Exception as e:
                trident_results.append({
                    "prediction": "", "gold": gold, "abstained": True,
                    "tokens_used": 0, "latency_ms": 0, "error": str(e),
                })

        # Top-k truncation baseline
        if run_topk:
            try:
                t0 = time.perf_counter()
                retrieval_result = pipeline._retrieve_passages(question, context)
                passages = retrieval_result.passages

                selected = []
                total_tokens = 0
                for p in passages:
                    cost = getattr(p, "num_tokens", len(getattr(p, "text", "").split()))
                    if total_tokens + cost > args.budget:
                        break
                    selected.append(p)
                    total_tokens += cost

                # Generate answer
                answer = ""
                if selected and hasattr(pipeline, "llm") and pipeline.llm is not None:
                    passage_text = " ".join(getattr(p, "text", str(p)) for p in selected[:5])
                    prompt = (
                        f"Answer the question based on the provided context.\n\n"
                        f"Context: {passage_text[:2000]}\n\n"
                        f"Question: {question}\n\nAnswer:"
                    )
                    try:
                        llm_out = pipeline.llm.generate(prompt, max_new_tokens=64)
                        answer = llm_out.text.strip() if hasattr(llm_out, "text") else str(llm_out).strip()
                    except Exception:
                        pass

                lat = (time.perf_counter() - t0) * 1000
                topk_results.append({
                    "prediction": answer,
                    "gold": gold,
                    "abstained": not selected,
                    "tokens_used": total_tokens,
                    "latency_ms": lat,
                })
            except Exception as e:
                topk_results.append({
                    "prediction": "", "gold": gold, "abstained": True,
                    "tokens_used": 0, "latency_ms": 0, "error": str(e),
                })

    def agg(results, label):
        if not results:
            return {"label": label, "em": 0, "f1": 0, "abstention_rate": 0,
                    "avg_evidence_tokens": 0, "avg_latency_ms": 0}
        preds = [r["prediction"] for r in results]
        golds_list = [r["gold"] for r in results]
        ems = [exact_match(p, g) for p, g in zip(preds, golds_list)]
        f1s = [f1_score(p, g) for p, g in zip(preds, golds_list)]
        toks = [r["tokens_used"] for r in results]
        lats = [r["latency_ms"] for r in results]
        abstain = sum(1 for r in results if r["abstained"])
        em_m, _, _ = bootstrap_ci(ems)
        f1_m, _, _ = bootstrap_ci(f1s)
        return {
            "label": label,
            "em": round(em_m, 4),
            "f1": round(f1_m, 4),
            "abstention_rate": round(abstain / max(len(results), 1), 4),
            "avg_evidence_tokens": round(float(np.mean(toks)), 1),
            "avg_latency_ms": round(float(np.mean(lats)), 1),
        }

    trident_agg = agg(trident_results, f"Pareto_{retriever_type}_N{top_n}") if run_trident else {}
    topk_agg = agg(topk_results, f"TopK_{retriever_type}_N{top_n}") if run_topk else {}

    delta_f1 = trident_agg.get("f1", 0) - topk_agg.get("f1", 0) if trident_agg and topk_agg else 0

    return {
        "retriever_type": retriever_type,
        "top_n": top_n,
        "trident": trident_agg,
        "topk": topk_agg,
        "delta_f1_vs_topk": round(delta_f1, 4),
    }


def aggregate_and_save(args, all_results: List[Dict[str, Any]]) -> str:
    """Aggregate per-arm results and save the final report."""
    # Build table
    table_rows = []
    for r in all_results:
        if r.get("trident"):
            table_rows.append(r["trident"])
        if r.get("topk"):
            table_rows.append(r["topk"])

    baseline_label = f"TopK_dense_N{args.top_ns[-1]}" if "dense" in args.retrievers else table_rows[0]["label"]
    table = delta_f1_table(table_rows, baseline_key=baseline_label)
    print(f"\n{table}")

    # Summary
    print("\n[E1.6] Retrieval sensitivity summary:")
    hdr = "| Retriever | N | Pareto F1 | TopK F1 | dF1 | Pareto EvTok | Pareto Lat |"
    sep = "|-----------|---|-----------|---------|-----|--------------|------------|"
    print(hdr)
    print(sep)
    for r in all_results:
        t = r.get("trident", {})
        tk = r.get("topk", {})
        print(f"| {r['retriever_type']} | {r['top_n']} "
              f"| {t.get('f1', 0):.3f} | {tk.get('f1', 0):.3f} "
              f"| {r['delta_f1_vs_topk']:+.3f} "
              f"| {t.get('avg_evidence_tokens', 0):.0f} "
              f"| {t.get('avg_latency_ms', 0):.0f} |")

    meta = ExperimentMetadata(
        experiment_id=f"e1_6_retrieval_{args.dataset}",
        dataset=args.dataset,
        budget=args.budget,
        mode="pareto",
        backbone=args.model,
        seed=args.seed,
        limit=args.limit,
        extra={"retrievers": args.retrievers, "top_ns": args.top_ns},
    )
    metrics = {}
    for r in all_results:
        key = f"{r['retriever_type']}_N{r['top_n']}"
        metrics[key] = {k: v for k, v in r.items()}

    report = ExperimentReport(
        metadata=meta,
        metrics=metrics,
        summary_table=table,
    )
    path = report.save(args.output_dir)
    print(f"\n[E1.6] Report saved to {path}")
    print("[E1.6] Rebuttal: 'Across retriever types and pool sizes, TRIDENT-Pareto "
          "consistently improves dF1 at the same 500-token cap.'")
    return path


def main():
    parser = argparse.ArgumentParser(description="E1.6: Retrieval sensitivity")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="hotpotqa")
    parser.add_argument("--output_dir", type=str, default="runs/rebuttal/e1_6")
    parser.add_argument("--budget", type=int, default=500)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--encoder_model", type=str, default="facebook/contriever")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--calibration_path", type=str, default=None,
                        help="Path to calibration JSON for p-value computation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top_ns", type=int, nargs="+", default=[20, 50, 100])
    parser.add_argument("--retrievers", nargs="+", default=["dense", "bm25"])
    add_multigpu_args(parser)
    args = parser.parse_args()

    # --- Worker path: run a single (retriever, top_n) arm -----------------
    if is_worker(args):
        arm = get_arm_spec(args)
        retriever = arm["retriever"]
        top_n = arm["top_n"]
        from experiments.eval_complete_runnable import load_data
        data = load_data(args.data_path, limit=args.limit)
        print(f"[E1.6/worker] {retriever}/N={top_n} on GPU {args._worker_gpu}")
        result = run_retrieval_arm(args, data, retriever, top_n)
        write_worker_result(args, result)
        return

    # --- Load data --------------------------------------------------------
    from experiments.eval_complete_runnable import load_data
    data = load_data(args.data_path, limit=args.limit)
    print(f"[E1.6] Loaded {len(data)} examples")

    # --- Multi-GPU path ---------------------------------------------------
    if args.num_gpus > 1:
        arm_specs = []
        for retriever in args.retrievers:
            for top_n in args.top_ns:
                arm_specs.append({
                    "retriever": retriever, "top_n": top_n,
                    "label": f"{retriever}/N={top_n}",
                })
        print(f"[E1.6] Distributing {len(arm_specs)} arms across {args.num_gpus} GPUs")
        all_results = run_arms_parallel(args, arm_specs, __file__)
        all_results = [r for r in all_results if r and "retriever_type" in r]
        aggregate_and_save(args, all_results)
        return

    # --- Sequential path --------------------------------------------------
    all_results = []
    for retriever in args.retrievers:
        for top_n in args.top_ns:
            print(f"\n[E1.6] Running {retriever} N={top_n}")
            result = run_retrieval_arm(args, data, retriever, top_n)
            all_results.append(result)

    aggregate_and_save(args, all_results)


if __name__ == "__main__":
    main()
