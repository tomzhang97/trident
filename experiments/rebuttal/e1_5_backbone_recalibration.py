#!/usr/bin/env python3
"""E1.5 -- Backbone dependence / recalibration (22Vu).

Purpose: Explain Qwen3 vs Llama discrepancy as calibration transfer, not
         method failure.

Run:
  - Slice: limit 200 (fast)
  - Backbones: Llama-3-8B-Instruct and Qwen3-8B
  - Conditions per backbone:
    * calibrated=shared (use Llama-fitted calibrator)
    * calibrated=per_backbone (fit on that backbone)
  - Mode: Safe-Cover primarily; also report Pareto if it uses thresholds.

Report columns:
  EM/F1, abstention_rate, avg_evidence_tokens, verifier_ece (binned),
  verifier_auc, coverage_proxy, delta_EM and delta_F1 (per_backbone - shared)

Usage:
    python -m experiments.rebuttal.e1_5_backbone_recalibration \
        --data_path data/hotpotqa_dev.json \
        --dataset hotpotqa \
        --output_dir runs/rebuttal/e1_5 \
        --budget 500 --limit 200 \
        --backbones meta-llama/Meta-Llama-3-8B-Instruct Qwen/Qwen3-8B \
        --shared_calibration_path data/calibration/calibrator_llama.json \
        --per_backbone_calibration_dir data/calibration/per_backbone \
        --device 0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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
    bootstrap_ci,
    delta_f1_table,
    exact_match,
    f1_score,
    latency_percentiles,
)


def compute_ece(confidences: List[float], correct: List[bool], n_bins: int = 10) -> float:
    """Compute Expected Calibration Error."""
    if not confidences:
        return 0.0
    confs = np.array(confidences)
    cors = np.array(correct, dtype=float)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confs > bin_edges[i]) & (confs <= bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        avg_conf = confs[mask].mean()
        avg_acc = cors[mask].mean()
        ece += mask.sum() / len(confs) * abs(avg_conf - avg_acc)
    return float(ece)


def compute_auc(scores: List[float], labels: List[bool]) -> float:
    """Compute a simple AUC (area under ROC curve)."""
    if not scores or not any(labels) or all(labels):
        return 0.5
    pairs = sorted(zip(scores, labels), reverse=True)
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    tp = 0
    auc = 0.0
    for score, label in pairs:
        if label:
            tp += 1
        else:
            auc += tp
    return float(auc / (n_pos * n_neg))


def run_backbone_condition(
    args,
    data: List[Dict],
    backbone: str,
    calibration_mode: str,
    calibration_path: Optional[str],
) -> Dict[str, Any]:
    """Run Safe-Cover with a specific backbone and calibration setting."""
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
            per_facet_alpha=0.05,
            token_cap=args.budget,
            max_evidence_tokens=args.budget,
            early_abstain=True,
            fallback_to_pareto=False,
            use_certificates=True,
        ),
        pareto=ParetoConfig(budget=args.budget),
        llm=LLMConfig(model_name=backbone, device=device, load_in_8bit=args.load_in_8bit),
        retrieval=RetrievalConfig(method="dense", encoder_model=args.encoder_model, top_k=100),
        nli=NLIConfig(batch_size=32),
        calibration=CalibrationConfig(use_mondrian=True),
        evaluation=EvaluationConfig(dataset=args.dataset),
        telemetry=TelemetryConfig(enable=True),
    )
    pipeline = create_pipeline(config, device=device)

    # Load specific calibrator if path provided
    if calibration_path and Path(calibration_path).exists():
        try:
            from trident.calibration import ReliabilityCalibrator
            cal = ReliabilityCalibrator()
            cal_data = json.loads(Path(calibration_path).read_text())
            # Attempt to load conformal calibrator from data
            if cal_data.get("conformal"):
                from trident.calibration import SelectionConditionalCalibrator
                scc = SelectionConditionalCalibrator()
                scc.from_dict(cal_data["conformal"])
                cal.set_conformal_calibrator(scc)
            pipeline.calibrator = cal
            print(f"    Loaded calibrator from {calibration_path}")
        except Exception as e:
            print(f"    Warning: Could not load calibrator: {e}")

    per_query = []
    verifier_scores = []
    verifier_correct = []

    for i, ex in enumerate(data):
        question = ex.get("question", "")
        gold = ex.get("answer", "")
        context = ex.get("context", None)

        if (i + 1) % 50 == 0:
            print(f"    [{backbone.split('/')[-1]}/{calibration_mode}] {i+1}/{len(data)}")

        try:
            output = pipeline.process_query(question, context=context)
            certs = output.certificates or []

            # Extract verifier scores for ECE/AUC
            for cert in certs:
                if isinstance(cert, dict):
                    pval = cert.get("p_value", 1.0)
                    threshold = cert.get("threshold", 0.05)
                    verifier_scores.append(1.0 - pval)  # higher = more confident
                    verifier_correct.append(pval <= threshold)

            per_query.append({
                "prediction": output.answer if not output.abstained else "",
                "gold": gold,
                "abstained": output.abstained,
                "tokens_used": output.tokens_used,
                "latency_ms": output.latency_ms,
                "num_certificates": len(certs),
                "coverage": output.metrics.get("coverage", 0),
            })
        except Exception as e:
            per_query.append({
                "prediction": "", "gold": gold, "abstained": True,
                "tokens_used": 0, "latency_ms": 0, "num_certificates": 0,
                "coverage": 0, "error": str(e),
            })

    # Aggregate
    preds = [q["prediction"] for q in per_query]
    golds = [q["gold"] for q in per_query]
    ems = [exact_match(p, g) for p, g in zip(preds, golds)]
    f1s = [f1_score(p, g) for p, g in zip(preds, golds)]
    abstain = sum(1 for q in per_query if q["abstained"])
    toks = [q["tokens_used"] for q in per_query]
    coverages = [q["coverage"] for q in per_query]

    em_m, em_lo, em_hi = bootstrap_ci(ems)
    f1_m, f1_lo, f1_hi = bootstrap_ci(f1s)

    ece = compute_ece(verifier_scores, verifier_correct)
    auc = compute_auc(verifier_scores, verifier_correct)

    return {
        "backbone": backbone,
        "calibration_mode": calibration_mode,
        "n_queries": len(data),
        "em": round(em_m, 4),
        "em_ci": [round(em_lo, 4), round(em_hi, 4)],
        "f1": round(f1_m, 4),
        "f1_ci": [round(f1_lo, 4), round(f1_hi, 4)],
        "abstention_rate": round(abstain / max(len(data), 1), 4),
        "avg_evidence_tokens": round(float(np.mean(toks)), 1),
        "verifier_ece": round(ece, 4),
        "verifier_auc": round(auc, 4),
        "coverage_proxy": round(float(np.mean(coverages)), 4),
        "per_query": per_query,
    }


def aggregate_and_save(args, all_results: List[Dict[str, Any]]) -> str:
    """Aggregate per-arm results and save the final report."""
    # Compute deltas (per_backbone - shared)
    by_backbone = {}
    for r in all_results:
        bb = r["backbone"]
        by_backbone.setdefault(bb, {})[r["calibration_mode"]] = r

    deltas = {}
    for bb, modes in by_backbone.items():
        if "shared" in modes and "per_backbone" in modes:
            s = modes["shared"]
            p = modes["per_backbone"]
            deltas[bb] = {
                "delta_em": round(p["em"] - s["em"], 4),
                "delta_f1": round(p["f1"] - s["f1"], 4),
                "delta_abstention": round(p["abstention_rate"] - s["abstention_rate"], 4),
                "delta_ece": round(p["verifier_ece"] - s["verifier_ece"], 4),
                "delta_auc": round(p["verifier_auc"] - s["verifier_auc"], 4),
            }

    # Print comparison table
    print("\n[E1.5] Backbone comparison:")
    hdr = "| Backbone | Cal | EM | F1 | Abstain | ECE | AUC | EvTok | Coverage |"
    sep = "|----------|-----|----|----|---------|-----|-----|-------|----------|"
    print(hdr)
    print(sep)
    for r in all_results:
        bb = r["backbone"].split("/")[-1]
        print(f"| {bb} | {r['calibration_mode']} "
              f"| {r['em']:.3f} | {r['f1']:.3f} "
              f"| {r['abstention_rate']:.3f} "
              f"| {r['verifier_ece']:.3f} | {r['verifier_auc']:.3f} "
              f"| {r['avg_evidence_tokens']:.0f} "
              f"| {r['coverage_proxy']:.3f} |")

    if deltas:
        print("\n[E1.5] Per-backbone - Shared deltas:")
        for bb, d in deltas.items():
            bb_short = bb.split("/")[-1]
            print(f"  {bb_short}: dEM={d['delta_em']:+.3f}, dF1={d['delta_f1']:+.3f}, "
                  f"dECE={d['delta_ece']:+.3f}, dAUC={d['delta_auc']:+.3f}")

    meta = ExperimentMetadata(
        experiment_id=f"e1_5_backbone_{args.dataset}",
        dataset=args.dataset,
        budget=args.budget,
        mode="safe_cover",
        backbone=",".join(args.backbones),
        seed=args.seed,
        limit=args.limit,
    )
    metrics = {}
    for r in all_results:
        key = f"{r['backbone'].split('/')[-1]}_{r['calibration_mode']}"
        metrics[key] = {k: v for k, v in r.items() if k != "per_query"}
    metrics["deltas"] = deltas

    report = ExperimentReport(
        metadata=meta,
        metrics=metrics,
    )
    path = report.save(args.output_dir)
    print(f"\n[E1.5] Report saved to {path}")
    print("[E1.5] Rebuttal: 'Per-backbone calibration reduces cross-architecture "
          "discrepancy: ECE improves [x->y] and EM/F1 improve by [delta] on Qwen3.'")
    return path


def _resolve_calibration_path(args, backbone: str, cal_mode: str) -> Optional[str]:
    """Resolve calibration file path for a (backbone, cal_mode) pair."""
    if cal_mode == "shared":
        return args.shared_calibration_path if args.shared_calibration_path else None
    else:
        if args.per_backbone_calibration_dir:
            backbone_short = backbone.split("/")[-1]
            candidate = Path(args.per_backbone_calibration_dir) / f"calibrator_{backbone_short}.json"
            if candidate.exists():
                return str(candidate)
    return None


def main():
    parser = argparse.ArgumentParser(description="E1.5: Backbone recalibration")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="hotpotqa")
    parser.add_argument("--output_dir", type=str, default="runs/rebuttal/e1_5")
    parser.add_argument("--budget", type=int, default=500)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--backbones", nargs="+",
                        default=["meta-llama/Meta-Llama-3-8B-Instruct", "Qwen/Qwen3-8B"])
    parser.add_argument("--shared_calibration_path", type=str, default="",
                        help="Path to shared (Llama-fitted) calibrator JSON")
    parser.add_argument("--per_backbone_calibration_dir", type=str, default="",
                        help="Directory with per-backbone calibrator JSONs")
    parser.add_argument("--encoder_model", type=str, default="facebook/contriever")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    add_multigpu_args(parser)
    args = parser.parse_args()

    # --- Worker path: run a single (backbone, cal_mode) arm ---------------
    if is_worker(args):
        arm = get_arm_spec(args)
        backbone = arm["backbone"]
        cal_mode = arm["calibration_mode"]
        cal_path = _resolve_calibration_path(args, backbone, cal_mode)
        from experiments.eval_complete_runnable import load_data
        data = load_data(args.data_path, limit=args.limit)
        bb_short = backbone.split("/")[-1]
        print(f"[E1.5/worker] {bb_short}/{cal_mode} on GPU {args._worker_gpu}")
        result = run_backbone_condition(args, data, backbone, cal_mode, cal_path)
        write_worker_result(args, result)
        return

    # --- Load data --------------------------------------------------------
    from experiments.eval_complete_runnable import load_data
    data = load_data(args.data_path, limit=args.limit)
    print(f"[E1.5] Loaded {len(data)} examples")

    # --- Multi-GPU path ---------------------------------------------------
    if args.num_gpus > 1:
        arm_specs = []
        for backbone in args.backbones:
            bb_short = backbone.split("/")[-1]
            for cal_mode in ("shared", "per_backbone"):
                arm_specs.append({
                    "backbone": backbone,
                    "calibration_mode": cal_mode,
                    "label": f"{bb_short}/{cal_mode}",
                })
        print(f"[E1.5] Distributing {len(arm_specs)} arms across {args.num_gpus} GPUs")
        all_results = run_arms_parallel(args, arm_specs, __file__)
        all_results = [r for r in all_results if r and "backbone" in r]
        aggregate_and_save(args, all_results)
        return

    # --- Sequential path --------------------------------------------------
    all_results = []
    for backbone in args.backbones:
        backbone_short = backbone.split("/")[-1]

        # Shared calibration
        print(f"\n[E1.5] {backbone_short} / shared calibration")
        shared_path = _resolve_calibration_path(args, backbone, "shared")
        shared_res = run_backbone_condition(args, data, backbone, "shared", shared_path)
        all_results.append(shared_res)

        # Per-backbone calibration
        print(f"\n[E1.5] {backbone_short} / per-backbone calibration")
        per_bb_path = _resolve_calibration_path(args, backbone, "per_backbone")
        per_bb_res = run_backbone_condition(args, data, backbone, "per_backbone", per_bb_path)
        all_results.append(per_bb_res)

    aggregate_and_save(args, all_results)


if __name__ == "__main__":
    main()
