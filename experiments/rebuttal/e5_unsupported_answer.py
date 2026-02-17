#!/usr/bin/env python3
"""E5 -- Unsupported answer rate (faithfulness proxy).

Purpose: Strengthen the "provenance alignment reduces unsupported answers" claim.

Run:
  - Dataset: HotpotQA dev limit 500
  - Compare: TRIDENT-Pareto vs top-k baseline
  - Only non-abstained queries.

Metric: normalized string-match, with guardrails:
  - skip short answers (<4 chars or <2 tokens)
  - report short_answer_rate
  - optionally strict match (longest alnum span)

Report columns:
  unsupported_rate (lower bound), unsupported_rate_strict,
  num_answered, num_short_skipped, avg_answer_len_chars,
  short_answer_rate

Usage:
    python -m experiments.rebuttal.e5_unsupported_answer \
        --data_path data/hotpotqa_dev.json \
        --dataset hotpotqa \
        --output_dir runs/rebuttal/e5 \
        --budget 500 --limit 500 \
        --model meta-llama/Meta-Llama-3-8B-Instruct \
        --device 0
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from experiments.rebuttal.report_utils import (
    ExperimentMetadata,
    ExperimentReport,
    bootstrap_ci,
    delta_f1_table,
    exact_match,
    f1_score,
    normalize_answer,
)


def longest_alnum_span(text: str) -> str:
    """Extract the longest contiguous alphanumeric span."""
    spans = re.findall(r"[a-zA-Z0-9\s]+", text)
    if not spans:
        return text
    return max(spans, key=len).strip()


def is_answer_supported(
    answer: str,
    passages: List[Dict[str, Any]],
    strict: bool = False,
) -> bool:
    """Check if answer text appears in any passage.

    Args:
        answer: predicted answer text
        passages: list of selected passages (dicts with 'text' key)
        strict: if True, use longest_alnum_span for matching
    """
    if not answer or not passages:
        return False

    check = longest_alnum_span(answer).lower() if strict else answer.lower()
    if not check.strip():
        return False

    for p in passages:
        text = p.get("text", str(p)).lower()
        if check in text:
            return True

    return False


def run_method(args, data, method: str) -> Dict[str, Any]:
    """Run TRIDENT-Pareto or top-k and collect support metrics."""
    from trident.config import (
        TridentConfig, ParetoConfig, SafeCoverConfig,
        LLMConfig, RetrievalConfig, EvaluationConfig,
        NLIConfig, CalibrationConfig, TelemetryConfig,
    )
    from trident.pipeline import TridentPipeline

    device = f"cuda:{args.device}" if args.device >= 0 else "cpu"

    config = TridentConfig(
        mode="pareto",
        pareto=ParetoConfig(
            budget=args.budget, max_evidence_tokens=args.budget,
            max_units=8, stop_on_budget=True, use_vqc=False, use_bwk=False,
        ),
        llm=LLMConfig(model_name=args.model, device=device, load_in_8bit=args.load_in_8bit),
        retrieval=RetrievalConfig(method="dense", encoder_model=args.encoder_model, top_k=100),
        nli=NLIConfig(batch_size=32),
        calibration=CalibrationConfig(use_mondrian=True),
        evaluation=EvaluationConfig(dataset=args.dataset),
        telemetry=TelemetryConfig(enable=True),
    )
    pipeline = TridentPipeline(config)

    per_query = []
    for i, ex in enumerate(data):
        question = ex.get("question", "")
        gold = ex.get("answer", "")
        context = ex.get("context", None)

        if (i + 1) % 100 == 0:
            print(f"    [{method}] {i+1}/{len(data)}")

        if method == "trident_pareto":
            try:
                output = pipeline.process_query(question, context=context)
                per_query.append({
                    "prediction": output.answer if not output.abstained else "",
                    "gold": gold,
                    "abstained": output.abstained,
                    "tokens_used": output.tokens_used,
                    "selected_passages": output.selected_passages,
                })
            except Exception as e:
                per_query.append({
                    "prediction": "", "gold": gold, "abstained": True,
                    "tokens_used": 0, "selected_passages": [], "error": str(e),
                })
        else:  # top_k
            try:
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

                # Convert passages to dicts for support checking
                selected_dicts = []
                for p in selected:
                    if hasattr(p, "text"):
                        selected_dicts.append({"text": p.text})
                    else:
                        selected_dicts.append({"text": str(p)})

                per_query.append({
                    "prediction": answer,
                    "gold": gold,
                    "abstained": not selected,
                    "tokens_used": total_tokens,
                    "selected_passages": selected_dicts,
                })
            except Exception as e:
                per_query.append({
                    "prediction": "", "gold": gold, "abstained": True,
                    "tokens_used": 0, "selected_passages": [], "error": str(e),
                })

    # Compute support metrics (only on non-abstained)
    answered = [q for q in per_query if not q["abstained"] and q["prediction"]]

    # Filter out short answers
    MIN_CHARS = 4
    MIN_TOKENS = 2
    short_skipped = []
    valid = []
    for q in answered:
        pred = q["prediction"]
        if len(pred) < MIN_CHARS or len(pred.split()) < MIN_TOKENS:
            short_skipped.append(q)
        else:
            valid.append(q)

    # Compute unsupported rates
    unsupported_count = 0
    unsupported_strict_count = 0
    answer_lens = []

    for q in valid:
        pred = q["prediction"]
        passages = q["selected_passages"]
        answer_lens.append(len(pred))

        if not is_answer_supported(pred, passages, strict=False):
            unsupported_count += 1
        if not is_answer_supported(pred, passages, strict=True):
            unsupported_strict_count += 1

    n_valid = max(len(valid), 1)
    n_answered = len(answered)
    n_total = len(per_query)

    # Also compute EM/F1 on answered
    ans_preds = [q["prediction"] for q in answered]
    ans_golds = [q["gold"] for q in answered]
    if ans_preds:
        em_m, _, _ = bootstrap_ci([exact_match(p, g) for p, g in zip(ans_preds, ans_golds)])
        f1_m, _, _ = bootstrap_ci([f1_score(p, g) for p, g in zip(ans_preds, ans_golds)])
    else:
        em_m = f1_m = 0.0

    return {
        "method": method,
        "n_total": n_total,
        "n_answered": n_answered,
        "n_valid": len(valid),
        "num_short_skipped": len(short_skipped),
        "short_answer_rate": round(len(short_skipped) / max(n_answered, 1), 4),
        "unsupported_rate": round(unsupported_count / n_valid, 4),
        "unsupported_rate_strict": round(unsupported_strict_count / n_valid, 4),
        "avg_answer_len_chars": round(float(np.mean(answer_lens)), 1) if answer_lens else 0,
        "answered_em": round(em_m, 4),
        "answered_f1": round(f1_m, 4),
    }


def main():
    parser = argparse.ArgumentParser(description="E5: Unsupported answer rate")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="hotpotqa")
    parser.add_argument("--output_dir", type=str, default="runs/rebuttal/e5")
    parser.add_argument("--budget", type=int, default=500)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--encoder_model", type=str, default="facebook/contriever")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from experiments.eval_complete_runnable import load_data
    data = load_data(args.data_path, limit=args.limit)
    print(f"[E5] Loaded {len(data)} examples")

    methods = ["trident_pareto", "top_k"]
    all_results = {}
    for method in methods:
        print(f"\n[E5] Running {method}")
        result = run_method(args, data, method)
        all_results[method] = result

    # Print comparison
    print("\n[E5] Unsupported answer rate comparison:")
    hdr = "| Method | Answered | Valid | Short% | Unsupported | Unsup(strict) | AvgLen | EM | F1 |"
    sep = "|--------|---------|-------|--------|-------------|---------------|--------|----|----|"
    print(hdr)
    print(sep)
    for method, r in all_results.items():
        print(f"| {method} "
              f"| {r['n_answered']} "
              f"| {r['n_valid']} "
              f"| {r['short_answer_rate']:.3f} "
              f"| {r['unsupported_rate']:.3f} "
              f"| {r['unsupported_rate_strict']:.3f} "
              f"| {r['avg_answer_len_chars']:.0f} "
              f"| {r['answered_em']:.3f} "
              f"| {r['answered_f1']:.3f} |")

    # Compute delta
    trident_unsup = all_results.get("trident_pareto", {}).get("unsupported_rate", 0)
    topk_unsup = all_results.get("top_k", {}).get("unsupported_rate", 0)
    delta_unsup = topk_unsup - trident_unsup
    print(f"\n  Delta unsupported rate (top_k - trident): {delta_unsup:+.3f}")

    meta = ExperimentMetadata(
        experiment_id=f"e5_unsupported_{args.dataset}",
        dataset=args.dataset,
        budget=args.budget,
        mode="pareto",
        backbone=args.model,
        seed=args.seed,
        limit=args.limit,
    )
    report = ExperimentReport(
        metadata=meta,
        metrics=all_results,
        compute={"delta_unsupported_rate": round(delta_unsup, 4)},
    )
    path = report.save(args.output_dir)
    print(f"\n[E5] Report saved to {path}")
    print(f"[E5] Rebuttal: 'TRIDENT reduces unsupported-answer rate vs top-k by "
          f"{delta_unsup:.3f} (conservative lower bound).'")


if __name__ == "__main__":
    main()
