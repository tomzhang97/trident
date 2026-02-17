#!/usr/bin/env python3
"""E0.2 -- Calibration pool protocol write-up (no GPU required).

Produces a reproducible artifact documenting:
  - Pool size, pos/neg ratio, hard negative mining
  - Refresh policy, per-backbone vs shared mapping
  - One JSON per dataset/backbone + 1 paragraph summary

Usage:
    python -m experiments.rebuttal.e0_2_calibration_protocol \
        --calibration_dir data/calibration \
        --output_dir runs/rebuttal/e0_2
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from experiments.rebuttal.report_utils import ExperimentMetadata, ExperimentReport


def load_calibrator_json(path: str) -> Dict[str, Any]:
    """Load a calibrator JSON and extract protocol-relevant fields."""
    with open(path) as f:
        data = json.load(f)
    return data


def extract_pool_stats(cal_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract pool statistics from calibrator data."""
    conformal = cal_data.get("conformal", {})
    bins_data = conformal.get("bins", {})
    version = cal_data.get("version", "unknown")
    use_mondrian = cal_data.get("use_mondrian", False)

    total_negatives = 0
    total_positives = 0
    bin_details = {}

    for bin_key, bin_info in bins_data.items():
        if isinstance(bin_info, dict):
            negs = bin_info.get("negatives", [])
            n_neg = len(negs) if isinstance(negs, list) else 0
            n_pos = bin_info.get("n_positives", 0)
        elif isinstance(bin_info, list):
            n_neg = len(bin_info)
            n_pos = 0
        else:
            n_neg = 0
            n_pos = 0

        total_negatives += n_neg
        total_positives += n_pos
        bin_details[bin_key] = {
            "n_negatives": n_neg,
            "n_positives": n_pos,
            "neg_score_percentiles": {},
        }
        if isinstance(bin_info, dict) and isinstance(bin_info.get("negatives"), list) and bin_info["negatives"]:
            negs = np.array(bin_info["negatives"])
            bin_details[bin_key]["neg_score_percentiles"] = {
                "p5": round(float(np.percentile(negs, 5)), 4),
                "p25": round(float(np.percentile(negs, 25)), 4),
                "p50": round(float(np.percentile(negs, 50)), 4),
                "p75": round(float(np.percentile(negs, 75)), 4),
                "p95": round(float(np.percentile(negs, 95)), 4),
            }

    pool_size = total_negatives + total_positives
    neg_ratio = total_negatives / max(pool_size, 1)

    return {
        "version": version,
        "use_mondrian": use_mondrian,
        "pool_size": pool_size,
        "total_negatives": total_negatives,
        "total_positives": total_positives,
        "neg_ratio": round(neg_ratio, 4),
        "pos_neg_ratio": f"{total_positives}:{total_negatives}",
        "num_bins": len(bins_data),
        "bin_details": bin_details,
    }


def generate_protocol_paragraph(
    stats_by_key: Dict[str, Dict[str, Any]],
) -> str:
    """Generate a one-paragraph summary of the calibration protocol."""
    if not stats_by_key:
        return (
            "Calibration protocol: No calibrator files were found. "
            "Please provide calibrator JSON files via --calibration_dir."
        )

    first = next(iter(stats_by_key.values()))
    paragraphs = []

    paragraphs.append(
        f"Calibration Protocol Summary: "
        f"The calibration pool is constructed from the training split "
        f"to ensure zero leakage with dev/test evaluation data. "
        f"For each (facet_type, length_bucket, retriever_score_bucket) Mondrian bin, "
        f"we collect negative scores (non-entailing passage-facet pairs). "
    )

    pool_sizes = [s["pool_size"] for s in stats_by_key.values()]
    n_bins = [s["num_bins"] for s in stats_by_key.values()]
    neg_ratios = [s["neg_ratio"] for s in stats_by_key.values()]

    paragraphs.append(
        f"Across {len(stats_by_key)} calibrator file(s), "
        f"pool sizes range from {min(pool_sizes)} to {max(pool_sizes)} "
        f"(negative ratio: {min(neg_ratios):.2f}--{max(neg_ratios):.2f}), "
        f"with {min(n_bins)}--{max(n_bins)} active bins. "
        f"Bins with fewer than n_min=25 negatives are merged upward "
        f"(retriever_score -> length -> facet_type) per the Mondrian merge policy. "
    )

    paragraphs.append(
        "Hard negative mining: negatives are drawn from shortlisted but non-entailing "
        "passage-facet pairs (stage-1 candidates that fail NLI verification). "
        "This ensures the calibration distribution matches the selection-conditional "
        "distribution seen at inference. "
        "Refresh policy: recalibrate when PSI drift exceeds 0.25 or after "
        "N_recal=1000 new scored pairs. "
        "Per-backbone vs shared: by default a single shared calibrator is used; "
        "E1.5 tests per-backbone recalibration to explain cross-architecture discrepancies."
    )

    return " ".join(paragraphs)


def find_calibrator_files(cal_dir: str) -> Dict[str, str]:
    """Find all calibrator JSON files and label them."""
    found = {}
    cdir = Path(cal_dir)
    if not cdir.exists():
        return found
    for p in sorted(cdir.rglob("*.json")):
        # Use relative path as label
        label = str(p.relative_to(cdir)).replace("/", "__").replace(".json", "")
        found[label] = str(p)
    return found


def main():
    parser = argparse.ArgumentParser(description="E0.2: Calibration pool protocol documentation")
    parser.add_argument("--calibration_dir", type=str, default="data/calibration",
                        help="Directory containing calibrator JSON files")
    parser.add_argument("--output_dir", type=str, default="runs/rebuttal/e0_2")
    args = parser.parse_args()

    cal_files = find_calibrator_files(args.calibration_dir)
    stats_by_key = {}

    if cal_files:
        for label, path in sorted(cal_files.items()):
            print(f"[E0.2] Analyzing calibrator: {label}")
            try:
                cal_data = load_calibrator_json(path)
                stats = extract_pool_stats(cal_data)
                stats_by_key[label] = stats
            except Exception as e:
                print(f"  Warning: Could not parse {path}: {e}")
    else:
        print(f"[E0.2] No calibrator files found in {args.calibration_dir}")
        print("[E0.2] Generating protocol template with default values")

    # Generate paragraph summary
    paragraph = generate_protocol_paragraph(stats_by_key)
    print(f"\n[E0.2] Protocol summary:\n{paragraph}\n")

    # Build per-dataset/backbone JSONs
    protocol_artifacts = {}
    for label, stats in stats_by_key.items():
        protocol_artifacts[label] = {
            "pool_size": stats["pool_size"],
            "pos_neg_ratio": stats["pos_neg_ratio"],
            "neg_ratio": stats["neg_ratio"],
            "num_bins": stats["num_bins"],
            "use_mondrian": stats["use_mondrian"],
            "version": stats["version"],
            "hard_negative_mining": "shortlist_non_entailing",
            "refresh_policy": "psi_threshold=0.25 OR n_recal=1000",
            "per_backbone_vs_shared": "shared (default); per-backbone tested in E1.5",
            "merge_policy": "retriever_score -> length -> facet_type (n_min=25)",
            "bin_summary": {
                k: {
                    "n_neg": v["n_negatives"],
                    "n_pos": v["n_positives"],
                    "percentiles": v.get("neg_score_percentiles", {}),
                }
                for k, v in stats["bin_details"].items()
            },
        }

    # If no cal files found, provide template
    if not protocol_artifacts:
        protocol_artifacts["template"] = {
            "pool_size": "TBD",
            "pos_neg_ratio": "TBD",
            "neg_ratio": "TBD",
            "num_bins": "TBD (typical: 45 = 5 facet_types x 3 length x 3 retriever_score)",
            "use_mondrian": True,
            "version": "v2.2",
            "hard_negative_mining": "shortlist_non_entailing",
            "refresh_policy": "psi_threshold=0.25 OR n_recal=1000",
            "per_backbone_vs_shared": "shared (default); per-backbone tested in E1.5",
            "merge_policy": "retriever_score -> length -> facet_type (n_min=25)",
        }

    meta = ExperimentMetadata(
        experiment_id="e0_2_calibration_protocol",
        dataset="all",
        mode="documentation",
    )
    report = ExperimentReport(
        metadata=meta,
        metrics={
            "protocol_artifacts": protocol_artifacts,
            "paragraph_summary": paragraph,
        },
        summary_table=paragraph,
    )
    path = report.save(args.output_dir)
    print(f"[E0.2] Report saved to {path}")
    print("[E0.2] Rebuttal: include protocol_artifacts JSON and paragraph in supplementary material.")


if __name__ == "__main__":
    main()
