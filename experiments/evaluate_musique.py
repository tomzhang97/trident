#!/usr/bin/env python3
"""
MuSiQue evaluation wrapper for TRIDENT results.

This is a thin wrapper around musique/evaluate_v1.0.py that provides
a convenient interface for evaluating TRIDENT experiment results.

Usage:
    # Evaluate a results directory
    python experiments/evaluate_musique.py --results_dir runs/musique/results/

    # Evaluate a single results.json file
    python experiments/evaluate_musique.py --results_file results/results.json

    # Specify ground truth split
    python experiments/evaluate_musique.py --results_dir runs/musique/results/ --split full_dev
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from musique.evaluate_v1 import evaluate_trident, evaluate


# Ground truth file mapping
SPLIT_TO_FILE = {
    'ans_dev': 'musique_ans_v1.0_dev.jsonl',
    'ans_test': 'musique_ans_v1.0_test.jsonl',
    'full_dev': 'musique_full_v1.0_dev.jsonl',
    'full_test': 'musique_full_v1.0_test.jsonl',
}


def get_ground_truth_path(split: str) -> str:
    """Get path to ground truth file for a split."""
    if split not in SPLIT_TO_FILE:
        raise ValueError(f"Unknown split: {split}. Available: {list(SPLIT_TO_FILE.keys())}")

    gt_path = Path(__file__).parent.parent / "musique" / "data" / SPLIT_TO_FILE[split]
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")

    return str(gt_path)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate TRIDENT results on MuSiQue",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate TRIDENT results directory
  python experiments/evaluate_musique.py --results_dir runs/musique/results/

  # Evaluate a single results.json file
  python experiments/evaluate_musique.py --results_file results/results.json

  # Use a specific ground truth split
  python experiments/evaluate_musique.py --results_dir runs/musique/results/ --split full_dev

  # Save converted predictions for inspection
  python experiments/evaluate_musique.py --results_dir runs/musique/results/ \\
      --output_predictions predictions.jsonl
        """
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--results_dir",
        type=str,
        help="Directory containing TRIDENT experiment results"
    )
    input_group.add_argument(
        "--results_file",
        type=str,
        help="Single results.json file to evaluate"
    )
    input_group.add_argument(
        "--predictions",
        type=str,
        help="MuSiQue format predictions.jsonl file (for standard evaluation)"
    )

    # Ground truth options
    parser.add_argument(
        "--split",
        type=str,
        default="ans_dev",
        choices=list(SPLIT_TO_FILE.keys()),
        help="MuSiQue data split for ground truth"
    )
    parser.add_argument(
        "--ground_truth",
        type=str,
        help="Path to ground truth file (overrides --split)"
    )

    # Output options
    parser.add_argument(
        "--output_filepath",
        type=str,
        help="Path to save metrics JSON"
    )
    parser.add_argument(
        "--output_predictions",
        type=str,
        help="Path to save converted predictions"
    )

    args = parser.parse_args()

    # Determine ground truth path
    if args.ground_truth:
        gt_path = args.ground_truth
    else:
        gt_path = get_ground_truth_path(args.split)

    print(f"Ground truth: {gt_path}")

    # Run evaluation
    if args.predictions:
        # Standard MuSiQue evaluation
        print(f"Evaluating predictions: {args.predictions}")
        metrics = evaluate(args.predictions, gt_path, lenient=True)
    else:
        # TRIDENT evaluation
        trident_path = args.results_dir or args.results_file
        print(f"Evaluating TRIDENT results: {trident_path}")
        metrics = evaluate_trident(
            trident_path,
            gt_path,
            output_predictions=args.output_predictions,
            lenient=True
        )

    # Print results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    for metric, value in metrics.items():
        print(f"  {metric}: {value}")

    # Save metrics if requested
    if args.output_filepath:
        with open(args.output_filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to: {args.output_filepath}")


if __name__ == "__main__":
    main()
