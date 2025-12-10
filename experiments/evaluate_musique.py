#!/usr/bin/env python3
"""
MuSiQue evaluation wrapper for TRIDENT results.

This script provides an easy way to evaluate TRIDENT experiment results
against MuSiQue ground truth, supporting both the official MuSiQue metrics
and TRIDENT's internal metrics.

Usage:
    # Evaluate a results directory
    python experiments/evaluate_musique.py --results_dir runs/musique/safe_cover_ans_dev_xxx/

    # Evaluate a single results.json file
    python experiments/evaluate_musique.py --results_file results/results.json --split ans_dev

    # Convert and evaluate predictions.jsonl
    python experiments/evaluate_musique.py --predictions predictions.jsonl --split ans_dev
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_trident_results(results_path: str) -> List[Dict[str, Any]]:
    """Load TRIDENT results from a JSON file."""
    with open(results_path, 'r') as f:
        data = json.load(f)

    if isinstance(data, dict) and 'results' in data:
        return data['results']
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"Unexpected results format in {results_path}")


def aggregate_results_from_dir(results_dir: str) -> List[Dict[str, Any]]:
    """Aggregate results from multiple shard directories."""
    all_results = []

    results_path = Path(results_dir)

    # Check for direct results.json
    if (results_path / "results.json").exists():
        return load_trident_results(str(results_path / "results.json"))

    # Check for shards in results/ subdirectory
    shard_dirs = list((results_path / "results").glob("*"))
    if not shard_dirs:
        shard_dirs = list(results_path.glob("*"))

    for shard_dir in shard_dirs:
        results_file = shard_dir / "results.json"
        if results_file.exists():
            results = load_trident_results(str(results_file))
            all_results.extend(results)

    return all_results


def convert_to_musique_format(
    results: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Convert TRIDENT results to MuSiQue prediction format."""
    predictions = []

    for result in results:
        # Extract the original MuSiQue ID from the query_id
        query_id = result.get('query_id', '')

        # Handle different ID formats
        # TRIDENT format: "musique_ans_dev_0" or direct MuSiQue ID
        if query_id.startswith('musique_'):
            # Try to extract original ID from the indexed format
            parts = query_id.split('_')
            # This is a generated ID, we need to use a different approach
            musique_id = query_id
        else:
            musique_id = query_id

        # Get predicted answer
        predicted_answer = result.get('prediction', '')
        if not predicted_answer:
            predicted_answer = result.get('answer', '')

        # Get predicted support indices
        predicted_support = []
        passages = result.get('selected_passages', [])
        for p in passages:
            if 'idx' in p:
                predicted_support.append(p['idx'])
            elif 'paragraph_idx' in p:
                predicted_support.append(p['paragraph_idx'])

        # Determine answerability (not abstained and has answer)
        predicted_answerable = (
            not result.get('abstained', False) and
            bool(predicted_answer.strip())
        )

        predictions.append({
            'id': musique_id,
            'predicted_answer': predicted_answer,
            'predicted_support_idxs': predicted_support,
            'predicted_answerable': predicted_answerable
        })

    return predictions


def compute_internal_metrics(
    results: List[Dict[str, Any]]
) -> Dict[str, float]:
    """Compute TRIDENT internal metrics."""
    import re
    import string
    from collections import Counter

    def normalize(text: str) -> str:
        """Normalize text for comparison."""
        text = text.lower().strip()
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r'\b(a|an|the)\b', ' ', text)
        return ' '.join(text.split())

    def f1_score(pred: str, gt: str) -> float:
        """Compute F1 score."""
        pred_tokens = normalize(pred).split()
        gt_tokens = normalize(gt).split()

        if not pred_tokens or not gt_tokens:
            return 0.0

        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_same = sum(common.values())

        if num_same == 0:
            return 0.0

        precision = num_same / len(pred_tokens)
        recall = num_same / len(gt_tokens)

        return (2 * precision * recall) / (precision + recall)

    em_scores = []
    f1_scores = []
    abstained = 0
    tokens_used = []
    latencies = []

    for result in results:
        if result.get('abstained') or 'error' in result:
            abstained += 1
            continue

        pred = result.get('prediction', '')
        gts = result.get('ground_truth', [])

        if isinstance(gts, str):
            gts = [gts]

        if pred and gts:
            # Compute best EM and F1 across all ground truths
            best_em = max(
                1.0 if normalize(pred) == normalize(gt) else 0.0
                for gt in gts
            )
            best_f1 = max(f1_score(pred, gt) for gt in gts)

            em_scores.append(best_em)
            f1_scores.append(best_f1)

        if result.get('tokens_used'):
            tokens_used.append(result['tokens_used'])
        if result.get('latency_ms'):
            latencies.append(result['latency_ms'])

    return {
        'total': len(results),
        'valid': len(results) - abstained,
        'abstained': abstained,
        'abstention_rate': abstained / len(results) if results else 0,
        'exact_match': sum(em_scores) / len(em_scores) if em_scores else 0,
        'f1_score': sum(f1_scores) / len(f1_scores) if f1_scores else 0,
        'avg_tokens': sum(tokens_used) / len(tokens_used) if tokens_used else 0,
        'avg_latency_ms': sum(latencies) / len(latencies) if latencies else 0
    }


def run_official_evaluation(
    predictions: List[Dict[str, Any]],
    ground_truth_path: str,
    output_dir: Optional[str] = None
) -> Dict[str, float]:
    """Run official MuSiQue evaluation."""
    # Save predictions to temp file
    import tempfile

    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for pred in predictions:
            f.write(json.dumps(pred) + '\n')
        predictions_path = f.name

    try:
        # Import MuSiQue evaluation
        sys.path.insert(0, str(Path(__file__).parent.parent / "musique"))
        from evaluate_v1 import evaluate

        metrics = evaluate(predictions_path, ground_truth_path)

        # Save predictions if output_dir provided
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Copy predictions
            import shutil
            shutil.copy(predictions_path, output_path / "predictions.jsonl")

            # Save metrics
            with open(output_path / "official_metrics.json", 'w') as f:
                json.dump(metrics, f, indent=2)

        return metrics

    except Exception as e:
        print(f"Warning: Official evaluation failed: {e}")
        return {}

    finally:
        # Clean up temp file
        import os
        os.unlink(predictions_path)


def print_metrics(
    internal_metrics: Dict[str, float],
    official_metrics: Dict[str, float],
    title: str = "Evaluation Results"
) -> None:
    """Print formatted metrics."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

    print("\n📊 TRIDENT Internal Metrics:")
    print(f"   Total examples: {internal_metrics['total']}")
    print(f"   Valid (non-abstained): {internal_metrics['valid']}")
    print(f"   Abstained: {internal_metrics['abstained']}")
    print(f"   Abstention rate: {internal_metrics['abstention_rate']:.4f}")
    print(f"   Exact Match: {internal_metrics['exact_match']:.4f}")
    print(f"   F1 Score: {internal_metrics['f1_score']:.4f}")
    if internal_metrics['avg_tokens'] > 0:
        print(f"   Avg Tokens: {internal_metrics['avg_tokens']:.1f}")
    if internal_metrics['avg_latency_ms'] > 0:
        print(f"   Avg Latency: {internal_metrics['avg_latency_ms']:.1f}ms")

    if official_metrics:
        print("\n📊 Official MuSiQue Metrics:")
        for metric, value in official_metrics.items():
            print(f"   {metric}: {value}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate TRIDENT results on MuSiQue",
        formatter_class=argparse.RawDescriptionHelpFormatter
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
        help="MuSiQue format predictions.jsonl file"
    )

    # Ground truth options
    parser.add_argument(
        "--split",
        type=str,
        default="ans_dev",
        choices=["ans_dev", "ans_test", "full_dev", "full_test"],
        help="MuSiQue data split for ground truth"
    )
    parser.add_argument(
        "--ground_truth",
        type=str,
        help="Path to ground truth file (overrides --split)"
    )

    # Output options
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--skip_official",
        action="store_true",
        help="Skip official MuSiQue evaluation"
    )

    args = parser.parse_args()

    # Determine ground truth path
    if args.ground_truth:
        gt_path = args.ground_truth
    else:
        split_to_file = {
            'ans_dev': 'musique_ans_v1.0_dev.jsonl',
            'ans_test': 'musique_ans_v1.0_test.jsonl',
            'full_dev': 'musique_full_v1.0_dev.jsonl',
            'full_test': 'musique_full_v1.0_test.jsonl'
        }
        gt_path = str(
            Path(__file__).parent.parent / "musique" / "data" / split_to_file[args.split]
        )

    print(f"Ground truth: {gt_path}")

    # Load results
    if args.results_dir:
        print(f"Loading results from directory: {args.results_dir}")
        results = aggregate_results_from_dir(args.results_dir)
    elif args.results_file:
        print(f"Loading results from file: {args.results_file}")
        results = load_trident_results(args.results_file)
    else:
        # Load predictions directly
        print(f"Loading predictions from: {args.predictions}")
        with open(args.predictions, 'r') as f:
            predictions = [json.loads(line) for line in f if line.strip()]
        results = None

    print(f"Loaded {len(results) if results else len(predictions)} examples")

    # Compute internal metrics if we have TRIDENT results
    internal_metrics = {}
    if results:
        internal_metrics = compute_internal_metrics(results)

        # Convert to MuSiQue format
        predictions = convert_to_musique_format(results)
    else:
        internal_metrics = {
            'total': len(predictions),
            'valid': len(predictions),
            'abstained': 0,
            'abstention_rate': 0,
            'exact_match': 0,
            'f1_score': 0,
            'avg_tokens': 0,
            'avg_latency_ms': 0
        }

    # Run official evaluation
    official_metrics = {}
    if not args.skip_official and Path(gt_path).exists():
        print("Running official MuSiQue evaluation...")
        official_metrics = run_official_evaluation(
            predictions,
            gt_path,
            args.output_dir
        )

    # Print results
    print_metrics(internal_metrics, official_metrics)

    # Save combined results
    if args.output_dir:
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        combined = {
            'internal_metrics': internal_metrics,
            'official_metrics': official_metrics
        }

        with open(output_path / "combined_metrics.json", 'w') as f:
            json.dump(combined, f, indent=2)

        print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
