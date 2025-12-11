#!/usr/bin/env python3
"""
End-to-end MuSiQue experiment runner for TRIDENT.

This script handles the complete workflow:
1. Load and prepare MuSiQue data
2. Run TRIDENT or baseline systems
3. Evaluate results using MuSiQue official metrics
4. Generate comprehensive reports

Usage:
    # Run TRIDENT Safe-Cover on MuSiQue dev set
    python experiments/run_musique_experiment.py --mode safe_cover --split ans_dev

    # Run Self-RAG baseline
    python experiments/run_musique_experiment.py --mode self_rag --split ans_dev

    # Run with limited examples for testing
    python experiments/run_musique_experiment.py --mode safe_cover --split ans_dev --limit 100

    # Run on full dataset with answerability
    python experiments/run_musique_experiment.py --mode safe_cover --split full_dev
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from musique.data_loader import MuSiQueDataLoader


def setup_output_dirs(base_dir: str, mode: str, split: str) -> Dict[str, Path]:
    """Create output directory structure."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{mode}_{split}_{timestamp}"

    run_dir = Path(base_dir) / run_name
    shards_dir = run_dir / "shards"
    results_dir = run_dir / "results"
    eval_dir = run_dir / "evaluation"

    for d in [run_dir, shards_dir, results_dir, eval_dir]:
        d.mkdir(parents=True, exist_ok=True)

    return {
        'run_dir': run_dir,
        'shards_dir': shards_dir,
        'results_dir': results_dir,
        'eval_dir': eval_dir,
        'run_name': run_name
    }


def prepare_data(
    split: str,
    shards_dir: Path,
    shard_size: int = 100,
    limit: Optional[int] = None
) -> Dict[str, Any]:
    """Load and prepare MuSiQue data."""
    print(f"\n{'='*60}")
    print("STEP 1: Loading and preparing MuSiQue data")
    print(f"{'='*60}")

    loader = MuSiQueDataLoader()

    print(f"Loading split: {split}")
    if limit:
        print(f"Limiting to {limit} examples")

    examples = loader.load_and_convert(split, limit)
    stats = loader.get_stats(examples)

    print(f"\nDataset statistics:")
    print(f"  Total examples: {stats['total_examples']}")
    print(f"  Answerable: {stats['answerable']}")
    print(f"  Unanswerable: {stats['unanswerable']}")
    print(f"  Hop types: {stats['hop_types']}")

    print(f"\nCreating shards (size={shard_size})...")
    shard_paths = loader.save_shards(examples, str(shards_dir), shard_size, split)

    print(f"Created {len(shard_paths)} shards")

    return {
        'examples': examples,
        'stats': stats,
        'shard_paths': shard_paths
    }


def run_experiment(
    shard_paths: List[str],
    results_dir: Path,
    mode: str,
    config_path: str,
    model: str,
    device: int,
    num_gpus: int = 1,
    additional_args: Optional[Dict] = None
) -> List[str]:
    """Run TRIDENT/baseline on all shards."""
    print(f"\n{'='*60}")
    print("STEP 2: Running experiments")
    print(f"{'='*60}")

    print(f"Mode: {mode}")
    print(f"Config: {config_path}")
    print(f"Model: {model}")
    print(f"GPUs: {num_gpus}")

    result_paths = []

    for i, shard_path in enumerate(shard_paths):
        shard_name = Path(shard_path).stem
        output_dir = results_dir / shard_name
        gpu_id = i % num_gpus if device < 0 else device

        print(f"\nProcessing shard {i+1}/{len(shard_paths)}: {shard_name}")
        print(f"  GPU: {gpu_id}")

        cmd = [
            "python", "experiments/eval_complete_runnable.py",
            "--worker",
            "--data_path", shard_path,
            "--output_dir", str(output_dir),
            "--mode", mode,
            "--config", config_path,
            "--model", model,
            "--device", str(gpu_id),
            "--dataset", "musique"
        ]

        # Add additional arguments
        if additional_args:
            for key, value in additional_args.items():
                if value is not None:
                    if isinstance(value, bool):
                        if value:
                            cmd.append(f"--{key}")
                    else:
                        cmd.extend([f"--{key}", str(value)])

        # Execute command
        import subprocess
        print(f"  Command: {' '.join(cmd)}")

        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - start_time

        if result.returncode != 0:
            print(f"  ERROR: {result.stderr}")
        else:
            print(f"  Completed in {elapsed:.1f}s")

        result_path = output_dir / "results.json"
        if result_path.exists():
            result_paths.append(str(result_path))

    return result_paths


def aggregate_results(result_paths: List[str], eval_dir: Path) -> Dict[str, Any]:
    """Aggregate results from all shards."""
    print(f"\n{'='*60}")
    print("STEP 3: Aggregating results")
    print(f"{'='*60}")

    all_results = []
    all_predictions = []

    for path in result_paths:
        with open(path, 'r') as f:
            data = json.load(f)

        if 'results' in data:
            all_results.extend(data['results'])

    print(f"Total results: {len(all_results)}")

    # Compute aggregate metrics
    em_scores = []
    f1_scores = []
    abstained = 0
    tokens_used = []
    latencies = []

    for result in all_results:
        if result.get('abstained') or 'error' in result:
            abstained += 1
            continue

        metrics = result.get('metrics', {})
        if 'em' in metrics:
            em_scores.append(metrics['em'])
        if 'f1' in metrics:
            f1_scores.append(metrics['f1'])

        if result.get('tokens_used'):
            tokens_used.append(result['tokens_used'])
        if result.get('latency_ms'):
            latencies.append(result['latency_ms'])

    # Convert results to MuSiQue prediction format
    for result in all_results:
        pred = {
            'id': result.get('query_id', '').replace('musique_', '').replace('_dev_', '_'),
            'predicted_answer': result.get('prediction', ''),
            'predicted_support_idxs': [],
            'predicted_answerable': not result.get('abstained', False)
        }

        # Extract support indices from selected passages if available
        passages = result.get('selected_passages', [])
        for p in passages:
            if 'idx' in p:
                pred['predicted_support_idxs'].append(p['idx'])

        all_predictions.append(pred)

    # Save predictions in MuSiQue format
    predictions_path = eval_dir / "predictions.jsonl"
    with open(predictions_path, 'w') as f:
        for pred in all_predictions:
            f.write(json.dumps(pred) + '\n')

    # Save aggregated results
    aggregated = {
        'total': len(all_results),
        'valid': len(all_results) - abstained,
        'abstained': abstained,
        'abstention_rate': abstained / len(all_results) if all_results else 0,
        'avg_em': sum(em_scores) / len(em_scores) if em_scores else 0,
        'avg_f1': sum(f1_scores) / len(f1_scores) if f1_scores else 0,
        'avg_tokens': sum(tokens_used) / len(tokens_used) if tokens_used else 0,
        'avg_latency_ms': sum(latencies) / len(latencies) if latencies else 0
    }

    aggregated_path = eval_dir / "aggregated_metrics.json"
    with open(aggregated_path, 'w') as f:
        json.dump(aggregated, f, indent=2)

    print(f"\nAggregated metrics:")
    print(f"  EM: {aggregated['avg_em']:.4f}")
    print(f"  F1: {aggregated['avg_f1']:.4f}")
    print(f"  Abstention rate: {aggregated['abstention_rate']:.4f}")
    print(f"  Avg tokens: {aggregated['avg_tokens']:.1f}")
    print(f"  Avg latency: {aggregated['avg_latency_ms']:.1f}ms")

    return {
        'aggregated': aggregated,
        'predictions_path': str(predictions_path),
        'all_predictions': all_predictions
    }


def run_official_evaluation(
    predictions_path: str,
    ground_truth_path: str,
    eval_dir: Path
) -> Dict[str, float]:
    """Run official MuSiQue evaluation."""
    print(f"\n{'='*60}")
    print("STEP 4: Running official MuSiQue evaluation")
    print(f"{'='*60}")

    # Import MuSiQue evaluation (using importlib for module with dot in name)
    import importlib.util
    eval_path = Path(__file__).parent.parent / "musique" / "evaluate_v1.0.py"
    spec = importlib.util.spec_from_file_location("evaluate_v1_0", eval_path)
    evaluate_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluate_module)
    evaluate = evaluate_module.evaluate

    try:
        metrics = evaluate(predictions_path, ground_truth_path)

        print(f"\nOfficial MuSiQue metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")

        # Save official metrics
        official_path = eval_dir / "official_metrics.json"
        with open(official_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        return metrics

    except Exception as e:
        print(f"Warning: Could not run official evaluation: {e}")
        print("This may be because prediction format doesn't match expected format.")
        return {}


def generate_report(
    run_dir: Path,
    args: argparse.Namespace,
    data_stats: Dict,
    aggregated: Dict,
    official_metrics: Dict
) -> None:
    """Generate a comprehensive experiment report."""
    print(f"\n{'='*60}")
    print("STEP 5: Generating report")
    print(f"{'='*60}")

    report = f"""
# MuSiQue Experiment Report

## Configuration
- **Mode**: {args.mode}
- **Split**: {args.split}
- **Model**: {args.model}
- **Config**: {args.config}
- **Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Dataset Statistics
- Total examples: {data_stats['total_examples']}
- Answerable: {data_stats['answerable']}
- Unanswerable: {data_stats['unanswerable']}
- Hop types: {data_stats['hop_types']}

## Results (Aggregated)
- Exact Match: {aggregated['avg_em']:.4f}
- F1 Score: {aggregated['avg_f1']:.4f}
- Abstention Rate: {aggregated['abstention_rate']:.4f}
- Avg Tokens Used: {aggregated['avg_tokens']:.1f}
- Avg Latency: {aggregated['avg_latency_ms']:.1f}ms

## Official MuSiQue Metrics
"""

    if official_metrics:
        for metric, value in official_metrics.items():
            report += f"- {metric}: {value}\n"
    else:
        report += "- (Official evaluation not available)\n"

    report += f"""

## Files
- Predictions: evaluation/predictions.jsonl
- Aggregated Metrics: evaluation/aggregated_metrics.json
- Official Metrics: evaluation/official_metrics.json
- Raw Results: results/
"""

    report_path = run_dir / "REPORT.md"
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"Report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run MuSiQue experiments with TRIDENT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run TRIDENT Safe-Cover on answerable dev set
  python experiments/run_musique_experiment.py --mode safe_cover --split ans_dev

  # Run with limited examples for quick testing
  python experiments/run_musique_experiment.py --mode safe_cover --split ans_dev --limit 50

  # Run Self-RAG baseline
  python experiments/run_musique_experiment.py --mode self_rag --split ans_dev

  # Run on full dataset (with unanswerable questions)
  python experiments/run_musique_experiment.py --mode safe_cover --split full_dev
        """
    )

    # Required arguments
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["safe_cover", "pareto", "self_rag", "graphrag", "ketrag"],
        help="System mode to run"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="ans_dev",
        choices=["ans_dev", "ans_test", "full_dev", "full_test", "singlehop"],
        help="MuSiQue data split"
    )

    # Data options
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples")
    parser.add_argument("--shard_size", type=int, default=100, help="Examples per shard")

    # Model options
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="LLM model to use"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/musique.json",
        help="Config file path"
    )

    # Hardware options
    parser.add_argument("--device", type=int, default=0, help="GPU device ID (-1 for CPU)")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs for parallel processing")

    # Output options
    parser.add_argument(
        "--output_dir",
        type=str,
        default="runs/musique",
        help="Base output directory"
    )

    # Evaluation options
    parser.add_argument(
        "--skip_eval",
        action="store_true",
        help="Skip official MuSiQue evaluation"
    )

    # Additional model options
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM temperature")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max tokens to generate")
    parser.add_argument("--load_in_8bit", action="store_true", help="Load model in 8-bit")

    # Baseline-specific options
    parser.add_argument("--common_k", type=int, default=8, help="Retrieval k for baselines")
    parser.add_argument("--selfrag_use_critic", action="store_true", help="Use critic in Self-RAG")

    args = parser.parse_args()

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║           MuSiQue Experiment Runner for TRIDENT              ║
╠══════════════════════════════════════════════════════════════╣
║  Mode: {args.mode:<54}║
║  Split: {args.split:<53}║
║  Model: {args.model:<53}║
╚══════════════════════════════════════════════════════════════╝
""")

    # Setup directories
    dirs = setup_output_dirs(args.output_dir, args.mode, args.split)
    print(f"Run directory: {dirs['run_dir']}")

    # Step 1: Prepare data
    data_info = prepare_data(
        args.split,
        dirs['shards_dir'],
        args.shard_size,
        args.limit
    )

    # Step 2: Run experiments
    additional_args = {
        'temperature': args.temperature,
        'max_new_tokens': args.max_new_tokens,
        'load_in_8bit': args.load_in_8bit,
        'common_k': args.common_k,
        'selfrag_use_critic': args.selfrag_use_critic
    }

    result_paths = run_experiment(
        data_info['shard_paths'],
        dirs['results_dir'],
        args.mode,
        args.config,
        args.model,
        args.device,
        args.num_gpus,
        additional_args
    )

    # Step 3: Aggregate results
    agg_info = aggregate_results(result_paths, dirs['eval_dir'])

    # Step 4: Official evaluation
    official_metrics = {}
    if not args.skip_eval:
        # Determine ground truth file path
        split_to_file = {
            'ans_dev': 'musique_ans_v1.0_dev.jsonl',
            'ans_test': 'musique_ans_v1.0_test.jsonl',
            'full_dev': 'musique_full_v1.0_dev.jsonl',
            'full_test': 'musique_full_v1.0_test.jsonl'
        }

        if args.split in split_to_file:
            gt_path = Path(__file__).parent.parent / "musique" / "data" / split_to_file[args.split]
            if gt_path.exists():
                official_metrics = run_official_evaluation(
                    agg_info['predictions_path'],
                    str(gt_path),
                    dirs['eval_dir']
                )

    # Step 5: Generate report
    generate_report(
        dirs['run_dir'],
        args,
        data_info['stats'],
        agg_info['aggregated'],
        official_metrics
    )

    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {dirs['run_dir']}")


if __name__ == "__main__":
    main()
