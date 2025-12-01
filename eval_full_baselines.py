#!/usr/bin/env python3
"""
Evaluation script for full baseline systems (GraphRAG, Self-RAG, KET-RAG) on HotpotQA.

Usage:
    python eval_full_baselines.py \
        --data_path data/hotpotqa_dev_shards/shard_0.jsonl \
        --output_dir results/full_baselines \
        --baselines graphrag selfrag ketrag \
        --max_samples 100

Environment variables:
    GRAPHRAG_API_KEY or OPENAI_API_KEY: Required for GraphRAG and KET-RAG
    HF_TOKEN: Required for Self-RAG model download
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List
import time
from tqdm import tqdm
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from baselines.full_baseline_interface import compute_exact_match, compute_f1
from baselines.full_graphrag_adapter import FullGraphRAGAdapter
from baselines.full_selfrag_adapter import FullSelfRAGAdapter
from baselines.full_ketrag_adapter import FullKETRAGAdapter


def load_hotpotqa_data(data_path: str, max_samples: int = None) -> List[Dict[str, Any]]:
    """Load HotpotQA data from JSONL file."""
    data = []
    with open(data_path, 'r') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            data.append(json.loads(line))
    return data


def evaluate_baseline(
    baseline_name: str,
    baseline_system,
    data: List[Dict[str, Any]],
    output_path: str
) -> Dict[str, Any]:
    """
    Evaluate a baseline system on HotpotQA data.

    Metrics Separation (Aligned with Fair Baseline Comparison):
    - Query-only metrics (tokens_used, latency_ms): Online inference costs only
    - Total metrics: Include offline indexing costs from stats['indexing_*']
    - This matches how original papers report performance

    Args:
        baseline_name: Name of the baseline ('graphrag', 'selfrag', 'ketrag')
        baseline_system: Initialized baseline system
        data: List of HotpotQA examples
        output_path: Path to save results

    Returns:
        Summary statistics with separated query/total metrics
    """
    print(f"\n{'='*80}")
    print(f"Evaluating {baseline_name.upper()}")
    print(f"{'='*80}\n")

    results = []
    em_scores = []
    f1_scores = []

    # Query-only metrics (matches original paper claims)
    query_tokens = []
    query_latencies = []

    # Total metrics (includes indexing overhead)
    total_tokens = []
    total_latencies = []

    # Indexing metrics (for reference)
    indexing_tokens = []
    indexing_latencies = []

    abstention_count = 0

    for example in tqdm(data, desc=f"{baseline_name}"):
        question = example['question']
        context = example.get('context', [])
        supporting_facts = example.get('supporting_facts', [])
        answer = example.get('answer', '')
        question_id = example.get('_id', 'unknown')
        question_type = example.get('type', 'unknown')

        try:
            # Generate answer
            response = baseline_system.answer(
                question=question,
                context=context,
                supporting_facts=supporting_facts,
                metadata={
                    'question_id': question_id,
                    'type': question_type,
                }
            )

            # Compute metrics
            em = compute_exact_match(response.answer, answer)
            f1 = compute_f1(response.answer, answer)

            em_scores.append(em)
            f1_scores.append(f1)

            # Query-only metrics (PRIMARY)
            query_tokens.append(response.tokens_used)
            query_latencies.append(response.latency_ms)

            # Extract indexing metrics from stats
            idx_tokens = response.stats.get('indexing_tokens', 0)
            idx_latency = response.stats.get('indexing_latency_ms', 0.0)
            tot_tokens = response.stats.get('total_cost_tokens', response.tokens_used)

            indexing_tokens.append(idx_tokens)
            indexing_latencies.append(idx_latency)
            total_tokens.append(tot_tokens)
            total_latencies.append(response.latency_ms + idx_latency)

            if response.abstained:
                abstention_count += 1

            # Save result
            result = {
                'question_id': question_id,
                'question': question,
                'answer': answer,
                'prediction': response.answer,
                'em': em,
                'f1': f1,
                # Query-only (PRIMARY)
                'tokens_used': response.tokens_used,
                'latency_ms': response.latency_ms,
                # Total costs
                'total_tokens': tot_tokens,
                'total_latency_ms': response.latency_ms + idx_latency,
                'indexing_tokens': idx_tokens,
                'indexing_latency_ms': idx_latency,
                'abstained': response.abstained,
                'mode': response.mode,
                'stats': response.stats,
                'type': question_type,
            }
            results.append(result)

        except Exception as e:
            print(f"\nError processing question {question_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Compute summary statistics
    summary = {
        'baseline': baseline_name,
        'num_examples': len(data),
        'num_processed': len(results),
        'num_abstained': abstention_count,
        'abstention_rate': abstention_count / len(results) if results else 0.0,

        # Accuracy metrics
        'avg_em': np.mean(em_scores) if em_scores else 0.0,
        'avg_f1': np.mean(f1_scores) if f1_scores else 0.0,

        # Query-only metrics (PRIMARY - matches original papers)
        'avg_query_tokens': np.mean(query_tokens) if query_tokens else 0.0,
        'median_query_tokens': np.median(query_tokens) if query_tokens else 0.0,
        'avg_query_latency_ms': np.mean(query_latencies) if query_latencies else 0.0,
        'median_query_latency_ms': np.median(query_latencies) if query_latencies else 0.0,

        # Total metrics (includes indexing)
        'avg_total_tokens': np.mean(total_tokens) if total_tokens else 0.0,
        'median_total_tokens': np.median(total_tokens) if total_tokens else 0.0,
        'avg_total_latency_ms': np.mean(total_latencies) if total_latencies else 0.0,
        'median_total_latency_ms': np.median(total_latencies) if total_latencies else 0.0,

        # Indexing overhead (for reference)
        'avg_indexing_tokens': np.mean(indexing_tokens) if indexing_tokens else 0.0,
        'avg_indexing_latency_ms': np.mean(indexing_latencies) if indexing_latencies else 0.0,
    }

    # Save results
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Save individual results
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    # Save summary
    summary_file = output_file.parent / f"{baseline_name}_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{baseline_name.upper()} Results:")
    print(f"  EM: {summary['avg_em']:.4f}")
    print(f"  F1: {summary['avg_f1']:.4f}")
    print(f"  Query Tokens (PRIMARY): {summary['avg_query_tokens']:.1f}")
    print(f"  Total Tokens (w/ indexing): {summary['avg_total_tokens']:.1f}")
    print(f"  Query Latency (PRIMARY): {summary['avg_query_latency_ms']:.1f}ms")
    print(f"  Total Latency (w/ indexing): {summary['avg_total_latency_ms']:.1f}ms")
    print(f"  Abstention Rate: {summary['abstention_rate']:.2%}")
    print(f"\nResults saved to: {output_file}")
    print(f"Summary saved to: {summary_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate full baseline systems on HotpotQA")

    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to HotpotQA data file (JSONL format)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/full_baselines",
        help="Output directory for results"
    )
    parser.add_argument(
        "--baselines",
        nargs='+',
        choices=['graphrag', 'selfrag', 'ketrag', 'all'],
        default=['all'],
        help="Which baselines to evaluate"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (for testing)"
    )

    # GraphRAG/KET-RAG options
    parser.add_argument(
        "--graphrag_model",
        type=str,
        default="gpt-4o-mini",
        help="Model to use for GraphRAG and KET-RAG"
    )

    # Self-RAG options
    parser.add_argument(
        "--selfrag_model",
        type=str,
        default="selfrag/selfrag_llama2_7b",
        help="Self-RAG model name (7b or 13b)"
    )
    parser.add_argument(
        "--selfrag_max_tokens",
        type=int,
        default=100,
        help="Max tokens for Self-RAG generation"
    )

    args = parser.parse_args()

    # Expand 'all' to all baselines
    if 'all' in args.baselines:
        args.baselines = ['graphrag', 'selfrag', 'ketrag']

    # Load data
    print(f"Loading data from: {args.data_path}")
    data = load_hotpotqa_data(args.data_path, args.max_samples)
    print(f"Loaded {len(data)} examples")

    # Check API keys
    api_key = os.getenv("GRAPHRAG_API_KEY") or os.getenv("OPENAI_API_KEY")
    if ('graphrag' in args.baselines or 'ketrag' in args.baselines) and not api_key:
        print("ERROR: GraphRAG and KET-RAG require GRAPHRAG_API_KEY or OPENAI_API_KEY environment variable")
        sys.exit(1)

    # Evaluate each baseline
    summaries = {}

    for baseline_name in args.baselines:
        print(f"\n{'#'*80}")
        print(f"# Setting up {baseline_name.upper()}")
        print(f"{'#'*80}\n")

        try:
            # Initialize baseline system
            if baseline_name == 'graphrag':
                baseline_system = FullGraphRAGAdapter(
                    api_key=api_key,
                    model=args.graphrag_model,
                    temperature=0.0,
                    max_tokens=500,
                )
            elif baseline_name == 'selfrag':
                baseline_system = FullSelfRAGAdapter(
                    model_name=args.selfrag_model,
                    max_tokens=args.selfrag_max_tokens,
                    temperature=0.0,
                    provide_context=True,  # Use HotpotQA context
                )
            elif baseline_name == 'ketrag':
                baseline_system = FullKETRAGAdapter(
                    api_key=api_key,
                    model=args.graphrag_model,
                    temperature=0.0,
                    max_tokens=500,
                    skeleton_ratio=0.3,
                    max_skeleton_triples=10,
                    max_keyword_chunks=5,
                )
            else:
                print(f"Unknown baseline: {baseline_name}")
                continue

            # Evaluate
            output_path = os.path.join(args.output_dir, f"{baseline_name}_results.jsonl")
            summary = evaluate_baseline(baseline_name, baseline_system, data, output_path)
            summaries[baseline_name] = summary

        except Exception as e:
            print(f"\nERROR setting up {baseline_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Print comparison
    if len(summaries) > 1:
        print(f"\n{'='*80}")
        print("COMPARISON SUMMARY (Query-Only Metrics)")
        print(f"{'='*80}\n")
        print(f"{'Baseline':<15} {'EM':<10} {'F1':<10} {'Query Tokens':<15} {'Query Latency':<15}")
        print("-" * 80)
        for name, summary in summaries.items():
            print(f"{name:<15} {summary['avg_em']:<10.4f} {summary['avg_f1']:<10.4f} "
                  f"{summary['avg_query_tokens']:<15.1f} {summary['avg_query_latency_ms']:<15.1f}ms")

        print(f"\n{'='*80}")
        print("COMPARISON SUMMARY (Total Metrics w/ Indexing)")
        print(f"{'='*80}\n")
        print(f"{'Baseline':<15} {'EM':<10} {'F1':<10} {'Total Tokens':<15} {'Total Latency':<15}")
        print("-" * 80)
        for name, summary in summaries.items():
            print(f"{name:<15} {summary['avg_em']:<10.4f} {summary['avg_f1']:<10.4f} "
                  f"{summary['avg_total_tokens']:<15.1f} {summary['avg_total_latency_ms']:<15.1f}ms")

    # Save combined summary
    combined_summary_path = os.path.join(args.output_dir, "combined_summary.json")
    with open(combined_summary_path, 'w') as f:
        json.dump(summaries, f, indent=2)
    print(f"\nCombined summary saved to: {combined_summary_path}")


if __name__ == "__main__":
    main()
