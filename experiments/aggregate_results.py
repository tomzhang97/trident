#!/usr/bin/env python3
"""Script to aggregate results from parallel TRIDENT evaluation."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd

try:
    from trident.experimental_utils import aggregate_certificate_audit
    EXPERIMENTAL_UTILS_AVAILABLE = True
except Exception:
    EXPERIMENTAL_UTILS_AVAILABLE = False


def load_shard_results(results_dir: Path) -> List[Dict[str, Any]]:
    """Load results from all shard directories."""
    all_results = []
    
    for shard_dir in results_dir.iterdir():
        if not shard_dir.is_dir():
            continue
        
        results_file = shard_dir / "results.json"
        if not results_file.exists():
            print(f"Warning: No results found in {shard_dir}")
            continue
        
        with open(results_file) as f:
            data = json.load(f)
            all_results.append(data)
    
    return all_results


def aggregate_metrics(shard_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate metrics across all shards."""
    
    # Collect all individual predictions
    all_predictions = []
    for shard in shard_results:
        if 'results' in shard:
            all_predictions.extend(shard['results'])
    
    if not all_predictions:
        return {}
    
    # Compute aggregate metrics
    metrics = {
        'total_queries': len(all_predictions),
        'abstained': sum(1 for p in all_predictions if p.get('abstained', False)),
        'errors': sum(1 for p in all_predictions if 'error' in p),
    }
    
    # Filter valid predictions
    valid_predictions = [p for p in all_predictions 
                        if not p.get('abstained', False) and 'error' not in p]
    
    if valid_predictions:
        # Compute EM and F1
        em_scores = []
        f1_scores = []
        
        for pred in valid_predictions:
            # Simple EM check
            if 'prediction' in pred and 'ground_truth' in pred:
                pred_text = pred['prediction'].strip().lower()
                for gt in pred['ground_truth']:
                    if pred_text == gt.strip().lower():
                        em_scores.append(1.0)
                        break
                else:
                    em_scores.append(0.0)
                
                # Simple F1 (token overlap)
                pred_tokens = set(pred_text.split())
                best_f1 = 0.0
                for gt in pred['ground_truth']:
                    gt_tokens = set(gt.strip().lower().split())
                    if pred_tokens and gt_tokens:
                        overlap = len(pred_tokens & gt_tokens)
                        precision = overlap / len(pred_tokens)
                        recall = overlap / len(gt_tokens)
                        if precision + recall > 0:
                            f1 = 2 * precision * recall / (precision + recall)
                            best_f1 = max(best_f1, f1)
                f1_scores.append(best_f1)
        
        metrics['exact_match'] = np.mean(em_scores) if em_scores else 0.0
        metrics['f1_score'] = np.mean(f1_scores) if f1_scores else 0.0
        
        # Token and latency metrics
        token_values = [p.get('tokens_used', 0) for p in valid_predictions]
        evidence_values = [
            p.get('metrics', {}).get('evidence_tokens', p.get('stats', {}).get('evidence_tokens', 0))
            for p in valid_predictions
        ]
        overhead_values = [
            max(p.get('tokens_used', 0) - p.get('metrics', {}).get('evidence_tokens', 0), 0)
            for p in valid_predictions
        ]

        metrics['avg_tokens'] = float(np.mean(token_values)) if token_values else 0.0
        metrics['avg_latency_ms'] = float(np.mean([p.get('latency_ms', 0) for p in valid_predictions]))
        metrics['avg_evidence_tokens'] = float(np.mean(evidence_values)) if evidence_values else 0.0
        metrics['avg_overhead_tokens'] = float(np.mean(overhead_values)) if overhead_values else 0.0

        if token_values:
            metrics['tokens_p50'] = float(np.percentile(token_values, 50))
            metrics['tokens_p90'] = float(np.percentile(token_values, 90))
            metrics['tokens_p95'] = float(np.percentile(token_values, 95))
        if evidence_values:
            metrics['evidence_p50'] = float(np.percentile(evidence_values, 50))
            metrics['evidence_p90'] = float(np.percentile(evidence_values, 90))
            metrics['evidence_p95'] = float(np.percentile(evidence_values, 95))
        
        # Mode-specific metrics
        if valid_predictions[0].get('mode') == 'safe_cover':
            # Certificate statistics
            num_with_certs = sum(1 for p in valid_predictions 
                                if p.get('certificates'))
            metrics['queries_with_certificates'] = num_with_certs
            
            # Coverage statistics
            coverage_rates = []
            for pred in valid_predictions:
                if 'metrics' in pred:
                    coverage_rates.append(pred['metrics'].get('coverage', 0))
            metrics['avg_coverage_rate'] = np.mean(coverage_rates) if coverage_rates else 0.0
        
        elif valid_predictions[0].get('mode') == 'pareto':
            # Utility statistics
            utilities = []
            for pred in valid_predictions:
                if 'metrics' in pred:
                    utilities.append(pred['metrics'].get('utility', 0))
            metrics['avg_utility'] = np.mean(utilities) if utilities else 0.0
    
    # Abstention analysis
    metrics['abstention_rate'] = metrics['abstained'] / metrics['total_queries']
    metrics['error_rate'] = metrics['errors'] / metrics['total_queries']
    
    if EXPERIMENTAL_UTILS_AVAILABLE and valid_predictions:
        mode = valid_predictions[0].get('mode', '')
        if mode in ('safe_cover', 'trident_safe_cover'):
            budget_cap = None
            if shard_results and 'config' in shard_results[0]:
                config = shard_results[0]['config']
                safe_cover_cfg = config.get('safe_cover', {})
                budget_cap = safe_cover_cfg.get('max_evidence_tokens') or safe_cover_cfg.get('token_cap')
            audit_metrics = aggregate_certificate_audit(valid_predictions, budget_cap)
            metrics['certificate_audit'] = audit_metrics.to_audit_table()

    return metrics


def generate_report(
    metrics: Dict[str, Any],
    shard_results: List[Dict[str, Any]],
    output_path: Path
) -> None:
    """Generate detailed evaluation report."""
    
    report = []
    report.append("=" * 60)
    report.append("TRIDENT EVALUATION REPORT")
    report.append("=" * 60)
    
    # Basic statistics
    report.append("\nDATASET STATISTICS:")
    report.append(f"  Total queries: {metrics.get('total_queries', 0)}")
    report.append(f"  Abstained: {metrics.get('abstained', 0)}")
    report.append(f"  Errors: {metrics.get('errors', 0)}")
    report.append(f"  Valid predictions: {metrics['total_queries'] - metrics['abstained'] - metrics['errors']}")
    
    # Performance metrics
    report.append("\nPERFORMANCE METRICS:")
    report.append(f"  Exact Match: {metrics.get('exact_match', 0):.3f}")
    report.append(f"  F1 Score: {metrics.get('f1_score', 0):.3f}")
    report.append(f"  Abstention Rate: {metrics.get('abstention_rate', 0):.3f}")
    
    # Efficiency metrics
    report.append("\nEFFICIENCY METRICS:")
    report.append(f"  Avg Tokens Used: {metrics.get('avg_tokens', 0):.1f}")
    report.append(f"  Tokens p50/p90/p95: {metrics.get('tokens_p50', 0):.1f}/"
                  f"{metrics.get('tokens_p90', 0):.1f}/{metrics.get('tokens_p95', 0):.1f}")
    report.append(f"  Avg Evidence Tokens: {metrics.get('avg_evidence_tokens', 0):.1f}")
    report.append(f"  Evidence p50/p90/p95: {metrics.get('evidence_p50', 0):.1f}/"
                  f"{metrics.get('evidence_p90', 0):.1f}/{metrics.get('evidence_p95', 0):.1f}")
    report.append(f"  Avg Overhead Tokens: {metrics.get('avg_overhead_tokens', 0):.1f}")
    report.append(f"  Avg Latency (ms): {metrics.get('avg_latency_ms', 0):.1f}")
    
    # Mode-specific metrics
    if 'avg_coverage_rate' in metrics:
        report.append("\nSAFE-COVER METRICS:")
        report.append(f"  Avg Coverage Rate: {metrics['avg_coverage_rate']:.3f}")
        report.append(f"  Queries with Certificates: {metrics.get('queries_with_certificates', 0)}")
        if 'certificate_audit' in metrics:
            audit = metrics['certificate_audit']
            report.append("  Certificate Audit:")
            report.append(f"    - Certificate rate: {audit['certificate_metrics']['certificate_rate']:.1%}")
            report.append(f"    - Randomized p-values: {audit['pvalue_modes']['randomized_fraction']:.1%}")
            report.append(f"    - Min/Median n_b: {audit['bin_coverage']['min_n_b_across_bins']:.0f}/"
                          f"{audit['bin_coverage']['median_n_b_across_bins']:.0f}")
            report.append(f"    - Near-threshold count: {audit['threshold_analysis']['near_threshold_count']}")
            report.append(f"    - Abstentions (dual-LB/no-cover): {audit['abstention_breakdown']['dual_lb']:.1f}%/"
                          f"{audit['abstention_breakdown']['no_cover']:.1f}%")
            report.append(f"    - Avg LB margin: {audit['lb_margin']['avg_margin']:.2f}")
    
    if 'avg_utility' in metrics:
        report.append("\nPARETO METRICS:")
        report.append(f"  Avg Utility: {metrics['avg_utility']:.3f}")
    
    # Configuration info (from first shard)
    if shard_results and 'config' in shard_results[0]:
        config = shard_results[0]['config']
        report.append("\nCONFIGURATION:")
        report.append(f"  Mode: {config.get('mode', 'unknown')}")
        report.append(f"  Model: {config.get('llm', {}).get('model_name', 'unknown')}")
        report.append(f"  Budget: {config.get('safe_cover', {}).get('token_cap', 'N/A')}")
    
    # Write report
    report_text = "\n".join(report)
    print(report_text)
    
    with open(output_path / "evaluation_report.txt", 'w') as f:
        f.write(report_text)
    
    # Also save as JSON
    with open(output_path / "aggregated_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)


def create_error_analysis(
    shard_results: List[Dict[str, Any]],
    output_path: Path
) -> None:
    """Create error analysis report."""
    
    errors = []
    abstentions = []
    
    for shard in shard_results:
        if 'results' not in shard:
            continue
        
        for result in shard['results']:
            if 'error' in result:
                errors.append({
                    'query_id': result.get('query_id'),
                    'question': result.get('question', '')[:100],
                    'error': result['error']
                })
            elif result.get('abstained'):
                abstentions.append({
                    'query_id': result.get('query_id'),
                    'question': result.get('question', '')[:100],
                    'reason': 'abstained'
                })
    
    # Save error analysis
    if errors:
        pd.DataFrame(errors).to_csv(output_path / "errors.csv", index=False)
        print(f"Found {len(errors)} errors - saved to errors.csv")
    
    if abstentions:
        pd.DataFrame(abstentions).to_csv(output_path / "abstentions.csv", index=False)
        print(f"Found {len(abstentions)} abstentions - saved to abstentions.csv")


def create_pareto_curves(
    shard_results: List[Dict[str, Any]],
    output_path: Path
) -> None:
    """Extract and save Pareto curve data if available."""
    
    pareto_data = []
    
    for shard in shard_results:
        if 'results' not in shard:
            continue
        
        for result in shard['results']:
            if result.get('mode') == 'pareto' and 'metrics' in result:
                pareto_data.append({
                    'utility': result['metrics'].get('utility', 0),
                    'cost': result.get('tokens_used', 0),
                    'coverage': result['metrics'].get('coverage', 0)
                })
    
    if pareto_data:
        df = pd.DataFrame(pareto_data)
        df.to_csv(output_path / "pareto_data.csv", index=False)
        print(f"Saved Pareto curve data for {len(pareto_data)} queries")


def main():
    parser = argparse.ArgumentParser(description="Aggregate TRIDENT evaluation results")
    
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory containing shard result folders"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for aggregated results (defaults to results_dir)"
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory {results_dir} does not exist")
        return
    
    output_dir = Path(args.output_dir) if args.output_dir else results_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading results from {results_dir}...")
    shard_results = load_shard_results(results_dir)
    
    if not shard_results:
        print("No results found!")
        return
    
    print(f"Found {len(shard_results)} shard results")
    
    # Aggregate metrics
    metrics = aggregate_metrics(shard_results)
    
    # Generate reports
    generate_report(metrics, shard_results, output_dir)
    create_error_analysis(shard_results, output_dir)
    create_pareto_curves(shard_results, output_dir)
    
    print(f"\nResults aggregated and saved to {output_dir}")


if __name__ == "__main__":
    main()
