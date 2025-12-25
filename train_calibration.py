#!/usr/bin/env python3
"""
Script to train calibration models for TRIDENT.

This trains the Mondrian/isotonic calibration on labeled data
to map NLI scores to calibrated p-values.
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Sequence

import numpy as np
from sklearn.model_selection import train_test_split

from trident.calibration import ReliabilityCalibrator, SelectionConditionalCalibrator
from trident.facets import FacetType
try:
    from trident.experimental_utils import CalibrationProvenance
    EXPERIMENTAL_UTILS_AVAILABLE = True
except Exception:
    EXPERIMENTAL_UTILS_AVAILABLE = False


def load_calibration_data(path: str, filter_lexical_false: bool = True, use_entail_prob: bool = True) -> Tuple[List[float], List[int], List[Dict], List[str]]:
    """
    Load calibration dataset.

    Expected format: JSONL with fields:
    - score: NLI/CE score (or use probs.entail if use_entail_prob=True)
    - probs: {entail, neutral, contra} probabilities
    - label: 1 if passage sufficient for facet, 0 otherwise
    - metadata: facet_type, text_length, lexical_match, etc.

    Args:
        path: Path to JSONL file
        filter_lexical_false: If True, skip samples where lexical_match=False
                             (since they're gated at runtime anyway)
        use_entail_prob: If True, use probs.entail instead of score
                        (more stable for calibration)
    """
    scores = []
    labels = []
    metadata = []
    ids = []

    n_total = 0
    n_filtered_lex = 0

    with open(path) as f:
        for line in f:
            item = json.loads(line.strip())
            n_total += 1

            meta = item.get('metadata', {})

            # ✅ Filter out lexical_match=False cases (they're gated at runtime)
            if filter_lexical_false and meta.get('lexical_match', None) is False:
                n_filtered_lex += 1
                continue

            # Use entail prob or score
            if use_entail_prob and 'probs' in item:
                probs = item['probs']
                # Use entailment probability directly (most stable)
                score = float(probs.get('entail', item.get('score', 0.0)))
            else:
                score = float(item.get('score', 0.0))

            scores.append(score)
            labels.append(int(item['label']))
            metadata.append(meta)
            ids.append(str(item.get('id', item.get('query_id', item.get('example_id', '')))))

    if n_filtered_lex > 0:
        print(f"  Filtered {n_filtered_lex}/{n_total} samples with lexical_match=False")
        print(f"  (Keeping {len(scores)} samples for calibration)")

    return scores, labels, metadata, ids


def generate_synthetic_calibration_data(
    n_samples: int = 1000
) -> Tuple[List[float], List[int], List[Dict], List[str]]:
    """Generate synthetic calibration data for testing."""
    
    np.random.seed(42)
    scores = []
    labels = []
    metadata = []
    ids = []
    
    facet_types = ["ENTITY", "RELATION", "TEMPORAL", "NUMERIC", "BRIDGE"]
    
    for _ in range(n_samples):
        # Generate score (higher scores should correlate with positive labels)
        score = np.random.beta(2, 2)  # Scores between 0 and 1
        
        # Generate label with probability based on score
        # Add noise to make it realistic
        prob_positive = score ** 2  # Quadratic relationship
        label = 1 if np.random.random() < prob_positive else 0
        
        # Generate metadata
        meta = {
            'facet_type': np.random.choice(facet_types),
            'text_length': np.random.randint(10, 500),
            'passage_id': f"p_{_}",
            'facet_id': f"f_{np.random.randint(10)}"
        }
        
        scores.append(score)
        labels.append(label)
        metadata.append(meta)
        ids.append(meta["facet_id"])
    
    return scores, labels, metadata, ids


def load_eval_ids(path: str) -> List[str]:
    """Load evaluation IDs from JSON or JSONL."""
    eval_ids: List[str] = []
    with open(path) as f:
        first_line = f.readline()
        f.seek(0)
        if first_line.lstrip().startswith("["):
            data = json.load(f)
            for item in data:
                if isinstance(item, dict):
                    eval_ids.append(str(item.get('_id', item.get('id', ''))))
        else:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                if isinstance(item, dict):
                    eval_ids.append(str(item.get('_id', item.get('id', ''))))
    return [eid for eid in eval_ids if eid]


def build_calibration_provenance(
    scores: Sequence[float],
    labels: Sequence[int],
    metadata: Sequence[Dict[str, Any]],
    ids: Sequence[str],
    args: argparse.Namespace
) -> Optional[Dict[str, Any]]:
    """Build calibration provenance record with bin statistics."""
    if not EXPERIMENTAL_UTILS_AVAILABLE:
        return None

    provenance = CalibrationProvenance(
        corpus_name=args.corpus_name or "",
        corpus_version=args.corpus_version or "",
        corpus_size=len(scores),
        calibration_split=args.calibration_split,
        retriever_version=args.retriever_version or "",
        reranker_version=args.reranker_version or "",
        shortlister_version=args.shortlister_version or "",
        verifier_version=args.verifier_version or "",
    )

    if ids:
        provenance.compute_corpus_hash([cid for cid in ids if cid])

    if args.eval_ids_path:
        eval_ids = load_eval_ids(args.eval_ids_path)
        if ids and eval_ids:
            provenance.validate_no_leakage(ids, eval_ids)

    calibrator = SelectionConditionalCalibrator(n_min=args.calibration_n_min)
    for score, label, meta in zip(scores, labels, metadata):
        calibrator.add_calibration_sample(
            score=score,
            is_sufficient=bool(label),
            facet_type=meta.get('facet_type', 'UNKNOWN'),
            text_length=int(meta.get('text_length', 100)),
            retriever_score=float(meta.get('retriever_score', 0.5)),
        )
    calibrator.finalize()

    merged_bins = set(calibrator.merged_bins.keys())
    for bin_key, bin_obj in calibrator.bins.items():
        if bin_key in merged_bins:
            continue
        score_percentiles = None
        if bin_obj.negative_scores:
            score_percentiles = {
                "p50": float(np.percentile(bin_obj.negative_scores, 50)),
                "p5": float(np.percentile(bin_obj.negative_scores, 5)),
                "p95": float(np.percentile(bin_obj.negative_scores, 95)),
            }
        provenance.add_bin_statistics(
            bin_key=bin_key,
            n_negatives=bin_obj.n_negatives,
            n_positives=bin_obj.n_positives,
            score_percentiles=score_percentiles,
        )

    return provenance.to_provenance_record()


def train_calibrator(
    scores: List[float],
    labels: List[int],
    metadata: List[Dict],
    use_mondrian: bool = True
) -> ReliabilityCalibrator:
    """Train calibrator on labeled data."""
    
    calibrator = ReliabilityCalibrator(
        use_mondrian=use_mondrian,
        version="v1.0"
    )
    
    if use_mondrian:
        print("Training Mondrian calibration...")
        
        # Group by facet type for Mondrian
        facet_groups = {}
        for score, label, meta in zip(scores, labels, metadata):
            facet_type = meta.get('facet_type', 'UNKNOWN')
            if facet_type not in facet_groups:
                facet_groups[facet_type] = {
                    'scores': [],
                    'labels': [],
                    'lengths': []
                }
            facet_groups[facet_type]['scores'].append(score)
            facet_groups[facet_type]['labels'].append(label)
            facet_groups[facet_type]['lengths'].append(meta.get('text_length', 100))
        
        # Train per facet type
        for facet_type, group_data in facet_groups.items():
            print(f"  Training for {facet_type}: {len(group_data['scores'])} samples")
            calibrator.fit(
                scores=group_data['scores'],
                sufficiency_labels=group_data['labels'],
                bucket_key=facet_type,
                text_lengths=group_data['lengths']
            )
    else:
        print("Training simple isotonic calibration...")
        
        # Simple calibration without Mondrian
        facet_groups = {}
        for score, label, meta in zip(scores, labels, metadata):
            facet_type = meta.get('facet_type', 'UNKNOWN')
            if facet_type not in facet_groups:
                facet_groups[facet_type] = {'scores': [], 'labels': []}
            facet_groups[facet_type]['scores'].append(score)
            facet_groups[facet_type]['labels'].append(label)
        
        for facet_type, group_data in facet_groups.items():
            print(f"  Training for {facet_type}: {len(group_data['scores'])} samples")
            calibrator.fit(
                scores=group_data['scores'],
                sufficiency_labels=group_data['labels'],
                bucket_key=facet_type
            )
    
    return calibrator


def evaluate_calibration(
    calibrator: ReliabilityCalibrator,
    test_scores: List[float],
    test_labels: List[int],
    test_metadata: List[Dict]
) -> Dict[str, float]:
    """Evaluate calibration quality."""
    
    metrics = {}
    
    # Group by facet type
    facet_groups = {}
    for score, label, meta in zip(test_scores, test_labels, test_metadata):
        facet_type = meta.get('facet_type', 'UNKNOWN')
        if facet_type not in facet_groups:
            facet_groups[facet_type] = {
                'scores': [],
                'labels': [],
                'lengths': [],
                'p_values': []
            }
        
        # Get calibrated p-value
        text_length = meta.get('text_length', 100)
        p_value = calibrator.to_pvalue(score, facet_type, text_length)
        
        facet_groups[facet_type]['scores'].append(score)
        facet_groups[facet_type]['labels'].append(label)
        facet_groups[facet_type]['lengths'].append(text_length)
        facet_groups[facet_type]['p_values'].append(p_value)
    
    # Compute metrics per facet type
    for facet_type, group_data in facet_groups.items():
        p_values = np.array(group_data['p_values'])
        labels = np.array(group_data['labels'])

        # Calibration error at different alpha levels
        alphas = [0.01, 0.05, 0.1]
        for alpha in alphas:
            pred_pos = p_values <= alpha
            pos = labels == 1
            neg = ~pos

            tp = int((pred_pos & pos).sum())
            fp = int((pred_pos & neg).sum())
            fn = int((~pred_pos & pos).sum())
            tn = int((~pred_pos & neg).sum())

            # ✅ TRUE FPR: FP / N_neg (not FP / predicted_positive!)
            fpr = fp / max(1, int(neg.sum()))
            tpr = tp / max(1, int(pos.sum()))  # recall / coverage

            precision = tp / max(1, (tp + fp))
            selection_rate = int(pred_pos.sum()) / max(1, len(labels))

            metrics[f"{facet_type}_fpr_at_{alpha}"] = float(fpr)
            metrics[f"{facet_type}_tpr_at_{alpha}"] = float(tpr)  # Same as coverage
            metrics[f"{facet_type}_precision_at_{alpha}"] = float(precision)
            metrics[f"{facet_type}_selection_rate_at_{alpha}"] = float(selection_rate)

            # Keep old coverage name for backwards compatibility
            metrics[f"{facet_type}_coverage_at_{alpha}"] = float(tpr)
        
        # Expected calibration error (ECE)
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            bin_mask = (p_values >= bin_boundaries[i]) & (p_values < bin_boundaries[i + 1])
            if bin_mask.sum() > 0:
                bin_p_values = p_values[bin_mask]
                bin_labels = labels[bin_mask]
                
                # Expected frequency in bin
                expected_freq = bin_p_values.mean()
                # Actual frequency
                actual_freq = (1 - bin_labels).mean()  # p-value represents P(not sufficient)
                
                ece += bin_mask.sum() * abs(expected_freq - actual_freq)
        
        ece /= len(p_values)
        metrics[f"{facet_type}_ece"] = ece
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train TRIDENT calibration")
    
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to calibration data (JSONL)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="calibration_model.pkl",
        help="Path to save calibrator"
    )
    parser.add_argument(
        "--use_synthetic",
        action="store_true",
        help="Use synthetic data for testing"
    )
    parser.add_argument(
        "--use_mondrian",
        action="store_true",
        default=True,
        help="Use Mondrian calibration (stratified)"
    )
    parser.add_argument(
        "--test_split",
        type=float,
        default=0.2,
        help="Fraction of data to use for testing"
    )
    parser.add_argument(
        "--filter_lexical_false",
        action="store_true",
        default=True,
        help="Filter out samples with lexical_match=False (default: True)"
    )
    parser.add_argument(
        "--no_filter_lexical_false",
        action="store_false",
        dest="filter_lexical_false",
        help="Don't filter lexical_match=False samples"
    )
    parser.add_argument(
        "--use_entail_prob",
        action="store_true",
        default=True,
        help="Use probs.entail instead of score (default: True)"
    )
    parser.add_argument(
        "--no_use_entail_prob",
        action="store_false",
        dest="use_entail_prob",
        help="Use score instead of probs.entail"
    )
    parser.add_argument(
        "--output_provenance",
        type=str,
        default=None,
        help="Optional path to save calibration provenance metadata (JSON)"
    )
    parser.add_argument("--corpus_name", type=str, default="", help="Calibration corpus name")
    parser.add_argument("--corpus_version", type=str, default="", help="Calibration corpus version")
    parser.add_argument("--calibration_split", type=str, default="train", help="Calibration split name")
    parser.add_argument("--retriever_version", type=str, default="", help="Retriever version/hash")
    parser.add_argument("--reranker_version", type=str, default="", help="Reranker version/hash")
    parser.add_argument("--shortlister_version", type=str, default="", help="Shortlister version/hash")
    parser.add_argument("--verifier_version", type=str, default="", help="Verifier version/hash")
    parser.add_argument("--eval_ids_path", type=str, default=None, help="Optional eval IDs for leakage check")
    parser.add_argument(
        "--calibration_n_min",
        type=int,
        default=50,
        help="Minimum negatives per bin before merging (for provenance audit)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("TRIDENT Calibration Training")
    print("=" * 60)
    
    # Load or generate data
    if args.use_synthetic or not args.data_path:
        print("\nGenerating synthetic calibration data...")
        scores, labels, metadata, ids = generate_synthetic_calibration_data(2000)
    else:
        print(f"\nLoading calibration data from {args.data_path}...")
        print(f"  Filter lexical_match=False: {args.filter_lexical_false}")
        print(f"  Use probs.entail: {args.use_entail_prob}")
        scores, labels, metadata, ids = load_calibration_data(
            args.data_path,
            filter_lexical_false=args.filter_lexical_false,
            use_entail_prob=args.use_entail_prob
        )
    
    print(f"Total samples: {len(scores)}")
    print(f"Positive rate: {sum(labels) / len(labels):.3f}")
    
    # Split data
    indices = list(range(len(scores)))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=args.test_split,
        random_state=42,
        stratify=labels
    )
    
    train_scores = [scores[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    train_metadata = [metadata[i] for i in train_idx]
    
    test_scores = [scores[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]
    test_metadata = [metadata[i] for i in test_idx]
    
    print(f"\nTraining samples: {len(train_scores)}")
    print(f"Test samples: {len(test_scores)}")
    
    # Train calibrator
    print("\n" + "=" * 40)
    calibrator = train_calibrator(
        train_scores,
        train_labels,
        train_metadata,
        use_mondrian=args.use_mondrian
    )
    
    # Evaluate
    print("\n" + "=" * 40)
    print("Evaluating calibration...")
    metrics = evaluate_calibration(
        calibrator,
        test_scores,
        test_labels,
        test_metadata
    )
    
    print("\nCalibration Metrics:")
    
    # Aggregate metrics by type
    metric_types = {}
    for key, value in metrics.items():
        parts = key.split('_')
        facet_type = parts[0]
        metric_name = '_'.join(parts[1:])
        
        if facet_type not in metric_types:
            metric_types[facet_type] = {}
        metric_types[facet_type][metric_name] = value
    
    for facet_type, type_metrics in metric_types.items():
        print(f"\n  {facet_type}:")
        for metric_name, value in type_metrics.items():
            print(f"    {metric_name}: {value:.3f}")
    
    # Save calibrator
    print("\n" + "=" * 40)
    output_path = Path(args.output_path)

    # ✅ Ensure .pkl extension to avoid confusion
    if output_path.suffix != '.pkl':
        print(f"⚠️  Warning: output_path should use .pkl extension (got {output_path.suffix})")
        print(f"   Auto-correcting to .pkl")
        output_path = output_path.with_suffix('.pkl')

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as pickle
    with open(output_path, 'wb') as f:
        pickle.dump(calibrator, f)
    print(f"Calibrator saved to {output_path}")

    # Also save as JSON
    json_path = output_path.with_suffix('.json')
    calibrator.save(str(json_path))
    print(f"Calibrator JSON saved to {json_path}")

    provenance_record = build_calibration_provenance(scores, labels, metadata, ids, args)
    if provenance_record and args.output_provenance:
        with open(args.output_provenance, "w") as f:
            json.dump(provenance_record, f, indent=2)
        print(f"Calibration provenance saved to {args.output_provenance}")
    
    # Print usage instructions
    print("\n" + "=" * 40)
    print("To use this calibrator:")
    print("\n1. Load in Python:")
    print("   import pickle")
    print(f"   with open('{output_path}', 'rb') as f:")
    print("       calibrator = pickle.load(f)")
    print("\n2. Or load from JSON:")
    print("   from trident.calibration import ReliabilityCalibrator")
    print(f"   calibrator = ReliabilityCalibrator.load('{json_path}')")
    print("\n3. Use for calibration:")
    print("   p_value = calibrator.to_pvalue(score, facet_type, text_length)")


if __name__ == "__main__":
    main()
