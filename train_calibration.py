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
from typing import Dict, List, Tuple, Any

import numpy as np
from sklearn.model_selection import train_test_split

from trident.calibration import ReliabilityCalibrator
from trident.facets import FacetType


def load_calibration_data(path: str) -> Tuple[List[float], List[int], List[Dict]]:
    """
    Load calibration dataset.
    
    Expected format: JSONL with fields:
    - score: NLI/CE score
    - label: 1 if passage sufficient for facet, 0 otherwise
    - metadata: facet_type, text_length, etc.
    """
    scores = []
    labels = []
    metadata = []
    
    with open(path) as f:
        for line in f:
            item = json.loads(line.strip())
            scores.append(item['score'])
            labels.append(item['label'])
            metadata.append(item.get('metadata', {}))
    
    return scores, labels, metadata


def generate_synthetic_calibration_data(n_samples: int = 1000) -> Tuple[List[float], List[int], List[Dict]]:
    """Generate synthetic calibration data for testing."""
    
    np.random.seed(42)
    scores = []
    labels = []
    metadata = []
    
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
    
    return scores, labels, metadata


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
            # Predicted positives
            predicted_positive = p_values <= alpha
            actual_positive = labels == 1
            
            # False positive rate (should be ≤ alpha for good calibration)
            if predicted_positive.sum() > 0:
                fpr = ((predicted_positive & ~actual_positive).sum() / 
                      predicted_positive.sum())
            else:
                fpr = 0.0
            
            metrics[f"{facet_type}_fpr_at_{alpha}"] = fpr
            
            # Coverage (fraction of positives detected)
            if actual_positive.sum() > 0:
                coverage = ((predicted_positive & actual_positive).sum() / 
                          actual_positive.sum())
            else:
                coverage = 0.0
            
            metrics[f"{facet_type}_coverage_at_{alpha}"] = coverage
        
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
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("TRIDENT Calibration Training")
    print("=" * 60)
    
    # Load or generate data
    if args.use_synthetic or not args.data_path:
        print("\nGenerating synthetic calibration data...")
        scores, labels, metadata = generate_synthetic_calibration_data(2000)
    else:
        print(f"\nLoading calibration data from {args.data_path}...")
        scores, labels, metadata = load_calibration_data(args.data_path)
    
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as pickle
    with open(output_path, 'wb') as f:
        pickle.dump(calibrator, f)
    print(f"Calibrator saved to {output_path}")
    
    # Also save as JSON
    json_path = output_path.with_suffix('.json')
    calibrator.save(str(json_path))
    print(f"Calibrator JSON saved to {json_path}")
    
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