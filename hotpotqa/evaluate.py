#!/usr/bin/env python3
"""
HotpotQA Evaluation Script (for TRIDENT)

This provides official HotpotQA evaluation metrics:
1. Answer EM and F1 (token-level, SQuAD-style)
2. Supporting Facts EM and F1 (set-based)
3. Joint EM and F1 (product of answer and support metrics)

Based on the official HotpotQA evaluation: https://github.com/hotpotqa/hotpot

Usage:
    # Standard HotpotQA evaluation
    python evaluate.py predictions.json ground_truth.json

    # Evaluate TRIDENT results
    python evaluate.py --trident_results results.json --ground_truth hotpot_dev_distractor_v1.json

    # Evaluate directory of shard results
    python evaluate.py --trident_dir runs/hotpotqa/results/ --ground_truth hotpot_dev_distractor_v1.json
"""

import json
import argparse
import re
import string
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set, Optional
from collections import Counter


# =============================================================================
# Answer Normalization (from SQuAD)
# =============================================================================

def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 score."""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(prediction_tokens) if prediction_tokens else 0.0
    recall = num_same / len(ground_truth_tokens) if ground_truth_tokens else 0.0

    if precision + recall == 0:
        return 0.0

    return (2 * precision * recall) / (precision + recall)


def exact_match_score(prediction: str, ground_truth: str) -> float:
    """Compute exact match score."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


# =============================================================================
# Supporting Facts Evaluation
# =============================================================================

def support_f1_score(
    prediction: List[Tuple[str, int]],
    ground_truth: List[Tuple[str, int]]
) -> float:
    """Compute F1 score for supporting facts."""
    # Convert to sets of (title, sent_idx) tuples
    pred_set: Set[Tuple[str, int]] = set()
    for item in prediction:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            pred_set.add((str(item[0]), int(item[1])))

    gt_set: Set[Tuple[str, int]] = set()
    for item in ground_truth:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            gt_set.add((str(item[0]), int(item[1])))

    if not pred_set and not gt_set:
        return 1.0
    if not pred_set or not gt_set:
        return 0.0

    intersection = pred_set & gt_set
    precision = len(intersection) / len(pred_set)
    recall = len(intersection) / len(gt_set)

    if precision + recall == 0:
        return 0.0

    return (2 * precision * recall) / (precision + recall)


def support_em_score(
    prediction: List[Tuple[str, int]],
    ground_truth: List[Tuple[str, int]]
) -> float:
    """Compute exact match score for supporting facts."""
    pred_set: Set[Tuple[str, int]] = set()
    for item in prediction:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            pred_set.add((str(item[0]), int(item[1])))

    gt_set: Set[Tuple[str, int]] = set()
    for item in ground_truth:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            gt_set.add((str(item[0]), int(item[1])))

    return float(pred_set == gt_set)


# =============================================================================
# Main Evaluation
# =============================================================================

def evaluate(
    predictions: Dict[str, Dict[str, Any]],
    ground_truths: Dict[str, Dict[str, Any]]
) -> Dict[str, float]:
    """
    Evaluate predictions against ground truths.

    Args:
        predictions: Dict of {id: {'answer': str, 'sp': List[Tuple[str, int]]}}
        ground_truths: Dict of {id: {'answer': str, 'supporting_facts': List}}

    Returns:
        Dictionary of metrics
    """
    answer_em_scores = []
    answer_f1_scores = []
    sp_em_scores = []
    sp_f1_scores = []

    for qid, gt in ground_truths.items():
        if qid not in predictions:
            # Missing prediction - count as 0
            answer_em_scores.append(0.0)
            answer_f1_scores.append(0.0)
            sp_em_scores.append(0.0)
            sp_f1_scores.append(0.0)
            continue

        pred = predictions[qid]

        # Answer evaluation
        pred_answer = pred.get('answer', '')
        gt_answer = gt.get('answer', '')

        answer_em_scores.append(exact_match_score(pred_answer, gt_answer))
        answer_f1_scores.append(f1_score(pred_answer, gt_answer))

        # Supporting facts evaluation
        pred_sp = pred.get('sp', pred.get('supporting_facts', []))
        gt_sp = gt.get('supporting_facts', [])

        sp_em_scores.append(support_em_score(pred_sp, gt_sp))
        sp_f1_scores.append(support_f1_score(pred_sp, gt_sp))

    # Compute averages
    ans_em = sum(answer_em_scores) / len(answer_em_scores) if answer_em_scores else 0.0
    ans_f1 = sum(answer_f1_scores) / len(answer_f1_scores) if answer_f1_scores else 0.0
    sp_em = sum(sp_em_scores) / len(sp_em_scores) if sp_em_scores else 0.0
    sp_f1 = sum(sp_f1_scores) / len(sp_f1_scores) if sp_f1_scores else 0.0

    # Joint metrics (product of answer and support)
    joint_em_scores = [a * s for a, s in zip(answer_em_scores, sp_em_scores)]
    joint_f1_scores = [a * s for a, s in zip(answer_f1_scores, sp_f1_scores)]
    joint_em = sum(joint_em_scores) / len(joint_em_scores) if joint_em_scores else 0.0
    joint_f1 = sum(joint_f1_scores) / len(joint_f1_scores) if joint_f1_scores else 0.0

    return {
        'answer_em': round(ans_em, 4),
        'answer_f1': round(ans_f1, 4),
        'sp_em': round(sp_em, 4),
        'sp_f1': round(sp_f1, 4),
        'joint_em': round(joint_em, 4),
        'joint_f1': round(joint_f1, 4),
        'num_evaluated': len(ground_truths),
        'num_predictions': len(predictions)
    }


# =============================================================================
# File I/O
# =============================================================================

def read_json(file_path: str) -> Any:
    """Read JSON file."""
    with open(file_path, "r") as file:
        return json.load(file)


def load_ground_truth(file_path: str) -> Dict[str, Dict[str, Any]]:
    """Load HotpotQA ground truth file."""
    data = read_json(file_path)

    # Handle list format (raw HotpotQA)
    if isinstance(data, list):
        gt_dict = {}
        for item in data:
            qid = item.get('_id', item.get('id', ''))
            gt_dict[qid] = {
                'answer': item.get('answer', ''),
                'supporting_facts': item.get('supporting_facts', [])
            }
        return gt_dict

    # Handle dict format
    return data


def load_predictions(file_path: str) -> Dict[str, Dict[str, Any]]:
    """Load predictions file."""
    data = read_json(file_path)

    # Handle list format
    if isinstance(data, list):
        pred_dict = {}
        for item in data:
            qid = item.get('_id', item.get('id', ''))
            pred_dict[qid] = {
                'answer': item.get('answer', item.get('predicted_answer', '')),
                'sp': item.get('sp', item.get('supporting_facts', []))
            }
        return pred_dict

    # Handle dict format (official HotpotQA prediction format)
    # Format: {'answer': {id: answer}, 'sp': {id: [[title, sent_idx], ...]}}
    if 'answer' in data and 'sp' in data:
        pred_dict = {}
        for qid, answer in data['answer'].items():
            pred_dict[qid] = {
                'answer': answer,
                'sp': data['sp'].get(qid, [])
            }
        return pred_dict

    return data


# =============================================================================
# TRIDENT Results Conversion
# =============================================================================

def convert_trident_to_predictions(
    trident_results: List[Dict[str, Any]],
    ground_truth_dict: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """
    Convert TRIDENT results to HotpotQA prediction format.

    TRIDENT format:
    {
        "query_id": "...",
        "prediction": "answer",
        "selected_passages": [...],
        "abstained": false,
        ...
    }

    HotpotQA format:
    {
        id: {'answer': str, 'sp': [[title, sent_idx], ...]}
    }
    """
    predictions = {}

    for result in trident_results:
        query_id = result.get('query_id', '')

        # Try to match to ground truth ID
        hotpot_id = None

        # Direct match
        if query_id in ground_truth_dict:
            hotpot_id = query_id
        # Check if it's the original _id field preserved
        elif result.get('_id') in ground_truth_dict:
            hotpot_id = result['_id']
        else:
            # Try to extract ID from query_id patterns
            # e.g., "hotpotqa_validation_42" or "hotpot_dev_distractor_42"
            for gt_id in ground_truth_dict.keys():
                if gt_id in query_id or query_id.endswith(gt_id):
                    hotpot_id = gt_id
                    break

        if hotpot_id is None:
            # Skip if we can't match
            continue

        # Get predicted answer
        predicted_answer = result.get('prediction', '')
        if not predicted_answer:
            predicted_answer = result.get('answer', '')

        # Get predicted supporting facts
        predicted_sp = []
        passages = result.get('selected_passages', [])
        for p in passages:
            if isinstance(p, dict):
                title = p.get('title', '')
                sent_idx = p.get('sent_idx', p.get('sentence_idx', 0))
                if title:
                    predicted_sp.append([title, sent_idx])

        predictions[hotpot_id] = {
            'answer': predicted_answer,
            'sp': predicted_sp
        }

    return predictions


def load_trident_results(path: str) -> List[Dict[str, Any]]:
    """Load TRIDENT results from a JSON file or directory."""
    path = Path(path)

    if path.is_file():
        data = read_json(str(path))
        if isinstance(data, dict) and 'results' in data:
            return data['results']
        elif isinstance(data, list):
            return data
        else:
            raise ValueError(f"Unexpected format in {path}")

    elif path.is_dir():
        all_results = []
        # Look for results.json files in subdirectories
        for results_file in path.rglob("results.json"):
            data = read_json(str(results_file))
            if isinstance(data, dict) and 'results' in data:
                all_results.extend(data['results'])
            elif isinstance(data, list):
                all_results.extend(data)
        return all_results

    else:
        raise FileNotFoundError(f"Path not found: {path}")


def evaluate_trident(
    trident_path: str,
    ground_truth_path: str,
    output_predictions: Optional[str] = None
) -> Dict[str, float]:
    """
    Evaluate TRIDENT results against HotpotQA ground truth.

    Args:
        trident_path: Path to TRIDENT results (JSON file or directory)
        ground_truth_path: Path to HotpotQA ground truth JSON
        output_predictions: Optional path to save converted predictions

    Returns:
        Dictionary of metrics
    """
    print(f"Loading TRIDENT results from: {trident_path}")
    trident_results = load_trident_results(trident_path)
    print(f"Loaded {len(trident_results)} results")

    print(f"Loading ground truth from: {ground_truth_path}")
    ground_truth_dict = load_ground_truth(ground_truth_path)
    print(f"Loaded {len(ground_truth_dict)} ground truth instances")

    # Convert to HotpotQA format
    predictions = convert_trident_to_predictions(trident_results, ground_truth_dict)
    print(f"Converted {len(predictions)} predictions")

    if not predictions:
        print("WARNING: No predictions could be matched to ground truth!")
        return {"error": "No matched predictions"}

    # Save predictions if requested
    if output_predictions:
        with open(output_predictions, 'w') as f:
            json.dump(predictions, f, indent=2)
        print(f"Saved predictions to: {output_predictions}")

    # Evaluate
    metrics = evaluate(predictions, ground_truth_dict)

    return metrics


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate HotpotQA predictions (supports TRIDENT format)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard HotpotQA evaluation
  python evaluate.py predictions.json hotpotqa/data/hotpot_dev_distractor_v1.json

  # Evaluate TRIDENT results file
  python evaluate.py --trident_results results.json \\
      --ground_truth hotpotqa/data/hotpot_dev_distractor_v1.json

  # Evaluate TRIDENT results directory
  python evaluate.py --trident_dir runs/hotpotqa/results/ \\
      --ground_truth hotpotqa/data/hotpot_dev_distractor_v1.json
        """
    )

    # Standard HotpotQA evaluation arguments
    parser.add_argument(
        "filepath_with_predictions",
        type=str,
        nargs='?',
        help="JSON file with predictions"
    )
    parser.add_argument(
        "filepath_with_ground_truths",
        type=str,
        nargs='?',
        help="JSON file with ground truths"
    )

    # TRIDENT-specific arguments
    parser.add_argument(
        "--trident_results",
        type=str,
        help="Path to TRIDENT results.json file"
    )
    parser.add_argument(
        "--trident_dir",
        type=str,
        help="Path to directory containing TRIDENT shard results"
    )
    parser.add_argument(
        "--ground_truth",
        type=str,
        help="Path to HotpotQA ground truth JSON (for TRIDENT evaluation)"
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
        help="Path to save converted predictions (TRIDENT mode)"
    )

    args = parser.parse_args()

    # Determine evaluation mode
    if args.trident_results or args.trident_dir:
        # TRIDENT evaluation mode
        trident_path = args.trident_results or args.trident_dir
        gt_path = args.ground_truth

        if not gt_path:
            # Try to infer ground truth path
            gt_path = str(Path(__file__).parent / "data" / "hotpot_dev_distractor_v1.json")
            if not Path(gt_path).exists():
                parser.error("--ground_truth is required for TRIDENT evaluation")

        metrics = evaluate_trident(
            trident_path,
            gt_path,
            args.output_predictions
        )

    elif args.filepath_with_predictions and args.filepath_with_ground_truths:
        # Standard HotpotQA evaluation mode
        predictions = load_predictions(args.filepath_with_predictions)
        ground_truths = load_ground_truth(args.filepath_with_ground_truths)
        metrics = evaluate(predictions, ground_truths)

    else:
        parser.error("Provide either (predictions, ground_truth) or (--trident_results/--trident_dir, --ground_truth)")

    # Output results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    for metric, value in metrics.items():
        print(f"  {metric}: {value}")

    if args.output_filepath:
        with open(args.output_filepath, "w") as file:
            json.dump(metrics, file, indent=4)
        print(f"\nMetrics saved to: {args.output_filepath}")


if __name__ == "__main__":
    main()
