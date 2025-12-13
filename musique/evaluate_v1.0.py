#!/usr/bin/env python3
"""
MuSiQue v1.0 Evaluation Script (Modified for TRIDENT)

This is the official MuSiQue evaluation script, modified to:
1. Be self-contained (all metrics inline, no external dependencies)
2. Support TRIDENT experiment results format
3. Provide flexible evaluation modes

Original: https://github.com/stonybrooknlp/musique

Usage:
    # Standard MuSiQue evaluation
    python evaluate_v1.0.py predictions.jsonl ground_truth.jsonl

    # Evaluate TRIDENT results
    python evaluate_v1.0.py --trident_results results.json --ground_truth musique_ans_v1.0_dev.jsonl

    # Evaluate directory of shard results
    python evaluate_v1.0.py --trident_dir runs/musique/results/ --ground_truth musique_ans_v1.0_dev.jsonl
"""

import json
import argparse
import re
import string
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set, Optional
from collections import Counter, defaultdict


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


def metric_max_over_ground_truths(metric_fn, prediction: str, ground_truths: List[str]) -> float:
    """Compute the max metric over all ground truths."""
    if not ground_truths:
        return 0.0
    scores = [metric_fn(prediction, gt) for gt in ground_truths]
    return max(scores)


# =============================================================================
# Metric Classes
# =============================================================================

class AnswerMetric:
    """Track answer EM and F1 metrics."""

    def __init__(self):
        self.em_scores = []
        self.f1_scores = []

    def __call__(self, predicted_answer: str, ground_truth_answers: List[str]) -> None:
        if not predicted_answer:
            predicted_answer = ""

        em = metric_max_over_ground_truths(exact_match_score, predicted_answer, ground_truth_answers)
        f1 = metric_max_over_ground_truths(f1_score, predicted_answer, ground_truth_answers)

        self.em_scores.append(em)
        self.f1_scores.append(f1)

    def get_metric(self) -> Tuple[float, float]:
        """Return (EM, F1) averages."""
        if not self.em_scores:
            return (0.0, 0.0)
        return (
            sum(self.em_scores) / len(self.em_scores),
            sum(self.f1_scores) / len(self.f1_scores)
        )


class SupportMetric:
    """Track supporting facts EM and F1 metrics."""

    def __init__(self):
        self.em_scores = []
        self.f1_scores = []

    def __call__(self, predicted_indices: List[int], ground_truth_indices: List[int]) -> None:
        pred_set: Set[int] = set(predicted_indices)
        gt_set: Set[int] = set(ground_truth_indices)

        em = 1.0 if pred_set == gt_set else 0.0

        if not pred_set and not gt_set:
            f1 = 1.0
        elif not pred_set or not gt_set:
            f1 = 0.0
        else:
            intersection = pred_set & gt_set
            precision = len(intersection) / len(pred_set)
            recall = len(intersection) / len(gt_set)
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        self.em_scores.append(em)
        self.f1_scores.append(f1)

    def get_metric(self) -> Tuple[float, float]:
        if not self.em_scores:
            return (0.0, 0.0)
        return (
            sum(self.em_scores) / len(self.em_scores),
            sum(self.f1_scores) / len(self.f1_scores)
        )


class GroupAnswerSufficiencyMetric:
    """Track grouped answer sufficiency metrics for unanswerable questions."""

    def __init__(self):
        self.question_groups: Dict[str, List[Dict]] = defaultdict(list)

    def __call__(
        self,
        predicted_answer: str,
        ground_truth_answers: List[str],
        predicted_sufficiency: bool,
        ground_truth_sufficiency: bool,
        question_id: str
    ) -> None:
        self.question_groups[question_id].append({
            'predicted_answer': predicted_answer or "",
            'ground_truth_answers': ground_truth_answers,
            'predicted_sufficiency': predicted_sufficiency,
            'ground_truth_sufficiency': ground_truth_sufficiency,
        })

    def get_metric(self) -> Dict[str, float]:
        if not self.question_groups:
            return {'f1': 0.0, 'em': 0.0}

        group_scores = []
        for predictions in self.question_groups.values():
            scores = []
            for pred in predictions:
                sufficiency_correct = pred['predicted_sufficiency'] == pred['ground_truth_sufficiency']
                if pred['ground_truth_sufficiency']:
                    if sufficiency_correct:
                        score = metric_max_over_ground_truths(
                            f1_score, pred['predicted_answer'], pred['ground_truth_answers']
                        )
                    else:
                        score = 0.0
                else:
                    score = 1.0 if sufficiency_correct else 0.0
                scores.append(score)
            group_scores.append(sum(scores) / len(scores) if scores else 0.0)

        return {'f1': sum(group_scores) / len(group_scores) if group_scores else 0.0}


class GroupSupportSufficiencyMetric:
    """Track grouped support sufficiency metrics for unanswerable questions."""

    def __init__(self):
        self.question_groups: Dict[str, List[Dict]] = defaultdict(list)

    def __call__(
        self,
        predicted_indices: List[int],
        ground_truth_indices: List[int],
        predicted_sufficiency: bool,
        ground_truth_sufficiency: bool,
        question_id: str
    ) -> None:
        self.question_groups[question_id].append({
            'predicted_support': set(predicted_indices),
            'ground_truth_support': set(ground_truth_indices),
            'predicted_sufficiency': predicted_sufficiency,
            'ground_truth_sufficiency': ground_truth_sufficiency,
        })

    def get_metric(self) -> Dict[str, float]:
        if not self.question_groups:
            return {'f1': 0.0}

        group_scores = []
        for predictions in self.question_groups.values():
            scores = []
            for pred in predictions:
                sufficiency_correct = pred['predicted_sufficiency'] == pred['ground_truth_sufficiency']
                if pred['ground_truth_sufficiency']:
                    if sufficiency_correct:
                        pred_set = pred['predicted_support']
                        gt_set = pred['ground_truth_support']
                        if not pred_set and not gt_set:
                            score = 1.0
                        elif not pred_set or not gt_set:
                            score = 0.0
                        else:
                            intersection = pred_set & gt_set
                            precision = len(intersection) / len(pred_set)
                            recall = len(intersection) / len(gt_set)
                            score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                    else:
                        score = 0.0
                else:
                    score = 1.0 if sufficiency_correct else 0.0
                scores.append(score)
            group_scores.append(sum(scores) / len(scores) if scores else 0.0)

        return {'f1': sum(group_scores) / len(group_scores) if group_scores else 0.0}


# =============================================================================
# File I/O
# =============================================================================

def read_jsonl(file_path: str) -> List[Dict]:
    """Read JSONL file."""
    with open(file_path, "r") as file:
        instances = [json.loads(line.strip()) for line in file if line.strip()]
    return instances


def read_json(file_path: str) -> Any:
    """Read JSON file."""
    with open(file_path, "r") as file:
        return json.load(file)


# =============================================================================
# TRIDENT Results Conversion
# =============================================================================

def convert_trident_to_predictions(
    trident_results: List[Dict[str, Any]],
    ground_truth_instances: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Convert TRIDENT results to MuSiQue prediction format.

    TRIDENT format:
    {
        "query_id": "...",
        "prediction": "answer",
        "selected_passages": [...],
        "abstained": false,
        ...
    }

    MuSiQue format:
    {
        "id": "...",
        "predicted_answer": "answer",
        "predicted_support_idxs": [...],
        "predicted_answerable": true
    }
    """
    # Build lookup from ground truth IDs
    gt_by_id = {gt['id']: gt for gt in ground_truth_instances}

    # Also build lookup by index position for fallback matching
    gt_by_idx = {i: gt for i, gt in enumerate(ground_truth_instances)}

    predictions = []

    for result in trident_results:
        query_id = result.get('query_id', '')

        # Try to match to ground truth ID
        musique_id = None

        # Direct match
        if query_id in gt_by_id:
            musique_id = query_id
        # Check if it's the original _id field preserved
        elif result.get('_id') in gt_by_id:
            musique_id = result['_id']
        else:
            # Try to extract index from query_id like "musique_ans_dev_42"
            parts = query_id.split('_')
            if len(parts) >= 4 and parts[-1].isdigit():
                idx = int(parts[-1])
                if idx in gt_by_idx:
                    musique_id = gt_by_idx[idx]['id']

        if musique_id is None:
            # Skip if we can't match
            continue

        # Get predicted answer
        predicted_answer = result.get('prediction', '')
        if not predicted_answer:
            predicted_answer = result.get('answer', '')

        # Get predicted support indices
        predicted_support = []
        passages = result.get('selected_passages', [])
        for p in passages:
            if isinstance(p, dict):
                if 'idx' in p:
                    predicted_support.append(p['idx'])
                elif 'paragraph_idx' in p:
                    predicted_support.append(p['paragraph_idx'])

        # Determine answerability
        predicted_answerable = not result.get('abstained', False)

        predictions.append({
            'id': musique_id,
            'predicted_answer': predicted_answer,
            'predicted_support_idxs': predicted_support,
            'predicted_answerable': predicted_answerable
        })

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


# =============================================================================
# Main Evaluation Functions
# =============================================================================

def evaluate(
    filepath_with_predictions: str,
    filepath_with_ground_truths: str,
    lenient: bool = False
) -> Dict[str, float]:
    """
    Official MuSiQue evaluation.

    Args:
        filepath_with_predictions: Path to predictions JSONL
        filepath_with_ground_truths: Path to ground truth JSONL
        lenient: If True, allow mismatched lengths (skip unmatched)

    Returns:
        Dictionary of metrics
    """
    prediction_instances = read_jsonl(filepath_with_predictions)
    ground_truth_instances = read_jsonl(filepath_with_ground_truths)

    # Build prediction lookup
    pred_by_id = {p['id']: p for p in prediction_instances}

    if not lenient:
        assert len(prediction_instances) == len(ground_truth_instances), \
            f"Prediction count ({len(prediction_instances)}) != ground truth count ({len(ground_truth_instances)})"

    do_sufficiency_eval = False
    answer_metric = AnswerMetric()
    support_metric = SupportMetric()
    group_answer_sufficiency_metric = GroupAnswerSufficiencyMetric()
    group_support_sufficiency_metric = GroupSupportSufficiencyMetric()

    matched_count = 0

    for ground_truth_instance in ground_truth_instances:
        question_id = ground_truth_instance["id"]

        if question_id not in pred_by_id:
            if lenient:
                continue
            else:
                raise AssertionError(f"Missing prediction for ID: {question_id}")

        prediction_instance = pred_by_id[question_id]
        matched_count += 1

        predicted_answer = prediction_instance.get("predicted_answer", "")
        ground_truth_answers = [ground_truth_instance["answer"]] + ground_truth_instance.get("answer_aliases", [])

        predicted_support_indices = prediction_instance.get("predicted_support_idxs", [])
        ground_truth_support_indices = [
            paragraph["idx"]
            for paragraph in ground_truth_instance["paragraphs"]
            if paragraph["is_supporting"]
        ]

        predicted_sufficiency = prediction_instance.get("predicted_answerable", True)
        ground_truth_sufficiency = ground_truth_instance.get("answerable", True)

        if ground_truth_sufficiency:
            answer_metric(predicted_answer, ground_truth_answers)
            support_metric(predicted_support_indices, ground_truth_support_indices)

        group_answer_sufficiency_metric(
            predicted_answer,
            ground_truth_answers,
            predicted_sufficiency,
            ground_truth_sufficiency,
            question_id,
        )
        group_support_sufficiency_metric(
            predicted_support_indices,
            ground_truth_support_indices,
            predicted_sufficiency,
            ground_truth_sufficiency,
            question_id,
        )

        if not ground_truth_sufficiency:
            do_sufficiency_eval = True

    metrics = {
        "answer_f1": round(answer_metric.get_metric()[1], 3),
        "answer_em": round(answer_metric.get_metric()[0], 3),
        "support_f1": round(support_metric.get_metric()[1], 3),
        "num_evaluated": matched_count,
    }

    if do_sufficiency_eval:
        metrics["group_answer_sufficiency_f1"] = round(
            group_answer_sufficiency_metric.get_metric()["f1"], 3
        )
        metrics["group_support_sufficiency_f1"] = round(
            group_support_sufficiency_metric.get_metric()["f1"], 3
        )

    return metrics


def evaluate_trident(
    trident_path: str,
    ground_truth_path: str,
    output_predictions: Optional[str] = None,
    lenient: bool = True
) -> Dict[str, float]:
    """
    Evaluate TRIDENT results against MuSiQue ground truth.

    Args:
        trident_path: Path to TRIDENT results (JSON file or directory)
        ground_truth_path: Path to MuSiQue ground truth JSONL
        output_predictions: Optional path to save converted predictions
        lenient: If True, allow partial matching

    Returns:
        Dictionary of metrics
    """
    print(f"Loading TRIDENT results from: {trident_path}")
    trident_results = load_trident_results(trident_path)
    print(f"Loaded {len(trident_results)} results")

    print(f"Loading ground truth from: {ground_truth_path}")
    ground_truth_instances = read_jsonl(ground_truth_path)
    print(f"Loaded {len(ground_truth_instances)} ground truth instances")

    # Convert to MuSiQue format
    predictions = convert_trident_to_predictions(trident_results, ground_truth_instances)
    print(f"Converted {len(predictions)} predictions")

    if not predictions:
        print("WARNING: No predictions could be matched to ground truth!")
        return {"error": "No matched predictions"}

    # Save predictions if requested
    if output_predictions:
        with open(output_predictions, 'w') as f:
            for pred in predictions:
                f.write(json.dumps(pred) + '\n')
        print(f"Saved predictions to: {output_predictions}")

    # Create temp file for evaluation
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for pred in predictions:
            f.write(json.dumps(pred) + '\n')
        temp_path = f.name

    try:
        metrics = evaluate(temp_path, ground_truth_path, lenient=lenient)
    finally:
        import os
        os.unlink(temp_path)

    return metrics


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MuSiQue predictions (supports TRIDENT format)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard MuSiQue evaluation
  python evaluate_v1.0.py predictions.jsonl musique/data/musique_ans_v1.0_dev.jsonl

  # Evaluate TRIDENT results file
  python evaluate_v1.0.py --trident_results results.json \\
      --ground_truth musique/data/musique_ans_v1.0_dev.jsonl

  # Evaluate TRIDENT results directory
  python evaluate_v1.0.py --trident_dir runs/musique/results/ \\
      --ground_truth musique/data/musique_ans_v1.0_dev.jsonl
        """
    )

    # Standard MuSiQue evaluation arguments
    parser.add_argument(
        "filepath_with_predictions",
        type=str,
        nargs='?',
        help="JSONL file with predictions (MuSiQue format)"
    )
    parser.add_argument(
        "filepath_with_ground_truths",
        type=str,
        nargs='?',
        help="JSONL file with ground truths"
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
        help="Path to MuSiQue ground truth JSONL (for TRIDENT evaluation)"
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

    # Evaluation options
    parser.add_argument(
        "--lenient",
        action="store_true",
        help="Allow partial matching (skip unmatched predictions)"
    )

    args = parser.parse_args()

    # Determine evaluation mode
    if args.trident_results or args.trident_dir:
        # TRIDENT evaluation mode
        trident_path = args.trident_results or args.trident_dir
        gt_path = args.ground_truth

        if not gt_path:
            # Try to infer ground truth path
            gt_path = str(Path(__file__).parent / "data" / "musique_ans_v1.0_dev.jsonl")
            if not Path(gt_path).exists():
                parser.error("--ground_truth is required for TRIDENT evaluation")

        metrics = evaluate_trident(
            trident_path,
            gt_path,
            args.output_predictions,
            lenient=args.lenient or True
        )

    elif args.filepath_with_predictions and args.filepath_with_ground_truths:
        # Standard MuSiQue evaluation mode
        metrics = evaluate(
            args.filepath_with_predictions,
            args.filepath_with_ground_truths,
            lenient=args.lenient
        )

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