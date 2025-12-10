"""Group support sufficiency metrics for MuSiQue (handles unanswerable questions)."""

from typing import Dict, List, Set
from collections import defaultdict


class GroupSupportSufficiencyMetric:
    """
    Track grouped support sufficiency metrics.

    For questions with both answerable and unanswerable versions,
    compute joint metrics that account for correctly predicting answerability
    AND identifying correct supporting facts when answerable.
    """

    def __init__(self):
        self.question_groups: Dict[str, List[Dict]] = defaultdict(list)

    def __call__(
        self,
        predicted_support_indices: List[int],
        ground_truth_support_indices: List[int],
        predicted_sufficiency: bool,
        ground_truth_sufficiency: bool,
        question_id: str
    ) -> None:
        """Update metrics with a new prediction."""
        base_id = self._get_base_id(question_id)

        self.question_groups[base_id].append({
            'predicted_support': set(predicted_support_indices),
            'ground_truth_support': set(ground_truth_support_indices),
            'predicted_sufficiency': predicted_sufficiency,
            'ground_truth_sufficiency': ground_truth_sufficiency,
            'question_id': question_id
        })

    def _get_base_id(self, question_id: str) -> str:
        """Get base question ID."""
        return question_id

    def get_metric(self) -> Dict[str, float]:
        """Return group support sufficiency metrics."""
        if not self.question_groups:
            return {'f1': 0.0, 'em': 0.0}

        group_f1_scores = []
        group_em_scores = []

        for base_id, predictions in self.question_groups.items():
            group_score = self._compute_group_score(predictions)
            group_f1_scores.append(group_score['f1'])
            group_em_scores.append(group_score['em'])

        return {
            'f1': sum(group_f1_scores) / len(group_f1_scores) if group_f1_scores else 0.0,
            'em': sum(group_em_scores) / len(group_em_scores) if group_em_scores else 0.0
        }

    def _compute_group_score(self, predictions: List[Dict]) -> Dict[str, float]:
        """Compute score for a group of predictions."""
        scores = []

        for pred in predictions:
            sufficiency_correct = pred['predicted_sufficiency'] == pred['ground_truth_sufficiency']

            if pred['ground_truth_sufficiency']:
                # For answerable: need correct sufficiency AND correct support
                if sufficiency_correct:
                    support_f1 = self._support_f1(
                        pred['predicted_support'],
                        pred['ground_truth_support']
                    )
                    support_em = self._support_em(
                        pred['predicted_support'],
                        pred['ground_truth_support']
                    )
                else:
                    support_f1 = 0.0
                    support_em = 0.0
            else:
                # For unanswerable: only need correct sufficiency
                support_f1 = 1.0 if sufficiency_correct else 0.0
                support_em = 1.0 if sufficiency_correct else 0.0

            scores.append({'f1': support_f1, 'em': support_em})

        if not scores:
            return {'f1': 0.0, 'em': 0.0}

        return {
            'f1': sum(s['f1'] for s in scores) / len(scores),
            'em': sum(s['em'] for s in scores) / len(scores)
        }

    def _support_f1(self, pred_set: Set[int], gt_set: Set[int]) -> float:
        """Compute F1 for supporting facts."""
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

    def _support_em(self, pred_set: Set[int], gt_set: Set[int]) -> float:
        """Compute EM for supporting facts."""
        return 1.0 if pred_set == gt_set else 0.0

    def reset(self) -> None:
        """Reset all scores."""
        self.question_groups = defaultdict(list)
