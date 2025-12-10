"""Group answer sufficiency metrics for MuSiQue (handles unanswerable questions)."""

from typing import Dict, List, Optional
from collections import defaultdict
from .answer import f1_score, exact_match_score, metric_max_over_ground_truths


class GroupAnswerSufficiencyMetric:
    """
    Track grouped answer sufficiency metrics.

    For questions with both answerable and unanswerable versions,
    compute joint metrics that account for correctly predicting answerability
    AND providing the correct answer when answerable.
    """

    def __init__(self):
        # Store predictions grouped by question (without sufficiency suffix)
        self.question_groups: Dict[str, List[Dict]] = defaultdict(list)

    def __call__(
        self,
        predicted_answer: str,
        ground_truth_answers: List[str],
        predicted_sufficiency: bool,
        ground_truth_sufficiency: bool,
        question_id: str
    ) -> None:
        """Update metrics with a new prediction."""
        # Extract base question ID (remove any suffix like _ans or _unans)
        base_id = self._get_base_id(question_id)

        self.question_groups[base_id].append({
            'predicted_answer': predicted_answer or "",
            'ground_truth_answers': ground_truth_answers,
            'predicted_sufficiency': predicted_sufficiency,
            'ground_truth_sufficiency': ground_truth_sufficiency,
            'question_id': question_id
        })

    def _get_base_id(self, question_id: str) -> str:
        """Get base question ID by removing sufficiency-related suffixes."""
        # MuSiQue uses IDs like "2hop__123_456"
        # For full dataset, each question has 2 variants (same ID)
        return question_id

    def get_metric(self) -> Dict[str, float]:
        """
        Return group answer sufficiency metrics.

        For each question group:
        - If both variants are correctly predicted (answer + sufficiency), score = 1
        - Otherwise, use F1 of individual predictions
        """
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
        """Compute score for a group of predictions (answerable + unanswerable variants)."""
        scores = []

        for pred in predictions:
            # Check sufficiency prediction
            sufficiency_correct = pred['predicted_sufficiency'] == pred['ground_truth_sufficiency']

            if pred['ground_truth_sufficiency']:
                # For answerable questions: need correct sufficiency AND correct answer
                if sufficiency_correct:
                    answer_f1 = metric_max_over_ground_truths(
                        f1_score,
                        pred['predicted_answer'],
                        pred['ground_truth_answers']
                    )
                    answer_em = metric_max_over_ground_truths(
                        exact_match_score,
                        pred['predicted_answer'],
                        pred['ground_truth_answers']
                    )
                else:
                    answer_f1 = 0.0
                    answer_em = 0.0
            else:
                # For unanswerable questions: only need correct sufficiency
                answer_f1 = 1.0 if sufficiency_correct else 0.0
                answer_em = 1.0 if sufficiency_correct else 0.0

            scores.append({'f1': answer_f1, 'em': answer_em})

        # Return average across group
        if not scores:
            return {'f1': 0.0, 'em': 0.0}

        return {
            'f1': sum(s['f1'] for s in scores) / len(scores),
            'em': sum(s['em'] for s in scores) / len(scores)
        }

    def reset(self) -> None:
        """Reset all scores."""
        self.question_groups = defaultdict(list)
