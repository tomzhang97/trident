"""Answer evaluation metrics for MuSiQue."""

import re
import string
from collections import Counter
from typing import List, Tuple


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
    scores = [metric_fn(prediction, gt) for gt in ground_truths]
    return max(scores) if scores else 0.0


class AnswerMetric:
    """Track answer EM and F1 metrics."""

    def __init__(self):
        self.em_scores = []
        self.f1_scores = []

    def __call__(self, predicted_answer: str, ground_truth_answers: List[str]) -> None:
        """Update metrics with a new prediction."""
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

        avg_em = sum(self.em_scores) / len(self.em_scores)
        avg_f1 = sum(self.f1_scores) / len(self.f1_scores)

        return (avg_em, avg_f1)

    def reset(self) -> None:
        """Reset all scores."""
        self.em_scores = []
        self.f1_scores = []
