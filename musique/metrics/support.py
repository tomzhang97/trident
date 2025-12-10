"""Support evaluation metrics for MuSiQue."""

from typing import List, Set, Tuple


class SupportMetric:
    """Track supporting facts EM and F1 metrics."""

    def __init__(self):
        self.em_scores = []
        self.f1_scores = []

    def __call__(
        self,
        predicted_support_indices: List[int],
        ground_truth_support_indices: List[int]
    ) -> None:
        """Update metrics with a new prediction."""
        pred_set: Set[int] = set(predicted_support_indices)
        gt_set: Set[int] = set(ground_truth_support_indices)

        # Exact match: predictions exactly match ground truth
        em = 1.0 if pred_set == gt_set else 0.0

        # F1 score
        if not pred_set and not gt_set:
            f1 = 1.0
        elif not pred_set or not gt_set:
            f1 = 0.0
        else:
            intersection = pred_set & gt_set
            precision = len(intersection) / len(pred_set)
            recall = len(intersection) / len(gt_set)

            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = (2 * precision * recall) / (precision + recall)

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
