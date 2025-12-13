"""HotpotQA dataset support for TRIDENT pipeline."""

from .data_loader import HotpotQADataLoader
from .evaluate import (
    evaluate,
    evaluate_trident,
    load_ground_truth,
    load_predictions,
    normalize_answer,
    f1_score,
    exact_match_score,
    support_f1_score,
    support_em_score,
)

__all__ = [
    "HotpotQADataLoader",
    "evaluate",
    "evaluate_trident",
    "load_ground_truth",
    "load_predictions",
    "normalize_answer",
    "f1_score",
    "exact_match_score",
    "support_f1_score",
    "support_em_score",
]
