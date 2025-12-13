"""Unified interface for full baseline systems (GraphRAG, Self-RAG, KET-RAG)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import time


@dataclass
class BaselineResponse:
    """Standardized response from any baseline system."""

    answer: str
    tokens_used: int  # Total tokens including indexing/overhead
    latency_ms: float  # Total latency including indexing
    selected_passages: List[Dict[str, Any]]
    abstained: bool
    mode: str  # 'graphrag', 'selfrag', 'ketrag', 'trident'

    # Detailed stats
    stats: Dict[str, Any]  # System-specific statistics

    # Optional debugging information
    raw_answer: Optional[str] = None  # Unprocessed model output
    extracted_answer: Optional[str] = None  # Post-processed answer

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "answer": self.answer,
            "tokens_used": self.tokens_used,
            "latency_ms": self.latency_ms,
            "selected_passages": self.selected_passages,
            "abstained": self.abstained,
            "mode": self.mode,
            "stats": self.stats,
            "raw_answer": self.raw_answer,
            "extracted_answer": self.extracted_answer,
        }


class BaselineSystem(ABC):
    """
    Abstract base class for all baseline systems.

    Provides a unified interface for:
    - GraphRAG (Microsoft)
    - Self-RAG (Asai et al.)
    - KET-RAG
    - TRIDENT (for comparison)
    """

    def __init__(self, name: str, **kwargs):
        """
        Initialize baseline system.

        Args:
            name: System name ('graphrag', 'selfrag', 'ketrag', 'trident')
            **kwargs: System-specific configuration
        """
        self.name = name
        self.config = kwargs

    @abstractmethod
    def answer(
        self,
        question: str,
        context: Optional[List[List[str]]] = None,
        supporting_facts: Optional[List[tuple]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> BaselineResponse:
        """
        Answer a question using the baseline system.

        Args:
            question: The question to answer
            context: Optional pre-provided context in HotpotQA format:
                     List of [title, sentences] pairs
            supporting_facts: Optional supporting facts (for interface compatibility)
            metadata: Optional metadata (question_id, type, etc.)

        Returns:
            BaselineResponse with answer, metrics, and stats
        """
        pass

    def batch_answer(
        self,
        questions: List[str],
        contexts: Optional[List[List[List[str]]]] = None,
        supporting_facts: Optional[List[List[tuple]]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> List[BaselineResponse]:
        """
        Answer multiple questions in batch.

        Default implementation processes sequentially.
        Subclasses can override for true batch processing.

        Args:
            questions: List of questions
            contexts: Optional list of contexts (one per question)
            supporting_facts: Optional list of supporting facts
            metadata: Optional list of metadata dicts

        Returns:
            List of BaselineResponse objects
        """
        results = []
        n = len(questions)

        for i in range(n):
            ctx = contexts[i] if contexts else None
            sf = supporting_facts[i] if supporting_facts else None
            meta = metadata[i] if metadata else None

            result = self.answer(
                question=questions[i],
                context=ctx,
                supporting_facts=sf,
                metadata=meta
            )
            results.append(result)

        return results

    @abstractmethod
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get system information and configuration.

        Returns:
            Dictionary with system name, version, config, etc.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class TokenTracker:
    """Helper class for tracking token usage across multiple LLM calls."""

    def __init__(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.num_calls = 0
        self.call_details = []

    def add_call(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        purpose: str = "unknown"
    ):
        """Record a single LLM call."""
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_tokens += (prompt_tokens + completion_tokens)
        self.num_calls += 1

        self.call_details.append({
            "purpose": purpose,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        })

    def get_stats(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "num_calls": self.num_calls,
            "call_details": self.call_details,
        }

    def reset(self):
        """Reset all counters."""
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.num_calls = 0
        self.call_details = []


class LatencyTracker:
    """Helper class for tracking latency across operations."""

    def __init__(self):
        self.start_time = None
        self.total_latency_ms = 0.0
        self.operation_latencies = []

    def start(self):
        """Start timing."""
        self.start_time = time.time()

    def stop(self, operation: str = "unknown"):
        """Stop timing and record latency."""
        if self.start_time is None:
            return 0.0

        elapsed_ms = (time.time() - self.start_time) * 1000
        self.total_latency_ms += elapsed_ms
        self.operation_latencies.append({
            "operation": operation,
            "latency_ms": elapsed_ms,
        })
        self.start_time = None
        return elapsed_ms

    def get_total_latency(self) -> float:
        """Get total latency in milliseconds."""
        return self.total_latency_ms

    def get_stats(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            "total_latency_ms": self.total_latency_ms,
            "operation_latencies": self.operation_latencies,
        }

    def reset(self):
        """Reset all trackers."""
        self.start_time = None
        self.total_latency_ms = 0.0
        self.operation_latencies = []


def normalize_answer(s: str) -> str:
    """
    Normalize answer for evaluation.

    - Lowercase
    - Remove punctuation
    - Remove articles (a, an, the)
    - Remove extra whitespace
    """
    import string
    import re

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


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """Compute exact match score."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Compute F1 score between prediction and ground truth."""
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()

    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return float(len(pred_tokens) == len(truth_tokens))

    common_tokens = set(pred_tokens) & set(truth_tokens)
    num_common = sum(min(pred_tokens.count(t), truth_tokens.count(t)) for t in common_tokens)

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(truth_tokens)
    f1 = 2 * precision * recall / (precision + recall)

    return f1