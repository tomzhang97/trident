"""Evaluation utilities for TRIDENT across QA datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, List, Optional

from .config import QueryConfig
from .facets import Facet
from .pipeline import TridentPipeline

try:  # pragma: no cover - optional dependency
    from datasets import load_dataset
except Exception:  # pragma: no cover
    load_dataset = None  # type: ignore


@dataclass
class DatasetExample:
    dataset: str
    question: str
    answer: str
    context: Optional[List[str]] = None


FacetMiner = Callable[[DatasetExample], List[Facet]]


def simple_facet_miner(example: DatasetExample) -> List[Facet]:
    """Derive a small set of facets from a dataset example."""

    return [
        Facet(
            facet_id=f"answer-{i}",
            facet_type="ENTITY",
            template={"mention": token},
        )
        for i, token in enumerate(example.answer.split())
        if token
    ]


@dataclass
class EvaluationMetrics:
    dataset: str
    queries: int
    safe_abstained: int
    average_tokens: float


def load_dataset_examples(name: str, split: str = "validation", limit: Optional[int] = None) -> Iterator[DatasetExample]:
    """Load QA dataset examples using the Hugging Face datasets API."""

    if load_dataset is None:
        raise RuntimeError("datasets library is required to load benchmark data")
    dataset = load_dataset(name, split=split)
    for idx, row in enumerate(dataset):
        question = row.get("question") or row.get("input") or row.get("query")
        answer = row.get("answer") or row.get("answers", {}).get("text", [""])[0]
        context = row.get("context")
        if question and answer:
            yield DatasetExample(dataset=name, question=question, answer=answer, context=context)
        if limit is not None and idx + 1 >= limit:
            break


def evaluate_pipeline(
    pipeline: TridentPipeline,
    examples: Iterable[DatasetExample],
    miner: FacetMiner = simple_facet_miner,
    mode: str = "safe_cover",
) -> EvaluationMetrics:
    queries = 0
    abstained = 0
    total_tokens = 0
    dataset_name = "unknown"
    for example in examples:
        dataset_name = example.dataset
        facets = miner(example)
        config = QueryConfig(query=example.question, facets=facets, mode=mode)
        output = pipeline.run(config)
        queries += 1
        if output.safe_cover and output.safe_cover.abstained:
            abstained += 1
        total_tokens += sum(p.cost for p in output.passages_examined)
    average_tokens = total_tokens / max(queries, 1)
    return EvaluationMetrics(
        dataset=dataset_name,
        queries=queries,
        safe_abstained=abstained,
        average_tokens=average_tokens,
    )
