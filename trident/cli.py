"""Command-line interface for TRIDENT experiments."""

from __future__ import annotations

import argparse
from typing import List

from .calibration import ReliabilityCalibrator
from .evaluation import DatasetExample, evaluate_pipeline, load_dataset_examples, simple_facet_miner
from .pipeline import TridentPipeline
from .retrieval import SimpleRetriever


def _build_retriever_from_examples(examples: List[DatasetExample]) -> SimpleRetriever:
    documents: List[str] = []
    ids: List[str] = []
    for idx, example in enumerate(examples):
        contexts = example.context or []
        if contexts:
            for jdx, ctx in enumerate(contexts):
                documents.append(ctx)
                ids.append(f"{example.dataset}-{idx}-{jdx}")
        else:
            documents.append(example.question)
            ids.append(f"{example.dataset}-{idx}-question")
    return SimpleRetriever(documents=documents, ids=ids)


def run_dataset(args: argparse.Namespace) -> None:
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")
    examples = list(load_dataset_examples(args.dataset, split=args.split, limit=args.limit))
    if not examples:
        raise RuntimeError("No examples loaded from dataset")
    retriever = _build_retriever_from_examples(examples)
    pipeline = TridentPipeline(retriever=retriever, calibrator=ReliabilityCalibrator())
    metrics = evaluate_pipeline(
        pipeline=pipeline,
        examples=examples,
        miner=simple_facet_miner,
        mode=args.mode,
    )
    print(f"Dataset: {metrics.dataset}")
    print(f"Queries evaluated: {metrics.queries}")
    print(f"Safe-Cover abstentions: {metrics.safe_abstained}")
    print(f"Average passage tokens processed: {metrics.average_tokens:.2f}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TRIDENT risk-controlled RAG pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    dataset_parser = subparsers.add_parser("dataset", help="Run evaluation over a dataset")
    dataset_parser.add_argument("--dataset", required=True, help="Dataset name (e.g., hotpot_qa)")
    dataset_parser.add_argument("--split", default="validation", help="Dataset split")
    dataset_parser.add_argument("--limit", type=int, default=10, help="Maximum number of queries")
    dataset_parser.add_argument(
        "--mode",
        choices=["safe_cover", "pareto", "both"],
        default="safe_cover",
        help="Operating mode",
    )
    dataset_parser.set_defaults(func=run_dataset)

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
