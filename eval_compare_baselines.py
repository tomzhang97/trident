#!/usr/bin/env python3
"""Comparison evaluation script for TRIDENT and baseline systems."""

import argparse
import json
import os
import sys
import time
import re
import string
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

import torch
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from trident.config import TridentConfig
from trident.pipeline import TridentPipeline
from trident.llm_interface import LLMInterface
from trident.llm_instrumentation import InstrumentedLLM
from trident.retrieval import DenseRetriever, HybridRetriever
from trident.logging_utils import setup_logger

# Import baseline systems
from baselines.self_rag_system import SelfRAGSystem
from baselines.graphrag_system import GraphRAGSystem
from baselines.trident_wrapper import TridentSystemWrapper


def _normalize(text: str) -> str:
    """Normalize text for EM/F1 calculation."""
    text = text.lower().strip()
    # remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # remove articles
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    # normalize whitespace
    text = ' '.join(text.split())
    return text


def _f1(pred: str, gt: str) -> float:
    """Calculate F1 score between prediction and ground truth."""
    pred_tokens = _normalize(pred).split()
    gt_tokens = _normalize(gt).split()
    common = set(pred_tokens) & set(gt_tokens)
    if not pred_tokens or not gt_tokens:
        return 0.0
    if not common:
        return 0.0
    prec = len(common) / len(pred_tokens)
    rec = len(common) / len(gt_tokens)
    return 2 * prec * rec / (prec + rec)


@dataclass
class SystemResult:
    """Results from a single system on one example."""
    query_id: str
    question: str
    prediction: str
    ground_truth: List[str]
    tokens_used: int
    latency_ms: float
    em: float
    f1: float
    system: str
    abstained: bool
    stats: Dict[str, Any]


class BaselineComparator:
    """Main comparison orchestrator for TRIDENT and baselines."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger(args.output_dir)
        self.device = f"cuda:{args.device}" if args.device >= 0 else "cpu"

        # Load configuration
        self.config = self._load_config()

        # Initialize shared components
        self.llm = self._init_llm()
        self.instrumented_llm = InstrumentedLLM(self.llm)
        self.retriever = self._init_retriever()

        # Load data
        self.data = self._load_data()

        # Initialize systems based on which to run
        self.systems = self._init_systems()

    def _load_config(self) -> TridentConfig:
        """Load or create configuration."""
        config_path = self.args.config or "configs/default.json"

        if Path(config_path).exists():
            with open(config_path) as f:
                config_dict = json.load(f)
        else:
            # Default configuration
            config_dict = {
                "mode": "pareto",  # Default to pareto for TRIDENT
                "safe_cover": {
                    "per_facet_alpha": 0.01,
                    "token_cap": self.args.budget_tokens,
                },
                "pareto": {
                    "budget": self.args.budget_tokens,
                    "relaxed_alpha": 0.5
                },
                "llm": {
                    "model_name": self.args.model,
                    "temperature": 0.0,
                    "max_new_tokens": 256,
                    "device": self.device,
                    "load_in_8bit": self.args.load_in_8bit
                },
                "retrieval": {
                    "method": "dense",
                    "top_k": 100,
                    "encoder_model": "facebook/contriever"
                },
                "baselines": {
                    "selfrag_k": 8,
                    "selfrag_use_critic": self.args.selfrag_use_critic,
                    "graphrag_k": 20,
                    "graphrag_max_seeds": 10
                }
            }

        return TridentConfig.from_dict(config_dict)

    def _init_llm(self) -> LLMInterface:
        """Initialize the local LLM."""
        return LLMInterface(
            model_name=self.config.llm.model_name,
            device=self.device,
            temperature=self.config.llm.temperature,
            max_new_tokens=self.config.llm.max_new_tokens,
            load_in_8bit=self.config.llm.load_in_8bit,
            seed=self.args.seed
        )

    def _init_retriever(self) -> Any:
        """Initialize the retrieval system."""
        # Build corpus from data if needed
        if self.args.data_path:
            corpus_texts = []
            corpus_ids = []

            with open(self.args.data_path) as f:
                data = json.load(f)

            for idx, example in enumerate(data):
                if 'context' in example:
                    for ctx_idx, (title, sentences) in enumerate(example['context']):
                        text = " ".join(sentences) if isinstance(sentences, list) else sentences
                        corpus_texts.append(text)
                        corpus_ids.append(f"doc_{idx}_{ctx_idx}_{title}")

            # Create retriever
            retriever = DenseRetriever(
                encoder_model=self.config.retrieval.encoder_model,
                device=self.device,
                top_k=self.config.retrieval.top_k
            )

            # Load corpus
            retriever.corpus = corpus_texts
            retriever.build_index()

            return retriever
        return None

    def _load_data(self) -> List[Dict[str, Any]]:
        """Load evaluation data."""
        with open(self.args.data_path) as f:
            data = json.load(f)

        # Limit to max_examples if specified
        if self.args.max_examples > 0:
            data = data[:self.args.max_examples]

        return data

    def _init_systems(self) -> Dict[str, Any]:
        """Initialize all systems to evaluate."""
        systems = {}

        # Determine which systems to run
        systems_to_run = self.args.systems.split(',') if self.args.systems else ['all']
        if 'all' in systems_to_run:
            systems_to_run = ['trident_pareto', 'trident_safe_cover', 'self_rag', 'graphrag']

        # Initialize TRIDENT variants
        if any('trident' in s for s in systems_to_run):
            pipeline = TridentPipeline(
                config=self.config,
                llm=self.llm,
                retriever=self.retriever,
                device=self.device
            )

            if 'trident_pareto' in systems_to_run:
                systems['trident_pareto'] = TridentSystemWrapper(pipeline, mode='pareto')

            if 'trident_safe_cover' in systems_to_run:
                systems['trident_safe_cover'] = TridentSystemWrapper(pipeline, mode='safe_cover')

        # Initialize Self-RAG
        if 'self_rag' in systems_to_run:
            systems['self_rag'] = SelfRAGSystem(
                llm=self.instrumented_llm,
                retriever=self.retriever,
                k=self.config.baselines.selfrag_k,
                use_critic=self.config.baselines.selfrag_use_critic
            )

        # Initialize GraphRAG
        if 'graphrag' in systems_to_run:
            systems['graphrag'] = GraphRAGSystem(
                llm=self.instrumented_llm,
                retriever=self.retriever,
                k=self.config.baselines.graphrag_k,
                topk_nodes=self.config.baselines.graphrag_topk_nodes,
                max_seeds=self.config.baselines.graphrag_max_seeds
            )

        return systems

    def evaluate_system(
        self,
        name: str,
        system: Any
    ) -> List[SystemResult]:
        """Evaluate a single system on all examples."""
        self.logger.info(f"Evaluating {name}...")
        results = []

        for idx, example in enumerate(self.data):
            query_id = example.get('_id', str(idx))
            question = example['question']
            ground_truth = [example['answer']] if isinstance(example['answer'], str) else example['answer']

            self.logger.info(f"[{name}] {idx+1}/{len(self.data)}: {query_id}")

            try:
                # Run system
                output = system.answer(
                    question=question,
                    context=example.get('context'),
                    supporting_facts=example.get('supporting_facts')
                )

                # Calculate metrics
                prediction = output['answer']
                em_scores = [1.0 if _normalize(prediction) == _normalize(gt) else 0.0 for gt in ground_truth]
                f1_scores = [_f1(prediction, gt) for gt in ground_truth]

                result = SystemResult(
                    query_id=query_id,
                    question=question,
                    prediction=prediction,
                    ground_truth=ground_truth,
                    tokens_used=output.get('tokens_used', 0),
                    latency_ms=output.get('latency_ms', 0.0),
                    em=max(em_scores),
                    f1=max(f1_scores),
                    system=name,
                    abstained=output.get('abstained', False),
                    stats=output.get('stats', {})
                )

                results.append(result)

            except Exception as e:
                self.logger.error(f"[{name}] Error on {query_id}: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())

                # Add failed result
                results.append(SystemResult(
                    query_id=query_id,
                    question=question,
                    prediction="ERROR",
                    ground_truth=ground_truth,
                    tokens_used=0,
                    latency_ms=0.0,
                    em=0.0,
                    f1=0.0,
                    system=name,
                    abstained=True,
                    stats={'error': str(e)}
                ))

        return results

    def compute_summary(self, results: List[SystemResult]) -> Dict[str, Any]:
        """Compute summary statistics for a system."""
        valid_results = [r for r in results if not r.abstained]

        if not valid_results:
            return {
                'num_examples': len(results),
                'num_valid': 0,
                'error': 'No valid results'
            }

        return {
            'system': results[0].system,
            'num_examples': len(results),
            'num_valid': len(valid_results),
            'num_abstained': len(results) - len(valid_results),
            'abstention_rate': (len(results) - len(valid_results)) / len(results),
            'avg_em': np.mean([r.em for r in valid_results]),
            'avg_f1': np.mean([r.f1 for r in valid_results]),
            'avg_tokens': np.mean([r.tokens_used for r in valid_results]),
            'median_tokens': np.median([r.tokens_used for r in valid_results]),
            'total_tokens': sum(r.tokens_used for r in valid_results),
            'avg_latency_ms': np.mean([r.latency_ms for r in valid_results]),
            'median_latency_ms': np.median([r.latency_ms for r in valid_results]),
        }

    def run_comparison(self) -> None:
        """Run full comparison across all systems."""
        all_results = {}
        all_summaries = []

        for name, system in self.systems.items():
            # Evaluate system
            results = self.evaluate_system(name, system)

            # Save detailed results
            results_file = self.output_dir / f"{name}_results.json"
            with open(results_file, 'w') as f:
                json.dump([asdict(r) for r in results], f, indent=2)

            self.logger.info(f"Saved {name} results to {results_file}")

            # Compute summary
            summary = self.compute_summary(results)
            all_summaries.append(summary)
            all_results[name] = results

        # Save summary
        summary_file = self.output_dir / "token_usage_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(all_summaries, f, indent=2)

        # Print comparison table
        self._print_comparison_table(all_summaries)

        self.logger.info(f"Comparison complete! Results saved to {self.output_dir}")

    def _print_comparison_table(self, summaries: List[Dict[str, Any]]) -> None:
        """Print a formatted comparison table."""
        print("\n" + "="*100)
        print("TOKEN USAGE COMPARISON")
        print("="*100)
        print(f"{'System':<25} {'Avg Tokens':<12} {'Med Tokens':<12} {'Avg Latency':<15} {'EM':<8} {'F1':<8}")
        print("-"*100)

        for s in summaries:
            print(
                f"{s['system']:<25} "
                f"{s['avg_tokens']:<12.1f} "
                f"{s['median_tokens']:<12.1f} "
                f"{s['avg_latency_ms']:<15.1f} "
                f"{s['avg_em']:<8.3f} "
                f"{s['avg_f1']:<8.3f}"
            )

        print("="*100 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Compare TRIDENT with baseline systems")

    # Required arguments
    parser.add_argument("--data_path", type=str, required=True, help="Path to evaluation data")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")

    # System selection
    parser.add_argument(
        "--systems",
        type=str,
        default="all",
        help="Comma-separated list of systems to run (trident_pareto,trident_safe_cover,self_rag,graphrag) or 'all'"
    )

    # Model configuration
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf", help="LLM model name")
    parser.add_argument("--device", type=int, default=0, help="CUDA device (-1 for CPU)")
    parser.add_argument("--load_in_8bit", action="store_true", help="Load model in 8-bit mode")

    # Pipeline configuration
    parser.add_argument("--budget_tokens", type=int, default=2000, help="Token budget for TRIDENT")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Baseline-specific options
    parser.add_argument("--selfrag_use_critic", action="store_true", help="Use critic in Self-RAG")

    # Evaluation options
    parser.add_argument("--max_examples", type=int, default=0, help="Maximum examples to evaluate (0 for all)")
    parser.add_argument("--config", type=str, help="Path to configuration file")

    args = parser.parse_args()

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Run comparison
    comparator = BaselineComparator(args)
    comparator.run_comparison()


if __name__ == "__main__":
    main()
