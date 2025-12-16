#!/usr/bin/env python3
"""Main experiment runner for TRIDENT evaluation with local LLMs.

Supports:
- Single worker mode (--worker): Process a single data shard
- Multi-GPU mode (--num_gpus N): Automatically shard data and run in parallel
- Both JSON and JSONL data formats (for HotpotQA and MuSiQue)

Usage:
    # Single GPU worker mode
    python eval_complete_runnable.py --worker --data_path data.json --output_dir results/

    # Multi-GPU parallel mode (automatically shards and distributes)
    python eval_complete_runnable.py --data_path data.jsonl --output_dir results/ --num_gpus 4

    # With MuSiQue JSONL format
    python eval_complete_runnable.py --data_path musique_ans_v1.0_dev.jsonl --output_dir results/ --num_gpus 4
"""

import argparse
import json
import os
import sys
import time
import traceback
import re
import string
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
from threading import Lock

import importlib
import importlib.util
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trident.config import TridentConfig, ExperimentConfig, ParetoConfig, SafeCoverConfig
from trident.config_families import get_config, get_selfrag_config, ALL_CONFIGS, SELFRAG_CONFIGS
from trident.pipeline import TridentPipeline
from trident.evaluation import BenchmarkEvaluator, DatasetLoader
from trident.llm_interface import LLMInterface
from trident.llm_instrumentation import InstrumentedLLM
from trident.retrieval import DenseRetriever, HybridRetriever, BM25Retriever
from trident.logging_utils import setup_logger, log_metrics

# Import baseline systems
from baselines.self_rag_system import SelfRAGSystem
from baselines.graphrag_system import GraphRAGSystem
from baselines.ketrag_system import KETRAGSystem
from baselines.trident_wrapper import TridentSystemWrapper


@dataclass
class WorkerResult:
    """Results from a single worker evaluation."""
    query_id: str
    question: str
    prediction: str
    ground_truth: List[str]
    selected_passages: List[Dict[str, Any]]
    certificates: List[Dict[str, Any]]
    metrics: Dict[str, float]
    abstained: bool
    latency_ms: float
    tokens_used: int
    mode: str


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


def load_data(data_path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load data from JSON or JSONL file.

    Supports:
    - JSON: List of examples or HotpotQA format
    - JSONL: One JSON object per line (MuSiQue format)

    Automatically converts to standard format with _id, question, answer, context, supporting_facts.
    """
    path = Path(data_path)

    if path.suffix == '.jsonl':
        # JSONL format (MuSiQue)
        examples = []
        with open(path, 'r') as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))
                    if limit and len(examples) >= limit:
                        break
        # Convert MuSiQue format to standard format
        return convert_musique_format(examples)
    else:
        # JSON format (HotpotQA or pre-processed)
        with open(path, 'r') as f:
            data = json.load(f)

        if isinstance(data, list):
            examples = data[:limit] if limit else data
        else:
            examples = [data]

        # Check if it's raw HotpotQA format and convert
        if examples and '_id' not in examples[0] and 'id' in examples[0]:
            return convert_hotpotqa_format(examples)

        return examples


def convert_musique_format(examples: List[Dict]) -> List[Dict]:
    """Convert MuSiQue JSONL format to standard format."""
    converted = []
    for ex in examples:
        # Extract context from paragraphs
        context = []
        supporting_facts = []

        for para in ex.get('paragraphs', []):
            title = para.get('title', f"para_{para.get('idx', 0)}")
            text = para.get('paragraph_text', '')
            # Split into sentences
            sentences = re.split(r'(?<=[.!?])\s+', text.strip())
            sentences = [s.strip() for s in sentences if s.strip()]
            context.append((title, sentences))

            # Track supporting paragraphs
            if para.get('is_supporting', False):
                for sent_idx in range(len(sentences)):
                    supporting_facts.append((title, sent_idx))

        converted.append({
            '_id': ex.get('id', ''),
            'question': ex.get('question', ''),
            'answer': ex.get('answer', ''),
            'answer_aliases': ex.get('answer_aliases', []),
            'context': context,
            'supporting_facts': supporting_facts,
            'type': ex.get('id', '').split('__')[0] if '__' in ex.get('id', '') else 'unknown',
            'answerable': ex.get('answerable', True),
        })
    return converted


def convert_hotpotqa_format(examples: List[Dict]) -> List[Dict]:
    """Convert raw HotpotQA JSON format to standard format."""
    converted = []
    for ex in examples:
        # Handle context format
        context = ex.get('context', [])
        if isinstance(context, dict):
            # HuggingFace format
            titles = context.get('title', [])
            sentences_list = context.get('sentences', [])
            context = list(zip(titles, sentences_list))

        # Handle supporting_facts format
        sf = ex.get('supporting_facts', [])
        if isinstance(sf, dict):
            sf_titles = sf.get('title', [])
            sf_sents = sf.get('sent_id', [])
            sf = list(zip(sf_titles, sf_sents))

        converted.append({
            '_id': ex.get('_id', ex.get('id', '')),
            'question': ex.get('question', ''),
            'answer': ex.get('answer', ''),
            'context': context,
            'supporting_facts': sf,
            'type': ex.get('type', 'unknown'),
            'level': ex.get('level', 'unknown'),
        })
    return converted


def save_shard(examples: List[Dict], output_path: str) -> None:
    """Save examples as a JSON shard."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(examples, f, indent=2)


def create_shards(
    examples: List[Dict],
    output_dir: str,
    shard_size: int = 100,
    prefix: str = 'shard'
) -> List[str]:
    """Create data shards for parallel processing."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    shard_paths = []
    for i in range(0, len(examples), shard_size):
        shard = examples[i:i + shard_size]
        start_idx = i
        end_idx = min(i + shard_size, len(examples)) - 1

        shard_name = f"{prefix}_{start_idx}_{end_idx}.json"
        shard_path = output_path / shard_name
        save_shard(shard, str(shard_path))
        shard_paths.append(str(shard_path))

    return shard_paths


class ExperimentRunner:
    """Main experiment orchestrator."""
    
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.logger = setup_logger(args.output_dir)
        self.device = f"cuda:{args.device}" if args.device >= 0 else "cpu"

        # Load configuration
        self.logger.info("Step 1/5: Loading configuration...")
        self.config = self._load_config()

        # Initialize components
        self.logger.info("Step 2/5: Loading LLM model (this may take a few minutes)...")
        self.llm = self._init_llm()
        self.logger.info("Step 2/5: LLM loaded successfully")

        self.instrumented_llm = InstrumentedLLM(self.llm)

        self.logger.info("Step 3/5: Initializing retriever and building index...")
        self.retriever = self._init_retriever()
        self.logger.info("Step 3/5: Retriever ready")

        # Initialize the appropriate system based on mode
        self.logger.info("Step 4/5: Initializing system...")
        self.system = self._init_system()
        self.logger.info("Step 4/5: System initialized")

        self.logger.info("Step 5/5: Setting up evaluator...")
        self.evaluator = BenchmarkEvaluator(self.config.evaluation)
        self.logger.info("Initialization complete!")
        
    def _load_config(self) -> TridentConfig:
        """Load or create configuration."""
        config_path = self.args.config or "configs/default.json"

        if Path(config_path).exists():
            with open(config_path) as f:
                config_dict = json.load(f)
        else:
            # Use config family if specified, otherwise use default configuration
            if self.args.config_family:
                return self._load_config_family()

            # Default configuration (backward compatible)
            config_dict = {
                "mode": self.args.mode or "safe_cover",
                "safe_cover": {
                    "per_facet_alpha": 0.01,
                    "token_cap": self.args.budget_tokens,
                    "early_abstain": True
                },
                "pareto": {
                    "budget": self.args.budget_tokens,
                    "relaxed_alpha": 0.5
                },
                "llm": {
                    "model_name": self.args.model or "meta-llama/Llama-2-7b-hf",
                    "temperature": self.args.temperature or 0.0,
                    "max_new_tokens": self.args.max_new_tokens or 512,
                    "device": self.device,
                    "load_in_8bit": self.args.load_in_8bit
                },
                "retrieval": {
                    "method": self.args.retrieval_method or "dense",
                    "top_k": self.args.top_k or 100,
                    "encoder_model": self.args.encoder_model or "facebook/contriever",
                    "corpus_path": self.args.corpus_path
                },
                "evaluation": {
                    "metrics": ["em", "f1", "support_em", "faithfulness"],
                    "dataset": self.args.dataset or "hotpotqa"
                },
                "baselines": {
                    "common_k": getattr(self.args, 'common_k', 8),
                    "selfrag_k": getattr(self.args, 'selfrag_k', 8),
                    "selfrag_use_critic": getattr(self.args, 'selfrag_use_critic', False),
                    "selfrag_allow_oracle_context": getattr(self.args, 'selfrag_allow_oracle_context', False),
                    "graphrag_k": getattr(self.args, 'graphrag_k', 8),
                    "graphrag_topk_nodes": getattr(self.args, 'graphrag_topk_nodes', 20),
                    "graphrag_max_seeds": getattr(self.args, 'graphrag_max_seeds', 10),
                    "graphrag_max_hops": getattr(self.args, 'graphrag_max_hops', 2),
                }
            }
        
        return TridentConfig.from_dict(config_dict)

    def _load_config_family(self) -> TridentConfig:
        """Load configuration from a named config family."""
        config_family_name = self.args.config_family

        # Check if it's a Self-RAG config
        if config_family_name in SELFRAG_CONFIGS:
            baseline_cfg = get_selfrag_config(config_family_name)
            mode = "self_rag"
        # Check if it's a TRIDENT config
        elif config_family_name in ALL_CONFIGS:
            trident_cfg = get_config(config_family_name)
            # Determine mode from config name
            if config_family_name.startswith("pareto"):
                mode = "pareto"
                pareto_cfg = trident_cfg
                safe_cover_cfg = SafeCoverConfig()  # Use defaults
            elif config_family_name.startswith("safe_cover"):
                mode = "safe_cover"
                safe_cover_cfg = trident_cfg
                pareto_cfg = ParetoConfig()  # Use defaults
            else:
                raise ValueError(f"Unknown config family type: {config_family_name}")
        else:
            raise ValueError(
                f"Unknown config family: {config_family_name}. "
                f"Available: {', '.join(list(ALL_CONFIGS.keys()) + list(SELFRAG_CONFIGS.keys()))}"
            )

        # Build full config dict
        config_dict = {
            "mode": mode,
            "llm": {
                "model_name": self.args.model or "meta-llama/Llama-2-7b-hf",
                "temperature": self.args.temperature or 0.0,
                "max_new_tokens": self.args.max_new_tokens or 512,
                "device": self.device,
                "load_in_8bit": self.args.load_in_8bit
            },
            "retrieval": {
                "method": self.args.retrieval_method or "dense",
                "top_k": self.args.top_k or 100,
                "encoder_model": self.args.encoder_model or "facebook/contriever",
                "corpus_path": self.args.corpus_path
            },
            "evaluation": {
                "metrics": ["em", "f1", "support_em", "faithfulness"],
                "dataset": self.args.dataset or "hotpotqa"
            }
        }

        # Add mode-specific config
        if mode == "pareto":
            config_dict["pareto"] = {
                "budget": pareto_cfg.budget,
                "max_evidence_tokens": pareto_cfg.max_evidence_tokens,
                "max_units": pareto_cfg.max_units,
                "stop_on_budget": pareto_cfg.stop_on_budget,
                "relaxed_alpha": pareto_cfg.relaxed_alpha,
                "weight_default": pareto_cfg.weight_default,
                "use_vqc": pareto_cfg.use_vqc,
                "use_bwk": pareto_cfg.use_bwk,
                "max_vqc_iterations": pareto_cfg.max_vqc_iterations,
                "bwk_exploration_bonus": pareto_cfg.bwk_exploration_bonus,
            }
            config_dict["safe_cover"] = {}  # Empty defaults
        elif mode == "safe_cover":
            config_dict["safe_cover"] = {
                "per_facet_alpha": safe_cover_cfg.per_facet_alpha,
                "token_cap": safe_cover_cfg.token_cap,
                "max_evidence_tokens": safe_cover_cfg.max_evidence_tokens,
                "max_units": safe_cover_cfg.max_units,
                "stop_on_budget": safe_cover_cfg.stop_on_budget,
                "abstain_on_infeasible": safe_cover_cfg.abstain_on_infeasible,
                "coverage_threshold": safe_cover_cfg.coverage_threshold,
                "dual_tolerance": safe_cover_cfg.dual_tolerance,
                "early_abstain": safe_cover_cfg.early_abstain,
                "use_certificates": safe_cover_cfg.use_certificates,
                "monitor_drift": safe_cover_cfg.monitor_drift,
                "psi_threshold": safe_cover_cfg.psi_threshold,
                "fallback_to_pareto": safe_cover_cfg.fallback_to_pareto,
            }
            config_dict["pareto"] = {}  # Empty defaults
        elif mode == "self_rag":
            config_dict["baselines"] = {
                "common_k": baseline_cfg.common_k,
                "selfrag_k": baseline_cfg.selfrag_k,
                "selfrag_use_critic": baseline_cfg.selfrag_use_critic,
                "selfrag_allow_oracle_context": baseline_cfg.selfrag_allow_oracle_context,
                "graphrag_k": baseline_cfg.graphrag_k,
                "graphrag_topk_nodes": baseline_cfg.graphrag_topk_nodes,
                "graphrag_max_seeds": baseline_cfg.graphrag_max_seeds,
                "graphrag_max_hops": baseline_cfg.graphrag_max_hops,
            }
            config_dict["safe_cover"] = {}
            config_dict["pareto"] = {}

        self.logger.info(f"Loaded config family: {config_family_name} (mode: {mode})")
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

        # If we have a data file with contexts, build corpus from it
        if hasattr(self.args, 'data_path') and self.args.data_path:
            corpus_texts = []
            corpus_ids = []

            # Use the new load_data function that handles JSON and JSONL
            data = load_data(self.args.data_path)

            for idx, example in enumerate(data):
                if 'context' in example:
                    for ctx_idx, (title, sentences) in enumerate(example['context']):
                        text = " ".join(sentences) if isinstance(sentences, list) else sentences
                        corpus_texts.append(text)
                        corpus_ids.append(f"doc_{idx}_{ctx_idx}_{title}")

            # Create retriever with this corpus
            if self.config.retrieval.method == "hybrid":
                retriever = HybridRetriever(
                    encoder_model=self.config.retrieval.encoder_model,
                    device=self.device,
                    top_k=self.config.retrieval.top_k
                )
            elif self.config.retrieval.method == "sparse":
                retriever = BM25Retriever()
            else:
                retriever = DenseRetriever(
                    encoder_model=self.config.retrieval.encoder_model,
                    device=self.device,
                    top_k=self.config.retrieval.top_k
                )

            # Load the corpus into retriever
            retriever.corpus = corpus_texts

            # Build appropriate index
            if hasattr(retriever, "build_index"):
                retriever.build_index()

            return retriever
    
    def _init_system(self) -> Any:
        """Initialize the appropriate system based on mode."""
        mode = self.config.mode

        if mode in ["safe_cover", "pareto", "both"]:
            # Initialize TRIDENT pipeline
            pipeline = TridentPipeline(
                config=self.config,
                llm=self.llm,
                retriever=self.retriever,
                device=self.device
            )
            return TridentSystemWrapper(pipeline, mode=mode)

        elif mode == "self_rag":
            # Initialize Self-RAG system
            return SelfRAGSystem(
                llm=self.instrumented_llm,
                retriever=self.retriever,
                k=self.config.baselines.selfrag_k,
                use_critic=self.config.baselines.selfrag_use_critic,
                allow_oracle_context=getattr(self.config.baselines, 'selfrag_allow_oracle_context', False)
            )

        elif mode == "graphrag":
            # Initialize GraphRAG system
            return GraphRAGSystem(
                llm=self.instrumented_llm,
                retriever=self.retriever,
                k=self.config.baselines.graphrag_k,
                topk_nodes=self.config.baselines.graphrag_topk_nodes,
                max_seeds=self.config.baselines.graphrag_max_seeds,
                max_hops=getattr(self.config.baselines, 'graphrag_max_hops', 2)
            )

        elif mode == "ketrag":
            # Initialize KET-RAG system
            return KETRAGSystem(
                llm=self.instrumented_llm,
                retriever=self.retriever,
                k=getattr(self.config.baselines, 'ketrag_k', 8),
                skeleton_ratio=getattr(self.config.baselines, 'ketrag_skeleton_ratio', 0.3),
                max_skeleton_triples=getattr(self.config.baselines, 'ketrag_max_skeleton_triples', 10),
                max_keyword_chunks=getattr(self.config.baselines, 'ketrag_max_keyword_chunks', 5)
            )

        else:
            raise ValueError(f"Unknown mode: {mode}. Must be one of: safe_cover, pareto, both, self_rag, graphrag, ketrag")
    
    def run_worker(self) -> None:
        """Run evaluation on a shard of data."""
        # Load data shard (handles both JSON and JSONL)
        data = load_data(self.args.data_path)
        
        results = []
        total_time = 0
        
        for idx, example in enumerate(data):
            self.logger.info(f"Processing {idx+1}/{len(data)}: {example['_id']}")
            
            try:
                start_time = time.time()

                # Determine whether to pass oracle context based on mode
                # For TRIDENT modes: pass context for facet mining from supporting facts
                # For baseline modes: only pass context if explicitly allowed
                if self.args.mode in ['safe_cover', 'pareto', 'both']:
                    # TRIDENT modes: pass context and supporting_facts
                    output = self.system.answer(
                        question=example['question'],
                        context=example.get('context'),
                        supporting_facts=example.get('supporting_facts')
                    )
                elif self.args.mode == 'self_rag':
                    # Self-RAG: only pass context if allow_oracle_context is enabled
                    use_context = getattr(self.config.baselines, 'selfrag_allow_oracle_context', False)
                    output = self.system.answer(
                        question=example['question'],
                        context=example.get('context') if use_context else None,
                        supporting_facts=example.get('supporting_facts')
                    )
                elif self.args.mode == 'graphrag':
                    # GraphRAG: pass context for graph building if flag is set, else use retrieval
                    use_oracle = getattr(self.args, 'graphrag_use_oracle_context', False)
                    output = self.system.answer(
                        question=example['question'],
                        context=example.get('context') if use_oracle else None,
                        supporting_facts=example.get('supporting_facts')
                    )
                elif self.args.mode == 'ketrag':
                    # KET-RAG: uses retrieval by default (dual-channel approach)
                    # Can optionally use oracle context for ablation studies
                    output = self.system.answer(
                        question=example['question'],
                        context=example.get('context'),  # KET-RAG can use context to build skeleton+keyword indices
                        supporting_facts=example.get('supporting_facts')
                    )
                else:
                    # Fallback: pass everything
                    output = self.system.answer(
                        question=example['question'],
                        context=example.get('context'),
                        supporting_facts=example.get('supporting_facts')
                    )

                elapsed_ms = (time.time() - start_time) * 1000

                # Extract results (handle both TRIDENT and baseline output formats)
                result = WorkerResult(
                    query_id=example['_id'],
                    question=example['question'],
                    prediction=output.get('answer', ''),
                    ground_truth=[example['answer']] if isinstance(example['answer'], str) else example['answer'],
                    selected_passages=output.get('selected_passages', []),
                    certificates=output.get('certificates', []),
                    metrics=output.get('stats', {}),
                    abstained=output.get('abstained', False),
                    latency_ms=output.get('latency_ms', elapsed_ms),
                    tokens_used=output.get('tokens_used', 0),
                    mode=output.get('mode', self.config.mode)
                )
                
                results.append(asdict(result))
                total_time += elapsed_ms
                
                # Log progress
                if (idx + 1) % 10 == 0:
                    avg_time = total_time / (idx + 1)
                    self.logger.info(f"Progress: {idx+1}/{len(data)}, Avg time: {avg_time:.2f}ms")
                
            except Exception as e:
                self.logger.error(f"Error processing {example['_id']}: {str(e)}")
                self.logger.error(traceback.format_exc())
                
                # Add failed result
                results.append({
                    'query_id': example['_id'],
                    'error': str(e),
                    'abstained': True
                })
        
        # Save results
        output_path = Path(self.args.output_dir) / "results.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump({
                'config': asdict(self.config),
                'results': results,
                'summary': self._compute_summary(results)
            }, f, indent=2)
        
        self.logger.info(f"Results saved to {output_path}")
    
    def _compute_summary(self, results: List[Dict]) -> Dict:
        """Compute summary statistics."""
        valid_results = [r for r in results if 'error' not in r]

        if not valid_results:
            return {'error': 'No valid results'}

        # Compute metrics
        em_scores = []
        f1_scores = []
        tokens_used = []
        evidence_tokens = []
        num_units = []
        latencies = []
        abstained_count = 0

        for result in valid_results:
            if result.get('abstained'):
                abstained_count += 1
                continue

            pred = result.get('prediction')
            gts = result.get('ground_truth') or []
            if pred and gts:
                best_em = 0.0
                best_f1 = 0.0
                for gt in gts:
                    em = 1.0 if _normalize(pred) == _normalize(gt) else 0.0
                    f1 = _f1(pred, gt)
                    best_em = max(best_em, em)
                    best_f1 = max(best_f1, f1)
                em_scores.append(best_em)
                f1_scores.append(best_f1)

            tokens_used.append(result.get('tokens_used', 0))
            latencies.append(result.get('latency_ms', 0))

            # Track evidence tokens and num_units if available (new config families)
            metrics = result.get('metrics', {})
            if 'evidence_tokens' in metrics:
                evidence_tokens.append(metrics['evidence_tokens'])
            if 'num_units' in metrics:
                num_units.append(metrics['num_units'])

        summary = {
            'num_examples': len(results),
            'num_valid': len(valid_results),
            'num_abstained': abstained_count,
            'abstention_rate': abstained_count / len(valid_results) if valid_results else 0,
            'avg_em': np.mean(em_scores) if em_scores else 0,
            'avg_f1': np.mean(f1_scores) if f1_scores else 0,
            'avg_tokens_total': np.mean(tokens_used) if tokens_used else 0,
            'avg_latency_ms': np.mean(latencies) if latencies else 0,
            'total_tokens': sum(tokens_used),
            'mode': self.config.mode
        }

        # Add evidence token metrics if available
        if evidence_tokens:
            summary['avg_evidence_tokens'] = np.mean(evidence_tokens)
            summary['total_evidence_tokens'] = sum(evidence_tokens)

        # Add num_units metrics if available
        if num_units:
            summary['avg_num_units'] = np.mean(num_units)

        return summary


def run_multi_gpu(args: argparse.Namespace) -> None:
    """Run evaluation in parallel across multiple GPUs."""
    print(f"\n{'='*60}")
    print("MULTI-GPU EVALUATION MODE")
    print(f"{'='*60}")

    # Load and optionally limit data
    print(f"Loading data from: {args.data_path}")
    examples = load_data(args.data_path, limit=args.limit)
    print(f"Loaded {len(examples)} examples")

    # Create output directories
    output_dir = Path(args.output_dir)
    shards_dir = output_dir / "shards"
    results_dir = output_dir / "results"
    shards_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Create shards
    print(f"Creating shards (size={args.shard_size})...")
    shard_paths = create_shards(examples, str(shards_dir), args.shard_size)
    print(f"Created {len(shard_paths)} shards")

    # Resolve GPU IDs
    gpu_ids = _resolve_gpu_ids(args.device, args.num_gpus)
    if not gpu_ids:
        raise RuntimeError(f"No GPUs available (device={args.device}, num_gpus={args.num_gpus})")
    print(f"Using GPUs: {gpu_ids}")

    # Queue for shard distribution
    q: Queue = Queue()
    for sp in shard_paths:
        q.put(sp)

    result_paths: List[str] = []
    result_lock = Lock()
    print_lock = Lock()

    def run_worker_on_gpu(gpu_id: str):
        """Worker bound to a single GPU."""
        while True:
            try:
                shard_path = q.get_nowait()
            except Empty:
                return

            shard_name = Path(shard_path).stem
            shard_output_dir = results_dir / shard_name
            shard_output_dir.mkdir(parents=True, exist_ok=True)

            # Build command
            cmd = [
                sys.executable, __file__,
                "--worker",
                "--data_path", shard_path,
                "--output_dir", str(shard_output_dir),
                "--mode", args.mode,
                "--model", args.model,
                "--device", "0",  # Always 0 since we mask CUDA_VISIBLE_DEVICES
                "--dataset", args.dataset,
                "--temperature", str(args.temperature),
                "--max_new_tokens", str(args.max_new_tokens),
                "--seed", str(args.seed),
            ]

            if args.config:
                cmd.extend(["--config", args.config])
            if args.config_family:
                cmd.extend(["--config_family", args.config_family])
            if args.load_in_8bit:
                cmd.append("--load_in_8bit")

            # Per-process environment: pin to exactly one GPU
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

            log_path = shard_output_dir / f"worker_gpu{gpu_id}.log"
            with print_lock:
                print(f"[GPU {gpu_id}] Processing: {shard_name}")

            start_time = time.time()
            with open(log_path, "w", encoding="utf-8") as logf:
                result = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, text=True, env=env)
            elapsed = time.time() - start_time

            if result.returncode != 0:
                with print_lock:
                    print(f"[GPU {gpu_id}] ERROR on {shard_name} (see {log_path})")
            else:
                with print_lock:
                    print(f"[GPU {gpu_id}] Done {shard_name} in {elapsed:.1f}s")

            rp = shard_output_dir / "results.json"
            if rp.exists():
                with result_lock:
                    result_paths.append(str(rp))

            q.task_done()

    # Launch workers
    print(f"\nStarting {len(gpu_ids)} workers...")
    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as ex:
        futures = [ex.submit(run_worker_on_gpu, gid) for gid in gpu_ids]
        for f in futures:
            f.result()

    # Aggregate results
    print(f"\n{'='*60}")
    print("AGGREGATING RESULTS")
    print(f"{'='*60}")

    all_results = []
    for rp in result_paths:
        with open(rp, 'r') as f:
            data = json.load(f)
        if 'results' in data:
            all_results.extend(data['results'])

    # Compute aggregate metrics
    em_scores = []
    f1_scores = []
    abstained = 0
    tokens_used = []

    for result in all_results:
        if result.get('abstained') or 'error' in result:
            abstained += 1
            continue

        pred = result.get('prediction', '')
        gts = result.get('ground_truth', [])
        if pred and gts:
            best_em = max(1.0 if _normalize(pred) == _normalize(gt) else 0.0 for gt in gts)
            best_f1 = max(_f1(pred, gt) for gt in gts)
            em_scores.append(best_em)
            f1_scores.append(best_f1)

        tokens_used.append(result.get('tokens_used', 0))

    summary = {
        'total': len(all_results),
        'valid': len(all_results) - abstained,
        'abstained': abstained,
        'abstention_rate': abstained / len(all_results) if all_results else 0,
        'avg_em': np.mean(em_scores) if em_scores else 0,
        'avg_f1': np.mean(f1_scores) if f1_scores else 0,
        'avg_tokens': np.mean(tokens_used) if tokens_used else 0,
    }

    # Save aggregated results
    aggregated_path = output_dir / "aggregated_results.json"
    with open(aggregated_path, 'w') as f:
        json.dump({
            'summary': summary,
            'results': all_results
        }, f, indent=2)

    print(f"\nResults:")
    print(f"  Total examples: {summary['total']}")
    print(f"  EM: {summary['avg_em']:.4f}")
    print(f"  F1: {summary['avg_f1']:.4f}")
    print(f"  Abstention rate: {summary['abstention_rate']:.4f}")
    print(f"\nSaved to: {aggregated_path}")


def _resolve_gpu_ids(start_gpu: int, num_gpus: int) -> List[str]:
    """Resolve physical GPU IDs to use."""
    if num_gpus <= 0:
        return []

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        vis_list = [x.strip() for x in visible.split(",") if x.strip()]
        return vis_list[start_gpu:start_gpu + num_gpus]

    return [str(i) for i in range(start_gpu, start_gpu + num_gpus)]


def main():
    parser = argparse.ArgumentParser(
        description="TRIDENT Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single worker mode
  python eval_complete_runnable.py --worker --data_path shard.json --output_dir results/

  # Multi-GPU parallel mode (automatically shards data)
  python eval_complete_runnable.py --data_path musique_ans_v1.0_dev.jsonl --output_dir results/ --num_gpus 4

  # With limit and custom shard size
  python eval_complete_runnable.py --data_path data.jsonl --output_dir results/ --num_gpus 4 --limit 1000 --shard_size 50
        """
    )

    # Required arguments
    parser.add_argument("--worker", action="store_true", help="Run as single worker (internal use)")
    parser.add_argument("--data_path", type=str, required=True, help="Path to data file (JSON or JSONL)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")

    # Multi-GPU options
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs for parallel processing")
    parser.add_argument("--shard_size", type=int, default=100, help="Examples per shard for multi-GPU mode")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples to process")
    
    # Model configuration
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf", help="LLM model name")
    parser.add_argument("--device", type=int, default=0, help="CUDA device (-1 for CPU)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max tokens to generate")
    parser.add_argument("--load_in_8bit", action="store_true", help="Load model in 8-bit mode")
    
    # Pipeline configuration
    parser.add_argument(
        "--mode",
        choices=["safe_cover", "pareto", "both", "self_rag", "graphrag", "ketrag"],
        default="safe_cover",
        help="System mode: safe_cover/pareto/both (TRIDENT modes), self_rag, graphrag, or ketrag"
    )
    parser.add_argument("--budget_tokens", type=int, default=2000, help="Token budget (legacy, use --config_family instead)")
    parser.add_argument("--config_family", type=str, help="Named config family (e.g., pareto_cheap_1500, safe_cover_equal_2500, selfrag_base)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Retrieval configuration
    parser.add_argument("--retrieval_method", choices=["dense", "hybrid", "sparse"], default="dense")
    parser.add_argument("--corpus_path", type=str, help="Path to corpus for retrieval")
    parser.add_argument("--encoder_model", type=str, default="facebook/contriever")
    parser.add_argument("--top_k", type=int, default=100, help="Top-k passages to retrieve")
    
    # Dataset configuration
    parser.add_argument("--dataset", type=str, default="hotpotqa", help="Dataset name")
    parser.add_argument("--config", type=str, help="Path to configuration file")

    # Baseline-specific configuration
    parser.add_argument("--common_k", type=int, default=8, help="Common k for retrieval across all baselines")
    parser.add_argument("--selfrag_k", type=int, help="Number of documents for Self-RAG (defaults to --common_k)")
    parser.add_argument("--selfrag_use_critic", action="store_true", help="Use critic in Self-RAG")
    parser.add_argument("--selfrag_allow_oracle_context", action="store_true", help="Allow Self-RAG to use oracle context")
    parser.add_argument("--graphrag_k", type=int, help="Number of documents for GraphRAG (defaults to --common_k)")
    parser.add_argument("--graphrag_max_seeds", type=int, default=10, help="Max seed nodes for GraphRAG")
    parser.add_argument("--graphrag_topk_nodes", type=int, default=20, help="Top-k nodes for GraphRAG")
    parser.add_argument("--graphrag_max_hops", type=int, default=2, help="Max hops for GraphRAG BFS expansion")
    parser.add_argument("--graphrag_use_oracle_context", action="store_true", help="Use oracle context for GraphRAG (default: uses retrieval)")
    parser.add_argument("--ketrag_k", type=int, help="Number of documents for KET-RAG (defaults to --common_k)")
    parser.add_argument("--ketrag_skeleton_ratio", type=float, default=0.3, help="Ratio of chunks for skeleton KG (default: 0.3)")
    parser.add_argument("--ketrag_max_skeleton_triples", type=int, default=10, help="Max triples from skeleton KG (default: 10)")
    parser.add_argument("--ketrag_max_keyword_chunks", type=int, default=5, help="Max chunks from keyword index (default: 5)")

    args = parser.parse_args()

    # Validate config_family and mode compatibility
    if args.config_family:
        if args.config_family.startswith("pareto") and args.mode not in ["pareto", None]:
            print(f"Warning: config_family '{args.config_family}' is for Pareto mode, but --mode={args.mode}")
            args.mode = "pareto"
        elif args.config_family.startswith("safe_cover") and args.mode not in ["safe_cover", None]:
            print(f"Warning: config_family '{args.config_family}' is for Safe-Cover mode, but --mode={args.mode}")
            args.mode = "safe_cover"
        elif args.config_family.startswith("selfrag") and args.mode not in ["self_rag", None]:
            print(f"Warning: config_family '{args.config_family}' is for Self-RAG mode, but --mode={args.mode}")
            args.mode = "self_rag"

    # Apply common_k defaults
    if args.selfrag_k is None:
        args.selfrag_k = args.common_k
    if args.graphrag_k is None:
        args.graphrag_k = args.common_k
    if args.ketrag_k is None:
        args.ketrag_k = args.common_k

    # Set random seeds
    np.random.seed(args.seed)

    torch_spec = importlib.util.find_spec("torch")
    if torch_spec is not None:
        torch = importlib.import_module("torch")
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # Determine run mode
    if args.worker:
        # Single worker mode (used internally by multi-GPU orchestration)
        runner = ExperimentRunner(args)
        runner.run_worker()
    elif args.num_gpus > 1:
        # Multi-GPU parallel mode
        run_multi_gpu(args)
    else:
        # Single GPU mode - just run as worker directly
        print(f"Running on single GPU (device={args.device})")
        runner = ExperimentRunner(args)
        runner.run_worker()


if __name__ == "__main__":
    main()