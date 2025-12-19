#!/usr/bin/env python3
# IMPORTANT: These must be at the very top, before ANY other imports
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import multiprocessing
# Set multiprocessing start method to 'spawn' before any CUDA operations
# This fixes the "Cannot re-initialize CUDA in forked subprocess" error
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

"""
Evaluation script for full baseline systems (Self-RAG, KET-RAG, Vanilla RAG, HippoRAG).

Supports multiple datasets: HotpotQA, MuSiQue, 2WikiMultiHopQA
Supports both OpenAI API and local LLMs via HuggingFace

Baselines:
    - Self-RAG: Self-reflective retrieval-augmented generation
    - KET-RAG (reimpl): In-framework reimplementation of skeleton + keyword RAG
    - KET-RAG (official): Faithful wrapper using precomputed contexts from official KET-RAG
    - Vanilla RAG: Simple TF-IDF retrieval + LLM generation (baseline)
    - HippoRAG: Memory-enhanced RAG with personalized PageRank

Usage with OpenAI:
    # Run all baselines on HotpotQA
    python eval_full_baselines.py \
        --data_path data/hotpotqa_dev_shards/shard_0.jsonl \
        --output_dir results/full_baselines \
        --baselines ketrag_reimpl selfrag vanillarag hipporag

    # Run Self-RAG on 2WikiMultiHopQA (auto-detected from path)
    python eval_full_baselines.py \
        --data_path data/2wikimultihop_dev.json \
        --output_dir results/2wiki_baselines \
        --baselines selfrag

    # Explicit dataset specification
    python eval_full_baselines.py \
        --data_path data/my_data.json \
        --output_dir results/baselines \
        --dataset 2wiki \
        --baselines selfrag

    # KET-RAG official (uses original KET-RAG prompts and outputs)
    python eval_full_baselines.py \
        --data_path data/hotpotqa_dev_shards/shard_0.jsonl \
        --output_dir results/full_baselines \
        --baselines ketrag_official \
        --ketrag_context_file KET-RAG/ragtest-hotpot/output/ragtest-hotpot-keyword-0.5.json

Usage with Local LLM:
    python eval_full_baselines.py \
        --data_path data/hotpotqa_dev_shards/shard_0.jsonl \
        --output_dir results/full_baselines \
        --baselines ketrag_reimpl selfrag \
        --use_local_llm \
        --local_llm_model Qwen/Qwen2.5-7B-Instruct \
        --local_llm_device cuda:0

Environment variables:
    OPENAI_API_KEY: Required for KET-RAG, Vanilla RAG, HippoRAG when not using local LLM
    HF_TOKEN: Required for Self-RAG model download
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Any, List
import time
from tqdm import tqdm
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from baselines.full_baseline_interface import compute_exact_match, compute_f1

# Note: Baseline adapters are imported lazily in the evaluation loop
# to avoid loading heavy dependencies (like HippoRAG) when not needed.
# This prevents multiprocessing conflicts with libraries that initialize
# at import time.


def convert_musique_example(ex: Dict) -> Dict:
    """Convert a single MuSiQue example to standard format."""
    context = []
    supporting_facts = []

    for para in ex.get('paragraphs', []):
        title = para.get('title', f"para_{para.get('idx', 0)}")
        text = para.get('paragraph_text', '')
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        context.append([title, sentences])

        # Track supporting paragraphs
        if para.get('is_supporting', False):
            for sent_idx in range(len(sentences)):
                supporting_facts.append([title, sent_idx])

    return {
        '_id': ex.get('id', ''),
        'question': ex.get('question', ''),
        'answer': ex.get('answer', ''),
        'answer_aliases': ex.get('answer_aliases', []),
        'context': context,
        'supporting_facts': supporting_facts,
        'type': ex.get('id', '').split('__')[0] if '__' in ex.get('id', '') else 'unknown',
        'answerable': ex.get('answerable', True),
    }


def convert_2wiki_example(ex: Dict) -> Dict:
    """Convert a single 2WikiMultiHopQA example to standard format.

    2WikiMultiHop format is similar to HotpotQA with some differences:
    - May have 'evidences' field instead of 'supporting_facts'
    - Context may be in different formats (dict or list)
    - Types: comparison, inference, compositional, bridge
    """
    # Handle context format - 2Wiki uses similar format to HotpotQA
    context = ex.get('context', [])
    if isinstance(context, dict):
        # Handle dict format: {'title': [...], 'sentences': [[...], ...]}
        titles = context.get('title', [])
        sentences_list = context.get('sentences', [])
        context = list(zip(titles, sentences_list))
    else:
        # Ensure context is list of [title, sentences] pairs
        normalized_context = []
        for item in context:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                normalized_context.append([item[0], item[1]])
            elif isinstance(item, dict):
                title = item.get('title', '')
                sentences = item.get('sentences', item.get('text', ''))
                if isinstance(sentences, str):
                    sentences = [sentences]
                normalized_context.append([title, sentences])
        context = normalized_context

    # Handle supporting_facts - 2Wiki may use 'evidences' or 'supporting_facts'
    sf = ex.get('supporting_facts', ex.get('evidences', []))
    if isinstance(sf, dict):
        sf_titles = sf.get('title', [])
        sf_sents = sf.get('sent_id', [])
        sf = list(zip(sf_titles, sf_sents))

    return {
        '_id': ex.get('_id', ex.get('id', '')),
        'question': ex.get('question', ''),
        'answer': ex.get('answer', ''),
        'context': context,
        'supporting_facts': sf,
        'type': ex.get('type', 'unknown'),
        'level': ex.get('level', 'unknown'),
    }


def load_hotpotqa_data(data_path: str, max_samples: int = None, dataset: str = None) -> List[Dict[str, Any]]:
    """Load HotpotQA/MuSiQue/2WikiMultiHop data from JSONL or JSON array files.

    The loader skips empty lines in JSONL files and also supports files that
    contain a single JSON array (common when exporting small evaluation shards).

    Automatically detects and converts dataset formats:
    - MuSiQue: uses 'paragraphs' instead of 'context'
    - 2WikiMultiHop: similar to HotpotQA but may use 'evidences' instead of 'supporting_facts'

    Args:
        data_path: Path to the data file (JSON or JSONL)
        max_samples: Maximum number of samples to load
        dataset: Explicit dataset name ('hotpotqa', 'musique', '2wiki'). Auto-detected if None.
    """
    path_lower = data_path.lower()

    # Auto-detect dataset from path if not specified
    if dataset is None:
        if 'musique' in path_lower:
            dataset = 'musique'
        elif '2wiki' in path_lower or 'wikimultihop' in path_lower:
            dataset = '2wiki'
        else:
            dataset = 'hotpotqa'

    with open(data_path, 'r') as f:
        # Peek at the first non-empty line to determine file shape
        first_non_empty = ""
        for line in f:
            stripped = line.strip()
            if stripped:
                first_non_empty = stripped
                break

        f.seek(0)

        if first_non_empty.startswith("["):
            data = json.load(f)
            if max_samples:
                data = data[:max_samples]
        else:
            data: List[Dict[str, Any]] = []
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                if not line.strip():
                    continue
                data.append(json.loads(line))

    # Convert based on dataset type
    if dataset == 'musique' or (data and 'paragraphs' in data[0]):
        print(f"Detected MuSiQue format, converting {len(data)} examples...")
        data = [convert_musique_example(ex) for ex in data]
    elif dataset == '2wiki':
        print(f"Detected 2WikiMultiHop format, converting {len(data)} examples...")
        data = [convert_2wiki_example(ex) for ex in data]
    # HotpotQA format is already in the correct format, no conversion needed

    return data


def evaluate_baseline(
    baseline_name: str,
    baseline_system,
    data: List[Dict[str, Any]],
    output_path: str
) -> Dict[str, Any]:
    """
    Evaluate a baseline system on HotpotQA data.

    Metrics Separation (Aligned with Fair Baseline Comparison):
    - Query-only metrics (tokens_used, latency_ms): Online inference costs only
    - Total metrics: Include offline indexing costs from stats['indexing_*']
    - This matches how original papers report performance

    Args:
        baseline_name: Name of the baseline ('selfrag', 'ketrag')
        baseline_system: Initialized baseline system
        data: List of HotpotQA examples
        output_path: Path to save results

    Returns:
        Summary statistics with separated query/total metrics
    """
    print(f"\n{'='*80}")
    print(f"Evaluating {baseline_name.upper()}")
    print(f"{'='*80}\n")

    results = []
    em_scores = []
    f1_scores = []

    # Query-only metrics (matches original paper claims)
    query_tokens = []
    query_latencies = []

    # Total metrics (includes indexing overhead)
    total_tokens = []
    total_latencies = []

    # Indexing metrics (for reference)
    indexing_tokens = []
    indexing_latencies = []

    abstention_count = 0

    for example in tqdm(data, desc=f"{baseline_name}"):
        question = example['question']
        context = example.get('context', [])
        supporting_facts = example.get('supporting_facts', [])
        answer = example.get('answer', '')
        question_id = example.get('_id', 'unknown')
        question_type = example.get('type', 'unknown')

        try:
            # Generate answer
            response = baseline_system.answer(
                question=question,
                context=context,
                supporting_facts=supporting_facts,
                metadata={
                    'question_id': question_id,
                    'type': question_type,
                }
            )

            # Compute metrics
            em = compute_exact_match(response.answer, answer)
            f1 = compute_f1(response.answer, answer)

            em_scores.append(em)
            f1_scores.append(f1)

            # Query-only metrics (PRIMARY)
            query_tokens.append(response.tokens_used)
            query_latencies.append(response.latency_ms)

            # Extract indexing metrics from stats
            idx_tokens = response.stats.get('indexing_tokens', 0)
            idx_latency = response.stats.get('indexing_latency_ms', 0.0)
            tot_tokens = response.stats.get('total_cost_tokens', response.tokens_used)

            indexing_tokens.append(idx_tokens)
            indexing_latencies.append(idx_latency)
            total_tokens.append(tot_tokens)
            total_latencies.append(response.latency_ms + idx_latency)

            if response.abstained:
                abstention_count += 1

            # Save result
            result = {
                'question_id': question_id,
                'question': question,
                'answer': answer,
                'prediction': response.answer,
                'raw_answer': response.raw_answer,
                'extracted_answer': response.extracted_answer,
                'em': em,
                'f1': f1,
                # Query-only (PRIMARY)
                'tokens_used': response.tokens_used,
                'latency_ms': response.latency_ms,
                # Total costs
                'total_tokens': tot_tokens,
                'total_latency_ms': response.latency_ms + idx_latency,
                'indexing_tokens': idx_tokens,
                'indexing_latency_ms': idx_latency,
                'abstained': response.abstained,
                'mode': response.mode,
                'stats': response.stats,
                'type': question_type,
            }
            results.append(result)

        except Exception as e:
            print(f"\nError processing question {question_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Compute summary statistics
    summary = {
        'baseline': baseline_name,
        'num_examples': len(data),
        'num_processed': len(results),
        'num_abstained': abstention_count,
        'abstention_rate': abstention_count / len(results) if results else 0.0,

        # Accuracy metrics
        'avg_em': np.mean(em_scores) if em_scores else 0.0,
        'avg_f1': np.mean(f1_scores) if f1_scores else 0.0,

        # Query-only metrics (PRIMARY - matches original papers)
        'avg_query_tokens': np.mean(query_tokens) if query_tokens else 0.0,
        'median_query_tokens': np.median(query_tokens) if query_tokens else 0.0,
        'avg_query_latency_ms': np.mean(query_latencies) if query_latencies else 0.0,
        'median_query_latency_ms': np.median(query_latencies) if query_latencies else 0.0,

        # Total metrics (includes indexing)
        'avg_total_tokens': np.mean(total_tokens) if total_tokens else 0.0,
        'median_total_tokens': np.median(total_tokens) if total_tokens else 0.0,
        'avg_total_latency_ms': np.mean(total_latencies) if total_latencies else 0.0,
        'median_total_latency_ms': np.median(total_latencies) if total_latencies else 0.0,

        # Indexing overhead (for reference)
        'avg_indexing_tokens': np.mean(indexing_tokens) if indexing_tokens else 0.0,
        'avg_indexing_latency_ms': np.mean(indexing_latencies) if indexing_latencies else 0.0,
    }

    # Save results
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Save individual results
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    # Save summary
    summary_file = output_file.parent / f"{baseline_name}_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{baseline_name.upper()} Results:")
    print(f"  EM: {summary['avg_em']:.4f}")
    print(f"  F1: {summary['avg_f1']:.4f}")
    print(f"  Query Tokens (PRIMARY): {summary['avg_query_tokens']:.1f}")
    print(f"  Total Tokens (w/ indexing): {summary['avg_total_tokens']:.1f}")
    print(f"  Query Latency (PRIMARY): {summary['avg_query_latency_ms']:.1f}ms")
    print(f"  Total Latency (w/ indexing): {summary['avg_total_latency_ms']:.1f}ms")
    print(f"  Abstention Rate: {summary['abstention_rate']:.2%}")
    print(f"\nResults saved to: {output_file}")
    print(f"Summary saved to: {summary_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate full baseline systems on HotpotQA")

    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to HotpotQA data file (JSONL format)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/full_baselines",
        help="Output directory for results"
    )
    parser.add_argument(
        "--baselines",
        nargs='+',
        choices=['selfrag', 'ketrag_reimpl', 'ketrag_official', 'vanillarag', 'hipporag', 'graphrag', 'all'],
        default=['all'],
        help="Which baselines to evaluate (ketrag_reimpl=in-framework reimpl, ketrag_official=faithful wrapper, graphrag=vanilla GraphRAG)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (for testing)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=['hotpotqa', 'musique', '2wiki'],
        default=None,
        help="Dataset type (auto-detected from path if not specified)"
    )

    # KET-RAG options
    parser.add_argument(
        "--ketrag_model",
        type=str,
        default="gpt-4o-mini",
        help="Model to use for KET-RAG (OpenAI model name)"
    )
    parser.add_argument(
        "--ketrag_context_file",
        type=str,
        default=None,
        help="Path to precomputed context JSON for ketrag_official (from official KET-RAG pipeline)"
    )
    parser.add_argument(
        "--use_local_llm",
        action="store_true",
        help="Use local LLM instead of OpenAI for GraphRAG/KET-RAG"
    )
    parser.add_argument(
        "--local_llm_model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace model name for local LLM"
    )
    parser.add_argument(
        "--local_llm_device",
        type=str,
        default="cuda:0",
        help="Device for local LLM (cuda:0, cuda:1, cpu)"
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Use 8-bit quantization for local LLM"
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Use 4-bit quantization for local LLM"
    )

    # Self-RAG options
    parser.add_argument(
        "--selfrag_model",
        type=str,
        default="selfrag/selfrag_llama2_7b",
        help="Self-RAG model name (7b or 13b)"
    )
    parser.add_argument(
        "--selfrag_max_tokens",
        type=int,
        default=100,
        help="Max tokens for Self-RAG generation"
    )
    parser.add_argument(
        "--selfrag_gpu_memory_utilization",
        type=float,
        default=0.5,
        help="GPU memory utilization fraction for Self-RAG (default 0.5)"
    )
    parser.add_argument(
        "--selfrag_mode",
        type=str,
        default="adaptive_retrieval",
        choices=["adaptive_retrieval", "no_retrieval", "always_retrieve"],
        help="Self-RAG retrieval mode (default: adaptive_retrieval)"
    )
    parser.add_argument(
        "--selfrag_threshold",
        type=float,
        default=None,
        help="Probability threshold for adaptive retrieval. None = check generated text for [Retrieval] token (recommended)"
    )
    parser.add_argument(
        "--selfrag_use_groundness",
        action="store_true",
        default=True,
        help="Use groundedness/support tokens for scoring (default: True)"
    )
    parser.add_argument(
        "--selfrag_use_utility",
        action="store_true",
        default=True,
        help="Use utility tokens for scoring (default: True)"
    )
    parser.add_argument(
        "--selfrag_use_seqscore",
        action="store_true",
        help="Include sequence probability in scoring"
    )
    parser.add_argument(
        "--selfrag_ndocs",
        type=int,
        default=10,
        help="Number of documents for retrieval scoring (default: 10)"
    )

    # Vanilla RAG options
    parser.add_argument(
        "--vanillarag_model",
        type=str,
        default="gpt-4o-mini",
        help="Model to use for Vanilla RAG (OpenAI model name)"
    )
    parser.add_argument(
        "--vanillarag_top_k",
        type=int,
        default=5,
        help="Number of chunks to retrieve for Vanilla RAG"
    )

    # HippoRAG options
    parser.add_argument(
        "--hipporag_model",
        type=str,
        default="gpt-4o-mini",
        help="Model to use for HippoRAG (OpenAI model name or local model)"
    )
    parser.add_argument(
        "--hipporag_embedding_model",
        type=str,
        default="nvidia/NV-Embed-v2",
        help="Embedding model for HippoRAG (nvidia/NV-Embed-v2, GritLM, Contriever)"
    )
    parser.add_argument(
        "--hipporag_num_retrieve",
        type=int,
        default=5,
        help="Number of passages to retrieve for HippoRAG"
    )
    parser.add_argument(
        "--hipporag_save_dir",
        type=str,
        default=None,
        help="Directory to save HippoRAG outputs (default: temp dir)"
    )
    parser.add_argument(
        "--hipporag_local_base_url",
        type=str,
        default=None,
        help="Base URL for local vLLM server (e.g., http://localhost:8000/v1)"
    )

    # GraphRAG options
    parser.add_argument(
        "--graphrag_index_dir",
        type=str,
        default=None,
        help="Path to GraphRAG index output directory (from graphrag index command)"
    )
    parser.add_argument(
        "--graphrag_search_method",
        type=str,
        choices=["local", "global"],
        default="local",
        help="GraphRAG search method (default: local)"
    )
    parser.add_argument(
        "--graphrag_community_level",
        type=int,
        default=2,
        help="GraphRAG community hierarchy level (default: 2)"
    )

    args = parser.parse_args()

    # Normalize device format (handle both "3" and "cuda:3")
    if args.local_llm_device and args.local_llm_device.isdigit():
        args.local_llm_device = f"cuda:{args.local_llm_device}"

    # Expand 'all' to all baselines
    if 'all' in args.baselines:
        args.baselines = ['selfrag', 'ketrag_reimpl', 'ketrag_official', 'vanillarag', 'hipporag', 'graphrag']

    # Load data
    print(f"Loading data from: {args.data_path}")
    data = load_hotpotqa_data(args.data_path, args.max_samples, args.dataset)
    print(f"Loaded {len(data)} examples")

    # Check API keys (only if not using local LLM)
    api_key = None
    if not args.use_local_llm:
        api_key = os.getenv("OPENAI_API_KEY")
        openai_baselines = {'ketrag_reimpl', 'ketrag_official', 'vanillarag', 'hipporag'}
        needs_api_key = openai_baselines & set(args.baselines)
        if needs_api_key and not api_key:
            print(f"ERROR: {', '.join(needs_api_key)} require(s) OPENAI_API_KEY when not using --use_local_llm")
            sys.exit(1)

    # Evaluate each baseline
    summaries = {}

    for baseline_name in args.baselines:
        print(f"\n{'#'*80}")
        print(f"# Setting up {baseline_name.upper()}")
        print(f"{'#'*80}\n")

        try:
            # Initialize baseline system with lazy imports
            # (avoids loading unused dependencies that may conflict with multiprocessing)
            if baseline_name == 'selfrag':
                from baselines.full_selfrag_adapter import FullSelfRAGAdapter
                baseline_system = FullSelfRAGAdapter(
                    model_name=args.selfrag_model,
                    max_tokens=args.selfrag_max_tokens,
                    temperature=0.0,
                    gpu_memory_utilization=args.selfrag_gpu_memory_utilization,
                    device=args.local_llm_device,
                    # Vanilla Self-RAG settings
                    mode=args.selfrag_mode,
                    threshold=args.selfrag_threshold,
                    use_groundness=args.selfrag_use_groundness,
                    use_utility=args.selfrag_use_utility,
                    use_seqscore=args.selfrag_use_seqscore,
                    ndocs=args.selfrag_ndocs,
                )
            elif baseline_name == 'ketrag_reimpl':
                from baselines.full_ketrag_reimpl_adapter import FullKETRAGReimplAdapter
                baseline_system = FullKETRAGReimplAdapter(
                    api_key=api_key,
                    model=args.ketrag_model,
                    temperature=0.0,
                    max_tokens=500,
                    skeleton_ratio=0.3,
                    max_skeleton_triples=10,
                    max_keyword_chunks=5,
                    use_local_llm=args.use_local_llm,
                    local_llm_model=args.local_llm_model,
                    local_llm_device=args.local_llm_device,
                    load_in_8bit=args.load_in_8bit,
                    load_in_4bit=args.load_in_4bit,
                )
            elif baseline_name == 'ketrag_official':
                from baselines.full_ketrag_official_adapter import FullKETRAGAdapter
                # For official KET-RAG, we need the precomputed context file
                context_file = getattr(args, 'ketrag_context_file', None)
                if not context_file:
                    print("ERROR: ketrag_official requires --ketrag_context_file")
                    print("  Run the official KET-RAG pipeline first:")
                    print("    cd KET-RAG")
                    print("    poetry run graphrag index --root ragtest-hotpot/")
                    print("    poetry run python indexing_sket/create_context.py ragtest-hotpot/ keyword 0.5")
                    continue
                baseline_system = FullKETRAGAdapter(
                    context_file=context_file,
                    api_key=api_key,
                    model=args.ketrag_model,
                    temperature=0.0,
                    max_tokens=500,
                    use_local_llm=args.use_local_llm,
                    local_llm_model=args.local_llm_model,
                    local_llm_device=args.local_llm_device,
                    load_in_8bit=args.load_in_8bit,
                    load_in_4bit=args.load_in_4bit,
                )
            elif baseline_name == 'vanillarag':
                from baselines.full_vanillarag_adapter import FullVanillaRAGAdapter
                baseline_system = FullVanillaRAGAdapter(
                    api_key=api_key,
                    model=args.vanillarag_model,
                    temperature=0.0,
                    max_tokens=500,
                    top_k=args.vanillarag_top_k,
                    use_local_llm=args.use_local_llm,
                    local_llm_model=args.local_llm_model,
                    local_llm_device=args.local_llm_device,
                    load_in_8bit=args.load_in_8bit,
                    load_in_4bit=args.load_in_4bit,
                )
            elif baseline_name == 'hipporag':
                from baselines.full_hipporag_adapter import FullHippoRAGAdapter
                baseline_system = FullHippoRAGAdapter(
                    api_key=api_key,
                    model=args.hipporag_model,
                    embedding_model=args.hipporag_embedding_model,
                    temperature=0.0,
                    max_tokens=500,
                    num_to_retrieve=args.hipporag_num_retrieve,
                    save_dir=args.hipporag_save_dir,
                    use_local_llm=args.use_local_llm,
                    local_llm_base_url=args.hipporag_local_base_url,
                    local_llm_model=args.local_llm_model,
                )
            elif baseline_name == 'graphrag':
                from baselines.full_graphrag_adapter import FullGraphRAGAdapter
                # For GraphRAG, we need the precomputed index directory
                index_dir = getattr(args, 'graphrag_index_dir', None)
                if not index_dir:
                    print("ERROR: graphrag requires --graphrag_index_dir")
                    print("  Run the GraphRAG indexing pipeline first:")
                    print("    cd graphrag")
                    print("    graphrag index --root <project_root>")
                    continue
                baseline_system = FullGraphRAGAdapter(
                    index_dir=index_dir,
                    search_method=args.graphrag_search_method,
                    community_level=args.graphrag_community_level,
                    api_key=api_key,
                )
            else:
                print(f"Unknown baseline: {baseline_name}")
                continue

            # Evaluate
            output_path = os.path.join(args.output_dir, f"{baseline_name}_results.jsonl")
            summary = evaluate_baseline(baseline_name, baseline_system, data, output_path)
            summaries[baseline_name] = summary

        except Exception as e:
            print(f"\nERROR setting up {baseline_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Print comparison
    if len(summaries) > 1:
        print(f"\n{'='*80}")
        print("COMPARISON SUMMARY (Query-Only Metrics)")
        print(f"{'='*80}\n")
        print(f"{'Baseline':<15} {'EM':<10} {'F1':<10} {'Query Tokens':<15} {'Query Latency':<15}")
        print("-" * 80)
        for name, summary in summaries.items():
            print(f"{name:<15} {summary['avg_em']:<10.4f} {summary['avg_f1']:<10.4f} "
                  f"{summary['avg_query_tokens']:<15.1f} {summary['avg_query_latency_ms']:<15.1f}ms")

        print(f"\n{'='*80}")
        print("COMPARISON SUMMARY (Total Metrics w/ Indexing)")
        print(f"{'='*80}\n")
        print(f"{'Baseline':<15} {'EM':<10} {'F1':<10} {'Total Tokens':<15} {'Total Latency':<15}")
        print("-" * 80)
        for name, summary in summaries.items():
            print(f"{name:<15} {summary['avg_em']:<10.4f} {summary['avg_f1']:<10.4f} "
                  f"{summary['avg_total_tokens']:<15.1f} {summary['avg_total_latency_ms']:<15.1f}ms")

    # Save combined summary
    combined_summary_path = os.path.join(args.output_dir, "combined_summary.json")
    with open(combined_summary_path, 'w') as f:
        json.dump(summaries, f, indent=2)
    print(f"\nCombined summary saved to: {combined_summary_path}")


if __name__ == "__main__":
    main()