"""Evaluation module for TRIDENT with benchmark dataset support."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from datasets import load_dataset


@dataclass
class EvaluationMetrics:
    """Evaluation metrics for QA tasks."""
    exact_match: float
    f1_score: float
    support_em: float = 0.0  # EM on supporting facts
    support_f1: float = 0.0
    faithfulness: float = 0.0  # Answer faithful to retrieved passages
    abstention_rate: float = 0.0
    avg_tokens_used: float = 0.0
    avg_latency_ms: float = 0.0
    coverage_rate: float = 0.0  # Facet coverage
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'exact_match': self.exact_match,
            'f1_score': self.f1_score,
            'support_em': self.support_em,
            'support_f1': self.support_f1,
            'faithfulness': self.faithfulness,
            'abstention_rate': self.abstention_rate,
            'avg_tokens_used': self.avg_tokens_used,
            'avg_latency_ms': self.avg_latency_ms,
            'coverage_rate': self.coverage_rate
        }


class BenchmarkEvaluator:
    """Evaluator for QA benchmarks."""
    
    def __init__(self, config: Any):
        self.config = config
        self.metrics_to_compute = config.metrics if hasattr(config, 'metrics') else ['em', 'f1']
    
    def evaluate_batch(
        self,
        predictions: List[Dict[str, Any]],
        references: List[Dict[str, Any]]
    ) -> EvaluationMetrics:
        """Evaluate a batch of predictions."""
        em_scores = []
        f1_scores = []
        support_em_scores = []
        support_f1_scores = []
        faithfulness_scores = []
        abstention_count = 0
        token_counts = []
        latencies = []
        coverage_rates = []
        
        for pred, ref in zip(predictions, references):
            # Handle abstentions
            if pred.get('abstained', False):
                abstention_count += 1
                continue
            
            # Exact Match and F1
            pred_answer = self._normalize_answer(pred.get('prediction', ''))
            ref_answers = ref.get('answer', [])
            if isinstance(ref_answers, str):
                ref_answers = [ref_answers]
            
            # Compute EM
            em = max(self._exact_match(pred_answer, self._normalize_answer(ans)) 
                    for ans in ref_answers)
            em_scores.append(em)
            
            # Compute F1
            f1 = max(self._f1_score(pred_answer, self._normalize_answer(ans)) 
                    for ans in ref_answers)
            f1_scores.append(f1)
            
            # Support EM/F1 for multi-hop
            if 'support_em' in self.metrics_to_compute:
                pred_facts = pred.get('supporting_facts', [])
                ref_facts = ref.get('supporting_facts', [])
                if pred_facts and ref_facts:
                    support_em = self._support_exact_match(pred_facts, ref_facts)
                    support_f1 = self._support_f1_score(pred_facts, ref_facts)
                    support_em_scores.append(support_em)
                    support_f1_scores.append(support_f1)
            
            # Faithfulness (if passages provided)
            if 'faithfulness' in self.metrics_to_compute:
                faith_score = self._compute_faithfulness(
                    pred_answer,
                    pred.get('selected_passages', [])
                )
                faithfulness_scores.append(faith_score)
            
            # Token usage
            token_counts.append(pred.get('tokens_used', 0))
            
            # Latency
            latencies.append(pred.get('latency_ms', 0))
            
            # Coverage rate
            if pred.get('facets'):
                total_facets = len(pred['facets'])
                covered_facets = sum(1 for f in pred['facets'] 
                                   if f.get('covered', False))
                coverage_rates.append(covered_facets / max(total_facets, 1))
        
        # Compute averages
        num_valid = len(predictions) - abstention_count
        
        return EvaluationMetrics(
            exact_match=np.mean(em_scores) if em_scores else 0.0,
            f1_score=np.mean(f1_scores) if f1_scores else 0.0,
            support_em=np.mean(support_em_scores) if support_em_scores else 0.0,
            support_f1=np.mean(support_f1_scores) if support_f1_scores else 0.0,
            faithfulness=np.mean(faithfulness_scores) if faithfulness_scores else 0.0,
            abstention_rate=abstention_count / len(predictions) if predictions else 0.0,
            avg_tokens_used=np.mean(token_counts) if token_counts else 0.0,
            avg_latency_ms=np.mean(latencies) if latencies else 0.0,
            coverage_rate=np.mean(coverage_rates) if coverage_rates else 0.0
        )
    
    def _normalize_answer(self, text: str) -> str:
        """Normalize answer for evaluation."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove articles
        text = re.sub(r'\b(a|an|the)\b', ' ', text)
        
        # Remove punctuation
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def _exact_match(self, pred: str, ref: str) -> float:
        """Compute exact match score."""
        return float(pred == ref)
    
    def _f1_score(self, pred: str, ref: str) -> float:
        """Compute token-level F1 score."""
        pred_tokens = pred.split()
        ref_tokens = ref.split()
        
        if not pred_tokens or not ref_tokens:
            return 0.0
        
        common = Counter(pred_tokens) & Counter(ref_tokens)
        num_same = sum(common.values())
        
        if num_same == 0:
            return 0.0
        
        precision = num_same / len(pred_tokens)
        recall = num_same / len(ref_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        
        return f1
    
    def _support_exact_match(
        self,
        pred_facts: List[Tuple[str, int]],
        ref_facts: List[Tuple[str, int]]
    ) -> float:
        """Compute exact match for supporting facts."""
        pred_set = set(tuple(f) if isinstance(f, list) else f for f in pred_facts)
        ref_set = set(tuple(f) if isinstance(f, list) else f for f in ref_facts)
        return float(pred_set == ref_set)
    
    def _support_f1_score(
        self,
        pred_facts: List[Tuple[str, int]],
        ref_facts: List[Tuple[str, int]]
    ) -> float:
        """Compute F1 for supporting facts."""
        pred_set = set(tuple(f) if isinstance(f, list) else f for f in pred_facts)
        ref_set = set(tuple(f) if isinstance(f, list) else f for f in ref_facts)
        
        if not pred_set and not ref_set:
            return 1.0
        if not pred_set or not ref_set:
            return 0.0
        
        intersection = pred_set & ref_set
        precision = len(intersection) / len(pred_set)
        recall = len(intersection) / len(ref_set)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * precision * recall / (precision + recall)
    
    def _compute_faithfulness(
        self,
        answer: str,
        passages: List[Dict[str, Any]]
    ) -> float:
        """Compute faithfulness of answer to retrieved passages."""
        if not passages:
            return 0.0
        
        # Simple heuristic: check if answer tokens appear in passages
        answer_tokens = set(self._normalize_answer(answer).split())
        
        if not answer_tokens:
            return 1.0
        
        passage_text = " ".join(p.get('text', '') for p in passages)
        passage_tokens = set(self._normalize_answer(passage_text).split())
        
        overlap = answer_tokens & passage_tokens
        return len(overlap) / len(answer_tokens)


class DatasetLoader:
    """Load and preprocess benchmark datasets."""
    
    SUPPORTED_DATASETS = {
        'hotpotqa': 'hotpot_qa',
        'hotpot_qa': 'hotpot_qa',
        '2wikimultihop': 'multi_hop_qa',
        'musique': 'musique',
        'nq': 'natural_questions',
        'natural_questions': 'natural_questions',
        'triviaqa': 'trivia_qa',
        'squad': 'squad',
        'squad_v2': 'squad_v2'
    }
    
    @classmethod
    def load_dataset(
        cls,
        dataset_name: str,
        split: str = 'validation',
        limit: Optional[int] = None,
        cache_dir: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Load a benchmark dataset."""
        # Map to HuggingFace dataset name
        hf_name = cls.SUPPORTED_DATASETS.get(dataset_name.lower())
        if not hf_name:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        # Load dataset
        if dataset_name == 'hotpotqa':
            dataset = load_dataset('hotpot_qa', 'fullwiki', split=split, cache_dir=cache_dir)
        elif dataset_name == 'natural_questions':
            dataset = load_dataset('natural_questions', split=split, cache_dir=cache_dir)
        elif dataset_name == 'triviaqa':
            dataset = load_dataset('trivia_qa', 'unfiltered', split=split, cache_dir=cache_dir)
        else:
            dataset = load_dataset(hf_name, split=split, cache_dir=cache_dir)
        
        # Convert to standard format
        examples = []
        for idx, item in enumerate(dataset):
            if limit and idx >= limit:
                break
            
            example = cls._convert_to_standard_format(item, dataset_name)
            example['_id'] = f"{dataset_name}_{split}_{idx}"
            examples.append(example)
        
        return examples
    
    @classmethod
    def _convert_to_standard_format(
        cls,
        item: Dict[str, Any],
        dataset_name: str
    ) -> Dict[str, Any]:
        """Convert dataset item to standard format."""
        if dataset_name in ['hotpotqa', 'hotpot_qa']:
            return {
                'question': item['question'],
                'answer': item['answer'],
                'type': item['type'],
                'level': item['level'],
                'supporting_facts': item['supporting_facts'],
                'context': list(zip(item['context']['title'], item['context']['sentences']))
            }
        
        elif dataset_name == 'natural_questions':
            # Extract short answer
            answer = ''
            if item['annotations']['short_answers']:
                start = item['annotations']['short_answers'][0]['start_token']
                end = item['annotations']['short_answers'][0]['end_token']
                tokens = item['document']['tokens'][start:end]
                answer = ' '.join(t['token'] for t in tokens if not t['is_html'])
            
            return {
                'question': item['question']['text'],
                'answer': answer,
                'context': []  # Would need to process document HTML
            }
        
        elif dataset_name == 'triviaqa':
            return {
                'question': item['question'],
                'answer': item['answer']['value'] if isinstance(item['answer'], dict) else item['answer'],
                'context': []  # Would need to process evidence
            }
        
        elif dataset_name in ['squad', 'squad_v2']:
            answers = item.get('answers', {})
            answer_text = answers['text'][0] if answers.get('text') else ''
            
            return {
                'question': item['question'],
                'answer': answer_text,
                'context': [['Context', [item['context']]]]
            }
        
        else:
            # Generic conversion
            return {
                'question': item.get('question', item.get('query', '')),
                'answer': item.get('answer', item.get('answers', '')),
                'context': item.get('context', [])
            }
    
    @classmethod
    def save_shard(
        cls,
        examples: List[Dict[str, Any]],
        output_path: str
    ) -> None:
        """Save examples as a shard."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(examples, f, indent=2)
    
    @classmethod
    def create_shards(
        cls,
        dataset_name: str,
        split: str,
        shard_size: int = 100,
        output_dir: str = 'runs/_shards',
        limit: Optional[int] = None
    ) -> List[str]:
        """Create data shards for parallel processing."""
        # Load full dataset
        examples = cls.load_dataset(dataset_name, split, limit)
        
        # Create shards
        shard_paths = []
        for i in range(0, len(examples), shard_size):
            shard = examples[i:i + shard_size]
            start_idx = i
            end_idx = min(i + shard_size, len(examples)) - 1
            
            shard_path = f"{output_dir}/{split}_{start_idx}_{end_idx}.json"
            cls.save_shard(shard, shard_path)
            shard_paths.append(shard_path)
        
        return shard_paths