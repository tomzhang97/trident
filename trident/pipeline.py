"""Enhanced TRIDENT pipeline implementation with full specification support."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from .config import TridentConfig, SafeCoverConfig, ParetoConfig
from .facets import Facet, FacetMiner, FacetType
from .candidates import Passage
from .calibration import ReliabilityCalibrator, CalibrationMonitor
from .safe_cover import SafeCoverAlgorithm, SafeCoverResult
from .pareto import ParetoKnapsackOptimizer, ParetoResult
from .retrieval import RetrievalResult
from .nli_scorer import NLIScorer
from .llm_interface import LLMInterface
from .monitoring import DriftMonitor
from .logging_utils import TelemetryTracker
from .vqc import VerifierQueryCompiler
from .bwk import BwKController


@dataclass
class PipelineOutput:
    """Complete output from TRIDENT pipeline."""
    answer: str
    selected_passages: List[Dict[str, Any]]
    certificates: Optional[List[Dict[str, Any]]]
    abstained: bool
    tokens_used: int
    latency_ms: float
    metrics: Dict[str, float]
    mode: str
    facets: List[Dict[str, Any]]
    trace: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TwoStageScores:
    """Scores from two-stage scoring process."""
    stage1_scores: Dict[Tuple[str, str], float]  # (passage_id, facet_id) -> score
    stage2_scores: Dict[Tuple[str, str], float]  # CE/NLI scores
    p_values: Dict[Tuple[str, str], float]  # Calibrated p-values


class TridentPipeline:
    """Main TRIDENT pipeline orchestrator."""
    
    def __init__(
        self,
        config: TridentConfig,
        llm: LLMInterface,
        retriever: Any,
        device: str = "cuda:0"
    ):
        self.config = config
        self.llm = llm
        self.retriever = retriever
        self.device = device
        
        # Initialize components
        self.facet_miner = FacetMiner(config)
        self.nli_scorer = NLIScorer(config.nli, device)
        self.calibrator = ReliabilityCalibrator(config.calibration)
        self.telemetry = TelemetryTracker(config.telemetry)
        
        # Initialize mode-specific components
        if config.mode in ["safe_cover", "both"]:
            self.safe_cover_algo = SafeCoverAlgorithm(
                config=config.safe_cover,
                calibrator=self.calibrator
            )
            self.drift_monitor = DriftMonitor(config.safe_cover) if config.safe_cover.monitor_drift else None
            
        if config.mode in ["pareto", "both"]:
            self.pareto_optimizer = ParetoKnapsackOptimizer(config.pareto)
            if config.pareto.use_vqc:
                self.vqc = VerifierQueryCompiler(config, self.nli_scorer)
            if config.pareto.use_bwk:
                self.bwk = BwKController(config.pareto)
        
        # Caches
        self.score_cache: Dict[Tuple[str, str, str], float] = {}  # (passage_id, facet_id, model_version)
        self.retrieval_cache: Dict[str, RetrievalResult] = {}
    
    def process_query(
        self,
        query: str,
        supporting_facts: Optional[List[Tuple[str, int]]] = None,
        context: Optional[List[List[str]]] = None,
        mode: Optional[str] = None
    ) -> PipelineOutput:
        """Process a query through the TRIDENT pipeline."""
        start_time = time.time()
        mode = mode or self.config.mode
        
        # Track telemetry
        self.telemetry.start_query(query)
        
        # Step 1: Facet mining
        facets = self.facet_miner.extract_facets(query, supporting_facts)
        self.telemetry.log("facet_mining", {"num_facets": len(facets)})
        
        # Step 2: Retrieval
        retrieval_result = self._retrieve_passages(query, context)
        passages = retrieval_result.passages
        self.telemetry.log("retrieval", {"num_passages": len(passages)})
        
        # Step 3: Two-stage scoring
        scores = self._two_stage_scoring(passages, facets)
        self.telemetry.log("scoring", {"num_scores": len(scores.stage2_scores)})
        
        # Step 4: Mode-specific selection
        if mode == "safe_cover":
            result = self._run_safe_cover(facets, passages, scores)
        elif mode == "pareto":
            result = self._run_pareto(query, facets, passages, scores)
        elif mode == "both":
            # Run both modes for comparison
            safe_result = self._run_safe_cover(facets, passages, scores)
            pareto_result = self._run_pareto(query, facets, passages, scores)
            # Use Safe-Cover if it didn't abstain, otherwise use Pareto
            result = safe_result if not safe_result['abstained'] else pareto_result
            result['comparison'] = {
                'safe_cover': safe_result,
                'pareto': pareto_result
            }
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # Step 5: Generate answer
        answer = ""
        if not result['abstained']:
            prompt = self.llm.build_multi_hop_prompt(
                query=query,
                passages=result['selected_passages'],
                facets=[f.to_dict() for f in facets]
            )
            llm_output = self.llm.generate(prompt)
            answer = self.llm.extract_answer(llm_output.text)
            tokens_used = llm_output.tokens_used
        else:
            tokens_used = 0
        
        # Calculate total latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Prepare output
        return PipelineOutput(
            answer=answer,
            selected_passages=result['selected_passages'],
            certificates=result.get('certificates'),
            abstained=result['abstained'],
            tokens_used=tokens_used + result.get('retrieval_tokens', 0),
            latency_ms=latency_ms,
            metrics=result.get('metrics', {}),
            mode=mode,
            facets=[f.to_dict() for f in facets],
            trace=self.telemetry.get_trace()
        )
    
    def _retrieve_passages(
        self,
        query: str,
        context: Optional[List[List[str]]] = None
    ) -> RetrievalResult:
        """Retrieve passages for query."""
        # Check cache
        cache_key = query
        if cache_key in self.retrieval_cache:
            return self.retrieval_cache[cache_key]
        
        # If context is provided (e.g., HotpotQA), use it directly
        if context:
            passages = []
            for idx, (title, sentences) in enumerate(context):
                text = " ".join(sentences) if isinstance(sentences, list) else sentences
                passage = Passage(
                    pid=f"context_{idx}",
                    text=text,
                    cost=self.llm.compute_token_cost(text),
                    metadata={'title': title, 'source': 'provided_context'}
                )
                passages.append(passage)
            result = RetrievalResult(passages=passages, scores=[1.0] * len(passages))
        else:
            # Use retriever
            result = self.retriever.retrieve(query, top_k=self.config.retrieval.top_k)
        
        # Cache result
        self.retrieval_cache[cache_key] = result
        return result
    
    def _two_stage_scoring(
        self,
        passages: List[Passage],
        facets: List[Facet]
    ) -> TwoStageScores:
        """Perform two-stage scoring with shortlisting and CE/NLI."""
        stage1_scores = {}
        stage2_scores = {}
        p_values = {}
        
        for facet in facets:
            # Stage 1: Shortlist candidates per facet
            shortlist = self._shortlist_for_facet(passages, facet)
            
            # Stage 2: CE/NLI scoring for shortlisted pairs
            for passage in shortlist:
                cache_key = (passage.pid, facet.facet_id, self.config.calibration.version)
                
                if cache_key in self.score_cache:
                    score = self.score_cache[cache_key]
                else:
                    score = self.nli_scorer.score(passage, facet)
                    self.score_cache[cache_key] = score
                
                stage2_scores[(passage.pid, facet.facet_id)] = score
                
                # Calibrate to p-value
                bucket = self._get_calibration_bucket(passage, facet)
                p_value = self.calibrator.to_pvalue(score, bucket)
                p_values[(passage.pid, facet.facet_id)] = p_value
        
        return TwoStageScores(
            stage1_scores=stage1_scores,
            stage2_scores=stage2_scores,
            p_values=p_values
        )
    
    def _shortlist_for_facet(
        self,
        passages: List[Passage],
        facet: Facet,
        max_candidates: int = 10
    ) -> List[Passage]:
        """Shortlist passages for a facet using bi-encoder or lexical scoring."""
        # Simple lexical scoring for now (can be replaced with bi-encoder)
        scores = []
        for passage in passages:
            score = self._lexical_score(passage.text, facet.get_keywords())
            scores.append((passage, score))
        
        # Sort by score and take top candidates
        scores.sort(key=lambda x: x[1], reverse=True)
        return [p for p, _ in scores[:max_candidates]]
    
    def _lexical_score(self, text: str, keywords: List[str]) -> float:
        """Simple lexical overlap score."""
        if not keywords:
            return 0.0
        text_lower = text.lower()
        matches = sum(1 for kw in keywords if kw.lower() in text_lower)
        return matches / len(keywords)
    
    def _get_calibration_bucket(self, passage: Passage, facet: Facet) -> str:
        """Get calibration bucket for passage-facet pair."""
        # Bucket by facet type and length bin
        facet_type = facet.facet_type
        text_length = len(passage.text.split())
        
        if text_length < 50:
            length_bin = "short"
        elif text_length < 150:
            length_bin = "medium"
        else:
            length_bin = "long"
        
        return f"{facet_type}_{length_bin}"
    
    def _run_safe_cover(
        self,
        facets: List[Facet],
        passages: List[Passage],
        scores: TwoStageScores
    ) -> Dict[str, Any]:
        """Run Safe-Cover mode with certificates."""
        # Monitor drift if enabled
        if self.drift_monitor:
            drift_detected = self.drift_monitor.check_drift(scores.stage2_scores)
            if drift_detected:
                self.telemetry.log("drift_detected", {"timestamp": time.time()})
                # Apply fallback thresholds
                self.safe_cover_algo.apply_fallback()
        
        # Run RC-MCFC algorithm
        result = self.safe_cover_algo.run(
            facets=facets,
            passages=passages,
            p_values=scores.p_values
        )
        
        # Convert to output format
        selected = []
        for passage in result.selected_passages:
            selected.append({
                'pid': passage.pid,
                'text': passage.text,
                'cost': passage.cost,
                'covered_facets': result.coverage_map.get(passage.pid, [])
            })
        
        certificates = []
        for cert in result.certificates:
            certificates.append({
                'facet_id': cert.facet_id,
                'passage_id': cert.passage_id,
                'alpha_bar': cert.alpha_bar,
                'p_value': cert.p_value,
                'timestamp': cert.timestamp,
                'calibrator_version': self.config.calibration.version
            })
        
        return {
            'selected_passages': selected,
            'certificates': certificates,
            'abstained': result.abstained,
            'dual_lower_bound': result.dual_lower_bound,
            'retrieval_tokens': sum(p.cost for p in result.selected_passages),
            'metrics': {
                'coverage': len(result.covered_facets) / len(facets) if facets else 0,
                'efficiency': result.dual_lower_bound / sum(p.cost for p in result.selected_passages) if result.selected_passages else 0
            }
        }
    
    def _run_pareto(
        self,
        query: str,
        facets: List[Facet],
        passages: List[Passage],
        scores: TwoStageScores
    ) -> Dict[str, Any]:
        """Run Pareto-Knapsack mode with optional VQC and BwK."""
        current_passages = passages.copy()
        current_facets = facets.copy()
        vqc_iterations = 0
        
        # BwK controller for action selection
        if self.config.pareto.use_bwk:
            episode_state = {
                'query': query,
                'facets': current_facets,
                'budget_remaining': self.config.pareto.budget,
                'iterations': 0
            }
            self.bwk.start_episode(episode_state)
        
        while vqc_iterations < self.config.pareto.max_vqc_iterations:
            # Run Pareto optimization
            result = self.pareto_optimizer.optimize(
                facets=current_facets,
                passages=current_passages,
                p_values=scores.p_values,
                budget=self.config.pareto.budget
            )
            
            # Check if we should use VQC to improve coverage
            if self.config.pareto.use_vqc and result.uncovered_facets:
                # Use BwK to decide on action
                if self.config.pareto.use_bwk:
                    action = self.bwk.select_action(
                        deficit=result.uncovered_facets,
                        budget_remaining=self.config.pareto.budget - result.total_cost
                    )
                    
                    if action != 'vqc_rewrite':
                        break
                
                # Generate query rewrites for uncovered facets
                rewrites = self.vqc.generate_rewrites(
                    query=query,
                    uncovered_facets=result.uncovered_facets,
                    current_passages=result.selected_passages
                )
                
                # Retrieve new passages
                new_passages = []
                for rewrite in rewrites:
                    retrieval_result = self.retriever.retrieve(
                        rewrite,
                        top_k=self.config.retrieval.top_k // 2  # Fewer for rewrites
                    )
                    new_passages.extend(retrieval_result.passages)
                
                # Update passages and re-score
                current_passages.extend(new_passages)
                new_scores = self._two_stage_scoring(new_passages, current_facets)
                scores.stage2_scores.update(new_scores.stage2_scores)
                scores.p_values.update(new_scores.p_values)
                
                vqc_iterations += 1
            else:
                break
        
        # Update BwK with reward
        if self.config.pareto.use_bwk:
            reward = result.achieved_utility / max(result.total_cost, 1)
            self.bwk.update_reward(reward)
        
        # Convert to output format
        selected = []
        for passage in result.selected_passages:
            selected.append({
                'pid': passage.pid,
                'text': passage.text,
                'cost': passage.cost,
                'utility_contribution': passage.metadata.get('utility', 0)
            })
        
        return {
            'selected_passages': selected,
            'certificates': None,  # No certificates in Pareto mode
            'abstained': False,
            'retrieval_tokens': result.total_cost,
            'metrics': {
                'utility': result.achieved_utility,
                'coverage': len(result.covered_facets) / len(facets) if facets else 0,
                'efficiency': result.achieved_utility / max(result.total_cost, 1),
                'vqc_iterations': vqc_iterations
            }
        }
    
    def get_telemetry(self) -> Dict[str, Any]:
        """Get telemetry data."""
        return self.telemetry.get_summary()
    
    def reset_caches(self) -> None:
        """Reset all caches."""
        self.score_cache.clear()
        self.retrieval_cache.clear()
        self.nli_scorer.clear_cache()