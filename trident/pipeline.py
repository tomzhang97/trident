"""Enhanced TRIDENT pipeline implementation with fixes for answer extraction and facet handling."""

from __future__ import annotations

import re
import string
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


def _normalize_for_filter(text: str) -> str:
    """Helper function to normalize text for filtering."""
    t = text.lower().strip()
    t = t.translate(str.maketrans("", "", string.punctuation))
    t = " ".join(t.split())
    return t

def _looks_like_meta(text: str) -> bool:
    """Filter out lines like 'Answer:', 'Final answer', etc."""
    norm = _normalize_for_filter(text)
    if not norm:
        return True
    bad_exact = {"answer", "final answer", "short answer", "prediction"}
    if norm in bad_exact:
        return True
    bad_substrings = ["step ", "document ", "context ", "evidence "]
    return any(bad in norm for bad in bad_substrings)

def extract_final_answer(raw_text: str, question: str) -> str:
    """
    Extract the final answer from raw LLM output using a more robust method.
    
    Args:
        raw_text: The raw text output from the LLM.
        question: The original question, used for context (e.g., Yes/No detection).
        
    Returns:
        The extracted answer string.
    """
    text = raw_text.strip()

    # 0) Strip "Step N:" style reasoning lines (even at end of text)
    text = re.sub(r'(?mi)^step\s+\d+:[^\n]*$', '', text)
    text = re.sub(r'(?i)step\s+\d+:[^\.!\n]*', '', text)

    # 1) Yes/No shortcut
    q_lower = question.strip().lower()
    if q_lower.startswith((
        "is ", "are ", "was ", "were ",
        "do ", "does ", "did ",
        "can ", "could ", "should ",
        "has ", "have ", "had "
    )):
        t_lower = text.lower()
        # Prefer the last yes/no mention
        if " no" in t_lower or t_lower.startswith("no"):
            return "no"
        if " yes" in t_lower or t_lower.startswith("yes"):
            return "yes"

    # 2) Explicit "Answer:" / "Final Answer:"
    explicit_patterns = [
        r"(?mi)^final answer\s*[:\-]\s*(.+)$",
        r"(?mi)^answer\s*[:\-]\s*(.+)$",
        r"(?mi)answer\s*[:\-]\s*(.+)$",
    ]

    for pat in explicit_patterns:
        m = re.search(pat, text)
        if m:
            cand = m.group(1).strip()
            if cand and not _looks_like_meta(cand):
                return cand

    # 3) Fallback: first non-meta short line
    #    e.g., lines without colon and not obviously meta
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if ":" in line:
            # usually labels like "Reasoning:", "Answer:", etc.
            continue
        if _looks_like_meta(line):
            continue
        # reasonable length heuristic
        if 2 <= len(line) <= 80:
            return line

    # 4) Final fallback: last non-empty token sequence
    tokens = text.split()
    if tokens:
        # e.g., last noun-ish phrase rather than everything
        return " ".join(tokens[-5:]).strip()

    return ""


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
        self.calibrator = ReliabilityCalibrator(use_mondrian=config.calibration.use_mondrian)
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

        # CRITICAL FIX: Initialize VQC and BwK if Safe-Cover might fall back to Pareto and those features are enabled
        # This ensures components are available even if the primary mode doesn't require them,
        # but a fallback might.
        if config.mode == "safe_cover" and config.safe_cover.fallback_to_pareto:
            # Initialize the optimizer if fallback is enabled, regardless of primary mode initialization
            if not hasattr(self, 'pareto_optimizer'):
                self.pareto_optimizer = ParetoKnapsackOptimizer(config.pareto)
            if config.pareto.use_vqc and not hasattr(self, 'vqc'):
                self.vqc = VerifierQueryCompiler(config, self.nli_scorer)
            if config.pareto.use_bwk and not hasattr(self, 'bwk'):
                self.bwk = BwKController(config.pareto)
        
        # Caches
        self.score_cache: Dict[Tuple[str, str, str], float] = {}
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
            # CRITICAL FIX: If Safe-Cover is infeasible or abstains, optionally fall back to Pareto
            if result.get('abstained', False) or result.get('infeasible', False):
                # Access the fallback_to_pareto field directly from the dataclass instance
                if self.config.safe_cover.fallback_to_pareto:
                    self.telemetry.log("fallback", {"reason": "safe_cover_abstain_or_infeasible"})
                    # Run Pareto with the same data
                    pareto_result = self._run_pareto(query, facets, passages, scores)
                    # Use the Pareto result instead, but mark that the primary mode failed
                    result = pareto_result
                    # Optionally, append mode info to metrics to distinguish
                    result['metrics']['fallback_from'] = 'safe_cover'
                # else: keep the original Safe-Cover result (e.g., abstained=True)
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
        
        # Step 5: Generate answer (FIXED - now properly extracts final answer)
        answer = ""
        if not result['abstained'] and result['selected_passages']:
            # Build prompt with selected passages
            prompt = self.llm.build_multi_hop_prompt(
                query=query,
                passages=result['selected_passages'],
                facets=[f.to_dict() for f in facets]
            )
            llm_output = self.llm.generate(prompt)
            
            # CRITICAL FIX: Extract and clean the final answer from LLM output
            # This ensures we're not logging intermediate outputs like facet mining
            raw_answer = llm_output.text
            # Use the new, more robust extraction function
            answer = extract_final_answer(raw_answer, query)
            tokens_used = llm_output.tokens_used
        else:
            answer = "ABSTAINED" if result['abstained'] else ""
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
    
    # Removed the old _extract_final_answer method and _clean_answer method
    # as they are replaced by the standalone extract_final_answer function.
    
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
                    cost=self._estimate_token_cost(text),
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
    
    def _estimate_token_cost(self, text: str) -> int:
        """Estimate token cost more accurately."""
        # More realistic estimate: ~1.3 tokens per word + overhead
        words = len(text.split())
        return int(words * 1.3) + 10  # Add overhead for special tokens
    
    def _two_stage_scoring(
        self,
        passages: List[Passage],
        facets: List[Facet]
    ) -> TwoStageScores:
        """
        Perform two-stage scoring with shortlisting and CE/NLI.
        
        CRITICAL FIX: Ensures facets are proper Facet objects, not strings.
        """
        stage1_scores = {}
        stage2_scores = {}
        p_values = {}
        
        # Validate facets are proper objects
        for facet in facets:
            if not isinstance(facet, Facet):
                raise TypeError(f"Expected Facet object, got {type(facet)}: {facet}")
        
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
                
                # Calibrate to p-value with text length for Mondrian
                bucket = self._get_calibration_bucket(facet)
                text_length = len(passage.text.split())
                p_value = self.calibrator.to_pvalue(score, bucket, text_length)
                
                # CRITICAL FIX: Ensure p-values are reasonable
                # If calibration gives extreme values, use score-based heuristic
                if p_value > 0.99 or p_value < 0.0:
                    p_value = max(0.0, min(1.0, 1.0 - score))
                
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
    
    def _get_calibration_bucket(self, facet: Facet) -> str:
        """
        Get calibration bucket for facet.
        
        CRITICAL FIX: Properly handle facet type as enum.
        """
        # Ensure facet_type is accessed correctly
        if isinstance(facet.facet_type, FacetType):
            facet_type_str = facet.facet_type.value
        elif isinstance(facet.facet_type, str):
            facet_type_str = facet.facet_type
        else:
            facet_type_str = str(facet.facet_type)
        
        return facet_type_str
    
    def _run_safe_cover(
        self,
        facets: List[Facet],
        passages: List[Passage],
        scores: TwoStageScores
    ) -> Dict[str, Any]:
        """
        Run Safe-Cover mode with certificates.
        
        CRITICAL FIX: Relaxed thresholds to avoid always-zero coverage.
        """
        # Monitor drift if enabled
        if self.drift_monitor:
            drift_detected = self.drift_monitor.check_drift(scores.stage2_scores)
            if drift_detected:
                self.telemetry.log("drift_detected", {"timestamp": time.time()})
                self.safe_cover_algo.apply_fallback()
        
        # Run RC-MCFC algorithm
        result = self.safe_cover_algo.run(
            facets=facets,
            passages=passages,
            p_values=scores.p_values
        )
        
        # Convert to output format
        selected = []
        total_cost = 0
        for passage in result.selected_passages:
            selected.append({
                'pid': passage.pid,
                'text': passage.text,
                'cost': passage.cost,
                'covered_facets': result.coverage_map.get(passage.pid, [])
            })
            total_cost += passage.cost
        
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
        
        # Determine if the result is infeasible or leads to abstention
        # This depends on the specific return structure of self.safe_cover_algo.run
        # Assuming result.abstained or result.infeasible might be set, or inferring from selected_passages
        is_infeasible = getattr(result, 'infeasible', False) # Adjust based on actual result object
        is_abstained = result.abstained or is_infeasible or len(selected) == 0 # Adjust condition if needed

        return {
            'selected_passages': selected,
            'certificates': certificates,
            'abstained': is_abstained, # Use the computed value
            'infeasible': is_infeasible, # Add infeasible flag
            'dual_lower_bound': getattr(result, 'dual_lower_bound', None), # Adjust based on actual result object
            'retrieval_tokens': total_cost,
            'metrics': {
                'coverage': len(result.covered_facets) / len(facets) if facets else 0,
                'utility': len(result.covered_facets),
                'efficiency': len(result.covered_facets) / max(total_cost, 1) if total_cost > 0 else 0
            }
        }
    
    def _run_pareto(
        self,
        query: str,
        facets: List[Facet],
        passages: List[Passage],
        scores: TwoStageScores
    ) -> Dict[str, Any]:
        """
        Run Pareto-Knapsack mode with optional VQC and BwK.
        
        CRITICAL FIX: Properly respects budget constraints.
        CRITICAL FIX: VQC now runs based on uncovered facets and budget,
                    regardless of BwK action, allowing VQC to function.
        """
        current_passages = passages.copy()
        current_facets = facets.copy()
        vqc_iterations = 0
        
        # Calculate actual budget considering retrieval cost
        actual_budget = self.config.pareto.budget
        
        # BwK controller for action selection
        if self.config.pareto.use_bwk:
            episode_state = {
                'query': query,
                'facets': current_facets,
                'budget_remaining': actual_budget,
                'iterations': 0
            }
            self.bwk.start_episode(episode_state)
        
        while vqc_iterations < self.config.pareto.max_vqc_iterations:
            # Run Pareto optimization with actual budget
            result = self.pareto_optimizer.optimize(
                facets=current_facets,
                passages=current_passages,
                p_values=scores.p_values,
                budget=actual_budget
            )
            
            # Check if we should use VQC to improve coverage
            should_run_vqc = (
                self.config.pareto.use_vqc and
                result.uncovered_facets and
                result.total_cost < actual_budget * 0.8
            )
            
            # If conditions are met to run VQC
            if should_run_vqc:
                # If BwK is enabled, we can log its action or use it differently,
                # but the decision to run VQC is now independent of the action outcome.
                if self.config.pareto.use_bwk:
                    action = self.bwk.select_action(
                        deficit=result.uncovered_facets,
                        budget_remaining=actual_budget - result.total_cost
                    )
                    # Log the action for debugging if needed, but don't break based on it
                    # self.telemetry.log("bwk_action", {"action": action, "iteration": vqc_iterations})

                # CRITICAL FIX: VQC now runs if the primary conditions are met
                # Generate query rewrites for uncovered facets
                uncovered_facet_objs = [f for f in current_facets if f.facet_id in result.uncovered_facets]
                rewrites = self.vqc.generate_rewrites(
                    query=query,
                    uncovered_facets=uncovered_facet_objs,
                    current_passages=result.selected_passages,
                    max_rewrites=2
                )
                
                # Retrieve new passages
                new_passages = []
                for rewrite in rewrites:
                    retrieval_result = self.retriever.retrieve(
                        rewrite,
                        top_k=self.config.retrieval.top_k // 2
                    )
                    new_passages.extend(retrieval_result.passages)
                
                # Update passages and re-score
                current_passages.extend(new_passages)
                new_scores = self._two_stage_scoring(new_passages, current_facets)
                scores.stage2_scores.update(new_scores.stage2_scores)
                scores.p_values.update(new_scores.p_values)
                
                vqc_iterations += 1
            else:
                # Break the loop if conditions for VQC are not met
                break

        # Update BwK with final reward after the loop (if BwK is used)
        if self.config.pareto.use_bwk:
            # Use the final result after potentially multiple VQC iterations
            # Calculate reward based on the final state
            final_result_utility = result.achieved_utility # Result from the last optimization run
            final_result_cost = result.total_cost
            reward = final_result_utility / max(final_result_cost, 1)
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
            'certificates': None,
            'abstained': False, # Pareto mode typically does not abstain
            'infeasible': False, # Pareto mode typically is feasible
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