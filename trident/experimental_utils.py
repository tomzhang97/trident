"""
Experimental utilities for rigorous evaluation and reporting.

Provides:
1. Certificate Audit Table - Safe-Cover metrics per dataset
2. Calibration Provenance - Corpus tracking and leakage validation
3. Statistical Uncertainty - Bootstrap CIs, latency percentiles
4. Baseline Protocol Table - Fair comparison documentation

Per reviewer feedback: Ensures reproducibility and statistical rigor.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Sequence
import numpy as np


# =============================================================================
# 1. Certificate Audit Table
# =============================================================================

@dataclass
class CertificateAuditMetrics:
    """
    Aggregated metrics from coverage certificates for audit reporting.

    Per reviewer: "certificate rate, fraction of randomized p-values,
    per-bin n_b coverage (min/median), near-threshold counts,
    % of abstentions caused by dual-LB vs. no-cover, and average LB margin."
    """
    # Certificate counts
    total_queries: int = 0
    queries_with_certificates: int = 0
    total_certificates: int = 0

    # P-value mode breakdown
    deterministic_pvalues: int = 0
    randomized_pvalues: int = 0

    # Per-bin statistics
    bin_sizes: Dict[str, List[int]] = field(default_factory=lambda: defaultdict(list))
    bin_pvalues: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))

    # Near-threshold analysis (p-value within 10% of threshold)
    near_threshold_count: int = 0
    near_threshold_margin: float = 0.10  # 10% of threshold

    # Abstention breakdown
    abstentions_dual_lb: int = 0
    abstentions_no_cover: int = 0
    abstentions_budget_exhausted: int = 0
    abstentions_pvalue_infeasible: int = 0

    # LB margin statistics
    lb_margins: List[float] = field(default_factory=list)

    # Coverage statistics
    facet_coverage_rates: List[float] = field(default_factory=list)

    @property
    def certificate_rate(self) -> float:
        """Fraction of queries that produced certificates."""
        if self.total_queries == 0:
            return 0.0
        return self.queries_with_certificates / self.total_queries

    @property
    def randomized_fraction(self) -> float:
        """Fraction of p-values that used randomized mode."""
        total = self.deterministic_pvalues + self.randomized_pvalues
        if total == 0:
            return 0.0
        return self.randomized_pvalues / total

    @property
    def abstention_breakdown(self) -> Dict[str, float]:
        """Percentage breakdown of abstention causes."""
        total = (self.abstentions_dual_lb + self.abstentions_no_cover +
                 self.abstentions_budget_exhausted + self.abstentions_pvalue_infeasible)
        if total == 0:
            return {"dual_lb": 0.0, "no_cover": 0.0, "budget_exhausted": 0.0, "pvalue_infeasible": 0.0}
        return {
            "dual_lb": self.abstentions_dual_lb / total,
            "no_cover": self.abstentions_no_cover / total,
            "budget_exhausted": self.abstentions_budget_exhausted / total,
            "pvalue_infeasible": self.abstentions_pvalue_infeasible / total,
        }

    @property
    def avg_lb_margin(self) -> float:
        """Average lower bound margin (budget - LB)."""
        if not self.lb_margins:
            return 0.0
        return float(np.mean(self.lb_margins))

    def get_bin_coverage_stats(self) -> Dict[str, Dict[str, float]]:
        """Get min/median n_b per bin."""
        stats = {}
        for bin_key, sizes in self.bin_sizes.items():
            if sizes:
                stats[bin_key] = {
                    "min": float(np.min(sizes)),
                    "median": float(np.median(sizes)),
                    "p5": float(np.percentile(sizes, 5)),
                    "count": len(sizes),
                }
        return stats

    def to_audit_table(self) -> Dict[str, Any]:
        """Generate audit table for reporting."""
        bin_stats = self.get_bin_coverage_stats()

        # Compute overall bin coverage summary
        all_mins = [s["min"] for s in bin_stats.values()] if bin_stats else [0]
        all_medians = [s["median"] for s in bin_stats.values()] if bin_stats else [0]

        return {
            "certificate_metrics": {
                "total_queries": self.total_queries,
                "certificate_rate": round(self.certificate_rate, 3),
                "total_certificates": self.total_certificates,
                "avg_certificates_per_query": round(
                    self.total_certificates / max(self.queries_with_certificates, 1), 2
                ),
            },
            "pvalue_modes": {
                "deterministic_count": self.deterministic_pvalues,
                "randomized_count": self.randomized_pvalues,
                "randomized_fraction": round(self.randomized_fraction, 3),
            },
            "bin_coverage": {
                "min_n_b_across_bins": round(float(np.min(all_mins)), 1),
                "median_n_b_across_bins": round(float(np.median(all_medians)), 1),
                "bins_with_data": len(bin_stats),
                "per_bin": bin_stats,
            },
            "threshold_analysis": {
                "near_threshold_count": self.near_threshold_count,
                "near_threshold_margin": self.near_threshold_margin,
            },
            "abstention_breakdown": {
                k: round(v * 100, 1) for k, v in self.abstention_breakdown.items()
            },
            "lb_margin": {
                "avg_margin": round(self.avg_lb_margin, 2),
                "n_observations": len(self.lb_margins),
            },
            "coverage": {
                "mean_facet_coverage": round(
                    float(np.mean(self.facet_coverage_rates)) if self.facet_coverage_rates else 0.0, 3
                ),
            },
        }

    def to_markdown_table(self) -> str:
        """Generate markdown table for paper/docs."""
        audit = self.to_audit_table()

        lines = [
            "## Certificate Audit Table",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Certificate Rate | {audit['certificate_metrics']['certificate_rate']:.1%} |",
            f"| Total Certificates | {audit['certificate_metrics']['total_certificates']} |",
            f"| Randomized P-values | {audit['pvalue_modes']['randomized_fraction']:.1%} |",
            f"| Min n_b (across bins) | {audit['bin_coverage']['min_n_b_across_bins']:.0f} |",
            f"| Median n_b (across bins) | {audit['bin_coverage']['median_n_b_across_bins']:.0f} |",
            f"| Near-threshold Count | {audit['threshold_analysis']['near_threshold_count']} |",
            f"| Abstentions: Dual-LB | {audit['abstention_breakdown']['dual_lb']:.1f}% |",
            f"| Abstentions: No-Cover | {audit['abstention_breakdown']['no_cover']:.1f}% |",
            f"| Avg LB Margin | {audit['lb_margin']['avg_margin']:.2f} |",
            f"| Mean Facet Coverage | {audit['coverage']['mean_facet_coverage']:.1%} |",
        ]
        return "\n".join(lines)


def aggregate_certificate_audit(
    results: List[Dict[str, Any]],
    budget_cap: Optional[int] = None,
) -> CertificateAuditMetrics:
    """
    Aggregate certificate metrics from evaluation results.

    Args:
        results: List of per-query result dictionaries containing:
            - certificates: List of certificate dicts
            - abstention_reason: String reason if abstained
            - dual_lower_bound: Float LB value
            - covered_facets: List of covered facet IDs
            - total_facets: Total number of facets
        budget_cap: Token budget cap for margin calculation

    Returns:
        CertificateAuditMetrics with aggregated statistics
    """
    metrics = CertificateAuditMetrics()

    for result in results:
        metrics.total_queries += 1

        # Certificate analysis
        certificates = result.get("certificates", [])
        if not certificates:
            certificates = result.get("metrics", {}).get("certificates_detail", [])
        if certificates:
            metrics.queries_with_certificates += 1
            metrics.total_certificates += len(certificates)

            for cert in certificates:
                # P-value mode
                pvalue_mode = cert.get("pvalue_mode", "deterministic")
                if pvalue_mode == "randomized":
                    metrics.randomized_pvalues += 1
                else:
                    metrics.deterministic_pvalues += 1

                # Bin statistics
                bin_key = cert.get("bin", "default")
                bin_size = cert.get("bin_size", 0)
                metrics.bin_sizes[bin_key].append(bin_size)

                # P-value and threshold
                pvalue = cert.get("p_value", 0.0)
                threshold = cert.get("threshold", 0.1)
                metrics.bin_pvalues[bin_key].append(pvalue)

                # Near-threshold check
                if threshold > 0 and pvalue > 0:
                    margin_ratio = abs(threshold - pvalue) / threshold
                    if margin_ratio <= metrics.near_threshold_margin:
                        metrics.near_threshold_count += 1

        # Abstention analysis
        abstention_reason = result.get(
            "abstention_reason",
            result.get("metrics", {}).get("abstention_reason", "none"),
        )
        if abstention_reason == "infeasibility_proven":
            metrics.abstentions_dual_lb += 1
        elif abstention_reason == "no_covering_passages":
            metrics.abstentions_no_cover += 1
        elif abstention_reason == "budget_exhausted":
            metrics.abstentions_budget_exhausted += 1
        elif abstention_reason == "pvalue_infeasible_small_bin":
            metrics.abstentions_pvalue_infeasible += 1

        # LB margin
        dual_lb = result.get(
            "dual_lower_bound",
            result.get("metrics", {}).get("dual_lower_bound", 0.0),
        )
        if budget_cap is not None and dual_lb < float('inf'):
            margin = budget_cap - dual_lb
            metrics.lb_margins.append(margin)

        # Coverage rate - CRITICAL FIX: Look in metrics dict, compute from utility/num_facets
        result_metrics = result.get("metrics", {})
        # Try to get utility (covered facets count) and num_facets
        utility = result_metrics.get("utility", 0)
        num_facets = result_metrics.get("num_facets", 0)
        # Also check for explicit coverage field
        coverage = result_metrics.get("coverage", None)

        if coverage is not None:
            # Use explicit coverage if available
            metrics.facet_coverage_rates.append(coverage)
        elif num_facets > 0:
            # Compute coverage from utility/num_facets
            metrics.facet_coverage_rates.append(utility / num_facets)

    return metrics


# =============================================================================
# 2. Calibration Provenance
# =============================================================================

@dataclass
class CalibrationProvenance:
    """
    Tracks calibration corpus provenance for reproducibility.

    Per reviewer: "Specify the calibration split, guarantee no leakage
    from eval sets, and report per-bin negatives."
    """
    # Corpus identification
    corpus_name: str = ""
    corpus_version: str = ""
    corpus_hash: str = ""
    corpus_size: int = 0

    # Split information
    calibration_split: str = "train"  # train, dev, or custom
    eval_splits_excluded: List[str] = field(default_factory=lambda: ["dev", "test"])

    # Component versions (must match deployment)
    retriever_version: str = ""
    reranker_version: str = ""
    shortlister_version: str = ""
    verifier_version: str = ""

    # Leakage validation
    leakage_check_performed: bool = False
    leakage_check_result: str = ""  # "passed", "failed", or "not_checked"
    overlapping_ids: List[str] = field(default_factory=list)

    # Per-bin statistics
    bin_statistics: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def validate_no_leakage(
        self,
        calibration_ids: Sequence[str],
        eval_ids: Sequence[str],
    ) -> Tuple[bool, List[str]]:
        """
        Validate no leakage between calibration and evaluation sets.

        Returns: (passed, overlapping_ids)
        """
        cal_set = set(calibration_ids)
        eval_set = set(eval_ids)
        overlap = cal_set & eval_set

        self.leakage_check_performed = True
        self.overlapping_ids = list(overlap)

        if overlap:
            self.leakage_check_result = "failed"
            return False, list(overlap)
        else:
            self.leakage_check_result = "passed"
            return True, []

    def compute_corpus_hash(self, ids: Sequence[str]) -> str:
        """Compute deterministic hash of corpus IDs."""
        sorted_ids = sorted(ids)
        content = json.dumps(sorted_ids)
        self.corpus_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        return self.corpus_hash

    def add_bin_statistics(
        self,
        bin_key: str,
        n_negatives: int,
        n_positives: int,
        score_percentiles: Optional[Dict[str, float]] = None,
    ) -> None:
        """Add statistics for a calibration bin."""
        self.bin_statistics[bin_key] = {
            "n_negatives": n_negatives,
            "n_positives": n_positives,
            "median_negatives": score_percentiles.get("p50", 0.0) if score_percentiles else 0.0,
            "p5_negatives": score_percentiles.get("p5", 0.0) if score_percentiles else 0.0,
            "p95_negatives": score_percentiles.get("p95", 0.0) if score_percentiles else 0.0,
        }

    def get_bin_summary(self) -> Dict[str, Any]:
        """Get summary of per-bin statistics."""
        if not self.bin_statistics:
            return {"bins": 0, "total_negatives": 0, "min_negatives": 0, "median_negatives": 0}

        n_negs = [s["n_negatives"] for s in self.bin_statistics.values()]
        return {
            "bins": len(self.bin_statistics),
            "total_negatives": sum(n_negs),
            "min_negatives": min(n_negs),
            "median_negatives": float(np.median(n_negs)),
            "p5_negatives": float(np.percentile(n_negs, 5)),
        }

    def to_provenance_record(self) -> Dict[str, Any]:
        """Generate provenance record for documentation."""
        bin_summary = self.get_bin_summary()
        leakage_result = (
            self.leakage_check_result if self.leakage_check_performed else "not_checked"
        )
        return {
            "corpus": {
                "name": self.corpus_name,
                "version": self.corpus_version,
                "hash": self.corpus_hash,
                "size": self.corpus_size,
                "split": self.calibration_split,
                "excluded_splits": self.eval_splits_excluded,
            },
            "component_versions": {
                "retriever": self.retriever_version,
                "reranker": self.reranker_version,
                "shortlister": self.shortlister_version,
                "verifier": self.verifier_version,
            },
            "leakage_validation": {
                "performed": self.leakage_check_performed,
                "result": leakage_result,
                "overlapping_count": len(self.overlapping_ids),
            },
            "bin_summary": bin_summary,
        }

    def to_markdown(self) -> str:
        """Generate markdown documentation."""
        record = self.to_provenance_record()

        lines = [
            "## Calibration Provenance",
            "",
            "### Corpus",
            f"- **Name**: {record['corpus']['name']}",
            f"- **Version**: {record['corpus']['version']}",
            f"- **Size**: {record['corpus']['size']} samples",
            f"- **Split**: {record['corpus']['split']}",
            f"- **Excluded from eval**: {', '.join(record['corpus']['excluded_splits'])}",
            f"- **Hash**: `{record['corpus']['hash']}`",
            "",
            "### Component Versions",
            f"- Retriever: `{record['component_versions']['retriever']}`",
            f"- Reranker: `{record['component_versions']['reranker']}`",
            f"- Shortlister: `{record['component_versions']['shortlister']}`",
            f"- Verifier: `{record['component_versions']['verifier']}`",
            "",
            "### Leakage Validation",
            f"- Check performed: {record['leakage_validation']['performed']}",
            f"- Result: **{record['leakage_validation']['result']}**",
            "",
            "### Per-bin Statistics",
            f"- Total bins: {record['bin_summary']['bins']}",
            f"- Total negatives: {record['bin_summary']['total_negatives']}",
            f"- Min negatives per bin: {record['bin_summary']['min_negatives']}",
            f"- Median negatives per bin: {record['bin_summary']['median_negatives']:.1f}",
            f"- 5th percentile: {record['bin_summary']['p5_negatives']:.1f}",
        ]
        return "\n".join(lines)


# =============================================================================
# 3. Statistical Uncertainty
# =============================================================================

@dataclass
class StatisticalResults:
    """
    Statistical results with uncertainty quantification.

    Per reviewer: "Report means +/- 95% CIs (bootstrap over questions),
    keep <= 2 decimals. For latency, show p50/p90/p95. State #seeds."
    """
    # Accuracy metrics
    em_mean: float = 0.0
    em_ci_lower: float = 0.0
    em_ci_upper: float = 0.0

    f1_mean: float = 0.0
    f1_ci_lower: float = 0.0
    f1_ci_upper: float = 0.0

    # Latency percentiles (ms)
    latency_p50: float = 0.0
    latency_p90: float = 0.0
    latency_p95: float = 0.0
    latency_mean: float = 0.0

    # Token usage
    tokens_mean: float = 0.0
    tokens_p50: float = 0.0
    tokens_p95: float = 0.0

    # Sample info
    n_samples: int = 0
    n_bootstrap: int = 1000
    n_seeds: int = 1
    confidence_level: float = 0.95

    def to_report_dict(self) -> Dict[str, Any]:
        """Generate report dictionary with proper precision."""
        return {
            "accuracy": {
                "em": f"{self.em_mean:.2f} +/- [{self.em_ci_lower:.2f}, {self.em_ci_upper:.2f}]",
                "f1": f"{self.f1_mean:.2f} +/- [{self.f1_ci_lower:.2f}, {self.f1_ci_upper:.2f}]",
            },
            "latency_ms": {
                "p50": round(self.latency_p50, 1),
                "p90": round(self.latency_p90, 1),
                "p95": round(self.latency_p95, 1),
                "mean": round(self.latency_mean, 1),
            },
            "tokens": {
                "mean": round(self.tokens_mean, 0),
                "p50": round(self.tokens_p50, 0),
                "p95": round(self.tokens_p95, 0),
            },
            "sample_info": {
                "n_samples": self.n_samples,
                "n_bootstrap": self.n_bootstrap,
                "n_seeds": self.n_seeds,
                "confidence_level": self.confidence_level,
            },
        }

    def to_markdown_row(self, method_name: str) -> str:
        """Generate markdown table row."""
        return (
            f"| {method_name} | "
            f"{self.em_mean:.2f} [{self.em_ci_lower:.2f}, {self.em_ci_upper:.2f}] | "
            f"{self.f1_mean:.2f} [{self.f1_ci_lower:.2f}, {self.f1_ci_upper:.2f}] | "
            f"{self.latency_p50:.0f}/{self.latency_p90:.0f}/{self.latency_p95:.0f} | "
            f"{self.tokens_mean:.0f} |"
        )


def bootstrap_confidence_interval(
    values: Sequence[float],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: Optional[int] = None,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for mean.

    Returns: (mean, ci_lower, ci_upper)
    """
    if len(values) == 0:
        return 0.0, 0.0, 0.0

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    values = np.array(values)
    n = len(values)

    # Bootstrap resampling
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))

    bootstrap_means = np.array(bootstrap_means)

    # Compute percentiles
    alpha = 1 - confidence_level
    ci_lower = float(np.percentile(bootstrap_means, alpha / 2 * 100))
    ci_upper = float(np.percentile(bootstrap_means, (1 - alpha / 2) * 100))
    mean = float(np.mean(values))

    return mean, ci_lower, ci_upper


def compute_statistical_results(
    em_scores: Sequence[float],
    f1_scores: Sequence[float],
    latencies_ms: Sequence[float],
    tokens: Sequence[float],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    n_seeds: int = 1,
    seed: Optional[int] = 42,
) -> StatisticalResults:
    """
    Compute statistical results with uncertainty quantification.

    Args:
        em_scores: Per-query exact match scores (0 or 1)
        f1_scores: Per-query F1 scores
        latencies_ms: Per-query latencies in milliseconds
        tokens: Per-query token counts
        n_bootstrap: Number of bootstrap iterations
        confidence_level: Confidence level for intervals
        n_seeds: Number of random seeds used (for reporting)
        seed: Random seed for reproducibility

    Returns:
        StatisticalResults with all metrics
    """
    # Bootstrap CIs for EM and F1
    em_mean, em_ci_lower, em_ci_upper = bootstrap_confidence_interval(
        em_scores, n_bootstrap, confidence_level, seed
    )
    f1_mean, f1_ci_lower, f1_ci_upper = bootstrap_confidence_interval(
        f1_scores, n_bootstrap, confidence_level, seed
    )

    # Latency percentiles
    latencies = np.array(latencies_ms) if latencies_ms else np.array([0.0])
    latency_p50 = float(np.percentile(latencies, 50))
    latency_p90 = float(np.percentile(latencies, 90))
    latency_p95 = float(np.percentile(latencies, 95))
    latency_mean = float(np.mean(latencies))

    # Token percentiles
    tokens_arr = np.array(tokens) if tokens else np.array([0.0])
    tokens_mean = float(np.mean(tokens_arr))
    tokens_p50 = float(np.percentile(tokens_arr, 50))
    tokens_p95 = float(np.percentile(tokens_arr, 95))

    return StatisticalResults(
        em_mean=em_mean,
        em_ci_lower=em_ci_lower,
        em_ci_upper=em_ci_upper,
        f1_mean=f1_mean,
        f1_ci_lower=f1_ci_lower,
        f1_ci_upper=f1_ci_upper,
        latency_p50=latency_p50,
        latency_p90=latency_p90,
        latency_p95=latency_p95,
        latency_mean=latency_mean,
        tokens_mean=tokens_mean,
        tokens_p50=tokens_p50,
        tokens_p95=tokens_p95,
        n_samples=len(em_scores),
        n_bootstrap=n_bootstrap,
        n_seeds=n_seeds,
        confidence_level=confidence_level,
    )


def generate_results_table_with_ci(
    results: Dict[str, StatisticalResults],
) -> str:
    """Generate markdown results table with confidence intervals."""
    lines = [
        "## Results with Statistical Uncertainty",
        "",
        "| Method | EM [95% CI] | F1 [95% CI] | Latency p50/p90/p95 (ms) | Tokens |",
        "|--------|-------------|-------------|--------------------------|--------|",
    ]

    for method_name, stats in results.items():
        lines.append(stats.to_markdown_row(method_name))

    lines.extend([
        "",
        f"*Bootstrap iterations: {list(results.values())[0].n_bootstrap if results else 1000}. "
        f"Seeds: {list(results.values())[0].n_seeds if results else 1}.*",
    ])

    return "\n".join(lines)


# =============================================================================
# 4. Baseline Protocol Table
# =============================================================================

@dataclass
class BaselineProtocol:
    """
    Protocol specification for a baseline method.

    Per reviewer: "retriever, k, reranker, evidence-cap, generator,
    decode params for each baseline."
    """
    name: str

    # Retrieval
    retriever: str = ""
    retriever_k: int = 0
    reranker: str = ""
    reranker_k: int = 0

    # Evidence
    evidence_cap_tokens: int = 0
    evidence_cap_passages: int = 0
    iterative_retrieval: bool = False
    total_evidence_if_iterative: str = ""  # e.g., "sum across steps"

    # Generator
    generator_model: str = ""
    generator_size: str = ""  # e.g., "7B", "13B"
    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0

    # Special modes
    special_modes: List[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "retrieval": {
                "retriever": self.retriever,
                "k": self.retriever_k,
                "reranker": self.reranker,
                "reranker_k": self.reranker_k,
            },
            "evidence": {
                "cap_tokens": self.evidence_cap_tokens,
                "cap_passages": self.evidence_cap_passages,
                "iterative": self.iterative_retrieval,
                "iterative_accounting": self.total_evidence_if_iterative,
            },
            "generator": {
                "model": self.generator_model,
                "size": self.generator_size,
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
            },
            "special_modes": self.special_modes,
            "notes": self.notes,
        }


# Default baseline protocols for TRIDENT comparison
BASELINE_PROTOCOLS = {
    "TRIDENT": BaselineProtocol(
        name="TRIDENT",
        retriever="Contriever-MS",
        retriever_k=100,
        reranker="NLI-based verifier",
        reranker_k=10,
        evidence_cap_tokens=2048,
        evidence_cap_passages=10,
        iterative_retrieval=False,
        generator_model="Qwen2.5-7B-Instruct",
        generator_size="7B",
        max_new_tokens=512,
        temperature=0.0,
        special_modes=["Safe-Cover", "Budget-aware"],
        notes="Selection-conditional calibration, FWER control",
    ),
    "SelfRAG": BaselineProtocol(
        name="SelfRAG",
        retriever="Contriever-MS",
        retriever_k=10,
        reranker="None (uses reflection tokens)",
        reranker_k=10,
        evidence_cap_tokens=0,  # No explicit cap
        evidence_cap_passages=10,
        iterative_retrieval=True,
        total_evidence_if_iterative="Total across steps (report cumulative evidence tokens)",
        generator_model="SelfRAG-Llama2-7B",
        generator_size="7B",
        max_new_tokens=512,
        temperature=0.0,
        special_modes=["Adaptive retrieval", "Critique tokens"],
        notes="Uses [Retrieval], [Relevant], [Supported] special tokens; report cumulative evidence tokens",
    ),
    "VanillaRAG": BaselineProtocol(
        name="VanillaRAG",
        retriever="TF-IDF",
        retriever_k=5,
        reranker="None",
        reranker_k=5,
        evidence_cap_tokens=0,
        evidence_cap_passages=5,
        iterative_retrieval=False,
        generator_model="GPT-4o-mini / Qwen2.5-7B",
        generator_size="7B",
        max_new_tokens=512,
        temperature=0.0,
        special_modes=[],
        notes="Simple TF-IDF + LLM baseline",
    ),
    "HippoRAG": BaselineProtocol(
        name="HippoRAG",
        retriever="NV-Embed-v2",
        retriever_k=5,
        reranker="PPR-based memory",
        reranker_k=5,
        evidence_cap_tokens=0,
        evidence_cap_passages=5,
        iterative_retrieval=False,
        generator_model="GPT-4o-mini",
        generator_size="N/A",
        max_new_tokens=512,
        temperature=0.0,
        special_modes=["Knowledge graph", "Personalized PageRank"],
        notes="Memory-enhanced with entity extraction",
    ),
    "KET-RAG": BaselineProtocol(
        name="KET-RAG",
        retriever="GraphRAG + Keyword",
        retriever_k=10,
        reranker="Skeleton-based",
        reranker_k=5,
        evidence_cap_tokens=0,
        evidence_cap_passages=5,
        iterative_retrieval=False,
        generator_model="GPT-4o-mini / Qwen2.5-7B",
        generator_size="7B",
        max_new_tokens=512,
        temperature=0.0,
        special_modes=["Skeleton extraction", "Keyword matching"],
        notes="Uses KET-RAG's official pipeline for context",
    ),
}


def generate_baseline_protocol_table(
    protocols: Optional[Dict[str, BaselineProtocol]] = None,
) -> str:
    """Generate markdown table of baseline protocols."""
    if protocols is None:
        protocols = BASELINE_PROTOCOLS

    lines = [
        "## Baseline Protocol Table",
        "",
        "| Method | Retriever | k | Reranker | Evidence Cap | Generator | max_tokens | Decode (T/top_p) | Special |",
        "|--------|-----------|---|----------|--------------|-----------|------------|------------------|---------|",
    ]

    for name, proto in protocols.items():
        evidence = f"{proto.evidence_cap_passages}p"
        if proto.evidence_cap_tokens:
            evidence += f"/{proto.evidence_cap_tokens}t"
        if proto.iterative_retrieval:
            evidence += "*"

        special = ", ".join(proto.special_modes[:2]) if proto.special_modes else "-"
        decode = f"{proto.temperature:.1f}/{proto.top_p:.2f}"

        lines.append(
            f"| {name} | {proto.retriever} | {proto.retriever_k} | "
            f"{proto.reranker[:15]}... | {evidence} | "
            f"{proto.generator_size} | {proto.max_new_tokens} | {decode} | {special} |"
        )

    lines.extend([
        "",
        "*Evidence cap format: passages/tokens. `*` indicates iterative retrieval (total across steps).*",
    ])

    return "\n".join(lines)


def generate_full_protocol_documentation(
    protocols: Optional[Dict[str, BaselineProtocol]] = None,
) -> str:
    """Generate full documentation for all baseline protocols."""
    if protocols is None:
        protocols = BASELINE_PROTOCOLS

    sections = ["# Baseline Protocol Documentation", ""]

    for name, proto in protocols.items():
        sections.extend([
            f"## {name}",
            "",
            "### Retrieval",
            f"- **Retriever**: {proto.retriever}",
            f"- **Top-k**: {proto.retriever_k}",
            f"- **Reranker**: {proto.reranker}",
            f"- **Reranker k**: {proto.reranker_k}",
            "",
            "### Evidence",
            f"- **Token cap**: {proto.evidence_cap_tokens or 'None'}",
            f"- **Passage cap**: {proto.evidence_cap_passages}",
            f"- **Iterative**: {proto.iterative_retrieval}",
        ])

        if proto.iterative_retrieval:
            sections.append(f"- **Iterative accounting**: {proto.total_evidence_if_iterative}")

        sections.extend([
            "",
            "### Generator",
            f"- **Model**: {proto.generator_model}",
            f"- **Size**: {proto.generator_size}",
            f"- **max_new_tokens**: {proto.max_new_tokens}",
            f"- **Temperature**: {proto.temperature}",
            f"- **top_p**: {proto.top_p}",
            "",
        ])

        if proto.special_modes:
            sections.append("### Special Modes")
            for mode in proto.special_modes:
                sections.append(f"- {mode}")
            sections.append("")

        if proto.notes:
            sections.extend([
                "### Notes",
                proto.notes,
                "",
            ])

    return "\n".join(sections)
