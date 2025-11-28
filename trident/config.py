"""Enhanced configuration module for TRIDENT with full specification support."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path


@dataclass
class LLMConfig:
    """Configuration for Large Language Model."""
    model_name: str = "meta-llama/Llama-2-7b-hf"
    temperature: float = 0.0
    max_new_tokens: int = 512
    device: str = "cuda:0"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    batch_size: int = 1
    use_flash_attention: bool = False
    dtype: str = "float16"  # float16, bfloat16, float32


@dataclass
class RetrievalConfig:
    """Configuration for retrieval system."""
    method: str = "dense"  # dense, sparse, hybrid
    encoder_model: str = "facebook/contriever"
    corpus_path: Optional[str] = None
    index_path: Optional[str] = None
    top_k: int = 100
    rerank_top_k: int = 20
    use_reranker: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass 
class FacetConfig:
    """Per-facet statistical controls."""
    alpha: float = 0.01
    max_tests: int = 10
    prefilter_tests: int = 3
    fallback_scale: float = 0.5
    weight: float = 1.0


@dataclass
class SafeCoverConfig:
    """Settings for RC-MCFC Safe-Cover algorithm."""
    per_facet_alpha: float = 0.1  # Default alpha for all facets
    coverage_threshold: float = 0.15
    per_facet_configs: Dict[str, FacetConfig] = field(default_factory=dict)
    token_cap: Optional[int] = 2000
    dual_tolerance: float = 1e-6
    early_abstain: bool = False
    use_certificates: bool = False
    monitor_drift: bool = False
    psi_threshold: float = 0.5
    coverage_threshold: float = 0.15
    # CRITICAL FIX: Add fallback_to_pareto field
    fallback_to_pareto: bool = True # Default to True for backward compatibility/good behavior
    # Budget control fields for config families
    max_evidence_tokens: Optional[int] = None  # Maximum evidence tokens to use
    max_units: Optional[int] = None  # Maximum number of passages/units to select
    stop_on_budget: bool = True  # Stop selection when budget is exhausted
    abstain_on_infeasible: bool = False  # Abstain if constraints cannot be satisfied under budget


@dataclass
class ParetoConfig:
    """Settings for Pareto-Knapsack mode."""
    budget: int = 2000
    relaxed_alpha: float = 0.3
    weight_default: float = 1.0
    use_vqc: bool = True  # Verifier-driven Query Compiler
    use_bwk: bool = True  # Bandits with Knapsacks
    max_vqc_iterations: int = 3
    bwk_exploration_bonus: float = 0.1
    # Budget control fields for config families
    max_evidence_tokens: Optional[int] = None  # Maximum evidence tokens to use
    max_units: Optional[int] = None  # Maximum number of passages/units to select
    stop_on_budget: bool = True  # Stop selection when budget is exhausted


@dataclass
class CalibrationConfig:
    """Settings for score calibration."""
    method: str = "isotonic"  # isotonic, platt, beta
    version: str = "v1.0"
    facet_bins: Dict[str, List[float]] = field(default_factory=dict)
    reliability_table_size: int = 20
    use_mondrian: bool = True
    recalibration_buffer_size: int = 1000


@dataclass
class NLIConfig:
    """Configuration for NLI/Cross-encoder scoring."""
    model_name: str = "microsoft/deberta-v2-xlarge-mnli"
    batch_size: int = 32
    max_length: int = 512
    use_cache: bool = True
    cache_size: int = 10000
    score_threshold: float = 0.9


@dataclass
class EvaluationConfig:
    """Configuration for evaluation metrics."""
    metrics: List[str] = field(default_factory=lambda: ["em", "f1", "support_em", "faithfulness"])
    dataset: str = "hotpotqa"
    save_predictions: bool = True
    save_certificates: bool = True
    compute_pareto_curves: bool = True


@dataclass
class TelemetryConfig:
    """Configuration for telemetry and monitoring."""
    enable: bool = True
    log_level: str = "INFO"
    track_latency: bool = True
    track_gpu_util: bool = True
    track_cache_stats: bool = True
    profile_memory: bool = False
    save_traces: bool = True


@dataclass
class BaselineConfig:
    """Configuration for baseline systems (Self-RAG, GraphRAG, KET-RAG)."""
    # Common settings for fair comparison
    common_k: int = 8  # Shared retrieval k across all baselines

    # Self-RAG settings
    selfrag_k: int = 8  # Number of documents to retrieve
    selfrag_use_critic: bool = False  # Whether to use critic/verification
    selfrag_allow_oracle_context: bool = False  # Allow oracle context or enforce retrieval-only

    # GraphRAG settings
    graphrag_k: int = 8  # Number of documents to retrieve (harmonized with common_k)
    graphrag_topk_nodes: int = 20  # Candidate nodes to consider
    graphrag_max_seeds: int = 10  # Maximum seed nodes
    graphrag_max_hops: int = 2  # Maximum hops for subgraph expansion

    # KET-RAG settings
    ketrag_k: int = 8  # Number of documents to retrieve
    ketrag_skeleton_ratio: float = 0.3  # Ratio of chunks to use for skeleton KG
    ketrag_max_skeleton_triples: int = 10  # Maximum triples from skeleton KG
    ketrag_max_keyword_chunks: int = 5  # Maximum chunks from keyword index


@dataclass
class TridentConfig:
    """Complete TRIDENT system configuration."""
    mode: str = "safe_cover"  # safe_cover, pareto, both
    safe_cover: SafeCoverConfig = field(default_factory=SafeCoverConfig)
    pareto: ParetoConfig = field(default_factory=ParetoConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    nli: NLIConfig = field(default_factory=NLIConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    baselines: BaselineConfig = field(default_factory=BaselineConfig)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TridentConfig":
        """Create config from dictionary."""
        safe_cover = SafeCoverConfig(**config_dict.get("safe_cover", {}))
        pareto = ParetoConfig(**config_dict.get("pareto", {}))
        calibration = CalibrationConfig(**config_dict.get("calibration", {}))
        llm = LLMConfig(**config_dict.get("llm", {}))
        retrieval = RetrievalConfig(**config_dict.get("retrieval", {}))
        nli = NLIConfig(**config_dict.get("nli", {}))
        evaluation = EvaluationConfig(**config_dict.get("evaluation", {}))
        telemetry = TelemetryConfig(**config_dict.get("telemetry", {}))

        # Filter out non-dataclass fields like 'comments' from baselines config
        baselines_dict = config_dict.get("baselines", {})
        baselines_dict = {k: v for k, v in baselines_dict.items() if not k.startswith('_') and k != 'comments'}
        baselines = BaselineConfig(**baselines_dict)

        return cls(
            mode=config_dict.get("mode", "safe_cover"),
            safe_cover=safe_cover,
            pareto=pareto,
            calibration=calibration,
            llm=llm,
            retrieval=retrieval,
            nli=nli,
            evaluation=evaluation,
            telemetry=telemetry,
            baselines=baselines
        )
    
    @classmethod
    def from_file(cls, path: str) -> "TridentConfig":
        """Load configuration from JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))
    
    def save(self, path: str) -> None:
        """Save configuration to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)


@dataclass
class ExperimentConfig:
    """Configuration for running experiments."""
    name: str
    dataset: str
    data_path: str
    output_dir: str
    num_workers: int = 1
    shard_size: int = 100
    checkpoint_interval: int = 100
    resume_from: Optional[str] = None