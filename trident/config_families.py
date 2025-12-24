"""Named configuration families for TRIDENT cost-quality experiments.

This module defines concrete config presets for running cost-quality frontier experiments:

1. **Pareto configs**: Target Self-RAG-level costs with higher accuracy
2. **Safe-Cover configs**: Strict risk control with different budget regimes
3. **Self-RAG configs**: Fair baseline comparisons
4. **Dataset-specific helpers**: Create full TridentConfig for different datasets

Usage:
    from trident.config_families import PARETO_CHEAP_1500, SAFE_COVER_EQUAL_2500
     from trident.config_families import make_musique_config

    # Create a TridentConfig with a specific preset
    config = TridentConfig(
        mode="pareto",
        pareto=PARETO_CHEAP_1500,
        ...
    )
    # Or use dataset-specific helper
    config = make_musique_config("pareto_cheap_1500")
"""

from __future__ import annotations

from .config import (
    ParetoConfig, SafeCoverConfig, BaselineConfig,
    TridentConfig, EvaluationConfig, LLMConfig, RetrievalConfig,
    NLIConfig, CalibrationConfig, TelemetryConfig
)

# =============================================================================
# Pareto: "Cheap TRIDENT" configs
# =============================================================================
# Goal: Match or slightly exceed Self-RAG token usage (~1500-2000) with
#       significantly higher EM/F1

PARETO_CHEAP_1500 = ParetoConfig(
    budget=1500,
    max_evidence_tokens=1500,
    max_units=8,
    stop_on_budget=True,
    relaxed_alpha=0.3,
    weight_default=1.0,
    use_vqc=False,  # Disable VQC for cheaper config
    use_bwk=False,  # Disable BwK for cheaper config
    max_vqc_iterations=0,
    bwk_exploration_bonus=0.1,
)

PARETO_CHEAP_2000 = ParetoConfig(
    budget=2000,
    max_evidence_tokens=2000,
    max_units=10,
    stop_on_budget=True,
    relaxed_alpha=0.3,
    weight_default=1.0,
    use_vqc=False,
    use_bwk=False,
    max_vqc_iterations=0,
    bwk_exploration_bonus=0.1,
)

PARETO_MID_2500 = ParetoConfig(
    budget=2500,
    max_evidence_tokens=2500,
    max_units=12,
    stop_on_budget=True,
    relaxed_alpha=0.3,
    weight_default=1.0,
    use_vqc=True,  # Enable VQC for better coverage
    use_bwk=True,
    max_vqc_iterations=2,
    bwk_exploration_bonus=0.1,
)

# Self-RAG token-matching configs: Target ~1450 total tokens (Self-RAG level)
# These aim for total_tokens ≈ 1.5-1.6× budget due to answer overhead
PARETO_MATCH_1500 = ParetoConfig(
    budget=1000,
    max_evidence_tokens=1000,
    max_units=8,
    stop_on_budget=True,
    relaxed_alpha=0.5,  # More lenient to maintain coverage at low budget
    weight_default=1.0,
    use_vqc=False,  # Disable VQC to save tokens
    use_bwk=False,
    max_vqc_iterations=0,
    bwk_exploration_bonus=0.1,
)

PARETO_MATCH_1300 = ParetoConfig(
    budget=800,
    max_evidence_tokens=800,
    max_units=7,
    stop_on_budget=True,
    relaxed_alpha=0.5,
    weight_default=1.0,
    use_vqc=False,
    use_bwk=False,
    max_vqc_iterations=0,
    bwk_exploration_bonus=0.1,
)

PARETO_MATCH_1100 = ParetoConfig(
    budget=650,
    max_evidence_tokens=650,
    max_units=6,
    stop_on_budget=True,
    relaxed_alpha=0.5,
    weight_default=1.0,
    use_vqc=False,
    use_bwk=False,
    max_vqc_iterations=0,
    bwk_exploration_bonus=0.1,
)

# Variant with higher alpha for quality at low budget
PARETO_MATCH_1500_ALPHA06 = ParetoConfig(
    budget=1000,
    max_evidence_tokens=1000,
    max_units=8,
    stop_on_budget=True,
    relaxed_alpha=0.6,  # More lenient threshold for better coverage
    weight_default=1.0,
    use_vqc=False,
    use_bwk=False,
    max_vqc_iterations=0,
    bwk_exploration_bonus=0.1,
)

PARETO_MATCH_500_ALPHA06 = ParetoConfig(
    budget=500,
    max_evidence_tokens=500,
    max_units=5,
    stop_on_budget=True,
    relaxed_alpha=0.6,  # More lenient threshold for better coverage
    weight_default=1.0,
    use_vqc=False,
    use_bwk=False,
    max_vqc_iterations=0,
    bwk_exploration_bonus=0.1,
)

PARETO_MATCH_400 = ParetoConfig(
    budget=400,
    max_evidence_tokens=400,
    max_units=4,
    stop_on_budget=True,
    relaxed_alpha=0.5,
    weight_default=1.0,
    use_vqc=True,
    use_bwk=True,
    max_vqc_iterations=0,
    bwk_exploration_bonus=0.1,
)

# Ablation: Pareto 500 without reranker
PARETO_MATCH_500_NO_RERANK = ParetoConfig(
    budget=500,
    max_evidence_tokens=500,
    max_units=5,
    stop_on_budget=True,
    relaxed_alpha=0.6,
    weight_default=1.0,
    use_vqc=False,
    use_bwk=False,
    max_vqc_iterations=0,
    bwk_exploration_bonus=0.1,
)

# =============================================================================
# Safe-Cover configs: Strict risk control
# =============================================================================

# Loose budget regime: Accept higher cost for stricter risk control
SAFE_COVER_LOOSE_4000 = SafeCoverConfig(
    per_facet_alpha=0.05,  # Stricter risk than default
    max_evidence_tokens=4000,
    max_units=16,
    stop_on_budget=True,
    abstain_on_infeasible=False,  # Accept relaxed solutions if needed
    fallback_to_pareto=True,  # Fall back to Pareto if infeasible
    coverage_threshold=0.15,
    token_cap=4000,
    dual_tolerance=1e-6,
    early_abstain=True,
    use_certificates=True,
    monitor_drift=False,
    psi_threshold=0.5,
)

# Equal-budget regime: Same budget as PARETO_MID_2500, strict risk
SAFE_COVER_EQUAL_2500 = SafeCoverConfig(
    per_facet_alpha=0.02,  # Very strict risk
    max_evidence_tokens=2500,
    max_units=12,
    stop_on_budget=True,
    abstain_on_infeasible=False,  # Allow partial solutions
    fallback_to_pareto=True,  # Fall back to Pareto when Safe-Cover fails
    coverage_threshold=0.15,
    token_cap=2500,
    dual_tolerance=1e-6,
    early_abstain=True,
    use_certificates=True,
    monitor_drift=False,
    psi_threshold=0.5,
)

# Equal-budget regime: Same budget as PARETO_CHEAP_2000
SAFE_COVER_EQUAL_2000 = SafeCoverConfig(
    per_facet_alpha=0.02,
    max_evidence_tokens=2000,
    max_units=10,
    stop_on_budget=True,
    abstain_on_infeasible=False,  # Allow partial solutions
    fallback_to_pareto=True,  # Fall back to Pareto when Safe-Cover fails
    coverage_threshold=0.15,
    token_cap=2000,
    dual_tolerance=1e-6,
    early_abstain=True,
    use_certificates=True,
    monitor_drift=False,
    psi_threshold=0.5,
)

# Ablation: Safe 2000 with NLI threshold 0.8
SAFE_COVER_2000_NLI08 = SafeCoverConfig(
    per_facet_alpha=0.02,
    max_evidence_tokens=2000,
    max_units=10,
    stop_on_budget=True,
    abstain_on_infeasible=False,
    fallback_to_pareto=True,
    coverage_threshold=0.15,
    token_cap=2000,
    dual_tolerance=1e-6,
    early_abstain=True,
    use_certificates=True,
    monitor_drift=False,
    psi_threshold=0.5,
)

# Ablation: Safe 2000 without Mondrian calibration
SAFE_COVER_2000_NO_MONDRIAN = SafeCoverConfig(
    per_facet_alpha=0.02,
    max_evidence_tokens=2000,
    max_units=10,
    stop_on_budget=True,
    abstain_on_infeasible=False,
    fallback_to_pareto=True,
    coverage_threshold=0.15,
    token_cap=2000,
    dual_tolerance=1e-6,
    early_abstain=True,
    use_certificates=True,
    monitor_drift=False,
    psi_threshold=0.5,
)


# =============================================================================
# Self-RAG configs for fair baseline comparison
# =============================================================================

SELFRAG_BASE = BaselineConfig(
    common_k=8,
    selfrag_k=8,
    selfrag_use_critic=False,
    selfrag_allow_oracle_context=False,
    # GraphRAG settings (unchanged)
    graphrag_k=8,
    graphrag_topk_nodes=20,
    graphrag_max_seeds=10,
    graphrag_max_hops=2,
    # KET-RAG settings (unchanged)
    ketrag_k=8,
    ketrag_skeleton_ratio=0.3,
    ketrag_max_skeleton_triples=10,
    ketrag_max_keyword_chunks=5,
)

SELFRAG_STRONG = BaselineConfig(
    common_k=16,
    selfrag_k=16,
    selfrag_use_critic=True,  # Enable critic for stronger baseline
    selfrag_allow_oracle_context=False,
    # GraphRAG settings (unchanged)
    graphrag_k=8,
    graphrag_topk_nodes=20,
    graphrag_max_seeds=10,
    graphrag_max_hops=2,
    # KET-RAG settings (unchanged)
    ketrag_k=8,
    ketrag_skeleton_ratio=0.3,
    ketrag_max_skeleton_triples=10,
    ketrag_max_keyword_chunks=5,
)

SELFRAG_ORACLE = BaselineConfig(
    common_k=8,
    selfrag_k=8,
    selfrag_use_critic=False,
    selfrag_allow_oracle_context=True,  # Upper bound: uses gold context
    # GraphRAG settings (unchanged)
    graphrag_k=8,
    graphrag_topk_nodes=20,
    graphrag_max_seeds=10,
    graphrag_max_hops=2,
    # KET-RAG settings (unchanged)
    ketrag_k=8,
    ketrag_skeleton_ratio=0.3,
    ketrag_max_skeleton_triples=10,
    ketrag_max_keyword_chunks=5,
)


# =============================================================================
# Config registry for easy access
# =============================================================================

PARETO_CONFIGS = {
    "pareto_cheap_1500": PARETO_CHEAP_1500,
    "pareto_cheap_2000": PARETO_CHEAP_2000,
    "pareto_mid_2500": PARETO_MID_2500,
    # Self-RAG token-matching configs
    "pareto_match_1500": PARETO_MATCH_1500,
    "pareto_match_1300": PARETO_MATCH_1300,
    "pareto_match_1100": PARETO_MATCH_1100,
    "pareto_match_400": PARETO_MATCH_400,
    "pareto_match_1500_alpha06": PARETO_MATCH_1500_ALPHA06,
    "pareto_match_500_alpha06": PARETO_MATCH_500_ALPHA06,
    # Ablation configs
    "pareto_match_500_no_rerank": PARETO_MATCH_500_NO_RERANK,
}

SAFE_COVER_CONFIGS = {
    "safe_cover_loose_4000": SAFE_COVER_LOOSE_4000,
    "safe_cover_equal_2500": SAFE_COVER_EQUAL_2500,
    "safe_cover_equal_2000": SAFE_COVER_EQUAL_2000,
    # Ablation configs
    "safe_cover_2000_nli08": SAFE_COVER_2000_NLI08,
    "safe_cover_2000_no_mondrian": SAFE_COVER_2000_NO_MONDRIAN,
}

SELFRAG_CONFIGS = {
    "selfrag_base": SELFRAG_BASE,
    "selfrag_strong": SELFRAG_STRONG,
    "selfrag_oracle": SELFRAG_ORACLE,
}

ALL_CONFIGS = {
    **PARETO_CONFIGS,
    **SAFE_COVER_CONFIGS,
}


def get_config(config_name: str):
    """Get a config by name.

    Args:
        config_name: Name of the config (e.g., "pareto_cheap_1500")

    Returns:
        The config object (ParetoConfig or SafeCoverConfig)

    Raises:
        ValueError: If config_name is not recognized
    """
    if config_name in ALL_CONFIGS:
        return ALL_CONFIGS[config_name]
    else:
        available = ", ".join(ALL_CONFIGS.keys())
        raise ValueError(
            f"Unknown config name: {config_name}. "
            f"Available configs: {available}"
        )


def get_selfrag_config(config_name: str) -> BaselineConfig:
    """Get a Self-RAG config by name.

    Args:
        config_name: Name of the config (e.g., "selfrag_base")

    Returns:
        The BaselineConfig object

    Raises:
        ValueError: If config_name is not recognized
    """
    if config_name in SELFRAG_CONFIGS:
        return SELFRAG_CONFIGS[config_name]
    else:
        available = ", ".join(SELFRAG_CONFIGS.keys())
        raise ValueError(
            f"Unknown Self-RAG config name: {config_name}. "
            f"Available configs: {available}"
        )
    
# =============================================================================
# Dataset-specific config helpers
# =============================================================================

def make_dataset_config(
    config_name: str,
    dataset: str = "hotpotqa",
    model_name: str = "meta-llama/Llama-2-7b-hf",
    device: str = "cuda:0",
) -> TridentConfig:
    """Create a full TridentConfig for a specific dataset.

    Args:
        config_name: Name of the config preset (e.g., "pareto_cheap_1500")
        dataset: Dataset name ("hotpotqa", "musique", etc.)
        model_name: LLM model name
        device: Device to run on

    Returns:
        Complete TridentConfig ready for use

    Example:
        config = make_dataset_config("pareto_cheap_1500", dataset="musique")
    """
    # Get the base config
    base_config = get_config(config_name)

    # Determine mode from config type
    if isinstance(base_config, ParetoConfig):
        mode = "pareto"
        pareto_config = base_config
        safe_cover_config = SafeCoverConfig()
    elif isinstance(base_config, SafeCoverConfig):
        mode = "safe_cover"
        safe_cover_config = base_config
        pareto_config = ParetoConfig()
    else:
        raise ValueError(f"Unexpected config type: {type(base_config)}")

    return TridentConfig(
        mode=mode,
        pareto=pareto_config,
        safe_cover=safe_cover_config,
        llm=LLMConfig(model_name=model_name, device=device),
        evaluation=EvaluationConfig(dataset=dataset),
        baselines=SELFRAG_BASE,
    )


def make_musique_config(
    config_name: str,
    model_name: str = "meta-llama/Llama-2-7b-hf",
    device: str = "cuda:0",
) -> TridentConfig:
    """Create a TridentConfig for MuSiQue dataset.

    Args:
        config_name: Name of the config preset (e.g., "pareto_cheap_1500")
        model_name: LLM model name
        device: Device to run on

    Returns:
        Complete TridentConfig configured for MuSiQue

    Example:
        from trident.config_families import make_musique_config
        config = make_musique_config("pareto_cheap_1500")
    """
    return make_dataset_config(config_name, dataset="musique", model_name=model_name, device=device)


def make_hotpotqa_config(
    config_name: str,
    model_name: str = "meta-llama/Llama-2-7b-hf",
    device: str = "cuda:0",
) -> TridentConfig:
    """Create a TridentConfig for HotpotQA dataset.

    Args:
        config_name: Name of the config preset (e.g., "pareto_cheap_1500")
        model_name: LLM model name
        device: Device to run on

    Returns:
        Complete TridentConfig configured for HotpotQA

    Example:
        from trident.config_families import make_hotpotqa_config
        config = make_hotpotqa_config("pareto_cheap_1500")
    """
    return make_dataset_config(config_name, dataset="hotpotqa", model_name=model_name, device=device)