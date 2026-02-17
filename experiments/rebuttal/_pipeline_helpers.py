"""Shared helpers for rebuttal experiment scripts.

Provides:
- ``build_config``: Build a TridentConfig from a config-family name + CLI args.
- ``create_pipeline``: Instantiate LLM + retriever and return a ready pipeline.
"""

from __future__ import annotations

import argparse
from typing import Any, Optional

from trident.config import (
    TridentConfig, SafeCoverConfig, ParetoConfig,
    LLMConfig, RetrievalConfig, EvaluationConfig,
    NLIConfig, CalibrationConfig, TelemetryConfig,
)
from trident.config_families import get_config, ALL_CONFIGS
from trident.llm_interface import LLMInterface
from trident.pipeline import TridentPipeline
from trident.retrieval import BM25Retriever, DenseRetriever, HybridRetriever


# ── Config building ────────────────────────────────────────────────────────

def build_config(
    args: argparse.Namespace,
    *,
    mode: Optional[str] = None,
    safe_cover_overrides: Optional[dict] = None,
    pareto_overrides: Optional[dict] = None,
    retrieval_overrides: Optional[dict] = None,
    nli_overrides: Optional[dict] = None,
    telemetry_overrides: Optional[dict] = None,
) -> TridentConfig:
    """Build a :class:`TridentConfig` from ``--config_family`` + CLI args.

    The config family (e.g. ``pareto_match_500_alpha06``) supplies the
    Pareto/SafeCover preset.  Everything else (LLM, retrieval, NLI,
    calibration, evaluation, telemetry) is filled from *args* and optional
    per-experiment overrides.

    Parameters
    ----------
    args : argparse.Namespace
        Must contain at least ``config_family``, ``model``, ``device``,
        ``dataset``, ``load_in_8bit``.  ``encoder_model`` is optional
        (defaults to ``"facebook/contriever"``).
    mode : str, optional
        Force a specific mode (``"pareto"`` or ``"safe_cover"``).
        If *None*, derived from the config family name.
    safe_cover_overrides, pareto_overrides, retrieval_overrides,
    nli_overrides, telemetry_overrides : dict, optional
        Per-experiment field overrides applied on top of the base preset.
    """
    family_name = args.config_family
    base = get_config(family_name)

    # Determine mode
    if mode is None:
        if family_name.startswith("pareto"):
            mode = "pareto"
        elif family_name.startswith("safe_cover"):
            mode = "safe_cover"
        else:
            raise ValueError(f"Cannot infer mode from config family: {family_name}")

    # Build Pareto / SafeCover configs.
    # The family may be a ParetoConfig while the experiment needs safe_cover
    # (e.g. E1 runs both modes).  In that case we derive the missing config
    # from the family's budget so both modes use consistent token caps.
    if isinstance(base, ParetoConfig):
        if mode == "pareto":
            pareto_cfg = _apply_overrides(base, pareto_overrides)
            safe_cover_cfg = SafeCoverConfig(**(safe_cover_overrides or {}))
        else:  # safe_cover mode requested but family is pareto
            budget = base.budget
            sc_defaults = dict(
                per_facet_alpha=0.05,
                token_cap=budget,
                max_evidence_tokens=base.max_evidence_tokens or budget,
                early_abstain=True,
                fallback_to_pareto=False,
                use_certificates=True,
            )
            if safe_cover_overrides:
                sc_defaults.update(safe_cover_overrides)
            safe_cover_cfg = SafeCoverConfig(**sc_defaults)
            pareto_cfg = ParetoConfig(budget=budget, **(pareto_overrides or {}))
    elif isinstance(base, SafeCoverConfig):
        if mode == "safe_cover":
            safe_cover_cfg = _apply_overrides(base, safe_cover_overrides)
            pareto_cfg = ParetoConfig(**(pareto_overrides or {}))
        else:  # pareto mode requested but family is safe_cover
            budget = base.token_cap or base.max_evidence_tokens or 500
            p_defaults = dict(
                budget=budget,
                max_evidence_tokens=base.max_evidence_tokens or budget,
                max_units=8, stop_on_budget=True,
                use_vqc=False, use_bwk=False,
            )
            if pareto_overrides:
                p_defaults.update(pareto_overrides)
            pareto_cfg = ParetoConfig(**p_defaults)
            safe_cover_cfg = SafeCoverConfig(**(safe_cover_overrides or {}))
    else:
        raise TypeError(f"Unexpected config type from family: {type(base)}")

    device = f"cuda:{args.device}" if args.device >= 0 else "cpu"
    encoder_model = getattr(args, "encoder_model", "facebook/contriever")

    retrieval_kw: dict = dict(method="dense", encoder_model=encoder_model, top_k=100)
    if retrieval_overrides:
        retrieval_kw.update(retrieval_overrides)

    nli_kw: dict = dict(batch_size=32)
    if nli_overrides:
        nli_kw.update(nli_overrides)

    telemetry_kw: dict = dict(enable=True)
    if telemetry_overrides:
        telemetry_kw.update(telemetry_overrides)

    return TridentConfig(
        mode=mode,
        pareto=pareto_cfg,
        safe_cover=safe_cover_cfg,
        llm=LLMConfig(
            model_name=args.model,
            device=device,
            load_in_8bit=getattr(args, "load_in_8bit", False),
        ),
        retrieval=RetrievalConfig(**retrieval_kw),
        nli=NLIConfig(**nli_kw),
        calibration=CalibrationConfig(use_mondrian=True),
        evaluation=EvaluationConfig(dataset=args.dataset),
        telemetry=TelemetryConfig(**telemetry_kw),
    )


def available_families() -> str:
    """Return a comma-separated list of config family names."""
    return ", ".join(sorted(ALL_CONFIGS.keys()))


# ── Pipeline creation ──────────────────────────────────────────────────────

def create_pipeline(
    config: TridentConfig,
    device: str = "cuda:0",
    calibration_path: Optional[str] = None,
) -> TridentPipeline:
    """Instantiate LLM + retriever from *config* and return a ready pipeline."""
    llm = LLMInterface(
        model_name=config.llm.model_name,
        device=device,
        temperature=config.llm.temperature,
        max_new_tokens=config.llm.max_new_tokens,
        load_in_8bit=config.llm.load_in_8bit,
    )

    retriever: Any
    method = config.retrieval.method
    if method == "hybrid":
        retriever = HybridRetriever(
            encoder_model=config.retrieval.encoder_model,
            device=device,
            top_k=config.retrieval.top_k,
        )
    elif method == "sparse":
        retriever = BM25Retriever()
    else:
        retriever = DenseRetriever(
            encoder_model=config.retrieval.encoder_model,
            device=device,
            top_k=config.retrieval.top_k,
        )

    return TridentPipeline(
        config=config,
        llm=llm,
        retriever=retriever,
        device=device,
        calibration_path=calibration_path,
    )


# ── Private helpers ────────────────────────────────────────────────────────

def _apply_overrides(base, overrides: Optional[dict]):
    """Return a copy of *base* dataclass with *overrides* applied."""
    if not overrides:
        return base
    from dataclasses import asdict
    merged = {**asdict(base), **overrides}
    return type(base)(**merged)
