"""Shared helper for creating TridentPipeline with LLM and retriever.

All rebuttal scripts need to instantiate LLM + retriever explicitly
(the old ``TridentPipeline(config)`` one-arg shortcut was removed).
This module provides a single ``create_pipeline`` function so every
experiment doesn't have to duplicate the boilerplate.
"""

from __future__ import annotations

from typing import Any, Optional

from trident.config import TridentConfig
from trident.llm_interface import LLMInterface
from trident.pipeline import TridentPipeline
from trident.retrieval import BM25Retriever, DenseRetriever, HybridRetriever


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
