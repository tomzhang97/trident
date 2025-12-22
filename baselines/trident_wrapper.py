"""Wrapper for TRIDENT to provide unified interface with baselines."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from trident.pipeline import TridentPipeline
from trident.llm_instrumentation import QueryStats, InstrumentedLLM


class TridentSystemWrapper:
    """
    Wrapper for TridentPipeline to provide the same interface as baseline systems.

    This wrapper doesn't change TRIDENT's core logic but ensures token tracking
    is consistent with the baselines for fair comparison.
    """

    def __init__(
        self,
        pipeline: TridentPipeline,
        mode: str = "pareto"
    ):
        """
        Initialize TRIDENT wrapper.

        Args:
            pipeline: The TridentPipeline instance
            mode: TRIDENT mode to use ('safe_cover', 'pareto', or 'both')
        """
        self.pipeline = pipeline
        self.mode = mode

    def answer(
        self,
        question: str,
        context: Optional[List[List[str]]] = None,
        supporting_facts: Optional[List[tuple]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Answer a question using TRIDENT.

        Args:
            question: The question to answer
            context: Optional pre-provided context (for datasets like HotpotQA)
            supporting_facts: Optional supporting facts for facet mining
            metadata: Optional metadata

        Returns:
            Dictionary with answer, tokens_used, latency_ms, and stats
        """
        # Call TRIDENT pipeline directly (it already tracks tokens)
        output = self.pipeline.process_query(
            query=question,
            supporting_facts=supporting_facts,
            context=context,
            mode=self.mode
        )

        # Convert to baseline-compatible format
        return {
            "answer": output.answer,
            "tokens_used": output.tokens_used,
            "latency_ms": output.latency_ms,
            "selected_passages": output.selected_passages,
            "certificates": output.certificates,
            "abstained": output.abstained,
            "mode": f"trident_{self.mode}",
            "stats": {
                **output.metrics,
                "certificates": len(output.certificates) if output.certificates else 0,
                "num_facets": len(output.facets),
            },
        }
