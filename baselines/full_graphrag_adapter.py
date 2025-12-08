"""Vanilla GraphRAG adapter using the official GraphRAG library.

This adapter uses the original GraphRAG code from the graphrag/ folder
to run queries exactly as the original paper intended.
"""

from __future__ import annotations

import sys
import os
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional

from baselines.full_baseline_interface import (
    BaselineSystem,
    BaselineResponse,
    TokenTracker,
    LatencyTracker,
)

# Add the graphrag folder to path for imports
GRAPHRAG_PATH = Path(__file__).parent.parent / "graphrag"
if str(GRAPHRAG_PATH) not in sys.path:
    sys.path.insert(0, str(GRAPHRAG_PATH))

try:
    import pandas as pd
    from graphrag.config.load_config import load_config
    from graphrag.config.resolve_path import resolve_paths
    import graphrag.api as graphrag_api
    GRAPHRAG_AVAILABLE = True
except ImportError as e:
    GRAPHRAG_AVAILABLE = False
    GRAPHRAG_IMPORT_ERROR = str(e)


class FullGraphRAGAdapter(BaselineSystem):
    """
    Vanilla GraphRAG adapter using the official GraphRAG library.

    This adapter uses precomputed indexes from the GraphRAG pipeline
    and runs queries using the original GraphRAG API.

    Supported search methods:
    - local: Entity-focused search using local context
    - global: Map-reduce search over community reports
    """

    def __init__(
        self,
        index_dir: str,
        search_method: str = "local",
        community_level: int = 2,
        response_type: str = "Single Sentence",
        config_file: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize vanilla GraphRAG adapter.

        Args:
            index_dir: Path to GraphRAG index output directory containing parquet files
            search_method: Search method - "local" or "global"
            community_level: Community hierarchy level for search (default: 2)
            response_type: Response type description (default: "Single Sentence")
            config_file: Optional path to GraphRAG settings.yaml
            api_key: OpenAI API key (if not set in environment)
        """
        super().__init__(name="graphrag", **kwargs)

        if not GRAPHRAG_AVAILABLE:
            raise ImportError(
                f"GraphRAG library not available.\n"
                f"Error: {GRAPHRAG_IMPORT_ERROR}"
            )

        self.index_dir = Path(index_dir)
        self.search_method = search_method
        self.community_level = community_level
        self.response_type = response_type

        # Set API key if provided
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

        # Load GraphRAG configuration
        root_dir = self.index_dir.parent if config_file is None else Path(config_file).parent
        self.config = load_config(root_dir=root_dir, config_filepath=config_file)
        resolve_paths(self.config, root_dir)

        # Load index data from parquet files
        self._load_index_data()

        print(f"GraphRAG initialized with method={search_method}, community_level={community_level}")

    def _load_index_data(self):
        """Load precomputed index data from parquet files."""
        try:
            # Load required dataframes
            self.entities_df = pd.read_parquet(self.index_dir / "create_final_entities.parquet")
            self.nodes_df = pd.read_parquet(self.index_dir / "create_final_nodes.parquet")
            self.community_reports_df = pd.read_parquet(self.index_dir / "create_final_community_reports.parquet")

            # Load optional dataframes
            try:
                self.communities_df = pd.read_parquet(self.index_dir / "create_final_communities.parquet")
            except FileNotFoundError:
                self.communities_df = None

            try:
                self.text_units_df = pd.read_parquet(self.index_dir / "create_final_text_units.parquet")
            except FileNotFoundError:
                self.text_units_df = None

            try:
                self.relationships_df = pd.read_parquet(self.index_dir / "create_final_relationships.parquet")
            except FileNotFoundError:
                self.relationships_df = None

            try:
                self.covariates_df = pd.read_parquet(self.index_dir / "create_final_covariates.parquet")
            except FileNotFoundError:
                self.covariates_df = None

            print(f"  [GraphRAG] Loaded index from {self.index_dir}")
            print(f"  [GraphRAG] Entities: {len(self.entities_df)}, Reports: {len(self.community_reports_df)}")

        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"GraphRAG index files not found in {self.index_dir}\n"
                f"You must run the GraphRAG indexing pipeline first:\n"
                f"  graphrag index --root <project_root>\n"
                f"Error: {e}"
            )

    def _count_tokens(self, text: str) -> int:
        """Estimate token count from text."""
        return int(len(text.split()) * 1.3)

    async def _run_local_search(self, query: str) -> tuple:
        """Run local search using GraphRAG API."""
        return await graphrag_api.local_search(
            config=self.config,
            nodes=self.nodes_df,
            entities=self.entities_df,
            community_reports=self.community_reports_df,
            text_units=self.text_units_df,
            relationships=self.relationships_df,
            covariates=self.covariates_df,
            community_level=self.community_level,
            response_type=self.response_type,
            query=query,
        )

    async def _run_global_search(self, query: str) -> tuple:
        """Run global search using GraphRAG API."""
        return await graphrag_api.global_search(
            config=self.config,
            nodes=self.nodes_df,
            entities=self.entities_df,
            communities=self.communities_df,
            community_reports=self.community_reports_df,
            community_level=self.community_level,
            dynamic_community_selection=False,
            response_type=self.response_type,
            query=query,
        )

    def answer(
        self,
        question: str,
        context: Optional[List[List[str]]] = None,
        supporting_facts: Optional[List[tuple]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> BaselineResponse:
        """
        Answer a question using vanilla GraphRAG.

        Note: GraphRAG uses precomputed indexes, so the context parameter is ignored.
        The index should be built beforehand using the GraphRAG indexing pipeline.

        Args:
            question: The question to answer
            context: Ignored (GraphRAG uses precomputed indexes)
            supporting_facts: Ignored
            metadata: Optional metadata

        Returns:
            BaselineResponse with answer and metrics
        """
        token_tracker = TokenTracker()
        latency_tracker = LatencyTracker()

        try:
            latency_tracker.start()

            # Run search based on method
            if self.search_method == "local":
                response, context_data = asyncio.run(self._run_local_search(question))
            elif self.search_method == "global":
                response, context_data = asyncio.run(self._run_global_search(question))
            else:
                raise ValueError(f"Unknown search method: {self.search_method}")

            latency_tracker.stop("query")

            # Extract answer - use original GraphRAG response without post-processing
            if isinstance(response, dict):
                answer = response.get("response", str(response))
            else:
                answer = str(response)

            # Count tokens
            prompt_tokens = self._count_tokens(question)
            completion_tokens = self._count_tokens(answer)
            token_tracker.add_call(prompt_tokens, completion_tokens, "query")

            # Prepare selected passages from context data
            selected_passages = []
            if isinstance(context_data, pd.DataFrame) and len(context_data) > 0:
                for _, row in context_data.head(5).iterrows():
                    text = row.get("text", row.get("content", str(row)))
                    selected_passages.append({"text": str(text)[:500]})

            return BaselineResponse(
                answer=answer,
                tokens_used=token_tracker.total_tokens,
                latency_ms=latency_tracker.get_total_latency(),
                selected_passages=selected_passages,
                abstained=False,
                mode="graphrag",
                stats={
                    "indexing_latency_ms": 0.0,  # Indexing done offline
                    "indexing_tokens": 0,
                    "query_tokens": token_tracker.total_tokens,
                    "total_cost_tokens": token_tracker.total_tokens,
                    "search_method": self.search_method,
                    "community_level": self.community_level,
                },
                raw_answer=str(response),
                extracted_answer=answer,
            )

        except Exception as e:
            print(f"Error in GraphRAG processing: {e}")
            import traceback
            traceback.print_exc()

            return BaselineResponse(
                answer="Error processing question.",
                tokens_used=token_tracker.total_tokens,
                latency_ms=latency_tracker.get_total_latency(),
                selected_passages=[],
                abstained=True,
                mode="graphrag",
                stats={"error": str(e)},
                raw_answer=None,
                extracted_answer=None,
            )

    def get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        return {
            "name": "GraphRAG (Vanilla)",
            "version": "vanilla",
            "index_dir": str(self.index_dir),
            "search_method": self.search_method,
            "community_level": self.community_level,
            "response_type": self.response_type,
            "num_entities": len(self.entities_df) if self.entities_df is not None else 0,
            "num_reports": len(self.community_reports_df) if self.community_reports_df is not None else 0,
        }
