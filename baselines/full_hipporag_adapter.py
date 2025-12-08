"""Full HippoRAG adapter for multi-dataset evaluation."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional

from baselines.full_baseline_interface import (
    BaselineSystem,
    BaselineResponse,
    TokenTracker,
    LatencyTracker,
)

try:
    from hipporag import HippoRAG
    HIPPORAG_AVAILABLE = True
except ImportError:
    HIPPORAG_AVAILABLE = False
    HIPPORAG_IMPORT_ERROR = "HippoRAG not installed. Install with: pip install hipporag"


class FullHippoRAGAdapter(BaselineSystem):
    """
    Full HippoRAG adapter using the official HippoRAG library.

    HippoRAG is a memory-enhanced RAG framework inspired by human long-term memory.
    It combines:
    1. Knowledge graph construction from documents
    2. Personalized PageRank for multi-hop retrieval
    3. LLM-based question answering

    Key features:
    - Improved associativity (multi-hop retrieval)
    - Better sense-making (integrating complex contexts)
    - Continuous knowledge integration across documents

    Supports multiple datasets: HotpotQA, MuSiQue, NarrativeQA
    Supports both OpenAI API and local LLMs
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        embedding_model: str = "nvidia/NV-Embed-v2",
        temperature: float = 0.0,
        max_tokens: int = 500,
        num_to_retrieve: int = 5,
        save_dir: Optional[str] = None,
        use_local_llm: bool = False,
        local_llm_base_url: Optional[str] = None,
        local_llm_model: str = "meta-llama/Llama-3.1-8B-Instruct",
        **kwargs
    ):
        """
        Initialize HippoRAG adapter.

        Args:
            api_key: OpenAI API key (required if use_local_llm=False)
            model: LLM model to use (OpenAI model name or local model)
            embedding_model: Embedding model name (nvidia/NV-Embed-v2, GritLM, Contriever)
            temperature: Sampling temperature
            max_tokens: Max tokens for generation
            num_to_retrieve: Number of passages to retrieve
            save_dir: Directory to save HippoRAG outputs (default: temp dir)
            use_local_llm: Use local LLM instead of OpenAI
            local_llm_base_url: Base URL for local vLLM server (e.g., http://localhost:8000/v1)
            local_llm_model: Model name for local LLM
            **kwargs: Additional config
        """
        super().__init__(name="hipporag", **kwargs)

        if not HIPPORAG_AVAILABLE:
            raise ImportError(HIPPORAG_IMPORT_ERROR)

        self.model = local_llm_model if use_local_llm else model
        self.embedding_model = embedding_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.num_to_retrieve = num_to_retrieve
        self.use_local_llm = use_local_llm

        # Set up save directory
        if save_dir:
            self.save_dir = Path(save_dir)
            self.save_dir.mkdir(parents=True, exist_ok=True)
        else:
            # Use a temporary directory
            self._temp_dir = tempfile.mkdtemp(prefix="hipporag_")
            self.save_dir = Path(self._temp_dir)

        # Set up API key for OpenAI
        if not use_local_llm:
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError(
                    "API key required for OpenAI. Set OPENAI_API_KEY "
                    "or use use_local_llm=True for local models."
                )
            # Set environment variable for HippoRAG
            os.environ["OPENAI_API_KEY"] = self.api_key
            print(f"  [HippoRAG] Using OpenAI: {model}")
        else:
            print(f"  [HippoRAG] Using local LLM: {local_llm_model}")
            if not local_llm_base_url:
                raise ValueError(
                    "local_llm_base_url is required when use_local_llm=True. "
                    "Example: http://localhost:8000/v1"
                )

        # Initialize HippoRAG
        print(f"  [HippoRAG] Initializing with embedding model: {embedding_model}")
        hipporag_kwargs = {
            "save_dir": str(self.save_dir),
            "llm_model_name": self.model,
            "embedding_model_name": self.embedding_model,
        }

        if use_local_llm and local_llm_base_url:
            hipporag_kwargs["llm_base_url"] = local_llm_base_url

        try:
            self.hipporag = HippoRAG(**hipporag_kwargs)
        except Exception as e:
            print(f"Error initializing HippoRAG: {e}")
            print("Make sure you have the required dependencies installed:")
            print("  pip install hipporag")
            raise

        # Track whether we've indexed for current context
        self._indexed_context_hash = None

    def _hash_context(self, context: List[List[str]]) -> str:
        """Create a hash of the context for caching."""
        import hashlib
        context_str = str(context)
        return hashlib.md5(context_str.encode()).hexdigest()

    def answer(
        self,
        question: str,
        context: Optional[List[List[str]]] = None,
        supporting_facts: Optional[List[tuple]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> BaselineResponse:
        """
        Answer a question using HippoRAG.

        Pipeline:
        1. Index documents (build knowledge graph + embeddings)
        2. Retrieve relevant passages using personalized PageRank
        3. Generate answer conditioned on retrieved context

        Note: HippoRAG uses its library's native prompt format (not modified)
        to preserve the original system's behavior. Only answer extraction
        is standardized to match Trident for fair comparison.

        Metrics Separation:
        - tokens_used, latency_ms: Query-only metrics (PRIMARY)
        - stats['indexing_*']: Offline indexing costs
        - stats['total_cost_*']: Full cost including indexing

        Args:
            question: The question to answer
            context: HotpotQA context (list of [title, sentences] pairs)
            supporting_facts: Not used by HippoRAG
            metadata: Optional metadata

        Returns:
            BaselineResponse with separated indexing/query metrics
        """
        # Separate trackers for indexing and querying
        indexing_token_tracker = TokenTracker()
        indexing_latency_tracker = LatencyTracker()

        query_token_tracker = TokenTracker()
        query_latency_tracker = LatencyTracker()

        if not context:
            return BaselineResponse(
                answer="I don't have enough information to answer this question.",
                tokens_used=0,
                latency_ms=0,
                selected_passages=[],
                abstained=True,
                mode="hipporag",
                stats={},
                raw_answer=None,
                extracted_answer=None,
            )

        try:
            # --- PHASE 1: INDEXING (Offline Simulation) ---
            print("  [HippoRAG] Indexing documents (Offline Simulation)...")
            indexing_latency_tracker.start()

            # Prepare documents from context
            docs = []
            for title, sentences in context:
                text = " ".join(sentences) if isinstance(sentences, list) else sentences
                # Optionally prepend title for better context
                doc_text = f"{title}: {text}"
                docs.append(doc_text)

            # Check if we need to re-index (for efficiency)
            context_hash = self._hash_context(context)
            if self._indexed_context_hash != context_hash:
                # Index documents (builds KG + embeddings)
                # Note: HippoRAG indexing can be expensive with LLM calls
                self.hipporag.index(docs=docs)
                self._indexed_context_hash = context_hash
            else:
                print("  [HippoRAG] Using cached index")

            indexing_time_ms = indexing_latency_tracker.stop("indexing")

            # Estimate indexing tokens (rough approximation)
            # HippoRAG uses LLM for entity extraction, assume ~2x doc length
            total_doc_length = sum(len(doc.split()) for doc in docs)
            estimated_indexing_tokens = int(total_doc_length * 2 * 1.3)
            indexing_token_tracker.add_call(
                estimated_indexing_tokens // 2,
                estimated_indexing_tokens // 2,
                "indexing"
            )

            print(f"  [HippoRAG] Index Built in {indexing_time_ms/1000:.2f}s")

            # --- PHASE 2: QUERYING (Online / "Primary" Metric) ---
            query_latency_tracker.start()

            # Retrieve and answer using HippoRAG
            # This combines retrieval + generation in one call
            queries = [question]
            rag_results = self.hipporag.rag_qa(
                queries=queries,
                num_to_retrieve=self.num_to_retrieve
            )

            query_latency_tracker.stop("query_and_generation")

            # Extract answer from results
            if rag_results and len(rag_results) > 0:
                result = rag_results[0]
                raw_answer = result.get("answer", "").strip()

                # Use original HippoRAG answer without Trident-style post-processing
                answer = raw_answer

                # Get retrieved passages if available
                retrieved_passages = result.get("retrieved_passages", [])
            else:
                raw_answer = ""
                answer = "Unable to generate answer."
                retrieved_passages = []

            # Estimate query tokens (question + retrieved context + answer)
            question_tokens = len(question.split()) * 1.3
            answer_tokens = len(answer.split()) * 1.3
            # Retrieved passages contribute to prompt
            retrieved_text_length = sum(
                len(str(p).split()) for p in retrieved_passages[:self.num_to_retrieve]
            )
            context_tokens = retrieved_text_length * 1.3

            query_token_tracker.add_call(
                int(question_tokens + context_tokens),
                int(answer_tokens),
                "query_generation"
            )

            # Format retrieved passages for response
            selected_passages = []
            for i, passage in enumerate(retrieved_passages[:self.num_to_retrieve]):
                passage_text = str(passage) if not isinstance(passage, dict) else passage.get("text", str(passage))
                selected_passages.append({
                    "text": passage_text[:200],
                    "passage_idx": i,
                })

            return BaselineResponse(
                answer=answer,
                # PRIMARY METRICS: Query Only
                tokens_used=query_token_tracker.total_tokens,
                latency_ms=query_latency_tracker.get_total_latency(),
                selected_passages=selected_passages,
                abstained=False,
                mode="hipporag",
                stats={
                    "indexing_latency_ms": indexing_time_ms,
                    "indexing_tokens": indexing_token_tracker.total_tokens,
                    "query_tokens": query_token_tracker.total_tokens,
                    "total_cost_tokens": indexing_token_tracker.total_tokens + query_token_tracker.total_tokens,
                    "num_docs": len(docs),
                    "num_retrieved": len(retrieved_passages),
                },
                # Debugging fields
                raw_answer=raw_answer,
                extracted_answer=answer,
            )

        except Exception as e:
            print(f"Error in HippoRAG processing: {e}")
            import traceback
            traceback.print_exc()

            return BaselineResponse(
                answer="Error processing question.",
                tokens_used=0,
                latency_ms=0,
                selected_passages=[],
                abstained=True,
                mode="hipporag",
                stats={"error": str(e)},
                raw_answer=None,
                extracted_answer=None,
            )

    def get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        return {
            "name": "HippoRAG",
            "version": "full",
            "model": self.model,
            "embedding_model": self.embedding_model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "num_to_retrieve": self.num_to_retrieve,
            "use_local_llm": self.use_local_llm,
        }

    def __del__(self):
        """Cleanup temporary directory if created."""
        if hasattr(self, '_temp_dir'):
            import shutil
            try:
                shutil.rmtree(self._temp_dir)
            except Exception:
                pass
