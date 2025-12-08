"""Vanilla KET-RAG adapter using the original KET-RAG implementation.

This adapter uses the original KET-RAG code from KET-RAG/indexing_sket/
to run queries exactly as the original paper intended.
"""

from __future__ import annotations

import sys
import os
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
import json

from baselines.full_baseline_interface import (
    BaselineSystem,
    BaselineResponse,
    TokenTracker,
    LatencyTracker,
)

# Add the KET-RAG folder to path for imports
KETRAG_PATH = Path(__file__).parent.parent / "KET-RAG"
KETRAG_SKET_PATH = KETRAG_PATH / "indexing_sket"
if str(KETRAG_SKET_PATH) not in sys.path:
    sys.path.insert(0, str(KETRAG_SKET_PATH))
if str(KETRAG_PATH) not in sys.path:
    sys.path.insert(0, str(KETRAG_PATH))

try:
    import tiktoken
    import numpy as np
    from util_v1 import MyLocalSearch, LOCAL_SEARCH_EXACT_SYSTEM_PROMPT, f1_score, exact_match_score
    from graphrag.query.llm.base import BaseLLM
    from graphrag.query.llm.text_utils import num_tokens
    from graphrag.query.structured_search.local_search.mixed_context import LocalSearchMixedContext
    from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
    KETRAG_AVAILABLE = True
except ImportError as e:
    KETRAG_AVAILABLE = False
    KETRAG_IMPORT_ERROR = str(e)

# Import local LLM wrapper
try:
    from baselines.local_llm_wrapper import LocalLLMWrapper
    LOCAL_LLM_AVAILABLE = True
except ImportError:
    LOCAL_LLM_AVAILABLE = False

# Import OpenAI
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class GraphRAGCompatibleLLM(BaseLLM):
    """
    LLM wrapper that conforms to GraphRAG's BaseLLM interface.
    Allows using custom LLMs (local or OpenAI-compatible) with KET-RAG.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        use_local_llm: bool = False,
        local_llm_wrapper: Optional[Any] = None,
        temperature: float = 0.0,
        max_tokens: int = 500,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.api_base = api_base
        self.use_local_llm = use_local_llm
        self.local_llm = local_llm_wrapper
        self.temperature = temperature
        self.max_tokens = max_tokens

        if not use_local_llm and OPENAI_AVAILABLE:
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
            )

    def _do_generate(self, messages, **kwargs) -> str:
        """Internal generation method."""
        if self.use_local_llm and self.local_llm:
            response = self.local_llm.generate(
                messages=messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
            )
            return response
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
            )
            return response.choices[0].message.content

    def generate(
        self,
        messages: list[dict[str, str]],
        streaming: bool = False,
        callbacks: list = None,
        **kwargs
    ) -> str:
        """Synchronous generation."""
        return self._do_generate(messages, **kwargs)

    def stream_generate(
        self,
        messages: list[dict[str, str]],
        callbacks: list = None,
        **kwargs
    ):
        """Synchronous streaming generation - yields full response (no true streaming)."""
        response = self._do_generate(messages, **kwargs)
        yield response

    async def agenerate(
        self,
        messages: list[dict[str, str]],
        streaming: bool = False,
        callbacks: list = None,
        **kwargs
    ) -> str:
        """Async generation - required by GraphRAG."""
        return self._do_generate(messages, **kwargs)

    async def astream_generate(
        self,
        messages: list[dict[str, str]],
        callbacks: list = None,
        **kwargs
    ):
        """Async streaming generation - yields full response (no true streaming)."""
        response = self._do_generate(messages, **kwargs)
        yield response


class FullKETRAGAdapter(BaselineSystem):
    """
    Vanilla KET-RAG adapter using the original KET-RAG implementation.

    This adapter uses the original KET-RAG code from KET-RAG/indexing_sket/
    including:
    - MyLocalSearch for answer generation with precomputed context
    - LOCAL_SEARCH_EXACT_SYSTEM_PROMPT for the system prompt
    - Original KET-RAG context format

    The LLM can be swapped to use local models or OpenAI-compatible APIs.
    """

    def __init__(
        self,
        context_file: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 500,
        use_local_llm: bool = False,
        local_llm_model: str = "Qwen/Qwen2.5-7B-Instruct",
        local_llm_device: str = "cuda:0",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        **kwargs
    ):
        """
        Initialize vanilla KET-RAG adapter.

        Args:
            context_file: Path to precomputed context JSON from KET-RAG create_context.py
            api_key: OpenAI API key (if using OpenAI)
            api_base: OpenAI API base URL (for OpenAI-compatible servers)
            model: LLM model name
            temperature: Sampling temperature
            max_tokens: Max tokens for generation
            use_local_llm: Use local LLM instead of OpenAI
            local_llm_model: HuggingFace model name for local LLM
            local_llm_device: Device for local LLM
            load_in_8bit: Use 8-bit quantization
            load_in_4bit: Use 4-bit quantization
        """
        super().__init__(name="ketrag", **kwargs)

        if not KETRAG_AVAILABLE:
            raise ImportError(
                f"KET-RAG imports not available. Check KET-RAG folder.\n"
                f"Error: {KETRAG_IMPORT_ERROR}"
            )

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_local_llm = use_local_llm

        # Load precomputed contexts
        self.context_by_qid = self._load_contexts(context_file)
        print(f"  [KET-RAG Vanilla] Loaded {len(self.context_by_qid)} precomputed contexts")

        # Initialize token encoder
        self.token_encoder = tiktoken.get_encoding("cl100k_base")

        # Initialize LLM
        local_llm_wrapper = None
        if use_local_llm:
            if not LOCAL_LLM_AVAILABLE:
                raise ImportError("LocalLLMWrapper not available")
            print(f"  [KET-RAG Vanilla] Using local LLM: {local_llm_model}")
            local_llm_wrapper = LocalLLMWrapper(
                model_name=local_llm_model,
                device=local_llm_device,
                temperature=temperature,
                max_tokens=max_tokens,
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
            )
        else:
            print(f"  [KET-RAG Vanilla] Using OpenAI: {model}")

        # Create GraphRAG-compatible LLM wrapper
        self.llm = GraphRAGCompatibleLLM(
            model=model,
            api_key=api_key,
            api_base=api_base,
            use_local_llm=use_local_llm,
            local_llm_wrapper=local_llm_wrapper,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Create a minimal context builder (we use precomputed contexts)
        self.context_builder = LocalSearchMixedContext(
            community_reports=None,
            text_units=None,
            entities=[],
            relationships=None,
            covariates=None,
            entity_text_embeddings=None,
            embedding_vectorstore_key=EntityVectorStoreKey.ID,
            text_embedder=None,
            token_encoder=self.token_encoder,
        )

        # LLM params matching original KET-RAG
        self.llm_params = {
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        # Create search engine using original KET-RAG's MyLocalSearch
        self.search_engine = MyLocalSearch(
            llm=self.llm,
            context_builder=self.context_builder,
            token_encoder=self.token_encoder,
            llm_params=self.llm_params,
            context_builder_params={},
        )

        print(f"  [KET-RAG Vanilla] Initialized with original KET-RAG search engine")

    def _load_contexts(self, context_file: str) -> Dict[str, str]:
        """Load precomputed contexts from KET-RAG output."""
        context_path = Path(context_file)
        if not context_path.exists():
            raise FileNotFoundError(
                f"KET-RAG context file not found: {context_file}\n"
                f"Run the KET-RAG context creation first:\n"
                f"  cd KET-RAG\n"
                f"  python indexing_sket/create_context.py <root_path> keyword 0.5"
            )

        with open(context_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        context_dict = {}
        for entry in data:
            qid = entry.get('id')
            ctx = entry.get('context', '')
            if qid:
                context_dict[qid] = ctx

        return context_dict

    def _count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken."""
        return len(self.token_encoder.encode(text))

    def answer(
        self,
        question: str,
        context: Optional[List[List[str]]] = None,
        supporting_facts: Optional[List[tuple]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> BaselineResponse:
        """
        Answer using vanilla KET-RAG with original search engine.

        Uses MyLocalSearch.asearch_with_context() from the original KET-RAG code
        with LOCAL_SEARCH_EXACT_SYSTEM_PROMPT for answer generation.

        Args:
            question: The question to answer
            context: Ignored (uses precomputed KET-RAG contexts)
            supporting_facts: Ignored
            metadata: Must contain 'question_id' to look up precomputed context

        Returns:
            BaselineResponse with answer and metrics
        """
        token_tracker = TokenTracker()
        latency_tracker = LatencyTracker()

        # Get question ID
        question_id = metadata.get('question_id') if metadata else None
        if not question_id:
            return BaselineResponse(
                answer="Missing question_id in metadata",
                tokens_used=0,
                latency_ms=0,
                selected_passages=[],
                abstained=True,
                mode="ketrag",
                stats={"error": "missing_question_id"},
                raw_answer=None,
                extracted_answer=None,
            )

        # Get precomputed context
        ketrag_context = self.context_by_qid.get(question_id)
        if not ketrag_context:
            return BaselineResponse(
                answer="No precomputed context found",
                tokens_used=0,
                latency_ms=0,
                selected_passages=[],
                abstained=True,
                mode="ketrag",
                stats={"error": "context_not_found", "question_id": question_id},
                raw_answer=None,
                extracted_answer=None,
            )

        try:
            latency_tracker.start()

            # Use original KET-RAG search engine to generate answer
            result = asyncio.run(
                self.search_engine.asearch_with_context(question, ketrag_context)
            )

            latency_tracker.stop("query")

            # Get answer from result
            answer = result.response
            raw_answer = answer

            # Track tokens
            prompt_tokens = result.prompt_tokens
            completion_tokens = result.output_tokens
            token_tracker.add_call(prompt_tokens, completion_tokens, "query")

            return BaselineResponse(
                answer=answer,
                tokens_used=token_tracker.total_tokens,
                latency_ms=latency_tracker.get_total_latency(),
                selected_passages=[{"text": ketrag_context[:500]}],
                abstained=False,
                mode="ketrag",
                stats={
                    "indexing_latency_ms": 0.0,
                    "indexing_tokens": 0,
                    "query_tokens": token_tracker.total_tokens,
                    "total_cost_tokens": token_tracker.total_tokens,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "llm_calls": result.llm_calls,
                    "completion_time": result.completion_time,
                },
                raw_answer=raw_answer,
                extracted_answer=answer,
            )

        except Exception as e:
            print(f"Error in KET-RAG processing: {e}")
            import traceback
            traceback.print_exc()

            return BaselineResponse(
                answer="Error processing question.",
                tokens_used=token_tracker.total_tokens,
                latency_ms=latency_tracker.get_total_latency(),
                selected_passages=[],
                abstained=True,
                mode="ketrag",
                stats={"error": str(e)},
                raw_answer=None,
                extracted_answer=None,
            )

    def get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        return {
            "name": "KET-RAG (Vanilla)",
            "version": "vanilla",
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "use_local_llm": self.use_local_llm,
            "num_contexts_loaded": len(self.context_by_qid),
        }
