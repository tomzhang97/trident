"""Vanilla GraphRAG adapter using the original GraphRAG library.

This adapter uses the original GraphRAG code from the graphrag/ folder
to run queries exactly as the original paper intended, with a swappable LLM.
"""

from __future__ import annotations

import sys
import os
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional, AsyncGenerator
from dataclasses import dataclass

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
    import tiktoken
    from graphrag.query.structured_search.local_search.search import LocalSearch
    from graphrag.query.structured_search.global_search.search import GlobalSearch
    from graphrag.query.structured_search.base import SearchResult
    from graphrag.query.context_builder.builders import (
        LocalContextBuilder,
        GlobalContextBuilder,
        ContextBuilderResult,
    )
    from graphrag.query.context_builder.conversation_history import ConversationHistory
    from graphrag.language_model.response.base import BaseModelResponse, BaseModelOutput
    from graphrag.prompts.query.local_search_system_prompt import LOCAL_SEARCH_SYSTEM_PROMPT
    from graphrag.prompts.query.global_search_map_system_prompt import MAP_SYSTEM_PROMPT
    from graphrag.prompts.query.global_search_reduce_system_prompt import REDUCE_SYSTEM_PROMPT
    GRAPHRAG_AVAILABLE = True
except ImportError as e:
    GRAPHRAG_AVAILABLE = False
    GRAPHRAG_IMPORT_ERROR = str(e)

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


class GraphRAGCompatibleChatModel:
    """
    LLM wrapper that conforms to GraphRAG's ChatModel protocol.
    Allows using custom LLMs (local or OpenAI-compatible) with GraphRAG.
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

        # Mock config for protocol compliance
        self.config = None

        if not use_local_llm and OPENAI_AVAILABLE:
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
            )

    def _build_messages(self, prompt: str, history: list | None = None) -> list:
        """Build messages list from prompt and history."""
        messages = []
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})
        return messages

    def _generate(self, messages: list, **kwargs) -> str:
        """Internal generation method."""
        temperature = kwargs.get("model_parameters", {}).get("temperature", self.temperature)
        max_tokens = kwargs.get("model_parameters", {}).get("max_tokens", self.max_tokens)

        if self.use_local_llm and self.local_llm:
            response = self.local_llm.generate(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content

    async def achat(
        self, prompt: str, history: list | None = None, **kwargs
    ) -> BaseModelResponse:
        """Async chat - required by GraphRAG."""
        messages = self._build_messages(prompt, history)
        content = self._generate(messages, **kwargs)
        return BaseModelResponse(
            output=BaseModelOutput(content=content, full_response=None),
            parsed_response=None,
            history=messages + [{"role": "assistant", "content": content}],
        )

    async def achat_stream(
        self, prompt: str, history: list | None = None, **kwargs
    ) -> AsyncGenerator[str, None]:
        """Async streaming chat - required by GraphRAG."""
        messages = self._build_messages(prompt, history)
        content = self._generate(messages, **kwargs)
        # Yield the full response (we don't support true streaming with local LLMs)
        yield content

    def chat(
        self, prompt: str, history: list | None = None, **kwargs
    ) -> BaseModelResponse:
        """Sync chat."""
        messages = self._build_messages(prompt, history)
        content = self._generate(messages, **kwargs)
        return BaseModelResponse(
            output=BaseModelOutput(content=content, full_response=None),
            parsed_response=None,
            history=messages + [{"role": "assistant", "content": content}],
        )

    def chat_stream(
        self, prompt: str, history: list | None = None, **kwargs
    ):
        """Sync streaming chat."""
        messages = self._build_messages(prompt, history)
        content = self._generate(messages, **kwargs)
        yield content


class PrecomputedLocalContextBuilder(LocalContextBuilder):
    """
    A LocalContextBuilder that returns precomputed context.
    Used when GraphRAG indexes are precomputed offline.
    """

    def __init__(self, context_by_qid: Dict[str, str], token_encoder):
        self.context_by_qid = context_by_qid
        self.token_encoder = token_encoder
        self._current_qid = None

    def set_current_qid(self, qid: str):
        """Set the current question ID for context lookup."""
        self._current_qid = qid

    def build_context(
        self,
        query: str,
        conversation_history: ConversationHistory | None = None,
        **kwargs,
    ) -> ContextBuilderResult:
        """Return precomputed context for the current question."""
        context = self.context_by_qid.get(self._current_qid, "")
        return ContextBuilderResult(
            context_chunks=context,
            context_records={},
            llm_calls=0,
            prompt_tokens=len(self.token_encoder.encode(context)) if context else 0,
            output_tokens=0,
        )


class PrecomputedGlobalContextBuilder(GlobalContextBuilder):
    """
    A GlobalContextBuilder that returns precomputed context chunks.
    Used for global search with precomputed community reports.
    """

    def __init__(self, context_by_qid: Dict[str, List[str]], token_encoder):
        self.context_by_qid = context_by_qid
        self.token_encoder = token_encoder
        self._current_qid = None

    def set_current_qid(self, qid: str):
        """Set the current question ID for context lookup."""
        self._current_qid = qid

    async def build_context(
        self,
        query: str,
        conversation_history: ConversationHistory | None = None,
        **kwargs,
    ) -> ContextBuilderResult:
        """Return precomputed context chunks for the current question."""
        context_chunks = self.context_by_qid.get(self._current_qid, [])
        if isinstance(context_chunks, str):
            context_chunks = [context_chunks]
        return ContextBuilderResult(
            context_chunks=context_chunks,
            context_records={},
            llm_calls=0,
            prompt_tokens=sum(len(self.token_encoder.encode(c)) for c in context_chunks),
            output_tokens=0,
        )


class FullGraphRAGAdapter(BaselineSystem):
    """
    Vanilla GraphRAG adapter using the original GraphRAG library.

    This adapter uses the original GraphRAG code from the graphrag/ folder
    including:
    - LocalSearch for entity-focused search
    - GlobalSearch for map-reduce over community reports
    - Original GraphRAG prompts

    The LLM can be swapped to use local models or OpenAI-compatible APIs.
    """

    def __init__(
        self,
        context_file: str,
        search_method: str = "local",
        response_type: str = "Single Sentence",
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
        Initialize vanilla GraphRAG adapter.

        Args:
            context_file: Path to precomputed context JSON from GraphRAG indexing
            search_method: Search method - "local" or "global"
            response_type: Response type description (default: "Single Sentence")
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
        super().__init__(name="graphrag", **kwargs)

        if not GRAPHRAG_AVAILABLE:
            raise ImportError(
                f"GraphRAG imports not available. Check graphrag folder.\n"
                f"Error: {GRAPHRAG_IMPORT_ERROR}"
            )

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_local_llm = use_local_llm
        self.search_method = search_method
        self.response_type = response_type

        # Load precomputed contexts
        self.context_by_qid = self._load_contexts(context_file)
        print(f"  [GraphRAG Vanilla] Loaded {len(self.context_by_qid)} precomputed contexts")

        # Initialize token encoder
        self.token_encoder = tiktoken.get_encoding("cl100k_base")

        # Initialize LLM
        local_llm_wrapper = None
        if use_local_llm:
            if not LOCAL_LLM_AVAILABLE:
                raise ImportError("LocalLLMWrapper not available")
            print(f"  [GraphRAG Vanilla] Using local LLM: {local_llm_model}")
            local_llm_wrapper = LocalLLMWrapper(
                model_name=local_llm_model,
                device=local_llm_device,
                temperature=temperature,
                max_tokens=max_tokens,
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
            )
        else:
            print(f"  [GraphRAG Vanilla] Using OpenAI: {model}")

        # Create GraphRAG-compatible LLM wrapper
        self.llm = GraphRAGCompatibleChatModel(
            model=model,
            api_key=api_key,
            api_base=api_base,
            use_local_llm=use_local_llm,
            local_llm_wrapper=local_llm_wrapper,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Create context builder and search engine based on method
        if search_method == "local":
            self.context_builder = PrecomputedLocalContextBuilder(
                context_by_qid=self.context_by_qid,
                token_encoder=self.token_encoder,
            )
            self.search_engine = LocalSearch(
                model=self.llm,
                context_builder=self.context_builder,
                tokenizer=self._create_tokenizer_wrapper(),
                system_prompt=LOCAL_SEARCH_SYSTEM_PROMPT,
                response_type=response_type,
            )
        else:  # global
            self.context_builder = PrecomputedGlobalContextBuilder(
                context_by_qid=self.context_by_qid,
                token_encoder=self.token_encoder,
            )
            self.search_engine = GlobalSearch(
                model=self.llm,
                context_builder=self.context_builder,
                tokenizer=self._create_tokenizer_wrapper(),
                map_system_prompt=MAP_SYSTEM_PROMPT,
                reduce_system_prompt=REDUCE_SYSTEM_PROMPT,
                response_type=response_type,
            )

        print(f"  [GraphRAG Vanilla] Initialized with {search_method} search engine")

    def _create_tokenizer_wrapper(self):
        """Create a tokenizer wrapper compatible with GraphRAG."""
        encoder = self.token_encoder

        class TiktokenWrapper:
            def encode(self, text: str) -> list:
                return encoder.encode(text)

            def decode(self, tokens: list) -> str:
                return encoder.decode(tokens)

        return TiktokenWrapper()

    def _load_contexts(self, context_file: str) -> Dict[str, Any]:
        """Load precomputed contexts from GraphRAG output."""
        import json
        context_path = Path(context_file)
        if not context_path.exists():
            raise FileNotFoundError(
                f"GraphRAG context file not found: {context_file}\n"
                f"Run the GraphRAG indexing and context creation first."
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
        Answer using vanilla GraphRAG with original search engine.

        Uses LocalSearch or GlobalSearch from the original GraphRAG code
        with the original prompts for answer generation.

        Args:
            question: The question to answer
            context: Ignored (uses precomputed GraphRAG contexts)
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
                mode="graphrag",
                stats={"error": "missing_question_id"},
                raw_answer=None,
                extracted_answer=None,
            )

        # Get precomputed context
        graphrag_context = self.context_by_qid.get(question_id)
        if not graphrag_context:
            return BaselineResponse(
                answer="No precomputed context found",
                tokens_used=0,
                latency_ms=0,
                selected_passages=[],
                abstained=True,
                mode="graphrag",
                stats={"error": "context_not_found", "question_id": question_id},
                raw_answer=None,
                extracted_answer=None,
            )

        try:
            latency_tracker.start()

            # Set current question ID for context builder
            self.context_builder.set_current_qid(question_id)

            # Use original GraphRAG search engine to generate answer
            result = asyncio.run(self.search_engine.search(question))

            latency_tracker.stop("query")

            # Get answer from result
            if isinstance(result.response, str):
                answer = result.response
            elif isinstance(result.response, dict):
                answer = result.response.get("response", str(result.response))
            else:
                answer = str(result.response)

            raw_answer = answer

            # Track tokens
            prompt_tokens = result.prompt_tokens
            completion_tokens = result.output_tokens
            token_tracker.add_call(prompt_tokens, completion_tokens, "query")

            # Prepare selected passages
            context_preview = graphrag_context[:500] if isinstance(graphrag_context, str) else str(graphrag_context)[:500]

            return BaselineResponse(
                answer=answer,
                tokens_used=token_tracker.total_tokens,
                latency_ms=latency_tracker.get_total_latency(),
                selected_passages=[{"text": context_preview}],
                abstained=False,
                mode="graphrag",
                stats={
                    "indexing_latency_ms": 0.0,
                    "indexing_tokens": 0,
                    "query_tokens": token_tracker.total_tokens,
                    "total_cost_tokens": token_tracker.total_tokens,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "llm_calls": result.llm_calls,
                    "completion_time": result.completion_time,
                    "search_method": self.search_method,
                },
                raw_answer=raw_answer,
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
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "use_local_llm": self.use_local_llm,
            "search_method": self.search_method,
            "response_type": self.response_type,
            "num_contexts_loaded": len(self.context_by_qid),
        }