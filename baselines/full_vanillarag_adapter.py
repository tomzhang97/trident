"""Full Vanilla RAG adapter for baseline comparison."""

from __future__ import annotations

import os
from typing import Dict, Any, List, Optional
from collections import defaultdict
import re

from baselines.full_baseline_interface import (
    BaselineSystem,
    BaselineResponse,
    TokenTracker,
    LatencyTracker,
)
from baselines.local_llm_wrapper import LocalLLMWrapper

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from graphrag.config.enums import ModelType
    from graphrag.config.models.language_model_config import LanguageModelConfig
    from graphrag.language_model.manager import ModelManager
    GRAPHRAG_AVAILABLE = True
except ImportError:
    GRAPHRAG_AVAILABLE = False


class OpenAILLMWrapper:
    """Wrapper for GraphRAG 2.7.0 chat model to provide compatible generate() method."""

    def __init__(self, chat_model):
        self.chat_model = chat_model

    def generate(self, messages, temperature=0.0, max_tokens=500, **kwargs):
        """Generate a response using the chat model."""
        if isinstance(messages, list) and len(messages) > 0 and isinstance(messages[0], dict):
            chat_messages = messages
        else:
            chat_messages = [{"role": "user", "content": str(messages)}]

        response = self.chat_model.generate(
            messages=chat_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

        if hasattr(response, 'text'):
            return response.text
        elif hasattr(response, 'content'):
            return response.content
        else:
            return str(response)


class TFIDFRetriever:
    """Simple TF-IDF based retriever for vanilla RAG."""

    def __init__(self):
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for Vanilla RAG. "
                "Install with: pip install scikit-learn"
            )
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2)
        )
        self.chunks = []
        self.tfidf_matrix = None

    def index(self, chunks: List[str]):
        """Index chunks using TF-IDF."""
        self.chunks = chunks
        if not chunks:
            return
        self.tfidf_matrix = self.vectorizer.fit_transform(chunks)

    def retrieve(self, query: str, top_k: int = 5) -> List[tuple[int, float, str]]:
        """
        Retrieve top-k most relevant chunks.

        Returns:
            List of (chunk_idx, score, chunk_text) tuples
        """
        if not self.chunks or self.tfidf_matrix is None:
            return []

        # Vectorize query
        query_vec = self.vectorizer.transform([query])

        # Compute similarity
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        # Get top-k indices
        top_indices = similarities.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # Only include non-zero scores
                results.append((int(idx), float(similarities[idx]), self.chunks[idx]))

        return results


class FullVanillaRAGAdapter(BaselineSystem):
    """
    Full Vanilla RAG adapter for baseline comparison.

    This is a simple RAG system that uses:
    1. TF-IDF for retrieval (no fancy graph indexing)
    2. LLM for generation (OpenAI or local)
    3. No special reasoning or self-reflection

    This provides a clean baseline to compare against more sophisticated
    systems like Self-RAG, KET-RAG, and HippoRAG.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 500,
        top_k: int = 5,
        use_local_llm: bool = False,
        local_llm_model: str = "Qwen/Qwen2.5-7B-Instruct",
        local_llm_device: str = "cuda:0",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        **kwargs
    ):
        """
        Initialize Vanilla RAG adapter.

        Args:
            api_key: OpenAI API key (required if use_local_llm=False)
            model: LLM model to use (OpenAI model name)
            temperature: Sampling temperature
            max_tokens: Max tokens for generation
            top_k: Number of chunks to retrieve
            use_local_llm: Use local LLM instead of OpenAI
            local_llm_model: HuggingFace model name for local LLM
            local_llm_device: Device for local LLM (cuda:0, cuda:1, cpu)
            load_in_8bit: Use 8-bit quantization for local LLM
            load_in_4bit: Use 4-bit quantization for local LLM
            **kwargs: Additional config
        """
        super().__init__(name="vanillarag", **kwargs)

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_k = top_k
        self.use_local_llm = use_local_llm

        # Initialize LLM client (OpenAI or Local)
        if use_local_llm:
            print(f"  [Vanilla RAG] Using local LLM: {local_llm_model}")
            self.llm = LocalLLMWrapper(
                model_name=local_llm_model,
                device=local_llm_device,
                temperature=temperature,
                max_tokens=max_tokens,
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
            )
            self.api_key = None
        else:
            if not GRAPHRAG_AVAILABLE:
                raise ImportError(
                    "GraphRAG is required for OpenAI support. "
                    "Use use_local_llm=True or install GraphRAG."
                )
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError(
                    "API key required for OpenAI. Set OPENAI_API_KEY "
                    "or use use_local_llm=True for local models."
                )
            print(f"  [Vanilla RAG] Using OpenAI: {model}")
            # GraphRAG 2.7.0 API for chat model
            chat_config = LanguageModelConfig(
                api_key=self.api_key,
                type=ModelType.Chat,
                model_provider="openai",
                model=self.model,
                max_retries=3,
            )
            self.llm_raw = ModelManager().get_or_create_chat_model(
                name="vanillarag_adapter",
                model_type=ModelType.Chat,
                config=chat_config,
            )
            # Wrap in a compatibility layer
            self.llm = OpenAILLMWrapper(self.llm_raw)

        # Initialize retriever
        self.retriever = TFIDFRetriever()

    def answer(
        self,
        question: str,
        context: Optional[List[List[str]]] = None,
        supporting_facts: Optional[List[tuple]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> BaselineResponse:
        """
        Answer a question using Vanilla RAG.

        Simple pipeline:
        1. Index all chunks with TF-IDF
        2. Retrieve top-k most similar chunks
        3. Generate answer conditioned on retrieved chunks

        Args:
            question: The question to answer
            context: HotpotQA context (list of [title, sentences] pairs)
            supporting_facts: Not used by Vanilla RAG
            metadata: Optional metadata

        Returns:
            BaselineResponse
        """
        token_tracker = TokenTracker()
        latency_tracker = LatencyTracker()

        if not context:
            return BaselineResponse(
                answer="I don't have enough information to answer this question.",
                tokens_used=0,
                latency_ms=0,
                selected_passages=[],
                abstained=True,
                mode="vanillarag",
                stats={},
                raw_answer=None,
                extracted_answer=None,
            )

        try:
            latency_tracker.start()

            # Prepare chunks
            chunks = []
            chunk_metadata = []
            for title, sentences in context:
                text = " ".join(sentences) if isinstance(sentences, list) else sentences
                chunks.append(text)
                chunk_metadata.append({"title": title})

            # Index chunks
            self.retriever.index(chunks)

            # Retrieve relevant chunks
            retrieval_results = self.retriever.retrieve(question, top_k=self.top_k)

            # Format retrieved chunks for simple vanilla prompt
            context_parts = []
            if retrieval_results:
                for i, (idx, score, chunk_text) in enumerate(retrieval_results, 1):
                    context_parts.append(f"[{i}] {chunk_text}")
            else:
                context_parts.append("(no relevant context found)")

            context_str = "\n\n".join(context_parts)

            # Use simple vanilla RAG prompt (no Trident-style formatting)
            answer_prompt = (
                f"Based on the following context, answer the question concisely.\n\n"
                f"Context:\n{context_str}\n\n"
                f"Question: {question}\n\n"
                f"Answer:"
            )

            raw_response = self.llm.generate(
                messages=[{"role": "user", "content": answer_prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            latency_tracker.stop("answer_generation")

            # Count tokens (rough estimate)
            prompt_tokens = len(answer_prompt.split()) * 1.3
            completion_tokens = len(raw_response.split()) * 1.3
            token_tracker.add_call(
                int(prompt_tokens),
                int(completion_tokens),
                "answer_generation"
            )

            # Use original response without Trident-style post-processing
            answer = raw_response.strip()

            # Prepare selected passages
            selected_passages = [
                {
                    "text": chunk_text[:200],
                    "chunk_idx": idx,
                    "score": float(score)
                }
                for idx, score, chunk_text in retrieval_results
            ]

            return BaselineResponse(
                answer=answer,  # Use original response without post-processing
                tokens_used=token_tracker.total_tokens,
                latency_ms=latency_tracker.get_total_latency(),
                selected_passages=selected_passages,
                abstained=False,
                mode="vanillarag",
                stats={
                    "num_chunks": len(chunks),
                    "num_retrieved": len(retrieval_results),
                    "retrieval_scores": [float(score) for _, score, _ in retrieval_results],
                },
                # Debugging fields
                raw_answer=raw_response,
                extracted_answer=answer,
            )

        except Exception as e:
            print(f"Error in Vanilla RAG processing: {e}")
            import traceback
            traceback.print_exc()

            return BaselineResponse(
                answer="Error processing question.",
                tokens_used=0,
                latency_ms=0,
                selected_passages=[],
                abstained=True,
                mode="vanillarag",
                stats={"error": str(e)},
                raw_answer=None,
                extracted_answer=None,
            )

    def get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        return {
            "name": "Vanilla RAG",
            "version": "full",
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_k": self.top_k,
            "retrieval_method": "TF-IDF",
        }