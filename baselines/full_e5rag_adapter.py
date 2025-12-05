"""E5-RAG adapter implementing standard retrieval-augmented generation.

This adapter implements the experimental setup from research papers using:
- E5-base-v2 as the retriever (5 passages per query)
- LLaMA-3-8B-instruct or Qwen-1.5-14B as the generator
- vLLM framework with greedy decoding
- Standardized prompts for fair comparison

Retriever Settings:
- Model: E5-base-v2 (dense retrieval)
- Top-k: 5 passages
- Max padding length: 512
- Query padding length: 128
- Batch size: 1024
- FP16 enabled
- Faiss Flat index for accuracy

Generator Settings:
- Model: LLaMA-3-8B-instruct (default) or Qwen-1.5-14B
- Max input length: 2048
- Max output length: 32
- vLLM framework with greedy decoding

Prompt Format:
System: "Answer the question based on the given passage. Only give me the answer
         and do not output any other words. The following are given passages:"
Passages: "Doc 1 (Title: {title}) {content}\nDoc 2 (Title: {title}) {content}"
User: "Question: {question}"
"""

from __future__ import annotations

import os
from typing import Dict, Any, List, Optional
import numpy as np

from baselines.full_baseline_interface import (
    BaselineSystem,
    BaselineResponse,
    TokenTracker,
    LatencyTracker,
)

try:
    from sentence_transformers import SentenceTransformer
    import torch
    import faiss
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False


class E5Retriever:
    """Dense retriever using E5-base-v2 embeddings and Faiss index."""

    def __init__(
        self,
        model_name: str = "intfloat/e5-base-v2",
        max_length: int = 512,
        batch_size: int = 1024,
        use_fp16: bool = True,
        device: str = "cuda:0",
    ):
        """
        Initialize E5 retriever.

        Args:
            model_name: HuggingFace model name (default: intfloat/e5-base-v2)
            max_length: Maximum padding length for documents (default: 512)
            batch_size: Batch size for encoding (default: 1024)
            use_fp16: Use FP16 for faster encoding (default: True)
            device: Device to run on (default: cuda:0)
        """
        if not EMBEDDINGS_AVAILABLE:
            raise ImportError(
                "sentence-transformers and faiss required. "
                "Install with: pip install sentence-transformers faiss-gpu"
            )

        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.use_fp16 = use_fp16
        self.device = device

        # Load E5 model
        self.encoder = SentenceTransformer(model_name, device=device)
        if use_fp16:
            self.encoder = self.encoder.half()

        # Faiss index (will be initialized during indexing)
        self.index = None
        self.chunks = []

    def index(self, chunks: List[str]):
        """
        Index chunks using E5 embeddings and Faiss Flat index.

        Args:
            chunks: List of text chunks to index
        """
        if not chunks:
            return

        self.chunks = chunks

        # Encode chunks in batches
        # E5 requires "passage: " prefix for documents
        passages = [f"passage: {chunk}" for chunk in chunks]

        embeddings = self.encoder.encode(
            passages,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2 normalization for cosine similarity
        )

        # Create Faiss Flat index (exact search for accuracy)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine with normalized vectors)
        self.index.add(embeddings.astype('float32'))

    def retrieve(self, query: str, top_k: int = 5, query_max_length: int = 128) -> List[tuple[int, float, str]]:
        """
        Retrieve top-k most relevant chunks.

        Args:
            query: Query text
            top_k: Number of passages to retrieve (default: 5)
            query_max_length: Maximum query padding length (default: 128)

        Returns:
            List of (chunk_idx, score, chunk_text) tuples
        """
        if not self.chunks or self.index is None:
            return []

        # E5 requires "query: " prefix for queries
        query_text = f"query: {query}"

        # Encode query
        query_embedding = self.encoder.encode(
            [query_text],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        # Search Faiss index
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)

        # Format results
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0 and idx < len(self.chunks):  # Valid index
                results.append((int(idx), float(score), self.chunks[idx]))

        return results


class FullE5RAGAdapter(BaselineSystem):
    """
    E5-RAG adapter implementing standard experimental setup.

    This baseline uses:
    1. E5-base-v2 for dense retrieval (5 passages)
    2. vLLM for efficient generation (LLaMA-3-8B or Qwen-1.5-14B)
    3. Standardized prompt format from research papers
    4. Greedy decoding for reproducibility
    """

    def __init__(
        self,
        retriever_model: str = "intfloat/e5-base-v2",
        generator_model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        top_k: int = 5,
        max_input_length: int = 2048,
        max_output_length: int = 32,
        temperature: float = 0.0,
        gpu_memory_utilization: float = 0.5,
        device: str = "cuda:0",
        retriever_batch_size: int = 1024,
        use_fp16: bool = True,
        **kwargs
    ):
        """
        Initialize E5-RAG adapter.

        Args:
            retriever_model: E5 model for retrieval (default: intfloat/e5-base-v2)
            generator_model: LLM for generation (default: Meta-Llama-3-8B-Instruct)
                           Alternatives: "Qwen/Qwen1.5-14B", "Qwen/Qwen2.5-7B-Instruct"
            top_k: Number of passages to retrieve (default: 5)
            max_input_length: Maximum input length for generator (default: 2048)
            max_output_length: Maximum output length (default: 32)
            temperature: Sampling temperature (0.0 = greedy decoding)
            gpu_memory_utilization: GPU memory fraction for vLLM (default: 0.5)
            device: GPU device (e.g., "cuda:0", "cuda:2", or "0", "2")
            retriever_batch_size: Batch size for retrieval (default: 1024)
            use_fp16: Use FP16 for retrieval (default: True)
        """
        super().__init__(name="e5rag", **kwargs)

        if not EMBEDDINGS_AVAILABLE:
            raise ImportError(
                "sentence-transformers and faiss required. "
                "Install with: pip install sentence-transformers faiss-gpu"
            )

        if not VLLM_AVAILABLE:
            raise ImportError(
                "vllm required. Install with: pip install vllm"
            )

        self.retriever_model = retriever_model
        self.generator_model = generator_model
        self.top_k = top_k
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.temperature = temperature
        self.device = device if not device.isdigit() else f"cuda:{device}"

        # Initialize E5 retriever
        print(f"  [E5-RAG] Initializing retriever: {retriever_model}")
        self.retriever = E5Retriever(
            model_name=retriever_model,
            max_length=512,
            batch_size=retriever_batch_size,
            use_fp16=use_fp16,
            device=self.device,
        )

        # Initialize vLLM generator
        print(f"  [E5-RAG] Initializing generator: {generator_model}")
        self.llm = LLM(
            model=generator_model,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_input_length,
            trust_remote_code=True,
        )

        # Sampling params for greedy decoding
        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_output_length,
            top_p=1.0,
        )

    def _build_e5rag_prompt(self, question: str, passages: List[Dict[str, Any]]) -> str:
        """
        Build prompt using E5-RAG format.

        System: "Answer the question based on the given passage. Only give me
                 the answer and do not output any other words."
        Passages: "Doc 1 (Title: {title}) {content}"
        User: "Question: {question}"

        Args:
            question: The question
            passages: List of passage dicts with 'text' and optionally 'title'

        Returns:
            Formatted prompt string
        """
        # System instruction
        system_msg = (
            "Answer the question based on the given passage. "
            "Only give me the answer and do not output any other words."
        )

        # Format passages
        if passages:
            passage_lines = []
            for i, passage in enumerate(passages, 1):
                title = passage.get('title', f'Document {i}')
                content = passage.get('text', passage.get('content', ''))
                passage_lines.append(f"Doc {i} (Title: {title}) {content}")

            passages_text = "\n".join(passage_lines)
            system_msg += f"\n\nThe following are given passages:\n{passages_text}"

        # User question
        user_msg = f"Question: {question}"

        # Combine into single prompt
        prompt = f"{system_msg}\n\n{user_msg}"

        return prompt

    def answer(
        self,
        question: str,
        context: Optional[List[List[str]]] = None,
        supporting_facts: Optional[List[tuple]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> BaselineResponse:
        """
        Answer using E5-RAG pipeline.

        Pipeline:
        1. Index all chunks with E5-base-v2 embeddings
        2. Retrieve top-k most similar chunks using Faiss
        3. Generate answer with vLLM (greedy decoding)

        Args:
            question: The question to answer
            context: HotpotQA context (list of [title, sentences] pairs)
            supporting_facts: Not used
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
                mode="e5rag",
                stats={},
            )

        try:
            latency_tracker.start()

            # Prepare chunks with titles
            chunks = []
            chunk_titles = []
            for title, sentences in context:
                text = " ".join(sentences) if isinstance(sentences, list) else sentences
                chunks.append(text)
                chunk_titles.append(title)

            # Index chunks with E5
            self.retriever.index(chunks)

            # Retrieve top-k passages
            retrieval_results = self.retriever.retrieve(
                question,
                top_k=self.top_k,
                query_max_length=128
            )

            latency_tracker.stop("retrieval")

            # Format passages for prompt
            passages = []
            if retrieval_results:
                for idx, score, chunk_text in retrieval_results:
                    passages.append({
                        "text": chunk_text,
                        "title": chunk_titles[idx] if idx < len(chunk_titles) else f"Document {idx+1}"
                    })
            else:
                # No results found
                passages.append({"text": "(no relevant context found)", "title": "N/A"})

            # Build E5-RAG style prompt
            latency_tracker.start()
            prompt = self._build_e5rag_prompt(question, passages)

            # Generate answer with vLLM (greedy decoding)
            outputs = self.llm.generate([prompt], self.sampling_params)
            response = outputs[0].outputs[0].text.strip()

            latency_tracker.stop("generation")

            # Token counting (approximate)
            prompt_tokens = len(prompt.split()) * 1.3
            completion_tokens = len(response.split()) * 1.3
            token_tracker.add_call(
                int(prompt_tokens),
                int(completion_tokens),
                "generation"
            )

            # Extract answer (should already be short due to prompt)
            answer = response.strip()

            # Remove common prefixes if present
            for prefix in ["Answer:", "A:", "The answer is:"]:
                if answer.startswith(prefix):
                    answer = answer[len(prefix):].strip()
                    break

            # Prepare selected passages
            selected_passages = [
                {
                    "text": chunk_text[:200],
                    "title": chunk_titles[idx] if idx < len(chunk_titles) else f"Doc {idx+1}",
                    "chunk_idx": idx,
                    "score": float(score)
                }
                for idx, score, chunk_text in retrieval_results
            ]

            return BaselineResponse(
                answer=answer,
                tokens_used=token_tracker.total_tokens,
                latency_ms=latency_tracker.get_total_latency(),
                selected_passages=selected_passages,
                abstained=False,
                mode="e5rag",
                stats={
                    "num_chunks": len(chunks),
                    "num_retrieved": len(retrieval_results),
                    "retrieval_scores": [float(score) for _, score, _ in retrieval_results],
                    "prompt_length": len(prompt),
                },
                raw_answer=response,
                extracted_answer=answer,
            )

        except Exception as e:
            print(f"Error in E5-RAG processing: {e}")
            import traceback
            traceback.print_exc()

            return BaselineResponse(
                answer="Error processing question.",
                tokens_used=0,
                latency_ms=0,
                selected_passages=[],
                abstained=True,
                mode="e5rag",
                stats={"error": str(e)},
            )

    def get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        return {
            "name": "E5-RAG",
            "version": "full",
            "retriever_model": self.retriever_model,
            "generator_model": self.generator_model,
            "top_k": self.top_k,
            "max_input_length": self.max_input_length,
            "max_output_length": self.max_output_length,
            "temperature": self.temperature,
            "retrieval_method": "E5-base-v2 + Faiss Flat",
            "generation_method": "vLLM greedy decoding",
        }
