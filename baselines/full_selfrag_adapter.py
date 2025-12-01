"""Full Self-RAG adapter for HotpotQA evaluation using the official Self-RAG model."""

from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import re

from baselines.full_baseline_interface import (
    BaselineSystem,
    BaselineResponse,
    TokenTracker,
    LatencyTracker,
)

try:
    from vllm import LLM, SamplingParams
    import tiktoken
    VLLM_AVAILABLE = True
except ImportError as e:
    VLLM_AVAILABLE = False
    VLLM_IMPORT_ERROR = str(e)


class FullSelfRAGAdapter(BaselineSystem):
    """
    Full Self-RAG adapter using the official Self-RAG model from HuggingFace.

    This adapter:
    1. Loads the Self-RAG fine-tuned model (7B or 13B)
    2. Formats HotpotQA context with Self-RAG's special token syntax
    3. Generates answers with reflection tokens ([Retrieval], [Relevant], etc.)
    4. Extracts final answer and tracks all tokens

    Self-RAG uses special reflection tokens:
    - [Retrieval]: Decides whether to retrieve
    - [No Retrieval]: Decides not to retrieve
    - [Relevant]: Retrieved content is relevant
    - [Irrelevant]: Retrieved content is not relevant
    - [Fully supported]: Answer is fully supported by evidence
    - [Partially supported]: Answer is partially supported
    - [No support]: Answer has no support
    - [Utility:1-5]: Utility score
    """

    # Self-RAG special tokens
    RETRIEVAL_TOKENS = ["[Retrieval]", "[No Retrieval]"]
    RELEVANCE_TOKENS = ["[Relevant]", "[Irrelevant]"]
    SUPPORT_TOKENS = ["[Fully supported]", "[Partially supported]", "[No support]"]
    UTILITY_TOKENS = [f"[Utility:{i}]" for i in range(1, 6)]

    def __init__(
        self,
        model_name: str = "selfrag/selfrag_llama2_7b",
        download_dir: Optional[str] = None,
        max_tokens: int = 100,
        temperature: float = 0.0,
        top_p: float = 1.0,
        use_critic: bool = True,
        provide_context: bool = True,
        **kwargs
    ):
        """
        Initialize Self-RAG adapter.

        Args:
            model_name: HuggingFace model name
                       - "selfrag/selfrag_llama2_7b" (recommended)
                       - "selfrag/selfrag_llama2_13b"
            download_dir: Directory to cache downloaded model
            max_tokens: Max tokens to generate
            temperature: Sampling temperature (0.0 for greedy)
            top_p: Top-p sampling
            use_critic: Whether to use critic tokens in generation
            provide_context: Whether to provide context to model (vs let it decide to retrieve)
            **kwargs: Additional config
        """
        super().__init__(name="selfrag", **kwargs)

        if not VLLM_AVAILABLE:
            raise ImportError(
                f"vllm library not available. Install with: pip install vllm\n"
                f"Error: {VLLM_IMPORT_ERROR}"
            )

        self.model_name = model_name
        self.download_dir = download_dir or os.getenv("HF_CACHE_DIR")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.use_critic = use_critic
        self.provide_context = provide_context

        # Load model
        print(f"Loading Self-RAG model: {model_name}...")
        self.model = LLM(
            model_name,
            download_dir=self.download_dir,
            dtype="half",  # Use FP16 for efficiency
            tensor_parallel_size=1,
        )

        # Sampling params
        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            skip_special_tokens=False,  # Important: keep reflection tokens
        )

        # Tokenizer for token counting
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer as approximation
        except:
            self.tokenizer = None

    def _format_prompt(self, question: str, context: Optional[List[List[str]]] = None) -> str:
        """
        Format prompt in Self-RAG style.

        Args:
            question: The question
            context: Optional HotpotQA context

        Returns:
            Formatted prompt string
        """
        prompt = f"### Instruction:\n{question}\n\n### Response:\n"

        if self.provide_context and context:
            # Format context as paragraphs
            paragraphs = []
            for title, sentences in context:
                text = " ".join(sentences) if isinstance(sentences, list) else sentences
                paragraphs.append(text)

            # Combine paragraphs
            combined_context = " ".join(paragraphs)

            # Add retrieval marker and context
            prompt += f"[Retrieval]<paragraph>{combined_context}</paragraph>"

        return prompt

    def _extract_answer(self, generation: str) -> str:
        """
        Extract final answer from Self-RAG generation with reflection tokens.

        Args:
            generation: Raw generation from model (includes reflection tokens)

        Returns:
            Extracted answer text
        """
        # Remove all reflection tokens
        answer = generation

        # Remove retrieval tokens
        for token in self.RETRIEVAL_TOKENS:
            answer = answer.replace(token, "")

        # Remove relevance tokens
        for token in self.RELEVANCE_TOKENS:
            answer = answer.replace(token, "")

        # Remove support tokens
        for token in self.SUPPORT_TOKENS:
            answer = answer.replace(token, "")

        # Remove utility tokens
        for token in self.UTILITY_TOKENS:
            answer = answer.replace(token, "")

        # Remove paragraph tags
        answer = re.sub(r'</?paragraph>', '', answer)

        # Remove special tokens
        answer = answer.replace('</s>', '')
        answer = answer.replace('<s>', '')

        # Strip whitespace
        answer = answer.strip()

        # Take first sentence or first line as answer
        sentences = answer.split('.')
        if sentences:
            answer = sentences[0].strip()

        return answer

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Fallback: approximate as ~1.3 tokens per word
            return int(len(text.split()) * 1.3)

    def answer(
        self,
        question: str,
        context: Optional[List[List[str]]] = None,
        supporting_facts: Optional[List[tuple]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> BaselineResponse:
        """
        Answer a question using Self-RAG.

        Args:
            question: The question to answer
            context: HotpotQA context (list of [title, sentences] pairs)
            supporting_facts: Not used by Self-RAG
            metadata: Optional metadata

        Returns:
            BaselineResponse with answer and metrics
        """
        token_tracker = TokenTracker()
        latency_tracker = LatencyTracker()

        try:
            # Format prompt
            prompt = self._format_prompt(question, context)

            # Generate with Self-RAG
            latency_tracker.start()
            outputs = self.model.generate([prompt], self.sampling_params)
            latency_tracker.stop("generation")

            # Extract generation
            if not outputs or not outputs[0].outputs:
                raise ValueError("No output generated")

            raw_generation = outputs[0].outputs[0].text

            # Extract answer
            answer = self._extract_answer(raw_generation)

            # Count tokens
            prompt_tokens = self._count_tokens(prompt)
            completion_tokens = self._count_tokens(raw_generation)
            token_tracker.add_call(prompt_tokens, completion_tokens, purpose="generation")

            # Parse reflection tokens for stats
            used_retrieval = "[Retrieval]" in raw_generation
            relevance_label = None
            for token in self.RELEVANCE_TOKENS:
                if token in raw_generation:
                    relevance_label = token
                    break

            support_label = None
            for token in self.SUPPORT_TOKENS:
                if token in raw_generation:
                    support_label = token
                    break

            utility_score = None
            for i in range(5, 0, -1):
                token = f"[Utility:{i}]"
                if token in raw_generation:
                    utility_score = i
                    break

            # Prepare selected passages
            selected_passages = []
            if context:
                for title, sentences in context[:3]:
                    text = " ".join(sentences) if isinstance(sentences, list) else sentences
                    selected_passages.append({"text": text, "title": title})

            return BaselineResponse(
                answer=answer,
                tokens_used=token_tracker.total_tokens,
                latency_ms=latency_tracker.get_total_latency(),
                selected_passages=selected_passages,
                abstained=False,
                mode="selfrag",
                stats={
                    **token_tracker.get_stats(),
                    "raw_generation": raw_generation,
                    "used_retrieval": used_retrieval,
                    "relevance_label": relevance_label,
                    "support_label": support_label,
                    "utility_score": utility_score,
                    "num_context_docs": len(context) if context else 0,
                },
            )

        except Exception as e:
            print(f"Error in Self-RAG processing: {e}")
            import traceback
            traceback.print_exc()

            return BaselineResponse(
                answer="Error processing question.",
                tokens_used=token_tracker.total_tokens,
                latency_ms=latency_tracker.get_total_latency(),
                selected_passages=[],
                abstained=True,
                mode="selfrag",
                stats={
                    **token_tracker.get_stats(),
                    "error": str(e),
                },
            )

    def get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        return {
            "name": "Self-RAG",
            "version": "full",
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "use_critic": self.use_critic,
            "provide_context": self.provide_context,
        }
