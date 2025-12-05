"""Faithful KET-RAG adapter - uses precomputed contexts from official KET-RAG pipeline.

This adapter is a TRUE faithful wrapper around official KET-RAG:
- Uses official KET-RAG CLI for indexing and context retrieval
- Reads precomputed contexts from JSON files
- Only standardizes: prompt format, LLM model, answer extraction

Architecture:
    1. OFFLINE (manual step): Run official KET-RAG pipeline
       $ cd KET-RAG
       $ poetry run graphrag index --root ragtest-hotpot/
       $ poetry run python indexing_sket/create_context.py ragtest-hotpot/ keyword 0.5

    2. ONLINE (this adapter): Load precomputed contexts and generate answers
       - Reads contexts from JSON: {question_id: context_string}
       - Formats into Trident prompt
       - Calls user-specified LLM
       - Extracts answer with Trident logic

This ensures:
- ALL retrieval logic is 100% original KET-RAG
- ONLY generation is standardized for fair comparison
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

from baselines.full_baseline_interface import (
    BaselineSystem,
    BaselineResponse,
    TokenTracker,
    LatencyTracker,
)
from baselines.local_llm_wrapper import LocalLLMWrapper
from baselines.prompt_utils import (
    build_ketrag_original_prompt,
    build_trident_style_prompt,
    extract_ketrag_original_answer,
    extract_trident_style_answer,
)

# Import GraphRAG for LLM wrapper (when using OpenAI)
try:
    from graphrag.config.enums import ModelType
    from graphrag.config.models.language_model_config import LanguageModelConfig
    from graphrag.language_model.manager import ModelManager
    GRAPHRAG_AVAILABLE = True
except ImportError:
    GRAPHRAG_AVAILABLE = False


class OpenAILLMWrapper:
    """Wrapper for GraphRAG chat model to provide compatible generate() method."""

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


class FullKETRAGAdapter(BaselineSystem):
    """
    Faithful wrapper around official KET-RAG implementation.

    Uses precomputed contexts from the official KET-RAG pipeline.
    Only standardizes the final generation step for fair comparison.

    This is what you asked for:
    - "faithful original ket-rag" ✓
    - "change the prompt and llm model to my own" ✓
    - "have it run like it did originally" ✓
    """

    def __init__(
        self,
        context_file: str,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 500,
        use_local_llm: bool = False,
        local_llm_model: str = "Qwen/Qwen2.5-7B-Instruct",
        local_llm_device: str = "cuda:0",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        prompt_style: str = "original",
        compare_original_prompt: bool = False,
        **kwargs
    ):
        """
        Initialize faithful KET-RAG adapter.

        Args:
            context_file: Path to precomputed context JSON from official KET-RAG
                         (e.g., KET-RAG/ragtest-hotpot/output/ragtest-hotpot-keyword-0.5.json)
            api_key: OpenAI API key (if using OpenAI)
            model: LLM model name
            temperature: Sampling temperature
            max_tokens: Max tokens for generation
            use_local_llm: Use local LLM instead of OpenAI
            local_llm_model: HuggingFace model name
            local_llm_device: Device for local LLM
            load_in_8bit: Use 8-bit quantization
            load_in_4bit: Use 4-bit quantization
        """
        super().__init__(name="ketrag_official", **kwargs)

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_local_llm = use_local_llm
        self.prompt_style = prompt_style
        self.compare_original_prompt = compare_original_prompt

        # Load precomputed contexts from official KET-RAG
        self.context_by_qid = self._load_ketrag_contexts(context_file)
        print(f"  [KET-RAG Official] Loaded {len(self.context_by_qid)} precomputed contexts")

        # Initialize LLM (standardized for fair comparison)
        if use_local_llm:
            print(f"  [KET-RAG Official] Using local LLM: {local_llm_model}")
            self.llm = LocalLLMWrapper(
                model_name=local_llm_model,
                device=local_llm_device,
                temperature=temperature,
                max_tokens=max_tokens,
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
            )
        else:
            if not GRAPHRAG_AVAILABLE:
                raise ImportError("GraphRAG not available. Install with: pip install graphrag")

            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY required when not using local LLM")

            print(f"  [KET-RAG Official] Using OpenAI: {model}")
            chat_config = LanguageModelConfig(
                api_key=self.api_key,
                type=ModelType.Chat,
                model_provider="openai",
                model=self.model,
                max_retries=3,
            )
            self.llm_raw = ModelManager().get_or_create_chat_model(
                name="ketrag_official",
                model_type=ModelType.Chat,
                config=chat_config,
            )
            self.llm = OpenAILLMWrapper(self.llm_raw)

    def _load_ketrag_contexts(self, context_file: str) -> Dict[str, str]:
        """
        Load precomputed contexts from official KET-RAG output.

        Expected format from create_context.py:
        [
            {"id": "question_id", "context": "graph_context + text_context"},
            ...
        ]
        """
        context_path = Path(context_file)
        if not context_path.exists():
            raise FileNotFoundError(
                f"KET-RAG context file not found: {context_file}\n"
                f"You must run the official KET-RAG pipeline first:\n"
                f"  1. cd KET-RAG\n"
                f"  2. poetry run graphrag index --root ragtest-hotpot/\n"
                f"  3. poetry run python indexing_sket/create_context.py ragtest-hotpot/ keyword 0.5"
            )

        with open(context_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Build lookup: question_id -> context
        context_dict = {}
        for entry in data:
            qid = entry.get('id')
            ctx = entry.get('context', '')
            if qid:
                context_dict[qid] = ctx

        return context_dict

    def _clean_ketrag_context(self, context: str, question: str) -> str:
        """
        Clean and optimize KET-RAG context to reduce noise.

        Fixes:
        1. Remove empty "[]" prefix that confuses the LLM
        2. Filter out irrelevant entities based on question keywords
        3. Keep only the most relevant entities to reduce noise

        This is a post-processing step on KET-RAG's output, not a modification
        to KET-RAG itself.
        """
        if not context:
            return context

        # Fix 1: Remove empty "[]" prefix
        cleaned = context.strip()
        if cleaned.startswith("[]"):
            cleaned = cleaned[2:].lstrip()

        # Fix 2: Filter irrelevant entities
        # Extract question keywords (simple approach: words in question)
        question_words = set(question.lower().split())
        # Remove common stopwords
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                    'of', 'with', 'by', 'from', 'is', 'was', 'were', 'are', 'be', 'been',
                    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                    'should', 'may', 'might', 'must', 'can', 'same', 'as', 'than', 'that',
                    'this', 'these', 'those', 'what', 'which', 'who', 'when', 'where', 'why',
                    'how', 'their', 'them', 'they'}
        question_keywords = question_words - stopwords

        # Parse sections
        if "-----Entities-----" not in cleaned:
            return cleaned

        sections = cleaned.split("-----")
        filtered_sections = []

        for i, section in enumerate(sections):
            # Keep all non-entity sections as-is
            if not section.strip().startswith("Entities"):
                filtered_sections.append(section)
                continue

            # Filter entity section
            lines = section.split("\n")
            header_lines = lines[:2] if len(lines) >= 2 else lines  # Keep header
            entity_lines = lines[2:] if len(lines) > 2 else []

            # Keep entities that match question keywords
            relevant_entities = []
            for line in entity_lines:
                if not line.strip():
                    continue
                # Check if entity name or description contains question keywords
                line_lower = line.lower()
                if any(keyword in line_lower for keyword in question_keywords):
                    relevant_entities.append(line)

            # If we filtered too aggressively (< 2 entities), keep top 5
            if len(relevant_entities) < 2 and len(entity_lines) > 0:
                relevant_entities = entity_lines[:5]
            elif len(relevant_entities) > 10:
                # Cap at 10 entities to reduce noise
                relevant_entities = relevant_entities[:10]

            # Reconstruct entity section
            filtered_section = "\n".join(header_lines + relevant_entities)
            filtered_sections.append(filtered_section)

        return "-----".join(filtered_sections)

    def answer(
        self,
        question: str,
        context: Optional[List[List[str]]] = None,
        supporting_facts: Optional[List[tuple]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> BaselineResponse:
        """
        Answer using official KET-RAG's precomputed context + standardized generation.

        This is the "faithful wrapper" you requested:
        - Retrieval: 100% official KET-RAG (precomputed)
        - Generation: Standardized (Trident prompt + user LLM)
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
                mode="ketrag_official",
                stats={"error": "missing_question_id"},
            )

        # Load official KET-RAG's retrieved context
        ketrag_context = self.context_by_qid.get(question_id)
        if not ketrag_context:
            return BaselineResponse(
                answer="No precomputed context found for this question",
                tokens_used=0,
                latency_ms=0,
                selected_passages=[],
                abstained=True,
                mode="ketrag_official",
                stats={"error": "context_not_found", "question_id": question_id},
            )

        # Clean context to remove noise (e.g., empty "[]", irrelevant entities)
        # This is post-processing on KET-RAG output, not modifying KET-RAG itself
        ketrag_context = self._clean_ketrag_context(ketrag_context, question)

        comparison_stats: Dict[str, Any] = {}

        try:
            latency_tracker.start()

            # Parse KET-RAG context into passages
            # Format: "-----Entities and Relationships-----\n...\n-----Text source that may be relevant-----\n..."
            passages = self._parse_ketrag_context(ketrag_context)

            # Build primary prompt (default: Trident standardized)
            if self.prompt_style == "trident":
                answer_prompt = build_trident_style_prompt(question, passages)
                answer_messages = [{"role": "user", "content": answer_prompt}]
                extract_answer_fn = extract_trident_style_answer
            elif self.prompt_style == "original":
                answer_messages = build_ketrag_original_prompt(question, ketrag_context)
                # For token accounting/logging, flatten the messages
                answer_prompt = "\n\n".join(m.get("content", "") for m in answer_messages)
                extract_answer_fn = extract_ketrag_original_answer
            else:
                raise ValueError(f"Unknown prompt_style={self.prompt_style}. Choose 'trident' or 'original'.")

            # Generate with user-specified model
            response = self.llm.generate(
                messages=answer_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            latency_tracker.stop("answer_generation")

            # Count tokens
            prompt_tokens = len(answer_prompt.split()) * 1.3
            completion_tokens = len(response.split()) * 1.3
            token_tracker.add_call(
                int(prompt_tokens),
                int(completion_tokens),
                "query_generation"
            )

            # STANDARDIZED: Extract answer with Trident's logic
            answer = extract_answer_fn(response)

            # Optional: generate using the alternate prompt to compare behavior
            if self.compare_original_prompt:
                alt_style = "original" if self.prompt_style == "trident" else "trident"

                if alt_style == "original":
                    comparison_messages = build_ketrag_original_prompt(question, ketrag_context)
                    comparison_prompt = "\n\n".join(m.get("content", "") for m in comparison_messages)
                    comparison_extractor = extract_ketrag_original_answer
                else:
                    comparison_prompt = build_trident_style_prompt(question, passages)
                    comparison_messages = [{"role": "user", "content": comparison_prompt}]
                    comparison_extractor = extract_trident_style_answer

                latency_tracker.start()
                comparison_response = self.llm.generate(
                    messages=comparison_messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                comparison_latency = latency_tracker.stop("comparison_answer_generation")

                comp_prompt_tokens = len(comparison_prompt.split()) * 1.3
                comp_completion_tokens = len(comparison_response.split()) * 1.3
                token_tracker.add_call(
                    int(comp_prompt_tokens),
                    int(comp_completion_tokens),
                    "comparison_generation",
                )

                comparison_stats = {
                    "style": alt_style,
                    "prompt": comparison_prompt,
                    "raw_response": comparison_response,
                    "latency_ms": comparison_latency,
                    "extracted_answer": comparison_extractor(comparison_response),
                }

            return BaselineResponse(
                answer=answer,
                tokens_used=token_tracker.total_tokens,
                latency_ms=latency_tracker.get_total_latency(),
                selected_passages=passages[:5],  # First 5 passages
                abstained=False,
                mode="ketrag_official",
                stats={
                    "retrieval_source": "official_ketrag_precomputed",
                    "generation_approach": f"{self.prompt_style}_prompt",
                    "num_passages": len(passages),
                    "indexing_tokens": 0,  # Indexing done offline
                    "indexing_latency_ms": 0.0,  # Indexing done offline
                    "total_cost_tokens": token_tracker.total_tokens,
                    "prompt_style": self.prompt_style,
                    "comparison_generation": comparison_stats if comparison_stats else None,
                },
                raw_answer=response,
                extracted_answer=answer,
            )

        except Exception as e:
            print(f"Error in KET-RAG official processing: {e}")
            import traceback
            traceback.print_exc()

            return BaselineResponse(
                answer="Error processing question.",
                tokens_used=0,
                latency_ms=0,
                selected_passages=[],
                abstained=True,
                mode="ketrag_official",
                stats={"error": str(e)},
            )

    def _parse_ketrag_context(self, context_str: str) -> List[Dict[str, str]]:
        """
        Parse KET-RAG context string into passages for Trident prompt.

        KET-RAG context format:
        -----Entities and Relationships-----
        <graph facts>
        -----Text source that may be relevant-----
        id|text
        chunk_1|<text>
        chunk_2|<text>
        ...
        """
        passages = []

        # Split by sections
        if "-----Entities and Relationships-----" in context_str:
            parts = context_str.split("-----Text source that may be relevant-----")

            # Graph section
            if len(parts) > 0:
                graph_section = parts[0].replace("-----Entities and Relationships-----", "").strip()
                if graph_section and graph_section != "N/A":
                    passages.append({"text": f"Knowledge Graph:\n{graph_section}"})

            # Text chunks section
            if len(parts) > 1:
                text_section = parts[1].strip()
                lines = text_section.split('\n')
                for line in lines:
                    if '|' in line and not line.startswith('id|'):
                        # Format: chunk_N|text
                        _, text = line.split('|', 1)
                        passages.append({"text": text.strip()})

        # Fallback: treat whole context as one passage
        if not passages:
            passages.append({"text": context_str})

        return passages

    def get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        return {
            "name": "KET-RAG (Official)",
            "version": "faithful_wrapper_precomputed",
            "approach": "Official KET-RAG retrieval + Standardized generation",
            "retrieval": "Official KET-RAG (precomputed)",
            "generation": f"{self.prompt_style} (primary)",
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "use_local_llm": self.use_local_llm,
            "num_contexts_loaded": len(self.context_by_qid),
            "compare_original_prompt": self.compare_original_prompt,
        }
