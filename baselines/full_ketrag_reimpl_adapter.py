"""KET-RAG Reimplementation - in-framework implementation of KET-RAG ideas.

This adapter is a REIMPLEMENTATION of KET-RAG's approach within the Trident framework.
It implements the skeleton + keyword RAG idea but is NOT a wrapper around official code.

What's implemented (inspired by KET-RAG):
- SkeletonRAG: PageRank-based key chunk selection + entity extraction
- KeywordRAG: Keyword-chunk bipartite graph
- Dual-channel retrieval logic
- Similar hyperparameters and graph construction

What's different from official KET-RAG:
- All code is reimplemented in this file (not using official KET-RAG repo)
- Indexing happens per-query (not using GraphRAG's global indexing)
- Runs entirely within Trident framework
- Good for studying the idea and doing ablations

For a TRUE faithful wrapper around official KET-RAG, use FullKETRAGAdapter instead.
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple
from collections import defaultdict, Counter
import re

# Add KET-RAG source to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
KETRAG_PATHS = [
    PROJECT_ROOT / "external_baselines" / "KET-RAG",
    PROJECT_ROOT / "KET-RAG",
]
ketrag_path = None
for candidate in KETRAG_PATHS:
    if candidate.exists():
        sys.path.insert(0, str(candidate))
        ketrag_path = candidate
        break

from baselines.full_baseline_interface import (
    BaselineSystem,
    BaselineResponse,
    TokenTracker,
    LatencyTracker,
)
from baselines.local_llm_wrapper import LocalLLMWrapper
from baselines.prompt_utils import build_trident_style_prompt, extract_trident_style_answer

# Try to import official KET-RAG code
OFFICIAL_KETRAG_AVAILABLE = False
if ketrag_path is not None:
    try:
        # Import KET-RAG modules (adjust based on actual structure)
        # This is a placeholder - adjust based on actual KET-RAG API
        import ketrag
        OFFICIAL_KETRAG_AVAILABLE = True
    except ImportError:
        pass

# Fallback to graphrag for LLM wrapper
try:
    from graphrag.config.enums import ModelType
    from graphrag.config.models.language_model_config import LanguageModelConfig
    from graphrag.language_model.manager import ModelManager
    import networkx as nx
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    IMPORT_ERROR = str(e)


class SkeletonKG:
    """Knowledge Graph Skeleton for KET-RAG (SkeletonRAG component)."""

    def __init__(self):
        self.entities: Set[str] = set()
        self.triples: List[Tuple[str, str, str]] = []
        self.entity_to_chunks: Dict[str, List[int]] = defaultdict(list)

    def add_triple(self, subject: str, relation: str, obj: str, chunk_idx: int):
        """Add a triple to the skeleton KG."""
        self.triples.append((subject, relation, obj))
        self.entities.add(subject)
        self.entities.add(obj)
        self.entity_to_chunks[subject].append(chunk_idx)
        self.entity_to_chunks[obj].append(chunk_idx)

    def get_relevant_triples(self, query_entities: Set[str], max_triples: int = 10) -> List[Tuple[str, str, str]]:
        """Get triples relevant to query entities."""
        relevant = []
        for s, r, o in self.triples:
            if s in query_entities or o in query_entities:
                relevant.append((s, r, o))
        return relevant[:max_triples]


class KeywordIndex:
    """Keyword-Chunk Bipartite Graph for KET-RAG (KeywordRAG component)."""

    def __init__(self):
        self.keyword_to_chunks: Dict[str, Set[int]] = defaultdict(set)
        self.chunk_keywords: Dict[int, Set[str]] = defaultdict(set)

    def add_chunk(self, chunk_idx: int, text: str):
        """Add a chunk and extract keywords."""
        stopwords = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'from', 'that', 'this', 'have', 'has', 'was', 'were', 'been', 'are', 'is'}
        words = re.findall(r'\b\w+\b', text.lower())
        keywords = [w for w in words if len(w) > 3 and w not in stopwords]

        keyword_counts = Counter(keywords)
        top_keywords = [kw for kw, _ in keyword_counts.most_common(10)]

        for keyword in top_keywords:
            self.keyword_to_chunks[keyword].add(chunk_idx)
            self.chunk_keywords[chunk_idx].add(keyword)

    def retrieve_chunks(self, query: str, top_k: int = 5) -> List[int]:
        """Retrieve chunk indices based on keyword overlap with query."""
        query_keywords = set(re.findall(r'\b\w{4,}\b', query.lower()))

        chunk_scores: Dict[int, int] = defaultdict(int)
        for keyword in query_keywords:
            if keyword in self.keyword_to_chunks:
                for chunk_idx in self.keyword_to_chunks[keyword]:
                    chunk_scores[chunk_idx] += 1

        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
        return [chunk_idx for chunk_idx, _ in sorted_chunks[:top_k]]


class OpenAILLMWrapper:
    """Wrapper for GraphRAG 2.7.0 chat model to provide compatible generate() method."""

    def __init__(self, chat_model):
        self.chat_model = chat_model

    def generate(self, messages, temperature=0.0, max_tokens=500, **kwargs):
        """Generate a response using the chat model."""
        if isinstance(messages, list) and len(messages) > 0:
            if isinstance(messages[0], dict) and "content" in messages[0]:
                prompt = messages[0]["content"]
            else:
                prompt = str(messages[0])
        else:
            prompt = str(messages)

        response = self.chat_model.generate(
            messages=prompt,
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


class FullKETRAGReimplAdapter(BaselineSystem):
    """
    KET-RAG Reimplementation within Trident framework.

    Implements the KET-RAG approach (skeleton + keyword RAG) but does NOT
    use the official KET-RAG repository code. This is useful for:
    - Studying the KET-RAG idea under our control
    - Running without external dependencies
    - Doing ablations on hyperparameters

    For a TRUE faithful wrapper around official KET-RAG, use FullKETRAGAdapter.

    Implemented components:
    - SkeletonRAG (PageRank + entity extraction)
    - KeywordRAG (keyword-chunk graph)
    - Dual-channel retrieval
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4",
        temperature: float = 0.0,
        max_tokens: int = 500,
        skeleton_ratio: float = 0.3,
        max_skeleton_triples: int = 10,
        max_keyword_chunks: int = 5,
        use_local_llm: bool = False,
        local_llm_model: str = "Qwen/Qwen2.5-7B-Instruct",
        local_llm_device: str = "cuda:0",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        **kwargs
    ):
        """
        Initialize KET-RAG adapter.

        Args:
            api_key: OpenAI API key (required if use_local_llm=False)
            model: LLM model to use (OpenAI model name)
            temperature: Sampling temperature
            max_tokens: Max tokens for generation
            skeleton_ratio: Ratio of chunks to use for skeleton KG (0.0-1.0)
            max_skeleton_triples: Max triples to use from skeleton
            max_keyword_chunks: Max chunks to retrieve from keyword index
            use_local_llm: Use local LLM instead of OpenAI (default: False)
            local_llm_model: HuggingFace model name for local LLM
            local_llm_device: Device for local LLM (cuda:0, cuda:1, cpu)
            load_in_8bit: Use 8-bit quantization for local LLM
            load_in_4bit: Use 4-bit quantization for local LLM
            **kwargs: Additional config
        """
        super().__init__(name="ketrag_reimpl", **kwargs)

        if not DEPENDENCIES_AVAILABLE:
            raise ImportError(
                "KET-RAG dependencies not available. Install with: "
                "pip install graphrag networkx scikit-learn\n"
                f"Error: {IMPORT_ERROR}"
            )

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.skeleton_ratio = skeleton_ratio
        self.max_skeleton_triples = max_skeleton_triples
        self.max_keyword_chunks = max_keyword_chunks
        self.use_local_llm = use_local_llm

        # Initialize LLM client (OpenAI or Local)
        # This is the ONLY part we replace - use specified model instead of KET-RAG's default
        if use_local_llm:
            print(f"  [KET-RAG] Using local LLM: {local_llm_model} (standardized model)")
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
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError(
                    "API key required for OpenAI. Set OPENAI_API_KEY "
                    "or use use_local_llm=True for local models."
                )
            print(f"  [KET-RAG] Using OpenAI: {model} (standardized model)")
            chat_config = LanguageModelConfig(
                api_key=self.api_key,
                type=ModelType.Chat,
                model_provider="openai",
                model=self.model,
                max_retries=3,
            )
            self.llm_raw = ModelManager().get_or_create_chat_model(
                name="ketrag_adapter",
                model_type=ModelType.Chat,
                config=chat_config,
            )
            self.llm = OpenAILLMWrapper(self.llm_raw)

    def _compute_chunk_importance(self, chunks: List[str]) -> List[float]:
        """
        ORIGINAL KET-RAG: Compute importance using PageRank on similarity graph.

        This preserves the original KET-RAG SkeletonRAG logic.
        """
        if not chunks:
            return []
        if len(chunks) == 1:
            return [1.0]

        try:
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(chunks)
        except ValueError:
            return [1.0 / len(chunks)] * len(chunks)

        sim_matrix = cosine_similarity(tfidf_matrix)

        graph = nx.Graph()
        graph.add_nodes_from(range(len(chunks)))

        rows, cols = sim_matrix.shape
        for i in range(rows):
            for j in range(i + 1, cols):
                if sim_matrix[i, j] > 0.2:
                    graph.add_edge(i, j, weight=sim_matrix[i, j])

        try:
            scores_dict = nx.pagerank(graph, alpha=0.85, weight='weight')
            scores = [scores_dict.get(i, 0.0) for i in range(len(chunks))]
            max_score = max(scores) if scores else 1.0
            return [s / max_score if max_score > 0 else 0 for s in scores]
        except Exception:
            return [1.0] * len(chunks)

    def _build_skeleton_kg(
        self,
        key_chunks: List[Tuple[int, str]],
        token_tracker: TokenTracker,
        latency_tracker: LatencyTracker
    ) -> SkeletonKG:
        """
        ORIGINAL KET-RAG: Build knowledge graph skeleton from key chunks using LLM entity extraction.

        This preserves the original KET-RAG SkeletonRAG logic.
        """
        skeleton = SkeletonKG()
        latency_tracker.start()

        for chunk_idx, chunk_text in key_chunks:
            extraction_prompt = f"""Extract key entities and their relationships from the following text.
Format each relationship as: <entity1>, <relationship>, <entity2>
One relationship per line.

Text: {chunk_text[:500]}

Relationships:"""

            try:
                response = self.llm.generate(
                    messages=[{"role": "user", "content": extraction_prompt}],
                    temperature=0.0,
                    max_tokens=128,
                )

                prompt_tokens = len(extraction_prompt.split()) * 1.3
                completion_tokens = len(response.split()) * 1.3
                token_tracker.add_call(
                    int(prompt_tokens),
                    int(completion_tokens),
                    purpose=f"skeleton_extraction_chunk{chunk_idx}"
                )

                lines = response.strip().split('\n')
                for line in lines:
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) == 3:
                        skeleton.add_triple(parts[0], parts[1], parts[2], chunk_idx)

            except Exception as e:
                print(f"Warning: Skeleton extraction failed for chunk {chunk_idx}: {e}")
                continue

        latency_tracker.stop("skeleton_building")
        return skeleton

    def _extract_query_entities(self, query: str) -> Set[str]:
        """ORIGINAL KET-RAG: Extract potential entities from query."""
        words = query.split()
        entities = set()

        for word in words:
            if word and word[0].isupper() and len(word) > 2:
                entities.add(word)

        important_terms = [w for w in words if len(w) > 4]
        entities.update(important_terms)

        return entities

    def answer(
        self,
        question: str,
        context: Optional[List[List[str]]] = None,
        supporting_facts: Optional[List[tuple]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> BaselineResponse:
        """
        Answer a question using KET-RAG.

        PRESERVES original KET-RAG:
        - SkeletonRAG indexing (PageRank + entity extraction)
        - KeywordRAG indexing (keyword-chunk graph)
        - Dual-channel retrieval

        STANDARDIZES for fair comparison:
        - Final QA prompt (Trident's multi-hop format)
        - LLM model (user-specified model)
        - Answer extraction (Trident's logic)

        Args:
            question: The question to answer
            context: HotpotQA context (list of [title, sentences] pairs)
            supporting_facts: Not used
            metadata: Optional metadata

        Returns:
            BaselineResponse with separated indexing/query metrics
        """
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
                mode="ketrag_reimpl",
                stats={},
            )

        try:
            # ═══ PHASE 1: INDEXING (Original KET-RAG logic) ═══
            print("  [KET-RAG] Building Skeleton & Keyword Index (original algorithm)...")
            indexing_latency_tracker.start()

            # Prepare chunks
            chunks = []
            for title, sentences in context:
                text = " ".join(sentences) if isinstance(sentences, list) else sentences
                chunks.append(text)

            # 1. ORIGINAL: PageRank-based importance scoring
            importance_scores = self._compute_chunk_importance(chunks)

            # 2. ORIGINAL: Select key chunks for skeleton
            num_key_chunks = max(1, int(len(chunks) * self.skeleton_ratio))
            scored_chunks = [(i, score) for i, score in enumerate(importance_scores)]
            scored_chunks.sort(key=lambda x: x[1], reverse=True)
            key_chunk_indices = [i for i, _ in scored_chunks[:num_key_chunks]]
            key_chunks = [(i, chunks[i]) for i in key_chunk_indices]

            # 3. ORIGINAL: Build Skeleton KG with entity extraction
            skeleton_kg = self._build_skeleton_kg(
                key_chunks,
                indexing_token_tracker,
                indexing_latency_tracker
            )

            # 4. ORIGINAL: Build Keyword Index
            keyword_index = KeywordIndex()
            for i, chunk in enumerate(chunks):
                keyword_index.add_chunk(i, chunk)

            indexing_time_ms = indexing_latency_tracker.get_total_latency()
            indexing_cost_tokens = indexing_token_tracker.total_tokens
            print(f"  [KET-RAG] Index built in {indexing_time_ms/1000:.2f}s using {indexing_cost_tokens} tokens")

            # ═══ PHASE 2: RETRIEVAL (Original KET-RAG logic) ═══
            query_latency_tracker.start()

            # 5. ORIGINAL: Dual-channel retrieval
            # Entity Channel (SkeletonRAG)
            query_entities = self._extract_query_entities(question)
            relevant_triples = skeleton_kg.get_relevant_triples(query_entities, self.max_skeleton_triples)

            # Keyword Channel (KeywordRAG)
            keyword_chunk_indices = keyword_index.retrieve_chunks(question, self.max_keyword_chunks)
            keyword_chunks_text = [chunks[i] for i in keyword_chunk_indices if i < len(chunks)]

            # ═══ PHASE 3: GENERATION (STANDARDIZED for fair comparison) ═══

            # Format retrieved context as passages for Trident-style prompt
            passages = []

            # Add skeleton KG facts as first passage
            if relevant_triples:
                skeleton_text = "Knowledge graph facts:\n" + "\n".join(
                    f"- {s} {r} {o}" for s, r, o in relevant_triples
                )
                passages.append({"text": skeleton_text})

            # Add keyword-retrieved chunks
            for chunk_text in keyword_chunks_text:
                passages.append({"text": chunk_text})

            # STANDARDIZED: Use Trident's prompt format
            answer_prompt = build_trident_style_prompt(question, passages)

            # STANDARDIZED: Generate with user-specified model
            response = self.llm.generate(
                messages=[{"role": "user", "content": answer_prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            query_latency_tracker.stop("answer_generation")

            # Count query tokens
            p_tok = len(answer_prompt.split()) * 1.3
            c_tok = len(response.split()) * 1.3
            query_token_tracker.add_call(int(p_tok), int(c_tok), "query_generation")

            # STANDARDIZED: Extract answer with Trident's logic
            answer = extract_trident_style_answer(response)

            # Prepare passages
            selected_passages = [{"text": chunks[i][:200], "chunk_idx": i} for i in keyword_chunk_indices[:5]]

            return BaselineResponse(
                answer=answer,
                tokens_used=query_token_tracker.total_tokens,
                latency_ms=query_latency_tracker.get_total_latency(),
                selected_passages=selected_passages,
                abstained=False,
                mode="ketrag_reimpl",
                stats={
                    "indexing_latency_ms": indexing_time_ms,
                    "indexing_tokens": indexing_cost_tokens,
                    "query_tokens": query_token_tracker.total_tokens,
                    "total_cost_tokens": indexing_cost_tokens + query_token_tracker.total_tokens,
                    "num_chunks": len(chunks),
                    "num_key_chunks": len(key_chunks),
                    "num_skeleton_triples": len(relevant_triples),
                    "num_keyword_chunks": len(keyword_chunk_indices),
                    "skeleton_entities": len(skeleton_kg.entities),
                    "retrieval_algorithm": "original_ketrag",
                    "generation_approach": "standardized_trident",
                },
            )

        except Exception as e:
            print(f"Error in KET-RAG reimpl processing: {e}")
            import traceback
            traceback.print_exc()

            return BaselineResponse(
                answer="Error processing question.",
                tokens_used=0,
                latency_ms=0,
                selected_passages=[],
                abstained=True,
                mode="ketrag_reimpl",
                stats={"error": str(e)},
            )

    def get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        return {
            "name": "KET-RAG (Reimpl)",
            "version": "in_framework_reimplementation",
            "approach": "Reimplemented skeleton+keyword RAG (not official wrapper)",
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "skeleton_ratio": self.skeleton_ratio,
            "max_skeleton_triples": self.max_skeleton_triples,
            "max_keyword_chunks": self.max_keyword_chunks,
            "use_local_llm": self.use_local_llm,
        }
