"""KET-RAG baseline system for comparison with TRIDENT.

KET-RAG (Knowledge-Enhanced Text RAG) is a cost-efficient multi-granular indexing framework
that combines SkeletonRAG (KG skeleton from key chunks) and KeywordRAG (keyword-chunk bipartite graph).

Paper: KET-RAG: A Cost-Efficient Multi-Granular Indexing Framework for Graph-RAG
https://arxiv.org/abs/2502.09304
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set
from collections import defaultdict, Counter
import re

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from trident.llm_instrumentation import QueryStats, timed_llm_call, InstrumentedLLM


# KET-RAG Prompts
KETRAG_ENTITY_EXTRACTION_PROMPT = """Extract key entities and their relationships from the following text.

Text:
{text}

Format your response as a list of (entity1, relationship, entity2) triples, one per line.
Example:
Albert Einstein, discovered, Theory of Relativity
Paris, is_capital_of, France

Entities and relationships:
"""


KETRAG_ANSWER_PROMPT = """You are a question answering assistant.

Question:
{question}

Knowledge from Skeleton (key facts):
{skeleton_context}

Knowledge from Keywords (related chunks):
{keyword_context}

Instructions:
- Use both the skeleton knowledge and keyword context to answer the question
- Skeleton knowledge contains important facts and relationships
- Keyword context provides additional details from relevant chunks
- If the information is insufficient, say you do not know
- Answer with a single short phrase or sentence that directly answers the question

Answer:
"""


class SkeletonKG:
    """Knowledge Graph Skeleton built from key text chunks."""

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
    """Keyword-Chunk Bipartite Graph for lightweight retrieval."""

    def __init__(self):
        self.keyword_to_chunks: Dict[str, Set[int]] = defaultdict(set)
        self.chunk_keywords: Dict[int, Set[str]] = defaultdict(set)

    def add_chunk(self, chunk_idx: int, text: str):
        """Add a chunk and extract keywords."""
        # Extract keywords (simple: words longer than 3 chars, excluding common words)
        stopwords = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'from', 'that', 'this', 'have', 'has', 'was', 'were', 'been', 'are', 'is'}
        words = re.findall(r'\b\w+\b', text.lower())
        keywords = [w for w in words if len(w) > 3 and w not in stopwords]

        # Count frequency
        keyword_counts = Counter(keywords)
        # Keep top keywords for this chunk
        top_keywords = [kw for kw, _ in keyword_counts.most_common(10)]

        for keyword in top_keywords:
            self.keyword_to_chunks[keyword].add(chunk_idx)
            self.chunk_keywords[chunk_idx].add(keyword)

    def retrieve_chunks(self, query: str, top_k: int = 5) -> List[int]:
        """Retrieve chunk indices based on keyword overlap with query."""
        query_keywords = set(re.findall(r'\b\w{4,}\b', query.lower()))

        # Score chunks by keyword overlap
        chunk_scores: Dict[int, int] = defaultdict(int)
        for keyword in query_keywords:
            if keyword in self.keyword_to_chunks:
                for chunk_idx in self.keyword_to_chunks[keyword]:
                    chunk_scores[chunk_idx] += 1

        # Return top-k chunks by score
        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
        return [chunk_idx for chunk_idx, _ in sorted_chunks[:top_k]]


class KETRAGSystem:
    """
    KET-RAG baseline system implementation.

    Implements a simplified version of KET-RAG with:
    1. SkeletonRAG: PageRank-style selection of key chunks + KG extraction
    2. KeywordRAG: Lightweight keyword-chunk bipartite graph
    3. Dual-channel retrieval: Entity + keyword channels
    4. Answer generation with both contexts
    """

    def __init__(
        self,
        llm: InstrumentedLLM,
        retriever: Any,
        k: int = 8,
        skeleton_ratio: float = 0.3,  # Ratio of chunks to use for skeleton
        max_skeleton_triples: int = 10,
        max_keyword_chunks: int = 5
    ):
        """
        Initialize KET-RAG system.

        Args:
            llm: Instrumented LLM wrapper
            retriever: Retrieval system
            k: Number of documents to retrieve initially
            skeleton_ratio: Proportion of top chunks to use for skeleton KG
            max_skeleton_triples: Max triples to use from skeleton
            max_keyword_chunks: Max chunks from keyword index
        """
        self.llm = llm
        self.retriever = retriever
        self.k = k
        self.skeleton_ratio = skeleton_ratio
        self.max_skeleton_triples = max_skeleton_triples
        self.max_keyword_chunks = max_keyword_chunks

    def _call_llm(self, prompt: str, qstats: QueryStats, **gen_kwargs) -> str:
        """Make a timed LLM call and update query stats."""
        return timed_llm_call(self.llm, prompt, qstats, **gen_kwargs)

    def _compute_chunk_importance(self, chunks: List[str]) -> List[float]:
        """
        Compute importance scores for chunks using PageRank-like algorithm.
        Simplified version: uses term overlap as edge weights.
        """
        n = len(chunks)
        if n == 0:
            return []

        # Build similarity matrix based on term overlap
        scores = [1.0] * n  # Initialize with equal scores

        # Simple approach: chunks with more unique terms are more important
        for i, chunk in enumerate(chunks):
            words = set(re.findall(r'\b\w{4,}\b', chunk.lower()))
            scores[i] = len(words)  # Simple importance: vocabulary richness

        # Normalize
        max_score = max(scores) if scores else 1.0
        return [s / max_score for s in scores]

    def _build_skeleton_kg(
        self,
        key_chunks: List[Tuple[int, str]],
        qstats: QueryStats
    ) -> SkeletonKG:
        """Build knowledge graph skeleton from key chunks."""
        skeleton = SkeletonKG()

        for chunk_idx, chunk_text in key_chunks:
            # Extract entities and relationships using LLM
            extraction_prompt = KETRAG_ENTITY_EXTRACTION_PROMPT.format(text=chunk_text[:500])

            try:
                entities_output = self._call_llm(extraction_prompt, qstats, max_new_tokens=128)

                # Parse triples
                lines = entities_output.strip().split('\n')
                for line in lines:
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) == 3:
                        skeleton.add_triple(parts[0], parts[1], parts[2], chunk_idx)
            except Exception:
                # If extraction fails, skip this chunk
                continue

        return skeleton

    def _extract_query_entities(self, query: str) -> Set[str]:
        """Extract potential entities from query (simple: capitalize words, multi-word phrases)."""
        # Simple heuristic: capitalized words and multi-word phrases
        words = query.split()
        entities = set()

        # Single capitalized words
        for word in words:
            if word[0].isupper() and len(word) > 2:
                entities.add(word)

        # Also add important terms (longer words)
        important_terms = [w for w in words if len(w) > 4]
        entities.update(important_terms)

        return entities

    def answer(
        self,
        question: str,
        context: Optional[List[List[str]]] = None,
        supporting_facts: Optional[List[tuple]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Answer a question using KET-RAG approach.

        Args:
            question: The question to answer
            context: Optional pre-provided context (for datasets like HotpotQA)
            supporting_facts: Optional supporting facts (not used by KET-RAG)
            metadata: Optional metadata

        Returns:
            Dictionary with answer, tokens_used, latency_ms, and stats
        """
        qstats = QueryStats()

        # 1) Retrieve or use provided documents
        docs = []
        if context is not None:
            # Use provided context
            for title, sentences in context:
                text = " ".join(sentences) if isinstance(sentences, list) else sentences
                docs.append({"text": text, "title": title})
        else:
            # Retrieve documents
            if hasattr(self.retriever, 'get_relevant_documents'):
                docs = self.retriever.get_relevant_documents(question)[: self.k]
            elif hasattr(self.retriever, 'retrieve'):
                result = self.retriever.retrieve(question, top_k=self.k)
                docs = result.passages if hasattr(result, 'passages') else result
            else:
                docs = []

        if not docs:
            # Fallback: direct answer without KET-RAG
            answer = "I don't have enough information to answer this question."
            return {
                "answer": answer,
                "tokens_used": qstats.total_tokens,
                "latency_ms": qstats.latency_ms,
                "selected_passages": [],
                "abstained": True,
                "mode": "ketrag",
                "stats": {
                    "prompt_tokens": qstats.total_prompt_tokens,
                    "completion_tokens": qstats.total_completion_tokens,
                    "num_calls": qstats.num_calls,
                    "num_docs": 0,
                    "num_skeleton_triples": 0,
                    "num_keyword_chunks": 0,
                }
            }

        # 2) Extract text chunks
        chunks = []
        for doc in docs:
            if isinstance(doc, dict):
                chunks.append(doc.get("text", str(doc)))
            elif hasattr(doc, "text"):
                chunks.append(doc.text)
            else:
                chunks.append(str(doc))

        # 3) SkeletonRAG: Select key chunks and build KG skeleton
        importance_scores = self._compute_chunk_importance(chunks)
        num_key_chunks = max(1, int(len(chunks) * self.skeleton_ratio))

        # Get indices of top key chunks
        scored_chunks = [(i, score) for i, score in enumerate(importance_scores)]
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        key_chunk_indices = [i for i, _ in scored_chunks[:num_key_chunks]]

        key_chunks = [(i, chunks[i]) for i in key_chunk_indices]
        skeleton_kg = self._build_skeleton_kg(key_chunks, qstats)

        # 4) KeywordRAG: Build keyword-chunk index
        keyword_index = KeywordIndex()
        for i, chunk in enumerate(chunks):
            keyword_index.add_chunk(i, chunk)

        # 5) Query-time dual-channel retrieval
        # Channel 1: Entity/skeleton channel
        query_entities = self._extract_query_entities(question)
        relevant_triples = skeleton_kg.get_relevant_triples(query_entities, self.max_skeleton_triples)

        # Format skeleton context
        if relevant_triples:
            skeleton_context = "\n".join(
                f"- {s} {r} {o}" for s, r, o in relevant_triples
            )
        else:
            skeleton_context = "(no relevant knowledge graph facts found)"

        # Channel 2: Keyword channel
        keyword_chunk_indices = keyword_index.retrieve_chunks(question, self.max_keyword_chunks)
        keyword_chunks_text = [chunks[i] for i in keyword_chunk_indices if i < len(chunks)]

        if keyword_chunks_text:
            keyword_context = "\n\n".join(
                f"[{i+1}] {text[:300]}..." if len(text) > 300 else f"[{i+1}] {text}"
                for i, text in enumerate(keyword_chunks_text)
            )
        else:
            keyword_context = "(no relevant chunks found)"

        # 6) Generate answer using both contexts
        answer_prompt = KETRAG_ANSWER_PROMPT.format(
            question=question,
            skeleton_context=skeleton_context,
            keyword_context=keyword_context
        )
        raw_answer = self._call_llm(answer_prompt, qstats, max_new_tokens=64)
        answer = raw_answer.strip()

        return {
            "answer": answer,
            "tokens_used": qstats.total_tokens,
            "latency_ms": qstats.latency_ms,
            "selected_passages": [{"text": chunks[i][:200]} for i in keyword_chunk_indices[:5]],
            "abstained": False,
            "mode": "ketrag",
            "stats": {
                "prompt_tokens": qstats.total_prompt_tokens,
                "completion_tokens": qstats.total_completion_tokens,
                "num_calls": qstats.num_calls,
                "num_docs": len(docs),
                "num_skeleton_triples": len(relevant_triples),
                "num_keyword_chunks": len(keyword_chunk_indices),
                "num_key_chunks_for_kg": len(key_chunks),
            },
        }
