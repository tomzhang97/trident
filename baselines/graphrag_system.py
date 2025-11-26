"""GraphRAG baseline system for comparison with TRIDENT."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import re

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from trident.llm_instrumentation import QueryStats, timed_llm_call, InstrumentedLLM


# GraphRAG Prompts
GRAPHRAG_NODE_SELECTION_PROMPT = """You are an expert in knowledge graphs.

You are given a user question and a list of candidate graph nodes with short descriptions.
Select the nodes that are most relevant to answering the question.

Question:
{question}

Candidate nodes (ID: description):
{nodes}

Instructions:
- Choose between 3 and 10 node IDs.
- Only output a comma-separated list of IDs, with no explanation.

Selected node IDs:
"""


GRAPHRAG_COMMUNITY_SUMMARY_PROMPT = """You are summarizing a subgraph for question answering.

Question:
{question}

Subgraph facts:
{triples}

Each fact is a triple of the form (subject, relation, object).
Write a concise summary (3-5 sentences) of the key facts in this subgraph that may help answer the question.
Do not mention the words "graph" or "triple". Just write the facts as natural language.

Summary:
"""


GRAPHRAG_ANSWER_PROMPT = """You are a question answering assistant.

Question:
{question}

Relevant knowledge summaries:
{summaries}

Instructions:
- Use the summaries above as the primary source of truth.
- If the summaries do not contain the answer, say you do not know.
- Answer with a single short phrase or sentence that directly answers the question. Do not add explanations.

Answer:
"""


class SimpleGraphIndex:
    """
    Simple graph index built from retrieved documents.
    This is a simplified version for baseline comparison.
    """

    def __init__(self):
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: List[Tuple[str, str, str]] = []

    def build_from_documents(self, docs: List[Any]) -> None:
        """
        Build a simple graph from documents.

        Args:
            docs: List of documents (with text attribute or dict with 'text' key)
        """
        self.nodes.clear()
        self.edges.clear()

        for i, doc in enumerate(docs):
            # Extract text
            if hasattr(doc, "text"):
                text = doc.text
            elif isinstance(doc, dict):
                text = doc.get("text", str(doc))
            else:
                text = str(doc)

            # Create node
            node_id = f"n{i}"
            # Simple description: first sentence or first 100 chars
            sentences = re.split(r'[.!?]+', text)
            description = sentences[0][:100] if sentences else text[:100]

            self.nodes[node_id] = {
                "id": node_id,
                "description": description,
                "text": text
            }

            # Simple edge creation: connect consecutive nodes
            if i > 0:
                self.edges.append((f"n{i-1}", "related_to", node_id))

    def search_nodes(self, question: str, topk: int = 20) -> List[Dict[str, Any]]:
        """
        Search for relevant nodes.

        Args:
            question: The question to search for
            topk: Number of top nodes to return

        Returns:
            List of node dictionaries
        """
        # Simple keyword-based search
        question_lower = question.lower()
        scored_nodes = []

        for node_id, node in self.nodes.items():
            text_lower = node["text"].lower()
            # Count keyword matches
            words = question_lower.split()
            score = sum(1 for word in words if len(word) > 3 and word in text_lower)
            scored_nodes.append((score, node))

        # Sort by score and return top-k
        scored_nodes.sort(key=lambda x: x[0], reverse=True)
        return [node for _, node in scored_nodes[:topk]]

    def expand_subgraph(
        self,
        seed_node_ids: List[str],
        max_hops: int = 2,
        max_nodes: int = 64
    ) -> Dict[str, List[Tuple[str, str, str]]]:
        """
        Expand subgraph from seed nodes using BFS.

        Args:
            seed_node_ids: Starting node IDs
            max_hops: Maximum hops from seed nodes
            max_nodes: Maximum nodes to include

        Returns:
            Dict mapping node_id to list of triples
        """
        from collections import defaultdict, deque

        # Build adjacency list
        adj = defaultdict(list)
        for s, r, o in self.edges:
            adj[s].append((s, r, o))
            adj[o].append((s, r, o))  # undirected for simplicity

        visited = set()
        queue = deque([(seed, 0) for seed in seed_node_ids if seed in self.nodes])
        subgraph: Dict[str, List[Tuple[str, str, str]]] = {}

        while queue and len(visited) < max_nodes:
            node_id, hop = queue.popleft()
            if node_id in visited or hop > max_hops:
                continue
            visited.add(node_id)

            node = self.nodes[node_id]
            triples = list(adj[node_id])
            triples.append((node_id, "has_content", node["description"]))
            subgraph[node_id] = triples

            # Enqueue neighbors
            for s, r, o in adj[node_id]:
                neighbor = o if s == node_id else s
                if neighbor not in visited:
                    queue.append((neighbor, hop + 1))

        return subgraph


class GraphRAGSystem:
    """
    GraphRAG baseline system implementation.

    This implements a simplified GraphRAG approach with:
    1. Graph construction from retrieved documents
    2. Seed node selection
    3. Subgraph expansion
    4. Community summarization
    5. Answer generation
    """

    def __init__(
        self,
        llm: InstrumentedLLM,
        retriever: Any,
        topk_nodes: int = 20,
        max_seeds: int = 10,
        max_hops: int = 2,
        k: int = 20
    ):
        """
        Initialize GraphRAG system.

        Args:
            llm: Instrumented LLM wrapper
            retriever: Retrieval system
            topk_nodes: Number of candidate nodes to consider
            max_seeds: Maximum seed nodes to select
            max_hops: Maximum hops for subgraph expansion
            k: Number of documents to retrieve initially
        """
        self.llm = llm
        self.retriever = retriever
        self.topk_nodes = topk_nodes
        self.max_seeds = max_seeds
        self.max_hops = max_hops
        self.k = k

    def _call_llm(self, prompt: str, qstats: QueryStats, **gen_kwargs) -> str:
        """Make a timed LLM call and update query stats."""
        return timed_llm_call(self.llm, prompt, qstats, **gen_kwargs)

    def _format_candidates(self, nodes: List[Dict[str, Any]]) -> str:
        """Format candidate nodes for prompt."""
        return "\n".join(f"{n['id']}: {n['description']}" for n in nodes)

    def _format_triples(self, triples: List[Tuple[str, str, str]]) -> str:
        """Format triples for prompt."""
        return "\n".join(f"({s}, {r}, {o})" for (s, r, o) in triples)

    def answer(
        self,
        question: str,
        context: Optional[List[List[str]]] = None,
        supporting_facts: Optional[List[tuple]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Answer a question using GraphRAG approach.

        Args:
            question: The question to answer
            context: Optional pre-provided context (for datasets like HotpotQA)
            supporting_facts: Optional supporting facts (not used by GraphRAG, for interface compatibility)
            metadata: Optional metadata

        Returns:
            Dictionary with answer, tokens_used, latency_ms, and stats
        """
        qstats = QueryStats()
        graph_index = SimpleGraphIndex()

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
            # Fallback: direct answer without graph
            prompt = GRAPHRAG_ANSWER_PROMPT.format(
                question=question,
                summaries="(no graph information available)",
            )
            answer = self._call_llm(prompt, qstats, max_new_tokens=64).strip()
            return {
                "answer": answer,
                "tokens_used": qstats.total_tokens,
                "latency_ms": qstats.latency_ms,
                "selected_passages": [],
                "abstained": False,
                "mode": "graphrag",
                "stats": {
                    "prompt_tokens": qstats.total_prompt_tokens,
                    "completion_tokens": qstats.total_completion_tokens,
                    "num_calls": qstats.num_calls,
                    "num_seed_nodes": 0,
                    "num_subgraphs": 0,
                }
            }

        # 2) Build graph from documents
        graph_index.build_from_documents(docs)

        # 3) Retrieve candidate nodes
        candidates = graph_index.search_nodes(question, topk=self.topk_nodes)

        # 4) Seed node selection
        node_str = self._format_candidates(candidates)
        seed_prompt = GRAPHRAG_NODE_SELECTION_PROMPT.format(
            question=question,
            nodes=node_str,
        )
        seed_output = self._call_llm(seed_prompt, qstats, max_new_tokens=32)
        seed_ids = [s.strip() for s in seed_output.split(",") if s.strip()]
        seed_ids = seed_ids[: self.max_seeds]

        # Validate seed IDs and fallback if needed
        seed_ids = [s for s in seed_ids if s in graph_index.nodes]

        if not seed_ids:
            # Fallback: use top candidates directly
            seed_ids = [c["id"] for c in candidates[: min(self.max_seeds, len(candidates))]]

        # 5) Subgraph expansion
        subgraph = graph_index.expand_subgraph(
            seed_ids, max_hops=self.max_hops, max_nodes=64
        )

        # 6) Generate summaries for each subgraph
        summaries = []
        for seed_id, triples in subgraph.items():
            triples_str = self._format_triples(triples)
            summary_prompt = GRAPHRAG_COMMUNITY_SUMMARY_PROMPT.format(
                question=question,
                triples=triples_str,
            )
            summary = self._call_llm(summary_prompt, qstats, max_new_tokens=128)
            summaries.append(summary.strip())

        summaries_str = "\n\n".join(f"- {s}" for s in summaries)

        # 7) Final answer
        answer_prompt = GRAPHRAG_ANSWER_PROMPT.format(
            question=question,
            summaries=summaries_str,
        )
        raw_answer = self._call_llm(answer_prompt, qstats, max_new_tokens=64)
        answer = raw_answer.strip()

        return {
            "answer": answer,
            "tokens_used": qstats.total_tokens,
            "latency_ms": qstats.latency_ms,
            "selected_passages": [{"text": d.get("text", str(d)) if isinstance(d, dict) else str(d)} for d in docs[:5]],
            "abstained": False,
            "mode": "graphrag",
            "stats": {
                "prompt_tokens": qstats.total_prompt_tokens,
                "completion_tokens": qstats.total_completion_tokens,
                "num_calls": qstats.num_calls,
                "num_seed_nodes": len(seed_ids),
                "num_subgraphs": len(subgraph),
            },
        }
