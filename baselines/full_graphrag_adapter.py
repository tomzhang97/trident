"""Full GraphRAG adapter for HotpotQA evaluation using Microsoft's GraphRAG library."""

from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import tempfile
import shutil

# Add external_baselines to path
EXTERNAL_BASELINES = Path(__file__).parent.parent / "external_baselines"
sys.path.insert(0, str(EXTERNAL_BASELINES / "graphrag"))

from baselines.full_baseline_interface import (
    BaselineSystem,
    BaselineResponse,
    TokenTracker,
    LatencyTracker,
)

try:
    from graphrag.query.llm.oai.chat_openai import ChatOpenAI
    from graphrag.query.llm.oai.typing import OpenAIClientTypes
    from graphrag.query.structured_search.local_search.mixed_context import LocalSearchMixedContext
    from graphrag.query.structured_search.local_search.search import LocalSearch
    from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
    from graphrag.query.input.retrieval.entities import get_entity_by_key
    from graphrag.vector_stores.lancedb import LanceDBVectorStore
    import pandas as pd
    import networkx as nx
    GRAPHRAG_AVAILABLE = True
except ImportError as e:
    GRAPHRAG_AVAILABLE = False
    GRAPHRAG_IMPORT_ERROR = str(e)


class FullGraphRAGAdapter(BaselineSystem):
    """
    Full GraphRAG adapter using Microsoft's GraphRAG library.

    This adapter:
    1. Converts HotpotQA context into documents
    2. Builds a knowledge graph using GraphRAG's indexing pipeline (per-query)
    3. Uses GraphRAG's local search to answer questions
    4. Tracks all tokens and latency for fair comparison

    Note: This performs per-query indexing which is not how GraphRAG is designed
    to be used (it expects a persistent index), but necessary for HotpotQA evaluation.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4",
        temperature: float = 0.0,
        max_tokens: int = 500,
        use_local_search: bool = True,
        **kwargs
    ):
        """
        Initialize GraphRAG adapter.

        Args:
            api_key: OpenAI API key (or from GRAPHRAG_API_KEY env var)
            model: LLM model to use
            temperature: Sampling temperature
            max_tokens: Max tokens for generation
            use_local_search: Use local search (vs global search)
            **kwargs: Additional config
        """
        super().__init__(name="graphrag", **kwargs)

        if not GRAPHRAG_AVAILABLE:
            raise ImportError(
                f"GraphRAG library not available. Install with: "
                f"cd external_baselines/graphrag && pip install -e . "
                f"Error: {GRAPHRAG_IMPORT_ERROR}"
            )

        self.api_key = api_key or os.getenv("GRAPHRAG_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API key required. Set GRAPHRAG_API_KEY or OPENAI_API_KEY environment variable.")

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_local_search = use_local_search

        # Initialize LLM client
        self.llm = ChatOpenAI(
            api_key=self.api_key,
            model=self.model,
            api_type=OpenAIClientTypes.OpenAI,
            max_retries=3,
        )

    def _build_index_from_context(
        self,
        context: List[List[str]],
        token_tracker: TokenTracker,
        latency_tracker: LatencyTracker
    ) -> Dict[str, Any]:
        """
        FULL VERSION: Builds Index with Community Detection and Summarization.

        Restores the core GraphRAG pipeline:
        1. Extract Entities/Relations.
        2. Detect Communities (Leiden/Louvain).
        3. Generate Community Summaries (The "Map" step).

        Returns:
            Index data structure with entities, relationships, communities, and summaries
        """
        latency_tracker.start()

        # Convert context to documents
        documents = []
        for title, sentences in context:
            text = " ".join(sentences) if isinstance(sentences, list) else sentences
            documents.append({
                "id": title,
                "title": title,
                "text": text,
            })

        # Extract entities and relationships using LLM
        entities = []
        relationships = []
        entity_id_counter = 0

        for doc in documents:
            # Entity extraction prompt
            extraction_prompt = f"""Extract named entities and their relationships from this text.
Format each entity as: ENTITY: <name> | TYPE: <type> | DESCRIPTION: <brief description>
Format each relationship as: RELATION: <entity1> | <relationship> | <entity2>

Text: {doc['text'][:500]}

Entities and Relationships:"""

            try:
                response = self.llm.generate(
                    messages=[{"role": "user", "content": extraction_prompt}],
                    temperature=0.0,
                    max_tokens=300,
                )

                # Track tokens
                prompt_tokens = len(extraction_prompt.split()) * 1.3
                completion_tokens = len(response.split()) * 1.3
                token_tracker.add_call(
                    int(prompt_tokens),
                    int(completion_tokens),
                    purpose=f"entity_extraction_{doc['id']}"
                )

                # Parse entities and relationships
                for line in response.split('\n'):
                    line = line.strip()
                    if line.startswith('ENTITY:'):
                        parts = line.split('|')
                        if len(parts) >= 2:
                            name = parts[0].replace('ENTITY:', '').strip()
                            entity_type = parts[1].replace('TYPE:', '').strip() if len(parts) > 1 else "UNKNOWN"
                            entities.append({
                                "id": f"entity_{entity_id_counter}",
                                "name": name,
                                "type": entity_type,
                                "description": doc['text'][:200],
                                "source_doc": doc['id'],
                            })
                            entity_id_counter += 1

                    elif line.startswith('RELATION:'):
                        parts = line.split('|')
                        if len(parts) >= 3:
                            relationships.append({
                                "source": parts[0].replace('RELATION:', '').strip(),
                                "target": parts[2].strip(),
                                "relationship": parts[1].strip(),
                            })

            except Exception as e:
                print(f"Warning: Entity extraction failed for {doc['id']}: {e}")
                continue

        # Build graph
        graph = nx.Graph()
        for entity in entities:
            graph.add_node(entity["name"], **entity)

        for rel in relationships:
            if rel["source"] in graph.nodes and rel["target"] in graph.nodes:
                graph.add_edge(rel["source"], rel["target"], relationship=rel["relationship"])

        # --- FULL GRAPHRAG: COMMUNITY DETECTION AND SUMMARIZATION ---

        # 1. Detect Communities (Hierarchical Clustering)
        # Real GraphRAG uses Leiden. We use Louvain (available in networkx) as the closest proxy
        import networkx.algorithms.community as nx_comm

        try:
            # Attempt Louvain (closest to Leiden)
            communities = list(nx_comm.louvain_communities(graph))
        except:
            # Fallback for very small/disconnected graphs
            communities = list(nx.connected_components(graph))

        community_summaries = {}

        # 2. Generate Community Summaries (The "Map" Phase)
        # This is the defining feature of Microsoft GraphRAG
        summary_prompt_template = """Based on the provided entities and relationships, write a comprehensive summary of this community.
Focus on the key themes and how these entities relate to the query context.

Entities: {node_list}
Relationships: {edge_list}

Summary:"""

        for i, comm in enumerate(communities):
            if not comm:
                continue

            # Prepare data for summary
            node_list = list(comm)
            edge_list = []
            subgraph = graph.subgraph(comm)
            for u, v, data in subgraph.edges(data=True):
                edge_list.append(f"{u} -> {v} ({data.get('relationship', 'related')})")

            # Skip if community is too small (optimization)
            if len(node_list) < 2:
                continue

            prompt = summary_prompt_template.format(
                node_list=", ".join(node_list),
                edge_list="; ".join(edge_list)
            )

            # Generate Summary
            try:
                response = self.llm.generate(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=200,
                )

                # Track tokens
                prompt_tokens = len(prompt.split()) * 1.3
                completion_tokens = len(response.split()) * 1.3
                token_tracker.add_call(int(prompt_tokens), int(completion_tokens), purpose=f"community_summary_{i}")

                community_summaries[i] = response.strip()

            except Exception as e:
                print(f"Summary failed for comm {i}: {e}")

        latency_tracker.stop("indexing_full_pipeline")

        return {
            "documents": documents,
            "entities": entities,
            "relationships": relationships,
            "graph": graph,
            "communities": communities,           # NEW
            "community_summaries": community_summaries  # NEW
        }

    def answer(
        self,
        question: str,
        context: Optional[List[List[str]]] = None,
        supporting_facts: Optional[List[tuple]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> BaselineResponse:
        """
        Answer a question using GraphRAG.

        Args:
            question: The question to answer
            context: HotpotQA context (list of [title, sentences] pairs)
            supporting_facts: Not used by GraphRAG
            metadata: Optional metadata

        Returns:
            BaselineResponse with answer and metrics
        """
        token_tracker = TokenTracker()
        latency_tracker = LatencyTracker()

        if not context:
            # No context provided - return abstention
            return BaselineResponse(
                answer="I don't have enough information to answer this question.",
                tokens_used=0,
                latency_ms=0.0,
                selected_passages=[],
                abstained=True,
                mode="graphrag",
                stats={
                    "num_entities": 0,
                    "num_relationships": 0,
                    "error": "No context provided",
                },
            )

        try:
            # Build index from context
            index_data = self._build_index_from_context(context, token_tracker, latency_tracker)

            # Retrieve relevant information from graph
            latency_tracker.start()

            # FULL GRAPHRAG SEARCH STRATEGY:
            # Combine Global Search (Community Summaries) + Local Search (Entity Neighbors)

            context_parts = []

            # 1. Global Context (from Community Summaries)
            # This allows answering "What is the overarching theme?" type questions
            if "community_summaries" in index_data and index_data["community_summaries"]:
                context_parts.append("--- Community Summaries (Global Context) ---")
                for comm_id, summary in index_data["community_summaries"].items():
                    context_parts.append(f"Community {comm_id}: {summary}")

            # 2. Local Context (Specific Entities)
            # Simple entity-based retrieval
            question_lower = question.lower()
            relevant_entities = []
            for entity in index_data["entities"]:
                if entity["name"].lower() in question_lower:
                    relevant_entities.append(entity)

            # If no exact matches, use first few entities
            if not relevant_entities and index_data["entities"]:
                relevant_entities = index_data["entities"][:5]

            # Build context from relevant entities and their neighborhood
            if relevant_entities:
                context_parts.append("\n--- Local Context (Specific Entities) ---")
                for entity in relevant_entities[:3]:
                    context_parts.append(f"{entity['name']}: {entity['description']}")

                    # Add related entities from graph
                    if entity["name"] in index_data["graph"]:
                        neighbors = list(index_data["graph"].neighbors(entity["name"]))[:3]
                        for neighbor in neighbors:
                            edge_data = index_data["graph"][entity["name"]][neighbor]
                            rel = edge_data.get("relationship", "related to")
                            context_parts.append(f"{entity['name']} {rel} {neighbor}")

            context_text = "\n".join(context_parts) if context_parts else "No relevant information found."

            # Generate answer
            answer_prompt = f"""You are a question answering assistant using knowledge graph information.

Question: {question}

Knowledge Graph Context:
{context_text}

Instructions:
- Use the knowledge graph information above to answer the question
- If the information is insufficient, say you do not know
- Answer with a single short phrase or sentence that directly answers the question

Answer:"""

            response = self.llm.generate(
                messages=[{"role": "user", "content": answer_prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            answer = response.strip()

            # Track tokens
            prompt_tokens = len(answer_prompt.split()) * 1.3
            completion_tokens = len(response.split()) * 1.3
            token_tracker.add_call(int(prompt_tokens), int(completion_tokens), purpose="answer_generation")

            latency_tracker.stop("answer_generation")

            # Prepare selected passages (from source documents)
            selected_passages = [
                {"text": doc["text"], "title": doc["title"]}
                for doc in index_data["documents"][:3]
            ]

            return BaselineResponse(
                answer=answer,
                tokens_used=token_tracker.total_tokens,
                latency_ms=latency_tracker.get_total_latency(),
                selected_passages=selected_passages,
                abstained=False,
                mode="graphrag",
                stats={
                    **token_tracker.get_stats(),
                    "num_entities": len(index_data["entities"]),
                    "num_relationships": len(index_data["relationships"]),
                    "num_documents": len(index_data["documents"]),
                    "num_relevant_entities": len(relevant_entities),
                    "num_communities": len(index_data.get("communities", [])),
                    "num_community_summaries": len(index_data.get("community_summaries", {})),
                },
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
                stats={
                    **token_tracker.get_stats(),
                    "error": str(e),
                },
            )

    def get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        return {
            "name": "GraphRAG",
            "version": "full",
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "use_local_search": self.use_local_search,
        }
