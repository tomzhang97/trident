"""Self-RAG baseline system for comparison with TRIDENT."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from trident.llm_instrumentation import QueryStats, timed_llm_call, InstrumentedLLM


# Self-RAG Prompts
SELF_RAG_RETRIEVE_PROMPT = """You are a question analyzer.

For the user question below, decide whether external documents need to be retrieved to answer it correctly.

Rules:
- If the answer is obvious common knowledge or purely opinion, respond "NO_RETRIEVE".
- If the question needs specific factual details (names, dates, numbers, events, technical facts), respond "RETRIEVE".

Output exactly one token: either RETRIEVE or NO_RETRIEVE.

Question:
{question}
"""


SELF_RAG_ANSWER_PROMPT = """You are a helpful question answering assistant.

Question:
{question}

Retrieved documents:
{context}

Instructions:
- Use the retrieved documents when they contain relevant facts.
- If the documents do not mention the answer, say you do not know.
- Answer with a single short phrase or sentence that directly answers the question. Do not add explanations.

Answer:
"""


SELF_RAG_CRITIC_PROMPT = """You are verifying whether an answer is supported by the given documents.

Question:
{question}

Answer:
{answer}

Documents:
{context}

Decide whether the answer is fully supported by the documents.

Reply with exactly one of:
- SUPPORTS (if the documents clearly support the answer),
- CONTRADICTS (if the documents directly contradict the answer),
- INSUFFICIENT (if the documents do not clearly support it either way).

Label:
"""


class SelfRAGSystem:
    """
    Self-RAG baseline system implementation.

    This implements a simplified Self-RAG approach with:
    1. Retrieval decision gate
    2. Context-aware answer generation
    3. Optional critic/verification
    """

    def __init__(
        self,
        llm: InstrumentedLLM,
        retriever: Any,
        k: int = 8,
        use_critic: bool = False
    ):
        """
        Initialize Self-RAG system.

        Args:
            llm: Instrumented LLM wrapper
            retriever: Retrieval system (should have get_relevant_documents or retrieve method)
            k: Number of documents to retrieve
            use_critic: Whether to use critic/verification step
        """
        self.llm = llm
        self.retriever = retriever
        self.k = k
        self.use_critic = use_critic

    def _call_llm(self, prompt: str, qstats: QueryStats, **gen_kwargs) -> str:
        """Make a timed LLM call and update query stats."""
        return timed_llm_call(self.llm, prompt, qstats, **gen_kwargs)

    def _format_docs(self, docs: List[Any]) -> str:
        """Format documents for prompt."""
        lines = []
        for i, d in enumerate(docs):
            # Handle different document types
            if hasattr(d, "page_content"):
                text = d.page_content
            elif hasattr(d, "text"):
                text = d.text
            elif isinstance(d, dict):
                text = d.get("text", d.get("content", str(d)))
            else:
                text = str(d)
            lines.append(f"[{i}] {text}")
        return "\n\n".join(lines)

    def answer(
        self,
        question: str,
        context: Optional[List[List[str]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Answer a question using Self-RAG approach.

        Args:
            question: The question to answer
            context: Optional pre-provided context (for datasets like HotpotQA)
            metadata: Optional metadata

        Returns:
            Dictionary with answer, tokens_used, latency_ms, and stats
        """
        qstats = QueryStats()

        # 1) Decide whether to retrieve (if context not provided)
        docs = []
        gate = "PROVIDED_CONTEXT"

        if context is not None:
            # Use provided context directly
            for title, sentences in context:
                text = " ".join(sentences) if isinstance(sentences, list) else sentences
                docs.append({"text": text, "title": title})
        else:
            # Use retrieval gate
            retrieve_prompt = SELF_RAG_RETRIEVE_PROMPT.format(question=question)
            gate_output = self._call_llm(retrieve_prompt, qstats, max_new_tokens=4)
            gate = gate_output.strip().split()[0].upper()
            do_retrieve = (gate == "RETRIEVE")

            # 2) Retrieve if needed
            if do_retrieve:
                # Try different retrieval methods based on retriever type
                if hasattr(self.retriever, 'get_relevant_documents'):
                    docs = self.retriever.get_relevant_documents(question)[: self.k]
                elif hasattr(self.retriever, 'retrieve'):
                    result = self.retriever.retrieve(question, top_k=self.k)
                    docs = result.passages if hasattr(result, 'passages') else result
                else:
                    docs = []

        context_str = self._format_docs(docs) if docs else "(none)"

        # 3) Answer
        answer_prompt = SELF_RAG_ANSWER_PROMPT.format(
            question=question,
            context=context_str,
        )
        raw_answer = self._call_llm(answer_prompt, qstats, max_new_tokens=64)
        answer = raw_answer.strip()

        # 4) Optional critic
        critic_label = None
        if self.use_critic and docs:
            critic_prompt = SELF_RAG_CRITIC_PROMPT.format(
                question=question,
                answer=answer,
                context=context_str,
            )
            critic_output = self._call_llm(critic_prompt, qstats, max_new_tokens=4)
            critic_label = critic_output.strip().split()[0].upper()

        return {
            "answer": answer,
            "tokens_used": qstats.total_tokens,
            "latency_ms": qstats.latency_ms,
            "selected_passages": [{"text": d.get("text", str(d)) if isinstance(d, dict) else str(d)} for d in docs[:5]],
            "abstained": False,
            "mode": "self_rag",
            "stats": {
                "prompt_tokens": qstats.total_prompt_tokens,
                "completion_tokens": qstats.total_completion_tokens,
                "num_calls": qstats.num_calls,
                "retrieval_gate": gate,
                "critic_label": critic_label,
                "num_docs_retrieved": len(docs)
            },
        }
