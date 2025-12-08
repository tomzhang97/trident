"""Full Self-RAG adapter using the vanilla Self-RAG implementation.

This adapter uses the original Self-RAG code from self-rag/retrieval_lm/
to run inference exactly as the original paper intended.
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

from baselines.full_baseline_interface import (
    BaselineSystem,
    BaselineResponse,
    TokenTracker,
    LatencyTracker,
)

# Add the self-rag folder to path for imports
SELFRAG_PATH = Path(__file__).parent.parent / "self-rag" / "retrieval_lm"
if str(SELFRAG_PATH) not in sys.path:
    sys.path.insert(0, str(SELFRAG_PATH))

try:
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    VLLM_AVAILABLE = True
except ImportError as e:
    VLLM_AVAILABLE = False
    VLLM_IMPORT_ERROR = str(e)

# Import from vanilla Self-RAG
try:
    from utils import (
        PROMPT_DICT,
        load_special_tokens,
        postprocess,
        control_tokens,
    )
    from run_short_form import call_model_rerank_w_scores_batch
    from metrics import qa_f1_score, normalize_answer
    SELFRAG_IMPORTS_AVAILABLE = True
except ImportError as e:
    SELFRAG_IMPORTS_AVAILABLE = False
    SELFRAG_IMPORT_ERROR = str(e)


class FullSelfRAGAdapter(BaselineSystem):
    """
    Full Self-RAG adapter using the vanilla Self-RAG implementation.

    This adapter uses the original Self-RAG code from self-rag/retrieval_lm/
    including:
    - call_model_rerank_w_scores_batch() for inference with reflection token scoring
    - postprocess() for cleaning output
    - load_special_tokens() for reflection token IDs
    - Original prompt format from PROMPT_DICT

    Self-RAG reflection tokens:
    - [Retrieval] / [No Retrieval]: Retrieval decision
    - [Relevant] / [Irrelevant]: Context relevance
    - [Fully supported] / [Partially supported] / [No support / Contradictory]: Groundedness
    - [Utility:1-5]: Utility score
    """

    def __init__(
        self,
        model_name: str = "selfrag/selfrag_llama2_7b",
        download_dir: Optional[str] = None,
        max_tokens: int = 100,
        temperature: float = 0.0,
        top_p: float = 1.0,
        use_groundness: bool = True,
        use_utility: bool = True,
        use_seqscore: bool = False,
        threshold: Optional[float] = None,  # None = check generated text for [Retrieval] token
        w_rel: float = 1.0,
        w_sup: float = 1.0,
        w_use: float = 0.5,
        mode: str = "adaptive_retrieval",
        ndocs: int = 10,
        gpu_memory_utilization: float = 0.5,
        device: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Self-RAG adapter with vanilla Self-RAG settings.

        Args:
            model_name: HuggingFace model name
                       - "selfrag/selfrag_llama2_7b" (recommended)
                       - "selfrag/selfrag_llama2_13b"
            download_dir: Directory to cache downloaded model
            max_tokens: Max tokens to generate
            temperature: Sampling temperature (0.0 for greedy)
            top_p: Top-p sampling
            use_groundness: Use groundedness/support tokens for scoring
            use_utility: Use utility tokens for scoring
            use_seqscore: Include sequence probability in scoring
            threshold: Probability threshold for adaptive retrieval. Set to None to check
                      generated text for [Retrieval] token (recommended for newer vLLM versions
                      which limit logprobs to 20)
            w_rel: Weight for relevance scores (default 1.0)
            w_sup: Weight for support/groundness scores (default 1.0)
            w_use: Weight for utility scores (default 0.5)
            mode: Retrieval mode - "adaptive_retrieval", "no_retrieval", "always_retrieve"
            ndocs: Number of documents to use for retrieval scoring
            gpu_memory_utilization: GPU memory fraction to use (default 0.5)
            device: GPU device to use (e.g., "cuda:0", "cuda:2", or "0", "2")
            **kwargs: Additional config
        """
        super().__init__(name="selfrag", **kwargs)

        if not VLLM_AVAILABLE:
            raise ImportError(
                f"vllm library not available. Install with: pip install vllm\n"
                f"Error: {VLLM_IMPORT_ERROR}"
            )

        if not SELFRAG_IMPORTS_AVAILABLE:
            raise ImportError(
                f"Self-RAG imports not available. Check self-rag/retrieval_lm/ folder.\n"
                f"Error: {SELFRAG_IMPORT_ERROR}"
            )

        self.model_name = model_name
        self.download_dir = download_dir or os.getenv("HF_CACHE_DIR")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

        # Self-RAG specific settings (from vanilla implementation)
        self.use_groundness = use_groundness
        self.use_utility = use_utility
        self.use_seqscore = use_seqscore
        self.threshold = threshold
        self.w_rel = w_rel
        self.w_sup = w_sup
        self.w_use = w_use
        self.mode = mode
        self.ndocs = ndocs

        # Set CUDA device if specified
        if device is not None:
            if device.startswith("cuda:"):
                device_id = device.split(":")[-1]
            else:
                device_id = device
            print(f"Setting CUDA_VISIBLE_DEVICES={device_id} for Self-RAG")
            os.environ["CUDA_VISIBLE_DEVICES"] = device_id

        # Load model using vLLM (same as vanilla Self-RAG)
        print(f"Loading Self-RAG model: {model_name}...")
        self.model = LLM(
            model_name,
            download_dir=self.download_dir,
            dtype="half",
            tensor_parallel_size=1,
            gpu_memory_utilization=gpu_memory_utilization,
        )

        # Load tokenizer for special token IDs
        print(f"Loading tokenizer for reflection tokens...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

        # Get reflection token IDs using vanilla Self-RAG function
        self.ret_tokens, self.rel_tokens, self.grd_tokens, self.ut_tokens = load_special_tokens(
            self.tokenizer,
            use_grounding=self.use_groundness,
            use_utility=self.use_utility
        )

        print(f"Self-RAG initialized with mode={mode}, threshold={threshold}")

    def _format_evidences(self, context: List[List[str]]) -> List[Dict[str, str]]:
        """
        Format HotpotQA context into Self-RAG evidence format.

        Args:
            context: HotpotQA context (list of [title, sentences] pairs)

        Returns:
            List of evidence dicts with 'title' and 'text' keys
        """
        evidences = []
        for title, sentences in context:
            text = " ".join(sentences) if isinstance(sentences, list) else sentences
            evidences.append({"title": title, "text": text})
        return evidences

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using the model's tokenizer."""
        return len(self.tokenizer.encode(text))

    def answer(
        self,
        question: str,
        context: Optional[List[List[str]]] = None,
        supporting_facts: Optional[List[tuple]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> BaselineResponse:
        """
        Answer a question using vanilla Self-RAG implementation.

        Uses call_model_rerank_w_scores_batch() from the original Self-RAG code
        for inference with full reflection token scoring.

        Args:
            question: The question to answer
            context: HotpotQA context (list of [title, sentences] pairs)
            supporting_facts: Not used by Self-RAG
            metadata: Optional metadata

        Returns:
            BaselineResponse with answer and metrics
        """
        query_token_tracker = TokenTracker()
        query_latency_tracker = LatencyTracker()

        try:
            # Format prompt using vanilla Self-RAG format
            prompt = PROMPT_DICT["prompt_no_input"].format_map({"instruction": question})

            # Format evidences from context
            evidences = self._format_evidences(context) if context else []

            # Limit evidences to ndocs
            evidences = evidences[:self.ndocs]

            # Generate using vanilla Self-RAG function
            query_latency_tracker.start()
            pred, results, do_retrieve = call_model_rerank_w_scores_batch(
                prompt=prompt,
                evidences=evidences,
                model=self.model,
                max_new_tokens=self.max_tokens,
                ret_tokens=self.ret_tokens,
                rel_tokens=self.rel_tokens,
                grd_tokens=self.grd_tokens,
                ut_tokens=self.ut_tokens,
                use_seqscore=self.use_seqscore,
                threshold=self.threshold,
                w_rel=self.w_rel,
                w_sup=self.w_sup,
                w_use=self.w_use,
                mode=self.mode,
                closed=False,  # HotpotQA is open-domain QA
            )
            query_latency_tracker.stop("generation")

            # pred is already postprocessed by call_model_rerank_w_scores_batch
            # Use the vanilla postprocess function to ensure clean output
            answer = postprocess(pred) if pred else pred

            # Get raw generation from results for debugging
            raw_generation = None
            if "no_retrieval" in results:
                raw_generation = results["no_retrieval"]
            elif results:
                # Get the best retrieval result
                for key in results:
                    if key.startswith("retrieval_"):
                        raw_generation = results[key].get("pred", "")
                        break

            # Count tokens
            prompt_tokens = self._count_tokens(prompt)
            completion_tokens = self._count_tokens(answer) if answer else 0
            query_token_tracker.add_call(prompt_tokens, completion_tokens, purpose="generation")

            # Prepare selected passages
            selected_passages = []
            if context:
                for title, sentences in context[:3]:
                    text = " ".join(sentences) if isinstance(sentences, list) else sentences
                    selected_passages.append({"text": text, "title": title})

            return BaselineResponse(
                answer=answer,
                tokens_used=query_token_tracker.total_tokens,
                latency_ms=query_latency_tracker.get_total_latency(),
                selected_passages=selected_passages,
                abstained=False,
                mode="selfrag",
                stats={
                    "indexing_latency_ms": 0.0,
                    "indexing_tokens": 0,
                    "query_tokens": query_token_tracker.total_tokens,
                    "total_cost_tokens": query_token_tracker.total_tokens,
                    # Self-RAG specific stats
                    "raw_generation": raw_generation,
                    "do_retrieve": do_retrieve,
                    "all_results": results,
                    "num_context_docs": len(context) if context else 0,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    # Vanilla Self-RAG settings
                    "mode": self.mode,
                    "threshold": self.threshold,
                    "use_groundness": self.use_groundness,
                    "use_utility": self.use_utility,
                },
                raw_answer=raw_generation,
                extracted_answer=answer,
            )

        except Exception as e:
            print(f"Error in Self-RAG processing: {e}")
            import traceback
            traceback.print_exc()

            return BaselineResponse(
                answer="Error processing question.",
                tokens_used=query_token_tracker.total_tokens,
                latency_ms=query_latency_tracker.get_total_latency(),
                selected_passages=[],
                abstained=True,
                mode="selfrag",
                stats={
                    "indexing_latency_ms": 0.0,
                    "indexing_tokens": 0,
                    "query_tokens": query_token_tracker.total_tokens,
                    "total_cost_tokens": query_token_tracker.total_tokens,
                    "error": str(e),
                },
                raw_answer=None,
                extracted_answer=None,
            )

    def get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        return {
            "name": "Self-RAG (Vanilla)",
            "version": "vanilla",
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "mode": self.mode,
            "threshold": self.threshold,
            "use_groundness": self.use_groundness,
            "use_utility": self.use_utility,
            "use_seqscore": self.use_seqscore,
            "w_rel": self.w_rel,
            "w_sup": self.w_sup,
            "w_use": self.w_use,
            "ndocs": self.ndocs,
        }
