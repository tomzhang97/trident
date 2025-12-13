#!/usr/bin/env python3
"""
KET-RAG Answer Generation with Local LLM

This script generates answers for all context files using a local LLM.
It replaces the original llm_answer.py when using local models.

Usage:
    # Using local LLM directly (HuggingFace Transformers)
    python scripts/llm_answer_local.py ragtest-musique \
        --use_local_llm \
        --local_llm_model meta-llama/Meta-Llama-3-8B-Instruct \
        --device cuda:0

    # Using vLLM server (OpenAI-compatible API)
    python scripts/llm_answer_local.py ragtest-musique \
        --api_base http://localhost:8000/v1 \
        --model meta-llama/Meta-Llama-3-8B-Instruct

    # Using OpenAI API (same as original)
    export GRAPHRAG_API_KEY=your_key
    python scripts/llm_answer_local.py ragtest-musique
"""

import asyncio
import argparse
import glob
import json
import os
import random
import sys
import time
from pathlib import Path

import tiktoken
from tqdm.asyncio import tqdm_asyncio

# Add paths for imports
SCRIPT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "external_baselines" /  "KET-RAG"))
sys.path.insert(0, str(SCRIPT_DIR / "external_baselines" /  "KET-RAG" / "indexing_sket"))

from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
from graphrag.query.structured_search.local_search.mixed_context import LocalSearchMixedContext
from util_v1 import MyLocalSearch


class LocalLLMWrapper:
    """
    LLM wrapper compatible with GraphRAG's BaseLLM interface.
    Supports both local HuggingFace models and OpenAI-compatible APIs.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str = None,
        api_base: str = None,
        use_local_llm: bool = False,
        local_llm_model: str = None,
        device: str = "cuda:0",
        temperature: float = 0.0,
        max_tokens: int = 2000,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_local_llm = use_local_llm

        if use_local_llm:
            from baselines.local_llm_wrapper import LocalLLMWrapper as TridentLLM
            self.local_llm = TridentLLM(
                model_name=local_llm_model or model,
                device=device,
                temperature=temperature,
                max_tokens=max_tokens,
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
            )
            self.client = None
        else:
            import openai
            self.local_llm = None
            self.client = openai.OpenAI(
                api_key=api_key or os.environ.get("GRAPHRAG_API_KEY", "EMPTY"),
                base_url=api_base,
            )

    def _do_generate(self, messages, **kwargs) -> str:
        """Internal generation method."""
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        if self.use_local_llm and self.local_llm:
            return self.local_llm.generate(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content

    def generate(self, messages, streaming=False, callbacks=None, **kwargs) -> str:
        """Synchronous generation."""
        return self._do_generate(messages, **kwargs)

    def stream_generate(self, messages, callbacks=None, **kwargs):
        """Synchronous streaming generation."""
        response = self._do_generate(messages, **kwargs)
        yield response

    async def agenerate(self, messages, streaming=False, callbacks=None, **kwargs) -> str:
        """Async generation - wraps blocking call in thread for local LLM."""
        if self.use_local_llm:
            # Local LLM is blocking, run in thread pool
            return await asyncio.to_thread(self._do_generate, messages, **kwargs)
        else:
            # API calls are already non-blocking via httpx
            return self._do_generate(messages, **kwargs)

    async def astream_generate(self, messages, callbacks=None, **kwargs):
        """Async streaming generation."""
        response = self._do_generate(messages, **kwargs)
        yield response


async def process_qa_pairs(search_engine, golden, context_data, file_path, file_name, max_concurrent=10):
    """Process QA pairs and generate answers."""
    answered_pairs = []
    tasks = []
    semaphore = asyncio.Semaphore(max_concurrent)

    async def limited_asearch(qa):
        async with semaphore:
            # Find the question for this ID
            question = None
            for qa_pair in golden:
                if qa_pair["id"] == qa["id"]:
                    question = qa_pair["question"]
                    break

            if question is None:
                print(f"Warning: No question found for ID {qa['id']}")
                return {"id": qa["id"], "answer": "ERROR: Question not found"}

            time.sleep(0.01 * random.randint(1, 9))  # Small random delay
            result = await search_engine.asearch_with_context(question, qa["context"])
            return {"id": qa["id"], "answer": result.response}

    for qa in context_data:
        tasks.append(limited_asearch(qa))

    results = await tqdm_asyncio.gather(*tasks, desc=f"Processing {file_name}")

    for result in results:
        answered_pairs.append(result)

    # Save answers
    output_path = f"{file_path}/answer-{file_name}"
    with open(output_path, "w") as fw:
        json.dump(answered_pairs, fw, indent=4)
    print(f"  Saved answers to {output_path}")

    return answered_pairs


async def main(args):
    """Main function to process all context files."""
    ROOT_PATH = args.root_path.rstrip("/")

    # Initialize LLM
    print("Initializing LLM...")
    if args.use_local_llm:
        print(f"  Using local LLM: {args.local_llm_model}")
        # Auto-adjust concurrency for local LLM (model can only handle 1 request at a time)
        if args.max_concurrent > 1:
            print(f"  ⚠️  Local LLM: reducing max_concurrent from {args.max_concurrent} to 1")
            args.max_concurrent = 1
        llm = LocalLLMWrapper(
            use_local_llm=True,
            local_llm_model=args.local_llm_model,
            device=args.device,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            load_in_8bit=args.load_in_8bit,
            load_in_4bit=args.load_in_4bit,
        )
    else:
        api_base = args.api_base
        model = args.model
        print(f"  Using API: {api_base or 'OpenAI'} with model {model}")
        llm = LocalLLMWrapper(
            model=model,
            api_key=args.api_key or os.environ.get("GRAPHRAG_API_KEY"),
            api_base=api_base,
            use_local_llm=False,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )

    # Initialize token encoder
    token_encoder = tiktoken.get_encoding("cl100k_base")

    # Local context params (from original llm_answer.py)
    local_context_params = {
        "text_unit_prop": 0.5,
        "community_prop": 0.1,
        "conversation_history_max_turns": 5,
        "conversation_history_user_turns_only": True,
        "top_k_mapped_entities": 10,
        "top_k_relationships": 10,
        "include_entity_rank": True,
        "include_relationship_weight": True,
        "include_community_rank": False,
        "return_candidate_context": False,
        "embedding_vectorstore_key": EntityVectorStoreKey.ID,
        "max_tokens": 12_000,
    }

    # LLM params
    llm_params = {
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }

    # Create minimal context builder (we use precomputed contexts)
    context_builder = LocalSearchMixedContext(
        community_reports=None,
        text_units=None,
        entities=[],
        relationships=None,
        covariates=None,
        entity_text_embeddings=None,
        embedding_vectorstore_key=EntityVectorStoreKey.ID,
        text_embedder=None,
        token_encoder=token_encoder,
    )

    # Create search engine
    search_engine = MyLocalSearch(
        llm=llm,
        context_builder=context_builder,
        token_encoder=token_encoder,
        llm_params=llm_params,
        context_builder_params=local_context_params
    )

    # Load golden QA pairs
    qa_file = f"{ROOT_PATH}/qa-pairs/qa-pairs.json"
    print(f"\nLoading QA pairs from {qa_file}...")
    with open(qa_file, 'r', encoding='utf-8') as f:
        golden = json.load(f)
    print(f"  Loaded {len(golden)} QA pairs")

    # Find context files
    directory_path = f'{ROOT_PATH}/output/'
    pattern = args.context_pattern or 'ragtest-*.json'
    json_files = glob.glob(os.path.join(directory_path, pattern))
    print(f"\nFound {len(json_files)} context files matching '{pattern}'")

    if len(json_files) == 0:
        print("No context files found. Please run context generation first:")
        print(f"  python KET-RAG/indexing_sket/create_context.py {ROOT_PATH} keyword 0.5")
        return

    # Process all context files
    semaphore = asyncio.Semaphore(args.max_files)

    async def limited_process(json_file):
        async with semaphore:
            file_path = os.path.dirname(json_file).rstrip("/")
            file_name = os.path.basename(json_file)
            with open(json_file, 'r', encoding='utf-8') as f:
                context_data = json.load(f)
            await process_qa_pairs(
                search_engine, golden, context_data, file_path, file_name,
                max_concurrent=args.max_concurrent
            )

    tasks = [limited_process(json_file) for json_file in json_files]
    await tqdm_asyncio.gather(*tasks, desc="Processing all files")

    print("\nDone! Answer files saved with 'answer-' prefix.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate answers with local LLM")
    parser.add_argument("root_path", help="Project root directory")

    # LLM options
    parser.add_argument("--use_local_llm", action="store_true",
                        help="Use local HuggingFace model instead of API")
    parser.add_argument("--local_llm_model", default="meta-llama/Meta-Llama-3-8B-Instruct",
                        help="Local LLM model name")
    parser.add_argument("--device", default="cuda:0", help="Device for local LLM")
    parser.add_argument("--load_in_8bit", action="store_true", help="Use 8-bit quantization")
    parser.add_argument("--load_in_4bit", action="store_true", help="Use 4-bit quantization")

    # API options
    parser.add_argument("--api_base", default=None,
                        help="API base URL (e.g., http://localhost:8000/v1)")
    parser.add_argument("--api_key", default=None, help="API key")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model name for API")

    # Generation options
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature")
    parser.add_argument("--max_tokens", type=int, default=2000, help="Max tokens")

    # Processing options
    parser.add_argument("--max_concurrent", type=int, default=10,
                        help="Max concurrent requests per file")
    parser.add_argument("--max_files", type=int, default=4,
                        help="Max concurrent files to process")
    parser.add_argument("--context_pattern", default=None,
                        help="Pattern for context files (default: ragtest-*.json)")

    args = parser.parse_args()
    asyncio.run(main(args))