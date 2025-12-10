import asyncio
import argparse
import copy
import math
import random
import sys
import json
import os
import glob
import re
import time
from util_v1 import MyLocalSearch
from itertools import product
from pathlib import Path

import pandas as pd
import tiktoken
from collections import Counter, defaultdict
from tqdm.asyncio import tqdm_asyncio

from graphrag.query.context_builder.entity_extraction import (
    EntityVectorStoreKey,
)
from graphrag.query.input.loaders.dfs import (
    store_entity_semantic_embeddings,
)
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.llm.oai.embedding import OpenAIEmbedding
from graphrag.query.llm.oai.typing import OpenaiApiType
from graphrag.query.question_gen.local_gen import LocalQuestionGen
from graphrag.query.structured_search.local_search.mixed_context import (
    LocalSearchMixedContext,
)
from graphrag.query.structured_search.local_search.search import LocalSearch
from graphrag.vector_stores.lancedb import LanceDBVectorStore


class LocalLLMWrapper:
    """
    LLM wrapper compatible with GraphRAG's ChatOpenAI interface.
    Supports local HuggingFace models.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        device: str = "cuda:0",
        temperature: float = 0.0,
        max_tokens: int = 2000,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ):
        # Import here to avoid loading if not using local LLM
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        import torch

        self.model_name = model_name
        self.device = device
        self.temperature = temperature
        self.max_tokens = max_tokens

        print(f"  Loading local LLM: {model_name} on {device}")

        # Configure quantization
        quantization_config = None
        if load_in_8bit or load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
                bnb_4bit_compute_dtype=torch.float16 if load_in_4bit else None
            )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map=device if device != "cpu" else "auto",
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            low_cpu_mem_usage=True
        )
        self.torch = torch

    def _messages_to_prompt(self, messages):
        """Convert ChatOpenAI-style messages to a single prompt."""
        prompt_parts = []
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            if role == 'system':
                prompt_parts.append(f"System: {content}")
            elif role == 'user':
                prompt_parts.append(f"{content}")
            elif role == 'assistant':
                prompt_parts.append(f"Assistant: {content}")
        return "\n\n".join(prompt_parts)

    def _generate_sync(self, messages, **kwargs):
        """Synchronous generation."""
        prompt = self._messages_to_prompt(messages)
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        ).to(self.device)

        input_length = inputs['input_ids'].shape[1]

        with self.torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else 1e-7,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = outputs[0][input_length:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def generate(self, messages, streaming=False, callbacks=None, **kwargs):
        """Synchronous generation (compatible with ChatOpenAI)."""
        return self._generate_sync(messages, **kwargs)

    async def agenerate(self, messages, streaming=False, callbacks=None, **kwargs):
        """Async generation - wraps blocking call in thread."""
        return await asyncio.to_thread(self._generate_sync, messages, **kwargs)

    def stream_generate(self, messages, callbacks=None, **kwargs):
        """Synchronous streaming generation."""
        response = self._generate_sync(messages, **kwargs)
        yield response

    async def astream_generate(self, messages, callbacks=None, **kwargs):
        """Async streaming generation."""
        response = await asyncio.to_thread(self._generate_sync, messages, **kwargs)
        yield response


async def process_qa_pairs(search_engine, golden, graphgraph, file_path, file_name, max_concurrent=10):
    answered_pairs = []
    tasks = []
    semaphore = asyncio.Semaphore(max_concurrent)

    async def limited_asearch(qa):
        async with semaphore:
            question = [qa_pair["question"] for qa_pair in golden if qa_pair["id"] == qa["id"]][0]
            time.sleep(0.01 * random.randint(1, 9))
            return await search_engine.asearch_with_context(question, qa["context"])

    for qa in graphgraph:
        tasks.append(limited_asearch(qa))

    results = await tqdm_asyncio.gather(*tasks, desc=f"Processing {file_name}")

    for qa, result in zip(graphgraph, results):
        answered_pairs.append({"id": qa["id"], "answer": result.response})

    with open(f"{file_path}/answer-{file_name}", "w") as fw:
        json.dump(answered_pairs, fw, indent=4)


async def main(args):
    ROOT_PATH = args.root_path.rstrip("/")

    # Initialize LLM
    print("Initializing LLM...")
    if args.use_local_llm:
        print(f"  Using local LLM: {args.local_llm_model}")
        # Auto-adjust concurrency for local LLM
        max_concurrent = 1
        print(f"  Setting max_concurrent=1 for local LLM")
        llm = LocalLLMWrapper(
            model_name=args.local_llm_model,
            device=args.device,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            load_in_8bit=args.load_in_8bit,
            load_in_4bit=args.load_in_4bit,
        )
    else:
        max_concurrent = 10
        api_key = args.api_key or os.environ.get("GRAPHRAG_API_KEY")
        if not api_key:
            print("Error: GRAPHRAG_API_KEY not set. Use --api_key or set environment variable.")
            return

        if args.api_base:
            # Using OpenAI-compatible API (e.g., vLLM server)
            print(f"  Using API: {args.api_base} with model {args.model}")
            import openai
            llm = ChatOpenAI(
                api_key=api_key,
                api_base=args.api_base,
                model=args.model,
                api_type=OpenaiApiType.OpenAI,
                max_retries=20,
            )
        else:
            # Using OpenAI API directly
            print(f"  Using OpenAI API with model {args.model}")
            llm = ChatOpenAI(
                api_key=api_key,
                model=args.model,
                api_type=OpenaiApiType.OpenAI,
                max_retries=20,
            )

    token_encoder = tiktoken.get_encoding("cl100k_base")

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

    llm_params = {
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }

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

    search_engine = MyLocalSearch(
        llm=llm,
        context_builder=context_builder,
        token_encoder=token_encoder,
        llm_params=llm_params,
        context_builder_params=local_context_params
    )

    raw_golden_file = open(f"{ROOT_PATH}/qa-pairs/qa-pairs.json", 'r', encoding='utf-8')
    golden = json.load(raw_golden_file)

    # Find matching files
    directory_path = f'{ROOT_PATH}/output/'
    pattern = args.context_pattern or 'ragtest-*.json'
    json_files = glob.glob(os.path.join(directory_path, pattern))

    print(f"Found {len(json_files)} context files matching '{pattern}'")

    if len(json_files) == 0:
        print("No context files found. Run context generation first.")
        return

    file_semaphore = asyncio.Semaphore(args.max_files)

    async def limited_process(json_file):
        async with file_semaphore:
            with open(json_file, 'r', encoding='utf-8') as f:
                file_path, file_name = os.path.dirname(json_file).rstrip("/"), os.path.basename(json_file)
                graphgraph = json.load(f)
                await process_qa_pairs(search_engine, golden, graphgraph, file_path, file_name, max_concurrent)

    tasks = [limited_process(json_file) for json_file in json_files]
    await tqdm_asyncio.gather(*tasks, desc="Processing all files")

    print("\nDone! Answer files saved with 'answer-' prefix.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate answers for KET-RAG context files")
    parser.add_argument("root_path", help="Project root directory (e.g., ragtest-hotpot)")

    # Local LLM options
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
    parser.add_argument("--max_tokens", type=int, default=2000, help="Max tokens for generation")

    # Processing options
    parser.add_argument("--max_files", type=int, default=4,
                        help="Max concurrent files to process")
    parser.add_argument("--context_pattern", default=None,
                        help="Pattern for context files (default: ragtest-*.json)")

    args = parser.parse_args()
    asyncio.run(main(args))
