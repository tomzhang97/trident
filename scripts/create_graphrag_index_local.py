#!/usr/bin/env python3
"""
GraphRAG indexing with local LLM (via vLLM or OpenAI-compatible server).

This script creates GraphRAG indexes using a local LLM server.

Prerequisites:
1. Start a vLLM server:
   python -m vllm.entrypoints.openai.api_server \
       --model meta-llama/Meta-Llama-3-8B-Instruct \
       --port 8000

2. For embeddings, either:
   a) Start a local embedding server, or
   b) Use HuggingFace models directly with this script

Usage:
    # Basic usage with vLLM server
    python scripts/create_graphrag_index_local.py \
        --input_dir /path/to/documents \
        --output_dir /path/to/output \
        --llm_api_base http://localhost:8000/v1 \
        --llm_model meta-llama/Meta-Llama-3-8B-Instruct

    # With local embeddings
    python scripts/create_graphrag_index_local.py \
        --input_dir /path/to/documents \
        --output_dir /path/to/output \
        --llm_api_base http://localhost:8000/v1 \
        --llm_model meta-llama/Meta-Llama-3-8B-Instruct \
        --use_local_embeddings \
        --embedding_model sentence-transformers/all-MiniLM-L6-v2
"""

import asyncio
import os
import sys
import json
import yaml
import argparse
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "graphrag"))


def generate_settings_yaml(
    input_dir: str,
    output_dir: str,
    llm_api_base: str,
    llm_model: str,
    llm_api_key: str = "EMPTY",
    embedding_api_base: str = None,
    embedding_model: str = "text-embedding-3-small",
    embedding_api_key: str = None,
    chunk_size: int = 1200,
    chunk_overlap: int = 100,
    max_tokens: int = 4000,
    temperature: float = 0.0,
) -> dict:
    """Generate GraphRAG settings for local LLM."""

    settings = {
        "input": {
            "type": "file",
            "file_type": "text",
            "base_dir": input_dir,
            "file_encoding": "utf-8",
            "file_pattern": ".*\\.txt$",
        },
        "storage": {
            "type": "file",
            "base_dir": output_dir,
        },
        "cache": {
            "type": "file",
            "base_dir": f"{output_dir}/cache",
        },
        "reporting": {
            "type": "file",
            "base_dir": f"{output_dir}/reports",
        },
        "llm": {
            "type": "openai_chat",
            "model": llm_model,
            "api_key": llm_api_key,
            "api_base": llm_api_base,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "request_timeout": 180.0,
            "max_retries": 10,
        },
        "embeddings": {
            "type": "openai_embedding",
            "model": embedding_model,
            "api_key": embedding_api_key or llm_api_key,
            "api_base": embedding_api_base or llm_api_base,
        },
        "chunks": {
            "size": chunk_size,
            "overlap": chunk_overlap,
        },
        "claim_extraction": {
            "enabled": False,
        },
        "community_reports": {
            "max_length": 2000,
            "max_input_length": 8000,
        },
        "entity_extraction": {
            "max_gleanings": 1,
        },
        "snapshots": {
            "graphml": True,
            "raw_entities": True,
            "top_level_nodes": True,
        },
        "cluster_graph": {
            "max_cluster_size": 10,
        },
        "umap": {
            "enabled": False,
        },
    }

    return settings


def generate_settings_yaml_local_embeddings(
    input_dir: str,
    output_dir: str,
    llm_api_base: str,
    llm_model: str,
    llm_api_key: str = "EMPTY",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    chunk_size: int = 1200,
    chunk_overlap: int = 100,
    max_tokens: int = 4000,
    temperature: float = 0.0,
) -> dict:
    """Generate GraphRAG settings with local HuggingFace embeddings."""

    settings = {
        "input": {
            "type": "file",
            "file_type": "text",
            "base_dir": input_dir,
            "file_encoding": "utf-8",
            "file_pattern": ".*\\.txt$",
        },
        "storage": {
            "type": "file",
            "base_dir": output_dir,
        },
        "cache": {
            "type": "file",
            "base_dir": f"{output_dir}/cache",
        },
        "reporting": {
            "type": "file",
            "base_dir": f"{output_dir}/reports",
        },
        "llm": {
            "type": "openai_chat",
            "model": llm_model,
            "api_key": llm_api_key,
            "api_base": llm_api_base,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "request_timeout": 180.0,
            "max_retries": 10,
        },
        # Use sentence-transformers for embeddings
        "embeddings": {
            "type": "sentence_transformers",
            "model": embedding_model,
        },
        "chunks": {
            "size": chunk_size,
            "overlap": chunk_overlap,
        },
        "claim_extraction": {
            "enabled": False,
        },
        "community_reports": {
            "max_length": 2000,
            "max_input_length": 8000,
        },
        "entity_extraction": {
            "max_gleanings": 1,
        },
        "snapshots": {
            "graphml": True,
            "raw_entities": True,
            "top_level_nodes": True,
        },
        "cluster_graph": {
            "max_cluster_size": 10,
        },
        "umap": {
            "enabled": False,
        },
    }

    return settings


async def run_indexing(settings_path: str, root_dir: str):
    """Run GraphRAG indexing pipeline."""
    from graphrag.config.load_config import load_config
    from graphrag.api.index import build_index
    from graphrag.config.enums import IndexingMethod

    print(f"Loading config from {settings_path}...")
    config = load_config(root_dir, settings_path)

    print("Starting indexing pipeline...")
    results = await build_index(
        config=config,
        method=IndexingMethod.Standard,
        verbose=True,
    )

    print(f"\nIndexing complete! {len(results)} workflows executed.")
    for result in results:
        status = "SUCCESS" if not result.errors else "FAILED"
        print(f"  - {result.workflow}: {status}")
        if result.errors:
            for error in result.errors:
                print(f"    Error: {error}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Create GraphRAG index with local LLM")
    parser.add_argument("--input_dir", required=True, help="Directory containing input documents")
    parser.add_argument("--output_dir", required=True, help="Output directory for index")
    parser.add_argument("--llm_api_base", required=True, help="LLM API base URL (e.g., http://localhost:8000/v1)")
    parser.add_argument("--llm_model", required=True, help="LLM model name")
    parser.add_argument("--llm_api_key", default="EMPTY", help="LLM API key (default: EMPTY for local)")

    # Embedding options
    parser.add_argument("--use_local_embeddings", action="store_true",
                        help="Use local HuggingFace embeddings instead of API")
    parser.add_argument("--embedding_api_base", default=None, help="Embedding API base URL")
    parser.add_argument("--embedding_model", default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Embedding model name")
    parser.add_argument("--embedding_api_key", default=None, help="Embedding API key")

    # Index parameters
    parser.add_argument("--chunk_size", type=int, default=1200, help="Chunk size")
    parser.add_argument("--chunk_overlap", type=int, default=100, help="Chunk overlap")
    parser.add_argument("--max_tokens", type=int, default=4000, help="Max tokens for LLM")
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM temperature")

    # Control
    parser.add_argument("--settings_only", action="store_true",
                        help="Only generate settings.yaml, don't run indexing")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate settings
    if args.use_local_embeddings:
        settings = generate_settings_yaml_local_embeddings(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            llm_api_base=args.llm_api_base,
            llm_model=args.llm_model,
            llm_api_key=args.llm_api_key,
            embedding_model=args.embedding_model,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
    else:
        settings = generate_settings_yaml(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            llm_api_base=args.llm_api_base,
            llm_model=args.llm_model,
            llm_api_key=args.llm_api_key,
            embedding_api_base=args.embedding_api_base,
            embedding_model=args.embedding_model,
            embedding_api_key=args.embedding_api_key,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )

    # Save settings
    settings_path = os.path.join(args.output_dir, "settings.yaml")
    with open(settings_path, "w") as f:
        yaml.dump(settings, f, default_flow_style=False)
    print(f"Settings saved to {settings_path}")

    if args.settings_only:
        print("\nSettings generated. Run indexing with:")
        print(f"  cd {args.output_dir}")
        print(f"  graphrag index --root .")
        return

    # Run indexing
    try:
        asyncio.run(run_indexing(settings_path, args.output_dir))
    except Exception as e:
        print(f"\nError during indexing: {e}")
        print("\nYou can manually run indexing with:")
        print(f"  cd {args.output_dir}")
        print(f"  graphrag index --root .")
        raise


if __name__ == "__main__":
    main()