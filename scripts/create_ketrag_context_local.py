#!/usr/bin/env python3
"""
KET-RAG context creation with local LLM and embeddings.

This script creates context files for KET-RAG using local models instead of OpenAI.
It requires pre-built GraphRAG indexes (parquet files).

Usage:
    python scripts/create_ketrag_context_local.py \
        --root_path /path/to/ketrag/ragtest-hotpot \
        --strategy keyword \
        --budget 0.5 \
        --embedding_model sentence-transformers/all-MiniLM-L6-v2 \
        --device cuda:0
"""

import asyncio
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd
import tiktoken
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm

# Add paths
SCRIPT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "KET-RAG"))
sys.path.insert(0, str(SCRIPT_DIR / "KET-RAG" / "indexing_sket"))

from graphrag.query.llm.text_utils import num_tokens
from graphrag.query.input.loaders.dfs import read_text_units


class LocalEmbedding:
    """Local embedding model using sentence-transformers."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cuda:0",
        normalize: bool = True
    ):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)
        self.normalize = normalize
        print(f"  [LocalEmbedding] Loaded {model_name} on {device}")

    def embed(self, text: str, **kwargs) -> list[float]:
        """Embed a single text."""
        embedding = self.model.encode(
            text,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True
        )
        return embedding.tolist()

    def embed_batch(self, texts: list[str], **kwargs) -> list[list[float]]:
        """Embed a batch of texts."""
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        return embeddings.tolist()


def normalize_embeddings(vectors):
    if len(vectors.shape) == 1:
        return vectors / np.linalg.norm(vectors)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms


def find_chunks_keyword(
    query: str,
    embedder: LocalEmbedding,
    encoder: tiktoken.Encoding,
    index: faiss.IndexFlatIP,
    words: list,
    word_chunk_data: dict,
    candidate_units_dict: dict,
    chunks_size: int,
) -> str:
    """Find relevant chunks using keyword-based retrieval."""
    if chunks_size == 0:
        return "\n\n-----Text source that may be relevant-----\n\nN/A\n"

    query_embedding = embedder.embed(query)
    query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)

    D, Idx = index.search(query_embedding, 1000)
    word_rank = [(words[i], float(D[0, idx])) for idx, i in enumerate(Idx[0])]

    selected_chunks = []
    tokens_original = 0
    visited_chunks = set()
    is_full = False

    for word, _ in word_rank:
        if is_full:
            break
        if word not in word_chunk_data:
            continue
        chunk_ids, chunk_embs, chunk_token_counts = word_chunk_data[word]

        # Compute similarities via dot product for normalized embeddings
        sims = (query_embedding @ chunk_embs.T).flatten()

        # Sort chunks by similarity (descending)
        idx_sorted = np.argsort(-sims)

        for idx_c in idx_sorted:
            chunk_id = chunk_ids[idx_c]
            sim = sims[idx_c]
            chunk_tokens = chunk_token_counts[idx_c]

            if tokens_original + chunk_tokens > chunks_size * 2:
                is_full = True
                break
            if chunk_id in visited_chunks:
                continue
            visited_chunks.add(chunk_id)
            tokens_original += chunk_tokens
            selected_chunks.append((chunk_id, sim))

    selected_chunks.sort(key=lambda x: x[1], reverse=True)

    text = "\n\n-----Text source that may be relevant-----\nid|text\n"
    text_tokens = num_tokens(text, encoder)

    for idx, (element_id, sim) in enumerate(selected_chunks):
        new_text_segment = f"chunk_{idx+1}|" + candidate_units_dict[element_id].text + "\n"
        newly_added_tokens = num_tokens(new_text_segment, encoder)
        if text_tokens + newly_added_tokens > chunks_size:
            break
        text += new_text_segment
        text_tokens += newly_added_tokens

    return text


def find_chunks_text(
    query: str,
    embedder: LocalEmbedding,
    encoder: tiktoken.Encoding,
    index: faiss.IndexFlatIP,
    ids: list,
    candidate_units_dict: dict,
    chunks_size: int = 0
) -> str:
    """Find relevant chunks using text similarity."""
    if chunks_size == 0:
        return "\n\n-----Text source that may be relevant-----\n\nN/A\n"

    query_embedding = embedder.embed(query)
    query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)

    D, Idx = index.search(query_embedding, 1000)
    text_ranks = [(ids[i], float(D[0, idx])) for idx, i in enumerate(Idx[0])]

    text = "\n\n-----Text source that may be relevant-----\nid|text\n"
    text_tokens = num_tokens(text, encoder)

    for idx, (element_id, rank) in enumerate(text_ranks):
        element_content = candidate_units_dict[element_id].text
        newly_added_tokens = num_tokens(f"chunk_{idx+1}|" + element_content + "\n", encoder)
        if text_tokens + newly_added_tokens > chunks_size:
            break
        text = text + f"chunk_{idx+1}|" + element_content + "\n"
        text_tokens += newly_added_tokens
    return text


def generate_context(
    strategy: str,
    budget: float,
    question: str,
    text_context_params: dict,
    graph_context_builder=None,
) -> str:
    """Generate context for a question."""
    if strategy == "keyword":
        text_content = find_chunks_keyword(question, **text_context_params)
    elif strategy == "text":
        text_content = find_chunks_text(question, **text_context_params)
    else:
        text_content = "\n\n-----Text source that may be relevant-----\n\nN/A\n"

    # Add graph context if available and budget > 0
    if budget > 0 and graph_context_builder is not None:
        context_params = {
            "text_unit_prop": 0.5,
            "community_prop": 0.0,
            "conversation_history_max_turns": 10,
            "conversation_history_user_turns_only": True,
            "top_k_mapped_entities": 10,
            "top_k_relationships": 10,
            "include_entity_rank": True,
            "include_relationship_weight": True,
            "include_community_rank": False,
            "return_candidate_context": False,
            "max_tokens": 12000 - num_tokens(text_content, text_context_params['encoder'])
        }
        context_result = graph_context_builder.build_context(question, **context_params)
        graph_context = context_result.context_chunks
    else:
        graph_context = ""

    return graph_context + text_content


async def process_pair(pair, strategy, budget, text_context_params, graph_context_builder, semaphore):
    """Process a single QA pair."""
    async with semaphore:
        question = pair['question']
        context = await asyncio.to_thread(
            generate_context,
            strategy,
            budget,
            question,
            text_context_params,
            graph_context_builder,
        )
        return {"id": pair["id"], "context": context}


async def process_all_pairs(valid_pairs, strategy, budget, text_context_params, graph_context_builder, max_concurrent):
    """Process all QA pairs."""
    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = [
        process_pair(pair, strategy, budget, text_context_params, graph_context_builder, semaphore)
        for pair in valid_pairs
    ]
    return await atqdm.gather(*tasks, desc="Processing", unit="pair")


def main():
    parser = argparse.ArgumentParser(description="Create KET-RAG context with local embeddings")
    parser.add_argument("--root_path", required=True, help="Path to KET-RAG data root (e.g., ragtest-hotpot)")
    parser.add_argument("--strategy", choices=["keyword", "text", "none"], default="keyword",
                        help="Context building strategy")
    parser.add_argument("--budget", type=float, default=0.5,
                        help="Budget for graph context (0.0-1.0)")
    parser.add_argument("--embedding_model", default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Local embedding model")
    parser.add_argument("--device", default="cuda:0", help="Device for embeddings")
    parser.add_argument("--max_concurrent", type=int, default=10, help="Max concurrent tasks")
    parser.add_argument("--qa_file", default=None, help="Path to QA pairs JSON (default: <root>/qa-pairs/qa-pairs.json)")
    parser.add_argument("--output", default=None, help="Output path (default: auto-generated)")
    args = parser.parse_args()

    ROOT_PATH = args.root_path.rstrip("/")
    OUTPUT_DIR = f"{ROOT_PATH}/output"

    # Constants
    TEXT_UNIT_TABLE = "create_final_text_units"

    print(f"Loading data from {ROOT_PATH}...")

    # Initialize token encoder
    token_encoder = tiktoken.get_encoding("cl100k_base")

    # Initialize local embedder
    print("Loading embedding model...")
    embedder = LocalEmbedding(
        model_name=args.embedding_model,
        device=args.device,
    )

    # Calculate context sizes
    length_text_context = int(12_000 * (1.0 - args.budget))

    # Load split text units and keyword index
    print("Loading text units and keyword index...")
    split_text_units_df = pd.read_parquet(f"{OUTPUT_DIR}/split_text_units.parquet")
    keyword_df = pd.read_parquet(f"{OUTPUT_DIR}/keyword_index.parquet")

    split_text_units = read_text_units(
        split_text_units_df,
        text_col='chunk',
        short_id_col=None,
        covariates_col=None,
        entities_col=None,
        relationships_col=None,
        embedding_col='text_embedding'
    )

    # Build index
    candidate_units_dict = {unit.id: unit for unit in split_text_units}

    # Check embedding dimensions and re-embed if needed
    sample_embedding = keyword_df.iloc[0]['embedding'] if len(keyword_df) > 0 else None
    index_dim = len(sample_embedding) if sample_embedding is not None else 0
    local_dim = embedder.model.get_sentence_embedding_dimension()

    print(f"  Index embedding dim: {index_dim}, Local model dim: {local_dim}")

    need_reembed = (index_dim != local_dim)
    if need_reembed:
        print(f"  ⚠️  Dimension mismatch! Re-embedding with local model...")

    if args.strategy == "keyword":
        word_chunks = {row['word']: row['chunk_ids'] for _, row in keyword_df.iterrows()}
        words = list(word_chunks.keys())

        if need_reembed:
            # Re-embed keywords with local model
            print(f"  Re-embedding {len(words)} keywords...")
            word_vecs = np.array(embedder.embed_batch(words), dtype=np.float32)
        else:
            word_embeddings = {row['word']: row['embedding'] for _, row in keyword_df.iterrows()}
            word_vecs = np.array([word_embeddings[w] for w in words], dtype=np.float32)

        index = faiss.IndexFlatIP(word_vecs.shape[1])
        index.add(word_vecs)

        token_counts = {unit.id: num_tokens(unit.text, token_encoder) for unit in split_text_units}

        # Pre-compute chunk embeddings
        if need_reembed:
            # Re-embed all chunks with local model
            all_chunk_ids = list(candidate_units_dict.keys())
            all_chunk_texts = [candidate_units_dict[cid].text for cid in all_chunk_ids]
            print(f"  Re-embedding {len(all_chunk_texts)} chunks...")
            all_chunk_embs = np.array(embedder.embed_batch(all_chunk_texts), dtype=np.float32)
            chunk_embedding_map = {cid: all_chunk_embs[i] for i, cid in enumerate(all_chunk_ids)}
        else:
            chunk_embedding_map = {unit.id: unit.text_embedding for unit in split_text_units}

        # Pre-compute chunk data for each word
        word_chunk_data = {}
        for w in words:
            chunk_ids = list(word_chunks.get(w, []))
            if not chunk_ids:
                continue
            chunk_embs = np.stack([chunk_embedding_map[cid] for cid in chunk_ids]).astype(np.float32)
            chunk_token_counts = [token_counts[cid] for cid in chunk_ids]
            word_chunk_data[w] = (chunk_ids, chunk_embs, chunk_token_counts)

        text_context_params = {
            'embedder': embedder,
            'encoder': token_encoder,
            'index': index,
            'words': words,
            'word_chunk_data': word_chunk_data,
            'candidate_units_dict': candidate_units_dict,
            'chunks_size': length_text_context
        }
    elif args.strategy == "text":
        chunk_ids = list(candidate_units_dict.keys())

        if need_reembed:
            # Re-embed all chunks with local model
            all_chunk_texts = [candidate_units_dict[cid].text for cid in chunk_ids]
            print(f"  Re-embedding {len(all_chunk_texts)} chunks...")
            chunk_vecs = np.array(embedder.embed_batch(all_chunk_texts), dtype=np.float32)
        else:
            chunk_vecs = np.array([unit.text_embedding for unit in candidate_units_dict.values()], dtype=np.float32)

        index = faiss.IndexFlatIP(chunk_vecs.shape[1])
        index.add(chunk_vecs)

        text_context_params = {
            'embedder': embedder,
            'encoder': token_encoder,
            'index': index,
            'ids': chunk_ids,
            'candidate_units_dict': candidate_units_dict,
            'chunks_size': length_text_context
        }
    else:
        text_context_params = {
            'embedder': embedder,
            'encoder': token_encoder,
            'chunks_size': 0
        }

    # Load QA pairs
    qa_file_path = args.qa_file or f"{ROOT_PATH}/qa-pairs/qa-pairs.json"
    print(f"Loading QA pairs from {qa_file_path}...")

    with open(qa_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Filter valid pairs
    valid_pairs = [entry for entry in data if
                   isinstance(entry.get('question'), str) and isinstance(entry.get('answer'), str)][:500]
    print(f"Processing {len(valid_pairs)} QA pairs...")

    # Process all pairs
    matched_pairs = asyncio.run(
        process_all_pairs(
            valid_pairs,
            args.strategy,
            args.budget,
            text_context_params,
            None,  # graph_context_builder - set to None for now
            args.max_concurrent
        )
    )

    # Save output
    folder_name = os.path.basename(os.path.normpath(ROOT_PATH))
    output_path = args.output or f"{OUTPUT_DIR}/{folder_name}-{args.strategy}-{args.budget}.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(matched_pairs, f, indent=4)

    print(f"Stored contexts in {output_path}")


if __name__ == "__main__":
    main()
