#!/usr/bin/env python3
"""
Standalone prompt-tune script for KET-RAG.

This bypasses the typer CLI issue in the older KET-RAG graphrag version.

Usage:
    python scripts/prompt_tune_ketrag.py ragtest-hotpot/
"""

import asyncio
import sys
from pathlib import Path

# Add KET-RAG to path
SCRIPT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SCRIPT_DIR / "external_baselines" / "KET-RAG"))

import graphrag.api as api
from graphrag.config import load_config
from graphrag.logging import PrintProgressReporter
from graphrag.prompt_tune.generator.community_report_summarization import (
    COMMUNITY_SUMMARIZATION_FILENAME,
)
from graphrag.prompt_tune.generator.entity_extraction_prompt import (
    ENTITY_EXTRACTION_FILENAME,
)
from graphrag.prompt_tune.generator.entity_summarization_prompt import (
    ENTITY_SUMMARIZATION_FILENAME,
)


async def prompt_tune(
    root: Path,
    config: Path | None = None,
    domain: str | None = None,
    selection_method: str = "random",
    limit: int = 15,
    max_tokens: int = 3000,
    chunk_size: int = 300,
    language: str | None = None,
    discover_entity_types: bool = True,
    output: Path = None,
    n_subset_max: int = 300,
    k: int = 15,
    min_examples_required: int = 2,
):
    """Prompt tune the model."""
    reporter = PrintProgressReporter("")
    root_path = Path(root).resolve()

    # Load config
    config_path = config if config else root_path / "settings.yaml"
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        print("Run 'python -m graphrag init --root <path>' first")
        return

    print(f"Loading config from {config_path}")
    graph_config = load_config(root_path, config_path if config else None)

    # Map selection method string to enum
    selection_method_enum = api.DocSelectionType(selection_method)

    print(f"Generating prompts with selection_method={selection_method}, discover_entity_types={discover_entity_types}")
    prompts = await api.generate_indexing_prompts(
        config=graph_config,
        root=str(root_path),
        chunk_size=chunk_size,
        limit=limit,
        selection_method=selection_method_enum,
        domain=domain,
        language=language,
        max_tokens=max_tokens,
        discover_entity_types=discover_entity_types,
        min_examples_required=min_examples_required,
        n_subset_max=n_subset_max,
        k=k,
    )

    output_path = (output or root_path / "prompts").resolve()
    reporter.info(f"Writing prompts to {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)

    entity_extraction_prompt_path = output_path / ENTITY_EXTRACTION_FILENAME
    entity_summarization_prompt_path = output_path / ENTITY_SUMMARIZATION_FILENAME
    community_summarization_prompt_path = output_path / COMMUNITY_SUMMARIZATION_FILENAME

    # Write files to output path
    with entity_extraction_prompt_path.open("wb") as file:
        file.write(prompts[0].encode(encoding="utf-8", errors="strict"))
    print(f"  Wrote {entity_extraction_prompt_path}")

    with entity_summarization_prompt_path.open("wb") as file:
        file.write(prompts[1].encode(encoding="utf-8", errors="strict"))
    print(f"  Wrote {entity_summarization_prompt_path}")

    with community_summarization_prompt_path.open("wb") as file:
        file.write(prompts[2].encode(encoding="utf-8", errors="strict"))
    print(f"  Wrote {community_summarization_prompt_path}")

    print("\nPrompt tuning complete!")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Tune prompts for KET-RAG")
    parser.add_argument("root", help="Project root directory")
    parser.add_argument("--config", default=None, help="Path to settings.yaml")
    parser.add_argument("--domain", default=None,
                        help="Domain of your data (e.g., 'multi-hop question answering')")
    parser.add_argument("--selection-method", default="random",
                        choices=["random", "top", "auto"],
                        help="Document selection method")
    parser.add_argument("--limit", type=int, default=15,
                        help="Number of documents to use")
    parser.add_argument("--max-tokens", type=int, default=5000,
                        help="Max tokens for prompts")
    parser.add_argument("--chunk-size", type=int, default=300,
                        help="Chunk size for text")
    parser.add_argument("--language", default=None,
                        help="Primary language")
    parser.add_argument("--no-discover-entity-types", action="store_true",
                        help="Don't discover entity types (use defaults)")
    parser.add_argument("--output", default=None,
                        help="Output directory for prompts")

    args = parser.parse_args()

    asyncio.run(prompt_tune(
        root=Path(args.root),
        config=Path(args.config) if args.config else None,
        domain=args.domain,
        selection_method=args.selection_method,
        limit=args.limit,
        max_tokens=args.max_tokens,
        chunk_size=args.chunk_size,
        language=args.language,
        discover_entity_types=not args.no_discover_entity_types,
        output=Path(args.output) if args.output else None,
    ))


if __name__ == "__main__":
    main()