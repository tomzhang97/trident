#!/usr/bin/env python3
"""
Standalone GraphRAG indexing script for KET-RAG.

This bypasses the typer CLI version incompatibility by calling the index API directly.

Usage:
    python scripts/run_ketrag_index.py ragtest-hotpot/
    python scripts/run_ketrag_index.py ragtest-hotpot/ --config ragtest-hotpot/settings.yaml
"""

import asyncio
import sys
import time
import argparse
import warnings
from pathlib import Path

# Ignore numba warnings
warnings.filterwarnings("ignore", message=".*NumbaDeprecationWarning.*")

# Add KET-RAG to path FIRST (before any other imports that might load graphrag)
SCRIPT_DIR = Path(__file__).parent.parent
KETRAG_PATH = SCRIPT_DIR / "KET-RAG"

# Remove any existing graphrag from path and add KET-RAG's version
sys.path = [p for p in sys.path if 'graphrag' not in p.lower() or 'KET-RAG' in p]
sys.path.insert(0, str(KETRAG_PATH))

import graphrag.api as api
from graphrag.config import (
    CacheType,
    enable_logging_with_config,
    load_config,
    resolve_paths,
)
from graphrag.index.emit.types import TableEmitterType
from graphrag.logging import PrintProgressReporter


async def run_index(
    root_dir: Path,
    config_path: Path | None = None,
    verbose: bool = False,
    resume: str | None = None,
    memprofile: bool = False,
    cache: bool = True,
    output_dir: Path | None = None,
):
    """Run the GraphRAG indexing pipeline."""
    root = root_dir.resolve()
    run_id = resume or time.strftime("%Y%m%d-%H%M%S")

    print(f"GraphRAG Indexing (KET-RAG)")
    print(f"  Root: {root}")
    print(f"  Config: {config_path or 'default'}")
    print(f"  Run ID: {run_id}")
    print()

    # Load config
    config = load_config(root, config_path)

    # Override output dir if specified
    if output_dir:
        config.storage.base_dir = str(output_dir)
        config.reporting.base_dir = str(output_dir)

    # Resolve paths with run_id
    resolve_paths(config, run_id)

    # Disable cache if requested
    if not cache:
        config.cache.type = CacheType.none

    # Enable logging
    enabled_logging, log_path = enable_logging_with_config(config, verbose)
    if enabled_logging:
        print(f"  Logging to: {log_path}")

    # Create reporter
    reporter = PrintProgressReporter("")

    # Run the pipeline
    emit = [TableEmitterType.Parquet]

    print("\nStarting indexing pipeline...")
    outputs = await api.build_index(
        config=config,
        run_id=run_id,
        is_resume_run=bool(resume),
        memory_profile=memprofile,
        progress_reporter=reporter,
        emit=emit,
    )

    # Check for errors
    encountered_errors = any(
        output.errors and len(output.errors) > 0 for output in outputs
    )

    print(f"\nIndexing complete! {len(outputs)} workflows executed.")

    if encountered_errors:
        print("\nErrors occurred:")
        for output in outputs:
            if output.errors:
                print(f"  {output.workflow}:")
                for error in output.errors:
                    print(f"    - {error}")
        return 1
    else:
        print("All workflows completed successfully.")
        return 0


def main():
    parser = argparse.ArgumentParser(description="Run KET-RAG GraphRAG indexing")
    parser.add_argument("root", help="Project root directory")
    parser.add_argument("--config", "-c", default=None, help="Path to settings.yaml")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    parser.add_argument("--resume", default=None, help="Resume a previous run")
    parser.add_argument("--memprofile", action="store_true", help="Memory profiling")
    parser.add_argument("--no-cache", action="store_true", help="Disable LLM cache")
    parser.add_argument("--output", "-o", default=None, help="Output directory override")

    args = parser.parse_args()

    root_dir = Path(args.root).resolve()

    # Check if root exists
    if not root_dir.exists():
        print(f"Error: Root directory does not exist: {root_dir}")
        sys.exit(1)

    # Check for input files
    input_dir = root_dir / "input"
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        print("Create it and add your text documents (.txt files)")
        sys.exit(1)

    txt_files = list(input_dir.glob("*.txt"))
    if not txt_files:
        print(f"Error: No .txt files found in {input_dir}")
        print("Add your text documents to this directory")
        sys.exit(1)

    print(f"Found {len(txt_files)} input files")

    # Check for settings.yaml
    config_path = Path(args.config) if args.config else root_dir / "settings.yaml"
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        print("Run setup script first or create settings.yaml")
        sys.exit(1)

    exit_code = asyncio.run(run_index(
        root_dir=root_dir,
        config_path=config_path if args.config else None,
        verbose=args.verbose,
        resume=args.resume,
        memprofile=args.memprofile,
        cache=not args.no_cache,
        output_dir=Path(args.output) if args.output else None,
    ))

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
