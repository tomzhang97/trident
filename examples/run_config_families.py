#!/usr/bin/env python3
"""Example script demonstrating how to use the config families.

This script shows how to run TRIDENT with the named config presets
for cost-quality experiments.

Usage:
    # Run Pareto cheap config
    python examples/run_config_families.py --config pareto_cheap_1500

    # Run Safe-Cover equal-budget config
    python examples/run_config_families.py --config safe_cover_equal_2500

    # Run Self-RAG baseline
    python examples/run_config_families.py --config selfrag_base --baseline selfrag
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trident.config import TridentConfig
from trident.config_families import (
    get_config,
    get_selfrag_config,
    PARETO_CONFIGS,
    SAFE_COVER_CONFIGS,
    SELFRAG_CONFIGS,
)


def create_config_from_preset(preset_name: str, baseline: str = "trident") -> TridentConfig:
    """Create a full TridentConfig from a preset name.

    Args:
        preset_name: Name of the config preset (e.g., "pareto_cheap_1500")
        baseline: Type of baseline ("trident" or "selfrag")

    Returns:
        Complete TridentConfig object
    """
    if baseline == "selfrag":
        # Self-RAG configs
        baseline_config = get_selfrag_config(preset_name)
        return TridentConfig(
            mode="pareto",  # Not used for Self-RAG
            baselines=baseline_config,
        )
    else:
        # TRIDENT configs
        preset_config = get_config(preset_name)

        # Determine mode from config name
        if preset_name.startswith("pareto"):
            mode = "pareto"
            return TridentConfig(
                mode=mode,
                pareto=preset_config,
            )
        elif preset_name.startswith("safe_cover"):
            mode = "safe_cover"
            return TridentConfig(
                mode=mode,
                safe_cover=preset_config,
            )
        else:
            raise ValueError(f"Unknown config type: {preset_name}")


def main():
    parser = argparse.ArgumentParser(description="Run TRIDENT with config families")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Config preset name (e.g., 'pareto_cheap_1500', 'safe_cover_equal_2500')",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="trident",
        choices=["trident", "selfrag"],
        help="Baseline type (default: trident)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file to save config (optional)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available config presets",
    )

    args = parser.parse_args()

    if args.list:
        print("Available TRIDENT configs:")
        for name in PARETO_CONFIGS:
            print(f"  - {name}")
        for name in SAFE_COVER_CONFIGS:
            print(f"  - {name}")
        print("\nAvailable Self-RAG configs:")
        for name in SELFRAG_CONFIGS:
            print(f"  - {name}")
        return

    # Create config from preset
    config = create_config_from_preset(args.config, args.baseline)

    # Print config summary
    print(f"\nCreated config for: {args.config}")
    print(f"Mode: {config.mode}")
    if args.baseline == "trident":
        if config.mode == "pareto":
            print(f"  Budget: {config.pareto.budget}")
            print(f"  Max evidence tokens: {config.pareto.max_evidence_tokens}")
            print(f"  Max units: {config.pareto.max_units}")
        elif config.mode == "safe_cover":
            print(f"  Per-facet alpha: {config.safe_cover.per_facet_alpha}")
            print(f"  Max evidence tokens: {config.safe_cover.max_evidence_tokens}")
            print(f"  Max units: {config.safe_cover.max_units}")
            print(f"  Abstain on infeasible: {config.safe_cover.abstain_on_infeasible}")
    else:
        print(f"  Self-RAG k: {config.baselines.selfrag_k}")
        print(f"  Use critic: {config.baselines.selfrag_use_critic}")
        print(f"  Allow oracle context: {config.baselines.selfrag_allow_oracle_context}")

    # Optionally save to file
    if args.output:
        output_path = Path(args.output)
        config.save(str(output_path))
        print(f"\nConfig saved to: {output_path}")


if __name__ == "__main__":
    main()
