#!/usr/bin/env python3
"""Thin wrapper to run the root-level eval_full_baselines.py from experiments/."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Ensure the project root is on the path so imports behave the same as root execution.
sys.path.insert(0, str(ROOT))

from eval_full_baselines import main

if __name__ == "__main__":
    main()
