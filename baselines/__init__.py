"""Baseline systems for comparison with TRIDENT."""

# Simplified baseline implementations (lightweight, no external dependencies)
from .self_rag_system import SelfRAGSystem
from .graphrag_system import GraphRAGSystem, SimpleGraphIndex
from .ketrag_system import KETRAGSystem
from .trident_wrapper import TridentSystemWrapper

# Full baseline interface (no heavy dependencies)
from .full_baseline_interface import BaselineSystem, BaselineResponse, TokenTracker, LatencyTracker

# Note: Full baseline adapters (FullGraphRAGAdapter, FullSelfRAGAdapter,
# FullKETRAGAdapter, FullVanillaRAGAdapter, FullHippoRAGAdapter) are NOT
# imported here to avoid loading heavy dependencies at module import time.
# Import them directly where needed:
#   from baselines.full_selfrag_adapter import FullSelfRAGAdapter
#   from baselines.full_ketrag_adapter import FullKETRAGAdapter
#   etc.

__all__ = [
    # Simplified baselines
    'SelfRAGSystem',
    'GraphRAGSystem',
    'SimpleGraphIndex',
    'KETRAGSystem',
    'TridentSystemWrapper',
    # Full baseline interface
    'BaselineSystem',
    'BaselineResponse',
    'TokenTracker',
    'LatencyTracker',
    # Note: Full adapters are not in __all__ - import them directly
]
