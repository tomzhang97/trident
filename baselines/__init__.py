"""Baseline systems for comparison with TRIDENT."""

# Simplified baseline implementations (lightweight, no external dependencies)
from .self_rag_system import SelfRAGSystem
from .graphrag_system import GraphRAGSystem, SimpleGraphIndex
from .ketrag_system import KETRAGSystem
from .trident_wrapper import TridentSystemWrapper

# Full baseline implementations (using official repositories)
from .full_baseline_interface import BaselineSystem, BaselineResponse, TokenTracker, LatencyTracker
from .full_graphrag_adapter import FullGraphRAGAdapter
from .full_selfrag_adapter import FullSelfRAGAdapter
from .full_ketrag_adapter import FullKETRAGAdapter

__all__ = [
    # Simplified baselines
    'SelfRAGSystem',
    'GraphRAGSystem',
    'SimpleGraphIndex',
    'KETRAGSystem',
    'TridentSystemWrapper',
    # Full baselines
    'BaselineSystem',
    'BaselineResponse',
    'TokenTracker',
    'LatencyTracker',
    'FullGraphRAGAdapter',
    'FullSelfRAGAdapter',
    'FullKETRAGAdapter',
]
