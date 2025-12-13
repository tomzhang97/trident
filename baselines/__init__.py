"""Baseline systems for comparison with TRIDENT."""

# Full baseline interface (no heavy dependencies)
from .full_baseline_interface import BaselineSystem, BaselineResponse, TokenTracker, LatencyTracker

# Simplified baseline implementations (lightweight, may have dependencies)
# These are imported conditionally to avoid breaking imports when dependencies are missing
try:
    from .self_rag_system import SelfRAGSystem
    from .graphrag_system import GraphRAGSystem, SimpleGraphIndex
    from .ketrag_system import KETRAGSystem
    from .trident_wrapper import TridentSystemWrapper
    SIMPLIFIED_BASELINES_AVAILABLE = True
except ImportError as e:
    # If dependencies are missing, simplified baselines won't be available
    # but full adapters can still be imported
    SIMPLIFIED_BASELINES_AVAILABLE = False
    _import_error = str(e)

# Note: Full baseline adapters (FullGraphRAGAdapter, FullSelfRAGAdapter,
# FullKETRAGAdapter, FullVanillaRAGAdapter, FullHippoRAGAdapter) are NOT
# imported here to avoid loading heavy dependencies at module import time.
# Import them directly where needed:
#   from baselines.full_selfrag_adapter import FullSelfRAGAdapter
#   from baselines.full_ketrag_adapter import FullKETRAGAdapter
#   etc.

__all__ = [
    # Full baseline interface (always available)
    'BaselineSystem',
    'BaselineResponse',
    'TokenTracker',
    'LatencyTracker',
]

# Add simplified baselines to __all__ if they're available
if SIMPLIFIED_BASELINES_AVAILABLE:
    __all__.extend([
        'SelfRAGSystem',
        'GraphRAGSystem',
        'SimpleGraphIndex',
        'KETRAGSystem',
        'TridentSystemWrapper',
    ])