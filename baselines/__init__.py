"""Baseline systems for comparison with TRIDENT."""

from .self_rag_system import SelfRAGSystem
from .graphrag_system import GraphRAGSystem, SimpleGraphIndex
from .trident_wrapper import TridentSystemWrapper

__all__ = ['SelfRAGSystem', 'GraphRAGSystem', 'SimpleGraphIndex', 'TridentSystemWrapper']
