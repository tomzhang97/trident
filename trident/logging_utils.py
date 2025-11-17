"""Logging and telemetry utilities for TRIDENT."""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


def setup_logger(
    output_dir: str,
    name: str = "trident",
    level: str = "INFO"
) -> logging.Logger:
    """Set up logger with file and console handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level))
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_format)
    
    # File handler
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(f"{output_dir}/trident.log")
    file_handler.setLevel(getattr(logging, level))
    file_handler.setFormatter(console_format)
    
    # Add handlers
    logger.handlers.clear()
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


def log_metrics(
    metrics: Dict[str, Any],
    output_path: str,
    append: bool = True
) -> None:
    """Log metrics to JSON file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    if append and Path(output_path).exists():
        with open(output_path, 'r') as f:
            existing = json.load(f)
        if isinstance(existing, list):
            existing.append(metrics)
            metrics = existing
        else:
            metrics = [existing, metrics]
    
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)


@dataclass
class TelemetryTracker:
    """Track telemetry data for pipeline execution."""
    
    config: Any
    events: List[Dict[str, Any]] = field(default_factory=list)
    timers: Dict[str, float] = field(default_factory=dict)
    counters: Dict[str, int] = field(default_factory=dict)
    current_query: Optional[str] = None
    query_start_time: Optional[float] = None
    
    def start_query(self, query: str) -> None:
        """Start tracking a new query."""
        self.current_query = query
        self.query_start_time = time.time()
        self.log("query_start", {"query": query})
    
    def start_timer(self, name: str) -> None:
        """Start a named timer."""
        self.timers[f"{name}_start"] = time.time()
    
    def stop_timer(self, name: str) -> float:
        """Stop a named timer and return elapsed time."""
        start_key = f"{name}_start"
        if start_key not in self.timers:
            return 0.0
        
        elapsed = time.time() - self.timers[start_key]
        self.timers[name] = elapsed
        del self.timers[start_key]
        
        self.log(f"{name}_complete", {"elapsed_ms": elapsed * 1000})
        return elapsed
    
    def increment(self, name: str, value: int = 1) -> None:
        """Increment a counter."""
        self.counters[name] = self.counters.get(name, 0) + value
    
    def log(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log an event."""
        if not self.config.enable:
            return
        
        event = {
            "timestamp": time.time(),
            "event_type": event_type,
            "data": data
        }
        
        if self.current_query:
            event["query"] = self.current_query
        
        self.events.append(event)
    
    def get_trace(self) -> Dict[str, Any]:
        """Get complete telemetry trace."""
        return {
            "events": self.events,
            "timers": self.timers,
            "counters": self.counters,
            "total_time": time.time() - self.query_start_time if self.query_start_time else 0
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            "num_events": len(self.events),
            "total_time": time.time() - self.query_start_time if self.query_start_time else 0,
            "timers": dict(self.timers),
            "counters": dict(self.counters)
        }
    
    def save_trace(self, output_path: str) -> None:
        """Save telemetry trace to file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(self.get_trace(), f, indent=2)
    
    def reset(self) -> None:
        """Reset telemetry for new query."""
        self.events.clear()
        self.timers.clear()
        self.counters.clear()
        self.current_query = None
        self.query_start_time = None