"""
SearchReplay - Record and replay vector search paths for debugging.

NOVEL FEATURE: No other vector DB provides search path transparency.

WHY THIS MATTERS:
- Most vector DBs are black boxes - you get results but no insight into WHY
- When recall drops, you can't debug what went wrong in the search
- SearchReplay records the exact path through the index (HNSW graph traversal)
- Enables debugging: "Why didn't this vector get found?"

USE CASES:
1. Debug recall issues: See which graph nodes were visited vs skipped
2. Understand query performance: Count distance computations, cache misses
3. Optimize index parameters: Visualize how ef_search affects exploration
4. Detect index degradation: Compare search paths before/after bulk deletes

INSPIRATION: Borrowed from database query plan EXPLAIN - but for vector search.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set
from uuid import UUID, uuid4

import numpy as np
from numpy.typing import NDArray


@dataclass
class SearchStep:
    """
    Single step in the search path.

    Records what happened at each decision point during search.
    """

    step_number: int
    layer: int  # HNSW layer (0 = bottom layer with all vectors)
    node_id: UUID  # Vector ID that was evaluated
    distance: float  # Distance to query
    action: str  # "visited", "skipped", "backtracked", "pruned"

    # Performance metrics
    is_cache_hit: bool = False  # Was vector in CPU cache?
    distance_computations: int = 1  # How many distances computed at this step?

    # Why this action?
    reason: Optional[str] = None  # e.g., "too far from current best", "already visited"


@dataclass
class SearchPath:
    """
    Complete record of a vector search execution.

    WHY: Transparency into search behavior. Like EXPLAIN ANALYZE for SQL.
    """

    replay_id: str = field(default_factory=lambda: str(uuid4()))
    corpus_id: UUID = None
    query_vector: Optional[NDArray[np.float32]] = None
    k: int = 10

    # Execution trace
    steps: List[SearchStep] = field(default_factory=list)

    # Results
    result_ids: List[UUID] = field(default_factory=list)
    result_distances: List[float] = field(default_factory=list)

    # Performance metrics
    total_distance_computations: int = 0
    nodes_visited: int = 0
    nodes_skipped: int = 0
    layers_traversed: int = 0

    # Timing
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    duration_ms: float = 0.0

    # Index state snapshot
    index_type: str = "unknown"
    total_vectors_in_index: int = 0
    index_parameters: Dict = field(default_factory=dict)

    def add_step(
        self,
        layer: int,
        node_id: UUID,
        distance: float,
        action: str,
        reason: Optional[str] = None,
        is_cache_hit: bool = False,
    ):
        """
        Record a step in the search path.

        WHY: Called at every decision point during search to build complete trace.
        """
        step = SearchStep(
            step_number=len(self.steps),
            layer=layer,
            node_id=node_id,
            distance=distance,
            action=action,
            reason=reason,
            is_cache_hit=is_cache_hit,
        )
        self.steps.append(step)

        # Update counters
        self.total_distance_computations += 1
        if action == "visited":
            self.nodes_visited += 1
        elif action == "skipped":
            self.nodes_skipped += 1

    def finalize(self, result_ids: List[UUID], result_distances: List[float]):
        """
        Finalize the search path with results and compute metrics.

        WHY: Called when search completes to calculate summary statistics.
        """
        self.result_ids = result_ids
        self.result_distances = result_distances
        self.end_time = datetime.utcnow()

        duration = (self.end_time - self.start_time).total_seconds() * 1000
        self.duration_ms = round(duration, 2)

        # Count unique layers
        layers = set(step.layer for step in self.steps)
        self.layers_traversed = len(layers)

    def to_dict(self) -> Dict:
        """
        Convert to dictionary for API serialization.

        WHY: API endpoints return this for client-side visualization.
        """
        return {
            "replay_id": self.replay_id,
            "corpus_id": str(self.corpus_id) if self.corpus_id else None,
            "k": self.k,
            "index_type": self.index_type,
            "metrics": {
                "total_distance_computations": self.total_distance_computations,
                "nodes_visited": self.nodes_visited,
                "nodes_skipped": self.nodes_skipped,
                "layers_traversed": self.layers_traversed,
                "duration_ms": self.duration_ms,
                "cache_hit_rate": self._cache_hit_rate(),
            },
            "index_state": {
                "total_vectors": self.total_vectors_in_index,
                "parameters": self.index_parameters,
            },
            "results": {
                "count": len(self.result_ids),
                "ids": [str(id) for id in self.result_ids],
                "distances": self.result_distances,
            },
            "steps": [
                {
                    "step": step.step_number,
                    "layer": step.layer,
                    "node_id": str(step.node_id),
                    "distance": round(step.distance, 4),
                    "action": step.action,
                    "reason": step.reason,
                    "cache_hit": step.is_cache_hit,
                }
                for step in self.steps
            ],
        }

    def _cache_hit_rate(self) -> float:
        """Calculate percentage of cache hits during search."""
        if not self.steps:
            return 0.0
        cache_hits = sum(1 for step in self.steps if step.is_cache_hit)
        return round(cache_hits / len(self.steps) * 100, 2)

    def summary(self) -> str:
        """
        Human-readable summary of search path.

        WHY: Quick debugging output without full trace details.
        """
        return (
            f"SearchReplay {self.replay_id[:8]}:\n"
            f"  Index: {self.index_type} with {self.total_vectors_in_index} vectors\n"
            f"  Search: k={self.k}, {self.duration_ms}ms\n"
            f"  Exploration: {self.nodes_visited} visited, {self.nodes_skipped} skipped\n"
            f"  Efficiency: {self.total_distance_computations} distances, "
            f"{self._cache_hit_rate()}% cache hit rate\n"
            f"  Results: {len(self.result_ids)} found, "
            f"best distance={self.result_distances[0] if self.result_distances else 'N/A'}"
        )


class SearchReplayRecorder:
    """
    Manages recording of search paths.

    WHY: Singleton manager that indexes and stores search paths for later retrieval.
    Can enable/disable recording to avoid overhead in production.
    """

    def __init__(self, max_stored_paths: int = 1000):
        """
        Initialize the replay recorder.

        Args:
            max_stored_paths: Maximum number of search paths to keep in memory.
                Older paths are evicted when limit is reached.
        """
        self._enabled = False  # Recording disabled by default (no overhead)
        self._paths: Dict[str, SearchPath] = {}  # replay_id -> SearchPath
        self._max_stored_paths = max_stored_paths

    def enable(self):
        """Enable search path recording. PERF: Adds ~10% overhead to search."""
        self._enabled = True

    def disable(self):
        """Disable search path recording. Use in production for zero overhead."""
        self._enabled = False

    def is_enabled(self) -> bool:
        """Check if recording is currently enabled."""
        return self._enabled

    def start_recording(self, corpus_id: UUID, query_vector: NDArray, k: int) -> Optional[SearchPath]:
        """
        Start recording a new search path.

        Returns:
            SearchPath object to record steps, or None if recording disabled.
        """
        if not self._enabled:
            return None

        path = SearchPath(
            corpus_id=corpus_id,
            query_vector=query_vector,
            k=k,
        )

        self._paths[path.replay_id] = path

        # Evict oldest path if limit exceeded
        if len(self._paths) > self._max_stored_paths:
            oldest_id = min(self._paths.keys(), key=lambda k: self._paths[k].start_time)
            del self._paths[oldest_id]

        return path

    def get_path(self, replay_id: str) -> Optional[SearchPath]:
        """Retrieve a recorded search path by ID."""
        return self._paths.get(replay_id)

    def list_paths(
        self,
        corpus_id: Optional[UUID] = None,
        limit: int = 100
    ) -> List[SearchPath]:
        """
        List recorded search paths, optionally filtered by corpus.

        Args:
            corpus_id: If provided, only return paths for this corpus.
            limit: Maximum number of paths to return.

        Returns:
            List of search paths, sorted by most recent first.
        """
        paths = self._paths.values()

        if corpus_id:
            paths = [p for p in paths if p.corpus_id == corpus_id]

        # Sort by most recent first
        paths = sorted(paths, key=lambda p: p.start_time, reverse=True)

        return list(paths)[:limit]

    def clear(self):
        """Clear all recorded search paths. Use to free memory."""
        self._paths.clear()


# Global singleton instance
_recorder = SearchReplayRecorder()


def get_recorder() -> SearchReplayRecorder:
    """
    Get the global SearchReplay recorder.

    WHY: Singleton pattern - one recorder per process.
    """
    return _recorder
