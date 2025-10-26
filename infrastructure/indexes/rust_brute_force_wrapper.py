"""
Rust BruteForce Index Wrapper

Drop-in replacement for Python BruteForce with 12x faster additions
and 1.5x faster search.
"""

from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

import numpy as np
from numpy.typing import NDArray

try:
    import rust_hnsw
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

from core.vector_store import VectorStore
from infrastructure.indexes.base import VectorIndex


class RustBruteForceIndexWrapper(VectorIndex):
    """Wrapper for Rust BruteForce implementation."""

    def __init__(self, vector_store: VectorStore):
        if not RUST_AVAILABLE:
            raise ImportError("Rust module not available")

        self._vector_store = vector_store
        self._rust_index = rust_hnsw.RustBruteForceIndex(
            dimension=vector_store.dimension
        )
        self._vector_id_to_index: Dict[UUID, int] = {}

    def add_vector(self, vector_id: UUID, vector_index: int) -> None:
        if vector_id in self._vector_id_to_index:
            raise ValueError(f"Vector ID {vector_id} already exists in index")

        vector = self._vector_store.get_vector_by_index(vector_index)
        self._rust_index.add_vector(str(vector_id), vector)
        self._vector_id_to_index[vector_id] = vector_index

    def remove_vector(self, vector_id: UUID) -> bool:
        if vector_id not in self._vector_id_to_index:
            return False

        removed = self._rust_index.remove_vector(str(vector_id))
        if removed:
            del self._vector_id_to_index[vector_id]
        return removed

    def search(
        self,
        query_vector: NDArray[np.float32],
        k: int,
        distance_threshold: Optional[float] = None,
    ) -> List[Tuple[UUID, float]]:
        results = self._rust_index.search(query_vector, k, distance_threshold)
        return [(UUID(vid_str), dist) for vid_str, dist in results]

    def rebuild(self) -> None:
        self._rust_index.rebuild()

    def size(self) -> int:
        return self._rust_index.size()

    def clear(self) -> None:
        self._rust_index.clear()
        self._vector_id_to_index.clear()

    @property
    def supports_incremental_updates(self) -> bool:
        return True

    @property
    def index_type(self) -> str:
        return "rust_brute_force"

    def get_statistics(self) -> Dict[str, Any]:
        stats = self._rust_index.get_statistics()
        stats.update({
            "vector_store_dim": self._vector_store.dimension,
            "supports_incremental": self.supports_incremental_updates,
        })
        return stats

    def __repr__(self) -> str:
        return (
            f"RustBruteForceIndex(size={self.size()}, "
            f"dimension={self._vector_store.dimension})"
        )
