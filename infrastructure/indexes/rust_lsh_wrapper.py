"""Rust LSH Index Wrapper - 2.5x faster search"""

from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID
import numpy as np
from numpy.typing import NDArray
import sys
from pathlib import Path

# Add rust/indexes to Python path
rust_indexes_path = Path(__file__).parent.parent.parent / "rust" / "indexes"
if str(rust_indexes_path) not in sys.path:
    sys.path.insert(0, str(rust_indexes_path))

try:
    import rust_hnsw
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

from core.vector_store import VectorStore
from infrastructure.indexes.base import VectorIndex


class RustLSHIndexWrapper(VectorIndex):
    """Wrapper for Rust LSH implementation."""

    def __init__(
        self,
        vector_store: VectorStore,
        num_tables: int = 10,
        hash_size: int = 10,
        seed: Optional[int] = None,
    ):
        if not RUST_AVAILABLE:
            raise ImportError("Rust module not available")

        self._vector_store = vector_store
        self._rust_index = rust_hnsw.RustLSHIndex(
            dimension=vector_store.dimension,
            num_tables=num_tables,
            hash_size=hash_size,
            seed=seed,
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
        return "rust_lsh"

    def get_statistics(self) -> Dict[str, Any]:
        stats = self._rust_index.get_statistics()
        stats.update({
            "vector_store_dim": self._vector_store.dimension,
        })
        return stats

    def __repr__(self) -> str:
        return f"RustLSHIndex(size={self.size()}, dimension={self._vector_store.dimension})"
