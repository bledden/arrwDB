"""
Brute Force index implementation.

This module provides a simple brute force search that compares the query
against all vectors. It's exact (no approximation) and works well for
small to medium datasets (< 100K vectors).

Time Complexity:
- Insert: O(1)
- Delete: O(1)
- Search: O(n*d) where n = number of vectors, d = dimension

Space Complexity: O(n) for the mapping
"""

import threading
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

import numpy as np
from numpy.typing import NDArray

from core.vector_store import VectorStore
from infrastructure.indexes.base import VectorIndex


class BruteForceIndex(VectorIndex):
    """
    Brute force linear search index.

    This index simply stores a mapping from vector IDs to their indices
    in the VectorStore. Search compares the query against every vector.

    Advantages:
    - Exact results (no approximation)
    - Fast insertion/deletion (O(1))
    - No index building time
    - Memory efficient (only stores ID mapping)
    - Works well for any dimension

    Disadvantages:
    - Slow search for large datasets (O(n*d))
    - No early termination optimization

    Best for:
    - Small datasets (< 100K vectors)
    - Cases where exact results are critical
    - Frequent updates with infrequent searches
    """

    def __init__(self, vector_store: VectorStore):
        """
        Initialize the BruteForce index.

        Args:
            vector_store: The VectorStore containing the actual vectors.
        """
        self._vector_store = vector_store
        self._vector_map: Dict[UUID, int] = {}  # vector_id -> vector_index
        self._lock = threading.RLock()

    def add_vector(self, vector_id: UUID, vector_index: int) -> None:
        """
        Add a vector to the index.

        Time Complexity: O(1)

        Args:
            vector_id: Unique identifier for the vector.
            vector_index: Index in the VectorStore.

        Raises:
            ValueError: If the vector_id already exists or vector_index is invalid.
        """
        with self._lock:
            if vector_id in self._vector_map:
                raise ValueError(f"Vector ID {vector_id} already exists in index")

            # Verify the vector exists in the store
            try:
                self._vector_store.get_vector_by_index(vector_index)
            except IndexError as e:
                raise ValueError(
                    f"Invalid vector_index {vector_index}: {e}"
                ) from e

            self._vector_map[vector_id] = vector_index

    def remove_vector(self, vector_id: UUID) -> bool:
        """
        Remove a vector from the index.

        Time Complexity: O(1)

        Args:
            vector_id: The ID of the vector to remove.

        Returns:
            True if the vector was removed, False if it didn't exist.
        """
        with self._lock:
            if vector_id in self._vector_map:
                del self._vector_map[vector_id]
                return True
            return False

    def search(
        self,
        query_vector: NDArray[np.float32],
        k: int,
        distance_threshold: Optional[float] = None,
    ) -> List[Tuple[UUID, float]]:
        """
        Search for k nearest neighbors using brute force.

        Time Complexity: O(n*d) where n = number of vectors, d = dimension

        This implementation:
        1. Computes cosine similarity with all vectors
        2. Sorts by similarity (descending)
        3. Returns top k results

        Cosine similarity is used since all vectors are normalized.
        Distance = 1 - cosine_similarity, so smaller distance = more similar.

        Args:
            query_vector: The query vector (must be normalized).
            k: Number of nearest neighbors to return.
            distance_threshold: Optional maximum distance threshold.

        Returns:
            List of (vector_id, distance) tuples sorted by distance.

        Raises:
            ValueError: If query_vector dimension doesn't match.
        """
        with self._lock:
            if len(self._vector_map) == 0:
                return []

            if k <= 0:
                raise ValueError(f"k must be positive, got {k}")

            # Validate query vector dimension
            expected_dim = self._vector_store.dimension
            if len(query_vector) != expected_dim:
                raise ValueError(
                    f"Query vector dimension {len(query_vector)} doesn't match "
                    f"store dimension {expected_dim}"
                )

            # Get all vector indices
            vector_ids = list(self._vector_map.keys())
            vector_indices = [self._vector_map[vid] for vid in vector_ids]

            # Retrieve all vectors from store
            vectors = self._vector_store.get_vectors_by_indices(vector_indices)

            # Compute cosine similarities (dot product since vectors are normalized)
            similarities = np.dot(vectors, query_vector)

            # Convert to distances (1 - similarity)
            distances = 1.0 - similarities

            # Apply distance threshold if specified
            if distance_threshold is not None:
                valid_mask = distances <= distance_threshold
                distances = distances[valid_mask]
                vector_ids = [vid for vid, valid in zip(vector_ids, valid_mask) if valid]

            # Sort by distance and take top k
            if len(distances) == 0:
                return []

            # Use argpartition for better performance when k << n
            if k < len(distances):
                # argpartition is O(n) vs O(n log n) for full sort
                partition_indices = np.argpartition(distances, k - 1)[:k]
                # Sort only the top k
                top_k_indices = partition_indices[np.argsort(distances[partition_indices])]
            else:
                # If k >= n, just sort everything
                top_k_indices = np.argsort(distances)

            # Build results
            results = [
                (vector_ids[idx], float(distances[idx]))
                for idx in top_k_indices[:k]
            ]

            return results

    def rebuild(self) -> None:
        """
        Rebuild the index.

        For BruteForce index, this is a no-op since there's no structure
        to rebuild. The method is provided for interface consistency.
        """
        # No-op: brute force has no structure to rebuild
        pass

    def size(self) -> int:
        """
        Get the number of vectors in the index.

        Returns:
            The number of indexed vectors.
        """
        with self._lock:
            return len(self._vector_map)

    def clear(self) -> None:
        """
        Remove all vectors from the index.
        """
        with self._lock:
            self._vector_map.clear()

    @property
    def supports_incremental_updates(self) -> bool:
        """
        BruteForce supports efficient incremental updates.

        Returns:
            Always True.
        """
        return True

    @property
    def index_type(self) -> str:
        """
        Get the index type.

        Returns:
            "brute_force"
        """
        return "brute_force"

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the index.

        Returns:
            Dictionary with:
            - type: Index type
            - size: Number of vectors
            - vector_store_dim: Dimension of vectors
            - supports_incremental: Whether incremental updates are supported
        """
        with self._lock:
            return {
                "type": self.index_type,
                "size": len(self._vector_map),
                "vector_store_dim": self._vector_store.dimension,
                "supports_incremental": self.supports_incremental_updates,
            }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"BruteForceIndex(size={self.size()}, "
            f"dimension={self._vector_store.dimension})"
        )
