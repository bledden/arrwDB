"""
Locality-Sensitive Hashing (LSH) index implementation.

This module provides an approximate nearest neighbor search using random
hyperplane projections. LSH trades accuracy for speed, making it suitable
for high-dimensional data and large datasets.

Time Complexity:
- Build: O(n * L * k) where L = num tables, k = hash size
- Insert: O(L * k)
- Delete: O(L * k)
- Search: O(L * b) where b = average bucket size

Space Complexity: O(n * L)
"""

import threading
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import UUID

import numpy as np
from numpy.typing import NDArray

from core.vector_store import VectorStore
from infrastructure.indexes.base import VectorIndex


class LSHIndex(VectorIndex):
    """
    Locality-Sensitive Hashing index for approximate nearest neighbor search.

    LSH uses random hyperplane projections to hash similar vectors to the
    same buckets. Multiple hash tables are used to increase recall.

    Hash Function:
    For each hash table, we generate k random hyperplanes. A vector's hash
    is determined by which side of each hyperplane it falls on, creating
    a k-bit binary hash.

    Advantages:
    - Sub-linear search time for large datasets
    - Works well in high dimensions
    - Tunable accuracy/speed tradeoff
    - Supports incremental updates

    Disadvantages:
    - Approximate results (may miss some nearest neighbors)
    - Requires tuning parameters (num_tables, hash_size)
    - Higher memory usage than exact methods

    Best for:
    - Large datasets (> 100K vectors)
    - High-dimensional data (> 50D)
    - Applications that can tolerate approximate results
    """

    def __init__(
        self,
        vector_store: VectorStore,
        num_tables: int = 10,
        hash_size: int = 10,
        seed: Optional[int] = None,
    ):
        """
        Initialize the LSH index.

        Args:
            vector_store: The VectorStore containing the actual vectors.
            num_tables: Number of hash tables (L). More tables increase
                recall but use more memory. Typical: 5-20.
            hash_size: Number of bits per hash (k). Larger values create
                more buckets with fewer items. Typical: 8-16.
            seed: Random seed for reproducibility.

        Raises:
            ValueError: If parameters are invalid.
        """
        if num_tables <= 0:
            raise ValueError(f"num_tables must be positive, got {num_tables}")
        if hash_size <= 0:
            raise ValueError(f"hash_size must be positive, got {hash_size}")

        self._vector_store = vector_store
        self._num_tables = num_tables
        self._hash_size = hash_size
        self._lock = threading.RLock()
        self._rng = np.random.RandomState(seed)

        # Generate random hyperplanes for hashing
        # Shape: (num_tables, hash_size, dimension)
        dimension = vector_store.dimension
        self._hyperplanes = self._rng.randn(num_tables, hash_size, dimension)

        # Normalize hyperplanes to unit length
        norms = np.linalg.norm(self._hyperplanes, axis=2, keepdims=True)
        self._hyperplanes = self._hyperplanes / norms

        # Hash tables: List of dicts mapping hash -> set of (vector_id, vector_index)
        self._tables: List[Dict[int, Set[Tuple[UUID, int]]]] = [
            defaultdict(set) for _ in range(num_tables)
        ]

        # Reverse mapping for removal: vector_id -> list of hashes (one per table)
        self._vector_hashes: Dict[UUID, List[int]] = {}

    def add_vector(self, vector_id: UUID, vector_index: int) -> None:
        """
        Add a vector to the index.

        Time Complexity: O(L * k) where L = num_tables, k = hash_size

        Args:
            vector_id: Unique identifier for the vector.
            vector_index: Index in the VectorStore.

        Raises:
            ValueError: If the vector_id already exists.
        """
        with self._lock:
            if vector_id in self._vector_hashes:
                raise ValueError(f"Vector ID {vector_id} already exists in index")

            # Get the vector
            try:
                vector = self._vector_store.get_vector_by_index(vector_index)
            except IndexError as e:
                raise ValueError(
                    f"Invalid vector_index {vector_index}: {e}"
                ) from e

            # Compute hashes for all tables
            hashes = self._compute_hashes(vector)

            # Insert into each table
            for table_idx, hash_val in enumerate(hashes):
                self._tables[table_idx][hash_val].add((vector_id, vector_index))

            # Store hashes for future removal
            self._vector_hashes[vector_id] = hashes

    def remove_vector(self, vector_id: UUID) -> bool:
        """
        Remove a vector from the index.

        Time Complexity: O(L * k)

        Args:
            vector_id: The ID of the vector to remove.

        Returns:
            True if the vector was removed, False if it didn't exist.
        """
        with self._lock:
            if vector_id not in self._vector_hashes:
                return False

            # Get stored hashes
            hashes = self._vector_hashes[vector_id]

            # Get vector_index before removing
            # Find it in the first table
            first_hash = hashes[0]
            vector_index = None
            for vid, vidx in self._tables[0][first_hash]:
                if vid == vector_id:
                    vector_index = vidx
                    break

            # Remove from all tables
            for table_idx, hash_val in enumerate(hashes):
                self._tables[table_idx][hash_val].discard(
                    (vector_id, vector_index)
                )

                # Clean up empty buckets
                if len(self._tables[table_idx][hash_val]) == 0:
                    del self._tables[table_idx][hash_val]

            # Remove from hash map
            del self._vector_hashes[vector_id]

            return True

    def search(
        self,
        query_vector: NDArray[np.float32],
        k: int,
        distance_threshold: Optional[float] = None,
    ) -> List[Tuple[UUID, float]]:
        """
        Search for k approximate nearest neighbors.

        Time Complexity: O(L * b) where b = average bucket size

        This searches all hash tables and collects candidate vectors
        from matching buckets. Candidates are then ranked by actual distance.

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
            if len(self._vector_hashes) == 0:
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

            # Compute hashes for query
            hashes = self._compute_hashes(query_vector)

            # Collect candidate vectors from all tables
            candidates: Set[Tuple[UUID, int]] = set()
            for table_idx, hash_val in enumerate(hashes):
                if hash_val in self._tables[table_idx]:
                    candidates.update(self._tables[table_idx][hash_val])

            if len(candidates) == 0:
                return []

            # Compute actual distances for candidates
            candidate_ids = [vid for vid, _ in candidates]
            candidate_indices = [vidx for _, vidx in candidates]

            vectors = self._vector_store.get_vectors_by_indices(candidate_indices)

            # Compute cosine distances
            similarities = np.dot(vectors, query_vector)
            distances = 1.0 - similarities

            # Apply distance threshold if specified
            if distance_threshold is not None:
                valid_mask = distances <= distance_threshold
                distances = distances[valid_mask]
                candidate_ids = [
                    vid for vid, valid in zip(candidate_ids, valid_mask) if valid
                ]

            if len(distances) == 0:
                return []

            # Sort by distance and take top k
            if k < len(distances):
                partition_indices = np.argpartition(distances, k - 1)[:k]
                top_k_indices = partition_indices[
                    np.argsort(distances[partition_indices])
                ]
            else:
                top_k_indices = np.argsort(distances)

            results = [
                (candidate_ids[idx], float(distances[idx]))
                for idx in top_k_indices[:k]
            ]

            return results

    def _compute_hashes(self, vector: NDArray[np.float32]) -> List[int]:
        """
        Compute hash values for a vector across all tables.

        For each table, computes which side of each hyperplane the vector
        is on, creating a binary hash.

        Args:
            vector: The vector to hash.

        Returns:
            List of hash values, one per table.
        """
        # Compute dot products with all hyperplanes
        # Shape: (num_tables, hash_size)
        projections = np.dot(self._hyperplanes, vector)

        # Convert to binary: 1 if positive, 0 if negative
        binary_hashes = (projections > 0).astype(int)

        # Convert binary arrays to integers
        # Each row is a binary number
        hashes = []
        for i in range(self._num_tables):
            # Convert binary array to integer
            hash_val = 0
            for bit in binary_hashes[i]:
                hash_val = (hash_val << 1) | bit
            hashes.append(hash_val)

        return hashes

    def rebuild(self) -> None:
        """
        Rebuild the index with new random hyperplanes.

        This creates entirely new hash functions, which may improve
        distribution if the current ones are suboptimal.

        Time Complexity: O(n * L * k)
        """
        with self._lock:
            if len(self._vector_hashes) == 0:
                return

            # Store current vectors
            vector_ids = list(self._vector_hashes.keys())

            # Find vector indices by looking in first table
            vector_indices = []
            for vid in vector_ids:
                first_hash = self._vector_hashes[vid][0]
                for stored_vid, vidx in self._tables[0][first_hash]:
                    if stored_vid == vid:
                        vector_indices.append(vidx)
                        break

            # Clear current state
            self._tables = [defaultdict(set) for _ in range(self._num_tables)]
            self._vector_hashes.clear()

            # Generate new random hyperplanes
            dimension = self._vector_store.dimension
            self._hyperplanes = self._rng.randn(
                self._num_tables, self._hash_size, dimension
            )
            norms = np.linalg.norm(self._hyperplanes, axis=2, keepdims=True)
            self._hyperplanes = self._hyperplanes / norms

            # Re-insert all vectors
            for vid, vidx in zip(vector_ids, vector_indices):
                self.add_vector(vid, vidx)

    def size(self) -> int:
        """Get the number of vectors in the index."""
        with self._lock:
            return len(self._vector_hashes)

    def clear(self) -> None:
        """Remove all vectors from the index."""
        with self._lock:
            self._tables = [defaultdict(set) for _ in range(self._num_tables)]
            self._vector_hashes.clear()

    @property
    def supports_incremental_updates(self) -> bool:
        """
        LSH supports efficient incremental updates.

        Returns:
            True
        """
        return True

    @property
    def index_type(self) -> str:
        """Get the index type."""
        return "lsh"

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the index.

        Returns:
            Dictionary with index statistics.
        """
        with self._lock:
            # Compute average bucket size
            total_buckets = 0
            total_items = 0
            for table in self._tables:
                total_buckets += len(table)
                for bucket in table.values():
                    total_items += len(bucket)

            avg_bucket_size = (
                total_items / total_buckets if total_buckets > 0 else 0
            )

            return {
                "type": self.index_type,
                "size": len(self._vector_hashes),
                "vector_store_dim": self._vector_store.dimension,
                "supports_incremental": self.supports_incremental_updates,
                "num_tables": self._num_tables,
                "hash_size": self._hash_size,
                "total_buckets": total_buckets,
                "avg_bucket_size": round(avg_bucket_size, 2),
            }

    def __repr__(self) -> str:
        """String representation."""
        stats = self.get_statistics()
        return (
            f"LSHIndex(size={stats['size']}, "
            f"dimension={stats['vector_store_dim']}, "
            f"tables={stats['num_tables']}, "
            f"hash_size={stats['hash_size']})"
        )
