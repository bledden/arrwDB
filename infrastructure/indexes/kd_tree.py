"""
KD-Tree index implementation.

This module provides a K-Dimensional tree for efficient nearest neighbor search.
Works well for low to medium dimensional data (< 20 dimensions) but degrades
to O(n) in very high dimensions due to the curse of dimensionality.

Time Complexity:
- Build: O(n log n)
- Insert: O(log n) but degrades tree balance
- Delete: O(log n) but degrades tree balance
- Search: O(log n) average, O(n) worst case in high dimensions

Space Complexity: O(n)
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID
import threading
from numpy.typing import NDArray
from dataclasses import dataclass
import heapq

from infrastructure.indexes.base import VectorIndex
from core.vector_store import VectorStore


@dataclass
class KDNode:
    """
    Node in the KD-Tree.

    Each node represents a point in k-dimensional space and splits
    the space along one dimension.
    """

    vector_id: UUID
    vector_index: int
    split_dim: int  # Dimension along which to split
    left: Optional["KDNode"] = None
    right: Optional["KDNode"] = None


class KDTreeIndex(VectorIndex):
    """
    K-Dimensional Tree index for nearest neighbor search.

    The KD-Tree recursively partitions space by splitting along dimensions
    with maximum variance. This creates a binary tree where each node
    represents a hyperplane splitting.

    Advantages:
    - Fast search for low-dimensional data (< 20D)
    - Memory efficient (only tree structure overhead)
    - Exact results (no approximation)

    Disadvantages:
    - Performance degrades in high dimensions (curse of dimensionality)
    - Requires periodic rebuilding after many updates
    - Not suitable for very high dimensional data (> 50D)

    Best for:
    - Low to medium dimensional embeddings (< 20D)
    - Static or slowly changing datasets
    - Applications needing exact results
    """

    def __init__(self, vector_store: VectorStore, rebuild_threshold: int = 100):
        """
        Initialize the KD-Tree index.

        Args:
            vector_store: The VectorStore containing the actual vectors.
            rebuild_threshold: Number of modifications before automatic rebuild.
                Set to 0 to disable automatic rebuilding.
        """
        self._vector_store = vector_store
        self._rebuild_threshold = rebuild_threshold
        self._root: Optional[KDNode] = None
        self._vector_map: Dict[UUID, int] = {}  # vector_id -> vector_index
        self._modifications_since_rebuild = 0
        self._lock = threading.RLock()

    def add_vector(self, vector_id: UUID, vector_index: int) -> None:
        """
        Add a vector to the index.

        This adds the vector to the map and triggers a rebuild if threshold
        is exceeded. For optimal performance, batch multiple additions and
        then call rebuild() manually.

        Args:
            vector_id: Unique identifier for the vector.
            vector_index: Index in the VectorStore.

        Raises:
            ValueError: If the vector_id already exists.
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
            self._modifications_since_rebuild += 1

            # Auto-rebuild if threshold exceeded
            if (
                self._rebuild_threshold > 0
                and self._modifications_since_rebuild >= self._rebuild_threshold
            ):
                self.rebuild()

    def remove_vector(self, vector_id: UUID) -> bool:
        """
        Remove a vector from the index.

        Marks the vector as removed and triggers rebuild if threshold exceeded.

        Args:
            vector_id: The ID of the vector to remove.

        Returns:
            True if the vector was removed, False if it didn't exist.
        """
        with self._lock:
            if vector_id not in self._vector_map:
                return False

            del self._vector_map[vector_id]
            self._modifications_since_rebuild += 1

            # Auto-rebuild if threshold exceeded
            if (
                self._rebuild_threshold > 0
                and self._modifications_since_rebuild >= self._rebuild_threshold
            ):
                self.rebuild()

            return True

    def search(
        self,
        query_vector: NDArray[np.float32],
        k: int,
        distance_threshold: Optional[float] = None,
    ) -> List[Tuple[UUID, float]]:
        """
        Search for k nearest neighbors using KD-Tree.

        Uses recursive branch-and-bound search with a priority queue
        to efficiently find nearest neighbors.

        Args:
            query_vector: The query vector (must be normalized).
            k: Number of nearest neighbors to return.
            distance_threshold: Optional maximum distance threshold.

        Returns:
            List of (vector_id, distance) tuples sorted by distance.

        Raises:
            ValueError: If query_vector dimension doesn't match or tree is empty.
        """
        with self._lock:
            if self._root is None:
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

            # Use a max-heap to track k nearest neighbors
            # Heap stores: (-distance, vector_id, vector_index)
            # Negative distance for max-heap behavior
            heap: List[Tuple[float, UUID, int]] = []

            self._search_recursive(
                self._root, query_vector, k, heap, distance_threshold
            )

            # Convert heap to sorted results
            results = [
                (vector_id, -dist) for dist, vector_id, _ in sorted(heap)
            ]

            return results

    def _search_recursive(
        self,
        node: Optional[KDNode],
        query: NDArray[np.float32],
        k: int,
        heap: List[Tuple[float, UUID, int]],
        distance_threshold: Optional[float],
    ) -> None:
        """
        Recursive KD-Tree search with branch-and-bound pruning.

        Args:
            node: Current node in the tree.
            query: Query vector.
            k: Number of neighbors to find.
            heap: Max-heap of current k nearest neighbors.
            distance_threshold: Optional distance threshold.
        """
        if node is None:
            return

        # Get the vector for this node
        vector = self._vector_store.get_vector_by_index(node.vector_index)

        # Compute distance (using cosine distance: 1 - dot product)
        similarity = np.dot(vector, query)
        distance = 1.0 - similarity

        # Check distance threshold
        if distance_threshold is not None and distance > distance_threshold:
            # Still need to explore children as they might be closer
            pass
        else:
            # Add to heap if we have room or if it's better than worst in heap
            if len(heap) < k:
                heapq.heappush(heap, (-distance, node.vector_id, node.vector_index))
            elif distance < -heap[0][0]:  # Better than worst in heap
                heapq.heapreplace(
                    heap, (-distance, node.vector_id, node.vector_index)
                )

        # Determine which branch to search first
        split_dim = node.split_dim
        if query[split_dim] < vector[split_dim]:
            near_node, far_node = node.left, node.right
        else:
            near_node, far_node = node.right, node.left

        # Search near branch first
        self._search_recursive(near_node, query, k, heap, distance_threshold)

        # Check if we need to search far branch
        # Compute distance to splitting plane
        plane_distance = abs(query[split_dim] - vector[split_dim])

        # Search far branch if:
        # 1. We don't have k neighbors yet, OR
        # 2. The splitting plane is closer than our worst neighbor
        should_search_far = len(heap) < k
        if not should_search_far and len(heap) > 0:
            worst_distance = -heap[0][0]
            # Use conservative bound: plane_distance might underestimate true distance
            should_search_far = plane_distance < worst_distance

        if should_search_far:
            self._search_recursive(far_node, query, k, heap, distance_threshold)

    def rebuild(self) -> None:
        """
        Rebuild the KD-Tree from scratch.

        This should be called periodically after many insertions/deletions
        to maintain balanced tree structure and optimal search performance.

        Time Complexity: O(n log n)
        """
        with self._lock:
            if len(self._vector_map) == 0:
                self._root = None
                self._modifications_since_rebuild = 0
                return

            # Get all vectors
            vector_ids = list(self._vector_map.keys())
            vector_indices = [self._vector_map[vid] for vid in vector_ids]

            # Build tree recursively
            self._root = self._build_tree(vector_ids, vector_indices, depth=0)
            self._modifications_since_rebuild = 0

    def _build_tree(
        self, vector_ids: List[UUID], vector_indices: List[int], depth: int
    ) -> Optional[KDNode]:
        """
        Recursively build a balanced KD-Tree.

        Splits on the dimension with maximum variance at each level
        and chooses the median as the splitting point.

        Args:
            vector_ids: List of vector IDs to build tree from.
            vector_indices: Corresponding vector indices.
            depth: Current depth in the tree.

        Returns:
            Root node of the subtree.
        """
        if len(vector_ids) == 0:
            return None

        # Get all vectors
        vectors = self._vector_store.get_vectors_by_indices(vector_indices)

        # Choose splitting dimension with maximum variance
        variances = np.var(vectors, axis=0)
        split_dim = int(np.argmax(variances))

        # Sort by split dimension and find median
        sorted_indices = np.argsort(vectors[:, split_dim])
        median_idx = len(sorted_indices) // 2

        # Create node for median point
        median_pos = sorted_indices[median_idx]
        node = KDNode(
            vector_id=vector_ids[median_pos],
            vector_index=vector_indices[median_pos],
            split_dim=split_dim,
        )

        # Recursively build left and right subtrees
        left_indices = sorted_indices[:median_idx]
        right_indices = sorted_indices[median_idx + 1 :]

        if len(left_indices) > 0:
            left_ids = [vector_ids[i] for i in left_indices]
            left_vecs = [vector_indices[i] for i in left_indices]
            node.left = self._build_tree(left_ids, left_vecs, depth + 1)

        if len(right_indices) > 0:
            right_ids = [vector_ids[i] for i in right_indices]
            right_vecs = [vector_indices[i] for i in right_indices]
            node.right = self._build_tree(right_ids, right_vecs, depth + 1)

        return node

    def size(self) -> int:
        """Get the number of vectors in the index."""
        with self._lock:
            return len(self._vector_map)

    def clear(self) -> None:
        """Remove all vectors from the index."""
        with self._lock:
            self._vector_map.clear()
            self._root = None
            self._modifications_since_rebuild = 0

    @property
    def supports_incremental_updates(self) -> bool:
        """
        KD-Tree supports incremental updates but requires periodic rebuilds.

        Returns:
            False, indicating rebuilds are recommended.
        """
        return False

    @property
    def index_type(self) -> str:
        """Get the index type."""
        return "kd_tree"

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the index.

        Returns:
            Dictionary with index statistics.
        """
        with self._lock:
            tree_depth = self._compute_tree_depth(self._root) if self._root else 0

            return {
                "type": self.index_type,
                "size": len(self._vector_map),
                "vector_store_dim": self._vector_store.dimension,
                "supports_incremental": self.supports_incremental_updates,
                "tree_depth": tree_depth,
                "modifications_since_rebuild": self._modifications_since_rebuild,
                "rebuild_threshold": self._rebuild_threshold,
            }

    def _compute_tree_depth(self, node: Optional[KDNode]) -> int:
        """Compute the maximum depth of the tree."""
        if node is None:
            return 0
        return 1 + max(
            self._compute_tree_depth(node.left),
            self._compute_tree_depth(node.right),
        )

    def __repr__(self) -> str:
        """String representation."""
        stats = self.get_statistics()
        return (
            f"KDTreeIndex(size={stats['size']}, "
            f"dimension={stats['vector_store_dim']}, "
            f"depth={stats['tree_depth']})"
        )
