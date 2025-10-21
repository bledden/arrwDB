"""
Hierarchical Navigable Small World (HNSW) index implementation.

This module provides state-of-the-art approximate nearest neighbor search
using a hierarchical graph structure with navigable small world properties.

Time Complexity:
- Build: O(n * log n * M * log M)
- Insert: O(log n * M * log M)
- Delete: O(M * log M)
- Search: O(log n * M) average case

Space Complexity: O(n * M)
"""

import numpy as np
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import UUID
import threading
from numpy.typing import NDArray
import heapq
from dataclasses import dataclass, field

from infrastructure.indexes.base import VectorIndex
from core.vector_store import VectorStore


@dataclass
class HNSWNode:
    """
    Node in the HNSW graph.

    Each node represents a vector and maintains connections to neighbors
    at multiple layers of the hierarchy.
    """

    vector_id: UUID
    vector_index: int
    level: int  # Maximum level this node appears in
    # neighbors[layer] = set of neighbor vector_ids at that layer
    neighbors: Dict[int, Set[UUID]] = field(default_factory=dict)


class HNSWIndex(VectorIndex):
    """
    Hierarchical Navigable Small World graph index.

    HNSW builds a multi-layer graph where upper layers contain sparse
    long-range connections for fast navigation, while lower layers contain
    dense short-range connections for precise local search.

    Key Properties:
    - Logarithmic search complexity
    - High recall with low query time
    - Efficient incremental construction
    - Robust to high dimensions

    Advantages:
    - Best search performance among approximate methods
    - Works excellently in high dimensions
    - Good balance of speed, accuracy, and memory
    - Supports incremental updates

    Disadvantages:
    - More complex implementation
    - Higher memory usage than simpler methods
    - Requires parameter tuning (M, ef_construction)

    Best for:
    - Large datasets requiring fast queries
    - Production systems needing best performance
    - High-dimensional embeddings
    """

    def __init__(
        self,
        vector_store: VectorStore,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        max_level: int = 16,
        seed: Optional[int] = None,
    ):
        """
        Initialize the HNSW index.

        Args:
            vector_store: The VectorStore containing the actual vectors.
            M: Maximum number of connections per node per layer.
                Higher M increases recall but uses more memory. Typical: 12-48.
            ef_construction: Size of dynamic candidate list during construction.
                Higher values improve index quality but slow construction. Typical: 100-500.
            ef_search: Size of dynamic candidate list during search.
                Higher values improve recall but slow search. Typical: 50-500.
            max_level: Maximum number of layers in the hierarchy.
            seed: Random seed for reproducibility.

        Raises:
            ValueError: If parameters are invalid.
        """
        if M <= 0:
            raise ValueError(f"M must be positive, got {M}")
        if ef_construction <= 0:
            raise ValueError(
                f"ef_construction must be positive, got {ef_construction}"
            )
        if ef_search <= 0:
            raise ValueError(f"ef_search must be positive, got {ef_search}")

        self._vector_store = vector_store
        self._M = M
        self._M_max = M  # Max connections at layer 0
        self._M_max_upper = M  # Max connections at upper layers
        self._ef_construction = ef_construction
        self._ef_search = ef_search
        self._max_level = max_level
        self._ml = 1.0 / np.log(2.0)  # Normalization factor for level generation
        self._lock = threading.RLock()
        self._rng = np.random.RandomState(seed)

        # Graph structure
        self._nodes: Dict[UUID, HNSWNode] = {}
        self._entry_point: Optional[UUID] = None  # Entry point for search

    def add_vector(self, vector_id: UUID, vector_index: int) -> None:
        """
        Add a vector to the index.

        Time Complexity: O(log n * M * log M)

        The vector is inserted at a random level and connected to nearby
        neighbors at each level from bottom to top.

        Args:
            vector_id: Unique identifier for the vector.
            vector_index: Index in the VectorStore.

        Raises:
            ValueError: If the vector_id already exists.
        """
        with self._lock:
            if vector_id in self._nodes:
                raise ValueError(f"Vector ID {vector_id} already exists in index")

            # Verify the vector exists in the store
            try:
                vector = self._vector_store.get_vector_by_index(vector_index)
            except IndexError as e:
                raise ValueError(
                    f"Invalid vector_index {vector_index}: {e}"
                ) from e

            # Randomly select level for new node
            level = self._random_level()

            # Create new node
            node = HNSWNode(
                vector_id=vector_id,
                vector_index=vector_index,
                level=level,
                neighbors={i: set() for i in range(level + 1)},
            )

            # If this is the first node, make it the entry point
            if self._entry_point is None:
                self._entry_point = vector_id
                self._nodes[vector_id] = node
                return

            # Add node to graph BEFORE inserting (so it exists during connection)
            self._nodes[vector_id] = node

            # Find nearest neighbors and connect
            self._insert_node(node, vector)

    def _insert_node(self, node: HNSWNode, vector: NDArray[np.float32]) -> None:
        """
        Insert a node into the graph by connecting it to neighbors.

        Args:
            node: The node to insert.
            vector: The vector for this node.
        """
        # Start from entry point
        entry_level = self._nodes[self._entry_point].level
        current = self._entry_point
        current_vector = self._vector_store.get_vector_by_index(
            self._nodes[current].vector_index
        )
        current_dist = self._compute_distance(vector, current_vector)

        # Search for nearest neighbors from top to target layer
        for lc in range(entry_level, node.level, -1):
            current, current_dist = self._search_layer(
                vector, current, 1, lc
            )[0]

        # Insert at layers from node.level down to 0
        for lc in range(node.level, -1, -1):
            # Find ef_construction nearest neighbors at this layer
            candidates = self._search_layer(
                vector, current, self._ef_construction, lc
            )

            # Select M best neighbors
            M = self._M_max if lc == 0 else self._M_max_upper
            neighbors = self._select_neighbors(candidates, M)

            # Connect bidirectionally
            for neighbor_id, _ in neighbors:
                node.neighbors[lc].add(neighbor_id)

                # Only add reverse connection if neighbor has this layer
                neighbor_node = self._nodes[neighbor_id]
                if lc in neighbor_node.neighbors:
                    neighbor_node.neighbors[lc].add(node.vector_id)
                    # Prune neighbor's connections if needed
                    self._prune_connections(neighbor_id, lc)

            # Update current for next layer
            if len(candidates) > 0:
                current = candidates[0][0]

        # Update entry point if new node is higher
        if node.level > self._nodes[self._entry_point].level:
            self._entry_point = node.vector_id

    def _search_layer(
        self,
        query: NDArray[np.float32],
        entry_point: UUID,
        ef: int,
        layer: int,
    ) -> List[Tuple[UUID, float]]:
        """
        Search for nearest neighbors at a specific layer.

        Uses greedy search with a dynamic candidate list.

        Args:
            query: Query vector.
            entry_point: Starting point for search.
            ef: Size of the dynamic candidate list.
            layer: Layer to search.

        Returns:
            List of (vector_id, distance) tuples.
        """
        # Visited set to avoid revisiting nodes
        visited: Set[UUID] = {entry_point}

        # Candidate heap: (distance, vector_id)
        entry_dist = self._compute_distance_by_id(query, entry_point)
        candidates = [(entry_dist, entry_point)]
        heapq.heapify(candidates)

        # Result heap: (-distance, vector_id) for max-heap behavior
        results = [(-entry_dist, entry_point)]

        while len(candidates) > 0:
            current_dist, current_id = heapq.heappop(candidates)

            # Stop if current is farther than worst result
            if current_dist > -results[0][0]:
                break

            # Explore neighbors
            if layer not in self._nodes[current_id].neighbors:
                continue

            for neighbor_id in self._nodes[current_id].neighbors[layer]:
                if neighbor_id not in visited:
                    visited.add(neighbor_id)

                    neighbor_dist = self._compute_distance_by_id(query, neighbor_id)

                    # Add to results if better than worst or we have room
                    if neighbor_dist < -results[0][0] or len(results) < ef:
                        heapq.heappush(candidates, (neighbor_dist, neighbor_id))
                        heapq.heappush(results, (-neighbor_dist, neighbor_id))

                        # Prune results to size ef
                        if len(results) > ef:
                            heapq.heappop(results)

        # Convert results to ascending order
        return [(vid, -dist) for dist, vid in sorted(results, reverse=True)]

    def _select_neighbors(
        self, candidates: List[Tuple[UUID, float]], M: int
    ) -> List[Tuple[UUID, float]]:
        """
        Select M best neighbors from candidates using a heuristic.

        Uses a simple strategy: select the M nearest neighbors.
        A more sophisticated heuristic could ensure diversity.

        Args:
            candidates: List of (vector_id, distance) tuples.
            M: Number of neighbors to select.

        Returns:
            Selected neighbors.
        """
        # Sort by distance and take top M
        sorted_candidates = sorted(candidates, key=lambda x: x[1])
        return sorted_candidates[:M]

    def _prune_connections(self, node_id: UUID, layer: int) -> None:
        """
        Prune a node's connections to maintain maximum connection count.

        Args:
            node_id: The node to prune.
            layer: The layer to prune at.
        """
        M_max = self._M_max if layer == 0 else self._M_max_upper
        neighbors = self._nodes[node_id].neighbors[layer]

        if len(neighbors) <= M_max:
            return

        # Get distances to all neighbors
        node_vector = self._vector_store.get_vector_by_index(
            self._nodes[node_id].vector_index
        )

        neighbor_dists = [
            (nid, self._compute_distance_by_id(node_vector, nid))
            for nid in neighbors
            if nid in self._nodes  # Only compute for nodes that exist
        ]

        # Keep only M_max nearest
        neighbor_dists.sort(key=lambda x: x[1])
        new_neighbors = {nid for nid, _ in neighbor_dists[:M_max]}

        # Remove pruned connections bidirectionally
        for nid in neighbors - new_neighbors:
            # Only remove if the node exists
            if nid in self._nodes and layer in self._nodes[nid].neighbors:
                self._nodes[nid].neighbors[layer].discard(node_id)

        self._nodes[node_id].neighbors[layer] = new_neighbors

    def remove_vector(self, vector_id: UUID) -> bool:
        """
        Remove a vector from the index.

        Removes all connections to this node from neighbors.

        Args:
            vector_id: The ID of the vector to remove.

        Returns:
            True if the vector was removed, False if it didn't exist.
        """
        with self._lock:
            if vector_id not in self._nodes:
                return False

            node = self._nodes[vector_id]

            # Remove all connections from neighbors
            for layer in range(node.level + 1):
                for neighbor_id in node.neighbors.get(layer, set()):
                    if neighbor_id in self._nodes:
                        # Only remove if the neighbor has this layer
                        if layer in self._nodes[neighbor_id].neighbors:
                            self._nodes[neighbor_id].neighbors[layer].discard(
                                vector_id
                            )

            # Remove node
            del self._nodes[vector_id]

            # Update entry point if needed
            if self._entry_point == vector_id:
                if len(self._nodes) > 0:
                    # Find new entry point with highest level
                    self._entry_point = max(
                        self._nodes.keys(),
                        key=lambda vid: self._nodes[vid].level,
                    )
                else:
                    self._entry_point = None

            return True

    def search(
        self,
        query_vector: NDArray[np.float32],
        k: int,
        distance_threshold: Optional[float] = None,
    ) -> List[Tuple[UUID, float]]:
        """
        Search for k nearest neighbors.

        Time Complexity: O(log n * M)

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
            if self._entry_point is None:
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

            # Start from entry point and navigate down
            entry_level = self._nodes[self._entry_point].level
            current = self._entry_point

            # Search upper layers
            for lc in range(entry_level, 0, -1):
                nearest = self._search_layer(query_vector, current, 1, lc)
                if len(nearest) > 0:
                    current = nearest[0][0]

            # Search layer 0 with ef_search
            candidates = self._search_layer(
                query_vector, current, max(self._ef_search, k), 0
            )

            # Apply distance threshold if specified
            if distance_threshold is not None:
                candidates = [
                    (vid, dist)
                    for vid, dist in candidates
                    if dist <= distance_threshold
                ]

            # Return top k
            return candidates[:k]

    def _compute_distance(
        self, vec1: NDArray[np.float32], vec2: NDArray[np.float32]
    ) -> float:
        """Compute cosine distance between two vectors."""
        similarity = np.dot(vec1, vec2)
        return 1.0 - similarity

    def _compute_distance_by_id(
        self, query: NDArray[np.float32], vector_id: UUID
    ) -> float:
        """Compute distance from query to a vector by ID."""
        vector = self._vector_store.get_vector_by_index(
            self._nodes[vector_id].vector_index
        )
        return self._compute_distance(query, vector)

    def _random_level(self) -> int:
        """
        Randomly generate a level for a new node.

        Uses exponential decay to ensure higher layers are sparser.
        """
        level = 0
        while self._rng.random() < 0.5 and level < self._max_level:
            level += 1
        return level

    def rebuild(self) -> None:
        """
        Rebuild the index from scratch.

        For HNSW, this is expensive but can improve graph quality
        after many modifications.
        """
        with self._lock:
            if len(self._nodes) == 0:
                return

            # Store all vectors
            vector_data = [
                (node.vector_id, node.vector_index)
                for node in self._nodes.values()
            ]

            # Clear graph
            self._nodes.clear()
            self._entry_point = None

            # Re-insert all vectors
            for vector_id, vector_index in vector_data:
                self.add_vector(vector_id, vector_index)

    def size(self) -> int:
        """Get the number of vectors in the index."""
        with self._lock:
            return len(self._nodes)

    def clear(self) -> None:
        """Remove all vectors from the index."""
        with self._lock:
            self._nodes.clear()
            self._entry_point = None

    @property
    def supports_incremental_updates(self) -> bool:
        """HNSW supports efficient incremental updates."""
        return True

    @property
    def index_type(self) -> str:
        """Get the index type."""
        return "hnsw"

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the index."""
        with self._lock:
            if len(self._nodes) == 0:
                return {
                    "type": self.index_type,
                    "size": 0,
                    "vector_store_dim": self._vector_store.dimension,
                    "supports_incremental": self.supports_incremental_updates,
                    "M": self._M,
                    "ef_construction": self._ef_construction,
                    "ef_search": self._ef_search,
                }

            # Count nodes per level
            level_counts = {}
            for node in self._nodes.values():
                for level in range(node.level + 1):
                    level_counts[level] = level_counts.get(level, 0) + 1

            # Average connections
            total_connections = sum(
                sum(len(neighbors) for neighbors in node.neighbors.values())
                for node in self._nodes.values()
            )
            avg_connections = total_connections / len(self._nodes)

            return {
                "type": self.index_type,
                "size": len(self._nodes),
                "vector_store_dim": self._vector_store.dimension,
                "supports_incremental": self.supports_incremental_updates,
                "M": self._M,
                "ef_construction": self._ef_construction,
                "ef_search": self._ef_search,
                "num_levels": len(level_counts),
                "level_counts": level_counts,
                "avg_connections": round(avg_connections, 2),
            }

    def __repr__(self) -> str:
        """String representation."""
        stats = self.get_statistics()
        return (
            f"HNSWIndex(size={stats['size']}, "
            f"dimension={stats['vector_store_dim']}, "
            f"M={stats['M']}, "
            f"ef_search={stats['ef_search']})"
        )
