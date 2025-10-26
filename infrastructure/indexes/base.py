"""
Base interface for vector indexes.

This module defines the abstract base class that all index implementations
must follow, ensuring a consistent interface across different index types.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from uuid import UUID

import numpy as np
from numpy.typing import NDArray


class VectorIndex(ABC):
    """
    Abstract base class for all vector indexes.

    All index implementations (BruteForce, KDTree, LSH, HNSW) must inherit
    from this class and implement its abstract methods.

    Thread-Safety: Implementations must be thread-safe for concurrent reads
    and exclusive writes.
    """

    @abstractmethod
    def add_vector(self, vector_id: UUID, vector_index: int) -> None:
        """
        Add a vector to the index.

        Args:
            vector_id: Unique identifier for the vector (usually chunk ID).
            vector_index: Index in the VectorStore where the vector is stored.

        Raises:
            ValueError: If the vector_id already exists in the index.
        """
        pass

    @abstractmethod
    def remove_vector(self, vector_id: UUID) -> bool:
        """
        Remove a vector from the index.

        Args:
            vector_id: The ID of the vector to remove.

        Returns:
            True if the vector was removed, False if it didn't exist.
        """
        pass

    @abstractmethod
    def search(
        self,
        query_vector: NDArray[np.float32],
        k: int,
        distance_threshold: Optional[float] = None,
    ) -> List[Tuple[UUID, float]]:
        """
        Search for the k nearest neighbors to the query vector.

        Args:
            query_vector: The query vector (must be normalized).
            k: The number of nearest neighbors to return.
            distance_threshold: Optional maximum distance threshold.
                Only return vectors within this distance.

        Returns:
            A list of (vector_id, distance) tuples, sorted by distance
            in ascending order. May contain fewer than k results if
            distance_threshold is specified.

        Raises:
            ValueError: If query_vector dimension doesn't match the index.
        """
        pass

    @abstractmethod
    def rebuild(self) -> None:
        """
        Rebuild the index from scratch.

        This may be necessary after many insertions/deletions to maintain
        optimal performance. Some index types (like KD-Tree) require rebuilding
        to stay balanced.
        """
        pass

    @abstractmethod
    def size(self) -> int:
        """
        Get the number of vectors in the index.

        Returns:
            The number of indexed vectors.
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """
        Remove all vectors from the index.
        """
        pass

    @property
    @abstractmethod
    def supports_incremental_updates(self) -> bool:
        """
        Whether this index supports efficient incremental updates.

        Returns:
            True if the index can efficiently handle add/remove operations,
            False if it requires periodic rebuilds for optimal performance.
        """
        pass

    @property
    @abstractmethod
    def index_type(self) -> str:
        """
        Get the type of this index.

        Returns:
            One of: "brute_force", "kd_tree", "lsh", "hnsw"
        """
        pass
