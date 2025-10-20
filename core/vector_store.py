"""
Centralized Vector Storage with Reference Counting.

This module provides the VectorStore which manages all vectors in memory
with reference counting for efficient memory usage. It supports both
in-memory and memory-mapped storage for large datasets.
"""

import numpy as np
from typing import Dict, Optional, Set, Tuple
from uuid import UUID
from pathlib import Path
import threading
from numpy.typing import NDArray


class VectorStore:
    """
    Centralized storage for all vectors with reference counting.

    The VectorStore maintains:
    - A single numpy array containing all vectors
    - Reference counts for each vector (how many chunks use it)
    - Mapping from chunk IDs to vector indices
    - Optional memory-mapped storage for large datasets

    Thread-Safety: This class uses a lock for all operations to ensure
    thread-safe access to the internal data structures.

    Memory Efficiency:
    - Vectors are deduplicated (identical vectors share storage)
    - Reference counting allows cleanup of unused vectors
    - Memory-mapping allows working with datasets larger than RAM
    """

    def __init__(
        self,
        dimension: int,
        initial_capacity: int = 1000,
        use_mmap: bool = False,
        mmap_path: Optional[Path] = None,
    ):
        """
        Initialize the VectorStore.

        Args:
            dimension: The dimensionality of vectors to store.
            initial_capacity: Initial capacity for the vector array.
                Will grow automatically as needed.
            use_mmap: Whether to use memory-mapped storage.
                Set to True for large datasets (>1M vectors).
            mmap_path: Path to the memory-mapped file.
                Required if use_mmap is True.

        Raises:
            ValueError: If dimension is invalid, or if use_mmap is True
                but mmap_path is None.
        """
        if dimension <= 0:
            raise ValueError(f"Dimension must be positive, got {dimension}")

        if initial_capacity <= 0:
            raise ValueError(
                f"Initial capacity must be positive, got {initial_capacity}"
            )

        if use_mmap and mmap_path is None:
            raise ValueError("mmap_path is required when use_mmap is True")

        self._dimension = dimension
        self._use_mmap = use_mmap
        self._mmap_path = mmap_path
        self._lock = threading.RLock()

        # Core data structures
        self._chunk_to_index: Dict[UUID, int] = {}  # chunk_id -> vector index
        self._ref_counts: Dict[int, int] = {}  # vector index -> reference count
        self._vector_hashes: Dict[int, int] = {}  # vector hash -> vector index
        self._next_index: int = 0  # Next available index for new vectors
        self._free_indices: Set[int] = set()  # Indices freed by deletions

        # Initialize storage
        if use_mmap:
            self._initialize_mmap_storage(initial_capacity)
        else:
            self._vectors = np.zeros(
                (initial_capacity, dimension), dtype=np.float32
            )

    def _initialize_mmap_storage(self, capacity: int) -> None:
        """
        Initialize memory-mapped storage.

        Creates a memory-mapped file for vector storage.
        """
        assert self._mmap_path is not None

        # Ensure directory exists
        self._mmap_path.parent.mkdir(parents=True, exist_ok=True)

        # Create memory-mapped array
        self._vectors = np.memmap(
            self._mmap_path,
            dtype=np.float32,
            mode="w+",
            shape=(capacity, self._dimension),
        )

    @property
    def dimension(self) -> int:
        """Get the vector dimension."""
        return self._dimension

    @property
    def count(self) -> int:
        """Get the number of unique vectors stored."""
        with self._lock:
            return self._next_index - len(self._free_indices)

    @property
    def total_references(self) -> int:
        """Get the total number of references to all vectors."""
        with self._lock:
            return sum(self._ref_counts.values())

    def add_vector(
        self, chunk_id: UUID, vector: NDArray[np.float32]
    ) -> int:
        """
        Add a vector to the store and associate it with a chunk.

        If an identical vector already exists, increments its reference count
        instead of storing a duplicate.

        Args:
            chunk_id: The ID of the chunk this vector belongs to.
            vector: The normalized vector to store (must be 1D array).

        Returns:
            The index where the vector is stored.

        Raises:
            ValueError: If the vector dimension doesn't match the store's dimension,
                or if the chunk_id is already associated with a vector.
        """
        with self._lock:
            # Validate vector
            if vector.ndim != 1:
                raise ValueError(
                    f"Expected 1D vector, got shape {vector.shape}"
                )

            if len(vector) != self._dimension:
                raise ValueError(
                    f"Vector dimension mismatch: expected {self._dimension}, "
                    f"got {len(vector)}"
                )

            # Check if chunk already has a vector
            if chunk_id in self._chunk_to_index:
                raise ValueError(
                    f"Chunk {chunk_id} already has an associated vector"
                )

            # Check for duplicate vector
            vector_hash = self._hash_vector(vector)
            if vector_hash in self._vector_hashes:
                # Reuse existing vector
                index = self._vector_hashes[vector_hash]
                # Verify it's actually identical (hash collision check)
                if np.allclose(self._vectors[index], vector, atol=1e-6):
                    self._chunk_to_index[chunk_id] = index
                    self._ref_counts[index] += 1
                    return index

            # Store new vector
            index = self._allocate_index()
            self._ensure_capacity(index + 1)

            self._vectors[index] = vector
            self._chunk_to_index[chunk_id] = index
            self._ref_counts[index] = 1
            self._vector_hashes[vector_hash] = index

            return index

    def get_vector(self, chunk_id: UUID) -> Optional[NDArray[np.float32]]:
        """
        Retrieve the vector associated with a chunk.

        Args:
            chunk_id: The ID of the chunk.

        Returns:
            The vector as a numpy array, or None if the chunk has no vector.
        """
        with self._lock:
            index = self._chunk_to_index.get(chunk_id)
            if index is None:
                return None
            return self._vectors[index].copy()

    def get_vector_by_index(self, index: int) -> NDArray[np.float32]:
        """
        Retrieve a vector by its index.

        Args:
            index: The vector index.

        Returns:
            The vector as a numpy array.

        Raises:
            IndexError: If the index is invalid or has been freed.
        """
        with self._lock:
            if index < 0 or index >= self._next_index:
                raise IndexError(f"Invalid vector index: {index}")

            if index in self._free_indices:
                raise IndexError(f"Vector index {index} has been freed")

            return self._vectors[index].copy()

    def remove_vector(self, chunk_id: UUID) -> bool:
        """
        Remove the association between a chunk and its vector.

        Decrements the reference count for the vector. If the reference count
        reaches zero, the vector is freed and its index can be reused.

        Args:
            chunk_id: The ID of the chunk.

        Returns:
            True if the chunk had an associated vector that was removed,
            False if the chunk had no vector.
        """
        with self._lock:
            index = self._chunk_to_index.get(chunk_id)
            if index is None:
                return False

            # Remove chunk association
            del self._chunk_to_index[chunk_id]

            # Decrement reference count
            self._ref_counts[index] -= 1

            # If no more references, free the vector
            if self._ref_counts[index] == 0:
                self._free_vector(index)

            return True

    def get_all_vectors(self) -> NDArray[np.float32]:
        """
        Get all active vectors as a dense array.

        Returns:
            A 2D array of shape (n_vectors, dimension) containing all
            active vectors (those with reference count > 0).

        Note: This creates a copy of the data, which can be expensive
        for large datasets. Use sparingly.
        """
        with self._lock:
            active_indices = [
                i
                for i in range(self._next_index)
                if i not in self._free_indices
            ]

            if not active_indices:
                return np.zeros((0, self._dimension), dtype=np.float32)

            return self._vectors[active_indices].copy()

    def get_vectors_by_indices(
        self, indices: list[int]
    ) -> NDArray[np.float32]:
        """
        Get multiple vectors by their indices.

        Args:
            indices: List of vector indices.

        Returns:
            A 2D array of shape (len(indices), dimension).

        Raises:
            IndexError: If any index is invalid or has been freed.
        """
        with self._lock:
            for idx in indices:
                if idx < 0 or idx >= self._next_index:
                    raise IndexError(f"Invalid vector index: {idx}")
                if idx in self._free_indices:
                    raise IndexError(f"Vector index {idx} has been freed")

            return self._vectors[indices].copy()

    def _allocate_index(self) -> int:
        """
        Allocate an index for a new vector.

        Reuses freed indices if available, otherwise uses the next index.

        Returns:
            The allocated index.
        """
        if self._free_indices:
            return self._free_indices.pop()

        index = self._next_index
        self._next_index += 1
        return index

    def _free_vector(self, index: int) -> None:
        """
        Free a vector at the given index.

        Removes all metadata and marks the index as available for reuse.

        Args:
            index: The index to free.
        """
        # Remove from reference count
        del self._ref_counts[index]

        # Remove from hash map
        vector_hash = self._hash_vector(self._vectors[index])
        if vector_hash in self._vector_hashes:
            if self._vector_hashes[vector_hash] == index:
                del self._vector_hashes[vector_hash]

        # Zero out the vector (helps with debugging)
        self._vectors[index] = 0

        # Mark index as free
        self._free_indices.add(index)

    def _ensure_capacity(self, required_size: int) -> None:
        """
        Ensure the vector array has sufficient capacity.

        Grows the array by 50% when more space is needed.

        Args:
            required_size: The minimum required size.
        """
        current_capacity = self._vectors.shape[0]

        if required_size <= current_capacity:
            return

        # Grow by 50%
        new_capacity = max(required_size, int(current_capacity * 1.5))

        if self._use_mmap:
            # For memory-mapped arrays, we need to create a new file
            # and copy data over
            old_vectors = self._vectors
            old_path = self._mmap_path
            new_path = old_path.with_suffix(".tmp")

            # Create new mmap file
            new_vectors = np.memmap(
                new_path,
                dtype=np.float32,
                mode="w+",
                shape=(new_capacity, self._dimension),
            )

            # Copy old data
            new_vectors[: current_capacity] = old_vectors

            # Replace old with new
            del old_vectors  # Close old mmap
            new_path.replace(old_path)
            self._vectors = np.memmap(
                old_path,
                dtype=np.float32,
                mode="r+",
                shape=(new_capacity, self._dimension),
            )
        else:
            # For in-memory arrays, use numpy resize
            new_vectors = np.zeros((new_capacity, self._dimension), dtype=np.float32)
            new_vectors[: current_capacity] = self._vectors
            self._vectors = new_vectors

    def _hash_vector(self, vector: NDArray[np.float32]) -> int:
        """
        Compute a hash for a vector.

        Uses a stable hash function that's consistent across runs.

        Args:
            vector: The vector to hash.

        Returns:
            An integer hash value.
        """
        # Round to reduce floating point precision issues
        rounded = np.round(vector, decimals=6)
        return hash(rounded.tobytes())

    def get_statistics(self) -> Dict[str, any]:
        """
        Get statistics about the vector store.

        Returns:
            A dictionary containing:
            - unique_vectors: Number of unique vectors
            - total_references: Total number of references
            - capacity: Total capacity of the storage
            - utilization: Percentage of capacity used
            - free_indices: Number of freed indices available for reuse
            - dimension: Vector dimension
            - storage_type: "memory-mapped" or "in-memory"
        """
        with self._lock:
            capacity = self._vectors.shape[0]
            unique = self.count
            utilization = (unique / capacity * 100) if capacity > 0 else 0

            return {
                "unique_vectors": unique,
                "total_references": self.total_references,
                "capacity": capacity,
                "utilization": round(utilization, 2),
                "free_indices": len(self._free_indices),
                "dimension": self._dimension,
                "storage_type": "memory-mapped" if self._use_mmap else "in-memory",
            }

    def flush(self) -> None:
        """
        Flush changes to disk (only relevant for memory-mapped storage).

        For in-memory storage, this is a no-op.
        """
        if self._use_mmap and hasattr(self._vectors, "flush"):
            with self._lock:
                self._vectors.flush()

    def __len__(self) -> int:
        """Get the number of unique vectors stored."""
        return self.count

    def __repr__(self) -> str:
        """String representation of the VectorStore."""
        stats = self.get_statistics()
        return (
            f"VectorStore(dimension={self._dimension}, "
            f"vectors={stats['unique_vectors']}, "
            f"references={stats['total_references']}, "
            f"storage={stats['storage_type']})"
        )
