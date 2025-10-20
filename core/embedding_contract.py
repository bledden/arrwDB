"""
Embedding Contract for enforcing dimensional consistency.

This module provides the LibraryEmbeddingContract which ensures that once
a library is initialized with a specific embedding dimension, all subsequent
vectors must match that dimension. This prevents dimension mismatches that
would cause runtime errors during vector operations.
"""

import numpy as np
from typing import List, Optional
from numpy.typing import NDArray


class LibraryEmbeddingContract:
    """
    Immutable contract that enforces embedding dimension consistency.

    Once initialized with a dimension, all vectors must match that dimension.
    This contract is created when the first vector is added to a library and
    cannot be changed afterward.

    Thread-Safety: This class is thread-safe for read operations after initialization.
    The dimension is set once and never modified, making it safe for concurrent reads.
    """

    def __init__(self, expected_dimension: int):
        """
        Initialize the contract with the expected embedding dimension.

        Args:
            expected_dimension: The required dimension for all vectors.
                Must be a positive integer.

        Raises:
            ValueError: If expected_dimension is not positive.
        """
        if expected_dimension <= 0:
            raise ValueError(
                f"Expected dimension must be positive, got {expected_dimension}"
            )

        self._dimension = expected_dimension

    @property
    def dimension(self) -> int:
        """Get the contracted embedding dimension."""
        return self._dimension

    def validate_vector(self, vector: List[float]) -> NDArray[np.float32]:
        """
        Validate and normalize a vector according to the contract.

        This method:
        1. Checks that the vector matches the expected dimension
        2. Converts to numpy array with float32 dtype for consistency
        3. Validates that there are no NaN or Inf values
        4. Normalizes the vector to unit length (L2 norm = 1)

        Args:
            vector: The vector to validate as a list of floats.

        Returns:
            A normalized numpy array with dtype float32.

        Raises:
            ValueError: If the vector dimension doesn't match the contract,
                or if the vector contains invalid values (NaN, Inf),
                or if the vector is the zero vector (cannot normalize).
        """
        # Convert to numpy array
        arr = np.array(vector, dtype=np.float32)

        # Check dimension
        if len(arr) != self._dimension:
            raise ValueError(
                f"Vector dimension mismatch: expected {self._dimension}, "
                f"got {len(arr)}"
            )

        # Check for invalid values
        if np.any(np.isnan(arr)):
            raise ValueError("Vector contains NaN values")

        if np.any(np.isinf(arr)):
            raise ValueError("Vector contains Inf values")

        # Normalize to unit length for cosine similarity
        norm = np.linalg.norm(arr)

        if norm == 0:
            raise ValueError(
                "Cannot normalize zero vector. Vector must have non-zero magnitude."
            )

        normalized = arr / norm

        return normalized

    def validate_vectors_batch(
        self, vectors: List[List[float]]
    ) -> NDArray[np.float32]:
        """
        Validate and normalize a batch of vectors.

        More efficient than calling validate_vector in a loop when processing
        multiple vectors, as it performs batch operations where possible.

        Args:
            vectors: List of vectors to validate.

        Returns:
            A 2D numpy array of shape (n_vectors, dimension) with dtype float32,
            where each row is a normalized vector.

        Raises:
            ValueError: If any vector fails validation.
        """
        if not vectors:
            raise ValueError("Cannot validate empty batch of vectors")

        # Convert all to numpy array at once
        try:
            arr = np.array(vectors, dtype=np.float32)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Failed to convert vectors to numpy array: {e}")

        # Check shape
        if arr.ndim != 2:
            raise ValueError(
                f"Expected 2D array of vectors, got shape {arr.shape}"
            )

        if arr.shape[1] != self._dimension:
            raise ValueError(
                f"Vector dimension mismatch: expected {self._dimension}, "
                f"got {arr.shape[1]}"
            )

        # Check for invalid values in entire batch
        if np.any(np.isnan(arr)):
            raise ValueError("Batch contains NaN values")

        if np.any(np.isinf(arr)):
            raise ValueError("Batch contains Inf values")

        # Normalize all vectors
        norms = np.linalg.norm(arr, axis=1, keepdims=True)

        # Check for zero vectors
        if np.any(norms == 0):
            zero_indices = np.where(norms.squeeze() == 0)[0]
            raise ValueError(
                f"Cannot normalize zero vectors at indices: {zero_indices.tolist()}"
            )

        normalized = arr / norms

        return normalized

    def __repr__(self) -> str:
        """String representation of the contract."""
        return f"LibraryEmbeddingContract(dimension={self._dimension})"

    def __eq__(self, other: object) -> bool:
        """Check equality based on dimension."""
        if not isinstance(other, LibraryEmbeddingContract):
            return NotImplemented
        return self._dimension == other._dimension

    def __hash__(self) -> int:
        """Hash based on dimension for use in sets/dicts."""
        return hash(self._dimension)
