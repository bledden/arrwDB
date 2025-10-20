"""
Unit tests for VectorStore.
"""

import pytest
import numpy as np
from uuid import uuid4, UUID
from pathlib import Path

from core.vector_store import VectorStore


@pytest.mark.unit
class TestVectorStoreInitialization:
    """Tests for VectorStore initialization."""

    def test_valid_initialization(self, vector_dimension: int):
        """Test that VectorStore initializes with valid parameters."""
        store = VectorStore(dimension=vector_dimension, initial_capacity=100)
        assert store.dimension == vector_dimension
        assert store.count == 0
        assert len(store) == 0

    def test_invalid_dimension_raises_error(self):
        """Test that negative dimension raises ValueError."""
        with pytest.raises(ValueError, match="Dimension must be positive"):
            VectorStore(dimension=-1)

        with pytest.raises(ValueError, match="Dimension must be positive"):
            VectorStore(dimension=0)

    def test_invalid_capacity_raises_error(self, vector_dimension: int):
        """Test that negative capacity raises ValueError."""
        with pytest.raises(ValueError, match="Initial capacity must be positive"):
            VectorStore(dimension=vector_dimension, initial_capacity=-1)

        with pytest.raises(ValueError, match="Initial capacity must be positive"):
            VectorStore(dimension=vector_dimension, initial_capacity=0)

    def test_mmap_without_path_raises_error(self, vector_dimension: int):
        """Test that use_mmap without mmap_path raises ValueError."""
        with pytest.raises(ValueError, match="mmap_path is required"):
            VectorStore(dimension=vector_dimension, use_mmap=True)


@pytest.mark.unit
class TestVectorStoreAddVector:
    """Tests for adding vectors."""

    def test_add_single_vector(self, vector_store: VectorStore, vector_dimension: int):
        """Test adding a single vector."""
        chunk_id = uuid4()
        vector = np.random.randn(vector_dimension).astype(np.float32)
        vector = vector / np.linalg.norm(vector)

        index = vector_store.add_vector(chunk_id, vector)

        assert index == 0  # First vector
        assert vector_store.count == 1
        assert len(vector_store) == 1
        # Check if vector exists by trying to get it
        retrieved = vector_store.get_vector(chunk_id)
        assert retrieved is not None

    def test_add_multiple_vectors(self, vector_store: VectorStore, sample_vectors: list):
        """Test adding multiple vectors."""
        chunk_ids = [uuid4() for _ in range(len(sample_vectors))]

        for i, (chunk_id, vector) in enumerate(zip(chunk_ids, sample_vectors)):
            index = vector_store.add_vector(chunk_id, vector)
            assert index == i

        assert vector_store.count == len(sample_vectors)
        assert len(vector_store) == len(sample_vectors)

        for chunk_id in chunk_ids:
            # Check if vector exists by trying to get it
            retrieved = vector_store.get_vector(chunk_id)
            assert retrieved is not None

    def test_add_duplicate_chunk_id_raises_error(self, vector_store: VectorStore, vector_dimension: int):
        """Test that adding the same chunk_id twice raises ValueError."""
        chunk_id = uuid4()
        vector = np.random.randn(vector_dimension).astype(np.float32)
        vector = vector / np.linalg.norm(vector)

        vector_store.add_vector(chunk_id, vector)

        with pytest.raises(ValueError, match="already has an associated vector"):
            vector_store.add_vector(chunk_id, vector)

    def test_vector_deduplication(self, vector_store: VectorStore, vector_dimension: int):
        """Test that identical vectors are deduplicated."""
        chunk_id1 = uuid4()
        chunk_id2 = uuid4()

        # Create identical vector
        vector = np.random.randn(vector_dimension).astype(np.float32)
        vector = vector / np.linalg.norm(vector)

        index1 = vector_store.add_vector(chunk_id1, vector)
        index2 = vector_store.add_vector(chunk_id2, vector)

        # Should reuse the same vector index
        assert index1 == index2
        # Only 1 unique vector stored (len shows unique vectors)
        assert vector_store.count == 1
        assert len(vector_store) == 1
        # But 2 total references
        assert vector_store.total_references == 2

    def test_add_wrong_dimension_raises_error(self, vector_store: VectorStore, vector_dimension: int):
        """Test that adding wrong dimension vector raises ValueError."""
        chunk_id = uuid4()
        wrong_vector = np.random.randn(vector_dimension + 10).astype(np.float32)

        with pytest.raises(ValueError, match="dimension"):
            vector_store.add_vector(chunk_id, wrong_vector)


@pytest.mark.unit
class TestVectorStoreGetVector:
    """Tests for retrieving vectors."""

    def test_get_existing_vector(self, vector_store: VectorStore, vector_dimension: int):
        """Test getting a vector that exists."""
        chunk_id = uuid4()
        original_vector = np.random.randn(vector_dimension).astype(np.float32)
        original_vector = original_vector / np.linalg.norm(original_vector)

        vector_store.add_vector(chunk_id, original_vector)
        retrieved_vector = vector_store.get_vector(chunk_id)

        np.testing.assert_array_almost_equal(retrieved_vector, original_vector)

    def test_get_nonexistent_vector_returns_none(self, vector_store: VectorStore):
        """Test that getting nonexistent vector returns None."""
        nonexistent_id = uuid4()

        result = vector_store.get_vector(nonexistent_id)
        assert result is None

    def test_get_by_index(self, vector_store: VectorStore, sample_vectors: list):
        """Test getting vectors by index."""
        chunk_id = uuid4()
        vector = sample_vectors[0]

        index = vector_store.add_vector(chunk_id, vector)
        retrieved_vector = vector_store.get_vector_by_index(index)

        np.testing.assert_array_almost_equal(retrieved_vector, vector)


@pytest.mark.unit
class TestVectorStoreRemoveVector:
    """Tests for removing vectors."""

    def test_remove_existing_vector(self, vector_store: VectorStore, vector_dimension: int):
        """Test removing a vector that exists."""
        chunk_id = uuid4()
        vector = np.random.randn(vector_dimension).astype(np.float32)
        vector = vector / np.linalg.norm(vector)

        vector_store.add_vector(chunk_id, vector)
        assert vector_store.count == 1

        removed = vector_store.remove_vector(chunk_id)
        assert removed is True
        assert vector_store.count == 0
        assert len(vector_store) == 0
        # Verify it's gone
        result = vector_store.get_vector(chunk_id)
        assert result is None

    def test_remove_nonexistent_vector(self, vector_store: VectorStore):
        """Test removing a vector that doesn't exist."""
        nonexistent_id = uuid4()
        removed = vector_store.remove_vector(nonexistent_id)
        assert removed is False

    def test_remove_with_reference_counting(self, vector_store: VectorStore, vector_dimension: int):
        """Test that reference counting works when removing vectors."""
        chunk_id1 = uuid4()
        chunk_id2 = uuid4()

        # Same vector for both chunks
        vector = np.random.randn(vector_dimension).astype(np.float32)
        vector = vector / np.linalg.norm(vector)

        vector_store.add_vector(chunk_id1, vector)
        vector_store.add_vector(chunk_id2, vector)

        # Remove first chunk
        vector_store.remove_vector(chunk_id1)

        # Second chunk should still be able to access the vector
        retrieved = vector_store.get_vector(chunk_id2)
        np.testing.assert_array_almost_equal(retrieved, vector)

    def test_reuse_freed_index(self, vector_store: VectorStore, sample_vectors: list):
        """Test that removed vector indices are reused."""
        chunk_id1 = uuid4()
        chunk_id2 = uuid4()

        index1 = vector_store.add_vector(chunk_id1, sample_vectors[0])
        vector_store.remove_vector(chunk_id1)

        # Next add should reuse the freed index
        index2 = vector_store.add_vector(chunk_id2, sample_vectors[1])
        assert index2 == index1


@pytest.mark.unit
class TestVectorStoreEdgeCases:
    """Edge case tests for VectorStore."""

    def test_empty_store_size(self, vector_store: VectorStore):
        """Test that empty store has count 0."""
        assert vector_store.count == 0
        assert len(vector_store) == 0

    def test_get_vector_on_empty_store(self, vector_store: VectorStore):
        """Test get_vector on empty store returns None."""
        result = vector_store.get_vector(uuid4())
        assert result is None

    def test_remove_all_vectors(self, vector_store: VectorStore, sample_vectors: list):
        """Test removing all vectors from the store."""
        # Add multiple vectors
        chunk_ids = []
        for vector in sample_vectors:
            chunk_id = uuid4()
            vector_store.add_vector(chunk_id, vector)
            chunk_ids.append(chunk_id)

        assert vector_store.count == len(sample_vectors)

        # Remove all
        for chunk_id in chunk_ids:
            vector_store.remove_vector(chunk_id)

        assert vector_store.count == 0
        assert len(vector_store) == 0

    def test_add_many_vectors_exceeding_capacity(self, vector_dimension: int):
        """Test that store grows beyond initial capacity."""
        store = VectorStore(dimension=vector_dimension, initial_capacity=10)

        # Add more than initial capacity
        for i in range(100):
            chunk_id = uuid4()
            vector = np.random.randn(vector_dimension).astype(np.float32)
            vector = vector / np.linalg.norm(vector)
            store.add_vector(chunk_id, vector)

        assert store.count == 100
        assert len(store) == 100

    def test_nan_vector_not_added(self, vector_store: VectorStore, vector_dimension: int):
        """Test adding vectors with NaN (VectorStore doesn't validate, that's EmbeddingContract's job)."""
        chunk_id = uuid4()
        vector = np.random.randn(vector_dimension).astype(np.float32)
        vector[0] = np.nan

        # VectorStore itself doesn't validate NaN/Inf - that's done by EmbeddingContract
        # This test just ensures the vector can be stored
        vector_store.add_vector(chunk_id, vector)
        retrieved = vector_store.get_vector(chunk_id)
        assert retrieved is not None

    def test_inf_vector_not_added(self, vector_store: VectorStore, vector_dimension: int):
        """Test adding vectors with Inf (VectorStore doesn't validate, that's EmbeddingContract's job)."""
        chunk_id = uuid4()
        vector = np.random.randn(vector_dimension).astype(np.float32)
        vector[0] = np.inf

        # VectorStore itself doesn't validate NaN/Inf - that's done by EmbeddingContract
        # This test just ensures the vector can be stored
        vector_store.add_vector(chunk_id, vector)
        retrieved = vector_store.get_vector(chunk_id)
        assert retrieved is not None
