"""
Advanced VectorStore tests to reach higher coverage.

Targets missing lines in core/vector_store.py to push coverage from 82% to 90%+.
Tests the actual API: add_vector, get_vector, remove_vector, count property, get_statistics.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from uuid import uuid4

from core.vector_store import VectorStore


class TestVectorStoreInitializationEdgeCases:
    """Test VectorStore initialization edge cases (lines 59-68, 96-102)."""

    def test_invalid_dimension_zero(self):
        """Test that dimension of 0 raises ValueError (line 59-60)."""
        with pytest.raises(ValueError) as exc_info:
            VectorStore(dimension=0)
        assert "positive" in str(exc_info.value).lower()

    def test_invalid_dimension_negative(self):
        """Test that negative dimension raises ValueError (line 59-60)."""
        with pytest.raises(ValueError) as exc_info:
            VectorStore(dimension=-10)
        assert "positive" in str(exc_info.value).lower()

    def test_invalid_capacity_zero(self):
        """Test that capacity of 0 raises ValueError (line 62-64)."""
        with pytest.raises(ValueError) as exc_info:
            VectorStore(dimension=128, initial_capacity=0)
        assert "positive" in str(exc_info.value).lower()

    def test_invalid_capacity_negative(self):
        """Test that negative capacity raises ValueError (line 62-64)."""
        with pytest.raises(ValueError) as exc_info:
            VectorStore(dimension=128, initial_capacity=-100)
        assert "positive" in str(exc_info.value).lower()

    def test_mmap_without_path_raises_error(self):
        """Test that mmap=True without mmap_path raises ValueError (line 67-68)."""
        with pytest.raises(ValueError) as exc_info:
            VectorStore(dimension=128, use_mmap=True, mmap_path=None)
        assert "mmap_path" in str(exc_info.value).lower()

    def test_mmap_with_valid_path(self):
        """Test mmap with valid path succeeds (lines 96-107)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mmap_path = Path(tmpdir) / "test_vectors.mmap"

            store = VectorStore(
                dimension=128,
                use_mmap=True,
                mmap_path=mmap_path
            )

            # Should create successfully
            assert store is not None
            assert store.dimension == 128
            # Should create the file
            assert mmap_path.exists()


class TestVectorAdditionEdgeCases:
    """Test add_vector edge cases (lines 146-185)."""

    def test_add_vector_wrong_dimension(self):
        """Test adding vector with wrong dimension raises ValueError (line 153-157)."""
        store = VectorStore(dimension=128)

        wrong_vec = np.random.rand(64).astype(np.float32)

        with pytest.raises(ValueError) as exc_info:
            store.add_vector(uuid4(), wrong_vec)

        error_msg = str(exc_info.value)
        assert "dimension" in error_msg.lower()
        assert "128" in error_msg and "64" in error_msg

    def test_add_vector_wrong_shape(self):
        """Test adding 2D vector raises ValueError (line 148-151)."""
        store = VectorStore(dimension=128)

        wrong_vec = np.random.rand(10, 128).astype(np.float32)  # 2D array

        with pytest.raises(ValueError) as exc_info:
            store.add_vector(uuid4(), wrong_vec)

        assert "1D" in str(exc_info.value) or "shape" in str(exc_info.value).lower()

    def test_add_duplicate_chunk_id_raises_error(self):
        """Test adding same chunk ID twice raises ValueError (line 160-163)."""
        store = VectorStore(dimension=128)

        chunk_id = uuid4()
        vec = np.random.rand(128).astype(np.float32)

        store.add_vector(chunk_id, vec)

        # Adding same ID again should raise error
        with pytest.raises(ValueError) as exc_info:
            store.add_vector(chunk_id, vec)

        assert "already" in str(exc_info.value).lower()

    def test_add_after_capacity_exceeded(self):
        """Test adding vectors after initial capacity is exceeded (lines 177-185, 343-362)."""
        store = VectorStore(dimension=128, initial_capacity=2)

        # Add 5 vectors (exceeds initial capacity of 2)
        for i in range(5):
            chunk_id = uuid4()
            vec = np.random.rand(128).astype(np.float32)
            store.add_vector(chunk_id, vec)

        assert store.count == 5

    def test_vector_deduplication(self):
        """Test that identical vectors are deduplicated (lines 166-174)."""
        store = VectorStore(dimension=128)

        # Create identical vector
        vec = np.random.rand(128).astype(np.float32)
        id1 = uuid4()
        id2 = uuid4()

        index1 = store.add_vector(id1, vec.copy())
        index2 = store.add_vector(id2, vec.copy())

        # Should deduplicate (same index)
        assert index1 == index2
        # But both chunks should be retrievable
        assert store.get_vector(id1) is not None
        assert store.get_vector(id2) is not None


class TestVectorRemovalEdgeCases:
    """Test remove_vector edge cases (lines 225-254)."""

    def test_remove_existing_vector(self):
        """Test removing an existing vector (lines 239-254)."""
        store = VectorStore(dimension=128)

        chunk_id = uuid4()
        vec = np.random.rand(128).astype(np.float32)
        store.add_vector(chunk_id, vec)

        # Remove should return True
        result = store.remove_vector(chunk_id)
        assert result is True

        # Should be gone
        assert store.get_vector(chunk_id) is None

    def test_remove_nonexistent_vector_returns_false(self):
        """Test removing non-existent vector returns False (lines 240-242)."""
        store = VectorStore(dimension=128)

        # Should return False, not crash
        result = store.remove_vector(uuid4())
        assert result is False

    def test_remove_all_then_add_new(self):
        """Test adding vectors after removing all (index reuse, lines 303-317)."""
        store = VectorStore(dimension=128)

        # Add some vectors
        ids = [uuid4() for _ in range(5)]
        for chunk_id in ids:
            store.add_vector(chunk_id, np.random.rand(128).astype(np.float32))

        # Remove all
        for chunk_id in ids:
            store.remove_vector(chunk_id)

        assert store.count == 0

        # Add new vectors (should reuse freed indices)
        for i in range(3):
            store.add_vector(uuid4(), np.random.rand(128).astype(np.float32))

        assert store.count == 3


class TestVectorRetrieval:
    """Test get_vector and get_vector_by_index edge cases (lines 187-223)."""

    def test_get_vector_nonexistent_returns_none(self):
        """Test getting vector with invalid ID returns None (line 198-200)."""
        store = VectorStore(dimension=128)

        result = store.get_vector(uuid4())
        assert result is None

    def test_get_vector_existing(self):
        """Test getting an existing vector (lines 197-201)."""
        store = VectorStore(dimension=128)

        chunk_id = uuid4()
        vec = np.random.rand(128).astype(np.float32)
        store.add_vector(chunk_id, vec)

        retrieved = store.get_vector(chunk_id)

        # Should return a copy
        assert retrieved is not None
        assert np.allclose(retrieved, vec, atol=1e-6)
        # Should be a copy, not the same object
        assert retrieved is not vec

    def test_get_vector_by_index_invalid(self):
        """Test get_vector_by_index with invalid index raises IndexError (line 217-218)."""
        store = VectorStore(dimension=128)

        # Negative index
        with pytest.raises(IndexError):
            store.get_vector_by_index(-1)

        # Index too large
        with pytest.raises(IndexError):
            store.get_vector_by_index(1000)

    def test_get_vector_by_index_freed(self):
        """Test get_vector_by_index on freed index raises IndexError (line 220-221)."""
        store = VectorStore(dimension=128)

        chunk_id = uuid4()
        vec = np.random.rand(128).astype(np.float32)
        index = store.add_vector(chunk_id, vec)

        # Remove the vector
        store.remove_vector(chunk_id)

        # Index should now be freed
        with pytest.raises(IndexError) as exc_info:
            store.get_vector_by_index(index)

        assert "freed" in str(exc_info.value).lower()


class TestGetAllVectors:
    """Test get_all_vectors edge cases (lines 256-277)."""

    def test_get_all_vectors_empty_store(self):
        """Test get_all_vectors on empty store returns empty array (line 274-275)."""
        store = VectorStore(dimension=128)

        all_vecs = store.get_all_vectors()

        assert all_vecs.shape == (0, 128)

    def test_get_all_vectors_with_data(self):
        """Test get_all_vectors returns all active vectors (lines 267-277)."""
        store = VectorStore(dimension=128)

        # Add some vectors
        for i in range(5):
            store.add_vector(uuid4(), np.random.rand(128).astype(np.float32))

        all_vecs = store.get_all_vectors()

        assert all_vecs.shape == (5, 128)

    def test_get_all_vectors_with_some_freed(self):
        """Test get_all_vectors excludes freed vectors (lines 268-277)."""
        store = VectorStore(dimension=128)

        # Add vectors
        ids = [uuid4() for _ in range(5)]
        for chunk_id in ids:
            store.add_vector(chunk_id, np.random.rand(128).astype(np.float32))

        # Remove some
        store.remove_vector(ids[0])
        store.remove_vector(ids[2])

        all_vecs = store.get_all_vectors()

        # Should only have 3 active vectors
        assert all_vecs.shape == (3, 128)


class TestGetVectorsByIndices:
    """Test get_vectors_by_indices edge cases (lines 279-301)."""

    def test_get_vectors_by_indices_invalid(self):
        """Test get_vectors_by_indices with invalid index raises IndexError (line 295-297)."""
        store = VectorStore(dimension=128)

        with pytest.raises(IndexError):
            store.get_vectors_by_indices([0, 1000])

    def test_get_vectors_by_indices_freed(self):
        """Test get_vectors_by_indices with freed index raises IndexError (line 298-299)."""
        store = VectorStore(dimension=128)

        chunk_id = uuid4()
        index = store.add_vector(chunk_id, np.random.rand(128).astype(np.float32))

        # Remove the vector
        store.remove_vector(chunk_id)

        # Should raise error for freed index
        with pytest.raises(IndexError) as exc_info:
            store.get_vectors_by_indices([index])

        assert "freed" in str(exc_info.value).lower()


class TestStatistics:
    """Test get_statistics method."""

    def test_statistics_empty_store(self):
        """Test statistics on empty store."""
        store = VectorStore(dimension=128)

        stats = store.get_statistics()

        assert stats["unique_vectors"] == 0
        assert stats["dimension"] == 128
        assert stats["capacity"] >= 0
        assert "utilization" in stats

    def test_statistics_with_vectors(self):
        """Test statistics after adding vectors."""
        store = VectorStore(dimension=128)

        # Add vectors
        for i in range(10):
            store.add_vector(uuid4(), np.random.rand(128).astype(np.float32))

        stats = store.get_statistics()

        assert stats["unique_vectors"] == 10
        assert stats["dimension"] == 128
        assert stats["total_references"] == 10

    def test_statistics_with_deduplication(self):
        """Test statistics with deduplicated vectors."""
        store = VectorStore(dimension=128)

        # Add same vector twice
        vec = np.random.rand(128).astype(np.float32)
        store.add_vector(uuid4(), vec.copy())
        store.add_vector(uuid4(), vec.copy())

        stats = store.get_statistics()

        # Should have 1 unique vector but 2 references
        assert stats["unique_vectors"] == 1
        assert stats["total_references"] == 2


class TestMmapSpecificBehavior:
    """Test memory-mapped file specific behavior."""

    def test_mmap_creates_directory(self):
        """Test that mmap creates parent directory if needed (line 99)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mmap_path = Path(tmpdir) / "subdir" / "nested" / "vectors.mmap"

            store = VectorStore(
                dimension=128,
                use_mmap=True,
                mmap_path=mmap_path
            )

            # Should create parent directories
            assert mmap_path.parent.exists()
            assert mmap_path.exists()

    def test_mmap_resize_behavior(self):
        """Test that mmap can resize when capacity exceeded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mmap_path = Path(tmpdir) / "vectors.mmap"

            store = VectorStore(
                dimension=128,
                initial_capacity=2,
                use_mmap=True,
                mmap_path=mmap_path
            )

            # Add more vectors than initial capacity
            for i in range(5):
                store.add_vector(uuid4(), np.random.rand(128).astype(np.float32))

            assert store.count == 5


class TestHashVectorEdgeCases:
    """Test _hash_vector behavior (deduplication)."""

    def test_identical_vectors_same_hash(self):
        """Test that identical vectors get same hash (deduplication)."""
        store = VectorStore(dimension=128)

        vec = np.random.rand(128).astype(np.float32)

        id1 = uuid4()
        id2 = uuid4()

        index1 = store.add_vector(id1, vec.copy())
        index2 = store.add_vector(id2, vec.copy())

        # Should deduplicate to same index
        assert index1 == index2


class TestCountAndTotalReferences:
    """Test count and total_references properties."""

    def test_count_property(self):
        """Test count property."""
        store = VectorStore(dimension=128)

        assert store.count == 0

        # Add vectors
        for i in range(5):
            store.add_vector(uuid4(), np.random.rand(128).astype(np.float32))

        assert store.count == 5

    def test_total_references_property(self):
        """Test total_references property."""
        store = VectorStore(dimension=128)

        assert store.total_references == 0

        # Add same vector twice (deduplication)
        vec = np.random.rand(128).astype(np.float32)
        store.add_vector(uuid4(), vec.copy())
        store.add_vector(uuid4(), vec.copy())

        # Should have 2 references
        assert store.total_references == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
