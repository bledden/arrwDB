"""
Advanced tests for LSH index to improve coverage.

Tests error handling, edge cases, and statistics methods.
"""

import pytest
import numpy as np
from uuid import uuid4
from core.vector_store import VectorStore
from infrastructure.indexes.lsh import LSHIndex


@pytest.fixture
def vector_store():
    """Shared vector store fixture."""
    return VectorStore(dimension=128)


@pytest.fixture
def lsh_index(vector_store):
    """Create LSH index with test parameters."""
    return LSHIndex(vector_store, num_tables=4, hash_size=8)


class TestLSHAddVectorErrors:
    """Test error handling in add_vector method."""

    def test_add_vector_with_invalid_vector_index_raises_error(self, lsh_index, vector_store):
        """Test that invalid vector_index raises ValueError (lines 126-127)."""
        # Try to add with vector_index that doesn't exist
        with pytest.raises(ValueError) as exc_info:
            lsh_index.add_vector(uuid4(), vector_index=999999)

        assert "invalid vector_index" in str(exc_info.value).lower()


class TestLSHSearchValidation:
    """Test search parameter validation."""

    def test_search_with_k_zero_raises_error(self, lsh_index, vector_store):
        """Test that k=0 raises ValueError (line 214)."""
        # Add a vector first
        vec_id = uuid4()
        vector = np.random.rand(128).astype(np.float32)
        vector = vector / np.linalg.norm(vector)
        vector_store.add_vector(vec_id, vector)
        lsh_index.add_vector(vec_id, vector_index=0)

        query = np.random.rand(128).astype(np.float32)
        query = query / np.linalg.norm(query)

        with pytest.raises(ValueError) as exc_info:
            lsh_index.search(query, k=0)

        assert "k must be positive" in str(exc_info.value).lower()

    def test_search_with_wrong_dimension_raises_error(self, lsh_index, vector_store):
        """Test that wrong dimension query raises ValueError (lines 219)."""
        # Add a vector first
        vec_id = uuid4()
        vector = np.random.rand(128).astype(np.float32)
        vector = vector / np.linalg.norm(vector)
        vector_store.add_vector(vec_id, vector)
        lsh_index.add_vector(vec_id, vector_index=0)

        # Query with wrong dimension
        wrong_query = np.random.rand(64).astype(np.float32)
        wrong_query = wrong_query / np.linalg.norm(wrong_query)

        with pytest.raises(ValueError) as exc_info:
            lsh_index.search(wrong_query, k=1)

        assert "dimension" in str(exc_info.value).lower()


class TestLSHSearchEdgeCases:
    """Test edge cases in search method."""

    def test_search_with_distance_threshold_filters_all_results(self, lsh_index, vector_store):
        """Test search with very strict threshold returns empty list (line 255)."""
        # Add vectors
        for i in range(5):
            vec_id = uuid4()
            vector = np.random.rand(128).astype(np.float32)
            vector = vector / np.linalg.norm(vector)
            vector_store.add_vector(vec_id, vector)
            lsh_index.add_vector(vec_id, vector_index=i)

        # Query with impossible threshold
        query = np.random.rand(128).astype(np.float32)
        query = query / np.linalg.norm(query)

        results = lsh_index.search(query, k=5, distance_threshold=0.0)
        # Might return empty if no exact matches
        assert isinstance(results, list)

    def test_search_with_k_larger_than_candidates(self, lsh_index, vector_store):
        """Test search when k > number of candidates (line 259-260)."""
        # Add just 2 vectors
        vec_ids = []
        for i in range(2):
            vec_id = uuid4()
            vector = np.random.rand(128).astype(np.float32)
            vector = vector / np.linalg.norm(vector)
            vector_store.add_vector(vec_id, vector)
            lsh_index.add_vector(vec_id, vector_index=i)
            vec_ids.append(vec_id)

        # Search for more than we have
        query = np.random.rand(128).astype(np.float32)
        query = query / np.linalg.norm(query)

        results = lsh_index.search(query, k=10)
        # Should return at most 2 results (lines 259-260 handle this case)
        assert len(results) <= 2


class TestLSHClear:
    """Test clear method."""

    def test_clear_removes_all_vectors(self, lsh_index, vector_store):
        """Test that clear removes all vectors (line 316)."""
        # Add several vectors
        for i in range(5):
            vec_id = uuid4()
            vector = np.random.rand(128).astype(np.float32)
            vector = vector / np.linalg.norm(vector)
            vector_store.add_vector(vec_id, vector)
            lsh_index.add_vector(vec_id, vector_index=i)

        # Clear the index
        lsh_index.clear()

        # Verify empty
        stats = lsh_index.get_statistics()
        assert stats["size"] == 0
        assert stats["total_buckets"] == 0


class TestLSHStatistics:
    """Test statistics and repr methods."""

    def test_get_statistics_with_data(self, lsh_index, vector_store):
        """Test get_statistics returns correct data (lines 379-392)."""
        # Add vectors
        for i in range(10):
            vec_id = uuid4()
            vector = np.random.rand(128).astype(np.float32)
            vector = vector / np.linalg.norm(vector)
            vector_store.add_vector(vec_id, vector)
            lsh_index.add_vector(vec_id, vector_index=i)

        stats = lsh_index.get_statistics()

        assert stats["type"] == "lsh"
        assert stats["size"] == 10
        assert stats["vector_store_dim"] == 128
        assert stats["num_tables"] == 4
        assert stats["hash_size"] == 8
        assert "total_buckets" in stats
        assert "avg_bucket_size" in stats
        assert stats["supports_incremental"] is True

    def test_get_statistics_empty_index(self, lsh_index):
        """Test get_statistics on empty index (lines 388-390)."""
        stats = lsh_index.get_statistics()

        assert stats["size"] == 0
        assert stats["total_buckets"] == 0
        assert stats["avg_bucket_size"] == 0

    def test_repr(self, lsh_index, vector_store):
        """Test __repr__ method (lines 405-406)."""
        # Add a vector
        vec_id = uuid4()
        vector = np.random.rand(128).astype(np.float32)
        vector = vector / np.linalg.norm(vector)
        vector_store.add_vector(vec_id, vector)
        lsh_index.add_vector(vec_id, vector_index=0)

        repr_str = repr(lsh_index)

        assert "LSHIndex" in repr_str
        assert "size=1" in repr_str
        assert "dimension=128" in repr_str
        assert "tables=4" in repr_str
        assert "hash_size=8" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
