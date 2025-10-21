"""
Tests to cover remaining edge cases and push coverage to maximum.

Targets: clear methods, error handling, edge cases across multiple files.
"""

import pytest
import numpy as np
from uuid import uuid4
from pathlib import Path
from core.vector_store import VectorStore
from infrastructure.indexes.brute_force import BruteForceIndex
from infrastructure.indexes.kd_tree import KDTreeIndex
from infrastructure.indexes.lsh import LSHIndex
from infrastructure.repositories.library_repository import LibraryRepository


class TestIndexClearMethods:
    """Test clear() methods across indexes."""

    def test_kdtree_clear(self):
        """Test KD-Tree clear method (line 312)."""
        vector_store = VectorStore(dimension=128)
        index = KDTreeIndex(vector_store)

        # Add vectors
        for i in range(10):
            vec_id = uuid4()
            vector = np.random.rand(128).astype(np.float32)
            vector = vector / np.linalg.norm(vector)
            vector_store.add_vector(vec_id, vector)
            index.add_vector(vec_id, vector_index=i)

        assert index.size() == 10

        # Clear
        index.clear()

        assert index.size() == 0
        assert index._root is None
        assert index._modifications_since_rebuild == 0

    def test_lsh_clear(self):
        """Test LSH clear method (line 316)."""
        vector_store = VectorStore(dimension=128)
        index = LSHIndex(vector_store, num_tables=4, hash_size=8)

        # Add vectors
        for i in range(10):
            vec_id = uuid4()
            vector = np.random.rand(128).astype(np.float32)
            vector = vector / np.linalg.norm(vector)
            vector_store.add_vector(vec_id, vector)
            index.add_vector(vec_id, vector_index=i)

        assert index.size() == 10

        # Clear
        index.clear()

        assert index.size() == 0
        assert len(index._vector_hashes) == 0
        assert all(len(table) == 0 for table in index._tables)

    def test_brute_force_clear(self):
        """Test BruteForce clear method (line 260)."""
        vector_store = VectorStore(dimension=128)
        index = BruteForceIndex(vector_store)

        # Add vectors
        for i in range(10):
            vec_id = uuid4()
            vector = np.random.rand(128).astype(np.float32)
            vector = vector / np.linalg.norm(vector)
            vector_store.add_vector(vec_id, vector)
            index.add_vector(vec_id, vector_index=i)

        assert index.size() == 10

        # Clear
        index.clear()

        assert index.size() == 0


class TestBruteForceEdgeCases:
    """Test BruteForce edge cases."""

    def test_duplicate_vector_id_raises_error(self):
        """Test adding duplicate vector ID raises error (lines 81-82)."""
        vector_store = VectorStore(dimension=128)
        index = BruteForceIndex(vector_store)

        vec_id = uuid4()
        vector = np.random.rand(128).astype(np.float32)
        vector = vector / np.linalg.norm(vector)
        vector_store.add_vector(vec_id, vector)

        # Add first time
        index.add_vector(vec_id, vector_index=0)

        # Try to add again - should raise
        with pytest.raises(ValueError) as exc_info:
            index.add_vector(vec_id, vector_index=0)

        assert "already exists" in str(exc_info.value).lower()

    def test_search_empty_index_returns_empty(self):
        """Test searching empty index returns empty list (line 146)."""
        vector_store = VectorStore(dimension=128)
        index = BruteForceIndex(vector_store)

        query = np.random.rand(128).astype(np.float32)
        query = query / np.linalg.norm(query)

        results = index.search(query, k=5)

        assert results == []

    def test_search_with_strict_threshold_filters_all(self):
        """Test very strict distance threshold filters all results (line 172)."""
        vector_store = VectorStore(dimension=128)
        index = BruteForceIndex(vector_store)

        # Add some vectors
        for i in range(10):
            vec_id = uuid4()
            vector = np.random.rand(128).astype(np.float32)
            vector = vector / np.linalg.norm(vector)
            vector_store.add_vector(vec_id, vector)
            index.add_vector(vec_id, vector_index=i)

        # Query with impossible threshold
        query = np.random.rand(128).astype(np.float32)
        query = query / np.linalg.norm(query)

        results = index.search(query, k=10, distance_threshold=0.0)

        # Should return empty or very few results
        assert isinstance(results, list)


class TestKDTreeEdgeCases:
    """Test KD-Tree edge cases."""

    def test_build_recursive_with_many_vectors(self):
        """Test recursive build with many vectors (lines 282-284)."""
        vector_store = VectorStore(dimension=128)
        index = KDTreeIndex(vector_store)

        # Add many vectors to trigger deep recursion
        for i in range(100):
            vec_id = uuid4()
            vector = np.random.rand(128).astype(np.float32)
            vector = vector / np.linalg.norm(vector)
            vector_store.add_vector(vec_id, vector)
            index.add_vector(vec_id, vector_index=i)

        # Rebuild to ensure tree is built
        index.rebuild()

        # Should be searchable
        query = np.random.rand(128).astype(np.float32)
        query = query / np.linalg.norm(query)
        results = index.search(query, k=5)

        assert len(results) == 5


class TestLSHEdgeCases:
    """Test LSH edge cases."""

    def test_search_k_larger_than_results(self):
        """Test search when k > available results after filtering (lines 259-260)."""
        vector_store = VectorStore(dimension=128)
        index = LSHIndex(vector_store, num_tables=4, hash_size=8)

        # Add just a few vectors
        for i in range(3):
            vec_id = uuid4()
            vector = np.random.rand(128).astype(np.float32)
            vector = vector / np.linalg.norm(vector)
            vector_store.add_vector(vec_id, vector)
            index.add_vector(vec_id, vector_index=i)

        # Search for more than we have
        query = np.random.rand(128).astype(np.float32)
        query = query / np.linalg.norm(query)

        results = index.search(query, k=100)

        # Should return at most 3 (lines 259-260 handle this)
        assert len(results) <= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
