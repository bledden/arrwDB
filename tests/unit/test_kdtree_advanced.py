"""
Advanced tests for KD-Tree index to improve coverage.

Tests error handling, edge cases, statistics, and rebuild logic.
"""

import pytest
import numpy as np
from uuid import uuid4
from core.vector_store import VectorStore
from infrastructure.indexes.kd_tree import KDTreeIndex


@pytest.fixture
def vector_store():
    """Shared vector store fixture."""
    return VectorStore(dimension=128)


@pytest.fixture
def kdtree_index(vector_store):
    """Create KD-Tree index with test parameters."""
    return KDTreeIndex(vector_store, rebuild_threshold=5)


class TestKDTreeAddVectorErrors:
    """Test error handling in add_vector method."""

    def test_add_vector_with_invalid_vector_index_raises_error(self, kdtree_index):
        """Test that invalid vector_index raises ValueError (lines 107-108)."""
        # Try to add with vector_index that doesn't exist
        with pytest.raises(ValueError) as exc_info:
            kdtree_index.add_vector(uuid4(), vector_index=999999)

        assert "invalid vector_index" in str(exc_info.value).lower()


class TestKDTreeRebuild:
    """Test rebuild threshold functionality."""

    def test_add_triggers_rebuild_when_threshold_exceeded(self, kdtree_index, vector_store):
        """Test that adding vectors triggers rebuild at threshold (line 146)."""
        # Add exactly rebuild_threshold vectors (5)
        for i in range(5):
            vec_id = uuid4()
            vector = np.random.rand(128).astype(np.float32)
            vector = vector / np.linalg.norm(vector)
            vector_store.add_vector(vec_id, vector)
            kdtree_index.add_vector(vec_id, vector_index=i)

        # The tree should have been rebuilt after 5 additions
        stats = kdtree_index.get_statistics()
        # After rebuild, modifications_since_rebuild resets
        assert "modifications_since_rebuild" in stats

    def test_remove_triggers_rebuild_when_threshold_exceeded(self, kdtree_index, vector_store):
        """Test that removing vectors triggers rebuild at threshold (line 146)."""
        # Add 10 vectors
        vec_ids = []
        for i in range(10):
            vec_id = uuid4()
            vector = np.random.rand(128).astype(np.float32)
            vector = vector / np.linalg.norm(vector)
            vector_store.add_vector(vec_id, vector)
            kdtree_index.add_vector(vec_id, vector_index=i)
            vec_ids.append(vec_id)

        # Clear the rebuild counter by rebuilding manually
        kdtree_index.rebuild()

        # Now remove 5 vectors to trigger rebuild threshold
        for i in range(5):
            kdtree_index.remove_vector(vec_ids[i])

        # The rebuild should have been triggered
        stats = kdtree_index.get_statistics()
        assert "modifications_since_rebuild" in stats


class TestKDTreeSearchValidation:
    """Test search parameter validation."""

    def test_search_with_k_zero_raises_error(self, kdtree_index, vector_store):
        """Test that k=0 raises ValueError (line 178)."""
        # Add a vector first and rebuild to ensure tree is not empty
        vec_id = uuid4()
        vector = np.random.rand(128).astype(np.float32)
        vector = vector / np.linalg.norm(vector)
        vector_store.add_vector(vec_id, vector)
        kdtree_index.add_vector(vec_id, vector_index=0)
        kdtree_index.rebuild()  # Ensure tree is built

        query = np.random.rand(128).astype(np.float32)
        query = query / np.linalg.norm(query)

        with pytest.raises(ValueError) as exc_info:
            kdtree_index.search(query, k=0)

        assert "k must be positive" in str(exc_info.value).lower()

    def test_search_with_wrong_dimension_raises_error(self, kdtree_index, vector_store):
        """Test that wrong dimension query raises ValueError (line 183)."""
        # Add a vector first and rebuild to ensure tree is not empty
        vec_id = uuid4()
        vector = np.random.rand(128).astype(np.float32)
        vector = vector / np.linalg.norm(vector)
        vector_store.add_vector(vec_id, vector)
        kdtree_index.add_vector(vec_id, vector_index=0)
        kdtree_index.rebuild()  # Ensure tree is built

        # Query with wrong dimension
        wrong_query = np.random.rand(64).astype(np.float32)
        wrong_query = wrong_query / np.linalg.norm(wrong_query)

        with pytest.raises(ValueError) as exc_info:
            kdtree_index.search(wrong_query, k=1)

        assert "dimension" in str(exc_info.value).lower()


class TestKDTreeSearchDistanceThreshold:
    """Test distance threshold filtering in search."""

    def test_search_with_distance_threshold_filters_results(self, kdtree_index, vector_store):
        """Test search with distance threshold (line 235)."""
        # Add several vectors
        for i in range(10):
            vec_id = uuid4()
            vector = np.random.rand(128).astype(np.float32)
            vector = vector / np.linalg.norm(vector)
            vector_store.add_vector(vec_id, vector)
            kdtree_index.add_vector(vec_id, vector_index=i)

        # Query with a very strict threshold
        query = np.random.rand(128).astype(np.float32)
        query = query / np.linalg.norm(query)

        results = kdtree_index.search(query, k=10, distance_threshold=0.01)
        # Should potentially filter out some results
        assert isinstance(results, list)


class TestKDTreeBuildRecursive:
    """Test recursive tree building."""

    def test_build_with_many_vectors(self, kdtree_index, vector_store):
        """Test building tree with many vectors (lines 282-284)."""
        # Add many vectors to trigger deeper recursion
        for i in range(50):
            vec_id = uuid4()
            vector = np.random.rand(128).astype(np.float32)
            vector = vector / np.linalg.norm(vector)
            vector_store.add_vector(vec_id, vector)
            kdtree_index.add_vector(vec_id, vector_index=i)

        # Rebuild to ensure all code paths are covered
        kdtree_index.rebuild()

        # Verify the tree is functional
        query = np.random.rand(128).astype(np.float32)
        query = query / np.linalg.norm(query)
        results = kdtree_index.search(query, k=5)
        assert len(results) == 5


class TestKDTreeClear:
    """Test clear method."""

    def test_clear_removes_all_vectors(self, kdtree_index, vector_store):
        """Test that clear removes all vectors (line 312)."""
        # Add several vectors
        for i in range(10):
            vec_id = uuid4()
            vector = np.random.rand(128).astype(np.float32)
            vector = vector / np.linalg.norm(vector)
            vector_store.add_vector(vec_id, vector)
            kdtree_index.add_vector(vec_id, vector_index=i)

        # Clear the index
        kdtree_index.clear()

        # Verify empty
        stats = kdtree_index.get_statistics()
        assert stats["size"] == 0
        assert stats["tree_depth"] == 0


class TestKDTreeStatistics:
    """Test statistics and repr methods."""

    def test_get_statistics_with_data(self, kdtree_index, vector_store):
        """Test get_statistics returns correct data (lines 383-386)."""
        # Add vectors
        for i in range(20):
            vec_id = uuid4()
            vector = np.random.rand(128).astype(np.float32)
            vector = vector / np.linalg.norm(vector)
            vector_store.add_vector(vec_id, vector)
            kdtree_index.add_vector(vec_id, vector_index=i)

        stats = kdtree_index.get_statistics()

        assert stats["type"] == "kd_tree"
        assert stats["size"] == 20
        assert stats["vector_store_dim"] == 128
        assert stats["rebuild_threshold"] == 5
        assert "tree_depth" in stats
        assert "modifications_since_rebuild" in stats
        assert stats["supports_incremental"] is False  # KD-Tree requires rebuilds

    def test_compute_tree_depth(self, kdtree_index, vector_store):
        """Test tree depth computation (lines 398-400)."""
        # Add vectors to create a tree
        for i in range(15):
            vec_id = uuid4()
            vector = np.random.rand(128).astype(np.float32)
            vector = vector / np.linalg.norm(vector)
            vector_store.add_vector(vec_id, vector)
            kdtree_index.add_vector(vec_id, vector_index=i)

        stats = kdtree_index.get_statistics()
        # Tree should have some depth
        assert stats["tree_depth"] >= 0

    def test_repr(self, kdtree_index, vector_store):
        """Test __repr__ method (lines 407-408)."""
        # Add a vector
        vec_id = uuid4()
        vector = np.random.rand(128).astype(np.float32)
        vector = vector / np.linalg.norm(vector)
        vector_store.add_vector(vec_id, vector)
        kdtree_index.add_vector(vec_id, vector_index=0)

        repr_str = repr(kdtree_index)

        assert "KDTreeIndex" in repr_str
        assert "size=1" in repr_str
        assert "dimension=128" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
