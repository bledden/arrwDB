"""
Advanced tests for HNSW index to push coverage from 90% to 95%+.

Targets specific uncovered lines in infrastructure/indexes/hnsw.py.
"""

import pytest
import numpy as np
from uuid import uuid4
from core.vector_store import VectorStore
from infrastructure.indexes.hnsw import HNSWIndex


@pytest.fixture
def vector_store():
    """Shared vector store fixture."""
    return VectorStore(dimension=128)


@pytest.fixture
def hnsw_index(vector_store):
    """Create HNSW index with test parameters."""
    return HNSWIndex(vector_store, M=16, ef_construction=200, ef_search=50)


class TestHNSWAddVectorErrors:
    """Test error handling in add_vector method."""

    def test_add_vector_with_invalid_vector_index_raises_error(self, hnsw_index):
        """Test that invalid vector_index raises ValueError (lines 148-149)."""
        # Try to add with vector_index that doesn't exist
        with pytest.raises(ValueError) as exc_info:
            hnsw_index.add_vector(uuid4(), vector_index=999999)

        assert "invalid vector_index" in str(exc_info.value).lower()


class TestHNSWRemoveVector:
    """Test remove_vector edge cases."""

    def test_remove_vector_updates_neighbor_references(self, hnsw_index, vector_store):
        """Test that removing a vector updates its neighbors (lines 367-368)."""
        # Add several connected vectors
        vec_ids = []
        for i in range(10):
            vec_id = uuid4()
            vector = np.random.rand(128).astype(np.float32)
            vector = vector / np.linalg.norm(vector)
            vector_store.add_vector(vec_id, vector)
            hnsw_index.add_vector(vec_id, vector_index=i)
            vec_ids.append(vec_id)

        # Remove a vector from the middle - this should update neighbor references
        removed_id = vec_ids[5]
        result = hnsw_index.remove_vector(removed_id)

        assert result is True
        assert hnsw_index.size() == 9

        # Verify the removed vector is not in the index
        assert removed_id not in hnsw_index._nodes

    def test_remove_entry_point_updates_to_new_entry(self, hnsw_index, vector_store):
        """Test that removing entry point selects new one (line 379)."""
        # Add multiple vectors
        vec_ids = []
        for i in range(5):
            vec_id = uuid4()
            vector = np.random.rand(128).astype(np.float32)
            vector = vector / np.linalg.norm(vector)
            vector_store.add_vector(vec_id, vector)
            hnsw_index.add_vector(vec_id, vector_index=i)
            vec_ids.append(vec_id)

        # Get current entry point
        entry_point = hnsw_index._entry_point
        assert entry_point is not None

        # Remove the entry point
        hnsw_index.remove_vector(entry_point)

        # Should have selected a new entry point
        assert hnsw_index._entry_point is not None
        assert hnsw_index._entry_point != entry_point
        assert hnsw_index.size() == 4


class TestHNSWSearchValidation:
    """Test search parameter validation."""

    def test_search_with_k_zero_raises_error(self, hnsw_index, vector_store):
        """Test that k=0 raises ValueError (line 415)."""
        # Add a vector first
        vec_id = uuid4()
        vector = np.random.rand(128).astype(np.float32)
        vector = vector / np.linalg.norm(vector)
        vector_store.add_vector(vec_id, vector)
        hnsw_index.add_vector(vec_id, vector_index=0)

        query = np.random.rand(128).astype(np.float32)
        query = query / np.linalg.norm(query)

        with pytest.raises(ValueError) as exc_info:
            hnsw_index.search(query, k=0)

        assert "k must be positive" in str(exc_info.value).lower()

    def test_search_with_wrong_dimension_raises_error(self, hnsw_index, vector_store):
        """Test that wrong dimension query raises ValueError (line 420)."""
        # Add a vector first
        vec_id = uuid4()
        vector = np.random.rand(128).astype(np.float32)
        vector = vector / np.linalg.norm(vector)
        vector_store.add_vector(vec_id, vector)
        hnsw_index.add_vector(vec_id, vector_index=0)

        # Query with wrong dimension
        wrong_query = np.random.rand(64).astype(np.float32)
        wrong_query = wrong_query / np.linalg.norm(wrong_query)

        with pytest.raises(ValueError) as exc_info:
            hnsw_index.search(wrong_query, k=1)

        assert "dimension" in str(exc_info.value).lower()


class TestHNSWRebuild:
    """Test rebuild functionality."""

    def test_rebuild_empty_index_returns_early(self, hnsw_index):
        """Test that rebuilding empty index returns early (line 487)."""
        # Verify index is empty
        assert hnsw_index.size() == 0

        # Rebuild should complete without error
        hnsw_index.rebuild()

        # Still empty
        assert hnsw_index.size() == 0

    def test_rebuild_reconstructs_graph(self, hnsw_index, vector_store):
        """Test that rebuild reconstructs the HNSW graph."""
        # Add vectors
        vec_ids = []
        for i in range(20):
            vec_id = uuid4()
            vector = np.random.rand(128).astype(np.float32)
            vector = vector / np.linalg.norm(vector)
            vector_store.add_vector(vec_id, vector)
            hnsw_index.add_vector(vec_id, vector_index=i)
            vec_ids.append(vec_id)

        original_size = hnsw_index.size()

        # Rebuild
        hnsw_index.rebuild()

        # Should have same number of vectors
        assert hnsw_index.size() == original_size

        # Should still be searchable
        query = np.random.rand(128).astype(np.float32)
        query = query / np.linalg.norm(query)
        results = hnsw_index.search(query, k=5)
        assert len(results) == 5


class TestHNSWStatistics:
    """Test statistics and repr methods."""

    def test_get_statistics_empty_index(self, hnsw_index):
        """Test get_statistics on empty index (lines 526-536)."""
        stats = hnsw_index.get_statistics()

        assert stats["type"] == "hnsw"
        assert stats["size"] == 0
        assert stats["vector_store_dim"] == 128
        assert stats["M"] == 16
        assert stats["ef_construction"] == 200
        assert stats["ef_search"] == 50
        assert stats["supports_incremental"] is True

    def test_get_statistics_with_data(self, hnsw_index, vector_store):
        """Test get_statistics with data computes averages (lines 538-551)."""
        # Add multiple vectors to create multi-layer graph
        for i in range(50):
            vec_id = uuid4()
            vector = np.random.rand(128).astype(np.float32)
            vector = vector / np.linalg.norm(vector)
            vector_store.add_vector(vec_id, vector)
            hnsw_index.add_vector(vec_id, vector_index=i)

        stats = hnsw_index.get_statistics()

        assert stats["type"] == "hnsw"
        assert stats["size"] == 50
        assert stats["vector_store_dim"] == 128
        assert "num_levels" in stats
        assert "level_counts" in stats
        assert "avg_connections" in stats
        assert stats["avg_connections"] > 0

    def test_repr_empty_index(self, hnsw_index):
        """Test __repr__ on empty index (lines 566-567)."""
        repr_str = repr(hnsw_index)

        assert "HNSWIndex" in repr_str
        assert "size=0" in repr_str
        assert "dimension=128" in repr_str
        assert "M=16" in repr_str
        assert "ef_search=50" in repr_str

    def test_repr_with_data(self, hnsw_index, vector_store):
        """Test __repr__ with data (lines 566-567)."""
        # Add some vectors
        for i in range(10):
            vec_id = uuid4()
            vector = np.random.rand(128).astype(np.float32)
            vector = vector / np.linalg.norm(vector)
            vector_store.add_vector(vec_id, vector)
            hnsw_index.add_vector(vec_id, vector_index=i)

        repr_str = repr(hnsw_index)

        assert "HNSWIndex" in repr_str
        assert "size=10" in repr_str
        assert "dimension=128" in repr_str


class TestHNSWClear:
    """Test clear method."""

    def test_clear_removes_all_vectors(self, hnsw_index, vector_store):
        """Test that clear removes all vectors."""
        # Add vectors
        for i in range(10):
            vec_id = uuid4()
            vector = np.random.rand(128).astype(np.float32)
            vector = vector / np.linalg.norm(vector)
            vector_store.add_vector(vec_id, vector)
            hnsw_index.add_vector(vec_id, vector_index=i)

        assert hnsw_index.size() == 10

        # Clear
        hnsw_index.clear()

        # Should be empty
        assert hnsw_index.size() == 0
        assert hnsw_index._entry_point is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
