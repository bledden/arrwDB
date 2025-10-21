"""
Shared validation tests for all index implementations.

Tests initialization validation for HNSW and LSH indexes.
Uses parametrize to avoid test duplication.
"""

import pytest
from core.vector_store import VectorStore
from infrastructure.indexes.hnsw import HNSWIndex
from infrastructure.indexes.lsh import LSHIndex


@pytest.fixture
def vector_store():
    """Shared vector store fixture for all index tests."""
    return VectorStore(dimension=128)


class TestHNSWValidation:
    """Test HNSW parameter validation."""

    @pytest.mark.parametrize("invalid_m", [0, -1, -10])
    def test_invalid_m_raises_error(self, vector_store, invalid_m):
        """Test that invalid M parameter raises ValueError (line 102)."""
        with pytest.raises(ValueError) as exc_info:
            HNSWIndex(vector_store, M=invalid_m)

        assert "m must be positive" in str(exc_info.value).lower()

    @pytest.mark.parametrize("invalid_ef_construction", [0, -1, -100])
    def test_invalid_ef_construction_raises_error(self, vector_store, invalid_ef_construction):
        """Test that invalid ef_construction raises ValueError (line 104)."""
        with pytest.raises(ValueError) as exc_info:
            HNSWIndex(vector_store, ef_construction=invalid_ef_construction)

        assert "ef_construction must be positive" in str(exc_info.value).lower()

    @pytest.mark.parametrize("invalid_ef_search", [0, -1, -50])
    def test_invalid_ef_search_raises_error(self, vector_store, invalid_ef_search):
        """Test that invalid ef_search raises ValueError (line 108)."""
        with pytest.raises(ValueError) as exc_info:
            HNSWIndex(vector_store, ef_search=invalid_ef_search)

        assert "ef_search must be positive" in str(exc_info.value).lower()


class TestLSHValidation:
    """Test LSH parameter validation."""

    @pytest.mark.parametrize("invalid_tables", [0, -1, -5])
    def test_invalid_num_tables_raises_error(self, vector_store, invalid_tables):
        """Test that invalid num_tables raises ValueError (line 79)."""
        with pytest.raises(ValueError) as exc_info:
            LSHIndex(vector_store, num_tables=invalid_tables)

        assert "num_tables must be positive" in str(exc_info.value).lower()

    @pytest.mark.parametrize("invalid_hash_size", [0, -1, -8])
    def test_invalid_hash_size_raises_error(self, vector_store, invalid_hash_size):
        """Test that invalid hash_size raises ValueError (line 81)."""
        with pytest.raises(ValueError) as exc_info:
            LSHIndex(vector_store, hash_size=invalid_hash_size)

        assert "hash_size must be positive" in str(exc_info.value).lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
