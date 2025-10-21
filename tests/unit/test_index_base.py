"""
Unit tests for VectorIndex abstract base class.

Tests that the ABC properly enforces the interface contract.
"""

import pytest
import numpy as np
from uuid import uuid4
from abc import ABC

from infrastructure.indexes.base import VectorIndex


class TestVectorIndexABC:
    """Test the VectorIndex abstract base class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that VectorIndex cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            VectorIndex()

    def test_incomplete_implementation_raises_error(self):
        """Test that incomplete implementations can't be instantiated."""

        # Create a class that only implements some methods
        class IncompleteIndex(VectorIndex):
            def add_vector(self, vector_id, vector_index):
                pass

            def remove_vector(self, vector_id):
                pass

            # Missing: search, rebuild, size, clear, supports_incremental_updates, index_type

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteIndex()

    def test_complete_implementation_can_be_instantiated(self):
        """Test that complete implementations can be instantiated."""

        class CompleteIndex(VectorIndex):
            def __init__(self):
                self._size = 0

            def add_vector(self, vector_id, vector_index):
                self._size += 1

            def remove_vector(self, vector_id):
                return True

            def search(self, query_vector, k, distance_threshold=None):
                return []

            def rebuild(self):
                pass

            def size(self):
                return self._size

            def clear(self):
                self._size = 0

            @property
            def supports_incremental_updates(self):
                return True

            @property
            def index_type(self):
                return "test"

        # Should not raise
        index = CompleteIndex()
        assert isinstance(index, VectorIndex)
        assert index.size() == 0
        assert index.index_type == "test"
        assert index.supports_incremental_updates is True

    def test_abc_enforces_method_signatures(self):
        """Test that the ABC properly defines method signatures."""
        # Verify all abstract methods are defined
        abstract_methods = VectorIndex.__abstractmethods__

        expected_methods = {
            'add_vector',
            'remove_vector',
            'search',
            'rebuild',
            'size',
            'clear',
            'supports_incremental_updates',
            'index_type'
        }

        assert abstract_methods == expected_methods

    def test_all_concrete_indexes_implement_interface(self):
        """Test that all concrete index implementations properly implement the interface."""
        from infrastructure.indexes.brute_force import BruteForceIndex
        from infrastructure.indexes.kd_tree import KDTreeIndex
        from infrastructure.indexes.lsh import LSHIndex
        from infrastructure.indexes.hnsw import HNSWIndex
        from core.vector_store import VectorStore

        dimension = 128
        vector_store = VectorStore(dimension=dimension)

        # Create instances with their default constructor parameters
        indexes = [
            ("brute_force", BruteForceIndex(vector_store=vector_store)),
            ("kd_tree", KDTreeIndex(vector_store=vector_store)),
            ("lsh", LSHIndex(vector_store=vector_store)),
            ("hnsw", HNSWIndex(vector_store=vector_store)),
        ]

        for expected_type, index in indexes:
            # Verify it's a VectorIndex
            assert isinstance(index, VectorIndex)

            # Verify all methods are callable
            assert callable(index.add_vector)
            assert callable(index.remove_vector)
            assert callable(index.search)
            assert callable(index.rebuild)
            assert callable(index.size)
            assert callable(index.clear)

            # Verify properties exist and have correct values
            assert hasattr(index, 'supports_incremental_updates')
            assert hasattr(index, 'index_type')
            assert isinstance(index.supports_incremental_updates, bool)
            assert isinstance(index.index_type, str)
            assert index.index_type == expected_type
