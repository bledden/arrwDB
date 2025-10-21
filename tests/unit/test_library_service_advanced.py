"""
Advanced LibraryService tests to increase coverage.

Targets missing lines in app/services/library_service.py to improve coverage
from 82% toward 95%+. Focuses on error handling and validation.
"""

import pytest
from uuid import uuid4
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from app.services.library_service import LibraryService
from app.services.embedding_service import EmbeddingService, EmbeddingServiceError
from infrastructure.repositories.library_repository import (
    LibraryRepository,
    LibraryNotFoundError,
    DimensionMismatchError,
)
from app.models.base import Library, Document, Chunk, LibraryMetadata, DocumentMetadata


class TestLibraryServiceCreateLibrary:
    """Test create_library error handling."""

    @pytest.fixture
    def service(self):
        """Create LibraryService with mocked dependencies."""
        # Create mock repository (first parameter)
        repository = MagicMock(spec=LibraryRepository)

        # Create a real embedding service with mocked Cohere client (second parameter)
        with patch('cohere.Client'):
            embedding_service = EmbeddingService(api_key="test_key", embedding_dimension=128)
            return LibraryService(repository, embedding_service)

    def test_create_library_invalid_index_type(self, service):
        """Test that invalid index_type raises ValueError (line 85-88)."""
        with pytest.raises(ValueError) as exc_info:
            service.create_library(
                name="test_lib",
                index_type="invalid_index",
                description="test"
            )

        error_msg = str(exc_info.value).lower()
        assert "invalid index_type" in error_msg or "invalid_index" in error_msg

    def test_create_library_repository_error(self, service):
        """Test repository error handling during creation (line 105-107)."""
        service._repository.create_library.side_effect = Exception("Database error")

        with pytest.raises(Exception) as exc_info:
            service.create_library(
                name="test_lib",
                index_type="brute_force",
                description="test"
            )

        assert "database error" in str(exc_info.value).lower()


class TestLibraryServiceAddDocumentWithText:
    """Test add_document_with_text error handling."""

    @pytest.fixture
    def service(self):
        """Create LibraryService with mocked dependencies."""
        # Create mock repository (first parameter)
        repository = MagicMock(spec=LibraryRepository)

        # Mock get_library to return a valid library
        mock_library = Library(
            id=uuid4(),
            name="test_lib",
            metadata=LibraryMetadata(
                index_type="brute_force",
                embedding_dimension=128,
                embedding_model="test"
            )
        )
        repository.get_library.return_value = mock_library

        # Create embedding service (second parameter)
        with patch('cohere.Client'):
            embedding_service = EmbeddingService(api_key="test_key", embedding_dimension=128)
            return LibraryService(repository, embedding_service)

    def test_add_document_empty_texts_raises_error(self, service):
        """Test that empty texts list raises ValueError (line 186-187)."""
        library_id = uuid4()

        with pytest.raises(ValueError) as exc_info:
            service.add_document_with_text(
                library_id=library_id,
                texts=[],
                title="Test Doc"
            )

        assert "empty" in str(exc_info.value).lower()

    def test_add_document_embedding_service_error(self, service):
        """Test EmbeddingServiceError handling (line 199-201)."""
        library_id = uuid4()

        # Mock embed_texts to raise error
        service._embedding_service.embed_texts = Mock(
            side_effect=EmbeddingServiceError("API error")
        )

        with pytest.raises(EmbeddingServiceError) as exc_info:
            service.add_document_with_text(
                library_id=library_id,
                texts=["chunk 1", "chunk 2"],
                title="Test Doc"
            )

        assert "api error" in str(exc_info.value).lower()

    def test_add_document_dimension_mismatch_error(self, service):
        """Test DimensionMismatchError handling (line 241-243)."""
        library_id = uuid4()

        # Mock embed_texts to return embeddings
        service._embedding_service.embed_texts = Mock(
            return_value=[
                np.random.rand(128).astype(np.float32),
                np.random.rand(128).astype(np.float32)
            ]
        )

        # Mock add_document to raise DimensionMismatchError
        service._repository.add_document.side_effect = DimensionMismatchError(
            "Expected 256 dimensions, got 128"
        )

        with pytest.raises(DimensionMismatchError) as exc_info:
            service.add_document_with_text(
                library_id=library_id,
                texts=["chunk 1", "chunk 2"],
                title="Test Doc"
            )

        assert "dimension" in str(exc_info.value).lower()

    def test_add_document_generic_repository_error(self, service):
        """Test generic Exception handling (line 244-246)."""
        library_id = uuid4()

        # Mock embed_texts to return embeddings
        service._embedding_service.embed_texts = Mock(
            return_value=[
                np.random.rand(128).astype(np.float32),
                np.random.rand(128).astype(np.float32)
            ]
        )

        # Mock add_document to raise generic error
        service._repository.add_document.side_effect = Exception("Repository error")

        with pytest.raises(Exception) as exc_info:
            service.add_document_with_text(
                library_id=library_id,
                texts=["chunk 1", "chunk 2"],
                title="Test Doc"
            )

        assert "repository error" in str(exc_info.value).lower()


class TestLibraryServiceAddDocumentWithEmbeddings:
    """Test add_document_with_embeddings error handling."""

    @pytest.fixture
    def service(self):
        """Create LibraryService with mocked dependencies."""
        # Create mock repository (first parameter)
        repository = MagicMock(spec=LibraryRepository)

        # Mock get_library to return a valid library
        mock_library = Library(
            id=uuid4(),
            name="test_lib",
            metadata=LibraryMetadata(
                index_type="brute_force",
                embedding_dimension=128,
                embedding_model="test"
            )
        )
        repository.get_library.return_value = mock_library

        # Create embedding service (second parameter)
        with patch('cohere.Client'):
            embedding_service = EmbeddingService(api_key="test_key", embedding_dimension=128)
            return LibraryService(repository, embedding_service)

    def test_add_document_with_embeddings_empty_pairs_raises_error(self, service):
        """Test that empty pairs list raises ValueError (line 280-281)."""
        library_id = uuid4()

        with pytest.raises(ValueError) as exc_info:
            service.add_document_with_embeddings(
                library_id=library_id,
                text_embedding_pairs=[],
                title="Test Doc"
            )

        assert "empty" in str(exc_info.value).lower()


class TestLibraryServiceSearch:
    """Test search error handling."""

    @pytest.fixture
    def service(self):
        """Create LibraryService with mocked dependencies."""
        # Create mock repository (first parameter)
        repository = MagicMock(spec=LibraryRepository)

        # Mock get_library to return a valid library
        mock_library = Library(
            id=uuid4(),
            name="test_lib",
            metadata=LibraryMetadata(
                index_type="brute_force",
                embedding_dimension=128,
                embedding_model="test"
            )
        )
        repository.get_library.return_value = mock_library

        # Create embedding service (second parameter) with _input_type attribute
        with patch('cohere.Client'):
            embedding_service = EmbeddingService(
                api_key="test_key",
                embedding_dimension=128,
                input_type="search_document"
            )
            return LibraryService(repository, embedding_service)

    def test_search_embedding_service_error(self, service):
        """Test EmbeddingServiceError during search (line 399-401)."""
        library_id = uuid4()

        # Mock embed_text to raise error
        service._embedding_service.embed_text = Mock(
            side_effect=EmbeddingServiceError("API error")
        )

        with pytest.raises(EmbeddingServiceError) as exc_info:
            service.search_with_text(
                library_id=library_id,
                query_text="test query"
            )

        assert "api error" in str(exc_info.value).lower()

    def test_search_with_embedding_repository_error(self, service):
        """Test repository error during search_with_embedding (line 442-444)."""
        library_id = uuid4()
        query_embedding = [0.1] * 128

        # Mock search to raise error
        service._repository.search.side_effect = Exception("Search failed")

        with pytest.raises(Exception) as exc_info:
            service.search_with_embedding(
                library_id=library_id,
                query_embedding=query_embedding
            )

        assert "search failed" in str(exc_info.value).lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
