"""
Unit tests for LibraryService core business logic.

Tests the service layer methods with mocked repository and embedding service.
Focuses on actual business logic, not just pass-through methods.
"""

import pytest
from unittest.mock import MagicMock, Mock
from uuid import uuid4
from datetime import datetime

from app.services.library_service import LibraryService
from app.models.base import Library, LibraryMetadata, Document, DocumentMetadata, Chunk, ChunkMetadata
from infrastructure.repositories.library_repository import LibraryNotFoundError, DocumentNotFoundError


@pytest.fixture
def mock_repository():
    """Create mock repository."""
    return MagicMock()


@pytest.fixture
def mock_embedding_service():
    """Create mock embedding service."""
    mock = MagicMock()
    mock.embedding_dimension = 1024
    mock.model = "embed-english-v3.0"
    mock.input_type = "search_document"
    return mock


@pytest.fixture
def library_service(mock_repository, mock_embedding_service):
    """Create LibraryService with mocked dependencies."""
    return LibraryService(
        repository=mock_repository,
        embedding_service=mock_embedding_service
    )


class TestCreateLibrary:
    """Test library creation business logic."""

    def test_create_library_success(self, library_service, mock_repository, mock_embedding_service):
        """Test successful library creation."""
        library = Library(
            name="Test Library",
            documents=[],
            metadata=LibraryMetadata(
                description="Test",
                created_at=datetime.utcnow(),
                index_type="brute_force",
                embedding_dimension=1024,
                embedding_model="embed-english-v3.0",
            ),
        )
        mock_repository.create_library.return_value = library

        result = library_service.create_library(
            name="Test Library",
            description="Test",
            index_type="brute_force",
        )

        assert result.name == "Test Library"
        mock_repository.create_library.assert_called_once()
        # Verify it uses embedding service dimensions
        call_args = mock_repository.create_library.call_args[0][0]
        assert call_args.metadata.embedding_dimension == mock_embedding_service.embedding_dimension

    def test_create_library_uses_custom_model(self, library_service, mock_repository):
        """Test library creation with custom embedding model."""
        library = Library(
            name="Custom Model Library",
            documents=[],
            metadata=LibraryMetadata(
                description="Test",
                created_at=datetime.utcnow(),
                index_type="hnsw",
                embedding_dimension=1024,
                embedding_model="custom-model-v1",
            ),
        )
        mock_repository.create_library.return_value = library

        result = library_service.create_library(
            name="Custom Model Library",
            embedding_model="custom-model-v1",
            index_type="hnsw",
        )

        call_args = mock_repository.create_library.call_args[0][0]
        assert call_args.metadata.embedding_model == "custom-model-v1"


class TestGetLibrary:
    """Test get library operations."""

    def test_get_library_success(self, library_service, mock_repository):
        """Test getting an existing library."""
        library_id = uuid4()
        library = Library(
            id=library_id,
            name="Test Library",
            documents=[],
            metadata=LibraryMetadata(
                description="Test",
                created_at=datetime.utcnow(),
                index_type="brute_force",
                embedding_dimension=1024,
                embedding_model="embed-english-v3.0",
            ),
        )
        mock_repository.get_library.return_value = library

        result = library_service.get_library(library_id)

        assert result.id == library_id
        assert result.name == "Test Library"
        mock_repository.get_library.assert_called_once_with(library_id)


class TestListLibraries:
    """Test list libraries operation."""

    def test_list_libraries_empty(self, library_service, mock_repository):
        """Test listing when no libraries exist."""
        mock_repository.list_libraries.return_value = []

        result = library_service.list_libraries()

        assert result == []
        mock_repository.list_libraries.assert_called_once()

    def test_list_libraries_multiple(self, library_service, mock_repository):
        """Test listing multiple libraries."""
        libraries = [
            Library(
                name=f"Library {i}",
                documents=[],
                metadata=LibraryMetadata(
                    created_at=datetime.utcnow(),
                    index_type="brute_force",
                    embedding_dimension=1024,
                    embedding_model="embed-english-v3.0",
                ),
            )
            for i in range(3)
        ]
        mock_repository.list_libraries.return_value = libraries

        result = library_service.list_libraries()

        assert len(result) == 3
        mock_repository.list_libraries.assert_called_once()


class TestDeleteLibrary:
    """Test delete library operation."""

    def test_delete_library_success(self, library_service, mock_repository):
        """Test successful library deletion."""
        library_id = uuid4()
        mock_repository.delete_library.return_value = True

        result = library_service.delete_library(library_id)

        assert result is True
        mock_repository.delete_library.assert_called_once_with(library_id)

    def test_delete_library_not_found(self, library_service, mock_repository):
        """Test deleting non-existent library."""
        library_id = uuid4()
        mock_repository.delete_library.return_value = False

        result = library_service.delete_library(library_id)

        assert result is False
        mock_repository.delete_library.assert_called_once_with(library_id)


class TestGetDocument:
    """Test get document operation."""

    def test_get_document_success(self, library_service, mock_repository):
        """Test getting an existing document."""
        doc_id = uuid4()
        document = Document(
            id=doc_id,
            chunks=[
                Chunk(
                    text="Test",
                    embedding=[0.1] * 1024,
                    metadata=ChunkMetadata(
                        created_at=datetime.utcnow(),
                        chunk_index=0,
                        source_document_id=doc_id,
                    ),
                )
            ],
            metadata=DocumentMetadata(
                title="Test Doc",
                created_at=datetime.utcnow(),
                document_type="text",
                tags=[],
            ),
        )
        mock_repository.get_document.return_value = document

        result = library_service.get_document(doc_id)

        assert result.id == doc_id
        assert result.metadata.title == "Test Doc"
        mock_repository.get_document.assert_called_once_with(doc_id)


class TestDeleteDocument:
    """Test delete document operation."""

    def test_delete_document_success(self, library_service, mock_repository):
        """Test successful document deletion."""
        doc_id = uuid4()
        mock_repository.delete_document.return_value = True

        result = library_service.delete_document(doc_id)

        assert result is True
        mock_repository.delete_document.assert_called_once_with(doc_id)

    def test_delete_document_not_found(self, library_service, mock_repository):
        """Test deleting non-existent document."""
        doc_id = uuid4()
        mock_repository.delete_document.return_value = False

        result = library_service.delete_document(doc_id)

        assert result is False


class TestGetLibraryStatistics:
    """Test get library statistics operation."""

    def test_get_library_statistics(self, library_service, mock_repository):
        """Test getting library statistics."""
        library_id = uuid4()
        stats = {
            "library_id": str(library_id),
            "library_name": "Test Library",
            "num_documents": 10,
            "num_chunks": 50,
            "embedding_dimension": 1024,
            "index_type": "brute_force",
            "vector_store_stats": {"total_vectors": 50},
            "index_stats": {"size": 50},
        }
        mock_repository.get_library_statistics.return_value = stats

        result = library_service.get_library_statistics(library_id)

        assert result["num_documents"] == 10
        assert result["num_chunks"] == 50
        assert result["library_name"] == "Test Library"
        mock_repository.get_library_statistics.assert_called_once_with(library_id)
