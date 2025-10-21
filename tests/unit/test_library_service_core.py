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


class TestAddDocumentWithText:
    """Test adding documents with text (auto-embeddings)."""

    def test_add_document_with_text_success(self, library_service, mock_repository, mock_embedding_service):
        """Test successful document addition with text chunks."""
        import numpy as np

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

        # Mock embedding service to return embeddings
        embeddings = [np.array([0.1] * 1024), np.array([0.2] * 1024)]
        mock_embedding_service.embed_texts.return_value = embeddings

        # Mock successful document addition - needs to match what we're adding
        def capture_document(lib_id, doc):
            return doc

        mock_repository.add_document.side_effect = capture_document

        result = library_service.add_document_with_text(
            library_id=library_id,
            title="Test Doc",
            texts=["Chunk 1", "Chunk 2"],
        )

        assert len(result.chunks) == 2
        assert result.chunks[0].text == "Chunk 1"
        assert result.chunks[1].text == "Chunk 2"
        assert result.metadata.title == "Test Doc"
        mock_embedding_service.embed_texts.assert_called_once_with(["Chunk 1", "Chunk 2"])
        mock_repository.add_document.assert_called_once()


class TestAddDocumentWithEmbeddings:
    """Test adding documents with pre-computed embeddings."""

    def test_add_document_with_embeddings_success(self, library_service, mock_repository):
        """Test successful document addition with pre-computed embeddings."""
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

        # Mock successful document addition
        doc_id = uuid4()
        added_doc = Document(
            id=doc_id,
            chunks=[
                Chunk(
                    text="Chunk 1",
                    embedding=[0.1] * 1024,
                    metadata=ChunkMetadata(
                        created_at=datetime.utcnow(),
                        chunk_index=0,
                        source_document_id=doc_id,
                    ),
                ),
            ],
            metadata=DocumentMetadata(
                title="Test Doc",
                created_at=datetime.utcnow(),
                document_type="text",
                tags=["test"],
            ),
        )
        mock_repository.add_document.return_value = added_doc

        text_embedding_pairs = [
            ("Chunk 1", [0.1] * 1024),
        ]

        result = library_service.add_document_with_embeddings(
            library_id=library_id,
            title="Test Doc",
            text_embedding_pairs=text_embedding_pairs,
            tags=["test"],
        )

        assert result.id == doc_id
        assert len(result.chunks) == 1
        mock_repository.get_library.assert_called_once_with(library_id)
        mock_repository.add_document.assert_called_once()

    def test_add_document_with_embeddings_empty_pairs_raises_error(self, library_service):
        """Test that empty text_embedding_pairs raises ValueError."""
        library_id = uuid4()

        with pytest.raises(ValueError, match="text_embedding_pairs cannot be empty"):
            library_service.add_document_with_embeddings(
                library_id=library_id,
                title="Empty Doc",
                text_embedding_pairs=[],
            )

    def test_add_document_with_embeddings_with_metadata(self, library_service, mock_repository):
        """Test adding document with embeddings and full metadata."""
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

        doc_id = uuid4()
        added_doc = Document(
            id=doc_id,
            chunks=[
                Chunk(
                    text="Chunk 1",
                    embedding=[0.1] * 1024,
                    metadata=ChunkMetadata(
                        created_at=datetime.utcnow(),
                        chunk_index=0,
                        source_document_id=doc_id,
                    ),
                ),
            ],
            metadata=DocumentMetadata(
                title="Test Doc",
                author="Test Author",
                document_type="research",
                source_url="https://example.com",
                tags=["ai", "ml"],
                created_at=datetime.utcnow(),
            ),
        )
        mock_repository.add_document.return_value = added_doc

        text_embedding_pairs = [
            ("Chunk 1", [0.1] * 1024),
        ]

        result = library_service.add_document_with_embeddings(
            library_id=library_id,
            title="Test Doc",
            text_embedding_pairs=text_embedding_pairs,
            author="Test Author",
            document_type="research",
            source_url="https://example.com",
            tags=["ai", "ml"],
        )

        assert result.id == doc_id
        assert result.metadata.author == "Test Author"
        assert result.metadata.document_type == "research"
        assert result.metadata.source_url == "https://example.com"
        assert result.metadata.tags == ["ai", "ml"]


class TestSearchWithText:
    """Test search with text operation."""

    def test_search_with_text_success(self, library_service, mock_repository, mock_embedding_service):
        """Test successful search with text query."""
        import numpy as np

        library_id = uuid4()

        # Mock embedding service
        query_embedding = np.array([0.5] * 1024)
        mock_embedding_service.embed_text.return_value = query_embedding
        mock_embedding_service.input_type = "search_document"

        # Mock search results
        chunk_id = uuid4()
        doc_id = uuid4()
        chunk = Chunk(
            id=chunk_id,
            text="Relevant chunk",
            embedding=[0.5] * 1024,
            metadata=ChunkMetadata(
                created_at=datetime.utcnow(),
                chunk_index=0,
                source_document_id=doc_id,
            ),
        )
        search_results = [(chunk, 0.1)]
        mock_repository.search.return_value = search_results

        results = library_service.search_with_text(
            library_id=library_id,
            query_text="test query",
            k=10,
        )

        assert len(results) == 1
        assert results[0][0].id == chunk_id
        assert results[0][1] == 0.1

        # Verify input type was changed and restored
        mock_embedding_service.change_input_type.assert_any_call("search_query")
        mock_embedding_service.change_input_type.assert_any_call("search_document")
        mock_embedding_service.embed_text.assert_called_once_with("test query")
        mock_repository.search.assert_called_once()


class TestSearchWithEmbedding:
    """Test search with embedding operation."""

    def test_search_with_embedding_success(self, library_service, mock_repository):
        """Test successful search with pre-computed embedding."""
        library_id = uuid4()
        query_embedding = [0.5] * 1024

        # Mock search results
        chunk_id = uuid4()
        doc_id = uuid4()
        chunk = Chunk(
            id=chunk_id,
            text="Relevant chunk",
            embedding=[0.5] * 1024,
            metadata=ChunkMetadata(
                created_at=datetime.utcnow(),
                chunk_index=0,
                source_document_id=doc_id,
            ),
        )
        search_results = [(chunk, 0.1), (chunk, 0.2)]
        mock_repository.search.return_value = search_results

        results = library_service.search_with_embedding(
            library_id=library_id,
            query_embedding=query_embedding,
            k=10,
        )

        assert len(results) == 2
        assert results[0][1] == 0.1
        assert results[1][1] == 0.2
        mock_repository.search.assert_called_once_with(
            library_id, query_embedding, 10, None
        )

    def test_search_with_embedding_with_distance_threshold(self, library_service, mock_repository):
        """Test search with distance threshold."""
        library_id = uuid4()
        query_embedding = [0.5] * 1024

        mock_repository.search.return_value = []

        results = library_service.search_with_embedding(
            library_id=library_id,
            query_embedding=query_embedding,
            k=5,
            distance_threshold=0.5,
        )

        assert len(results) == 0
        mock_repository.search.assert_called_once_with(
            library_id, query_embedding, 5, 0.5
        )
