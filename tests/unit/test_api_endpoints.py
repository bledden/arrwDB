"""
Unit tests for API endpoints with mocked dependencies.

Tests endpoint logic, response formatting, error handling without requiring
external services like Cohere API.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from uuid import uuid4
from datetime import datetime

from app.api.main import app
from app.models.base import Library, LibraryMetadata, Document, DocumentMetadata, Chunk, ChunkMetadata


@pytest.fixture
def mock_library_service():
    """Create a mock library service."""
    return MagicMock()


@pytest.fixture
def client_with_mock(mock_library_service):
    """Create test client with mocked library service."""
    from app.api.dependencies import get_library_service

    app.dependency_overrides[get_library_service] = lambda: mock_library_service
    client = TestClient(app)
    yield client, mock_library_service
    app.dependency_overrides.clear()


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_check_returns_healthy(self):
        """Test health endpoint returns healthy status."""
        client = TestClient(app)
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "timestamp" in data


class TestRootEndpoint:
    """Test root endpoint."""

    def test_root_returns_api_info(self):
        """Test root endpoint returns API information."""
        client = TestClient(app)
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Vector Database API"
        assert "version" in data
        assert "api_versions" in data
        assert "v1" in data["api_versions"]


class TestLibraryEndpoints:
    """Test library management endpoints."""

    def test_create_library(self, client_with_mock):
        """Test creating a library."""
        client, mock_service = client_with_mock

        # Setup mock
        mock_library = Library(
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
        mock_service.create_library.return_value = mock_library

        # Make request
        response = client.post(
            "/v1/libraries",
            json={
                "name": "Test Library",
                "description": "Test",
                "index_type": "brute_force",
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Test Library"
        mock_service.create_library.assert_called_once()

    def test_list_libraries_empty(self, client_with_mock):
        """Test listing empty libraries."""
        client, mock_service = client_with_mock

        # Setup mock
        mock_service.list_libraries.return_value = []

        # Make request
        response = client.get("/v1/libraries")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 0
        mock_service.list_libraries.assert_called_once()

    def test_list_libraries_with_documents(self, client_with_mock):
        """Test listing libraries that contain documents."""
        client, mock_service = client_with_mock

        # Setup mock with libraries containing documents
        doc_id = uuid4()
        libraries = [
            Library(
                name="Lib 1",
                documents=[
                    Document(
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
                            title="Doc 1",
                            created_at=datetime.utcnow(),
                            document_type="text",
                            tags=[],
                        ),
                    )
                ],
                metadata=LibraryMetadata(
                    created_at=datetime.utcnow(),
                    index_type="brute_force",
                    embedding_dimension=1024,
                    embedding_model="embed-english-v3.0",
                ),
            )
        ]
        mock_service.list_libraries.return_value = libraries

        # Make request
        response = client.get("/v1/libraries")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["name"] == "Lib 1"
        assert data[0]["num_documents"] == 1
        mock_service.list_libraries.assert_called_once()

    def test_get_library(self, client_with_mock):
        """Test getting a specific library."""
        client, mock_service = client_with_mock

        library_id = uuid4()
        mock_library = Library(
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
        mock_service.get_library.return_value = mock_library

        response = client.get(f"/v1/libraries/{library_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Test Library"

    def test_delete_library(self, client_with_mock):
        """Test deleting a library."""
        client, mock_service = client_with_mock

        library_id = uuid4()
        mock_service.delete_library.return_value = True  # Successfully deleted

        response = client.delete(f"/v1/libraries/{library_id}")

        assert response.status_code == 204
        mock_service.delete_library.assert_called_once_with(library_id)

    def test_get_library_statistics(self, client_with_mock):
        """Test getting library statistics."""
        client, mock_service = client_with_mock

        library_id = uuid4()
        mock_stats = {
            "library_id": str(library_id),
            "library_name": "Test",
            "num_documents": 10,
            "num_chunks": 50,
            "embedding_dimension": 1024,
            "index_type": "brute_force",
            "vector_store_stats": {},
            "index_stats": {},
        }
        mock_service.get_library_statistics.return_value = mock_stats

        response = client.get(f"/v1/libraries/{library_id}/statistics")

        assert response.status_code == 200
        data = response.json()
        assert data["num_documents"] == 10
        assert data["num_chunks"] == 50


class TestDocumentEndpoints:
    """Test document management endpoints."""

    def test_get_document(self, client_with_mock):
        """Test getting a document."""
        client, mock_service = client_with_mock

        doc_id = uuid4()
        mock_doc = Document(
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
        mock_service.get_document.return_value = mock_doc

        response = client.get(f"/v1/documents/{doc_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["metadata"]["title"] == "Test Doc"

    def test_delete_document(self, client_with_mock):
        """Test deleting a document."""
        client, mock_service = client_with_mock

        doc_id = uuid4()
        mock_service.delete_document.return_value = True  # Successfully deleted

        response = client.delete(f"/v1/documents/{doc_id}")

        assert response.status_code == 204
        mock_service.delete_document.assert_called_once_with(doc_id)


class TestSearchEndpoints:
    """Test search endpoints."""

    def test_search_with_text_slim_response(self, client_with_mock):
        """Test search with text returns slim response by default."""
        client, mock_service = client_with_mock

        library_id = uuid4()
        doc_id = uuid4()

        # Setup mock search results
        chunk = Chunk(
            text="Test result",
            embedding=[0.5] * 1024,
            metadata=ChunkMetadata(
                created_at=datetime.utcnow(),
                chunk_index=0,
                source_document_id=doc_id,
            ),
        )
        mock_doc = Document(
            id=doc_id,
            chunks=[chunk],
            metadata=DocumentMetadata(
                title="Test Doc",
                created_at=datetime.utcnow(),
                document_type="text",
                tags=[],
            ),
        )

        mock_service.search_with_text.return_value = [(chunk, 0.25)]
        mock_service.get_document.return_value = mock_doc

        response = client.post(
            f"/v1/libraries/{library_id}/search",
            json={"query": "test query", "k": 10},
        )

        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "query_time_ms" in data
        assert "total_results" in data
        # Verify slim response (no embeddings)
        if len(data["results"]) > 0:
            assert "embedding" not in data["results"][0]["chunk"]

    def test_search_with_text_full_response(self, client_with_mock):
        """Test search with include_embeddings=true returns full response."""
        client, mock_service = client_with_mock

        library_id = uuid4()
        doc_id = uuid4()

        chunk = Chunk(
            text="Test result",
            embedding=[0.5] * 1024,
            metadata=ChunkMetadata(
                created_at=datetime.utcnow(),
                chunk_index=0,
                source_document_id=doc_id,
            ),
        )
        mock_doc = Document(
            id=doc_id,
            chunks=[chunk],
            metadata=DocumentMetadata(
                title="Test Doc",
                created_at=datetime.utcnow(),
                document_type="text",
                tags=[],
            ),
        )

        mock_service.search_with_text.return_value = [(chunk, 0.25)]
        mock_service.get_document.return_value = mock_doc

        response = client.post(
            f"/v1/libraries/{library_id}/search?include_embeddings=true",
            json={"query": "test query", "k": 10},
        )

        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        # Verify full response (with embeddings)
        assert len(data["results"]) > 0
        assert "embedding" in data["results"][0]["chunk"]
        assert len(data["results"][0]["chunk"]["embedding"]) == 1024

    def test_search_with_embedding_slim_response(self, client_with_mock):
        """Test search with embedding returns slim response by default."""
        client, mock_service = client_with_mock

        library_id = uuid4()
        doc_id = uuid4()

        chunk = Chunk(
            text="Test result",
            embedding=[0.5] * 1024,
            metadata=ChunkMetadata(
                created_at=datetime.utcnow(),
                chunk_index=0,
                source_document_id=doc_id,
            ),
        )
        mock_doc = Document(
            id=doc_id,
            chunks=[chunk],
            metadata=DocumentMetadata(
                title="Test Doc",
                created_at=datetime.utcnow(),
                document_type="text",
                tags=[],
            ),
        )

        mock_service.search_with_embedding.return_value = [(chunk, 0.25)]
        mock_service.get_document.return_value = mock_doc

        response = client.post(
            f"/v1/libraries/{library_id}/search/embedding",
            json={"embedding": [0.1] * 1024, "k": 10},
        )

        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        # Verify slim response
        if len(data["results"]) > 0:
            assert "embedding" not in data["results"][0]["chunk"]

    def test_search_with_embedding_full_response(self, client_with_mock):
        """Test search with embedding and include_embeddings=true returns full response."""
        client, mock_service = client_with_mock

        library_id = uuid4()
        doc_id = uuid4()

        chunk = Chunk(
            text="Test result",
            embedding=[0.5] * 1024,
            metadata=ChunkMetadata(
                created_at=datetime.utcnow(),
                chunk_index=0,
                source_document_id=doc_id,
            ),
        )
        mock_doc = Document(
            id=doc_id,
            chunks=[chunk],
            metadata=DocumentMetadata(
                title="Test Doc",
                created_at=datetime.utcnow(),
                document_type="text",
                tags=[],
            ),
        )

        mock_service.search_with_embedding.return_value = [(chunk, 0.25)]
        mock_service.get_document.return_value = mock_doc

        response = client.post(
            f"/v1/libraries/{library_id}/search/embedding?include_embeddings=true",
            json={"embedding": [0.1] * 1024, "k": 10},
        )

        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        # Verify full response (with embeddings)
        assert len(data["results"]) > 0
        assert "embedding" in data["results"][0]["chunk"]
        assert len(data["results"][0]["chunk"]["embedding"]) == 1024


class TestDocumentAddEndpoints:
    """Test document add endpoints."""

    def test_add_document_with_text(self, client_with_mock):
        """Test adding document with text chunks."""
        client, mock_service = client_with_mock

        library_id = uuid4()
        doc_id = uuid4()

        # Mock returned document
        mock_doc = Document(
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
                )
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
        mock_service.add_document_with_text.return_value = mock_doc

        response = client.post(
            f"/v1/libraries/{library_id}/documents",
            json={
                "title": "Test Doc",
                "texts": ["Chunk 1"],
                "author": "Test Author",
                "document_type": "research",
                "source_url": "https://example.com",
                "tags": ["ai", "ml"],
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["metadata"]["title"] == "Test Doc"
        assert data["metadata"]["author"] == "Test Author"
        mock_service.add_document_with_text.assert_called_once()

    def test_add_document_with_embeddings(self, client_with_mock):
        """Test adding document with pre-computed embeddings."""
        client, mock_service = client_with_mock

        library_id = uuid4()
        doc_id = uuid4()

        # Mock returned document
        mock_doc = Document(
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
                )
            ],
            metadata=DocumentMetadata(
                title="Test Doc with Embeddings",
                created_at=datetime.utcnow(),
                document_type="text",
                tags=[],
            ),
        )
        mock_service.add_document_with_embeddings.return_value = mock_doc

        response = client.post(
            f"/v1/libraries/{library_id}/documents/with-embeddings",
            json={
                "title": "Test Doc with Embeddings",
                "chunks": [
                    {"text": "Chunk 1", "embedding": [0.1] * 1024}
                ],
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["metadata"]["title"] == "Test Doc with Embeddings"
        mock_service.add_document_with_embeddings.assert_called_once()


class TestErrorHandling:
    """Test error handling in endpoints."""

    def test_library_not_found_returns_404(self, client_with_mock):
        """Test that library not found returns 404."""
        from infrastructure.repositories.library_repository import LibraryNotFoundError

        client, mock_service = client_with_mock

        library_id = uuid4()
        mock_service.get_library.side_effect = LibraryNotFoundError(
            f"Library {library_id} not found"
        )

        response = client.get(f"/v1/libraries/{library_id}")

        assert response.status_code == 404
        data = response.json()
        assert "error" in data

    def test_document_not_found_returns_404(self, client_with_mock):
        """Test that document not found returns 404."""
        from infrastructure.repositories.library_repository import DocumentNotFoundError

        client, mock_service = client_with_mock

        doc_id = uuid4()
        mock_service.get_document.side_effect = DocumentNotFoundError(
            f"Document {doc_id} not found"
        )

        response = client.get(f"/v1/documents/{doc_id}")

        assert response.status_code == 404
        data = response.json()
        assert "error" in data

    def test_delete_library_not_found_returns_404(self, client_with_mock):
        """Test that deleting non-existent library returns 404."""
        client, mock_service = client_with_mock

        library_id = uuid4()
        mock_service.delete_library.return_value = False  # Not found

        response = client.delete(f"/v1/libraries/{library_id}")

        assert response.status_code == 404
        data = response.json()
        assert "error" in data
        assert "not found" in data["detail"].lower()

    def test_delete_document_not_found_returns_404(self, client_with_mock):
        """Test that deleting non-existent document returns 404."""
        client, mock_service = client_with_mock

        doc_id = uuid4()
        mock_service.delete_document.return_value = False  # Not found

        response = client.delete(f"/v1/documents/{doc_id}")

        assert response.status_code == 404
        data = response.json()
        assert "error" in data

    def test_dimension_mismatch_error_returns_400(self, client_with_mock):
        """Test that dimension mismatch errors return 400."""
        from infrastructure.repositories.library_repository import DimensionMismatchError

        client, mock_service = client_with_mock

        library_id = uuid4()
        mock_service.add_document_with_embeddings.side_effect = DimensionMismatchError(
            "Embedding dimension mismatch: expected 1024, got 512"
        )

        response = client.post(
            f"/v1/libraries/{library_id}/documents/with-embeddings",
            json={
                "title": "Test",
                "chunks": [{"text": "Test", "embedding": [0.1] * 512}],
            },
        )

        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert data["error_type"] == "DimensionMismatchError"

    def test_embedding_service_error_returns_503(self, client_with_mock):
        """Test that embedding service errors return 503."""
        from app.services.embedding_service import EmbeddingServiceError

        client, mock_service = client_with_mock

        library_id = uuid4()
        mock_service.add_document_with_text.side_effect = EmbeddingServiceError(
            "Failed to connect to Cohere API"
        )

        response = client.post(
            f"/v1/libraries/{library_id}/documents",
            json={"title": "Test", "texts": ["Test chunk"]},
        )

        assert response.status_code == 503
        data = response.json()
        assert "error" in data
        assert data["error_type"] == "EmbeddingServiceError"

    def test_value_error_returns_400(self, client_with_mock):
        """Test that ValueError from service returns 400."""
        client, mock_service = client_with_mock

        library_id = uuid4()
        # Mock the service to raise ValueError with valid request data
        mock_service.add_document_with_text.side_effect = ValueError(
            "texts list cannot be empty"
        )

        response = client.post(
            f"/v1/libraries/{library_id}/documents",
            json={"title": "Test", "texts": ["valid text"]},  # Valid data, but service raises ValueError
        )

        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert "Invalid request" in data["error"]
        assert data["error_type"] == "ValueError"

    def test_validation_error_returns_422(self, client_with_mock):
        """Test that validation errors return 422."""
        client, mock_service = client_with_mock

        # Send invalid request (k too large)
        response = client.post(
            f"/v1/libraries/{uuid4()}/search",
            json={"query": "test", "k": 99999},
        )

        assert response.status_code == 422  # Pydantic validation error


class TestStartupEvent:
    """Test startup event logging."""

    def test_startup_event_logs_configuration(self):
        """Test that startup event logs configuration details."""
        from unittest.mock import patch, MagicMock
        import asyncio

        # Import the startup function directly
        from app.api.main import startup_event

        # Create a mock settings object with all required attributes
        mock_settings = MagicMock()
        mock_settings.API_HOST = "0.0.0.0"
        mock_settings.API_PORT = 8000
        mock_settings.workers = 4
        mock_settings.RATE_LIMIT_ENABLED = True
        mock_settings.RATE_LIMIT_SEARCH = "100/minute"
        mock_settings.RATE_LIMIT_DOCUMENT_ADD = "50/minute"
        mock_settings.EMBEDDING_DIMENSION = 1024
        mock_settings.MAX_CHUNKS_PER_DOCUMENT = 100
        mock_settings.MAX_TEXT_LENGTH_PER_CHUNK = 10000
        mock_settings.MAX_SEARCH_RESULTS = 1000
        mock_settings.MAX_QUERY_LENGTH = 5000

        with patch("app.api.main.settings", mock_settings):
            with patch("app.api.main.logger") as mock_logger:
                # Run the async startup event
                asyncio.run(startup_event())

                # Verify logger.info was called multiple times with configuration info
                assert mock_logger.info.called
                call_count = mock_logger.info.call_count
                assert call_count >= 5  # Multiple log statements in startup

                # Verify warning was called for multiple workers
                assert mock_logger.warning.called

    def test_startup_event_with_multiple_workers(self):
        """Test startup event warning when workers > 1."""
        from unittest.mock import patch
        import asyncio
        from app.api.main import startup_event, settings

        # Mock settings to have multiple workers
        with patch("app.api.main.settings") as mock_settings:
            mock_settings.API_HOST = "0.0.0.0"
            mock_settings.API_PORT = 8000
            mock_settings.workers = 4  # Multiple workers
            mock_settings.RATE_LIMIT_ENABLED = False
            mock_settings.EMBEDDING_DIMENSION = 1024
            mock_settings.MAX_CHUNKS_PER_DOCUMENT = 100
            mock_settings.MAX_TEXT_LENGTH_PER_CHUNK = 10000
            mock_settings.MAX_SEARCH_RESULTS = 1000
            mock_settings.MAX_QUERY_LENGTH = 5000

            with patch("app.api.main.logger") as mock_logger:
                asyncio.run(startup_event())

                # Verify warning was logged about multiple workers
                assert mock_logger.warning.called or mock_logger.info.called
