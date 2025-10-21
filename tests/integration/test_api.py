"""
Integration tests for the REST API.

These tests verify end-to-end functionality of all API endpoints
using FastAPI's TestClient with real implementations (no mocking).

Requirements:
- COHERE_API_KEY environment variable must be set
- All components use real implementations including the embedding service
"""

import pytest
from fastapi.testclient import TestClient
from uuid import uuid4
import numpy as np
from pathlib import Path
import tempfile
import os

from app.api.main import app
from app.api.dependencies import get_library_repository
from infrastructure.repositories.library_repository import LibraryRepository


@pytest.fixture(scope="session", autouse=True)
def check_api_key():
    """Ensure COHERE_API_KEY is set before running integration tests."""
    if not os.getenv("COHERE_API_KEY"):
        pytest.skip("COHERE_API_KEY environment variable not set")


@pytest.fixture
def test_repository():
    """Create a test repository with temporary directory for isolation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo = LibraryRepository(Path(tmpdir))
        yield repo


@pytest.fixture
def client(test_repository):
    """
    Create a test client with dependency overrides.

    Only overrides the repository to use a temporary directory for test isolation.
    All other components (embedding service, indexes, etc.) use real implementations.
    """
    app.dependency_overrides[get_library_repository] = lambda: test_repository

    client = TestClient(app)
    yield client

    # Clean up overrides
    app.dependency_overrides.clear()


@pytest.fixture
def sample_library_request():
    """Sample request for creating a library."""
    return {
        "name": "Test Library",
        "description": "A test library for integration testing",
        "index_type": "brute_force",
        "embedding_model": "embed-english-v3.0",
    }


@pytest.fixture
def sample_document_request():
    """Sample request for adding a document."""
    return {
        "title": "Test Document",
        "texts": ["This is chunk 1", "This is chunk 2", "This is chunk 3"],
        "author": "Test Author",
        "document_type": "text",
        "source_url": "https://test.com/doc",
        "tags": ["test", "integration"],
    }


@pytest.fixture
def sample_document_with_embeddings_request():
    """Sample request for adding a document with embeddings."""
    # Use 1024 dimensions to match Cohere's default
    dimension = 1024
    chunks = []
    for i in range(3):
        vec = np.random.randn(dimension).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        chunks.append({"text": f"Chunk {i}", "embedding": vec.tolist()})

    return {
        "title": "Test Document with Embeddings",
        "chunks": chunks,
        "author": "Test Author",
        "document_type": "text",
    }


@pytest.mark.integration
class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_check(self, client):
        """Test the health check endpoint."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"
        assert "timestamp" in data


@pytest.mark.integration
class TestLibraryEndpoints:
    """Tests for library CRUD endpoints."""

    def test_create_library(self, client, sample_library_request):
        """Test creating a library."""
        response = client.post("/v1/libraries", json=sample_library_request)

        assert response.status_code == 201
        data = response.json()
        assert "id" in data
        assert data["name"] == sample_library_request["name"]
        assert data["metadata"]["index_type"] == sample_library_request["index_type"]

    def test_list_libraries_empty(self, client):
        """Test listing libraries when none exist."""
        response = client.get("/v1/libraries")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_list_libraries_with_data(self, client, sample_library_request):
        """Test listing libraries after creating one."""
        # Create a library
        create_response = client.post("/v1/libraries", json=sample_library_request)
        assert create_response.status_code == 201

        # List libraries
        list_response = client.get("/v1/libraries")
        assert list_response.status_code == 200
        libraries = list_response.json()

        assert len(libraries) >= 1
        assert any(lib["name"] == sample_library_request["name"] for lib in libraries)

    def test_get_library(self, client, sample_library_request):
        """Test getting a library by ID."""
        # Create library
        create_response = client.post("/v1/libraries", json=sample_library_request)
        library_id = create_response.json()["id"]

        # Get library
        get_response = client.get(f"/v1/libraries/{library_id}")
        assert get_response.status_code == 200
        data = get_response.json()
        assert data["id"] == library_id
        assert data["name"] == sample_library_request["name"]

    def test_get_nonexistent_library(self, client):
        """Test getting a library that doesn't exist."""
        fake_id = str(uuid4())
        response = client.get(f"/v1/libraries/{fake_id}")

        assert response.status_code == 404
        data = response.json()
        assert data["error_type"] == "LibraryNotFoundError"

    def test_delete_library(self, client, sample_library_request):
        """Test deleting a library."""
        # Create library
        create_response = client.post("/v1/libraries", json=sample_library_request)
        library_id = create_response.json()["id"]

        # Delete library
        delete_response = client.delete(f"/v1/libraries/{library_id}")
        assert delete_response.status_code == 204

        # Verify it's gone
        get_response = client.get(f"/v1/libraries/{library_id}")
        assert get_response.status_code == 404

    def test_delete_nonexistent_library(self, client):
        """Test deleting a library that doesn't exist."""
        fake_id = str(uuid4())
        response = client.delete(f"/v1/libraries/{fake_id}")

        assert response.status_code == 404

    def test_get_library_statistics(self, client, sample_library_request):
        """Test getting library statistics."""
        # Create library
        create_response = client.post("/v1/libraries", json=sample_library_request)
        library_id = create_response.json()["id"]

        # Get statistics
        stats_response = client.get(f"/v1/libraries/{library_id}/statistics")
        assert stats_response.status_code == 200

        stats = stats_response.json()
        assert "num_documents" in stats
        assert "num_chunks" in stats
        assert "vector_store_stats" in stats
        assert "index_stats" in stats
        assert stats["num_documents"] == 0  # Empty library


@pytest.mark.integration
class TestDocumentEndpoints:
    """Tests for document CRUD endpoints."""

    def test_add_document_with_text(self, client, sample_library_request, sample_document_request):
        """Test adding a document with text (auto-embedding)."""
        # Create library
        lib_response = client.post("/v1/libraries", json=sample_library_request)
        library_id = lib_response.json()["id"]

        # Add document
        doc_response = client.post(
            f"/v1/libraries/{library_id}/documents",
            json=sample_document_request
        )

        assert doc_response.status_code == 201
        doc = doc_response.json()
        assert "id" in doc
        assert doc["metadata"]["title"] == sample_document_request["title"]
        assert len(doc["chunks"]) == len(sample_document_request["texts"])

    def test_add_document_with_embeddings(
        self,
        client,
        sample_library_request,
        sample_document_with_embeddings_request
    ):
        """Test adding a document with pre-computed embeddings."""
        # Create library
        lib_response = client.post("/v1/libraries", json=sample_library_request)
        library_id = lib_response.json()["id"]

        # Add document with embeddings
        doc_response = client.post(
            f"/v1/libraries/{library_id}/documents/with-embeddings",
            json=sample_document_with_embeddings_request
        )

        assert doc_response.status_code == 201
        doc = doc_response.json()
        assert "id" in doc
        assert len(doc["chunks"]) == len(sample_document_with_embeddings_request["chunks"])

    def test_add_document_to_nonexistent_library(self, client, sample_document_request):
        """Test adding document to non-existent library."""
        fake_id = str(uuid4())
        response = client.post(
            f"/v1/libraries/{fake_id}/documents",
            json=sample_document_request
        )

        assert response.status_code == 404
        assert response.json()["error_type"] == "LibraryNotFoundError"

    def test_add_document_with_wrong_dimension(
        self,
        client,
        sample_library_request,
    ):
        """Test adding document with wrong embedding dimension raises error."""
        # Create library
        lib_response = client.post("/v1/libraries", json=sample_library_request)
        library_id = lib_response.json()["id"]

        # Try to add document with wrong dimension
        wrong_vec = np.random.randn(64).astype(np.float32)  # Wrong dimension
        wrong_vec = wrong_vec / np.linalg.norm(wrong_vec)

        wrong_request = {
            "title": "Wrong Dimension Doc",
            "chunks": [{"text": "test", "embedding": wrong_vec.tolist()}],
        }

        response = client.post(
            f"/v1/libraries/{library_id}/documents/with-embeddings",
            json=wrong_request
        )

        assert response.status_code == 400
        assert response.json()["error_type"] == "DimensionMismatchError"

    def test_get_document(self, client, sample_library_request, sample_document_request):
        """Test getting a document by ID."""
        # Create library and document
        lib_response = client.post("/v1/libraries", json=sample_library_request)
        library_id = lib_response.json()["id"]

        doc_response = client.post(
            f"/v1/libraries/{library_id}/documents",
            json=sample_document_request
        )
        document_id = doc_response.json()["id"]

        # Get document
        get_response = client.get(f"/v1/documents/{document_id}")
        assert get_response.status_code == 200
        doc = get_response.json()
        assert doc["id"] == document_id

    def test_get_nonexistent_document(self, client):
        """Test getting non-existent document."""
        fake_id = str(uuid4())
        response = client.get(f"/v1/documents/{fake_id}")

        assert response.status_code == 404
        assert response.json()["error_type"] == "DocumentNotFoundError"

    def test_delete_document(self, client, sample_library_request, sample_document_request):
        """Test deleting a document."""
        # Create library and document
        lib_response = client.post("/v1/libraries", json=sample_library_request)
        library_id = lib_response.json()["id"]

        doc_response = client.post(
            f"/v1/libraries/{library_id}/documents",
            json=sample_document_request
        )
        document_id = doc_response.json()["id"]

        # Delete document
        delete_response = client.delete(f"/v1/documents/{document_id}")
        assert delete_response.status_code == 204

        # Verify it's gone
        get_response = client.get(f"/v1/documents/{document_id}")
        assert get_response.status_code == 404

    def test_delete_nonexistent_document(self, client):
        """Test deleting non-existent document."""
        fake_id = str(uuid4())
        response = client.delete(f"/v1/documents/{fake_id}")

        assert response.status_code == 404


@pytest.mark.integration
class TestSearchEndpoints:
    """Tests for search endpoints."""

    def test_search_with_text(self, client, sample_library_request, sample_document_request):
        """Test searching with text query."""
        # Create library and document
        lib_response = client.post("/v1/libraries", json=sample_library_request)
        library_id = lib_response.json()["id"]

        client.post(
            f"/v1/libraries/{library_id}/documents",
            json=sample_document_request
        )

        # Search
        search_request = {
            "query": "chunk 1",
            "k": 5,
        }
        search_response = client.post(
            f"/v1/libraries/{library_id}/search",
            json=search_request
        )

        assert search_response.status_code == 200
        results = search_response.json()
        assert "results" in results
        assert "query_time_ms" in results
        assert "total_results" in results
        assert len(results["results"]) <= search_request["k"]

    def test_search_with_embedding(
        self,
        client,
        sample_library_request,
        sample_document_with_embeddings_request,
    ):
        """Test searching with pre-computed embedding."""
        # Create library and document
        lib_response = client.post("/v1/libraries", json=sample_library_request)
        library_id = lib_response.json()["id"]

        client.post(
            f"/v1/libraries/{library_id}/documents/with-embeddings",
            json=sample_document_with_embeddings_request
        )

        # Search with embedding (use one of the embeddings from the document)
        query_embedding = sample_document_with_embeddings_request["chunks"][0]["embedding"]
        search_request = {
            "embedding": query_embedding,
            "k": 5,
        }
        search_response = client.post(
            f"/v1/libraries/{library_id}/search/embedding",
            json=search_request
        )

        assert search_response.status_code == 200
        results = search_response.json()
        assert len(results["results"]) > 0
        # First result should be very close (exact match)
        assert results["results"][0]["distance"] < 0.01

    def test_search_empty_library(self, client, sample_library_request):
        """Test searching an empty library."""
        lib_response = client.post("/v1/libraries", json=sample_library_request)
        library_id = lib_response.json()["id"]

        search_request = {
            "query": "test query",
            "k": 5,
        }
        search_response = client.post(
            f"/v1/libraries/{library_id}/search",
            json=search_request
        )

        assert search_response.status_code == 200
        results = search_response.json()
        assert len(results["results"]) == 0

    def test_search_with_distance_threshold(
        self,
        client,
        sample_library_request,
        sample_document_with_embeddings_request,
    ):
        """Test search with distance threshold filtering."""
        # Create library and document
        lib_response = client.post("/v1/libraries", json=sample_library_request)
        library_id = lib_response.json()["id"]

        client.post(
            f"/v1/libraries/{library_id}/documents/with-embeddings",
            json=sample_document_with_embeddings_request
        )

        # Search with very restrictive threshold
        query_embedding = sample_document_with_embeddings_request["chunks"][0]["embedding"]
        search_request = {
            "embedding": query_embedding,
            "k": 10,
            "distance_threshold": 0.01,  # Very restrictive
        }
        search_response = client.post(
            f"/v1/libraries/{library_id}/search/embedding",
            json=search_request
        )

        assert search_response.status_code == 200
        results = search_response.json()
        # Only the exact match should be within threshold
        assert len(results["results"]) == 1


@pytest.mark.integration
class TestEndToEndWorkflows:
    """Tests for complete end-to-end workflows."""

    def test_complete_workflow(self, client, sample_library_request, sample_document_request):
        """Test a complete workflow from library creation to search."""
        # 1. Create library
        lib_response = client.post("/v1/libraries", json=sample_library_request)
        assert lib_response.status_code == 201
        library_id = lib_response.json()["id"]

        # 2. Add multiple documents
        doc_ids = []
        for i in range(3):
            doc_request = sample_document_request.copy()
            doc_request["title"] = f"Document {i}"
            doc_response = client.post(
                f"/v1/libraries/{library_id}/documents",
                json=doc_request
            )
            assert doc_response.status_code == 201
            doc_ids.append(doc_response.json()["id"])

        # 3. Check statistics
        stats_response = client.get(f"/v1/libraries/{library_id}/statistics")
        assert stats_response.status_code == 200
        stats = stats_response.json()
        assert stats["num_documents"] == 3
        assert stats["num_chunks"] == 3 * len(sample_document_request["texts"])

        # 4. Search
        search_response = client.post(
            f"/v1/libraries/{library_id}/search",
            json={"query": "chunk 1", "k": 5}
        )
        assert search_response.status_code == 200
        results = search_response.json()
        assert len(results["results"]) > 0

        # 5. Delete one document
        delete_response = client.delete(f"/v1/documents/{doc_ids[0]}")
        assert delete_response.status_code == 204

        # 6. Verify statistics updated
        stats_response = client.get(f"/v1/libraries/{library_id}/statistics")
        stats = stats_response.json()
        assert stats["num_documents"] == 2

        # 7. Delete library
        delete_lib_response = client.delete(f"/v1/libraries/{library_id}")
        assert delete_lib_response.status_code == 204

    def test_multiple_libraries_isolation(self, client, sample_library_request, sample_document_request):
        """Test that multiple libraries remain isolated."""
        # Create two libraries
        lib1_request = sample_library_request.copy()
        lib1_request["name"] = "Library 1"
        lib1_response = client.post("/v1/libraries", json=lib1_request)
        lib1_id = lib1_response.json()["id"]

        lib2_request = sample_library_request.copy()
        lib2_request["name"] = "Library 2"
        lib2_response = client.post("/v1/libraries", json=lib2_request)
        lib2_id = lib2_response.json()["id"]

        # Add document to library 1
        doc_response = client.post(
            f"/v1/libraries/{lib1_id}/documents",
            json=sample_document_request
        )
        assert doc_response.status_code == 201

        # Verify library 1 has documents
        stats1 = client.get(f"/v1/libraries/{lib1_id}/statistics").json()
        assert stats1["num_documents"] == 1

        # Verify library 2 is still empty
        stats2 = client.get(f"/v1/libraries/{lib2_id}/statistics").json()
        assert stats2["num_documents"] == 0
