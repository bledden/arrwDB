"""
Test suite for app/api/streaming.py

Coverage targets:
- NDJSON document ingestion (batch upload)
- Streaming search results
- SSE (Server-Sent Events) event streaming
- Error handling for malformed data
- Empty lines and whitespace handling
- Success/failure counting
"""

import asyncio
import json
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest
from fastapi import Request
from fastapi.responses import JSONResponse

from app.api.streaming import stream_documents_fixed, stream_search_results
from app.models.base import Chunk, ChunkMetadata, Document, DocumentMetadata


class TestNDJSONDocumentIngestion:
    """Test NDJSON document ingestion endpoint."""

    @pytest.fixture
    def mock_library_service(self):
        """Mock library service."""
        service = Mock()
        return service

    @pytest.fixture
    def mock_request(self):
        """Mock FastAPI Request."""
        request = Mock(spec=Request)
        return request

    @pytest.mark.asyncio
    async def test_ndjson_ingestion_success(self, mock_library_service, mock_request):
        """Test successful NDJSON batch ingestion."""
        library_id = uuid4()

        # Create NDJSON data
        ndjson_data = '\n'.join([
            json.dumps({"title": "Doc 1", "texts": ["Text 1"]}),
            json.dumps({"title": "Doc 2", "texts": ["Text 2"]}),
            json.dumps({"title": "Doc 3", "texts": ["Text 3"]}),
        ])

        mock_request.body = AsyncMock(return_value=ndjson_data.encode('utf-8'))

        # Mock successful document additions
        mock_doc = Mock()
        mock_doc.id = uuid4()
        mock_doc.chunks = [Mock(), Mock()]  # 2 chunks
        mock_library_service.add_document_with_text.return_value = mock_doc

        # Execute
        response = await stream_documents_fixed(
            library_id=library_id,
            request=mock_request,
            service=mock_library_service
        )

        # Verify response
        assert isinstance(response, JSONResponse)
        response_data = json.loads(response.body.decode())

        assert response_data["successful"] == 3
        assert response_data["failed"] == 0
        assert len(response_data["results"]) == 3

        # Verify all results are successful
        for result in response_data["results"]:
            assert result["success"] is True
            assert "document_id" in result
            assert "num_chunks" in result

    @pytest.mark.asyncio
    async def test_ndjson_ingestion_partial_failure(self, mock_library_service, mock_request):
        """Test NDJSON ingestion with some documents failing."""
        library_id = uuid4()

        ndjson_data = '\n'.join([
            json.dumps({"title": "Doc 1", "texts": ["Text 1"]}),
            json.dumps({"title": "Doc 2", "texts": ["Text 2"]}),
            json.dumps({"title": "Doc 3", "texts": ["Text 3"]}),
        ])

        mock_request.body = AsyncMock(return_value=ndjson_data.encode('utf-8'))

        # Mock: first succeeds, second fails, third succeeds
        mock_doc = Mock()
        mock_doc.id = uuid4()
        mock_doc.chunks = [Mock()]

        def side_effect_func(*args, **kwargs):
            # Fail on second call
            if mock_library_service.add_document_with_text.call_count == 2:
                raise ValueError("Document processing failed")
            return mock_doc

        mock_library_service.add_document_with_text.side_effect = side_effect_func

        response = await stream_documents_fixed(
            library_id=library_id,
            request=mock_request,
            service=mock_library_service
        )

        response_data = json.loads(response.body.decode())

        assert response_data["successful"] == 2
        assert response_data["failed"] == 1

        # Check that one result has error
        errors = [r for r in response_data["results"] if not r.get("success")]
        assert len(errors) == 1
        assert "error" in errors[0]

    @pytest.mark.asyncio
    async def test_ndjson_ingestion_empty_lines(self, mock_library_service, mock_request):
        """Test that empty lines are skipped."""
        library_id = uuid4()

        ndjson_data = '\n'.join([
            json.dumps({"title": "Doc 1", "texts": ["Text 1"]}),
            "",  # Empty line
            "   ",  # Whitespace only
            json.dumps({"title": "Doc 2", "texts": ["Text 2"]}),
            "",  # Another empty line
        ])

        mock_request.body = AsyncMock(return_value=ndjson_data.encode('utf-8'))

        mock_doc = Mock()
        mock_doc.id = uuid4()
        mock_doc.chunks = [Mock()]
        mock_library_service.add_document_with_text.return_value = mock_doc

        response = await stream_documents_fixed(
            library_id=library_id,
            request=mock_request,
            service=mock_library_service
        )

        response_data = json.loads(response.body.decode())

        # Should only process 2 valid documents
        assert response_data["successful"] == 2
        assert response_data["failed"] == 0
        assert len(response_data["results"]) == 2

    @pytest.mark.asyncio
    async def test_ndjson_ingestion_malformed_json(self, mock_library_service, mock_request):
        """Test handling of malformed JSON lines."""
        library_id = uuid4()

        ndjson_data = '\n'.join([
            json.dumps({"title": "Doc 1", "texts": ["Text 1"]}),
            "{ invalid json here }",  # Malformed
            json.dumps({"title": "Doc 2", "texts": ["Text 2"]}),
        ])

        mock_request.body = AsyncMock(return_value=ndjson_data.encode('utf-8'))

        mock_doc = Mock()
        mock_doc.id = uuid4()
        mock_doc.chunks = [Mock()]
        mock_library_service.add_document_with_text.return_value = mock_doc

        response = await stream_documents_fixed(
            library_id=library_id,
            request=mock_request,
            service=mock_library_service
        )

        response_data = json.loads(response.body.decode())

        # 2 successful, 1 failed (malformed JSON)
        assert response_data["successful"] == 2
        assert response_data["failed"] == 1

    @pytest.mark.asyncio
    async def test_ndjson_ingestion_with_metadata(self, mock_library_service, mock_request):
        """Test NDJSON ingestion with document metadata."""
        library_id = uuid4()

        ndjson_data = json.dumps({
            "title": "Research Paper",
            "texts": ["Abstract here", "Introduction here"],
            "author": "John Doe",
            "document_type": "research",
            "source_url": "https://example.com/paper.pdf",
            "tags": ["AI", "ML"]
        })

        mock_request.body = AsyncMock(return_value=ndjson_data.encode('utf-8'))

        mock_doc = Mock()
        mock_doc.id = uuid4()
        mock_doc.chunks = [Mock(), Mock()]
        mock_library_service.add_document_with_text.return_value = mock_doc

        response = await stream_documents_fixed(
            library_id=library_id,
            request=mock_request,
            service=mock_library_service
        )

        # Verify service was called with metadata
        call_kwargs = mock_library_service.add_document_with_text.call_args[1]
        assert call_kwargs["author"] == "John Doe"
        assert call_kwargs["document_type"] == "research"
        assert call_kwargs["source_url"] == "https://example.com/paper.pdf"
        assert call_kwargs["tags"] == ["AI", "ML"]

    @pytest.mark.asyncio
    async def test_ndjson_ingestion_missing_title(self, mock_library_service, mock_request):
        """Test NDJSON ingestion with missing title (should default to 'Untitled')."""
        library_id = uuid4()

        ndjson_data = json.dumps({"texts": ["Some text"]})  # No title

        mock_request.body = AsyncMock(return_value=ndjson_data.encode('utf-8'))

        mock_doc = Mock()
        mock_doc.id = uuid4()
        mock_doc.chunks = [Mock()]
        mock_library_service.add_document_with_text.return_value = mock_doc

        response = await stream_documents_fixed(
            library_id=library_id,
            request=mock_request,
            service=mock_library_service
        )

        # Verify service was called with "Untitled"
        call_kwargs = mock_library_service.add_document_with_text.call_args[1]
        assert call_kwargs["title"] == "Untitled"


class TestStreamingSearch:
    """Test streaming search results endpoint."""

    @pytest.fixture
    def mock_library_service(self):
        """Mock library service."""
        service = Mock()
        return service

    @pytest.fixture
    def mock_request(self):
        """Mock FastAPI Request."""
        request = Mock(spec=Request)
        return request

    @pytest.fixture
    def mock_search_results(self):
        """Create mock search results."""
        # Create dummy embeddings (required by Chunk model)
        dummy_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]

        chunk1 = Chunk(
            id=uuid4(),
            text="Test chunk 1",
            embedding=dummy_embedding,
            metadata=ChunkMetadata(
                source_document_id=uuid4(),
                chunk_index=0
            )
        )

        chunk2 = Chunk(
            id=uuid4(),
            text="Test chunk 2",
            embedding=dummy_embedding,
            metadata=ChunkMetadata(
                source_document_id=uuid4(),
                chunk_index=1
            )
        )

        return [(chunk1, 0.15), (chunk2, 0.25)]

    @pytest.mark.asyncio
    async def test_streaming_search_success(self, mock_library_service, mock_request, mock_search_results):
        """Test successful streaming search."""
        library_id = uuid4()

        search_params = {
            "query": "test query",
            "k": 10,
            "distance_threshold": 0.5
        }

        mock_request.body = AsyncMock(return_value=json.dumps(search_params).encode('utf-8'))
        mock_library_service.search_with_text.return_value = mock_search_results

        response = await stream_search_results(
            library_id=library_id,
            request=mock_request,
            service=mock_library_service
        )

        response_data = json.loads(response.body.decode())

        assert response_data["total"] == 2
        assert len(response_data["results"]) == 2

        # Verify result format
        result1 = response_data["results"][0]
        assert result1["rank"] == 1
        assert "chunk_id" in result1
        assert "document_id" in result1
        assert result1["distance"] == 0.15
        assert result1["text"] == "Test chunk 1"
        assert "metadata" in result1

    @pytest.mark.asyncio
    async def test_streaming_search_with_distance_threshold(self, mock_library_service, mock_request, mock_search_results):
        """Test search with distance threshold."""
        library_id = uuid4()

        search_params = {
            "query": "test query",
            "k": 5,
            "distance_threshold": 0.3
        }

        mock_request.body = AsyncMock(return_value=json.dumps(search_params).encode('utf-8'))
        mock_library_service.search_with_text.return_value = mock_search_results

        response = await stream_search_results(
            library_id=library_id,
            request=mock_request,
            service=mock_library_service
        )

        # Verify service was called with correct parameters
        call_kwargs = mock_library_service.search_with_text.call_args[1]
        assert call_kwargs["query_text"] == "test query"
        assert call_kwargs["k"] == 5
        assert call_kwargs["distance_threshold"] == 0.3

    @pytest.mark.asyncio
    async def test_streaming_search_empty_results(self, mock_library_service, mock_request):
        """Test search with no results."""
        library_id = uuid4()

        search_params = {"query": "nonexistent query", "k": 10}

        mock_request.body = AsyncMock(return_value=json.dumps(search_params).encode('utf-8'))
        mock_library_service.search_with_text.return_value = []

        response = await stream_search_results(
            library_id=library_id,
            request=mock_request,
            service=mock_library_service
        )

        response_data = json.loads(response.body.decode())

        assert response_data["total"] == 0
        assert response_data["results"] == []

    @pytest.mark.asyncio
    async def test_streaming_search_default_k(self, mock_library_service, mock_request, mock_search_results):
        """Test that k defaults to 10 if not specified."""
        library_id = uuid4()

        search_params = {"query": "test query"}  # No k specified

        mock_request.body = AsyncMock(return_value=json.dumps(search_params).encode('utf-8'))
        mock_library_service.search_with_text.return_value = mock_search_results

        response = await stream_search_results(
            library_id=library_id,
            request=mock_request,
            service=mock_library_service
        )

        # Verify default k=10 was used
        call_kwargs = mock_library_service.search_with_text.call_args[1]
        assert call_kwargs["k"] == 10

    @pytest.mark.asyncio
    async def test_streaming_search_result_ranking(self, mock_library_service, mock_request, mock_search_results):
        """Test that search results are ranked correctly."""
        library_id = uuid4()

        search_params = {"query": "test", "k": 5}

        mock_request.body = AsyncMock(return_value=json.dumps(search_params).encode('utf-8'))
        mock_library_service.search_with_text.return_value = mock_search_results

        response = await stream_search_results(
            library_id=library_id,
            request=mock_request,
            service=mock_library_service
        )

        response_data = json.loads(response.body.decode())

        # Ranks should be 1, 2, 3...
        for i, result in enumerate(response_data["results"], start=1):
            assert result["rank"] == i


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.fixture
    def mock_library_service(self):
        """Mock library service."""
        return Mock()

    @pytest.fixture
    def mock_request(self):
        """Mock FastAPI Request."""
        return Mock(spec=Request)

    @pytest.mark.asyncio
    async def test_ndjson_empty_body(self, mock_library_service, mock_request):
        """Test NDJSON ingestion with empty body."""
        library_id = uuid4()

        mock_request.body = AsyncMock(return_value=b"")

        response = await stream_documents_fixed(
            library_id=library_id,
            request=mock_request,
            service=mock_library_service
        )

        response_data = json.loads(response.body.decode())

        assert response_data["successful"] == 0
        assert response_data["failed"] == 0
        assert len(response_data["results"]) == 0

    @pytest.mark.asyncio
    async def test_ndjson_only_whitespace(self, mock_library_service, mock_request):
        """Test NDJSON ingestion with only whitespace."""
        library_id = uuid4()

        mock_request.body = AsyncMock(return_value=b"   \n\n   \n   ")

        response = await stream_documents_fixed(
            library_id=library_id,
            request=mock_request,
            service=mock_library_service
        )

        response_data = json.loads(response.body.decode())

        assert response_data["successful"] == 0
        assert response_data["failed"] == 0
