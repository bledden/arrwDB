"""
Comprehensive unit tests for API model validation.

Tests Pydantic v2 field validators, input limits, and response models.
"""

import pytest
from uuid import uuid4
from datetime import datetime
from app.api.models import (
    AddDocumentRequest,
    AddDocumentWithEmbeddingsRequest,
    ChunkWithEmbedding,
    SearchRequest,
    SearchWithEmbeddingRequest,
    CreateLibraryRequest,
    ChunkResponseSlim,
    DocumentResponseSlim,
    SearchResultResponseSlim,
    SearchResponseSlim,
)
from app.config import settings
from app.models.base import Chunk, ChunkMetadata, Document, DocumentMetadata


class TestCreateLibraryRequestValidation:
    """Test CreateLibraryRequest validation."""

    def test_valid_library_request(self):
        """Test creating a valid library request."""
        req = CreateLibraryRequest(
            name="Test Library",
            description="A test library",
            index_type="brute_force",
        )
        assert req.name == "Test Library"
        assert req.description == "A test library"
        assert req.index_type == "brute_force"

    def test_library_request_with_minimal_fields(self):
        """Test library request with only required fields."""
        req = CreateLibraryRequest(name="Minimal Library")
        assert req.name == "Minimal Library"
        assert req.index_type == "brute_force"  # default

    def test_all_index_types_valid(self):
        """Test all supported index types are valid."""
        for index_type in ["brute_force", "kd_tree", "lsh", "hnsw"]:
            req = CreateLibraryRequest(name="Test", index_type=index_type)
            assert req.index_type == index_type


class TestAddDocumentRequestValidation:
    """Test AddDocumentRequest validation with configurable limits."""

    def test_valid_document_request(self):
        """Test creating a valid document request."""
        req = AddDocumentRequest(
            title="Test Document",
            texts=["Chunk 1", "Chunk 2", "Chunk 3"],
            author="Test Author",
            document_type="text",
            tags=["test", "validation"],
        )
        assert req.title == "Test Document"
        assert len(req.texts) == 3
        assert req.author == "Test Author"
        assert len(req.tags) == 2

    def test_document_request_minimal_fields(self):
        """Test document request with only required fields."""
        req = AddDocumentRequest(
            title="Minimal",
            texts=["Single chunk"],
        )
        assert req.title == "Minimal"
        assert len(req.texts) == 1
        assert req.document_type == "text"  # default
        assert req.tags == []  # default

    def test_exactly_max_chunks_allowed(self):
        """Test document with exactly MAX_CHUNKS_PER_DOCUMENT chunks."""
        texts = [f"Chunk {i}" for i in range(settings.MAX_CHUNKS_PER_DOCUMENT)]
        req = AddDocumentRequest(title="Max Chunks", texts=texts)
        assert len(req.texts) == settings.MAX_CHUNKS_PER_DOCUMENT

    def test_too_many_chunks_rejected(self):
        """Test that exceeding MAX_CHUNKS_PER_DOCUMENT raises ValueError."""
        texts = [f"Chunk {i}" for i in range(settings.MAX_CHUNKS_PER_DOCUMENT + 1)]
        with pytest.raises(ValueError, match="Too many chunks"):
            AddDocumentRequest(title="Too Many", texts=texts)

    def test_exactly_max_text_length_allowed(self):
        """Test chunk with exactly MAX_TEXT_LENGTH_PER_CHUNK characters."""
        text = "a" * settings.MAX_TEXT_LENGTH_PER_CHUNK
        req = AddDocumentRequest(title="Max Length", texts=[text])
        assert len(req.texts[0]) == settings.MAX_TEXT_LENGTH_PER_CHUNK

    def test_text_too_long_rejected(self):
        """Test that exceeding MAX_TEXT_LENGTH_PER_CHUNK raises ValueError."""
        long_text = "a" * (settings.MAX_TEXT_LENGTH_PER_CHUNK + 1)
        with pytest.raises(ValueError, match="too long"):
            AddDocumentRequest(title="Too Long", texts=[long_text])

    def test_multiple_chunks_with_one_too_long(self):
        """Test that validator checks all chunks, not just first."""
        texts = [
            "Normal chunk 1",
            "Normal chunk 2",
            "a" * (settings.MAX_TEXT_LENGTH_PER_CHUNK + 1),  # This one is too long
        ]
        with pytest.raises(ValueError, match="Chunk 2 too long"):
            AddDocumentRequest(title="Mixed", texts=texts)

    def test_empty_title_rejected(self):
        """Test that empty title raises ValueError."""
        with pytest.raises(ValueError):
            AddDocumentRequest(title="", texts=["chunk"])

    def test_empty_texts_list_rejected(self):
        """Test that empty texts list raises ValueError."""
        with pytest.raises(ValueError):
            AddDocumentRequest(title="Empty", texts=[])


class TestChunkWithEmbeddingValidation:
    """Test ChunkWithEmbedding validation."""

    def test_valid_chunk_with_embedding(self):
        """Test creating a valid chunk with embedding."""
        chunk = ChunkWithEmbedding(
            text="Test chunk text",
            embedding=[0.1] * 1024,
        )
        assert chunk.text == "Test chunk text"
        assert len(chunk.embedding) == 1024

    def test_chunk_text_exactly_max_length(self):
        """Test chunk with exactly MAX_TEXT_LENGTH_PER_CHUNK."""
        text = "b" * settings.MAX_TEXT_LENGTH_PER_CHUNK
        chunk = ChunkWithEmbedding(text=text, embedding=[0.1] * 1024)
        assert len(chunk.text) == settings.MAX_TEXT_LENGTH_PER_CHUNK

    def test_chunk_text_too_long_rejected(self):
        """Test that chunk text exceeding limit raises ValueError."""
        long_text = "c" * (settings.MAX_TEXT_LENGTH_PER_CHUNK + 1)
        with pytest.raises(ValueError, match="Text too long"):
            ChunkWithEmbedding(text=long_text, embedding=[0.1] * 1024)

    def test_empty_embedding_rejected(self):
        """Test that empty embedding is rejected."""
        with pytest.raises(ValueError):
            ChunkWithEmbedding(text="Test", embedding=[])


class TestAddDocumentWithEmbeddingsRequestValidation:
    """Test AddDocumentWithEmbeddingsRequest validation."""

    def test_valid_request_with_embeddings(self):
        """Test creating a valid request with pre-computed embeddings."""
        chunks = [
            ChunkWithEmbedding(text=f"Chunk {i}", embedding=[0.1 * i] * 1024)
            for i in range(3)
        ]
        req = AddDocumentWithEmbeddingsRequest(
            title="Test Document",
            chunks=chunks,
            author="Test Author",
        )
        assert req.title == "Test Document"
        assert len(req.chunks) == 3
        assert req.author == "Test Author"

    def test_exactly_max_chunks_with_embeddings(self):
        """Test document with exactly MAX_CHUNKS_PER_DOCUMENT chunks."""
        chunks = [
            ChunkWithEmbedding(text=f"Chunk {i}", embedding=[0.1] * 1024)
            for i in range(settings.MAX_CHUNKS_PER_DOCUMENT)
        ]
        req = AddDocumentWithEmbeddingsRequest(title="Max Chunks", chunks=chunks)
        assert len(req.chunks) == settings.MAX_CHUNKS_PER_DOCUMENT

    def test_too_many_chunks_with_embeddings_rejected(self):
        """Test that exceeding MAX_CHUNKS_PER_DOCUMENT raises ValueError."""
        chunks = [
            ChunkWithEmbedding(text=f"Chunk {i}", embedding=[0.1] * 1024)
            for i in range(settings.MAX_CHUNKS_PER_DOCUMENT + 1)
        ]
        with pytest.raises(ValueError, match="Too many chunks"):
            AddDocumentWithEmbeddingsRequest(title="Too Many", chunks=chunks)


class TestSearchRequestValidation:
    """Test SearchRequest validation with configurable limits."""

    def test_valid_search_request(self):
        """Test creating a valid search request."""
        req = SearchRequest(query="machine learning", k=10)
        assert req.query == "machine learning"
        assert req.k == 10
        assert req.distance_threshold is None

    def test_search_request_with_all_fields(self):
        """Test search request with all optional fields."""
        req = SearchRequest(
            query="deep learning",
            k=20,
            distance_threshold=0.5,
        )
        assert req.query == "deep learning"
        assert req.k == 20
        assert req.distance_threshold == 0.5

    def test_default_k_value(self):
        """Test that k defaults to 10."""
        req = SearchRequest(query="test")
        assert req.k == 10

    def test_minimum_k_value(self):
        """Test k=1 is valid."""
        req = SearchRequest(query="test", k=1)
        assert req.k == 1

    def test_exactly_max_k_allowed(self):
        """Test k with exactly MAX_SEARCH_RESULTS."""
        req = SearchRequest(query="test", k=settings.MAX_SEARCH_RESULTS)
        assert req.k == settings.MAX_SEARCH_RESULTS

    def test_k_exceeds_max_rejected(self):
        """Test that k > MAX_SEARCH_RESULTS raises ValueError."""
        with pytest.raises(ValueError, match="Too many results"):
            SearchRequest(query="test", k=settings.MAX_SEARCH_RESULTS + 1)

    def test_exactly_max_query_length_allowed(self):
        """Test query with exactly MAX_QUERY_LENGTH characters."""
        query = "q" * settings.MAX_QUERY_LENGTH
        req = SearchRequest(query=query, k=10)
        assert len(req.query) == settings.MAX_QUERY_LENGTH

    def test_query_too_long_rejected(self):
        """Test that query exceeding MAX_QUERY_LENGTH raises ValueError."""
        long_query = "q" * (settings.MAX_QUERY_LENGTH + 1)
        with pytest.raises(ValueError, match="Query too long"):
            SearchRequest(query=long_query, k=10)

    def test_distance_threshold_min_value(self):
        """Test distance_threshold minimum value (0.0)."""
        req = SearchRequest(query="test", k=10, distance_threshold=0.0)
        assert req.distance_threshold == 0.0

    def test_distance_threshold_max_value(self):
        """Test distance_threshold maximum value (2.0)."""
        req = SearchRequest(query="test", k=10, distance_threshold=2.0)
        assert req.distance_threshold == 2.0


class TestSearchWithEmbeddingRequestValidation:
    """Test SearchWithEmbeddingRequest validation."""

    def test_valid_search_with_embedding(self):
        """Test creating a valid search with embedding request."""
        embedding = [0.1 * i for i in range(1024)]
        req = SearchWithEmbeddingRequest(embedding=embedding, k=10)
        assert len(req.embedding) == 1024
        assert req.k == 10

    def test_exactly_max_k_allowed(self):
        """Test k with exactly MAX_SEARCH_RESULTS."""
        embedding = [0.1] * 1024
        req = SearchWithEmbeddingRequest(
            embedding=embedding, k=settings.MAX_SEARCH_RESULTS
        )
        assert req.k == settings.MAX_SEARCH_RESULTS

    def test_k_exceeds_max_rejected(self):
        """Test that k > MAX_SEARCH_RESULTS raises ValueError."""
        embedding = [0.1] * 1024
        with pytest.raises(ValueError, match="Too many results"):
            SearchWithEmbeddingRequest(
                embedding=embedding, k=settings.MAX_SEARCH_RESULTS + 1
            )

    def test_empty_embedding_rejected(self):
        """Test that empty embedding is rejected."""
        with pytest.raises(ValueError):
            SearchWithEmbeddingRequest(embedding=[], k=10)

    def test_default_k_value(self):
        """Test that k defaults to 10."""
        embedding = [0.1] * 1024
        req = SearchWithEmbeddingRequest(embedding=embedding)
        assert req.k == 10


class TestSlimResponseModels:
    """Test slim response models exclude embeddings."""

    def test_chunk_response_slim_creation(self):
        """Test creating ChunkResponseSlim from data."""
        chunk_data = {
            "id": uuid4(),
            "text": "Test chunk text",
            "metadata": {
                "created_at": datetime.utcnow(),
                "page_number": None,
                "chunk_index": 0,
                "source_document_id": uuid4(),
            },
        }
        slim = ChunkResponseSlim(**chunk_data)
        assert slim.text == "Test chunk text"
        assert hasattr(slim, "id")
        assert hasattr(slim, "metadata")

    def test_chunk_response_slim_no_embedding_field(self):
        """Test ChunkResponseSlim doesn't have embedding in serialized form."""
        chunk_data = {
            "id": uuid4(),
            "text": "Test",
            "metadata": {
                "created_at": datetime.utcnow(),
                "page_number": None,
                "chunk_index": 0,
                "source_document_id": uuid4(),
            },
        }
        slim = ChunkResponseSlim(**chunk_data)
        dumped = slim.model_dump()
        assert "embedding" not in dumped

    def test_convert_full_chunk_to_slim(self):
        """Test converting a full Chunk (with embedding) to slim."""
        chunk = Chunk(
            text="Full chunk with embedding",
            embedding=[0.5] * 1024,
            metadata=ChunkMetadata(
                created_at=datetime.utcnow(),
                chunk_index=0,
                source_document_id=uuid4(),
            ),
        )

        # Convert to slim
        slim = ChunkResponseSlim.model_validate(chunk)
        assert slim.text == chunk.text
        assert slim.id == chunk.id

        # Verify embedding excluded in serialization
        dumped = slim.model_dump()
        assert "embedding" not in dumped

    def test_document_response_slim_has_slim_chunks(self):
        """Test DocumentResponseSlim contains ChunkResponseSlim objects."""
        doc = Document(
            chunks=[
                Chunk(
                    text=f"Chunk {i}",
                    embedding=[0.1 * i] * 1024,
                    metadata=ChunkMetadata(
                        created_at=datetime.utcnow(),
                        chunk_index=i,
                        source_document_id=uuid4(),
                    ),
                )
                for i in range(3)
            ],
            metadata=DocumentMetadata(
                title="Test Document",
                created_at=datetime.utcnow(),
                document_type="text",
                tags=["test"],
            ),
        )

        # Convert to slim
        slim = DocumentResponseSlim.model_validate(doc)
        assert len(slim.chunks) == 3
        assert all(isinstance(c, ChunkResponseSlim) for c in slim.chunks)

        # Verify no embeddings in any chunk
        dumped = slim.model_dump()
        for chunk_dict in dumped["chunks"]:
            assert "embedding" not in chunk_dict

    def test_search_response_slim_structure(self):
        """Test SearchResponseSlim contains SearchResultResponseSlim."""
        doc_id = uuid4()
        slim_chunk = ChunkResponseSlim(
            id=uuid4(),
            text="Test chunk",
            metadata={
                "created_at": datetime.utcnow(),
                "page_number": None,
                "chunk_index": 0,
                "source_document_id": doc_id,
            },
        )

        result = SearchResultResponseSlim(
            chunk=slim_chunk,
            distance=0.25,
            document_id=doc_id,
            document_title="Test Document",
        )

        response = SearchResponseSlim(
            results=[result],
            query_time_ms=15.75,
            total_results=1,
        )

        assert len(response.results) == 1
        assert response.query_time_ms == 15.75
        assert response.total_results == 1
        assert isinstance(response.results[0], SearchResultResponseSlim)

        # Verify no embeddings in serialized response
        dumped = response.model_dump()
        assert "embedding" not in dumped["results"][0]["chunk"]


class TestModelSerialization:
    """Test model serialization and deserialization."""

    def test_add_document_request_serialization(self):
        """Test AddDocumentRequest JSON serialization."""
        req = AddDocumentRequest(
            title="Serialization Test",
            texts=["chunk1", "chunk2"],
            author="Test Author",
            tags=["tag1", "tag2"],
        )
        json_data = req.model_dump()

        assert json_data["title"] == "Serialization Test"
        assert len(json_data["texts"]) == 2
        assert json_data["author"] == "Test Author"
        assert len(json_data["tags"]) == 2

    def test_search_request_serialization(self):
        """Test SearchRequest JSON serialization."""
        req = SearchRequest(query="test query", k=5, distance_threshold=0.8)
        json_data = req.model_dump()

        assert json_data["query"] == "test query"
        assert json_data["k"] == 5
        assert json_data["distance_threshold"] == 0.8

    def test_search_response_slim_serialization(self):
        """Test SearchResponseSlim serialization."""
        response = SearchResponseSlim(
            results=[],
            query_time_ms=12.34,
            total_results=0,
        )
        json_data = response.model_dump()

        assert json_data["query_time_ms"] == 12.34
        assert json_data["total_results"] == 0
        assert json_data["results"] == []
