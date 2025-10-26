"""
API models (DTOs) for request and response serialization.

These are separate from the domain models and handle API-level concerns
like validation, documentation, and serialization.
"""

from datetime import datetime
from typing import Any, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from app.config import settings

# Request Models


class CreateLibraryRequest(BaseModel):
    """Request model for creating a library."""

    name: str = Field(..., min_length=1, max_length=255, description="Library name")
    description: Optional[str] = Field(None, description="Optional description")
    index_type: str = Field(
        default="brute_force",
        pattern="^(brute_force|kd_tree|lsh|hnsw)$",
        description="Index type: brute_force, kd_tree, lsh, or hnsw",
    )
    embedding_model: Optional[str] = Field(
        None, description="Optional embedding model override"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "name": "Research Papers",
                "description": "Collection of AI research papers",
                "index_type": "hnsw",
            }
        }


class AddDocumentRequest(BaseModel):
    """Request model for adding a document with text chunks."""

    title: str = Field(..., min_length=1, description="Document title")
    texts: List[str] = Field(
        ..., min_length=1, description="List of text chunks"
    )
    author: Optional[str] = Field(None, description="Document author")
    document_type: str = Field(default="text", description="Type of document")
    source_url: Optional[str] = Field(None, description="Source URL")
    tags: List[str] = Field(default_factory=list, description="Tags")

    @field_validator("texts")
    @classmethod
    def validate_texts_count(cls, v: List[str]) -> List[str]:
        """Enforce maximum number of chunks per document."""
        if len(v) > settings.MAX_CHUNKS_PER_DOCUMENT:
            raise ValueError(
                f"Too many chunks: {len(v)}. Maximum allowed: {settings.MAX_CHUNKS_PER_DOCUMENT}"
            )
        return v

    @field_validator("texts")
    @classmethod
    def validate_text_length(cls, v: List[str]) -> List[str]:
        """Enforce maximum text length per chunk."""
        for i, text in enumerate(v):
            if len(text) > settings.MAX_TEXT_LENGTH_PER_CHUNK:
                raise ValueError(
                    f"Chunk {i} too long: {len(text)} chars. Maximum allowed: {settings.MAX_TEXT_LENGTH_PER_CHUNK}"
                )
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "title": "Introduction to Machine Learning",
                "texts": [
                    "Machine learning is a subset of artificial intelligence...",
                    "Supervised learning involves training with labeled data...",
                ],
                "author": "John Doe",
                "tags": ["machine-learning", "ai"],
            }
        }


class AddDocumentWithEmbeddingsRequest(BaseModel):
    """Request model for adding a document with pre-computed embeddings."""

    title: str = Field(..., min_length=1, description="Document title")
    chunks: List["ChunkWithEmbedding"] = Field(
        ..., min_length=1, description="List of text-embedding pairs"
    )
    author: Optional[str] = Field(None, description="Document author")
    document_type: str = Field(default="text", description="Type of document")
    source_url: Optional[str] = Field(None, description="Source URL")
    tags: List[str] = Field(default_factory=list, description="Tags")

    @field_validator("chunks")
    @classmethod
    def validate_chunks_count(cls, v: List["ChunkWithEmbedding"]) -> List["ChunkWithEmbedding"]:
        """Enforce maximum number of chunks per document."""
        if len(v) > settings.MAX_CHUNKS_PER_DOCUMENT:
            raise ValueError(
                f"Too many chunks: {len(v)}. Maximum allowed: {settings.MAX_CHUNKS_PER_DOCUMENT}"
            )
        return v


class ChunkWithEmbedding(BaseModel):
    """Model for a chunk with pre-computed embedding."""

    text: str = Field(..., min_length=1)
    embedding: List[float] = Field(..., min_length=1)

    @field_validator("text")
    @classmethod
    def validate_text_length(cls, v: str) -> str:
        """Enforce maximum text length per chunk."""
        if len(v) > settings.MAX_TEXT_LENGTH_PER_CHUNK:
            raise ValueError(
                f"Text too long: {len(v)} chars. Maximum allowed: {settings.MAX_TEXT_LENGTH_PER_CHUNK}"
            )
        return v


class SearchRequest(BaseModel):
    """Request model for searching with text query."""

    query: str = Field(..., min_length=1, description="Search query text")
    k: int = Field(default=10, ge=1, description="Number of results")
    distance_threshold: Optional[float] = Field(
        None,
        ge=0.0,
        le=2.0,
        description="Maximum distance threshold (0-2 for cosine)",
    )

    @field_validator("k")
    @classmethod
    def validate_k(cls, v: int) -> int:
        """Enforce maximum search results limit."""
        if v > settings.MAX_SEARCH_RESULTS:
            raise ValueError(
                f"Too many results requested: {v}. Maximum allowed: {settings.MAX_SEARCH_RESULTS}"
            )
        return v

    @field_validator("query")
    @classmethod
    def validate_query_length(cls, v: str) -> str:
        """Enforce maximum query length."""
        if len(v) > settings.MAX_QUERY_LENGTH:
            raise ValueError(
                f"Query too long: {len(v)} chars. Maximum allowed: {settings.MAX_QUERY_LENGTH}"
            )
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is machine learning?",
                "k": 5,
                "distance_threshold": 0.5,
            }
        }


class SearchWithEmbeddingRequest(BaseModel):
    """Request model for searching with embedding."""

    embedding: List[float] = Field(..., min_length=1, description="Query embedding")
    k: int = Field(default=10, ge=1, description="Number of results")
    distance_threshold: Optional[float] = Field(
        None,
        ge=0.0,
        le=2.0,
        description="Maximum distance threshold (0-2 for cosine)",
    )

    @field_validator("k")
    @classmethod
    def validate_k(cls, v: int) -> int:
        """Enforce maximum search results limit."""
        if v > settings.MAX_SEARCH_RESULTS:
            raise ValueError(
                f"Too many results requested: {v}. Maximum allowed: {settings.MAX_SEARCH_RESULTS}"
            )
        return v


# Response Models


class ChunkResponse(BaseModel):
    """Response model for a chunk."""

    id: UUID
    text: str
    embedding: List[float]
    metadata: "ChunkMetadataResponse"

    class Config:
        from_attributes = True


class ChunkMetadataResponse(BaseModel):
    """Response model for chunk metadata."""

    created_at: datetime
    page_number: Optional[int]
    chunk_index: int
    source_document_id: UUID

    class Config:
        from_attributes = True


class DocumentMetadataResponse(BaseModel):
    """Response model for document metadata."""

    title: str
    author: Optional[str]
    created_at: datetime
    document_type: str
    source_url: Optional[str]
    tags: List[str]

    class Config:
        from_attributes = True


class DocumentResponse(BaseModel):
    """Response model for a document."""

    id: UUID
    chunks: List[ChunkResponse]
    metadata: DocumentMetadataResponse

    class Config:
        from_attributes = True


class LibraryMetadataResponse(BaseModel):
    """Response model for library metadata."""

    description: Optional[str]
    created_at: datetime
    index_type: str
    embedding_dimension: int
    embedding_model: str

    class Config:
        from_attributes = True


class LibraryResponse(BaseModel):
    """Response model for a library."""

    id: UUID
    name: str
    documents: List[DocumentResponse]
    metadata: LibraryMetadataResponse

    class Config:
        from_attributes = True


class LibrarySummaryResponse(BaseModel):
    """Response model for library summary (without documents)."""

    id: UUID
    name: str
    num_documents: int
    metadata: LibraryMetadataResponse

    class Config:
        from_attributes = True


# Slim Response Models (without embeddings for bandwidth optimization)


class ChunkResponseSlim(BaseModel):
    """Slim response model for a chunk (without embedding)."""

    id: UUID
    text: str
    metadata: "ChunkMetadataResponse"

    class Config:
        from_attributes = True


class DocumentResponseSlim(BaseModel):
    """Slim response model for a document (without embeddings)."""

    id: UUID
    chunks: List[ChunkResponseSlim]
    metadata: DocumentMetadataResponse

    class Config:
        from_attributes = True


class SearchResultResponse(BaseModel):
    """Response model for a search result."""

    chunk: ChunkResponse
    distance: float
    document_id: UUID
    document_title: str


class SearchResultResponseSlim(BaseModel):
    """Slim response model for a search result (without embedding)."""

    chunk: ChunkResponseSlim
    distance: float
    document_id: UUID
    document_title: str


class SearchResponse(BaseModel):
    """Response model for search results."""

    results: List[SearchResultResponse]
    query_time_ms: float
    total_results: int


class SearchResponseSlim(BaseModel):
    """Slim response model for search results (without embeddings)."""

    results: List[SearchResultResponseSlim]
    query_time_ms: float
    total_results: int


class LibraryStatisticsResponse(BaseModel):
    """Response model for library statistics."""

    library_id: str
    library_name: str
    num_documents: int
    num_chunks: int
    embedding_dimension: int
    index_type: str
    vector_store_stats: dict
    index_stats: dict


class ErrorResponse(BaseModel):
    """Response model for errors."""

    error: str
    detail: Optional[str] = None
    error_type: str


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    version: str
    timestamp: datetime


# Metadata Filtering Models


class MetadataFilter(BaseModel):
    """
    A single metadata filter condition.

    Supports filtering on document and chunk metadata fields.
    """

    field: str = Field(..., description="Metadata field to filter on")
    operator: str = Field(
        ...,
        pattern="^(eq|ne|gt|lt|gte|lte|in|contains)$",
        description="Comparison operator: eq, ne, gt, lt, gte, lte, in, contains",
    )
    value: Any = Field(..., description="Value to compare against")

    class Config:
        json_schema_extra = {
            "example": {
                "field": "author",
                "operator": "eq",
                "value": "John Doe",
            }
        }


class SearchWithMetadataRequest(BaseModel):
    """Request model for searching with metadata filters."""

    query: str = Field(..., min_length=1, description="Search query text")
    k: int = Field(default=10, ge=1, description="Number of results")
    metadata_filters: List[MetadataFilter] = Field(
        default_factory=list,
        description="List of metadata filters to apply (AND logic)",
    )
    distance_threshold: Optional[float] = Field(
        None,
        ge=0.0,
        le=2.0,
        description="Maximum distance threshold (0-2 for cosine)",
    )

    @field_validator("k")
    @classmethod
    def validate_k(cls, v: int) -> int:
        """Enforce maximum search results limit."""
        if v > settings.MAX_SEARCH_RESULTS:
            raise ValueError(
                f"Too many results requested: {v}. Maximum allowed: {settings.MAX_SEARCH_RESULTS}"
            )
        return v

    @field_validator("query")
    @classmethod
    def validate_query_length(cls, v: str) -> str:
        """Enforce maximum query length."""
        if len(v) > settings.MAX_QUERY_LENGTH:
            raise ValueError(
                f"Query too long: {len(v)} chars. Maximum allowed: {settings.MAX_QUERY_LENGTH}"
            )
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is machine learning?",
                "k": 5,
                "metadata_filters": [
                    {"field": "author", "operator": "eq", "value": "John Doe"},
                    {"field": "tags", "operator": "contains", "value": "AI"},
                ],
                "distance_threshold": 0.5,
            }
        }


# Update forward references
AddDocumentWithEmbeddingsRequest.model_rebuild()
ChunkResponse.model_rebuild()
ChunkResponseSlim.model_rebuild()
