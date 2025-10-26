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


class QuantizationConfig(BaseModel):
    """Configuration for vector quantization (opt-in memory optimization)."""

    strategy: str = Field(
        default="none",
        pattern="^(none|scalar|hybrid)$",
        description="Quantization strategy: none (float32), scalar (int8), hybrid (best of both)",
    )
    # Scalar quantization params
    bits: int = Field(
        default=8,
        ge=4,
        le=8,
        description="Bits per dimension (4 or 8). Only for scalar/hybrid strategies.",
    )
    # Hybrid quantization params
    rerank_top_k: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Number of results to rerank with float32. Only for hybrid strategy.",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "strategy": "hybrid",
                "bits": 8,
                "rerank_top_k": 100
            }
        }


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
    quantization_config: Optional[QuantizationConfig] = Field(
        None,
        description="Optional quantization configuration for memory optimization (opt-in)",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "name": "Research Papers",
                "description": "Collection of AI research papers",
                "index_type": "hnsw",
                "quantization_config": {
                    "strategy": "hybrid",
                    "bits": 8,
                    "rerank_top_k": 100
                }
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


class QuantizationInfo(BaseModel):
    """Information about quantization settings and impact."""

    strategy: str = Field(..., description="Quantization strategy in use")
    bits: Optional[int] = Field(None, description="Bits per dimension (for scalar/hybrid)")
    memory_savings_percent: float = Field(..., description="Memory savings vs float32 (%)")
    estimated_accuracy_percent: float = Field(..., description="Estimated accuracy retention (%)")

    class Config:
        json_schema_extra = {
            "example": {
                "strategy": "hybrid",
                "bits": 8,
                "memory_savings_percent": 75.0,
                "estimated_accuracy_percent": 99.5
            }
        }


class LibraryMetadataResponse(BaseModel):
    """Response model for library metadata."""

    description: Optional[str]
    created_at: datetime
    index_type: str
    embedding_dimension: int
    embedding_model: str
    quantization: Optional[QuantizationInfo] = Field(None, description="Quantization settings (if enabled)")

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


class DetailedHealthResponse(BaseModel):
    """Response model for detailed health check."""

    status: str
    version: str
    timestamp: datetime
    uptime_seconds: float
    system_info: dict
    service_status: dict

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "timestamp": "2025-10-26T12:34:56Z",
                "uptime_seconds": 3600.5,
                "system_info": {
                    "python_version": "3.9.6",
                    "platform": "darwin",
                    "cpu_count": 8
                },
                "service_status": {
                    "embedding_service": "healthy",
                    "vector_store": "healthy",
                    "persistence": "healthy"
                }
            }
        }


class ReadinessResponse(BaseModel):
    """Response model for readiness probe."""

    ready: bool
    checks: dict
    message: str

    class Config:
        json_schema_extra = {
            "example": {
                "ready": True,
                "checks": {
                    "embedding_service": True,
                    "data_dir_writable": True,
                    "libraries_loaded": True
                },
                "message": "Service is ready to accept traffic"
            }
        }


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


# Batch Operation Models


class BatchAddDocumentsRequest(BaseModel):
    """Request model for batch adding documents with text chunks."""

    documents: List[AddDocumentRequest] = Field(
        ..., min_length=1, max_length=1000, description="List of documents to add (max 1000)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "documents": [
                    {
                        "title": "Document 1",
                        "texts": ["Chunk 1", "Chunk 2"],
                        "tags": ["batch-1"],
                    },
                    {
                        "title": "Document 2",
                        "texts": ["Chunk 1", "Chunk 2", "Chunk 3"],
                        "tags": ["batch-1"],
                    },
                ]
            }
        }


class BatchAddDocumentsWithEmbeddingsRequest(BaseModel):
    """Request model for batch adding documents with pre-computed embeddings."""

    documents: List[AddDocumentWithEmbeddingsRequest] = Field(
        ..., min_length=1, max_length=1000, description="List of documents to add (max 1000)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "documents": [
                    {
                        "title": "Document 1",
                        "chunks": [
                            {"text": "Chunk 1", "embedding": [0.1, 0.2, 0.3]},
                            {"text": "Chunk 2", "embedding": [0.4, 0.5, 0.6]},
                        ],
                    }
                ]
            }
        }


class BatchDeleteDocumentsRequest(BaseModel):
    """Request model for batch deleting documents."""

    document_ids: List[UUID] = Field(
        ..., min_length=1, max_length=1000, description="List of document IDs to delete (max 1000)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "document_ids": [
                    "550e8400-e29b-41d4-a716-446655440000",
                    "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
                ]
            }
        }


class BatchOperationResult(BaseModel):
    """Result for a single operation in a batch."""

    success: bool
    document_id: Optional[UUID] = None
    error: Optional[str] = None


class BatchAddDocumentsResponse(BaseModel):
    """Response model for batch document addition."""

    total_requested: int
    successful: int
    failed: int
    results: List[BatchOperationResult]
    total_chunks_added: int
    processing_time_ms: float

    class Config:
        json_schema_extra = {
            "example": {
                "total_requested": 100,
                "successful": 98,
                "failed": 2,
                "results": [
                    {"success": True, "document_id": "550e8400-e29b-41d4-a716-446655440000"},
                    {"success": False, "error": "Dimension mismatch"},
                ],
                "total_chunks_added": 1247,
                "processing_time_ms": 523.45,
            }
        }


class BatchDeleteDocumentsResponse(BaseModel):
    """Response model for batch document deletion."""

    total_requested: int
    successful: int
    failed: int
    results: List[BatchOperationResult]
    processing_time_ms: float

    class Config:
        json_schema_extra = {
            "example": {
                "total_requested": 50,
                "successful": 48,
                "failed": 2,
                "results": [
                    {"success": True, "document_id": "550e8400-e29b-41d4-a716-446655440000"},
                    {"success": False, "document_id": "6ba7b810-9dad-11d1-80b4-00c04fd430c8", "error": "Document not found"},
                ],
                "processing_time_ms": 145.32,
            }
        }


# Index Management Models


class RebuildIndexRequest(BaseModel):
    """Request model for rebuilding a library's index."""

    index_type: Optional[str] = Field(
        None,
        pattern="^(brute_force|kd_tree|lsh|hnsw)$",
        description="New index type (optional, keeps current if not specified)",
    )
    index_config: Optional[dict] = Field(
        None,
        description="Index-specific configuration parameters",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "index_type": "hnsw",
                "index_config": {
                    "M": 16,
                    "ef_construction": 200,
                    "ef_search": 100
                }
            }
        }


class RebuildIndexResponse(BaseModel):
    """Response model for index rebuild operation."""

    library_id: UUID
    old_index_type: str
    new_index_type: str
    total_vectors_reindexed: int
    rebuild_time_ms: float
    message: str

    class Config:
        json_schema_extra = {
            "example": {
                "library_id": "550e8400-e29b-41d4-a716-446655440000",
                "old_index_type": "brute_force",
                "new_index_type": "hnsw",
                "total_vectors_reindexed": 10000,
                "rebuild_time_ms": 2345.67,
                "message": "Index rebuilt successfully"
            }
        }


class IndexStatisticsResponse(BaseModel):
    """Response model for index statistics."""

    library_id: UUID
    library_name: str
    index_type: str
    total_vectors: int
    index_stats: dict
    vector_store_stats: dict

    class Config:
        json_schema_extra = {
            "example": {
                "library_id": "550e8400-e29b-41d4-a716-446655440000",
                "library_name": "Research Papers",
                "index_type": "hnsw",
                "total_vectors": 10000,
                "index_stats": {
                    "max_level": 4,
                    "entry_point": 42,
                    "avg_connections": 15.3
                },
                "vector_store_stats": {
                    "capacity": 20000,
                    "dimension": 384
                }
            }
        }


class OptimizeIndexResponse(BaseModel):
    """Response model for index optimization."""

    library_id: UUID
    index_type: str
    vectors_compacted: int
    memory_freed_bytes: int
    optimization_time_ms: float
    message: str

    class Config:
        json_schema_extra = {
            "example": {
                "library_id": "550e8400-e29b-41d4-a716-446655440000",
                "index_type": "hnsw",
                "vectors_compacted": 10000,
                "memory_freed_bytes": 1048576,
                "optimization_time_ms": 567.89,
                "message": "Index optimized successfully"
            }
        }


# Multi-Tenancy / Admin Models


class CreateTenantRequest(BaseModel):
    """Request model for creating a tenant."""

    name: str = Field(..., min_length=1, max_length=255, description="Tenant name (e.g., 'Acme Corp')")
    metadata: Optional[dict] = Field(None, description="Optional tenant metadata")
    expires_in_days: Optional[int] = Field(
        None,
        ge=1,
        description="API key expiration in days (None = never expires)",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "name": "Acme Corporation",
                "metadata": {"plan": "enterprise", "contact": "admin@acme.com"},
                "expires_in_days": 365,
            }
        }


class CreateTenantResponse(BaseModel):
    """Response model for tenant creation - includes API key (shown ONCE)."""

    tenant_id: str
    name: str
    api_key: str = Field(..., description="API key - SAVE THIS! It won't be shown again.")
    created_at: datetime
    expires_at: Optional[datetime] = None
    metadata: Optional[dict] = None

    class Config:
        json_schema_extra = {
            "example": {
                "tenant_id": "tenant_a1b2c3d4e5f67890",
                "name": "Acme Corporation",
                "api_key": "arrw_1234567890abcdef1234567890abcdef",
                "created_at": "2025-10-26T12:00:00Z",
                "expires_at": "2026-10-26T12:00:00Z",
                "metadata": {"plan": "enterprise"},
            }
        }


class TenantResponse(BaseModel):
    """Response model for tenant information (without API key)."""

    tenant_id: str
    name: str
    created_at: datetime
    is_active: bool
    expires_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    request_count: int
    metadata: Optional[dict] = None

    class Config:
        json_schema_extra = {
            "example": {
                "tenant_id": "tenant_a1b2c3d4e5f67890",
                "name": "Acme Corporation",
                "created_at": "2025-10-26T12:00:00Z",
                "is_active": True,
                "expires_at": "2026-10-26T12:00:00Z",
                "last_used_at": "2025-10-26T15:30:00Z",
                "request_count": 42567,
                "metadata": {"plan": "enterprise"},
            }
        }


class RotateKeyResponse(BaseModel):
    """Response model for API key rotation."""

    tenant_id: str
    new_api_key: str = Field(..., description="New API key - SAVE THIS! It won't be shown again.")
    rotated_at: datetime

    class Config:
        json_schema_extra = {
            "example": {
                "tenant_id": "tenant_a1b2c3d4e5f67890",
                "new_api_key": "arrw_fedcba0987654321fedcba0987654321",
                "rotated_at": "2025-10-26T16:45:00Z",
            }
        }


# Advanced Query Models (Hybrid Search, Reranking)


class HybridSearchRequest(BaseModel):
    """Request model for hybrid search combining vector + metadata scoring."""

    query: str = Field(..., min_length=1, description="Search query text")
    k: int = Field(default=10, ge=1, description="Number of results")
    vector_weight: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Weight for vector similarity (0-1)",
    )
    metadata_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for metadata boost (0-1, must sum to 1 with vector_weight)",
    )
    recency_boost: bool = Field(
        default=False,
        description="If True, boost recent documents with exponential decay",
    )
    recency_half_life_days: float = Field(
        default=30.0,
        ge=1.0,
        description="Documents lose half their boost after N days",
    )
    distance_threshold: Optional[float] = Field(
        None,
        ge=0.0,
        le=2.0,
        description="Maximum distance threshold (0-2 for cosine)",
    )

    @field_validator("metadata_weight")
    @classmethod
    def validate_weights_sum(cls, v: float, info) -> float:
        """Ensure vector_weight + metadata_weight = 1.0."""
        vector_weight = info.data.get("vector_weight", 0.7)
        if abs((vector_weight + v) - 1.0) > 0.01:
            raise ValueError(
                f"vector_weight + metadata_weight must equal 1.0 "
                f"(got {vector_weight} + {v} = {vector_weight + v})"
            )
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "query": "latest machine learning research",
                "k": 10,
                "vector_weight": 0.7,
                "metadata_weight": 0.3,
                "recency_boost": True,
                "recency_half_life_days": 30.0,
            }
        }


class RerankSearchRequest(BaseModel):
    """Request model for search with post-processing reranking."""

    query: str = Field(..., min_length=1, description="Search query text")
    k: int = Field(default=10, ge=1, description="Number of results")
    rerank_function: str = Field(
        default="recency",
        pattern="^(recency|position|length)$",
        description="Reranking function: recency, position, length",
    )
    rerank_params: Optional[dict] = Field(
        None,
        description="Parameters for reranking function",
    )
    distance_threshold: Optional[float] = Field(
        None,
        ge=0.0,
        le=2.0,
        description="Maximum distance threshold (0-2 for cosine)",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "AI research papers",
                "k": 10,
                "rerank_function": "recency",
                "rerank_params": {"half_life_days": 30.0},
            }
        }


class ScoreBreakdown(BaseModel):
    """Detailed score breakdown for hybrid search results."""

    vector_score: float = Field(..., description="Vector similarity score (0-1)")
    vector_distance: float = Field(..., description="Original cosine distance (0-2)")
    metadata_score: float = Field(..., description="Metadata boost score (0-1)")
    hybrid_score: float = Field(..., description="Final hybrid score (0-1)")
    vector_weight: float = Field(..., description="Weight applied to vector score")
    metadata_weight: float = Field(..., description="Weight applied to metadata score")
    recency_boost: Optional[float] = Field(None, description="Recency boost component")
    field_boost: Optional[dict] = Field(None, description="Field boost details")


class HybridSearchResultResponse(BaseModel):
    """Response model for a single hybrid search result."""

    chunk: ChunkResponseSlim
    score: float = Field(..., description="Hybrid score (0-1, higher is better)")
    score_breakdown: ScoreBreakdown
    document_id: UUID
    document_title: str


class HybridSearchResponse(BaseModel):
    """Response model for hybrid search results."""

    results: List[HybridSearchResultResponse]
    query_time_ms: float
    total_results: int
    scoring_config: dict

    class Config:
        json_schema_extra = {
            "example": {
                "results": [
                    {
                        "chunk": {"id": "...", "text": "...", "metadata": {}},
                        "score": 0.8542,
                        "score_breakdown": {
                            "vector_score": 0.89,
                            "metadata_score": 0.67,
                            "hybrid_score": 0.8542,
                        },
                        "document_id": "550e8400-e29b-41d4-a716-446655440000",
                        "document_title": "ML Research Paper",
                    }
                ],
                "query_time_ms": 45.23,
                "total_results": 10,
                "scoring_config": {"vector_weight": 0.7, "metadata_weight": 0.3},
            }
        }


class RerankSearchResultResponse(BaseModel):
    """Response model for a single reranked search result."""

    chunk: ChunkResponseSlim
    score: float = Field(..., description="Reranked score (0-1, higher is better)")
    document_id: UUID
    document_title: str


class RerankSearchResponse(BaseModel):
    """Response model for reranked search results."""

    results: List[RerankSearchResultResponse]
    query_time_ms: float
    total_results: int
    rerank_function: str
    rerank_params: dict

    class Config:
        json_schema_extra = {
            "example": {
                "results": [
                    {
                        "chunk": {"id": "...", "text": "...", "metadata": {}},
                        "score": 0.9123,
                        "document_id": "550e8400-e29b-41d4-a716-446655440000",
                        "document_title": "Recent Paper",
                    }
                ],
                "query_time_ms": 38.67,
                "total_results": 10,
                "rerank_function": "recency",
                "rerank_params": {"half_life_days": 30.0},
            }
        }


# Update forward references
AddDocumentWithEmbeddingsRequest.model_rebuild()
ChunkResponse.model_rebuild()
ChunkResponseSlim.model_rebuild()
HybridSearchResultResponse.model_rebuild()
RerankSearchResultResponse.model_rebuild()
