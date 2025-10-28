"""
FastAPI application with REST endpoints for the Vector Database.

This module provides the main FastAPI application with all CRUD endpoints
for libraries, documents, and chunks, plus search functionality.
"""

import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.api.dependencies import get_library_service
from app.api import metrics as custom_metrics
from app.api.models import (
    AddDocumentRequest,
    AddDocumentWithEmbeddingsRequest,
    BatchAddDocumentsRequest,
    BatchAddDocumentsResponse,
    BatchAddDocumentsWithEmbeddingsRequest,
    BatchDeleteDocumentsRequest,
    BatchDeleteDocumentsResponse,
    BatchOperationResult,
    ChunkResponseSlim,
    CreateLibraryRequest,
    CreateTenantRequest,
    CreateTenantResponse,
    DetailedHealthResponse,
    DocumentResponse,
    DocumentResponseSlim,
    ErrorResponse,
    HealthResponse,
    HybridSearchRequest,
    HybridSearchResponse,
    HybridSearchResultResponse,
    IndexStatisticsResponse,
    LibraryMetadataResponse,
    ReadinessResponse,
    LibraryResponse,
    LibraryStatisticsResponse,
    LibrarySummaryResponse,
    OptimizeIndexResponse,
    QuantizationInfo,
    RebuildIndexRequest,
    RebuildIndexResponse,
    RerankSearchRequest,
    RerankSearchResponse,
    RerankSearchResultResponse,
    RotateKeyResponse,
    ScoreBreakdown,
    SearchRequest,
    SearchResponse,
    SearchResponseSlim,
    SearchResultResponse,
    SearchResultResponseSlim,
    SearchWithEmbeddingRequest,
    SearchWithMetadataRequest,
    TenantResponse,
)
from app.config import settings
from app.services.embedding_service import EmbeddingServiceError
from app.services.library_service import LibraryService
from app.utils.quantization import calculate_memory_savings, estimate_accuracy
from infrastructure.repositories.library_repository import (
    DimensionMismatchError,
    DocumentNotFoundError,
    LibraryNotFoundError,
)

# Import Temporal client
try:
    from temporal.client import TemporalClient
    TEMPORAL_AVAILABLE = True
except Exception:
    TEMPORAL_AVAILABLE = False

# Configure logging (structured JSON or standard format)
from app.logging_config import configure_structured_logging
configure_structured_logging(
    level=settings.LOG_LEVEL,
    enable_json=settings.LOG_JSON_FORMAT,
)
logger = logging.getLogger(__name__)

# Initialize rate limiter
limiter = Limiter(
    key_func=get_remote_address,
    enabled=settings.RATE_LIMIT_ENABLED,
    storage_uri=settings.RATE_LIMIT_STORAGE_URI,
)

# API Version
API_VERSION = "1.0.0"
API_V1_PREFIX = "/v1"

# Track startup time for uptime monitoring
STARTUP_TIME = time.time()

# Create FastAPI app
app = FastAPI(
    title="Vector Database API",
    description="Production-grade REST API for vector similarity search with multiple indexing algorithms",
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add rate limiter to app state
app.state.limiter = limiter

# Add rate limit exception handler
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ============================================================
# Middleware Configuration
# ============================================================

# Add CORS middleware (if enabled)
if settings.CORS_ENABLED:
    from fastapi.middleware.cors import CORSMiddleware

    # Parse origins from comma-separated string
    origins = [origin.strip() for origin in settings.CORS_ORIGINS.split(",")]

    # Parse methods from comma-separated string
    methods = [method.strip() for method in settings.CORS_ALLOW_METHODS.split(",")]

    # Parse expose headers from comma-separated string
    expose_headers = [header.strip() for header in settings.CORS_EXPOSE_HEADERS.split(",")]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
        allow_methods=methods,
        allow_headers=["*"] if settings.CORS_ALLOW_HEADERS == "*" else settings.CORS_ALLOW_HEADERS.split(","),
        expose_headers=expose_headers,
        max_age=settings.CORS_MAX_AGE,
    )
    logger.info(f"CORS enabled with origins: {origins}")

# Add Security Headers middleware (if enabled)
if settings.SECURITY_HEADERS_ENABLED:
    from app.middleware.security import SecurityHeadersMiddleware

    app.add_middleware(SecurityHeadersMiddleware)
    logger.info("Security headers middleware enabled")

# Add Request Size Limit middleware
from app.middleware.security import RequestSizeLimitMiddleware

app.add_middleware(RequestSizeLimitMiddleware)
logger.info(f"Request size limit: {settings.MAX_REQUEST_SIZE / (1024 * 1024):.0f}MB")

# Add Request ID middleware (for tracing)
from app.middleware.security import RequestIDMiddleware

app.add_middleware(RequestIDMiddleware)
logger.info("Request ID tracking enabled")

# Add Prometheus metrics instrumentation
# This automatically tracks:
# - Request count by endpoint, method, status code
# - Request duration (latency) histograms
# - Requests in progress
instrumentator = Instrumentator(
    should_group_status_codes=False,
    should_ignore_untemplated=True,
    should_respect_env_var=True,
    should_instrument_requests_inprogress=True,
    excluded_handlers=["/metrics"],  # Don't track metrics endpoint itself
    inprogress_name="http_requests_inprogress",
    inprogress_labels=True,
)

# Instrument the app
instrumentator.instrument(app)

# Expose /metrics endpoint
# Prometheus will scrape this endpoint to collect metrics
instrumentator.expose(app, endpoint="/metrics", tags=["Metrics"])

# Create v1 API router
v1_router = APIRouter(prefix=API_V1_PREFIX)


# Exception Handlers


@app.exception_handler(LibraryNotFoundError)
async def library_not_found_handler(request: Request, exc: LibraryNotFoundError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={
            "error": "Library not found",
            "detail": str(exc),
            "error_type": "LibraryNotFoundError",
        },
    )


@app.exception_handler(DocumentNotFoundError)
async def document_not_found_handler(request: Request, exc: DocumentNotFoundError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={
            "error": "Document not found",
            "detail": str(exc),
            "error_type": "DocumentNotFoundError",
        },
    )


@app.exception_handler(DimensionMismatchError)
async def dimension_mismatch_handler(request: Request, exc: DimensionMismatchError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": "Dimension mismatch",
            "detail": str(exc),
            "error_type": "DimensionMismatchError",
        },
    )


@app.exception_handler(EmbeddingServiceError)
async def embedding_service_error_handler(request: Request, exc: EmbeddingServiceError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "error": "Embedding service error",
            "detail": str(exc),
            "error_type": "EmbeddingServiceError",
        },
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": "Invalid request",
            "detail": str(exc),
            "error_type": "ValueError",
        },
    )


# Health Check


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Basic health check",
)
async def health_check() -> HealthResponse:
    """
    Basic health check - returns 200 if API is running.

    This is a lightweight endpoint suitable for load balancer health checks.
    """
    return HealthResponse(
        status="healthy", version=API_VERSION, timestamp=datetime.utcnow()
    )


@app.get(
    "/health/detailed",
    response_model=DetailedHealthResponse,
    tags=["Health"],
    summary="Detailed health check",
)
async def detailed_health_check(
    service: LibraryService = Depends(get_library_service),
) -> DetailedHealthResponse:
    """
    Detailed health check with system information and service status.

    **Returns**:
    - Uptime in seconds
    - System information (Python version, platform, CPU count)
    - Service status (embedding service, vector store, persistence)

    Useful for monitoring dashboards and debugging.
    """
    import platform
    import sys
    from pathlib import Path

    uptime = time.time() - STARTUP_TIME

    # Check system info
    system_info = {
        "python_version": sys.version.split()[0],
        "platform": platform.system().lower(),
        "platform_release": platform.release(),
        "cpu_count": os.cpu_count() or 0,
    }

    # Check service status
    service_status = {}

    # Check embedding service
    try:
        from app.api.dependencies import embedding_service
        if embedding_service:
            service_status["embedding_service"] = "healthy"
        else:
            service_status["embedding_service"] = "unavailable"
    except Exception:
        service_status["embedding_service"] = "error"

    # Check vector store (try to list libraries)
    try:
        libraries = service.list_libraries()
        service_status["vector_store"] = "healthy"
        service_status["libraries_count"] = len(libraries)
    except Exception as e:
        service_status["vector_store"] = f"error: {str(e)}"
        service_status["libraries_count"] = 0

    # Check persistence (data directory writable)
    try:
        data_dir = Path(settings.VECTOR_DB_DATA_DIR)
        data_dir.mkdir(parents=True, exist_ok=True)
        test_file = data_dir / ".health_check"
        test_file.touch()
        test_file.unlink()
        service_status["persistence"] = "healthy"
        service_status["data_dir_writable"] = True
    except Exception:
        service_status["persistence"] = "warning"
        service_status["data_dir_writable"] = False

    return DetailedHealthResponse(
        status="healthy",
        version=API_VERSION,
        timestamp=datetime.utcnow(),
        uptime_seconds=round(uptime, 2),
        system_info=system_info,
        service_status=service_status,
    )


@app.get(
    "/readiness",
    response_model=ReadinessResponse,
    tags=["Health"],
    summary="Readiness probe (Kubernetes)",
)
async def readiness_probe(
    service: LibraryService = Depends(get_library_service),
) -> ReadinessResponse:
    """
    Readiness probe for Kubernetes deployments.

    Returns 200 if service is ready to accept traffic, 503 otherwise.

    **Checks**:
    - Embedding service available
    - Data directory writable
    - Libraries can be listed

    Use this for:
    - Kubernetes readiness probes
    - Load balancer health checks
    - Rolling deployment health checks
    """
    checks = {}
    all_ready = True

    # Check embedding service
    try:
        from app.api.dependencies import embedding_service
        checks["embedding_service"] = embedding_service is not None
        all_ready = all_ready and checks["embedding_service"]
    except Exception:
        checks["embedding_service"] = False
        all_ready = False

    # Check data directory writable
    try:
        from pathlib import Path
        data_dir = Path(settings.VECTOR_DB_DATA_DIR)
        data_dir.mkdir(parents=True, exist_ok=True)
        test_file = data_dir / ".readiness_check"
        test_file.touch()
        test_file.unlink()
        checks["data_dir_writable"] = True
    except Exception:
        checks["data_dir_writable"] = False
        all_ready = False

    # Check libraries can be loaded
    try:
        libraries = service.list_libraries()
        checks["libraries_loaded"] = True
    except Exception:
        checks["libraries_loaded"] = False
        all_ready = False

    message = "Service is ready to accept traffic" if all_ready else "Service is not ready"

    response = ReadinessResponse(ready=all_ready, checks=checks, message=message)

    # Return 503 if not ready
    if not all_ready:
        from fastapi import Response
        return Response(
            content=response.model_dump_json(),
            media_type="application/json",
            status_code=503,
        )

    return response


@app.get(
    "/liveness",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Liveness probe (Kubernetes)",
)
async def liveness_probe() -> HealthResponse:
    """
    Liveness probe for Kubernetes deployments.

    Returns 200 if process is alive (even if not ready).
    Kubernetes will restart the pod if this fails.

    This is simpler than readiness - just checks if the process is responsive.
    """
    return HealthResponse(
        status="alive", version=API_VERSION, timestamp=datetime.utcnow()
    )


# Library Endpoints

# Helper function to convert domain QuantizationMetadata to API QuantizationInfo
def quantization_metadata_to_info(qm, embedding_dimension: int) -> Optional[QuantizationInfo]:
    """Convert QuantizationMetadata to QuantizationInfo with calculated metrics."""
    if qm is None or qm.strategy == "none":
        return None

    memory_savings = calculate_memory_savings(
        strategy=qm.strategy,
        bits=qm.bits or 8,
        dimensions=embedding_dimension
    )

    accuracy = estimate_accuracy(
        strategy=qm.strategy,
        bits=qm.bits or 8
    )

    return QuantizationInfo(
        strategy=qm.strategy,
        bits=qm.bits,
        memory_savings_percent=memory_savings,
        estimated_accuracy_percent=accuracy
    )


@v1_router.post(
    "/libraries",
    response_model=LibraryResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Libraries"],
    summary="Create a new library",
)
def create_library(
    request: CreateLibraryRequest,
    service: LibraryService = Depends(get_library_service),
):
    """
    Create a new library for storing documents and vectors.

    - **name**: Unique name for the library
    - **description**: Optional description
    - **index_type**: Type of index (brute_force, kd_tree, lsh, hnsw, ivf)
    - **embedding_model**: Optional model override
    - **quantization_config**: Optional quantization configuration (opt-in)
    """
    # Convert quantization_config to dict if provided
    quantization_dict = None
    if request.quantization_config:
        quantization_dict = request.quantization_config.model_dump()

    library = service.create_library(
        name=request.name,
        description=request.description,
        index_type=request.index_type,
        embedding_model=request.embedding_model,
        quantization_config=quantization_dict,
    )

    # Convert quantization metadata to info for response
    quantization_info = quantization_metadata_to_info(
        library.metadata.quantization,
        library.metadata.embedding_dimension
    )

    # Create metadata response with quantization info
    metadata_response = LibraryMetadataResponse(
        description=library.metadata.description,
        created_at=library.metadata.created_at,
        index_type=library.metadata.index_type,
        embedding_dimension=library.metadata.embedding_dimension,
        embedding_model=library.metadata.embedding_model,
        quantization=quantization_info
    )

    # Create library response
    return LibraryResponse(
        id=library.id,
        name=library.name,
        documents=[],
        metadata=metadata_response
    )


@v1_router.get(
    "/libraries",
    response_model=List[LibrarySummaryResponse],
    tags=["Libraries"],
    summary="List all libraries",
)
def list_libraries(
    service: LibraryService = Depends(get_library_service),
):
    """
    Get a list of all libraries (without document details).
    """
    libraries = service.list_libraries()

    # Convert to summary responses
    summaries = []
    for lib in libraries:
        summaries.append(
            LibrarySummaryResponse(
                id=lib.id,
                name=lib.name,
                num_documents=len(lib.documents),
                metadata=lib.metadata,
            )
        )

    return summaries


@v1_router.get(
    "/libraries/{library_id}",
    response_model=LibraryResponse,
    tags=["Libraries"],
    summary="Get a library by ID",
)
def get_library(
    library_id: UUID,
    service: LibraryService = Depends(get_library_service),
):
    """
    Get full details of a library including all documents.
    """
    return service.get_library(library_id)


@v1_router.delete(
    "/libraries/{library_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["Libraries"],
    summary="Delete a library",
)
def delete_library(
    library_id: UUID,
    service: LibraryService = Depends(get_library_service),
):
    """
    Delete a library and all its documents and vectors.
    """
    deleted = service.delete_library(library_id)
    if not deleted:
        raise LibraryNotFoundError(f"Library {library_id} not found")


@v1_router.get(
    "/libraries/{library_id}/statistics",
    response_model=LibraryStatisticsResponse,
    tags=["Libraries"],
    summary="Get library statistics",
)
def get_library_statistics(
    library_id: UUID,
    service: LibraryService = Depends(get_library_service),
):
    """
    Get detailed statistics about a library including vector store and index metrics.
    """
    return service.get_library_statistics(library_id)


# Index Management Endpoints


@v1_router.post(
    "/libraries/{library_id}/index/rebuild",
    response_model=RebuildIndexResponse,
    tags=["Libraries", "Index Management"],
    summary="Rebuild library index",
)
def rebuild_library_index(
    library_id: UUID,
    request: RebuildIndexRequest,
    service: LibraryService = Depends(get_library_service),
):
    """
    Rebuild a library's index, optionally switching to a new index type.

    **Use Cases**:
    - Switch index types (e.g., brute_force → hnsw for better performance)
    - Optimize degraded index after many deletions
    - Tune index parameters for your dataset

    **Index Types**:
    - `brute_force`: Exact search, no build time, O(n) query
    - `kd_tree`: Fast for low dimensions (<20), O(log n) query
    - `lsh`: Approximate, fast build, good for high dimensions
    - `hnsw`: Best overall, fast queries, longer build time

    **Parameters by Index Type**:
    - `hnsw`: M (connections per node), ef_construction, ef_search
    - `kd_tree`: rebuild_threshold
    - `lsh`: num_tables, hash_size

    **Performance**:
    - Rebuilding acquires write lock (blocks other operations)
    - Time: ~1-5 seconds per 10,000 vectors
    - All data is preserved

    Returns the old/new index types and number of vectors reindexed.
    """
    start_time = time.time()

    old_type, new_type, vectors = service.rebuild_index(
        library_id,
        request.index_type,
        request.index_config,
    )

    rebuild_time_ms = (time.time() - start_time) * 1000

    return RebuildIndexResponse(
        library_id=library_id,
        old_index_type=old_type,
        new_index_type=new_type,
        total_vectors_reindexed=vectors,
        rebuild_time_ms=round(rebuild_time_ms, 2),
        message=f"Index rebuilt successfully ({old_type} → {new_type})",
    )


@v1_router.post(
    "/libraries/{library_id}/index/optimize",
    response_model=OptimizeIndexResponse,
    tags=["Libraries", "Index Management"],
    summary="Optimize library index",
)
def optimize_library_index(
    library_id: UUID,
    service: LibraryService = Depends(get_library_service),
):
    """
    Optimize a library's index by compacting and removing deleted entries.

    **Benefits**:
    - Improves search performance
    - Reduces memory usage
    - Removes fragmentation from deletions

    **When to Optimize**:
    - After bulk deletions
    - When search performance degrades
    - Periodically for maintenance (e.g., weekly)

    **Performance**:
    - Acquires write lock (blocks other operations)
    - Time: ~1-3 seconds per 10,000 vectors
    - All data is preserved

    Returns number of vectors compacted and memory freed.
    """
    start_time = time.time()

    vectors, memory_freed = service.optimize_index(library_id)

    optimization_time_ms = (time.time() - start_time) * 1000

    library = service.get_library(library_id)

    return OptimizeIndexResponse(
        library_id=library_id,
        index_type=library.metadata.index_type,
        vectors_compacted=vectors,
        memory_freed_bytes=memory_freed,
        optimization_time_ms=round(optimization_time_ms, 2),
        message="Index optimized successfully",
    )


@v1_router.get(
    "/libraries/{library_id}/index/statistics",
    response_model=IndexStatisticsResponse,
    tags=["Libraries", "Index Management"],
    summary="Get detailed index statistics",
)
def get_index_statistics(
    library_id: UUID,
    service: LibraryService = Depends(get_library_service),
):
    """
    Get detailed statistics about a library's index structure.

    **Returns**:
    - Index type and configuration
    - Number of vectors indexed
    - Index-specific metrics (e.g., HNSW graph depth, connections)
    - Vector store statistics (capacity, dimension)

    **Index-Specific Stats**:
    - `hnsw`: max_level, entry_point, avg_connections
    - `kd_tree`: tree_depth, leaf_size
    - `lsh`: num_tables, hash_size, bucket_stats
    - `brute_force`: num_vectors

    Useful for monitoring and tuning index performance.
    """
    stats = service.get_index_statistics(library_id)

    return IndexStatisticsResponse(
        library_id=UUID(stats["library_id"]),
        library_name=stats["library_name"],
        index_type=stats["index_type"],
        total_vectors=stats["total_vectors"],
        index_stats=stats["index_stats"],
        vector_store_stats=stats["vector_store_stats"],
    )


# Document Endpoints


@v1_router.post(
    "/libraries/{library_id}/documents",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Documents"],
    summary="Add a document with text chunks",
)
@limiter.limit(settings.RATE_LIMIT_DOCUMENT_ADD)
def add_document(
    req: Request,
    library_id: UUID,
    request: AddDocumentRequest,
    service: LibraryService = Depends(get_library_service),
):
    """
    Add a document to a library by providing text chunks.

    Embeddings will be generated automatically for each text chunk.

    - **title**: Document title
    - **texts**: List of text chunks
    - **author**: Optional author name
    - **document_type**: Type of document (default: "text")
    - **source_url**: Optional source URL
    - **tags**: Optional list of tags
    """
    document = service.add_document_with_text(
        library_id=library_id,
        title=request.title,
        texts=request.texts,
        author=request.author,
        document_type=request.document_type,
        source_url=request.source_url,
        tags=request.tags,
    )
    return document


@v1_router.post(
    "/libraries/{library_id}/documents/with-embeddings",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Documents"],
    summary="Add a document with pre-computed embeddings",
)
@limiter.limit(settings.RATE_LIMIT_DOCUMENT_ADD)
def add_document_with_embeddings(
    req: Request,
    library_id: UUID,
    request: AddDocumentWithEmbeddingsRequest,
    service: LibraryService = Depends(get_library_service),
):
    """
    Add a document to a library with pre-computed embeddings.

    Use this endpoint when you've already generated embeddings elsewhere.

    - **title**: Document title
    - **chunks**: List of {text, embedding} pairs
    - **author**: Optional author name
    - **document_type**: Type of document (default: "text")
    - **source_url**: Optional source URL
    - **tags**: Optional list of tags
    """
    # Convert chunks to tuples
    text_embedding_pairs = [
        (chunk.text, chunk.embedding) for chunk in request.chunks
    ]

    document = service.add_document_with_embeddings(
        library_id=library_id,
        title=request.title,
        text_embedding_pairs=text_embedding_pairs,
        author=request.author,
        document_type=request.document_type,
        source_url=request.source_url,
        tags=request.tags,
    )
    return document


@v1_router.get(
    "/documents/{document_id}",
    response_model=DocumentResponse,
    tags=["Documents"],
    summary="Get a document by ID",
)
def get_document(
    document_id: UUID,
    service: LibraryService = Depends(get_library_service),
):
    """
    Get a document by its ID.
    """
    return service.get_document(document_id)


@v1_router.delete(
    "/documents/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["Documents"],
    summary="Delete a document",
)
def delete_document(
    document_id: UUID,
    service: LibraryService = Depends(get_library_service),
):
    """
    Delete a document and all its chunks.
    """
    deleted = service.delete_document(document_id)
    if not deleted:
        raise DocumentNotFoundError(f"Document {document_id} not found")


# Batch Document Endpoints


@v1_router.post(
    "/libraries/{library_id}/documents/batch",
    response_model=BatchAddDocumentsResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Documents", "Batch Operations"],
    summary="Batch add documents with text chunks",
)
@limiter.limit("10/minute")
def batch_add_documents(
    req: Request,
    library_id: UUID,
    request: BatchAddDocumentsRequest,
    service: LibraryService = Depends(get_library_service),
):
    """
    Add multiple documents to a library in a single batch operation.

    This endpoint is 100-1000x faster than adding documents one at a time:
    - Batched embedding generation (all texts embedded at once)
    - Single write lock acquisition
    - Reduced API overhead

    **Performance**:
    - 1,000 documents with 10 chunks each = ~10 seconds (vs 10+ minutes individually)
    - Max 1,000 documents per batch

    **Example Use Cases**:
    - Initial data load
    - Bulk document import
    - ETL pipelines

    Returns success/failure for each document.
    """
    start_time = time.time()

    # Convert request models to dicts for service layer
    documents_data = [doc.model_dump() for doc in request.documents]

    try:
        successful, failed, total_chunks = service.add_documents_batch(
            library_id, documents_data
        )

        processing_time_ms = (time.time() - start_time) * 1000

        # Build results
        results = []
        for doc in successful:
            results.append(BatchOperationResult(success=True, document_id=doc.id))

        for idx, error in failed:
            results.append(BatchOperationResult(success=False, error=error))

        return BatchAddDocumentsResponse(
            total_requested=len(request.documents),
            successful=len(successful),
            failed=len(failed),
            results=results,
            total_chunks_added=total_chunks,
            processing_time_ms=round(processing_time_ms, 2),
        )

    except Exception as e:
        logger.error(f"Batch add documents failed: {e}")
        raise


@v1_router.post(
    "/libraries/{library_id}/documents/batch-with-embeddings",
    response_model=BatchAddDocumentsResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Documents", "Batch Operations"],
    summary="Batch add documents with pre-computed embeddings",
)
@limiter.limit("10/minute")
def batch_add_documents_with_embeddings(
    req: Request,
    library_id: UUID,
    request: BatchAddDocumentsWithEmbeddingsRequest,
    service: LibraryService = Depends(get_library_service),
):
    """
    Add multiple documents with pre-computed embeddings in a single batch operation.

    Use this when you've already generated embeddings elsewhere or want to use
    custom embedding models.

    **Performance**:
    - Even faster than regular batch add (no embedding generation)
    - Max 1,000 documents per batch

    Returns success/failure for each document.
    """
    start_time = time.time()

    # Convert request models to dicts for service layer
    documents_data = [doc.model_dump() for doc in request.documents]

    try:
        successful, failed, total_chunks = service.add_documents_batch_with_embeddings(
            library_id, documents_data
        )

        processing_time_ms = (time.time() - start_time) * 1000

        # Build results
        results = []
        for doc in successful:
            results.append(BatchOperationResult(success=True, document_id=doc.id))

        for idx, error in failed:
            results.append(BatchOperationResult(success=False, error=error))

        return BatchAddDocumentsResponse(
            total_requested=len(request.documents),
            successful=len(successful),
            failed=len(failed),
            results=results,
            total_chunks_added=total_chunks,
            processing_time_ms=round(processing_time_ms, 2),
        )

    except Exception as e:
        logger.error(f"Batch add documents with embeddings failed: {e}")
        raise


@v1_router.delete(
    "/documents/batch",
    response_model=BatchDeleteDocumentsResponse,
    status_code=status.HTTP_200_OK,
    tags=["Documents", "Batch Operations"],
    summary="Batch delete documents",
)
@limiter.limit("10/minute")
def batch_delete_documents(
    req: Request,
    request: BatchDeleteDocumentsRequest,
    service: LibraryService = Depends(get_library_service),
):
    """
    Delete multiple documents in a single batch operation.

    **Performance**:
    - Single write lock acquisition
    - Batched index/vector store operations
    - Max 1,000 documents per batch

    **Example Use Cases**:
    - Bulk cleanup
    - Data retention policies
    - Document expiration

    Returns success/failure for each document ID.
    """
    start_time = time.time()

    try:
        successful, failed = service.delete_documents_batch(request.document_ids)

        processing_time_ms = (time.time() - start_time) * 1000

        # Build results
        results = []
        for doc_id in successful:
            results.append(BatchOperationResult(success=True, document_id=doc_id))

        for doc_id, error in failed:
            results.append(
                BatchOperationResult(success=False, document_id=doc_id, error=error)
            )

        return BatchDeleteDocumentsResponse(
            total_requested=len(request.document_ids),
            successful=len(successful),
            failed=len(failed),
            results=results,
            processing_time_ms=round(processing_time_ms, 2),
        )

    except Exception as e:
        logger.error(f"Batch delete documents failed: {e}")
        raise


# Search Endpoints


@v1_router.post(
    "/libraries/{library_id}/search",
    tags=["Search"],
    summary="Search with text query",
)
@limiter.limit(settings.RATE_LIMIT_SEARCH)
def search(
    req: Request,
    library_id: UUID,
    request: SearchRequest,
    include_embeddings: bool = Query(default=False, description="Include embeddings in response"),
    service: LibraryService = Depends(get_library_service),
) -> Union[SearchResponse, SearchResponseSlim]:
    """
    Search a library using a natural language query.

    The query text will be embedded automatically and compared against
    all chunks in the library.

    - **query**: Natural language search query
    - **k**: Number of results to return (1-100)
    - **distance_threshold**: Optional maximum distance (0-2 for cosine)
    - **include_embeddings**: Include embeddings in response (default: false for reduced bandwidth)

    Returns the k most similar chunks ranked by cosine similarity.
    """
    start_time = time.time()

    # Get library to track index type
    library = service.get_library(library_id)
    index_type = library.metadata.index_type

    results = service.search_with_text(
        library_id=library_id,
        query_text=request.query,
        k=request.k,
        distance_threshold=request.distance_threshold,
    )

    query_time_ms = (time.time() - start_time) * 1000

    # Track metrics
    custom_metrics.vector_searches_total.labels(
        library_id=str(library_id),
        index_type=index_type,
    ).inc()

    custom_metrics.search_duration_seconds.labels(
        library_id=str(library_id),
        index_type=index_type,
    ).observe(query_time_ms / 1000)

    custom_metrics.search_results_count.observe(len(results))

    # Build response based on include_embeddings flag
    search_results = []
    for chunk, distance in results:
        # Get document info
        doc = service.get_document(chunk.metadata.source_document_id)

        if include_embeddings:
            search_results.append(
                SearchResultResponse(
                    chunk=chunk,
                    distance=distance,
                    document_id=doc.id,
                    document_title=doc.metadata.title,
                )
            )
        else:
            # Create slim chunk without embedding
            slim_chunk = ChunkResponseSlim(
                id=chunk.id,
                text=chunk.text,
                metadata=chunk.metadata,
            )
            search_results.append(
                SearchResultResponseSlim(
                    chunk=slim_chunk,
                    distance=distance,
                    document_id=doc.id,
                    document_title=doc.metadata.title,
                )
            )

    if include_embeddings:
        return SearchResponse(
            results=search_results,
            query_time_ms=round(query_time_ms, 2),
            total_results=len(search_results),
        )
    else:
        return SearchResponseSlim(
            results=search_results,
            query_time_ms=round(query_time_ms, 2),
            total_results=len(search_results),
        )


@v1_router.post(
    "/libraries/{library_id}/search/embedding",
    tags=["Search"],
    summary="Search with embedding",
)
@limiter.limit(settings.RATE_LIMIT_SEARCH)
def search_with_embedding(
    req: Request,
    library_id: UUID,
    request: SearchWithEmbeddingRequest,
    include_embeddings: bool = Query(default=False, description="Include embeddings in response"),
    service: LibraryService = Depends(get_library_service),
) -> Union[SearchResponse, SearchResponseSlim]:
    """
    Search a library using a pre-computed embedding.

    Use this when you've already generated a query embedding.

    - **embedding**: Query vector (must match library's dimension)
    - **k**: Number of results to return (1-100)
    - **distance_threshold**: Optional maximum distance (0-2 for cosine)
    - **include_embeddings**: Include embeddings in response (default: false for reduced bandwidth)

    Returns the k most similar chunks ranked by cosine similarity.
    """
    start_time = time.time()

    results = service.search_with_embedding(
        library_id=library_id,
        query_embedding=request.embedding,
        k=request.k,
        distance_threshold=request.distance_threshold,
    )

    query_time_ms = (time.time() - start_time) * 1000

    # Build response based on include_embeddings flag
    search_results = []
    for chunk, distance in results:
        # Get document info
        doc = service.get_document(chunk.metadata.source_document_id)

        if include_embeddings:
            search_results.append(
                SearchResultResponse(
                    chunk=chunk,
                    distance=distance,
                    document_id=doc.id,
                    document_title=doc.metadata.title,
                )
            )
        else:
            # Create slim chunk without embedding
            slim_chunk = ChunkResponseSlim(
                id=chunk.id,
                text=chunk.text,
                metadata=chunk.metadata,
            )
            search_results.append(
                SearchResultResponseSlim(
                    chunk=slim_chunk,
                    distance=distance,
                    document_id=doc.id,
                    document_title=doc.metadata.title,
                )
            )

    if include_embeddings:
        return SearchResponse(
            results=search_results,
            query_time_ms=round(query_time_ms, 2),
            total_results=len(search_results),
        )
    else:
        return SearchResponseSlim(
            results=search_results,
            query_time_ms=round(query_time_ms, 2),
            total_results=len(search_results),
        )


@v1_router.post(
    "/libraries/{library_id}/search/filtered",
    tags=["Search"],
    summary="Search with metadata filters",
)
@limiter.limit(settings.RATE_LIMIT_SEARCH)
def search_with_metadata(
    req: Request,
    library_id: UUID,
    request: SearchWithMetadataRequest,
    include_embeddings: bool = Query(default=False, description="Include embeddings in response"),
    service: LibraryService = Depends(get_library_service),
) -> Union[SearchResponse, SearchResponseSlim]:
    """
    Search a library with text query and apply metadata filters.

    This endpoint performs vector search first, then filters results
    based on chunk metadata. All filters use AND logic.

    **Available Metadata Fields:**
    - `created_at`: Chunk creation timestamp (datetime)
    - `page_number`: Page number (int, optional)
    - `chunk_index`: Position in document (int)
    - `source_document_id`: Parent document UUID

    **Supported Operators:**
    - `eq`, `ne`: Equality/inequality
    - `gt`, `lt`, `gte`, `lte`: Numeric comparisons
    - `in`: Check if value is in list
    - `contains`: String contains substring, or list contains element

    **Example Filters:**
    ```json
    {
      "query": "machine learning",
      "k": 10,
      "metadata_filters": [
        {"field": "page_number", "operator": "gte", "value": 5},
        {"field": "chunk_index", "operator": "lt", "value": 10}
      ]
    }
    ```

    Returns chunks that match the query AND all metadata filters.
    """
    start_time = time.time()

    # Convert MetadataFilter objects to dicts for service layer
    filters_dict = [
        {
            "field": f.field,
            "operator": f.operator,
            "value": f.value,
        }
        for f in request.metadata_filters
    ]

    results = service.search_with_metadata_filters(
        library_id=library_id,
        query_text=request.query,
        metadata_filters=filters_dict,
        k=request.k,
        distance_threshold=request.distance_threshold,
    )

    query_time_ms = (time.time() - start_time) * 1000

    # Build response (same format as regular search)
    search_results = []
    for chunk, distance in results:
        doc = service.get_document(chunk.metadata.source_document_id)

        if include_embeddings:
            search_results.append(
                SearchResultResponse(
                    chunk=chunk,
                    distance=distance,
                    document_id=doc.id,
                    document_title=doc.metadata.title,
                )
            )
        else:
            slim_chunk = ChunkResponseSlim(
                id=chunk.id,
                text=chunk.text,
                metadata=chunk.metadata,
            )
            search_results.append(
                SearchResultResponseSlim(
                    chunk=slim_chunk,
                    distance=distance,
                    document_id=doc.id,
                    document_title=doc.metadata.title,
                )
            )

    if include_embeddings:
        return SearchResponse(
            results=search_results,
            query_time_ms=round(query_time_ms, 2),
            total_results=len(search_results),
        )
    else:
        return SearchResponseSlim(
            results=search_results,
            query_time_ms=round(query_time_ms, 2),
            total_results=len(search_results),
        )


# Advanced Query Endpoints (Hybrid Search, Reranking)


@v1_router.post(
    "/libraries/{library_id}/search/hybrid",
    response_model=HybridSearchResponse,
    tags=["Search", "Advanced Queries"],
    summary="Hybrid search (vector + metadata)",
)
@limiter.limit(settings.RATE_LIMIT_SEARCH)
def hybrid_search(
    req: Request,
    library_id: UUID,
    request: HybridSearchRequest,
    service: LibraryService = Depends(get_library_service),
) -> HybridSearchResponse:
    """
    Perform hybrid search combining vector similarity with metadata-based scoring.

    **What is Hybrid Search?**

    Hybrid search combines two ranking signals:
    1. **Vector Similarity**: Semantic similarity from embeddings (70% by default)
    2. **Metadata Boost**: Signals from document metadata (30% by default)

    This provides production-grade ranking similar to:
    - Elasticsearch's "function_score" queries
    - Pinecone's metadata-boosted search
    - Weaviate's hybrid search mode

    **When to Use**:
    - When recency matters (e.g., news, research papers, blog posts)
    - When you want to boost specific metadata fields
    - When pure vector search isn't enough

    **Scoring Formula**:
    ```
    hybrid_score = (vector_weight * vector_similarity) + (metadata_weight * metadata_boost)
    ```

    **Parameters**:
    - `query`: Natural language search query
    - `k`: Number of results (1-100)
    - `vector_weight`: Weight for vector similarity (0-1, default 0.7)
    - `metadata_weight`: Weight for metadata boost (0-1, default 0.3)
    - `recency_boost`: Enable exponential decay based on document age
    - `recency_half_life_days`: Documents lose half their boost after N days

    **Returns**:
    - Ranked results with detailed score breakdown
    - Shows vector_score, metadata_score, hybrid_score for each result

    **Example**:
    ```bash
    curl -X POST http://localhost:8000/v1/libraries/{id}/search/hybrid \\
      -H "Content-Type: application/json" \\
      -d '{
        "query": "latest machine learning research",
        "k": 10,
        "vector_weight": 0.7,
        "metadata_weight": 0.3,
        "recency_boost": true,
        "recency_half_life_days": 30.0
      }'
    ```

    **Use Cases**:
    - News/blog search (boost recent articles)
    - Research papers (boost recent publications)
    - Documentation (boost official/verified sources)
    - E-commerce (boost popular/highly-rated products)
    """
    start_time = time.time()

    results = service.hybrid_search(
        library_id=library_id,
        query_text=request.query,
        k=request.k,
        vector_weight=request.vector_weight,
        metadata_weight=request.metadata_weight,
        recency_boost=request.recency_boost,
        recency_half_life_days=request.recency_half_life_days,
        distance_threshold=request.distance_threshold,
    )

    query_time_ms = (time.time() - start_time) * 1000

    # Build response
    search_results = []
    for chunk, hybrid_score, breakdown in results:
        # Get document info
        doc = service.get_document(chunk.metadata.source_document_id)

        # Create slim chunk
        slim_chunk = ChunkResponseSlim(
            id=chunk.id,
            text=chunk.text,
            metadata=chunk.metadata,
        )

        # Create score breakdown
        score_breakdown = ScoreBreakdown(
            vector_score=breakdown["vector_score"],
            vector_distance=breakdown["vector_distance"],
            metadata_score=breakdown["metadata_score"],
            hybrid_score=breakdown["hybrid_score"],
            vector_weight=breakdown["vector_weight"],
            metadata_weight=breakdown["metadata_weight"],
            recency_boost=breakdown.get("recency_boost"),
            field_boost=breakdown.get("field_boost"),
        )

        search_results.append(
            HybridSearchResultResponse(
                chunk=slim_chunk,
                score=hybrid_score,
                score_breakdown=score_breakdown,
                document_id=doc.id,
                document_title=doc.metadata.title,
            )
        )

    return HybridSearchResponse(
        results=search_results,
        query_time_ms=round(query_time_ms, 2),
        total_results=len(search_results),
        scoring_config={
            "vector_weight": request.vector_weight,
            "metadata_weight": request.metadata_weight,
            "recency_boost": request.recency_boost,
            "recency_half_life_days": request.recency_half_life_days,
        },
    )


@v1_router.post(
    "/libraries/{library_id}/search/rerank",
    response_model=RerankSearchResponse,
    tags=["Search", "Advanced Queries"],
    summary="Search with reranking",
)
@limiter.limit(settings.RATE_LIMIT_SEARCH)
def search_with_reranking(
    req: Request,
    library_id: UUID,
    request: RerankSearchRequest,
    service: LibraryService = Depends(get_library_service),
) -> RerankSearchResponse:
    """
    Search with post-processing reranking using predefined functions.

    **What is Reranking?**

    Reranking applies custom scoring logic AFTER initial vector search.
    This is similar to Elasticsearch's "rescore" feature.

    **Reranking Functions**:

    1. **recency**: Boost recent documents with exponential decay
       - Parameters: `half_life_days` (default: 30)
       - Use case: News, blog posts, research papers

    2. **position**: Boost chunks by position in document
       - Parameters: `prefer_early` (default: true)
       - Use case: Prefer intro/summary vs appendix

    3. **length**: Boost chunks by text length
       - Parameters: `prefer_longer` (default: true)
       - Use case: Prefer detailed vs short snippets

    **When to Use**:
    - When you want simple post-processing without full hybrid search
    - When you know the reranking signal you need
    - For quick experimentation with ranking strategies

    **Example - Recency Boost**:
    ```bash
    curl -X POST http://localhost:8000/v1/libraries/{id}/search/rerank \\
      -H "Content-Type: application/json" \\
      -d '{
        "query": "AI research",
        "k": 10,
        "rerank_function": "recency",
        "rerank_params": {"half_life_days": 30.0}
      }'
    ```

    **Example - Position Boost**:
    ```bash
    curl -X POST http://localhost:8000/v1/libraries/{id}/search/rerank \\
      -H "Content-Type: application/json" \\
      -d '{
        "query": "introduction to ML",
        "k": 10,
        "rerank_function": "position",
        "rerank_params": {"prefer_early": true}
      }'
    ```

    **Comparison to Hybrid Search**:
    - Reranking: Simple, fast, predefined functions
    - Hybrid Search: More control, configurable weights, score breakdowns
    """
    start_time = time.time()

    results = service.search_with_reranking(
        library_id=library_id,
        query_text=request.query,
        k=request.k,
        rerank_function=request.rerank_function,
        rerank_params=request.rerank_params or {},
        distance_threshold=request.distance_threshold,
    )

    query_time_ms = (time.time() - start_time) * 1000

    # Build response
    search_results = []
    for chunk, score in results:
        # Get document info
        doc = service.get_document(chunk.metadata.source_document_id)

        # Create slim chunk
        slim_chunk = ChunkResponseSlim(
            id=chunk.id,
            text=chunk.text,
            metadata=chunk.metadata,
        )

        search_results.append(
            RerankSearchResultResponse(
                chunk=slim_chunk,
                score=score,
                document_id=doc.id,
                document_title=doc.metadata.title,
            )
        )

    return RerankSearchResponse(
        results=search_results,
        query_time_ms=round(query_time_ms, 2),
        total_results=len(search_results),
        rerank_function=request.rerank_function,
        rerank_params=request.rerank_params or {},
    )


# Admin / Multi-Tenancy Endpoints


@v1_router.post(
    "/admin/tenants",
    response_model=CreateTenantResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Admin", "Multi-Tenancy"],
    summary="Create a new tenant",
)
def create_tenant(
    request: CreateTenantRequest,
):
    """
    Create a new tenant with API key authentication.

    **IMPORTANT**: The API key is only returned ONCE. Save it securely!

    **Use Cases**:
    - Onboard new customers/organizations
    - Create isolated namespaces for different users
    - Set up API access with expiration

    **Parameters**:
    - `name`: Tenant name (e.g., "Acme Corp", "Project Alpha")
    - `metadata`: Optional metadata (plan type, contact info, etc.)
    - `expires_in_days`: Optional expiration (None = never expires)

    **Security Features**:
    - Cryptographically secure key generation (128-bit entropy)
    - SHA-256 hashing for storage (keys never stored in plaintext)
    - Optional expiration for temporary access
    - Usage tracking (last_used_at, request_count)

    **Returns**:
    - `tenant_id`: Unique tenant identifier
    - `api_key`: API key (format: `arrw_<32 hex chars>`)
    - `created_at`: Timestamp
    - `expires_at`: Expiration timestamp (if set)

    **Example**:
    ```bash
    curl -X POST http://localhost:8000/v1/admin/tenants \\
      -H "Content-Type: application/json" \\
      -d '{
        "name": "Acme Corp",
        "metadata": {"plan": "enterprise"},
        "expires_in_days": 365
      }'
    ```

    Then use the API key:
    ```bash
    curl -H "X-API-Key: arrw_..." http://localhost:8000/v1/libraries
    # or
    curl -H "Authorization: Bearer arrw_..." http://localhost:8000/v1/libraries
    ```
    """
    from app.auth import get_api_key_manager

    manager = get_api_key_manager()
    tenant_id, api_key = manager.create_tenant(
        name=request.name,
        metadata=request.metadata,
        expires_in_days=request.expires_in_days,
    )

    tenant = manager.get_tenant(tenant_id)

    return CreateTenantResponse(
        tenant_id=tenant.tenant_id,
        name=tenant.name,
        api_key=api_key,  # ONLY TIME THIS IS SHOWN
        created_at=tenant.created_at,
        expires_at=tenant.key_expires_at,
        metadata=tenant.metadata,
    )


@v1_router.get(
    "/admin/tenants",
    response_model=List[TenantResponse],
    tags=["Admin", "Multi-Tenancy"],
    summary="List all tenants",
)
def list_tenants():
    """
    List all tenants (without API keys).

    **Returns**:
    - List of tenants with metadata
    - Usage statistics (request_count, last_used_at)
    - Status (active/inactive, expired/valid)

    **Use Cases**:
    - Monitor tenant usage
    - Audit active accounts
    - Identify inactive tenants

    **Example**:
    ```bash
    curl http://localhost:8000/v1/admin/tenants
    ```
    """
    from app.auth import get_api_key_manager

    manager = get_api_key_manager()
    tenants = manager.list_tenants()

    return [
        TenantResponse(
            tenant_id=t.tenant_id,
            name=t.name,
            created_at=t.created_at,
            is_active=t.is_active,
            expires_at=t.key_expires_at,
            last_used_at=t.last_used_at,
            request_count=t.request_count,
            metadata=t.metadata,
        )
        for t in tenants
    ]


@v1_router.get(
    "/admin/tenants/{tenant_id}",
    response_model=TenantResponse,
    tags=["Admin", "Multi-Tenancy"],
    summary="Get tenant details",
)
def get_tenant(tenant_id: str):
    """
    Get details for a specific tenant.

    **Returns**:
    - Tenant information (without API key)
    - Usage statistics
    - Status

    **Example**:
    ```bash
    curl http://localhost:8000/v1/admin/tenants/tenant_a1b2c3d4e5f67890
    ```
    """
    from app.auth import get_api_key_manager

    manager = get_api_key_manager()
    tenant = manager.get_tenant(tenant_id)

    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tenant {tenant_id} not found",
        )

    return TenantResponse(
        tenant_id=tenant.tenant_id,
        name=tenant.name,
        created_at=tenant.created_at,
        is_active=tenant.is_active,
        expires_at=tenant.key_expires_at,
        last_used_at=tenant.last_used_at,
        request_count=tenant.request_count,
        metadata=tenant.metadata,
    )


@v1_router.delete(
    "/admin/tenants/{tenant_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["Admin", "Multi-Tenancy"],
    summary="Deactivate a tenant",
)
def deactivate_tenant(tenant_id: str):
    """
    Deactivate a tenant (disable their API key).

    This doesn't delete the tenant record, just prevents API access.
    Usage history is preserved for audit purposes.

    **Use Cases**:
    - Suspend access for non-payment
    - Temporarily disable accounts
    - Revoke compromised keys

    **Example**:
    ```bash
    curl -X DELETE http://localhost:8000/v1/admin/tenants/tenant_a1b2c3d4e5f67890
    ```

    After deactivation, API requests with that key will return 401 Unauthorized.
    """
    from app.auth import get_api_key_manager

    manager = get_api_key_manager()
    success = manager.deactivate_tenant(tenant_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tenant {tenant_id} not found",
        )


@v1_router.post(
    "/admin/tenants/{tenant_id}/rotate",
    response_model=RotateKeyResponse,
    tags=["Admin", "Multi-Tenancy"],
    summary="Rotate tenant API key",
)
def rotate_tenant_key(tenant_id: str):
    """
    Rotate a tenant's API key (generate new, invalidate old).

    **IMPORTANT**: The new API key is only returned ONCE. Save it securely!

    **Use Cases**:
    - Regular key rotation (security best practice)
    - Key compromise response
    - Planned maintenance rotations

    **Behavior**:
    - Old key is immediately invalidated
    - New key is generated with same expiration settings
    - Tenant record and usage history preserved

    **Example**:
    ```bash
    curl -X POST http://localhost:8000/v1/admin/tenants/tenant_a1b2c3d4e5f67890/rotate
    ```

    Returns new API key - update your application configuration immediately!
    """
    from app.auth import get_api_key_manager

    manager = get_api_key_manager()
    new_api_key = manager.rotate_api_key(tenant_id)

    if not new_api_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tenant {tenant_id} not found",
        )

    return RotateKeyResponse(
        tenant_id=tenant_id,
        new_api_key=new_api_key,  # ONLY TIME THIS IS SHOWN
        rotated_at=datetime.utcnow(),
    )


# Startup event


@app.on_event("startup")
async def startup_event() -> None:
    """Log configuration summary and restore state on startup."""
    logger.info("=" * 60)
    logger.info("Vector Database API Starting")
    logger.info("=" * 60)
    logger.info(f"Server: {settings.HOST}:{settings.PORT}")
    logger.info(f"Workers: {settings.workers}")
    logger.info(f"Rate Limiting: {'Enabled' if settings.RATE_LIMIT_ENABLED else 'Disabled'}")
    if settings.RATE_LIMIT_ENABLED:
        logger.info(f"  - Search: {settings.RATE_LIMIT_SEARCH}")
        logger.info(f"  - Document Add: {settings.RATE_LIMIT_DOCUMENT_ADD}")
    logger.info(f"Embedding Dimension: {settings.EMBEDDING_DIMENSION}")
    logger.info(f"Max Chunks/Document: {settings.MAX_CHUNKS_PER_DOCUMENT}")
    logger.info(f"Max Text Length/Chunk: {settings.MAX_TEXT_LENGTH_PER_CHUNK}")
    logger.info(f"Max Search Results: {settings.MAX_SEARCH_RESULTS}")
    logger.info(f"Max Query Length: {settings.MAX_QUERY_LENGTH}")
    if settings.workers > 1:
        logger.warning("⚠️  Running with multiple workers. Be aware of in-memory state limitations.")

    # Start event bus for real-time notifications (CDC)
    from app.events.bus import get_event_bus, Event
    event_bus = get_event_bus()
    await event_bus.start()
    logger.info("✓ Event bus started for real-time notifications")

    # Connect event bus to WebSocket manager for real-time event broadcasting
    from app.websockets.manager import get_connection_manager
    connection_manager = get_connection_manager()

    async def forward_event_to_websockets(event: Event):
        """Forward events from event bus to WebSocket subscribers."""
        await connection_manager.broadcast_event(
            event_type=event.type.value,
            library_id=event.library_id,
            data=event.data,
        )

    # Subscribe to all events for WebSocket forwarding
    event_bus.subscribe(forward_event_to_websockets)
    logger.info("✓ WebSocket event forwarding enabled")

    # Start job queue for background operations
    from app.jobs.queue import get_job_queue
    from app.jobs.handlers import register_default_handlers
    from app.api.dependencies import get_library_service

    job_queue = get_job_queue()
    await job_queue.start()
    logger.info("✓ Job queue started for background operations")

    # Register default job handlers
    try:
        from app.api.dependencies import get_library_repository, get_embedding_service

        # Create actual service instances (not Depends wrappers)
        library_repository = get_library_repository()
        embedding_service_instance = get_embedding_service()
        library_service_instance = LibraryService(library_repository, embedding_service_instance)

        register_default_handlers(job_queue, library_service_instance)
        logger.info("✓ Job handlers registered")
    except Exception as e:
        logger.error(f"✗ Failed to register job handlers: {e}")

    # Restore state and regenerate embeddings for any loaded libraries
    try:
        from app.api.dependencies import library_service
        if library_service:
            libraries = library_service.list_libraries()
            if libraries:
                logger.info("=" * 60)
                logger.info(f"Restoring {len(libraries)} libraries from disk")
                logger.info("=" * 60)

                for library in libraries:
                    logger.info(f"Library '{library.name}' ({library.id}): {len(library.documents)} documents")

                    # Regenerate embeddings for this library
                    try:
                        chunks_reembedded = library_service.regenerate_embeddings(library.id)
                        if chunks_reembedded > 0:
                            logger.info(f"  ✓ Regenerated {chunks_reembedded} embeddings")
                        else:
                            logger.info(f"  ✓ All embeddings already present")
                    except Exception as e:
                        logger.error(f"  ✗ Failed to regenerate embeddings: {e}")

                logger.info("=" * 60)
                logger.info("State restoration complete")
                logger.info("=" * 60)
    except Exception as e:
        logger.warning(f"Failed to restore state: {e}")

    logger.info("=" * 60)


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Save state to disk on shutdown."""
    logger.info("=" * 60)
    logger.info("Vector Database API Shutting Down")
    logger.info("=" * 60)
    logger.info("Saving state to disk...")

    # Stop job queue
    try:
        from app.jobs.queue import get_job_queue
        job_queue = get_job_queue()
        await job_queue.stop()
        logger.info("✓ Job queue stopped")
    except Exception as e:
        logger.error(f"✗ Failed to stop job queue: {e}")

    # Stop event bus
    try:
        from app.events.bus import get_event_bus
        event_bus = get_event_bus()
        await event_bus.stop()
        logger.info("✓ Event bus stopped")
    except Exception as e:
        logger.error(f"✗ Failed to stop event bus: {e}")

    try:
        # Get the library service and trigger a save
        from app.api.dependencies import library_service
        if library_service and library_service.repository:
            library_service.repository.save_state()
            logger.info("✓ State saved successfully")
    except Exception as e:
        logger.error(f"✗ Failed to save state: {e}")

    logger.info("=" * 60)
    logger.info("Shutdown complete")
    logger.info("=" * 60)


# ============================================================
# Temporal Workflow Endpoints (Extra Feature)
# ============================================================

@v1_router.post(
    "/workflows/rag",
    tags=["workflows"],
    summary="Start RAG workflow",
    status_code=status.HTTP_202_ACCEPTED,
)
async def start_rag_workflow(
    library_id: UUID,
    query: str = Query(..., description="Search query"),
    k: int = Query(5, ge=1, le=20, description="Number of results"),
):
    """
    Start a Temporal RAG (Retrieval-Augmented Generation) workflow.

    This endpoint starts a durable workflow that:
    1. Preprocesses the query
    2. Generates embedding
    3. Retrieves relevant chunks
    4. Reranks results
    5. Generates an answer

    Returns the workflow ID for tracking.
    """
    if not TEMPORAL_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Temporal workflows not available. Start Temporal server with docker-compose."
        )

    try:
        client = TemporalClient()
        workflow_id = await client.start_rag_workflow(
            library_id=str(library_id),
            query=query,
            k=k
        )

        return {
            "workflow_id": workflow_id,
            "status": "started",
            "message": "RAG workflow started successfully",
            "query": query,
            "library_id": str(library_id)
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start workflow: {str(e)}"
        )


@v1_router.get(
    "/workflows/{workflow_id}",
    tags=["workflows"],
    summary="Get workflow status",
)
async def get_workflow_status(workflow_id: str):
    """
    Get the status and result of a RAG workflow.

    Returns the workflow execution status and result if completed.
    """
    if not TEMPORAL_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Temporal workflows not available"
        )

    try:
        client = TemporalClient()
        result = await client.get_workflow_result(workflow_id)

        return {
            "workflow_id": workflow_id,
            "status": "completed",
            "result": result
        }
    except Exception as e:
        return {
            "workflow_id": workflow_id,
            "status": "running or failed",
            "error": str(e)
        }


# Include v1 router (MUST be after all v1_router endpoints are defined)
app.include_router(v1_router)

# Include streaming router for async/real-time operations
from app.api import streaming
app.include_router(streaming.router, prefix=API_V1_PREFIX, tags=["streaming"])

# Include WebSocket router for bidirectional real-time communication
from app.api import websocket_routes
app.include_router(websocket_routes.router)

# Include job queue router for background operations
from app.api import job_routes
app.include_router(job_routes.router)

# Include event bus router for CDC and event monitoring
from app.api import event_routes
app.include_router(event_routes.router)

# Include health check endpoints for monitoring and probes
from app.api import health
app.include_router(health.router)

# Include webhook routes for event notifications
from app.api import webhook_routes
app.include_router(webhook_routes.router)

# Include SearchReplay routes for debugging vector search (NOVEL FEATURE)
from app.api import search_replay_routes
app.include_router(search_replay_routes.router, prefix=API_V1_PREFIX)


# Root endpoint


@app.get("/", tags=["Root"])
def root() -> Dict[str, Any]:
    """
    Root endpoint with API information and available versions.
    """
    return {
        "name": "Vector Database API",
        "version": API_VERSION,
        "description": "Production-grade vector similarity search API",
        "documentation": "/docs",
        "health_check": "/health",
        "api_versions": {
            "v1": {
                "prefix": API_V1_PREFIX,
                "status": "stable",
                "endpoints": f"{API_V1_PREFIX}/libraries, {API_V1_PREFIX}/documents, {API_V1_PREFIX}/search"
            }
        }
    }
