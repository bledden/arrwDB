"""
FastAPI application with REST endpoints for the Vector Database.

This module provides the main FastAPI application with all CRUD endpoints
for libraries, documents, and chunks, plus search functionality.
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Union
from uuid import UUID

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.api.dependencies import get_library_service
from app.api.models import (
    AddDocumentRequest,
    AddDocumentWithEmbeddingsRequest,
    ChunkResponseSlim,
    CreateLibraryRequest,
    DocumentResponse,
    DocumentResponseSlim,
    ErrorResponse,
    HealthResponse,
    LibraryResponse,
    LibraryStatisticsResponse,
    LibrarySummaryResponse,
    SearchRequest,
    SearchResponse,
    SearchResponseSlim,
    SearchResultResponse,
    SearchResultResponseSlim,
    SearchWithEmbeddingRequest,
    SearchWithMetadataRequest,
)
from app.config import settings
from app.services.embedding_service import EmbeddingServiceError
from app.services.library_service import LibraryService
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

# Configure logging
logging.basicConfig(level=logging.INFO)
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
    summary="Health check endpoint",
)
async def health_check() -> HealthResponse:
    """Check if the API is running."""
    return HealthResponse(
        status="healthy", version=API_VERSION, timestamp=datetime.utcnow()
    )


# Library Endpoints


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
    - **index_type**: Type of index (brute_force, kd_tree, lsh, hnsw)
    - **embedding_model**: Optional model override
    """
    library = service.create_library(
        name=request.name,
        description=request.description,
        index_type=request.index_type,
        embedding_model=request.embedding_model,
    )
    return library


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

    results = service.search_with_text(
        library_id=library_id,
        query_text=request.query,
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


# Startup event


@app.on_event("startup")
async def startup_event() -> None:
    """Log configuration summary on startup."""
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
