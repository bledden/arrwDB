"""
FastAPI application with REST endpoints for the Vector Database.

This module provides the main FastAPI application with all CRUD endpoints
for libraries, documents, and chunks, plus search functionality.
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from typing import List
from uuid import UUID
import time
from datetime import datetime
import logging

from app.api.models import (
    CreateLibraryRequest,
    AddDocumentRequest,
    AddDocumentWithEmbeddingsRequest,
    SearchRequest,
    SearchWithEmbeddingRequest,
    LibraryResponse,
    LibrarySummaryResponse,
    DocumentResponse,
    SearchResponse,
    SearchResultResponse,
    LibraryStatisticsResponse,
    ErrorResponse,
    HealthResponse,
)
from app.api.dependencies import get_library_service
from app.services.library_service import LibraryService
from infrastructure.repositories.library_repository import (
    LibraryNotFoundError,
    DocumentNotFoundError,
    DimensionMismatchError,
)
from app.services.embedding_service import EmbeddingServiceError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Vector Database API",
    description="Production-grade REST API for vector similarity search with multiple indexing algorithms",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


# Exception Handlers


@app.exception_handler(LibraryNotFoundError)
async def library_not_found_handler(request, exc):
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={
            "error": "Library not found",
            "detail": str(exc),
            "error_type": "LibraryNotFoundError",
        },
    )


@app.exception_handler(DocumentNotFoundError)
async def document_not_found_handler(request, exc):
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={
            "error": "Document not found",
            "detail": str(exc),
            "error_type": "DocumentNotFoundError",
        },
    )


@app.exception_handler(DimensionMismatchError)
async def dimension_mismatch_handler(request, exc):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": "Dimension mismatch",
            "detail": str(exc),
            "error_type": "DimensionMismatchError",
        },
    )


@app.exception_handler(EmbeddingServiceError)
async def embedding_service_error_handler(request, exc):
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "error": "Embedding service error",
            "detail": str(exc),
            "error_type": "EmbeddingServiceError",
        },
    )


@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
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
async def health_check():
    """Check if the API is running."""
    return HealthResponse(
        status="healthy", version="1.0.0", timestamp=datetime.utcnow()
    )


# Library Endpoints


@app.post(
    "/libraries",
    response_model=LibraryResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Libraries"],
    summary="Create a new library",
)
async def create_library(
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


@app.get(
    "/libraries",
    response_model=List[LibrarySummaryResponse],
    tags=["Libraries"],
    summary="List all libraries",
)
async def list_libraries(
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


@app.get(
    "/libraries/{library_id}",
    response_model=LibraryResponse,
    tags=["Libraries"],
    summary="Get a library by ID",
)
async def get_library(
    library_id: UUID,
    service: LibraryService = Depends(get_library_service),
):
    """
    Get full details of a library including all documents.
    """
    return service.get_library(library_id)


@app.delete(
    "/libraries/{library_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["Libraries"],
    summary="Delete a library",
)
async def delete_library(
    library_id: UUID,
    service: LibraryService = Depends(get_library_service),
):
    """
    Delete a library and all its documents and vectors.
    """
    deleted = service.delete_library(library_id)
    if not deleted:
        raise LibraryNotFoundError(f"Library {library_id} not found")


@app.get(
    "/libraries/{library_id}/statistics",
    response_model=LibraryStatisticsResponse,
    tags=["Libraries"],
    summary="Get library statistics",
)
async def get_library_statistics(
    library_id: UUID,
    service: LibraryService = Depends(get_library_service),
):
    """
    Get detailed statistics about a library including vector store and index metrics.
    """
    return service.get_library_statistics(library_id)


# Document Endpoints


@app.post(
    "/libraries/{library_id}/documents",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Documents"],
    summary="Add a document with text chunks",
)
async def add_document(
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


@app.post(
    "/libraries/{library_id}/documents/with-embeddings",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Documents"],
    summary="Add a document with pre-computed embeddings",
)
async def add_document_with_embeddings(
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


@app.get(
    "/documents/{document_id}",
    response_model=DocumentResponse,
    tags=["Documents"],
    summary="Get a document by ID",
)
async def get_document(
    document_id: UUID,
    service: LibraryService = Depends(get_library_service),
):
    """
    Get a document by its ID.
    """
    return service.get_document(document_id)


@app.delete(
    "/documents/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["Documents"],
    summary="Delete a document",
)
async def delete_document(
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


@app.post(
    "/libraries/{library_id}/search",
    response_model=SearchResponse,
    tags=["Search"],
    summary="Search with text query",
)
async def search(
    library_id: UUID,
    request: SearchRequest,
    service: LibraryService = Depends(get_library_service),
):
    """
    Search a library using a natural language query.

    The query text will be embedded automatically and compared against
    all chunks in the library.

    - **query**: Natural language search query
    - **k**: Number of results to return (1-100)
    - **distance_threshold**: Optional maximum distance (0-2 for cosine)

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

    # Build response
    search_results = []
    for chunk, distance in results:
        # Get document info
        doc = service.get_document(chunk.metadata.source_document_id)

        search_results.append(
            SearchResultResponse(
                chunk=chunk,
                distance=distance,
                document_id=doc.id,
                document_title=doc.metadata.title,
            )
        )

    return SearchResponse(
        results=search_results,
        query_time_ms=round(query_time_ms, 2),
        total_results=len(search_results),
    )


@app.post(
    "/libraries/{library_id}/search/embedding",
    response_model=SearchResponse,
    tags=["Search"],
    summary="Search with embedding",
)
async def search_with_embedding(
    library_id: UUID,
    request: SearchWithEmbeddingRequest,
    service: LibraryService = Depends(get_library_service),
):
    """
    Search a library using a pre-computed embedding.

    Use this when you've already generated a query embedding.

    - **embedding**: Query vector (must match library's dimension)
    - **k**: Number of results to return (1-100)
    - **distance_threshold**: Optional maximum distance (0-2 for cosine)

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

    # Build response
    search_results = []
    for chunk, distance in results:
        # Get document info
        doc = service.get_document(chunk.metadata.source_document_id)

        search_results.append(
            SearchResultResponse(
                chunk=chunk,
                distance=distance,
                document_id=doc.id,
                document_title=doc.metadata.title,
            )
        )

    return SearchResponse(
        results=search_results,
        query_time_ms=round(query_time_ms, 2),
        total_results=len(search_results),
    )


# Root endpoint


@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint with API information.
    """
    return {
        "name": "Vector Database API",
        "version": "1.0.0",
        "description": "Production-grade vector similarity search API",
        "documentation": "/docs",
        "health_check": "/health",
    }
