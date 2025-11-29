"""
FastAPI dependencies for dependency injection.

This module provides dependency injection functions that initialize
and provide services to API endpoints.
"""

import os
from functools import lru_cache
from pathlib import Path

from fastapi import Depends

from app.config import settings
from app.services.embedding_service import EmbeddingService
from app.services.library_service import LibraryService
from infrastructure.repositories.library_repository import LibraryRepository


@lru_cache()
def get_data_dir() -> Path:
    """
    Get the data directory path from environment or use default.

    Returns:
        Path to the data directory.
    """
    return Path(settings.VECTOR_DB_DATA_DIR).resolve()


@lru_cache()
def get_library_repository() -> LibraryRepository:
    """
    Get or create the library repository singleton.

    Returns:
        The library repository instance.
    """
    from app.events.bus import get_event_bus

    data_dir = get_data_dir()
    event_bus = get_event_bus()
    return LibraryRepository(data_dir, event_bus=event_bus)


@lru_cache()
def get_embedding_service() -> EmbeddingService:
    """
    Get or create the embedding service singleton.

    Returns:
        The embedding service instance.

    Raises:
        ValueError: If COHERE_API_KEY environment variable is not set.
    """
    if not settings.COHERE_API_KEY:
        raise ValueError(
            "COHERE_API_KEY environment variable must be set. "
            "Get your API key from https://dashboard.cohere.com/api-keys"
        )

    return EmbeddingService(
        api_key=settings.COHERE_API_KEY,
        model=settings.EMBEDDING_MODEL,
        input_type="search_document",
        embedding_dimension=settings.EMBEDDING_DIMENSION if settings.EMBEDDING_DIMENSION < 1024 else None,
    )


def get_library_service(
    repository: LibraryRepository = Depends(get_library_repository),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
) -> LibraryService:
    """
    Get the library service instance.

    This is not cached as it depends on injected services.

    Args:
        repository: Repository instance from dependency injection.
        embedding_service: Embedding service from dependency injection.

    Returns:
        The library service instance.
    """
    return LibraryService(repository, embedding_service)


# Module-level singletons for health checks and startup routines
# These are lazily initialized on first access
_embedding_service_instance = None
_library_service_instance = None


def _get_embedding_service_singleton():
    """Get or create the embedding service singleton."""
    global _embedding_service_instance
    if _embedding_service_instance is None:
        try:
            _embedding_service_instance = get_embedding_service()
        except Exception:
            return None
    return _embedding_service_instance


def _get_library_service_singleton():
    """Get or create the library service singleton."""
    global _library_service_instance
    if _library_service_instance is None:
        try:
            repo = get_library_repository()
            emb = _get_embedding_service_singleton()
            if emb:
                _library_service_instance = LibraryService(repo, emb)
        except Exception:
            return None
    return _library_service_instance


# Expose as module-level for backward compatibility with health checks
# These are evaluated lazily when first accessed
embedding_service = _get_embedding_service_singleton()
library_service = _get_library_service_singleton()
