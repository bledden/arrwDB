"""
FastAPI dependencies for dependency injection.

This module provides dependency injection functions that initialize
and provide services to API endpoints.
"""

from functools import lru_cache
from pathlib import Path
import os
from fastapi import Depends

from app.services.library_service import LibraryService
from app.services.embedding_service import EmbeddingService
from infrastructure.repositories.library_repository import LibraryRepository


@lru_cache()
def get_data_dir() -> Path:
    """
    Get the data directory path from environment or use default.

    Returns:
        Path to the data directory.
    """
    data_dir_str = os.getenv("VECTOR_DB_DATA_DIR", "./data")
    return Path(data_dir_str).resolve()


@lru_cache()
def get_library_repository() -> LibraryRepository:
    """
    Get or create the library repository singleton.

    Returns:
        The library repository instance.
    """
    data_dir = get_data_dir()
    return LibraryRepository(data_dir)


@lru_cache()
def get_embedding_service() -> EmbeddingService:
    """
    Get or create the embedding service singleton.

    Returns:
        The embedding service instance.

    Raises:
        ValueError: If COHERE_API_KEY environment variable is not set.
    """
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise ValueError(
            "COHERE_API_KEY environment variable must be set. "
            "Get your API key from https://dashboard.cohere.com/api-keys"
        )

    model = os.getenv("EMBEDDING_MODEL", "embed-english-v3.0")
    dimension_str = os.getenv("EMBEDDING_DIMENSION", "1024")

    try:
        dimension = int(dimension_str)
    except ValueError:
        dimension = 1024

    return EmbeddingService(
        api_key=api_key,
        model=model,
        input_type="search_document",
        embedding_dimension=dimension if dimension < 1024 else None,
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
