"""Service layer for the Vector Database."""

from app.services.embedding_service import EmbeddingService, EmbeddingServiceError

__all__ = ["EmbeddingService", "EmbeddingServiceError"]
