"""
Base models and shared components for the Vector Database.

This module provides the foundational Pydantic models that define
the data schema for Chunks, Documents, and Libraries with FIXED schemas.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime
from uuid import UUID, uuid4
import numpy as np

# Import settings for consistent defaults
from app.config import settings


class ChunkMetadata(BaseModel):
    """
    Fixed schema metadata for chunks.
    Users cannot add custom fields to this schema.
    """

    created_at: datetime = Field(default_factory=datetime.utcnow)
    page_number: Optional[int] = Field(None, ge=0)
    chunk_index: int = Field(..., ge=0)
    source_document_id: UUID

    class Config:
        json_schema_extra = {
            "example": {
                "created_at": "2024-01-01T00:00:00Z",
                "page_number": 1,
                "chunk_index": 0,
                "source_document_id": "123e4567-e89b-12d3-a456-426614174000",
            }
        }


class Chunk(BaseModel):
    """
    Immutable chunk with text and embedding.
    Once created, cannot be modified (frozen=True).

    The embedding is stored as List[float] for Pydantic compatibility,
    but will be converted to numpy arrays for computations.
    """

    id: UUID = Field(default_factory=uuid4)
    text: str = Field(..., min_length=1, max_length=10000)
    embedding: List[float] = Field(..., min_length=1)
    metadata: ChunkMetadata

    @field_validator("embedding")
    @classmethod
    def validate_embedding(cls, v: List[float]) -> List[float]:
        """Ensure embedding is valid and contains no invalid values."""
        if not v:
            raise ValueError("Embedding cannot be empty")

        # Check for NaN or Inf values
        arr = np.array(v)
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            raise ValueError("Embedding contains invalid values (NaN or Inf)")

        return v

    class Config:
        frozen = True  # Immutable after creation
        json_schema_extra = {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "text": "This is a sample chunk of text.",
                "embedding": [0.1, 0.2, 0.3],
                "metadata": {
                    "created_at": "2024-01-01T00:00:00Z",
                    "chunk_index": 0,
                    "source_document_id": "123e4567-e89b-12d3-a456-426614174001",
                },
            }
        }


class DocumentMetadata(BaseModel):
    """Fixed schema for document metadata."""

    title: str
    author: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    document_type: str = Field(default="text")
    source_url: Optional[str] = None
    tags: List[str] = Field(default_factory=list)

    class Config:
        json_schema_extra = {
            "example": {
                "title": "Sample Document",
                "author": "John Doe",
                "created_at": "2024-01-01T00:00:00Z",
                "document_type": "text",
                "source_url": "https://example.com/doc.pdf",
                "tags": ["research", "ai"],
            }
        }


class Document(BaseModel):
    """Document containing multiple chunks."""

    id: UUID = Field(default_factory=uuid4)
    chunks: List[Chunk] = Field(..., min_length=1)
    metadata: DocumentMetadata

    @field_validator("chunks")
    @classmethod
    def validate_chunks_consistency(cls, v: List[Chunk]) -> List[Chunk]:
        """Ensure all chunks have same embedding dimension."""
        if not v:
            return v

        first_dim = len(v[0].embedding)
        for i, chunk in enumerate(v[1:], 1):
            if len(chunk.embedding) != first_dim:
                raise ValueError(
                    f"Inconsistent embedding dimensions: "
                    f"chunk 0 has {first_dim}, chunk {i} has {len(chunk.embedding)}"
                )

        return v

    class Config:
        json_schema_extra = {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174002",
                "chunks": [],
                "metadata": {"title": "Sample Document"},
            }
        }


class LibraryMetadata(BaseModel):
    """Fixed schema for library metadata."""

    description: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    index_type: Literal["brute_force", "kd_tree", "lsh", "hnsw"] = "brute_force"
    embedding_dimension: int = Field(default=settings.EMBEDDING_DIMENSION, ge=1, le=4096)
    embedding_model: str = Field(default="embed-english-v3.0")

    class Config:
        json_schema_extra = {
            "example": {
                "description": "Research papers collection",
                "created_at": "2024-01-01T00:00:00Z",
                "index_type": "hnsw",
                "embedding_dimension": 1024,
                "embedding_model": "embed-english-v3.0",
            }
        }


class Library(BaseModel):
    """
    Library containing documents and managing indexes.

    Note: The actual index and VectorStore are not Pydantic fields
    as they're not easily serializable. They're managed separately
    in memory/disk by the IndexManager and VectorStoreManager.
    """

    id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., min_length=1, max_length=255)
    documents: List[Document] = Field(default_factory=list)
    metadata: LibraryMetadata = Field(default_factory=LibraryMetadata)

    class Config:
        json_schema_extra = {
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174003",
                "name": "My Research Library",
                "documents": [],
                "metadata": {
                    "description": "Collection of AI research papers",
                    "index_type": "hnsw",
                },
            }
        }
