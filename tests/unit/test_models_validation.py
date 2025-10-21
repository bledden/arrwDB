"""
Tests for Pydantic model validators to ensure data integrity.

Tests validation logic in app/models/base.py for Chunk and Document models.
"""

import pytest
import numpy as np
from uuid import uuid4
from app.models.base import Chunk, Document, ChunkMetadata, DocumentMetadata


class TestChunkEmbeddingValidation:
    """Test Chunk embedding validator (lines 55, 60)."""

    def test_embedding_with_nan_raises_error(self):
        """Test that embedding with NaN raises ValueError (line 60)."""
        doc_id = uuid4()
        metadata = ChunkMetadata(chunk_index=0, source_document_id=doc_id)

        with pytest.raises(ValueError) as exc_info:
            Chunk(
                id=uuid4(),
                text="Test text",
                embedding=[0.1, 0.2, float('nan'), 0.4],  # Contains NaN
                metadata=metadata
            )

        error_msg = str(exc_info.value).lower()
        assert "invalid" in error_msg or "nan" in error_msg

    def test_embedding_with_inf_raises_error(self):
        """Test that embedding with Inf raises ValueError (line 60)."""
        doc_id = uuid4()
        metadata = ChunkMetadata(chunk_index=0, source_document_id=doc_id)

        with pytest.raises(ValueError) as exc_info:
            Chunk(
                id=uuid4(),
                text="Test text",
                embedding=[0.1, float('inf'), 0.3],  # Contains Inf
                metadata=metadata
            )

        error_msg = str(exc_info.value).lower()
        assert "invalid" in error_msg or "inf" in error_msg

    def test_embedding_with_negative_inf_raises_error(self):
        """Test that embedding with -Inf raises ValueError (line 60)."""
        doc_id = uuid4()
        metadata = ChunkMetadata(chunk_index=0, source_document_id=doc_id)

        with pytest.raises(ValueError) as exc_info:
            Chunk(
                id=uuid4(),
                text="Test text",
                embedding=[0.1, float('-inf'), 0.3],  # Contains -Inf
                metadata=metadata
            )

        error_msg = str(exc_info.value).lower()
        assert "invalid" in error_msg or "inf" in error_msg

    def test_valid_embedding_succeeds(self):
        """Test that valid embedding passes validation."""
        doc_id = uuid4()
        metadata = ChunkMetadata(chunk_index=0, source_document_id=doc_id)

        chunk = Chunk(
            id=uuid4(),
            text="Test text",
            embedding=[0.1, 0.2, 0.3, 0.4],
            metadata=metadata
        )

        assert len(chunk.embedding) == 4
        assert chunk.text == "Test text"


class TestDocumentChunksValidation:
    """Test Document chunks validator (lines 114, 119)."""

    def test_inconsistent_chunk_dimensions_raises_error(self):
        """Test that inconsistent embedding dimensions raise ValueError (line 119)."""
        doc_id = uuid4()

        chunk1 = Chunk(
            id=uuid4(),
            text="First chunk",
            embedding=[0.1, 0.2, 0.3],  # 3 dimensions
            metadata=ChunkMetadata(chunk_index=0, source_document_id=doc_id)
        )

        chunk2 = Chunk(
            id=uuid4(),
            text="Second chunk",
            embedding=[0.4, 0.5, 0.6, 0.7],  # 4 dimensions - mismatch!
            metadata=ChunkMetadata(chunk_index=1, source_document_id=doc_id)
        )

        doc_metadata = DocumentMetadata(title="Test Document")

        with pytest.raises(ValueError) as exc_info:
            Document(
                id=doc_id,
                chunks=[chunk1, chunk2],  # Inconsistent dimensions
                metadata=doc_metadata
            )

        error_msg = str(exc_info.value).lower()
        assert "inconsistent" in error_msg or "dimension" in error_msg

    def test_consistent_chunk_dimensions_succeeds(self):
        """Test that consistent embedding dimensions pass validation."""
        doc_id = uuid4()

        chunks = [
            Chunk(
                id=uuid4(),
                text=f"Chunk {i}",
                embedding=[0.1 * i, 0.2 * i, 0.3 * i],  # All 3 dimensions
                metadata=ChunkMetadata(chunk_index=i, source_document_id=doc_id)
            )
            for i in range(5)
        ]

        doc_metadata = DocumentMetadata(title="Test Document", total_chunks=5)

        doc = Document(
            id=doc_id,
            chunks=chunks,
            metadata=doc_metadata
        )

        assert len(doc.chunks) == 5
        # Verify all chunks have same dimension
        first_dim = len(doc.chunks[0].embedding)
        assert all(len(chunk.embedding) == first_dim for chunk in doc.chunks)

    def test_single_chunk_always_valid(self):
        """Test that single chunk is always valid (no comparison needed)."""
        doc_id = uuid4()

        chunk = Chunk(
            id=uuid4(),
            text="Only chunk",
            embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
            metadata=ChunkMetadata(chunk_index=0, source_document_id=doc_id)
        )

        doc_metadata = DocumentMetadata(title="Single Chunk Doc")

        doc = Document(
            id=doc_id,
            chunks=[chunk],
            metadata=doc_metadata
        )

        assert len(doc.chunks) == 1


class TestChunkImmutability:
    """Test that Chunk is immutable (frozen)."""

    def test_chunk_is_frozen(self):
        """Test that Chunk cannot be modified after creation."""
        doc_id = uuid4()
        metadata = ChunkMetadata(chunk_index=0, source_document_id=doc_id)

        chunk = Chunk(
            id=uuid4(),
            text="Original text",
            embedding=[0.1, 0.2, 0.3],
            metadata=metadata
        )

        # Pydantic v2 frozen models raise ValidationError on modification
        with pytest.raises(Exception):  # Could be ValidationError or AttributeError
            chunk.text = "Modified text"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
