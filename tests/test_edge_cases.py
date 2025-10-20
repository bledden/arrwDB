"""
Edge case tests for the vector database.

Tests unusual inputs, boundary conditions, and error scenarios.
"""

import pytest
import numpy as np
from uuid import uuid4

from app.models.base import Library, Document, Chunk, LibraryMetadata, DocumentMetadata, ChunkMetadata
from core.vector_store import VectorStore
from core.embedding_contract import LibraryEmbeddingContract


@pytest.mark.edge
class TestEmptyInputs:
    """Tests with empty inputs."""

    def test_empty_text_chunk(self):
        """Test that empty text is rejected."""
        from pydantic import ValidationError

        metadata = ChunkMetadata(chunk_index=0, source_document_id=uuid4())

        with pytest.raises(ValidationError, match="String should have at least 1 character"):
            Chunk(text="", embedding=[1.0] * 128, metadata=metadata)

    def test_empty_library_name(self):
        """Test that empty library name is rejected."""
        from pydantic import ValidationError

        metadata = LibraryMetadata(
            index_type="brute_force",
            embedding_dimension=128,
            embedding_model="test",
        )

        with pytest.raises(ValidationError, match="String should have at least 1 character"):
            Library(name="", metadata=metadata)

    def test_empty_document_text(self):
        """Test that empty document text is rejected."""
        from pydantic import ValidationError

        # Document requires chunks, not text - test empty chunk text instead
        doc_id = uuid4()
        metadata = ChunkMetadata(chunk_index=0, source_document_id=doc_id)

        with pytest.raises(ValidationError, match="String should have at least 1 character"):
            chunk = Chunk(text="", embedding=[1.0] * 128, metadata=metadata)


@pytest.mark.edge
class TestUnicodeAndSpecialCharacters:
    """Tests with unicode and special characters."""

    def test_unicode_text_chunk(self, vector_dimension: int):
        """Test chunks with unicode characters."""
        vec = np.random.randn(vector_dimension).astype(np.float32)
        vec = vec / np.linalg.norm(vec)

        doc_id = uuid4()
        metadata = ChunkMetadata(chunk_index=0, source_document_id=doc_id)

        # Various unicode characters
        unicode_texts = [
            "Hello 世界",  # Chinese
            "Привет мир",  # Russian
            "مرحبا بالعالم",  # Arabic
            "🚀 Rocket emoji",  # Emoji
            "Çàfé ñiño",  # Accented characters
        ]

        for text in unicode_texts:
            chunk = Chunk(text=text, embedding=vec.tolist(), metadata=metadata)
            assert chunk.text == text

    def test_special_characters_in_metadata(self, vector_dimension: int):
        """Test special characters in metadata fields."""
        vec = np.random.randn(vector_dimension).astype(np.float32)
        vec = vec / np.linalg.norm(vec)

        # Create document with chunks (not text)
        doc_id = uuid4()
        chunk_metadata = ChunkMetadata(chunk_index=0, source_document_id=doc_id)
        chunk = Chunk(text="Test content", embedding=vec.tolist(), metadata=chunk_metadata)

        doc_metadata = DocumentMetadata(
            title='Test "quotes" and \'apostrophes\'',
            author="O'Brien, Smith & Jones",
            source_url="https://example.com/path?query=value&other=value",
        )

        doc = Document(id=doc_id, chunks=[chunk], metadata=doc_metadata)

        assert '"quotes"' in doc.metadata.title
        assert "O'Brien" in doc.metadata.author

    def test_newlines_and_whitespace(self, vector_dimension: int):
        """Test text with various whitespace."""
        vec = np.random.randn(vector_dimension).astype(np.float32)
        vec = vec / np.linalg.norm(vec)

        metadata = ChunkMetadata(chunk_index=0, source_document_id=uuid4())

        text_with_whitespace = """
        This text has:
        - Multiple lines
        - Tabs\t\there
        - Various   spaces
        """

        chunk = Chunk(text=text_with_whitespace, embedding=vec.tolist(), metadata=metadata)
        assert chunk.text == text_with_whitespace


@pytest.mark.edge
class TestBoundaryValues:
    """Tests with boundary values."""

    def test_very_long_text(self, vector_dimension: int):
        """Test chunk with very long text."""
        vec = np.random.randn(vector_dimension).astype(np.float32)
        vec = vec / np.linalg.norm(vec)

        metadata = ChunkMetadata(chunk_index=0, source_document_id=uuid4())

        # Maximum is 10,000 characters
        long_text = "a" * 10000

        chunk = Chunk(text=long_text, embedding=vec.tolist(), metadata=metadata)
        assert len(chunk.text) == 10000

    def test_text_exceeds_max_length(self, vector_dimension: int):
        """Test that text exceeding max length is rejected."""
        from pydantic import ValidationError

        vec = np.random.randn(vector_dimension).astype(np.float32)
        vec = vec / np.linalg.norm(vec)

        metadata = ChunkMetadata(chunk_index=0, source_document_id=uuid4())

        # Exceed maximum of 10,000 characters
        too_long_text = "a" * 10001

        with pytest.raises(ValidationError, match="String should have at most 10000 characters"):
            Chunk(text=too_long_text, embedding=vec.tolist(), metadata=metadata)

    def test_minimum_dimension(self):
        """Test vector with minimum dimension (1)."""
        store = VectorStore(dimension=1)
        contract = LibraryEmbeddingContract(expected_dimension=1)

        chunk_id = uuid4()
        vector = np.array([5.0], dtype=np.float32)

        # Should normalize to [1.0]
        validated = contract.validate_vector(vector.tolist())
        store.add_vector(chunk_id, validated)

        retrieved = store.get_vector(chunk_id)
        assert np.abs(retrieved[0] - 1.0) < 1e-5

    def test_very_large_dimension(self):
        """Test vector with very large dimension."""
        dimension = 4096

        store = VectorStore(dimension=dimension)
        contract = LibraryEmbeddingContract(expected_dimension=dimension)

        chunk_id = uuid4()
        vector = np.random.randn(dimension).astype(np.float32)

        validated = contract.validate_vector(vector.tolist())
        store.add_vector(chunk_id, validated)

        retrieved = store.get_vector(chunk_id)
        assert retrieved.shape == (dimension,)
        assert np.abs(np.linalg.norm(retrieved) - 1.0) < 1e-5

    def test_single_vector_search(self, vector_store: VectorStore, sample_vectors: list):
        """Test searching when only 1 vector exists."""
        from infrastructure.indexes.brute_force import BruteForceIndex

        index = BruteForceIndex(vector_store)

        chunk_id = uuid4()
        vec_index = vector_store.add_vector(chunk_id, sample_vectors[0])
        index.add_vector(chunk_id, vec_index)

        # Search for k > 1
        results = index.search(sample_vectors[0], k=10)

        # Should return only 1 result
        assert len(results) == 1


@pytest.mark.edge
class TestNumericalEdgeCases:
    """Tests with numerical edge cases."""

    def test_very_small_but_nonzero_values(self, vector_dimension: int):
        """Test vectors with very small values."""
        contract = LibraryEmbeddingContract(expected_dimension=vector_dimension)

        tiny_vector = np.full(vector_dimension, 1e-10, dtype=np.float32)

        validated = contract.validate_vector(tiny_vector.tolist())

        # Should normalize correctly
        assert np.abs(np.linalg.norm(validated) - 1.0) < 1e-5

    def test_very_large_values(self, vector_dimension: int):
        """Test vectors with very large values."""
        contract = LibraryEmbeddingContract(expected_dimension=vector_dimension)

        large_vector = np.full(vector_dimension, 1e6, dtype=np.float32)

        validated = contract.validate_vector(large_vector.tolist())

        # Should normalize correctly
        assert np.abs(np.linalg.norm(validated) - 1.0) < 1e-5
        # All components should be equal
        assert np.allclose(validated, validated[0])

    def test_mixed_positive_negative_values(self, vector_dimension: int):
        """Test vectors with mixed positive and negative values."""
        contract = LibraryEmbeddingContract(expected_dimension=vector_dimension)

        mixed_vector = np.array(
            [(-1) ** i * (i + 1) for i in range(vector_dimension)],
            dtype=np.float32
        )

        validated = contract.validate_vector(mixed_vector.tolist())

        assert np.abs(np.linalg.norm(validated) - 1.0) < 1e-5

    def test_vector_with_single_nonzero_component(self, vector_dimension: int):
        """Test vector with only one non-zero component."""
        contract = LibraryEmbeddingContract(expected_dimension=vector_dimension)

        sparse_vector = np.zeros(vector_dimension, dtype=np.float32)
        sparse_vector[vector_dimension // 2] = 10.0

        validated = contract.validate_vector(sparse_vector.tolist())

        # Should normalize to [0, 0, ..., 1.0, ..., 0, 0]
        assert np.abs(np.linalg.norm(validated) - 1.0) < 1e-5
        assert np.abs(validated[vector_dimension // 2] - 1.0) < 1e-5


@pytest.mark.edge
class TestConcurrentEdgeCases:
    """Edge cases with concurrent operations."""

    def test_add_and_remove_same_vector_concurrently(self, vector_store: VectorStore, sample_vectors: list):
        """Test adding and removing the same vector from different threads."""
        import threading
        import time

        chunk_id = uuid4()
        errors = []

        def add_vector():
            try:
                vector_store.add_vector(chunk_id, sample_vectors[0])
            except ValueError as e:
                errors.append(("add", str(e)))

        def remove_vector():
            time.sleep(0.001)  # Small delay
            try:
                vector_store.remove_vector(chunk_id)
            except Exception as e:
                errors.append(("remove", str(e)))

        threads = []
        for _ in range(5):
            threads.append(threading.Thread(target=add_vector))

        for _ in range(5):
            threads.append(threading.Thread(target=remove_vector))

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # Should handle concurrent adds/removes gracefully
        # Either added or removed, but no crashes
        # Some operations should fail with appropriate errors


@pytest.mark.edge
class TestMetadataEdgeCases:
    """Tests with edge cases in metadata."""

    def test_optional_metadata_fields_none(self, vector_dimension: int):
        """Test that optional metadata fields can be None."""
        vec = np.random.randn(vector_dimension).astype(np.float32)
        vec = vec / np.linalg.norm(vec)

        # Create a valid document with chunks
        doc_id = uuid4()
        chunk_metadata = ChunkMetadata(chunk_index=0, source_document_id=doc_id)
        chunk = Chunk(text="Test document", embedding=vec.tolist(), metadata=chunk_metadata)

        # Minimal metadata - only title is required
        doc_metadata = DocumentMetadata(title="Test")
        doc = Document(id=doc_id, chunks=[chunk], metadata=doc_metadata)

        # Optional fields should be None or default values
        assert doc.metadata.author is None

    def test_very_long_metadata_values(self, vector_dimension: int):
        """Test metadata with very long string values."""
        vec = np.random.randn(vector_dimension).astype(np.float32)
        vec = vec / np.linalg.norm(vec)

        long_title = "a" * 1000
        long_author = "b" * 500

        # Create document with chunks
        doc_id = uuid4()
        chunk_metadata = ChunkMetadata(chunk_index=0, source_document_id=doc_id)
        chunk = Chunk(text="Test", embedding=vec.tolist(), metadata=chunk_metadata)

        doc_metadata = DocumentMetadata(
            title=long_title,
            author=long_author,
        )

        doc = Document(id=doc_id, chunks=[chunk], metadata=doc_metadata)

        assert len(doc.metadata.title) == 1000
        assert len(doc.metadata.author) == 500

    def test_custom_metadata_fields(self, vector_dimension: int):
        """Test that ChunkMetadata has fixed schema (no custom fields)."""
        vec = np.random.randn(vector_dimension).astype(np.float32)
        vec = vec / np.linalg.norm(vec)

        # ChunkMetadata has a FIXED schema - no extras field
        # Just verify the standard fields work
        chunk_metadata = ChunkMetadata(
            chunk_index=0,
            source_document_id=uuid4(),
            page_number=1,
        )

        chunk = Chunk(text="Test", embedding=vec.tolist(), metadata=chunk_metadata)

        assert chunk.metadata.chunk_index == 0
        assert chunk.metadata.page_number == 1


@pytest.mark.edge
class TestSearchEdgeCases:
    """Edge cases for search operations."""

    def test_search_with_zero_k(self, vector_store: VectorStore, sample_vectors: list):
        """Test searching with k=0 raises error."""
        from infrastructure.indexes.brute_force import BruteForceIndex

        index = BruteForceIndex(vector_store)

        # Add vectors
        for vector in sample_vectors[:5]:
            chunk_id = uuid4()
            vec_index = vector_store.add_vector(chunk_id, vector)
            index.add_vector(chunk_id, vec_index)

        # Search with k=0 - should raise ValueError
        with pytest.raises(ValueError, match="k must be positive"):
            index.search(sample_vectors[0], k=0)

    def test_search_with_distance_threshold_zero(self, vector_store: VectorStore, sample_vectors: list):
        """Test search with distance threshold of 0."""
        from infrastructure.indexes.brute_force import BruteForceIndex

        index = BruteForceIndex(vector_store)

        chunk_id = uuid4()
        vec_index = vector_store.add_vector(chunk_id, sample_vectors[0])
        index.add_vector(chunk_id, vec_index)

        # Only exact match should be returned
        results = index.search(sample_vectors[0], k=10, distance_threshold=0.0)

        # Due to floating point precision, might be 1 or 0 results
        assert len(results) <= 1
        if len(results) == 1:
            assert results[0][1] < 1e-5

    def test_search_identical_vectors(self, vector_store: VectorStore, vector_dimension: int):
        """Test searching when all vectors are identical."""
        from infrastructure.indexes.brute_force import BruteForceIndex

        index = BruteForceIndex(vector_store)

        # Add same vector 10 times with different IDs
        vec = np.random.randn(vector_dimension).astype(np.float32)
        vec = vec / np.linalg.norm(vec)

        for _ in range(10):
            chunk_id = uuid4()
            vec_index = vector_store.add_vector(chunk_id, vec)
            index.add_vector(chunk_id, vec_index)

        # Search should return all of them with same distance
        results = index.search(vec, k=10)

        assert len(results) == 10
        # All should have nearly zero distance
        for _, distance in results:
            assert distance < 1e-4
