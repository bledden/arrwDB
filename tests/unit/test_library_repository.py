"""
Unit tests for LibraryRepository.
"""

import pytest
from uuid import uuid4
import numpy as np

from infrastructure.repositories.library_repository import (
    LibraryRepository,
    LibraryNotFoundError,
    DocumentNotFoundError,
    ChunkNotFoundError,
    DimensionMismatchError,
)
from app.models.base import Library, Document, Chunk, LibraryMetadata, DocumentMetadata, ChunkMetadata


@pytest.mark.unit
class TestLibraryRepositoryCRUD:
    """Tests for library CRUD operations."""

    def test_create_library(self, library_repository: LibraryRepository, sample_library: Library):
        """Test creating a library."""
        created = library_repository.create_library(sample_library)

        assert created.id == sample_library.id
        assert created.name == sample_library.name

    def test_get_library(self, library_repository: LibraryRepository, sample_library: Library):
        """Test getting a library."""
        library_repository.create_library(sample_library)

        retrieved = library_repository.get_library(sample_library.id)
        assert retrieved.id == sample_library.id

    def test_get_nonexistent_library_raises_error(self, library_repository: LibraryRepository):
        """Test getting nonexistent library raises LibraryNotFoundError."""
        with pytest.raises(LibraryNotFoundError):
            library_repository.get_library(uuid4())

    def test_list_libraries_empty(self, library_repository: LibraryRepository):
        """Test listing libraries when empty."""
        libraries = library_repository.list_libraries()
        assert len(libraries) == 0

    def test_list_libraries(self, library_repository: LibraryRepository, sample_library: Library):
        """Test listing libraries."""
        library_repository.create_library(sample_library)

        libraries = library_repository.list_libraries()
        assert len(libraries) == 1
        assert libraries[0].id == sample_library.id

    def test_delete_library(self, library_repository: LibraryRepository, sample_library: Library):
        """Test deleting a library."""
        library_repository.create_library(sample_library)

        deleted = library_repository.delete_library(sample_library.id)
        assert deleted is True

        # Should no longer exist
        with pytest.raises(LibraryNotFoundError):
            library_repository.get_library(sample_library.id)

    def test_delete_nonexistent_library(self, library_repository: LibraryRepository):
        """Test deleting nonexistent library returns False."""
        deleted = library_repository.delete_library(uuid4())
        assert deleted is False


@pytest.mark.unit
class TestDocumentOperations:
    """Tests for document operations."""

    def test_add_document(self, library_repository: LibraryRepository, sample_library: Library, sample_document: Document):
        """Test adding a document to a library."""
        library_repository.create_library(sample_library)
        library_repository.add_document(sample_library.id, sample_document)

        # Verify it was added
        stats = library_repository.get_library_statistics(sample_library.id)
        assert stats["num_documents"] == 1

    def test_add_document_to_nonexistent_library(self, library_repository: LibraryRepository, sample_document: Document):
        """Test adding document to nonexistent library raises error."""
        with pytest.raises(LibraryNotFoundError):
            library_repository.add_document(uuid4(), sample_document)

    def test_get_document(self, library_repository: LibraryRepository, sample_library: Library, sample_document: Document):
        """Test getting a document."""
        library_repository.create_library(sample_library)
        library_repository.add_document(sample_library.id, sample_document)

        retrieved = library_repository.get_document(sample_document.id)
        assert retrieved.id == sample_document.id

    def test_get_nonexistent_document(self, library_repository: LibraryRepository):
        """Test getting nonexistent document raises error."""
        with pytest.raises(DocumentNotFoundError):
            library_repository.get_document(uuid4())

    def test_delete_document(self, library_repository: LibraryRepository, sample_library: Library, sample_document: Document):
        """Test deleting a document."""
        library_repository.create_library(sample_library)
        library_repository.add_document(sample_library.id, sample_document)

        deleted = library_repository.delete_document(sample_document.id)
        assert deleted is True

        with pytest.raises(DocumentNotFoundError):
            library_repository.get_document(sample_document.id)


@pytest.mark.unit
class TestChunkOperations:
    """Tests for document with multiple chunks."""

    def test_add_document_with_chunk(self, library_repository: LibraryRepository, sample_library: Library, sample_document: Document):
        """Test adding a document with chunks."""
        library_repository.create_library(sample_library)
        library_repository.add_document(sample_library.id, sample_document)

        stats = library_repository.get_library_statistics(sample_library.id)
        assert stats["num_chunks"] == 1  # sample_document has 1 chunk

    def test_dimension_mismatch_raises_error(self, library_repository: LibraryRepository, sample_library: Library, vector_dimension: int):
        """Test that adding document with wrong dimension chunks raises DimensionMismatchError."""
        library_repository.create_library(sample_library)

        # Create document with wrong dimension chunk
        wrong_vec = np.random.randn(vector_dimension + 10).astype(np.float32)
        wrong_vec = wrong_vec / np.linalg.norm(wrong_vec)

        doc_id = uuid4()
        chunk_metadata = ChunkMetadata(chunk_index=0, source_document_id=doc_id)
        wrong_chunk = Chunk(text="test", embedding=wrong_vec.tolist(), metadata=chunk_metadata)

        doc_metadata = DocumentMetadata(title="Wrong Doc")
        wrong_doc = Document(id=doc_id, chunks=[wrong_chunk], metadata=doc_metadata)

        with pytest.raises(DimensionMismatchError):
            library_repository.add_document(sample_library.id, wrong_doc)

    def test_delete_document_removes_chunks(self, library_repository: LibraryRepository, sample_library: Library, sample_document: Document):
        """Test deleting a document removes its chunks from the index."""
        library_repository.create_library(sample_library)
        library_repository.add_document(sample_library.id, sample_document)

        # Verify chunks exist
        stats = library_repository.get_library_statistics(sample_library.id)
        assert stats["num_chunks"] == 1

        # Delete document
        library_repository.delete_document(sample_document.id)

        # Verify chunks are gone
        stats = library_repository.get_library_statistics(sample_library.id)
        assert stats["num_chunks"] == 0


@pytest.mark.unit
class TestSearchOperations:
    """Tests for vector search."""

    def test_search_vectors(self, library_repository: LibraryRepository, sample_library: Library, sample_vectors: list):
        """Test searching for vectors."""
        library_repository.create_library(sample_library)

        # Create document with multiple chunks
        doc_id = uuid4()
        chunks = []
        for i, vector in enumerate(sample_vectors[:5]):
            metadata = ChunkMetadata(chunk_index=i, source_document_id=doc_id)
            chunk = Chunk(text=f"Chunk {i}", embedding=vector.tolist(), metadata=metadata)
            chunks.append(chunk)

        doc_metadata = DocumentMetadata(title="Multi-chunk Doc")
        doc = Document(id=doc_id, chunks=chunks, metadata=doc_metadata)
        library_repository.add_document(sample_library.id, doc)

        # Search
        results = library_repository.search(sample_library.id, sample_vectors[0].tolist(), k=3)

        assert len(results) <= 3
        # First result should be the exact match
        assert results[0][1] < 0.01  # Very small distance

    def test_search_empty_library(self, library_repository: LibraryRepository, sample_library: Library, sample_vectors: list):
        """Test searching empty library."""
        library_repository.create_library(sample_library)

        results = library_repository.search(sample_library.id, sample_vectors[0].tolist(), k=5)
        assert len(results) == 0


@pytest.mark.unit
@pytest.mark.thread_safety
class TestRepositoryThreadSafety:
    """Thread safety tests for repository."""

    def test_concurrent_add_documents(self, library_repository: LibraryRepository, sample_library: Library, vector_dimension: int):
        """Test concurrent document additions."""
        import threading

        library_repository.create_library(sample_library)

        def add_doc(doc_id: int):
            # Create a document with a chunk
            doc_uuid = uuid4()
            vec = np.random.randn(vector_dimension).astype(np.float32)
            vec = vec / np.linalg.norm(vec)

            chunk_metadata = ChunkMetadata(chunk_index=0, source_document_id=doc_uuid)
            chunk = Chunk(text=f"Document {doc_id}", embedding=vec.tolist(), metadata=chunk_metadata)

            doc_metadata = DocumentMetadata(title=f"Doc {doc_id}")
            doc = Document(id=doc_uuid, chunks=[chunk], metadata=doc_metadata)
            library_repository.add_document(sample_library.id, doc)

        threads = []
        for i in range(10):
            t = threading.Thread(target=add_doc, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        stats = library_repository.get_library_statistics(sample_library.id)
        assert stats["num_documents"] == 10

    def test_concurrent_reads(self, library_repository: LibraryRepository, sample_library: Library):
        """Test concurrent reads don't interfere."""
        import threading

        library_repository.create_library(sample_library)

        def read_library():
            library = library_repository.get_library(sample_library.id)
            assert library.id == sample_library.id

        threads = []
        for _ in range(20):
            t = threading.Thread(target=read_library)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # All reads should succeed
