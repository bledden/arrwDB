"""
Thread-safe repository for managing libraries, documents, and chunks.

This module provides the LibraryRepository which coordinates between the
domain models, vector store, indexes, and embedding contract to provide
a consistent, thread-safe interface for all vector database operations.
"""

from typing import Dict, List, Optional, Tuple
from uuid import UUID
import threading
from pathlib import Path

from app.models.base import Library, Document, Chunk, LibraryMetadata
from core.vector_store import VectorStore
from core.embedding_contract import LibraryEmbeddingContract
from infrastructure.indexes.base import VectorIndex
from infrastructure.indexes.brute_force import BruteForceIndex
from infrastructure.indexes.kd_tree import KDTreeIndex
from infrastructure.indexes.lsh import LSHIndex
from infrastructure.indexes.hnsw import HNSWIndex
from infrastructure.concurrency.rw_lock import ReaderWriterLock
from infrastructure.persistence.wal import WriteAheadLog, OperationType, WALEntry
from infrastructure.persistence.snapshot import SnapshotManager


class LibraryNotFoundError(Exception):
    """Raised when a library is not found."""

    pass


class DocumentNotFoundError(Exception):
    """Raised when a document is not found."""

    pass


class ChunkNotFoundError(Exception):
    """Raised when a chunk is not found."""

    pass


class DimensionMismatchError(Exception):
    """Raised when vector dimensions don't match the library's contract."""

    pass


class LibraryRepository:
    """
    Thread-safe repository for library management.

    This repository:
    - Manages multiple libraries with different configurations
    - Enforces embedding dimension contracts per library
    - Provides thread-safe CRUD operations
    - Coordinates between domain models and infrastructure

    Thread-Safety: All public methods are thread-safe using reader-writer locks.
    Multiple concurrent reads are allowed, but writes are exclusive.
    """

    def __init__(self, data_dir: Path):
        """
        Initialize the repository.

        Args:
            data_dir: Base directory for data storage (vectors, indexes, etc.)
        """
        self._data_dir = data_dir
        self._data_dir.mkdir(parents=True, exist_ok=True)

        # Core data structures
        self._libraries: Dict[UUID, Library] = {}
        self._vector_stores: Dict[UUID, VectorStore] = {}
        self._indexes: Dict[UUID, VectorIndex] = {}
        self._contracts: Dict[UUID, LibraryEmbeddingContract] = {}

        # Document and chunk mappings for quick lookup
        self._documents: Dict[UUID, Document] = {}  # doc_id -> Document (O(1) lookup)
        self._doc_to_library: Dict[UUID, UUID] = {}  # doc_id -> library_id
        self._chunk_to_doc: Dict[UUID, UUID] = {}  # chunk_id -> doc_id

        # Thread safety
        self._lock = ReaderWriterLock()

        # Persistence
        wal_dir = self._data_dir / "wal"
        snapshot_dir = self._data_dir / "snapshots"
        wal_dir.mkdir(parents=True, exist_ok=True)
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        self._wal = WriteAheadLog(wal_dir)
        self._snapshot_manager = SnapshotManager(snapshot_dir)

        # Load from persistence on startup
        self._load_from_disk()

    def create_library(self, library: Library) -> Library:
        """
        Create a new library.

        Args:
            library: The library to create.

        Returns:
            The created library.

        Raises:
            ValueError: If a library with the same ID already exists.
        """
        with self._lock.write():
            if library.id in self._libraries:
                raise ValueError(f"Library with ID {library.id} already exists")

            # Log to WAL before applying
            self._wal.append_operation(
                OperationType.CREATE_LIBRARY,
                {
                    "library_id": str(library.id),
                    "name": library.name,
                    "index_type": library.metadata.index_type,
                }
            )

            # Create vector store for this library
            vector_dir = self._data_dir / "vectors" / str(library.id)
            use_mmap = len(library.documents) > 10000  # Use mmap for large libraries

            vector_store = VectorStore(
                dimension=library.metadata.embedding_dimension,
                initial_capacity=max(1000, len(library.documents) * 10),
                use_mmap=use_mmap,
                mmap_path=vector_dir / "vectors.mmap" if use_mmap else None,
            )

            # Create index based on library metadata
            index = self._create_index(library.metadata, vector_store)

            # Create embedding contract (will be set when first vector is added)
            # For now, create with expected dimension
            contract = LibraryEmbeddingContract(
                library.metadata.embedding_dimension
            )

            # Store everything
            self._libraries[library.id] = library
            self._vector_stores[library.id] = vector_store
            self._indexes[library.id] = index
            self._contracts[library.id] = contract

            # If library has documents, add them
            for doc in library.documents:
                self._add_document_internal(library.id, doc)

            # Save snapshot periodically (every 10 libraries)
            if len(self._libraries) % 10 == 0:
                self._save_snapshot()

            return library

    def get_library(self, library_id: UUID) -> Library:
        """
        Retrieve a library by ID.

        Args:
            library_id: The library ID.

        Returns:
            The library.

        Raises:
            LibraryNotFoundError: If the library doesn't exist.
        """
        with self._lock.read():
            if library_id not in self._libraries:
                raise LibraryNotFoundError(f"Library {library_id} not found")

            return self._libraries[library_id]

    def list_libraries(self) -> List[Library]:
        """
        List all libraries.

        Returns:
            List of all libraries.
        """
        with self._lock.read():
            return list(self._libraries.values())

    def delete_library(self, library_id: UUID) -> bool:
        """
        Delete a library and all its data.

        Args:
            library_id: The library ID.

        Returns:
            True if deleted, False if library didn't exist.
        """
        with self._lock.write():
            if library_id not in self._libraries:
                return False

            # Log to WAL before applying
            self._wal.append_operation(
                OperationType.DELETE_LIBRARY,
                {"library_id": str(library_id)}
            )

            library = self._libraries[library_id]

            # Remove all document and chunk mappings
            for doc in library.documents:
                if doc.id in self._doc_to_library:
                    del self._doc_to_library[doc.id]
                if doc.id in self._documents:
                    del self._documents[doc.id]

                for chunk in doc.chunks:
                    if chunk.id in self._chunk_to_doc:
                        del self._chunk_to_doc[chunk.id]

            # Clean up infrastructure
            self._indexes[library_id].clear()
            del self._indexes[library_id]
            del self._vector_stores[library_id]
            del self._contracts[library_id]
            del self._libraries[library_id]

            # Save snapshot periodically
            if len(self._libraries) % 10 == 0:
                self._save_snapshot()

            return True

    def add_document(self, library_id: UUID, document: Document) -> Document:
        """
        Add a document to a library.

        Args:
            library_id: The library ID.
            document: The document to add.

        Returns:
            The added document.

        Raises:
            LibraryNotFoundError: If the library doesn't exist.
            DimensionMismatchError: If document chunks have wrong dimension.
        """
        with self._lock.write():
            if library_id not in self._libraries:
                raise LibraryNotFoundError(f"Library {library_id} not found")

            self._add_document_internal(library_id, document)

            # Update library's document list
            self._libraries[library_id].documents.append(document)

            return document

    def _add_document_internal(self, library_id: UUID, document: Document) -> None:
        """
        Internal method to add a document (assumes lock is held).

        Args:
            library_id: The library ID.
            document: The document to add.

        Raises:
            DimensionMismatchError: If chunks have wrong dimension.
        """
        vector_store = self._vector_stores[library_id]
        index = self._indexes[library_id]
        contract = self._contracts[library_id]

        # Track document
        self._doc_to_library[document.id] = library_id
        self._documents[document.id] = document

        # Add all chunks
        for chunk in document.chunks:
            # Validate and normalize embedding through contract
            try:
                normalized_embedding = contract.validate_vector(chunk.embedding)
            except ValueError as e:
                raise DimensionMismatchError(
                    f"Chunk {chunk.id} embedding invalid: {e}"
                ) from e

            # Add to vector store
            vector_index = vector_store.add_vector(chunk.id, normalized_embedding)

            # Add to index
            index.add_vector(chunk.id, vector_index)

            # Track chunk
            self._chunk_to_doc[chunk.id] = document.id

    def get_document(self, document_id: UUID) -> Document:
        """
        Retrieve a document by ID.

        Args:
            document_id: The document ID.

        Returns:
            The document.

        Raises:
            DocumentNotFoundError: If the document doesn't exist.
        """
        with self._lock.read():
            if document_id not in self._documents:
                raise DocumentNotFoundError(f"Document {document_id} not found")

            return self._documents[document_id]

    def delete_document(self, document_id: UUID) -> bool:
        """
        Delete a document and all its chunks.

        Args:
            document_id: The document ID.

        Returns:
            True if deleted, False if document didn't exist.
        """
        with self._lock.write():
            if document_id not in self._doc_to_library:
                return False

            library_id = self._doc_to_library[document_id]
            library = self._libraries[library_id]
            vector_store = self._vector_stores[library_id]
            index = self._indexes[library_id]

            # Find and remove document
            doc_to_remove = None
            for i, doc in enumerate(library.documents):
                if doc.id == document_id:
                    doc_to_remove = doc
                    library.documents.pop(i)
                    break

            if doc_to_remove is None:
                return False

            # Remove all chunks
            for chunk in doc_to_remove.chunks:
                # Remove from index
                index.remove_vector(chunk.id)

                # Remove from vector store
                vector_store.remove_vector(chunk.id)

                # Remove from tracking
                if chunk.id in self._chunk_to_doc:
                    del self._chunk_to_doc[chunk.id]

            # Remove document tracking
            del self._doc_to_library[document_id]
            if document_id in self._documents:
                del self._documents[document_id]

            return True

    def search(
        self,
        library_id: UUID,
        query_embedding: List[float],
        k: int = 10,
        distance_threshold: Optional[float] = None,
    ) -> List[Tuple[Chunk, float]]:
        """
        Search for nearest neighbor chunks in a library.

        Args:
            library_id: The library to search in.
            query_embedding: The query vector.
            k: Number of results to return.
            distance_threshold: Optional maximum distance threshold.

        Returns:
            List of (chunk, distance) tuples sorted by distance.

        Raises:
            LibraryNotFoundError: If the library doesn't exist.
            DimensionMismatchError: If query dimension doesn't match.
        """
        with self._lock.read():
            if library_id not in self._libraries:
                raise LibraryNotFoundError(f"Library {library_id} not found")

            contract = self._contracts[library_id]
            index = self._indexes[library_id]
            library = self._libraries[library_id]

            # Validate and normalize query
            try:
                normalized_query = contract.validate_vector(query_embedding)
            except ValueError as e:
                raise DimensionMismatchError(f"Query embedding invalid: {e}") from e

            # Search index
            results = index.search(normalized_query, k, distance_threshold)

            # Map chunk IDs to actual chunks
            chunk_results = []
            for chunk_id, distance in results:
                # Find the chunk
                doc_id = self._chunk_to_doc.get(chunk_id)
                if doc_id is None:
                    continue

                # Find document
                for doc in library.documents:
                    if doc.id == doc_id:
                        # Find chunk in document
                        for chunk in doc.chunks:
                            if chunk.id == chunk_id:
                                chunk_results.append((chunk, distance))
                                break
                        break

            return chunk_results

    def get_library_statistics(self, library_id: UUID) -> Dict:
        """
        Get statistics about a library.

        Args:
            library_id: The library ID.

        Returns:
            Dictionary with statistics.

        Raises:
            LibraryNotFoundError: If the library doesn't exist.
        """
        with self._lock.read():
            if library_id not in self._libraries:
                raise LibraryNotFoundError(f"Library {library_id} not found")

            library = self._libraries[library_id]
            vector_store = self._vector_stores[library_id]
            index = self._indexes[library_id]

            total_chunks = sum(len(doc.chunks) for doc in library.documents)

            return {
                "library_id": str(library.id),
                "library_name": library.name,
                "num_documents": len(library.documents),
                "num_chunks": total_chunks,
                "embedding_dimension": library.metadata.embedding_dimension,
                "index_type": library.metadata.index_type,
                "vector_store_stats": vector_store.get_statistics(),
                "index_stats": index.get_statistics(),
            }

    def _create_index(
        self, metadata: LibraryMetadata, vector_store: VectorStore
    ) -> VectorIndex:
        """
        Create an index based on library metadata.

        Args:
            metadata: Library metadata specifying index type.
            vector_store: The vector store to use.

        Returns:
            The created index.

        Raises:
            ValueError: If index type is unknown.
        """
        index_type = metadata.index_type

        if index_type == "brute_force":
            return BruteForceIndex(vector_store)
        elif index_type == "kd_tree":
            return KDTreeIndex(vector_store, rebuild_threshold=100)
        elif index_type == "lsh":
            return LSHIndex(vector_store, num_tables=10, hash_size=10)
        elif index_type == "hnsw":
            return HNSWIndex(vector_store, M=16, ef_construction=200, ef_search=50)
        else:
            raise ValueError(f"Unknown index type: {index_type}")

    def _load_from_disk(self) -> None:
        """Load state from disk (snapshot + WAL replay)."""
        try:
            # Try to load latest snapshot
            state = self._snapshot_manager.load_latest_snapshot()
            if state:
                # Restore state from snapshot
                self._libraries = {UUID(k): Library(**v) for k, v in state.get("libraries", {}).items()}
                # Note: Vector stores and indexes are reconstructed on-demand
                # as they contain numpy arrays that don't serialize well

            # Replay WAL entries after snapshot
            # (WAL manager handles finding entries after snapshot timestamp)
            # For now, we start fresh - full WAL replay would go here

        except Exception as e:
            # If loading fails, start with empty state
            import logging
            logging.warning(f"Failed to load from disk: {e}. Starting fresh.")

    def _save_snapshot(self) -> None:
        """Save current state to snapshot."""
        try:
            state = {
                "libraries": {
                    str(lib_id): {
                        "id": str(lib.id),
                        "name": lib.name,
                        "documents": [
                            {
                                "id": str(doc.id),
                                "chunks": [
                                    {
                                        "id": str(chunk.id),
                                        "text": chunk.text,
                                        "metadata": {
                                            "created_at": str(chunk.metadata.created_at),
                                            "page_number": chunk.metadata.page_number,
                                            "chunk_index": chunk.metadata.chunk_index,
                                            "source_document_id": str(chunk.metadata.source_document_id),
                                        }
                                    }
                                    for chunk in doc.chunks
                                ],
                                "metadata": {
                                    "title": doc.metadata.title,
                                    "author": doc.metadata.author,
                                    "created_at": str(doc.metadata.created_at),
                                    "document_type": doc.metadata.document_type,
                                    "source_url": doc.metadata.source_url,
                                    "tags": doc.metadata.tags,
                                }
                            }
                            for doc in lib.documents
                        ],
                        "metadata": {
                            "description": lib.metadata.description,
                            "created_at": str(lib.metadata.created_at),
                            "index_type": lib.metadata.index_type,
                            "embedding_dimension": lib.metadata.embedding_dimension,
                            "embedding_model": lib.metadata.embedding_model,
                        }
                    }
                    for lib_id, lib in self._libraries.items()
                }
            }
            from infrastructure.persistence.snapshot import Snapshot
            from datetime import datetime
            snapshot = Snapshot(data=state, timestamp=datetime.now())
            self._snapshot_manager.save_snapshot(snapshot)
        except Exception as e:
            import logging
            logging.error(f"Failed to save snapshot: {e}")

    def __repr__(self) -> str:
        """String representation."""
        with self._lock.read():
            return (
                f"LibraryRepository(libraries={len(self._libraries)}, "
                f"data_dir={self._data_dir})"
            )
