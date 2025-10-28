"""
Thread-safe repository for managing corpora (collections), documents, and chunks.

This module provides the CorpusRepository (formerly LibraryRepository) which coordinates
between the domain models, vector store, indexes, and embedding contract to provide
a consistent, thread-safe interface for all vector database operations.

Rationale for terminology change: "Corpus" better represents a collection of documents
in the context of information retrieval and NLP, which is the core domain of this system.
The term "Library" was too generic and created confusion with Python's standard library concept.
"""

import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from uuid import UUID

import asyncio
import logging

from app.models.base import Chunk, Document, Corpus, CorpusMetadata
from core.embedding_contract import LibraryEmbeddingContract
from core.vector_store import VectorStore
from infrastructure.concurrency.rw_lock import ReaderWriterLock
from infrastructure.indexes.base import VectorIndex
from infrastructure.indexes.brute_force import BruteForceIndex
from infrastructure.indexes.hnsw import HNSWIndex
from infrastructure.indexes.ivf import IVFIndex
from infrastructure.indexes.kd_tree import KDTreeIndex
from infrastructure.indexes.lsh import LSHIndex
from infrastructure.persistence.snapshot import SnapshotManager
from infrastructure.persistence.wal import OperationType, WALEntry, WriteAheadLog

logger = logging.getLogger(__name__)


class CorpusNotFoundError(Exception):
    """Raised when a corpus is not found."""

    pass


# Backward compatibility alias
LibraryNotFoundError = CorpusNotFoundError


class DocumentNotFoundError(Exception):
    """Raised when a document is not found."""

    pass


class ChunkNotFoundError(Exception):
    """Raised when a chunk is not found."""

    pass


class DimensionMismatchError(Exception):
    """Raised when vector dimensions don't match the corpus's contract."""

    pass


class CorpusRepository:
    """
    Thread-safe repository for corpus (collection) management.

    Rationale for rename: "Corpus" is the standard term in information retrieval and NLP
    for a collection of documents. This rename improves clarity and aligns with domain
    terminology. Backward compatibility is maintained via the LibraryRepository alias.

    This repository:
    - Manages multiple corpora with different configurations
    - Enforces embedding dimension contracts per corpus
    - Provides thread-safe CRUD operations
    - Coordinates between domain models and infrastructure

    Thread-Safety: All public methods are thread-safe using reader-writer locks.
    Multiple concurrent reads are allowed, but writes are exclusive.
    """

    def __init__(self, data_dir: Path, event_bus=None):
        """
        Initialize the repository.

        Args:
            data_dir: Base directory for data storage (vectors, indexes, etc.)
            event_bus: Optional event bus for change data capture (CDC)
        """
        self._data_dir = data_dir
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._event_bus = event_bus  # Optional CDC event bus

        # Core data structures
        self._libraries: Dict[UUID, Corpus] = {}
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

    def _publish_event(self, event_type, library_id: UUID, data: dict):
        """
        Publish an event to the event bus (if configured).

        This is a fire-and-forget operation that doesn't block.

        Args:
            event_type: EventType enum value
            library_id: Library where event occurred
            data: Event-specific data payload
        """
        if self._event_bus is None:
            return

        try:
            # Import here to avoid circular dependency
            from app.events.bus import Event
            from datetime import datetime

            event = Event(
                type=event_type,
                library_id=library_id,
                data=data,
                timestamp=datetime.utcnow(),
            )

            # Publish asynchronously using thread-safe method
            # Get event loop from event bus (set during startup)
            event_loop = self._event_bus._loop if self._event_bus else None

            if event_loop and not event_loop.is_closed():
                # Use run_coroutine_threadsafe - designed for calling async from sync threads
                logger.info(f"Publishing event {event_type.value} to event bus")
                future = asyncio.run_coroutine_threadsafe(
                    self._event_bus.publish(event),
                    event_loop
                )
                # Don't wait for result - fire and forget
                logger.info(f"Event {event_type.value} queued successfully")
            else:
                logger.warning(f"Event {event_type.value} skipped (event bus not ready)")

        except Exception as e:
            # Never let event publishing break repository operations
            logger.error(f"Failed to publish event {event_type}: {e}")

    def create_library(self, library: Corpus) -> Corpus:
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

            # Publish event
            from app.events.bus import EventType
            self._publish_event(
                EventType.LIBRARY_CREATED,
                library.id,
                {
                    "library_id": str(library.id),
                    "name": library.name,
                    "index_type": library.metadata.index_type,
                    "embedding_dimension": library.metadata.embedding_dimension,
                },
            )

            return library

    def get_library(self, library_id: UUID) -> Corpus:
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

    def list_libraries(self) -> List[Corpus]:
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

            # Publish event
            from app.events.bus import EventType
            self._publish_event(
                EventType.DOCUMENT_ADDED,
                library_id,
                {
                    "document_id": str(document.id),
                    "title": document.metadata.title,
                    "num_chunks": len(document.chunks),
                },
            )

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

    def add_documents_batch(
        self, library_id: UUID, documents: List[Document]
    ) -> Tuple[List[Document], List[Tuple[int, str]]]:
        """
        Add multiple documents to a library in a batch.

        This is much more efficient than adding documents one at a time because:
        1. Single write lock acquisition
        2. Batched embedding validation
        3. Reduced WAL overhead
        4. Better cache locality

        Args:
            library_id: The library ID.
            documents: List of documents to add.

        Returns:
            Tuple of (successful_documents, failed_operations)
            where failed_operations is [(index, error_message), ...]

        Raises:
            LibraryNotFoundError: If the library doesn't exist.
        """
        with self._lock.write():
            if library_id not in self._libraries:
                raise LibraryNotFoundError(f"Library {library_id} not found")

            successful = []
            failed = []

            for idx, document in enumerate(documents):
                try:
                    self._add_document_internal(library_id, document)
                    self._libraries[library_id].documents.append(document)
                    successful.append(document)

                    # Log to WAL (batched)
                    self._wal.append_operation(
                        OperationType.ADD_DOCUMENT,
                        {
                            "library_id": str(library_id),
                            "document_id": str(document.id),
                            "num_chunks": len(document.chunks),
                        },
                    )
                except Exception as e:
                    failed.append((idx, str(e)))

            # Save snapshot periodically (every 100 documents in batch)
            if len(successful) > 0 and len(self._libraries[library_id].documents) % 100 < len(successful):
                self._save_snapshot()

            return successful, failed

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

    def delete_documents_batch(
        self, document_ids: List[UUID]
    ) -> Tuple[List[UUID], List[Tuple[UUID, str]]]:
        """
        Delete multiple documents in a batch.

        This is much more efficient than deleting documents one at a time because:
        1. Single write lock acquisition
        2. Batched index/vector store operations
        3. Reduced overhead

        Args:
            document_ids: List of document IDs to delete.

        Returns:
            Tuple of (successful_ids, failed_operations)
            where failed_operations is [(document_id, error_message), ...]
        """
        with self._lock.write():
            successful = []
            failed = []

            for document_id in document_ids:
                try:
                    if document_id not in self._doc_to_library:
                        failed.append((document_id, "Document not found"))
                        continue

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
                        failed.append((document_id, "Document not in library"))
                        continue

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

                    # Log to WAL
                    self._wal.append_operation(
                        OperationType.DELETE_DOCUMENT,
                        {
                            "library_id": str(library_id),
                            "document_id": str(document_id),
                        },
                    )

                    successful.append(document_id)

                except Exception as e:
                    failed.append((document_id, str(e)))

            return successful, failed

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

    def add_embeddings_to_library(
        self, library_id: UUID, chunks: List[Chunk], embeddings
    ) -> None:
        """
        Add embeddings for chunks to a library's vector store and index.

        This is used when re-embedding chunks after loading from disk.
        Assumes chunks already exist in documents, we're just adding their
        embeddings to the vector store and index.

        Args:
            library_id: The library ID.
            chunks: List of chunks to add embeddings for.
            embeddings: Numpy array of embeddings.

        Raises:
            LibraryNotFoundError: If the library doesn't exist.
            DimensionMismatchError: If embeddings have wrong dimension.
        """
        import numpy as np

        with self._lock.write():
            if library_id not in self._libraries:
                raise LibraryNotFoundError(f"Library {library_id} not found")

            vector_store = self._vector_stores[library_id]
            index = self._indexes[library_id]
            contract = self._contracts[library_id]

            # Add each embedding to vector store and index
            for chunk, embedding in zip(chunks, embeddings):
                # Validate and normalize embedding through contract
                try:
                    normalized_embedding = contract.validate_vector(embedding)
                except ValueError as e:
                    raise DimensionMismatchError(
                        f"Chunk {chunk.id} embedding invalid: {e}"
                    ) from e

                # Add to vector store
                vector_index = vector_store.add_vector(chunk.id, normalized_embedding)

                # Add to index
                index.add_vector(chunk.id, vector_index)

                # Update chunk tracking
                # Find the document this chunk belongs to
                for doc in self._libraries[library_id].documents:
                    if any(c.id == chunk.id for c in doc.chunks):
                        self._chunk_to_doc[chunk.id] = doc.id
                        break

    def rebuild_index(
        self, library_id: UUID, new_index_type: Optional[str] = None, index_config: Optional[dict] = None
    ) -> Tuple[str, str, int]:
        """
        Rebuild a library's index, optionally switching to a new index type.

        This operation:
        1. Creates a new empty index (with new type if specified)
        2. Re-indexes all existing vectors
        3. Replaces the old index

        Args:
            library_id: The library ID.
            new_index_type: Optional new index type (brute_force, kd_tree, lsh, hnsw).
                           If None, keeps the current type.
            index_config: Optional index-specific configuration parameters.

        Returns:
            Tuple of (old_index_type, new_index_type, vectors_reindexed)

        Raises:
            LibraryNotFoundError: If the library doesn't exist.
            ValueError: If new_index_type is invalid.
        """
        with self._lock.write():
            if library_id not in self._libraries:
                raise LibraryNotFoundError(f"Library {library_id} not found")

            library = self._libraries[library_id]
            vector_store = self._vector_stores[library_id]
            old_index = self._indexes[library_id]
            old_index_type = library.metadata.index_type

            # Determine new index type
            target_index_type = new_index_type if new_index_type else old_index_type

            # Update metadata if switching types
            if new_index_type:
                library.metadata.index_type = new_index_type

            # Create new index with updated metadata
            new_index = self._create_index_with_config(
                library.metadata, vector_store, index_config
            )

            # Re-index all existing vectors
            vectors_reindexed = 0
            for document in library.documents:
                for chunk in document.chunks:
                    if chunk.id in self._chunk_to_doc:
                        # Get vector from vector store
                        try:
                            vector_index = vector_store.get_vector_index(chunk.id)
                            new_index.add_vector(chunk.id, vector_index)
                            vectors_reindexed += 1
                        except Exception:
                            # Vector might not be in store yet (during restore)
                            pass

            # Replace old index
            old_index.clear()
            self._indexes[library_id] = new_index

            # Log to WAL
            self._wal.append_operation(
                OperationType.UPDATE_LIBRARY,
                {
                    "library_id": str(library_id),
                    "operation": "rebuild_index",
                    "old_index_type": old_index_type,
                    "new_index_type": target_index_type,
                },
            )

            return old_index_type, target_index_type, vectors_reindexed

    def optimize_index(self, library_id: UUID) -> Tuple[int, int]:
        """
        Optimize a library's index by compacting and removing deleted entries.

        This operation improves search performance and reduces memory usage.

        Args:
            library_id: The library ID.

        Returns:
            Tuple of (vectors_compacted, memory_freed_bytes)

        Raises:
            LibraryNotFoundError: If the library doesn't exist.
        """
        with self._lock.write():
            if library_id not in self._libraries:
                raise LibraryNotFoundError(f"Library {library_id} not found")

            library = self._libraries[library_id]
            index = self._indexes[library_id]
            vector_store = self._vector_stores[library_id]

            # Get stats before optimization
            stats_before = index.get_statistics()
            vectors_before = stats_before.get("num_vectors", 0)

            # Perform optimization
            # For now, we rebuild the index which effectively compacts it
            # In the future, indexes could have their own optimize() method
            new_index = self._create_index(library.metadata, vector_store)

            # Re-index all vectors
            vectors_compacted = 0
            for document in library.documents:
                for chunk in document.chunks:
                    if chunk.id in self._chunk_to_doc:
                        try:
                            vector_index = vector_store.get_vector_index(chunk.id)
                            new_index.add_vector(chunk.id, vector_index)
                            vectors_compacted += 1
                        except Exception:
                            pass

            # Estimate memory freed (rough approximation)
            # Each deleted entry might use ~100 bytes in the index
            memory_freed = (vectors_before - vectors_compacted) * 100

            # Replace old index
            index.clear()
            self._indexes[library_id] = new_index

            return vectors_compacted, memory_freed

    def get_index_statistics(self, library_id: UUID) -> Dict:
        """
        Get detailed statistics about a library's index.

        Args:
            library_id: The library ID.

        Returns:
            Dictionary with index statistics.

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
                "index_type": library.metadata.index_type,
                "total_vectors": total_chunks,
                "index_stats": index.get_statistics(),
                "vector_store_stats": vector_store.get_statistics(),
            }

    def _create_index(
        self, metadata: CorpusMetadata, vector_store: VectorStore
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
        elif index_type == "ivf":
            return IVFIndex(vector_store, n_clusters=256, nprobe=8)
        else:
            raise ValueError(f"Unknown index type: {index_type}")

    def _create_index_with_config(
        self, metadata: CorpusMetadata, vector_store: VectorStore, config: Optional[dict] = None
    ) -> VectorIndex:
        """
        Create an index with custom configuration.

        Args:
            metadata: Library metadata specifying index type.
            vector_store: The vector store to use.
            config: Optional index-specific configuration.

        Returns:
            The created index.

        Raises:
            ValueError: If index type is unknown.
        """
        index_type = metadata.index_type

        if not config:
            # Use default configuration
            return self._create_index(metadata, vector_store)

        # Create index with custom config
        if index_type == "brute_force":
            return BruteForceIndex(vector_store)
        elif index_type == "kd_tree":
            rebuild_threshold = config.get("rebuild_threshold", 100)
            return KDTreeIndex(vector_store, rebuild_threshold=rebuild_threshold)
        elif index_type == "lsh":
            num_tables = config.get("num_tables", 10)
            hash_size = config.get("hash_size", 10)
            return LSHIndex(vector_store, num_tables=num_tables, hash_size=hash_size)
        elif index_type == "hnsw":
            M = config.get("M", 16)
            ef_construction = config.get("ef_construction", 200)
            ef_search = config.get("ef_search", 50)
            return HNSWIndex(
                vector_store, M=M, ef_construction=ef_construction, ef_search=ef_search
            )
        elif index_type == "ivf":
            n_clusters = config.get("n_clusters", 256)
            nprobe = config.get("nprobe", 8)
            use_pq = config.get("use_pq", False)
            pq_subvectors = config.get("pq_subvectors", 8)
            return IVFIndex(
                vector_store,
                n_clusters=n_clusters,
                nprobe=nprobe,
                use_pq=use_pq,
                pq_subvectors=pq_subvectors,
            )
        else:
            raise ValueError(f"Unknown index type: {index_type}")

    def _load_from_disk(self) -> None:
        """Load state from disk (snapshot + WAL replay)."""
        import logging
        logger = logging.getLogger(__name__)

        try:
            # Try to load latest snapshot
            snapshot = self._snapshot_manager.load_latest_snapshot()
            if snapshot:
                logger.info(f"Loading snapshot from {snapshot.timestamp}")
                snapshot_data = snapshot.data

                # Restore libraries and documents
                libraries_data = snapshot_data.get("libraries", {})
                logger.info(f"Restoring {len(libraries_data)} libraries from snapshot...")

                for lib_id_str, lib_data in libraries_data.items():
                    try:
                        # Reconstruct library object
                        lib_id = UUID(lib_id_str)
                        library = Corpus(
                            id=lib_id,
                            name=lib_data["name"],
                            documents=[],  # Will be populated below
                            metadata=CorpusMetadata(**lib_data["metadata"])
                        )

                        # Restore documents and chunks
                        for doc_data in lib_data.get("documents", []):
                            chunks = []
                            for chunk_data in doc_data.get("chunks", []):
                                from app.models.base import ChunkMetadata
                                chunk = Chunk(
                                    id=UUID(chunk_data["id"]),
                                    text=chunk_data["text"],
                                    metadata=ChunkMetadata(**chunk_data["metadata"])
                                )
                                chunks.append(chunk)

                            from app.models.base import DocumentMetadata
                            document = Document(
                                id=UUID(doc_data["id"]),
                                chunks=chunks,
                                metadata=DocumentMetadata(**doc_data["metadata"])
                            )
                            library.documents.append(document)

                            # Update lookup maps
                            self._documents[document.id] = document
                            self._doc_to_library[document.id] = lib_id
                            for chunk in chunks:
                                self._chunk_to_doc[chunk.id] = document.id

                        # Store library
                        self._libraries[lib_id] = library

                        # Recreate vector store and index
                        logger.info(f"Rebuilding vector store and index for library {library.name}...")
                        vector_dir = self._data_dir / "vectors" / str(lib_id)
                        use_mmap = len(library.documents) > 10000

                        vector_store = VectorStore(
                            dimension=library.metadata.embedding_dimension,
                            initial_capacity=max(1000, len(library.documents) * 10),
                            use_mmap=use_mmap,
                            mmap_path=vector_dir / "vectors.mmap" if use_mmap else None,
                        )

                        index = self._create_index(library.metadata, vector_store)
                        contract = LibraryEmbeddingContract(library.metadata.embedding_dimension)

                        self._vector_stores[lib_id] = vector_store
                        self._indexes[lib_id] = index
                        self._contracts[lib_id] = contract

                        # Note: Embeddings need to be regenerated as they're not in snapshot
                        # This is intentional - embeddings can be large and should be regenerated
                        # from the text chunks using the embedding service

                    except Exception as e:
                        logger.error(f"Failed to restore library {lib_id_str}: {e}")
                        continue

                logger.info(f"Successfully restored {len(self._libraries)} libraries")

            else:
                logger.info("No snapshot found, starting with empty state")

            # TODO: Replay WAL entries after snapshot
            # This would apply any operations that happened after the last snapshot
            # For now, we rely on snapshots being created frequently enough

        except Exception as e:
            # If loading fails, start with empty state
            logger.warning(f"Failed to load from disk: {e}. Starting fresh.")
            self._libraries = {}
            self._vector_stores = {}
            self._indexes = {}
            self._contracts = {}
            self._documents = {}
            self._doc_to_library = {}
            self._chunk_to_doc = {}

    def _save_snapshot(self) -> None:
        """Save current state to snapshot."""
        import logging
        import numpy as np
        logger = logging.getLogger(__name__)

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

            # Note: Embeddings are NOT persisted in snapshots
            # They are regenerated from text chunks on startup
            # This is intentional for several reasons:
            # 1. Embeddings are large (384-1536 dims * 4 bytes * millions of chunks)
            # 2. Embedding models may change/improve over time
            # 3. Re-embedding on load ensures consistency with current model
            # 4. Snapshot files stay small and portable
            #
            # Future enhancement: Add optional embedding cache for faster startup

            from datetime import datetime
            self._snapshot_manager.create_snapshot(state)
            logger.info("Snapshot saved successfully")

        except Exception as e:
            logger.error(f"Failed to save snapshot: {e}")

    def save_state(self) -> None:
        """
        Save current state to disk (snapshot + flush WAL).

        This method is called on shutdown to persist all data.
        """
        with self._lock.read():
            import logging
            logger = logging.getLogger(__name__)

            try:
                # Save snapshot with all current state
                logger.info("Creating final snapshot...")
                self._save_snapshot()

                # Close WAL (flushes any pending writes)
                logger.info("Flushing WAL...")
                self._wal.close()

                # Save vector stores
                logger.info(f"Saving {len(self._vector_stores)} vector stores...")
                for lib_id, vector_store in self._vector_stores.items():
                    try:
                        vector_store.flush()
                    except Exception as e:
                        logger.warning(f"Failed to flush vector store for library {lib_id}: {e}")

                logger.info("State saved successfully")

            except Exception as e:
                logger.error(f"Error saving state: {e}")
                raise

    def __repr__(self) -> str:
        """String representation."""
        with self._lock.read():
            return (
                f"LibraryRepository(libraries={len(self._libraries)}, "
                f"data_dir={self._data_dir})"
            )


# Backward compatibility alias
LibraryRepository = CorpusRepository
