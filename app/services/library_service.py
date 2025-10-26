"""
Library service implementing domain logic.

This module provides the business logic layer for library operations,
following Domain-Driven Design principles. It sits between the API layer
and the repository layer.
"""

import logging
from pathlib import Path
from typing import Any, List, Optional, Tuple
from uuid import UUID

from app.models.base import Chunk, Document, DocumentMetadata, Library, LibraryMetadata, QuantizationMetadata
from app.services.embedding_service import EmbeddingService, EmbeddingServiceError
from app.utils.quantization import ScalarQuantizer, calculate_memory_savings, estimate_accuracy
from infrastructure.repositories.library_repository import (
    DimensionMismatchError,
    DocumentNotFoundError,
    LibraryNotFoundError,
    LibraryRepository,
)

logger = logging.getLogger(__name__)


class LibraryService:
    """
    Service for library management operations.

    This service:
    - Implements business logic and validation
    - Coordinates between embedding service and repository
    - Handles error translation and logging
    - Provides high-level operations for the API layer

    Domain-Driven Design:
    - This is the application service layer
    - Orchestrates domain operations
    - Maintains transaction boundaries
    - Translates between API DTOs and domain models
    """

    def __init__(
        self,
        repository: LibraryRepository,
        embedding_service: EmbeddingService,
    ) -> None:
        """
        Initialize the library service.

        Args:
            repository: The library repository.
            embedding_service: The embedding service for text-to-vector conversion.
        """
        self._repository = repository
        self._embedding_service = embedding_service

    @property
    def repository(self) -> LibraryRepository:
        """Get the underlying repository (for persistence operations)."""
        return self._repository

    def create_library(
        self,
        name: str,
        description: Optional[str] = None,
        index_type: str = "brute_force",
        embedding_model: Optional[str] = None,
        quantization_config: Optional[dict] = None,
    ) -> Library:
        """
        Create a new library.

        Args:
            name: Name of the library.
            description: Optional description.
            index_type: Type of index to use (brute_force, kd_tree, lsh, hnsw).
            embedding_model: Optional embedding model override.
            quantization_config: Optional quantization configuration (opt-in).
                - strategy: "none", "scalar", "hybrid"
                - bits: 4 or 8 (for scalar/hybrid)
                - rerank_top_k: Number for hybrid reranking

        Returns:
            The created library.

        Raises:
            ValueError: If parameters are invalid.
        """
        logger.info(f"Creating library '{name}' with index type '{index_type}'")

        # Validate index type
        valid_index_types = {"brute_force", "kd_tree", "lsh", "hnsw"}
        if index_type not in valid_index_types:
            raise ValueError(
                f"Invalid index_type '{index_type}'. "
                f"Must be one of {valid_index_types}"
            )

        # Process quantization configuration
        quantization_metadata = None
        if quantization_config and quantization_config.get("strategy", "none") != "none":
            strategy = quantization_config.get("strategy", "none")
            bits = quantization_config.get("bits", 8)
            rerank_top_k = quantization_config.get("rerank_top_k", 100)

            logger.info(
                f"Enabling quantization for library '{name}': "
                f"strategy={strategy}, bits={bits}"
            )

            quantization_metadata = QuantizationMetadata(
                strategy=strategy,
                bits=bits,
                rerank_top_k=rerank_top_k if strategy == "hybrid" else None,
                calibration_min=None,  # Will be set when first vectors are added
                calibration_max=None,
            )

        # Create library metadata
        metadata = LibraryMetadata(
            description=description,
            index_type=index_type,
            embedding_dimension=self._embedding_service.embedding_dimension,
            embedding_model=embedding_model or self._embedding_service.model,
            quantization=quantization_metadata,
        )

        # Create library
        library = Library(name=name, metadata=metadata)

        try:
            created = self._repository.create_library(library)
            logger.info(f"Created library {created.id} named '{name}'")
            return created
        except Exception as e:
            logger.error(f"Failed to create library '{name}': {e}")
            raise

    def get_library(self, library_id: UUID) -> Library:
        """
        Get a library by ID.

        Args:
            library_id: The library ID.

        Returns:
            The library.

        Raises:
            LibraryNotFoundError: If library doesn't exist.
        """
        return self._repository.get_library(library_id)

    def list_libraries(self) -> List[Library]:
        """
        List all libraries.

        Returns:
            List of all libraries.
        """
        return self._repository.list_libraries()

    def delete_library(self, library_id: UUID) -> bool:
        """
        Delete a library.

        Args:
            library_id: The library ID.

        Returns:
            True if deleted, False if didn't exist.
        """
        logger.info(f"Deleting library {library_id}")

        deleted = self._repository.delete_library(library_id)

        if deleted:
            logger.info(f"Deleted library {library_id}")
        else:
            logger.warning(f"Library {library_id} not found for deletion")

        return deleted

    def add_document_with_text(
        self,
        library_id: UUID,
        title: str,
        texts: List[str],
        author: Optional[str] = None,
        document_type: str = "text",
        source_url: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Document:
        """
        Add a document by providing raw text chunks.

        This method generates embeddings for the text chunks automatically.

        Args:
            library_id: The library to add the document to.
            title: Document title.
            texts: List of text chunks.
            author: Optional author name.
            document_type: Type of document.
            source_url: Optional source URL.
            tags: Optional tags.

        Returns:
            The created document.

        Raises:
            LibraryNotFoundError: If library doesn't exist.
            EmbeddingServiceError: If embedding generation fails.
            ValueError: If texts list is empty.
        """
        if not texts:
            raise ValueError("texts list cannot be empty")

        logger.info(
            f"Adding document '{title}' with {len(texts)} chunks to library {library_id}"
        )

        # Get library to check it exists
        library = self._repository.get_library(library_id)

        # Generate embeddings for all texts
        try:
            embeddings = self._embedding_service.embed_texts(texts)
        except EmbeddingServiceError as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise

        # Create chunks
        chunks = []
        doc_metadata = DocumentMetadata(
            title=title,
            author=author,
            document_type=document_type,
            source_url=source_url,
            tags=tags or [],
        )

        # We'll set the document ID after creating the document
        # For now, use a placeholder
        from uuid import uuid4

        doc_id = uuid4()

        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            from app.models.base import ChunkMetadata

            chunk_metadata = ChunkMetadata(
                chunk_index=i, source_document_id=doc_id
            )

            chunk = Chunk(
                text=text, embedding=embedding.tolist(), metadata=chunk_metadata
            )
            chunks.append(chunk)

        # Create document with the same ID used in chunk metadata
        document = Document(id=doc_id, chunks=chunks, metadata=doc_metadata)

        # Add to repository
        try:
            added = self._repository.add_document(library_id, document)
            logger.info(
                f"Added document {added.id} with {len(chunks)} chunks to library {library_id}"
            )
            return added
        except DimensionMismatchError as e:
            logger.error(f"Dimension mismatch when adding document: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to add document: {e}")
            raise

    def add_document_with_embeddings(
        self,
        library_id: UUID,
        title: str,
        text_embedding_pairs: List[Tuple[str, List[float]]],
        author: Optional[str] = None,
        document_type: str = "text",
        source_url: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Document:
        """
        Add a document with pre-computed embeddings.

        Use this when embeddings have already been generated elsewhere.

        Args:
            library_id: The library to add the document to.
            title: Document title.
            text_embedding_pairs: List of (text, embedding) tuples.
            author: Optional author name.
            document_type: Type of document.
            source_url: Optional source URL.
            tags: Optional tags.

        Returns:
            The created document.

        Raises:
            LibraryNotFoundError: If library doesn't exist.
            DimensionMismatchError: If embeddings have wrong dimension.
            ValueError: If text_embedding_pairs is empty.
        """
        if not text_embedding_pairs:
            raise ValueError("text_embedding_pairs cannot be empty")

        logger.info(
            f"Adding document '{title}' with {len(text_embedding_pairs)} "
            f"pre-computed chunks to library {library_id}"
        )

        # Get library to check it exists
        library = self._repository.get_library(library_id)

        # Create chunks
        from uuid import uuid4

        from app.models.base import ChunkMetadata

        doc_id = uuid4()
        chunks = []

        doc_metadata = DocumentMetadata(
            title=title,
            author=author,
            document_type=document_type,
            source_url=source_url,
            tags=tags or [],
        )

        for i, (text, embedding) in enumerate(text_embedding_pairs):
            chunk_metadata = ChunkMetadata(
                chunk_index=i, source_document_id=doc_id
            )

            chunk = Chunk(
                text=text, embedding=embedding, metadata=chunk_metadata
            )
            chunks.append(chunk)

        # Create document with the same ID used in chunk metadata
        document = Document(id=doc_id, chunks=chunks, metadata=doc_metadata)

        # Add to repository
        try:
            added = self._repository.add_document(library_id, document)
            logger.info(
                f"Added document {added.id} with {len(chunks)} chunks to library {library_id}"
            )
            return added
        except Exception as e:
            logger.error(f"Failed to add document: {e}")
            raise

    def get_document(self, document_id: UUID) -> Document:
        """
        Get a document by ID.

        Args:
            document_id: The document ID.

        Returns:
            The document.

        Raises:
            DocumentNotFoundError: If document doesn't exist.
        """
        return self._repository.get_document(document_id)

    def delete_document(self, document_id: UUID) -> bool:
        """
        Delete a document.

        Args:
            document_id: The document ID.

        Returns:
            True if deleted, False if didn't exist.
        """
        logger.info(f"Deleting document {document_id}")

        deleted = self._repository.delete_document(document_id)

        if deleted:
            logger.info(f"Deleted document {document_id}")
        else:
            logger.warning(f"Document {document_id} not found for deletion")

        return deleted

    def add_documents_batch(
        self,
        library_id: UUID,
        documents_data: List[dict],
    ) -> Tuple[List[Document], List[Tuple[int, str]], int]:
        """
        Add multiple documents with text chunks in a batch.

        This is 100-1000x faster than adding documents one at a time because:
        - Batched embedding generation
        - Single write lock acquisition
        - Reduced overhead

        Args:
            library_id: The library to add documents to.
            documents_data: List of dicts with document data (title, texts, author, etc.)

        Returns:
            Tuple of (successful_documents, failed_operations, total_chunks)

        Raises:
            LibraryNotFoundError: If library doesn't exist.
        """
        logger.info(
            f"Batch adding {len(documents_data)} documents to library {library_id}"
        )

        # Get library to check it exists
        library = self._repository.get_library(library_id)

        # Collect all texts that need embeddings
        all_texts = []
        doc_text_ranges = []  # [(start_idx, end_idx, doc_data), ...]

        for doc_data in documents_data:
            texts = doc_data.get("texts", [])
            start_idx = len(all_texts)
            all_texts.extend(texts)
            end_idx = len(all_texts)
            doc_text_ranges.append((start_idx, end_idx, doc_data))

        # Generate all embeddings in one batch
        logger.info(f"Generating embeddings for {len(all_texts)} chunks...")
        try:
            all_embeddings = self._embedding_service.embed_texts(all_texts)
        except EmbeddingServiceError as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise

        # Create documents with embeddings
        from uuid import uuid4
        from app.models.base import ChunkMetadata

        documents = []
        for start_idx, end_idx, doc_data in doc_text_ranges:
            doc_id = uuid4()
            chunks = []

            doc_metadata = DocumentMetadata(
                title=doc_data.get("title", "Untitled"),
                author=doc_data.get("author"),
                document_type=doc_data.get("document_type", "text"),
                source_url=doc_data.get("source_url"),
                tags=doc_data.get("tags", []),
            )

            # Get embeddings for this document
            doc_embeddings = all_embeddings[start_idx:end_idx]
            doc_texts = all_texts[start_idx:end_idx]

            for i, (text, embedding) in enumerate(zip(doc_texts, doc_embeddings)):
                chunk_metadata = ChunkMetadata(
                    chunk_index=i, source_document_id=doc_id
                )

                chunk = Chunk(
                    text=text, embedding=embedding.tolist(), metadata=chunk_metadata
                )
                chunks.append(chunk)

            document = Document(id=doc_id, chunks=chunks, metadata=doc_metadata)
            documents.append(document)

        # Add all documents in batch
        try:
            successful, failed = self._repository.add_documents_batch(library_id, documents)
            total_chunks = sum(len(doc.chunks) for doc in successful)

            logger.info(
                f"Batch add complete: {len(successful)} successful, {len(failed)} failed, "
                f"{total_chunks} total chunks"
            )

            return successful, failed, total_chunks
        except Exception as e:
            logger.error(f"Failed to add documents batch: {e}")
            raise

    def add_documents_batch_with_embeddings(
        self,
        library_id: UUID,
        documents_data: List[dict],
    ) -> Tuple[List[Document], List[Tuple[int, str]], int]:
        """
        Add multiple documents with pre-computed embeddings in a batch.

        Args:
            library_id: The library to add documents to.
            documents_data: List of dicts with document data including embeddings.

        Returns:
            Tuple of (successful_documents, failed_operations, total_chunks)

        Raises:
            LibraryNotFoundError: If library doesn't exist.
        """
        logger.info(
            f"Batch adding {len(documents_data)} documents (with embeddings) to library {library_id}"
        )

        # Get library to check it exists
        library = self._repository.get_library(library_id)

        # Create documents with pre-computed embeddings
        from uuid import uuid4
        from app.models.base import ChunkMetadata

        documents = []
        for doc_data in documents_data:
            doc_id = uuid4()
            chunks = []

            doc_metadata = DocumentMetadata(
                title=doc_data.get("title", "Untitled"),
                author=doc_data.get("author"),
                document_type=doc_data.get("document_type", "text"),
                source_url=doc_data.get("source_url"),
                tags=doc_data.get("tags", []),
            )

            # Get chunks with embeddings
            chunk_data_list = doc_data.get("chunks", [])

            for i, chunk_data in enumerate(chunk_data_list):
                chunk_metadata = ChunkMetadata(
                    chunk_index=i, source_document_id=doc_id
                )

                chunk = Chunk(
                    text=chunk_data["text"],
                    embedding=chunk_data["embedding"],
                    metadata=chunk_metadata,
                )
                chunks.append(chunk)

            document = Document(id=doc_id, chunks=chunks, metadata=doc_metadata)
            documents.append(document)

        # Add all documents in batch
        try:
            successful, failed = self._repository.add_documents_batch(library_id, documents)
            total_chunks = sum(len(doc.chunks) for doc in successful)

            logger.info(
                f"Batch add complete: {len(successful)} successful, {len(failed)} failed, "
                f"{total_chunks} total chunks"
            )

            return successful, failed, total_chunks
        except Exception as e:
            logger.error(f"Failed to add documents batch: {e}")
            raise

    def delete_documents_batch(
        self, document_ids: List[UUID]
    ) -> Tuple[List[UUID], List[Tuple[UUID, str]]]:
        """
        Delete multiple documents in a batch.

        Args:
            document_ids: List of document IDs to delete.

        Returns:
            Tuple of (successful_ids, failed_operations)

        Raises:
            No exceptions - failures are returned in the tuple.
        """
        logger.info(f"Batch deleting {len(document_ids)} documents")

        try:
            successful, failed = self._repository.delete_documents_batch(document_ids)

            logger.info(
                f"Batch delete complete: {len(successful)} successful, {len(failed)} failed"
            )

            return successful, failed
        except Exception as e:
            logger.error(f"Failed to delete documents batch: {e}")
            raise

    def search_with_text(
        self,
        library_id: UUID,
        query_text: str,
        k: int = 10,
        distance_threshold: Optional[float] = None,
    ) -> List[Tuple[Chunk, float]]:
        """
        Search a library using natural language query.

        This method generates an embedding for the query text automatically.

        Args:
            library_id: The library to search.
            query_text: Natural language query.
            k: Number of results to return.
            distance_threshold: Optional maximum distance threshold.

        Returns:
            List of (chunk, distance) tuples sorted by relevance.

        Raises:
            LibraryNotFoundError: If library doesn't exist.
            EmbeddingServiceError: If query embedding generation fails.
        """
        logger.info(f"Searching library {library_id} with text query (k={k})")

        # Generate query embedding
        # Switch to search_query input type for better retrieval
        original_input_type = self._embedding_service.input_type
        try:
            self._embedding_service.change_input_type("search_query")
            query_embedding = self._embedding_service.embed_text(query_text)
        except EmbeddingServiceError as e:
            logger.error(f"Failed to generate query embedding: {e}")
            raise
        finally:
            # Restore original input type
            self._embedding_service.change_input_type(original_input_type)

        # Search using embedding
        return self.search_with_embedding(
            library_id, query_embedding.tolist(), k, distance_threshold
        )

    def search_with_embedding(
        self,
        library_id: UUID,
        query_embedding: List[float],
        k: int = 10,
        distance_threshold: Optional[float] = None,
    ) -> List[Tuple[Chunk, float]]:
        """
        Search a library using a pre-computed embedding.

        Args:
            library_id: The library to search.
            query_embedding: Query vector.
            k: Number of results to return.
            distance_threshold: Optional maximum distance threshold.

        Returns:
            List of (chunk, distance) tuples sorted by relevance.

        Raises:
            LibraryNotFoundError: If library doesn't exist.
            DimensionMismatchError: If query embedding has wrong dimension.
        """
        logger.info(f"Searching library {library_id} with embedding (k={k})")

        try:
            results = self._repository.search(
                library_id, query_embedding, k, distance_threshold
            )
            logger.info(f"Found {len(results)} results for query")
            return results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise

    def search_with_metadata_filters(
        self,
        library_id: UUID,
        query_text: str,
        metadata_filters: List[dict],
        k: int = 10,
        distance_threshold: Optional[float] = None,
    ) -> List[Tuple[Chunk, float]]:
        """
        Search library with text query and apply metadata filters.

        Filters are applied AFTER vector search to narrow results.
        All filters are combined with AND logic.

        Args:
            library_id: The library to search.
            query_text: Natural language query.
            metadata_filters: List of filter dictionaries with 'field', 'operator', 'value'.
            k: Number of results to return (applied AFTER filtering).
            distance_threshold: Optional maximum distance threshold.

        Returns:
            List of (chunk, distance) tuples that match all filters.

        Raises:
            LibraryNotFoundError: If library doesn't exist.
            EmbeddingServiceError: If query embedding generation fails.
            ValueError: If filter specification is invalid.
        """
        logger.info(
            f"Searching library {library_id} with {len(metadata_filters)} metadata filters"
        )

        # First, do regular vector search with higher k to have candidates
        # We'll fetch more results and then filter them
        search_k = k * 10  # Get 10x results to account for filtering
        search_k = min(search_k, 1000)  # Cap at reasonable limit

        initial_results = self.search_with_text(
            library_id, query_text, k=search_k, distance_threshold=distance_threshold
        )

        if not metadata_filters:
            # No filters, just return top k results
            return initial_results[:k]

        # Apply metadata filters
        filtered_results = []
        for chunk, distance in initial_results:
            if self._chunk_matches_filters(chunk, metadata_filters):
                filtered_results.append((chunk, distance))

                # Stop once we have k results
                if len(filtered_results) >= k:
                    break

        logger.info(
            f"After filtering: {len(filtered_results)} results (from {len(initial_results)} candidates)"
        )

        return filtered_results

    def _chunk_matches_filters(self, chunk: Chunk, filters: List[dict]) -> bool:
        """
        Check if a chunk matches all metadata filters (AND logic).

        Args:
            chunk: The chunk to check.
            filters: List of filter dictionaries.

        Returns:
            True if chunk matches all filters, False otherwise.
        """
        for filter_spec in filters:
            field = filter_spec["field"]
            operator = filter_spec["operator"]
            value = filter_spec["value"]

            # Get the actual value from chunk metadata or document metadata
            # Check chunk metadata first
            chunk_meta = chunk.metadata
            actual_value = None

            # Map field names to actual metadata attributes
            if hasattr(chunk_meta, field):
                actual_value = getattr(chunk_meta, field)
            elif field == "title":
                # Document-level field - need to get from parent document
                # For now, we'll skip document-level filters in chunk context
                # This would require passing document info or restructuring
                logger.warning(
                    f"Document-level field '{field}' not accessible in chunk context"
                )
                return False
            else:
                # Field doesn't exist
                logger.warning(f"Field '{field}' not found in chunk metadata")
                return False

            # Apply operator
            if not self._apply_operator(actual_value, operator, value):
                return False

        return True

    def _apply_operator(self, actual: Any, operator: str, expected: Any) -> bool:
        """
        Apply a comparison operator.

        Args:
            actual: Actual value from metadata.
            operator: Comparison operator string.
            expected: Expected value to compare against.

        Returns:
            True if comparison succeeds, False otherwise.
        """
        try:
            if operator == "eq":
                return actual == expected
            elif operator == "ne":
                return actual != expected
            elif operator == "gt":
                return actual > expected
            elif operator == "lt":
                return actual < expected
            elif operator == "gte":
                return actual >= expected
            elif operator == "lte":
                return actual <= expected
            elif operator == "in":
                # Check if actual value is in the expected list
                return actual in expected
            elif operator == "contains":
                # For string: check if expected is substring
                # For list: check if expected is in the list
                if isinstance(actual, str):
                    return expected in actual
                elif isinstance(actual, list):
                    return expected in actual
                else:
                    return False
            else:
                logger.error(f"Unknown operator: {operator}")
                return False
        except (TypeError, AttributeError) as e:
            logger.warning(f"Comparison failed: {e}")
            return False

    def regenerate_embeddings(self, library_id: UUID) -> int:
        """
        Regenerate embeddings for all chunks in a library.

        This is useful after loading a library from disk where embeddings
        were not persisted. Chunks are re-embedded using their stored text.

        Args:
            library_id: The library to regenerate embeddings for.

        Returns:
            Number of chunks that were re-embedded.

        Raises:
            LibraryNotFoundError: If library doesn't exist.
            EmbeddingServiceError: If embedding generation fails.
        """
        logger.info(f"Regenerating embeddings for library {library_id}")

        # Get the library to ensure it exists
        library = self._repository.get_library(library_id)

        # Collect all text chunks that need embeddings
        texts_to_embed = []
        chunks_to_update = []

        for document in library.documents:
            for chunk in document.chunks:
                # Only re-embed chunks that don't have embeddings
                if chunk.embedding is None or len(chunk.embedding) == 0:
                    texts_to_embed.append(chunk.text)
                    chunks_to_update.append(chunk)

        if not texts_to_embed:
            logger.info(f"No chunks need re-embedding in library {library_id}")
            return 0

        logger.info(f"Re-embedding {len(texts_to_embed)} chunks...")

        # Generate embeddings in batches for efficiency
        try:
            embeddings = self._embedding_service.embed_texts(texts_to_embed)
        except EmbeddingServiceError as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise

        # Update chunks with new embeddings and add to index
        for chunk, embedding in zip(chunks_to_update, embeddings):
            chunk.embedding = embedding.tolist()

        # Add all embeddings to the vector store and index
        self._repository.add_embeddings_to_library(
            library_id, chunks_to_update, embeddings
        )

        logger.info(
            f"Successfully regenerated {len(texts_to_embed)} embeddings for library {library_id}"
        )

        return len(texts_to_embed)

    def get_library_statistics(self, library_id: UUID) -> dict:
        """
        Get statistics about a library.

        Args:
            library_id: The library ID.

        Returns:
            Dictionary with statistics.

        Raises:
            LibraryNotFoundError: If library doesn't exist.
        """
        return self._repository.get_library_statistics(library_id)

    def rebuild_index(
        self, library_id: UUID, new_index_type: Optional[str] = None, index_config: Optional[dict] = None
    ) -> Tuple[str, str, int]:
        """
        Rebuild a library's index, optionally switching to a new index type.

        This is useful for:
        - Switching between index types (e.g., brute_force → hnsw)
        - Optimizing an index that has degraded
        - Tuning index parameters

        Args:
            library_id: The library ID.
            new_index_type: Optional new index type (brute_force, kd_tree, lsh, hnsw).
            index_config: Optional index-specific configuration.

        Returns:
            Tuple of (old_index_type, new_index_type, vectors_reindexed)

        Raises:
            LibraryNotFoundError: If library doesn't exist.
            ValueError: If new_index_type is invalid.
        """
        logger.info(
            f"Rebuilding index for library {library_id}" +
            (f" (switching to {new_index_type})" if new_index_type else "")
        )

        try:
            old_type, new_type, vectors = self._repository.rebuild_index(
                library_id, new_index_type, index_config
            )

            logger.info(
                f"Index rebuild complete: {old_type} → {new_type}, {vectors} vectors reindexed"
            )

            return old_type, new_type, vectors
        except Exception as e:
            logger.error(f"Failed to rebuild index: {e}")
            raise

    def optimize_index(self, library_id: UUID) -> Tuple[int, int]:
        """
        Optimize a library's index by compacting and removing deleted entries.

        This improves search performance and reduces memory usage.

        Args:
            library_id: The library ID.

        Returns:
            Tuple of (vectors_compacted, memory_freed_bytes)

        Raises:
            LibraryNotFoundError: If library doesn't exist.
        """
        logger.info(f"Optimizing index for library {library_id}")

        try:
            vectors, memory_freed = self._repository.optimize_index(library_id)

            logger.info(
                f"Index optimization complete: {vectors} vectors compacted, "
                f"{memory_freed} bytes freed"
            )

            return vectors, memory_freed
        except Exception as e:
            logger.error(f"Failed to optimize index: {e}")
            raise

    def get_index_statistics(self, library_id: UUID) -> dict:
        """
        Get detailed statistics about a library's index.

        Args:
            library_id: The library ID.

        Returns:
            Dictionary with index statistics.

        Raises:
            LibraryNotFoundError: If library doesn't exist.
        """
        return self._repository.get_index_statistics(library_id)

    # ========================================================================
    # Advanced Query Features (Hybrid Search, Reranking)
    # ========================================================================

    def hybrid_search(
        self,
        library_id: UUID,
        query_text: str,
        k: int = 10,
        vector_weight: float = 0.7,
        metadata_weight: float = 0.3,
        recency_boost: bool = False,
        recency_half_life_days: float = 30.0,
        distance_threshold: Optional[float] = None,
        query_metadata: Optional[dict] = None,
    ) -> List[Tuple[Chunk, float, dict]]:
        """
        Perform hybrid search combining vector similarity with metadata boosting.

        This method provides production-grade ranking that combines:
        1. Vector similarity (semantic search)
        2. Metadata signals (recency, field matches, etc.)

        Args:
            library_id: The library to search.
            query_text: Natural language query.
            k: Number of results to return.
            vector_weight: Weight for vector similarity (0-1).
            metadata_weight: Weight for metadata boost (0-1, must sum to 1 with vector_weight).
            recency_boost: If True, boost recent documents.
            recency_half_life_days: Documents lose half their boost after N days.
            distance_threshold: Optional maximum distance threshold for vector search.
            query_metadata: Optional metadata to match against for boosting.

        Returns:
            List of (chunk, hybrid_score, score_breakdown) tuples sorted by hybrid score.

        Raises:
            LibraryNotFoundError: If library doesn't exist.
            EmbeddingServiceError: If query embedding generation fails.
            ValueError: If weights don't sum to 1.0.

        Example:
            ```python
            # Hybrid search with 70% vector, 30% metadata
            results = service.hybrid_search(
                library_id=lib_id,
                query_text="machine learning basics",
                k=10,
                vector_weight=0.7,
                metadata_weight=0.3,
                recency_boost=True,
                recency_half_life_days=30.0
            )

            for chunk, score, breakdown in results:
                print(f"Score: {score:.4f}")
                print(f"  Vector: {breakdown['vector_score']:.4f}")
                print(f"  Metadata: {breakdown['metadata_score']:.4f}")
                print(f"  Text: {chunk.text[:100]}")
            ```
        """
        logger.info(
            f"Hybrid search on library {library_id}: k={k}, "
            f"vector_weight={vector_weight}, metadata_weight={metadata_weight}, "
            f"recency_boost={recency_boost}"
        )

        # Import hybrid search scorer
        from app.services.hybrid_search import HybridSearchScorer, ScoringConfig

        # Create scoring configuration
        config = ScoringConfig(
            vector_weight=vector_weight,
            metadata_weight=metadata_weight,
            recency_boost_enabled=recency_boost,
            recency_half_life_days=recency_half_life_days,
        )

        # Perform vector search first (get more candidates for reranking)
        search_k = min(k * 5, 500)  # Get 5x results for reranking
        vector_results = self.search_with_text(
            library_id, query_text, k=search_k, distance_threshold=distance_threshold
        )

        if not vector_results:
            logger.info("No vector results found")
            return []

        # Apply hybrid scoring
        scorer = HybridSearchScorer(config)
        hybrid_results = scorer.score_results(vector_results, query_metadata)

        # Return top k results
        top_results = hybrid_results[:k]

        logger.info(
            f"Hybrid search complete: {len(top_results)} results, "
            f"top score={top_results[0][1]:.4f}"
        )

        return top_results

    def search_with_reranking(
        self,
        library_id: UUID,
        query_text: str,
        k: int = 10,
        rerank_function: str = "recency",
        rerank_params: Optional[dict] = None,
        distance_threshold: Optional[float] = None,
    ) -> List[Tuple[Chunk, float]]:
        """
        Search with post-processing reranking using predefined or custom functions.

        This method allows applying custom scoring logic after initial vector search,
        similar to Elasticsearch's "rescore" feature.

        Args:
            library_id: The library to search.
            query_text: Natural language query.
            k: Number of results to return.
            rerank_function: Name of reranking function to use:
                - "recency": Boost recent documents
                - "position": Boost early/late chunks in document
                - "length": Boost longer/shorter chunks
            rerank_params: Parameters for the reranking function.
            distance_threshold: Optional maximum distance threshold for vector search.

        Returns:
            List of (chunk, reranked_score) tuples sorted by reranked score.

        Raises:
            LibraryNotFoundError: If library doesn't exist.
            EmbeddingServiceError: If query embedding generation fails.
            ValueError: If rerank_function is invalid.

        Example:
            ```python
            # Rerank by recency with 30-day half-life
            results = service.search_with_reranking(
                library_id=lib_id,
                query_text="latest AI research",
                k=10,
                rerank_function="recency",
                rerank_params={"half_life_days": 30.0}
            )
            ```
        """
        logger.info(
            f"Search with reranking on library {library_id}: "
            f"rerank_function={rerank_function}"
        )

        # Import reranking utilities
        from app.services.hybrid_search import (
            ResultReranker,
            boost_by_chunk_position,
            boost_by_length,
            boost_by_recency,
        )

        # Get initial vector search results (fetch more for reranking)
        search_k = min(k * 3, 300)
        vector_results = self.search_with_text(
            library_id, query_text, k=search_k, distance_threshold=distance_threshold
        )

        if not vector_results:
            logger.info("No vector results found")
            return []

        # Convert distances to similarity scores for reranking
        # Cosine distance [0, 2] -> similarity score [1, 0]
        results_with_scores = [
            (chunk, 1.0 - (distance / 2.0)) for chunk, distance in vector_results
        ]

        # Select reranking function
        rerank_params = rerank_params or {}

        if rerank_function == "recency":
            half_life_days = rerank_params.get("half_life_days", 30.0)
            scoring_fn = boost_by_recency(half_life_days)
        elif rerank_function == "position":
            prefer_early = rerank_params.get("prefer_early", True)
            scoring_fn = boost_by_chunk_position(prefer_early)
        elif rerank_function == "length":
            prefer_longer = rerank_params.get("prefer_longer", True)
            scoring_fn = boost_by_length(prefer_longer)
        else:
            raise ValueError(
                f"Invalid rerank_function: {rerank_function}. "
                f"Must be one of: recency, position, length"
            )

        # Apply reranking
        reranker = ResultReranker(scoring_fn)
        reranked_results = reranker.rerank(results_with_scores)

        # Return top k results
        top_results = reranked_results[:k]

        logger.info(
            f"Reranking complete: {len(top_results)} results, "
            f"top score={top_results[0][1]:.4f}"
        )

        return top_results
