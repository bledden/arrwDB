"""
Library service implementing domain logic.

This module provides the business logic layer for library operations,
following Domain-Driven Design principles. It sits between the API layer
and the repository layer.
"""

from typing import List, Optional, Tuple
from uuid import UUID
import logging
from pathlib import Path

from app.models.base import Library, Document, Chunk, LibraryMetadata, DocumentMetadata
from app.services.embedding_service import EmbeddingService, EmbeddingServiceError
from infrastructure.repositories.library_repository import (
    LibraryRepository,
    LibraryNotFoundError,
    DocumentNotFoundError,
    DimensionMismatchError,
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
    ):
        """
        Initialize the library service.

        Args:
            repository: The library repository.
            embedding_service: The embedding service for text-to-vector conversion.
        """
        self._repository = repository
        self._embedding_service = embedding_service

    def create_library(
        self,
        name: str,
        description: Optional[str] = None,
        index_type: str = "brute_force",
        embedding_model: Optional[str] = None,
    ) -> Library:
        """
        Create a new library.

        Args:
            name: Name of the library.
            description: Optional description.
            index_type: Type of index to use (brute_force, kd_tree, lsh, hnsw).
            embedding_model: Optional embedding model override.

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

        # Create library metadata
        metadata = LibraryMetadata(
            description=description,
            index_type=index_type,
            embedding_dimension=self._embedding_service.embedding_dimension,
            embedding_model=embedding_model or self._embedding_service.model,
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
