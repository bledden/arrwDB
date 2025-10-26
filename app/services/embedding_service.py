"""
Embedding Service using Cohere API.

This module provides the EmbeddingService which generates embeddings
for text using the Cohere API. It handles batching, retries, and
error handling for robust production use.
"""

import logging
from typing import List, Optional

import cohere
import numpy as np
from numpy.typing import NDArray
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

# Handle different Cohere SDK versions
try:
    # Cohere SDK 5.x
    from cohere.errors import (
        BadRequestError,
        ForbiddenError,
        GatewayTimeoutError,
        InternalServerError,
        ServiceUnavailableError,
        TooManyRequestsError,
        UnauthorizedError,
    )
except (ImportError, AttributeError):
    # Cohere SDK 4.x fallback - use base exceptions
    TooManyRequestsError = Exception
    ServiceUnavailableError = Exception
    GatewayTimeoutError = Exception
    InternalServerError = Exception
    BadRequestError = Exception
    UnauthorizedError = Exception
    ForbiddenError = Exception


class EmbeddingServiceError(Exception):
    """Base exception for embedding service errors."""

    pass


class EmbeddingService:
    """
    Service for generating text embeddings using Cohere API.

    This service provides:
    - Text-to-vector embedding generation
    - Automatic batching for efficiency
    - Retry logic for transient failures
    - Dimension validation
    - Rate limiting awareness

    Thread-Safety: This class is thread-safe. The Cohere client handles
    concurrent requests internally.
    """

    # Cohere API limits
    MAX_BATCH_SIZE = 96  # Cohere's max batch size
    MAX_TEXT_LENGTH = 512 * 1024  # 512KB per text

    def __init__(
        self,
        api_key: str,
        model: str = "embed-english-v3.0",
        input_type: str = "search_document",
        embedding_dimension: Optional[int] = None,
    ) -> None:
        """
        Initialize the EmbeddingService.

        Args:
            api_key: Cohere API key for authentication.
            model: The embedding model to use. Default is embed-english-v3.0.
            input_type: The input type for embeddings. Options:
                - "search_document": For indexing documents
                - "search_query": For query embeddings
                - "classification": For classification tasks
                - "clustering": For clustering tasks
            embedding_dimension: Optional dimension to truncate embeddings to.
                If None, uses the model's default dimension (1024 for v3.0).

        Raises:
            ValueError: If parameters are invalid.
            EmbeddingServiceError: If client initialization fails.
        """
        if not api_key:
            raise ValueError("API key cannot be empty")

        valid_input_types = {
            "search_document",
            "search_query",
            "classification",
            "clustering",
        }
        if input_type not in valid_input_types:
            raise ValueError(
                f"Invalid input_type: {input_type}. "
                f"Must be one of {valid_input_types}"
            )

        if embedding_dimension is not None:
            if embedding_dimension <= 0 or embedding_dimension > 1024:
                raise ValueError(
                    f"Embedding dimension must be between 1 and 1024, "
                    f"got {embedding_dimension}"
                )

        try:
            self._client = cohere.Client(api_key)
            self._model = model
            self._input_type = input_type
            self._embedding_dimension = embedding_dimension
        except Exception as e:
            raise EmbeddingServiceError(
                f"Failed to initialize Cohere client: {e}"
            ) from e

        logger.info(
            f"Initialized EmbeddingService with model={model}, "
            f"input_type={input_type}, dimension={embedding_dimension}"
        )

    @property
    def model(self) -> str:
        """Get the embedding model name."""
        return self._model

    @property
    def input_type(self) -> str:
        """Get the current input type."""
        return self._input_type

    @property
    def embedding_dimension(self) -> int:
        """
        Get the embedding dimension.

        Returns:
            The dimension of embeddings produced by this service.
            Default is 1024 for embed-english-v3.0.
        """
        if self._embedding_dimension is not None:
            return self._embedding_dimension
        # Default dimension for embed-english-v3.0
        return 1024

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(
            (
                TooManyRequestsError,
                ServiceUnavailableError,
                GatewayTimeoutError,
                InternalServerError,
            )
        ),
        reraise=True,
    )
    def embed_text(self, text: str) -> NDArray[np.float32]:
        """
        Generate embedding for a single text.

        Args:
            text: The text to embed. Must not exceed MAX_TEXT_LENGTH.

        Returns:
            A 1D numpy array containing the normalized embedding.

        Raises:
            ValueError: If text is empty or too long.
            EmbeddingServiceError: If the API call fails after retries.
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        if len(text) > self.MAX_TEXT_LENGTH:
            raise ValueError(
                f"Text length ({len(text)}) exceeds maximum "
                f"({self.MAX_TEXT_LENGTH})"
            )

        try:
            response = self._client.embed(
                texts=[text],
                model=self._model,
                input_type=self._input_type,
                truncate="END",  # Truncate if text is too long
            )

            # Extract embedding
            embedding = np.array(response.embeddings[0], dtype=np.float32)

            # Truncate to desired dimension if specified
            if self._embedding_dimension is not None:
                embedding = embedding[: self._embedding_dimension]

            # Normalize to unit length
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            logger.debug(
                f"Generated embedding with dimension {len(embedding)} "
                f"for text of length {len(text)}"
            )

            return embedding

        except (
            BadRequestError,
            UnauthorizedError,
            ForbiddenError,
            TooManyRequestsError,
            InternalServerError,
            ServiceUnavailableError,
        ) as e:
            logger.error(f"Cohere API error: {e}")
            raise EmbeddingServiceError(
                f"Failed to generate embedding: {e}"
            ) from e
        except Exception as e:
            logger.error(f"Unexpected error generating embedding: {e}")
            raise EmbeddingServiceError(
                f"Unexpected error: {e}"
            ) from e

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(
            (
                TooManyRequestsError,
                ServiceUnavailableError,
                GatewayTimeoutError,
                InternalServerError,
            )
        ),
        reraise=True,
    )
    def embed_texts(self, texts: List[str]) -> List[NDArray[np.float32]]:
        """
        Generate embeddings for multiple texts in batch.

        This is more efficient than calling embed_text repeatedly.
        Automatically handles batching if the number of texts exceeds
        the API limit.

        Args:
            texts: List of texts to embed. Each must not exceed MAX_TEXT_LENGTH.

        Returns:
            List of 1D numpy arrays, one per input text.

        Raises:
            ValueError: If texts list is empty or any text is invalid.
            EmbeddingServiceError: If the API call fails after retries.
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")

        # Validate all texts
        for i, text in enumerate(texts):
            if not text or not text.strip():
                raise ValueError(f"Text at index {i} is empty")
            if len(text) > self.MAX_TEXT_LENGTH:
                raise ValueError(
                    f"Text at index {i} length ({len(text)}) exceeds "
                    f"maximum ({self.MAX_TEXT_LENGTH})"
                )

        # If batch size exceeds limit, process in chunks
        if len(texts) > self.MAX_BATCH_SIZE:
            logger.info(
                f"Batch size {len(texts)} exceeds maximum "
                f"{self.MAX_BATCH_SIZE}, processing in chunks"
            )
            return self._embed_texts_chunked(texts)

        try:
            response = self._client.embed(
                texts=texts,
                model=self._model,
                input_type=self._input_type,
                truncate="END",
            )

            # Extract and process embeddings
            embeddings = []
            for emb in response.embeddings:
                embedding = np.array(emb, dtype=np.float32)

                # Truncate to desired dimension if specified
                if self._embedding_dimension is not None:
                    embedding = embedding[: self._embedding_dimension]

                # Normalize to unit length
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm

                embeddings.append(embedding)

            logger.debug(
                f"Generated {len(embeddings)} embeddings with dimension "
                f"{len(embeddings[0])}"
            )

            return embeddings

        except (
            BadRequestError,
            UnauthorizedError,
            ForbiddenError,
            TooManyRequestsError,
            InternalServerError,
            ServiceUnavailableError,
        ) as e:
            logger.error(f"Cohere API error: {e}")
            raise EmbeddingServiceError(
                f"Failed to generate embeddings: {e}"
            ) from e
        except Exception as e:
            logger.error(f"Unexpected error generating embeddings: {e}")
            raise EmbeddingServiceError(
                f"Unexpected error: {e}"
            ) from e

    def _embed_texts_chunked(
        self, texts: List[str]
    ) -> List[NDArray[np.float32]]:
        """
        Process large batches by splitting into chunks.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embeddings.
        """
        all_embeddings = []

        for i in range(0, len(texts), self.MAX_BATCH_SIZE):
            chunk = texts[i : i + self.MAX_BATCH_SIZE]
            logger.debug(
                f"Processing chunk {i // self.MAX_BATCH_SIZE + 1} "
                f"with {len(chunk)} texts"
            )
            chunk_embeddings = self.embed_texts(chunk)
            all_embeddings.extend(chunk_embeddings)

        return all_embeddings

    def change_input_type(self, input_type: str) -> None:
        """
        Change the input type for future embeddings.

        Use "search_document" when indexing documents and "search_query"
        when embedding user queries for search.

        Args:
            input_type: The new input type.

        Raises:
            ValueError: If input_type is invalid.
        """
        valid_input_types = {
            "search_document",
            "search_query",
            "classification",
            "clustering",
        }
        if input_type not in valid_input_types:
            raise ValueError(
                f"Invalid input_type: {input_type}. "
                f"Must be one of {valid_input_types}"
            )

        self._input_type = input_type
        logger.info(f"Changed input_type to {input_type}")

    def __repr__(self) -> str:
        """String representation of the service."""
        return (
            f"EmbeddingService(model={self._model}, "
            f"input_type={self._input_type}, "
            f"dimension={self.embedding_dimension})"
        )
