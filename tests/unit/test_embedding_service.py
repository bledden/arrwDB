"""
Advanced EmbeddingService tests to increase coverage.

Targets missing lines in app/services/embedding_service.py to improve coverage
from 64% toward 90%+. Focuses on error handling, validation, and edge cases.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import cohere

from app.services.embedding_service import EmbeddingService, EmbeddingServiceError


class TestEmbeddingServiceInitialization:
    """Test EmbeddingService initialization edge cases."""

    def test_empty_api_key_raises_error(self):
        """Test that empty API key raises ValueError (line 74-75)."""
        with pytest.raises(ValueError) as exc_info:
            EmbeddingService(api_key="")

        assert "api key" in str(exc_info.value).lower()

    def test_invalid_input_type_raises_error(self):
        """Test that invalid input_type raises ValueError (line 83-87)."""
        with pytest.raises(ValueError) as exc_info:
            EmbeddingService(
                api_key="test_key",
                input_type="invalid_type"
            )

        error_msg = str(exc_info.value).lower()
        assert "invalid input_type" in error_msg or "invalid_type" in error_msg

    def test_embedding_dimension_zero_raises_error(self):
        """Test that dimension of 0 raises ValueError (line 90-94)."""
        with pytest.raises(ValueError) as exc_info:
            EmbeddingService(
                api_key="test_key",
                embedding_dimension=0
            )

        assert "dimension" in str(exc_info.value).lower()

    def test_embedding_dimension_negative_raises_error(self):
        """Test that negative dimension raises ValueError (line 90-94)."""
        with pytest.raises(ValueError) as exc_info:
            EmbeddingService(
                api_key="test_key",
                embedding_dimension=-10
            )

        assert "dimension" in str(exc_info.value).lower()

    def test_embedding_dimension_too_large_raises_error(self):
        """Test that dimension > 1024 raises ValueError (line 90-94)."""
        with pytest.raises(ValueError) as exc_info:
            EmbeddingService(
                api_key="test_key",
                embedding_dimension=2048
            )

        assert "1024" in str(exc_info.value)

    @patch('cohere.Client')
    def test_client_initialization_failure(self, mock_cohere_client):
        """Test that client initialization failure raises EmbeddingServiceError (line 102-104)."""
        # Make cohere.Client() raise an exception
        mock_cohere_client.side_effect = Exception("Connection failed")

        with pytest.raises(EmbeddingServiceError) as exc_info:
            EmbeddingService(api_key="test_key")

        assert "failed to initialize" in str(exc_info.value).lower()


class TestEmbedTextValidation:
    """Test embed_text validation and error handling."""

    @pytest.fixture
    def service(self):
        """Create a mock EmbeddingService for testing."""
        with patch('cohere.Client'):
            return EmbeddingService(api_key="test_key")

    def test_empty_text_raises_error(self, service):
        """Test that empty text raises ValueError (line 156-157)."""
        with pytest.raises(ValueError) as exc_info:
            service.embed_text("")

        assert "empty" in str(exc_info.value).lower()

    def test_whitespace_only_text_raises_error(self, service):
        """Test that whitespace-only text raises ValueError (line 156-157)."""
        with pytest.raises(ValueError) as exc_info:
            service.embed_text("   \n\t  ")

        assert "empty" in str(exc_info.value).lower()

    def test_text_too_long_raises_error(self, service):
        """Test that text exceeding MAX_TEXT_LENGTH raises ValueError (line 159-163)."""
        # Create text longer than 512KB
        long_text = "a" * (EmbeddingService.MAX_TEXT_LENGTH + 1)

        with pytest.raises(ValueError) as exc_info:
            service.embed_text(long_text)

        error_msg = str(exc_info.value).lower()
        assert "exceeds maximum" in error_msg or "too long" in error_msg


class TestEmbedTextErrorHandling:
    """Test embed_text API error handling (lines 193-209)."""

    @pytest.fixture
    def service(self):
        """Create an EmbeddingService with mocked client."""
        with patch('cohere.Client') as mock_client:
            service = EmbeddingService(api_key="test_key")
            service._client = mock_client.return_value
            return service

    def test_bad_request_error_handling(self, service):
        """Test BadRequestError handling (line 194, 201-204)."""
        service._client.embed.side_effect = cohere.errors.BadRequestError(
            body="Invalid request"
        )

        with pytest.raises(EmbeddingServiceError) as exc_info:
            service.embed_text("test text")

        assert "failed to generate embedding" in str(exc_info.value).lower()

    def test_unauthorized_error_handling(self, service):
        """Test UnauthorizedError handling (line 195, 201-204)."""
        service._client.embed.side_effect = cohere.errors.UnauthorizedError(
            body="Invalid API key"
        )

        with pytest.raises(EmbeddingServiceError) as exc_info:
            service.embed_text("test text")

        assert "failed to generate embedding" in str(exc_info.value).lower()

    def test_forbidden_error_handling(self, service):
        """Test ForbiddenError handling (line 196, 201-204)."""
        service._client.embed.side_effect = cohere.errors.ForbiddenError(
            body="Access denied"
        )

        with pytest.raises(EmbeddingServiceError) as exc_info:
            service.embed_text("test text")

        assert "failed to generate embedding" in str(exc_info.value).lower()

    def test_internal_server_error_handling(self, service):
        """Test InternalServerError handling (line 198, 201-204)."""
        service._client.embed.side_effect = cohere.errors.InternalServerError(
            body="Server error"
        )

        with pytest.raises(EmbeddingServiceError) as exc_info:
            service.embed_text("test text")

        assert "failed to generate embedding" in str(exc_info.value).lower()

    def test_service_unavailable_error_handling(self, service):
        """Test ServiceUnavailableError handling (line 199, 201-204)."""
        service._client.embed.side_effect = cohere.errors.ServiceUnavailableError(
            body="Service unavailable"
        )

        with pytest.raises(EmbeddingServiceError) as exc_info:
            service.embed_text("test text")

        assert "failed to generate embedding" in str(exc_info.value).lower()

    def test_unexpected_error_handling(self, service):
        """Test unexpected error handling (line 205-209)."""
        service._client.embed.side_effect = RuntimeError("Unexpected error")

        with pytest.raises(EmbeddingServiceError) as exc_info:
            service.embed_text("test text")

        assert "unexpected error" in str(exc_info.value).lower()


class TestEmbedTextDimensionTruncation:
    """Test dimension truncation feature (lines 178-179)."""

    @pytest.fixture
    def service_with_truncation(self):
        """Create service with dimension truncation."""
        with patch('cohere.Client') as mock_client:
            service = EmbeddingService(
                api_key="test_key",
                embedding_dimension=512
            )

            # Mock the embed response
            mock_response = Mock()
            mock_embedding = np.random.rand(1024).tolist()  # Full 1024-dim embedding
            mock_response.embeddings.float = [mock_embedding]

            service._client = mock_client.return_value
            service._client.embed.return_value = mock_response

            return service

    def test_dimension_truncation_applied(self, service_with_truncation):
        """Test that embeddings are truncated to specified dimension (line 178-179)."""
        result = service_with_truncation.embed_text("test text")

        # Should be truncated to 512
        assert len(result) == 512
        assert result.dtype == np.float32


class TestEmbedTextsValidation:
    """Test embed_texts validation (lines 241-252)."""

    @pytest.fixture
    def service(self):
        """Create a mock EmbeddingService for testing."""
        with patch('cohere.Client'):
            return EmbeddingService(api_key="test_key")

    def test_empty_texts_list_raises_error(self, service):
        """Test that empty list raises ValueError (line 241-242)."""
        with pytest.raises(ValueError) as exc_info:
            service.embed_texts([])

        assert "empty" in str(exc_info.value).lower()

    def test_empty_text_in_list_raises_error(self, service):
        """Test that empty text in list raises ValueError (line 246-247)."""
        with pytest.raises(ValueError) as exc_info:
            service.embed_texts(["valid text", "", "another text"])

        error_msg = str(exc_info.value).lower()
        assert "index 1" in error_msg and "empty" in error_msg

    def test_whitespace_text_in_list_raises_error(self, service):
        """Test that whitespace text in list raises ValueError (line 246-247)."""
        with pytest.raises(ValueError) as exc_info:
            service.embed_texts(["valid text", "  \n  ", "another text"])

        error_msg = str(exc_info.value).lower()
        assert "index 1" in error_msg and "empty" in error_msg

    def test_text_too_long_in_list_raises_error(self, service):
        """Test that too-long text in list raises ValueError (line 248-252)."""
        long_text = "a" * (EmbeddingService.MAX_TEXT_LENGTH + 1)

        with pytest.raises(ValueError) as exc_info:
            service.embed_texts(["valid text", long_text, "another text"])

        error_msg = str(exc_info.value).lower()
        assert "index 1" in error_msg and ("exceeds" in error_msg or "maximum" in error_msg)


class TestEmbedTextsChunking:
    """Test batch processing with chunking (lines 255-260, 324-335)."""

    @pytest.fixture
    def service_with_chunking(self):
        """Create service that will trigger chunking."""
        with patch('cohere.Client') as mock_client:
            service = EmbeddingService(api_key="test_key")

            # Mock the embed response
            def mock_embed(texts, **kwargs):
                mock_response = Mock()
                embeddings = [np.random.rand(1024).tolist() for _ in texts]
                mock_response.embeddings.float = embeddings
                return mock_response

            service._client = mock_client.return_value
            service._client.embed.side_effect = mock_embed

            return service

    def test_large_batch_triggers_chunking(self, service_with_chunking):
        """Test that batches > MAX_BATCH_SIZE are chunked (lines 255-260)."""
        # Create 200 texts (exceeds MAX_BATCH_SIZE of 96)
        texts = [f"text {i}" for i in range(200)]

        results = service_with_chunking.embed_texts(texts)

        # Should get 200 embeddings back
        assert len(results) == 200

        # Should have called embed multiple times
        assert service_with_chunking._client.embed.call_count > 1

    def test_chunked_processing_preserves_order(self, service_with_chunking):
        """Test that chunked processing preserves order (line 332-333)."""
        # Create 150 texts
        texts = [f"text_{i}" for i in range(150)]

        results = service_with_chunking.embed_texts(texts)

        # Should get exactly 150 embeddings in correct order
        assert len(results) == 150
        assert all(isinstance(emb, np.ndarray) for emb in results)


class TestEmbedTextsErrorHandling:
    """Test embed_texts API error handling (lines 294-310)."""

    @pytest.fixture
    def service(self):
        """Create an EmbeddingService with mocked client."""
        with patch('cohere.Client') as mock_client:
            service = EmbeddingService(api_key="test_key")
            service._client = mock_client.return_value
            return service

    def test_bad_request_error_in_batch(self, service):
        """Test BadRequestError handling in batch (line 295, 302-305)."""
        service._client.embed.side_effect = cohere.errors.BadRequestError(
            body="Invalid request"
        )

        with pytest.raises(EmbeddingServiceError) as exc_info:
            service.embed_texts(["text1", "text2"])

        assert "failed to generate embeddings" in str(exc_info.value).lower()

    def test_unauthorized_error_in_batch(self, service):
        """Test UnauthorizedError handling in batch (line 296, 302-305)."""
        service._client.embed.side_effect = cohere.errors.UnauthorizedError(
            body="Invalid API key"
        )

        with pytest.raises(EmbeddingServiceError) as exc_info:
            service.embed_texts(["text1", "text2"])

        assert "failed to generate embeddings" in str(exc_info.value).lower()

    def test_forbidden_error_in_batch(self, service):
        """Test ForbiddenError handling in batch (line 297, 302-305)."""
        service._client.embed.side_effect = cohere.errors.ForbiddenError(
            body="Access denied"
        )

        with pytest.raises(EmbeddingServiceError) as exc_info:
            service.embed_texts(["text1", "text2"])

        assert "failed to generate embeddings" in str(exc_info.value).lower()

    def test_unexpected_error_in_batch(self, service):
        """Test unexpected error handling in batch (line 306-310)."""
        service._client.embed.side_effect = RuntimeError("Unexpected error")

        with pytest.raises(EmbeddingServiceError) as exc_info:
            service.embed_texts(["text1", "text2"])

        assert "unexpected error" in str(exc_info.value).lower()


class TestChangeInputType:
    """Test change_input_type method."""

    @pytest.fixture
    def service(self):
        """Create a mock EmbeddingService."""
        with patch('cohere.Client'):
            return EmbeddingService(api_key="test_key")

    def test_change_to_search_query(self, service):
        """Test changing input type to search_query."""
        service.change_input_type("search_query")
        assert service._input_type == "search_query"

    def test_change_to_classification(self, service):
        """Test changing input type to classification."""
        service.change_input_type("classification")
        assert service._input_type == "classification"

    def test_change_to_clustering(self, service):
        """Test changing input type to clustering."""
        service.change_input_type("clustering")
        assert service._input_type == "clustering"

    def test_invalid_input_type_raises_error(self, service):
        """Test that invalid input type raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            service.change_input_type("invalid_type")

        assert "invalid input_type" in str(exc_info.value).lower()


class TestEmbeddingServiceProperties:
    """Test service properties."""

    def test_model_property(self):
        """Test model property returns correct value."""
        with patch('cohere.Client'):
            service = EmbeddingService(
                api_key="test_key",
                model="embed-multilingual-v3.0"
            )
            assert service.model == "embed-multilingual-v3.0"

    def test_embedding_dimension_with_custom_value(self):
        """Test embedding_dimension property with custom value."""
        with patch('cohere.Client'):
            service = EmbeddingService(
                api_key="test_key",
                embedding_dimension=768
            )
            assert service.embedding_dimension == 768

    def test_embedding_dimension_default(self):
        """Test embedding_dimension property returns default (line 125-128)."""
        with patch('cohere.Client'):
            service = EmbeddingService(
                api_key="test_key",
                embedding_dimension=None
            )
            assert service.embedding_dimension == 1024

    def test_repr(self):
        """Test __repr__ method."""
        with patch('cohere.Client'):
            service = EmbeddingService(
                api_key="test_key",
                model="embed-english-v3.0",
                embedding_dimension=512
            )
            repr_str = repr(service)
            assert "EmbeddingService" in repr_str
            assert "embed-english-v3.0" in repr_str
            assert "512" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
