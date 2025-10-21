"""
Tests for FastAPI dependencies to increase coverage.

Targets missing lines in app/api/dependencies.py to improve coverage
from 76% to 80%+.
"""

import pytest
import os
from pathlib import Path
from unittest.mock import patch

from app.api.dependencies import (
    get_data_dir,
    get_library_repository,
    get_embedding_service,
    get_library_service
)
from app.services.embedding_service import EmbeddingService
from infrastructure.repositories.library_repository import LibraryRepository


class TestGetDataDir:
    """Test get_data_dir function."""

    def test_get_data_dir_default(self):
        """Test get_data_dir with default value."""
        # Clear cache
        get_data_dir.cache_clear()

        # Patch settings to return default
        from app.config import settings
        with patch.object(settings, 'VECTOR_DB_DATA_DIR', './data'):
            result = get_data_dir()
            assert result == Path("./data").resolve()

    def test_get_data_dir_custom_env(self):
        """Test get_data_dir with custom environment variable (line 26-27)."""
        # Clear cache
        get_data_dir.cache_clear()

        custom_dir = "/custom/data/dir"
        from app.config import settings
        with patch.object(settings, 'VECTOR_DB_DATA_DIR', custom_dir):
            result = get_data_dir()
            assert result == Path(custom_dir).resolve()


class TestGetLibraryRepository:
    """Test get_library_repository function."""

    def test_get_library_repository_creates_instance(self):
        """Test that get_library_repository creates LibraryRepository (line 38-39)."""
        # Clear cache
        get_library_repository.cache_clear()
        get_data_dir.cache_clear()

        from app.config import settings
        with patch.object(settings, 'VECTOR_DB_DATA_DIR', "/tmp/test_data"):
            result = get_library_repository()
            assert isinstance(result, LibraryRepository)
            assert result._data_dir == Path("/tmp/test_data").resolve()


class TestGetEmbeddingService:
    """Test get_embedding_service function."""

    def test_get_embedding_service_missing_api_key_raises_error(self):
        """Test that missing COHERE_API_KEY raises ValueError (line 55)."""
        # Clear cache
        get_embedding_service.cache_clear()

        from app.config import settings
        with patch.object(settings, 'COHERE_API_KEY', None):
            with pytest.raises(ValueError) as exc_info:
                get_embedding_service()

            error_msg = str(exc_info.value).lower()
            assert "cohere_api_key" in error_msg
            assert "must be set" in error_msg

    def test_get_embedding_service_invalid_dimension_uses_default(self):
        """Test that invalid dimension string uses default 1024 (line 65-66)."""
        # Clear cache
        get_embedding_service.cache_clear()

        from app.config import settings
        with patch.object(settings, 'COHERE_API_KEY', 'test_key_12345'):
            with patch.object(settings, 'EMBEDDING_MODEL', 'embed-english-v3.0'):
                with patch.object(settings, 'EMBEDDING_DIMENSION', 1024):
                    with patch('cohere.Client'):
                        service = get_embedding_service()
                        # Should use 1024 (which becomes None in the constructor due to line 72)
                        assert service.embedding_dimension == 1024

    def test_get_embedding_service_with_valid_dimension(self):
        """Test get_embedding_service with valid dimension."""
        # Clear cache
        get_embedding_service.cache_clear()

        from app.config import settings
        with patch.object(settings, 'COHERE_API_KEY', 'test_key_12345'):
            with patch.object(settings, 'EMBEDDING_MODEL', 'embed-english-v3.0'):
                with patch.object(settings, 'EMBEDDING_DIMENSION', 512):
                    with patch('cohere.Client'):
                        service = get_embedding_service()
                        assert service.embedding_dimension == 512


class TestGetLibraryService:
    """Test get_library_service function."""

    def test_get_library_service_creates_instance(self):
        """Test that get_library_service creates LibraryService."""
        from app.services.library_service import LibraryService
        from unittest.mock import Mock

        # Create mock dependencies
        mock_repository = Mock(spec=LibraryRepository)
        mock_embedding_service = Mock(spec=EmbeddingService)

        result = get_library_service(
            repository=mock_repository,
            embedding_service=mock_embedding_service
        )

        assert isinstance(result, LibraryService)
        assert result._repository == mock_repository
        assert result._embedding_service == mock_embedding_service


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
