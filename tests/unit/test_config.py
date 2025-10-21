"""
Comprehensive unit tests for the configuration system.

Tests the pydantic-settings based configuration with environment variable
support, validation, properties, and edge cases.
"""

import pytest
import os
import multiprocessing
from unittest.mock import patch
from app.config import Settings


class TestSettingsDefaults:
    """Test default configuration values."""

    def test_server_defaults(self):
        """Test server configuration defaults."""
        settings = Settings()
        assert settings.HOST == "0.0.0.0"
        assert settings.PORT == 8000

    def test_worker_defaults(self):
        """Test worker configuration defaults."""
        settings = Settings()
        assert settings.GUNICORN_WORKERS is None
        assert settings.GUNICORN_WORKERS_FALLBACK == 4
        assert settings.GUNICORN_MAX_REQUESTS == 10000
        assert settings.GUNICORN_MAX_REQUESTS_JITTER == 1000
        assert settings.GUNICORN_TIMEOUT == 120

    def test_rate_limiting_defaults(self):
        """Test rate limiting defaults."""
        settings = Settings()
        assert settings.RATE_LIMIT_ENABLED is False
        assert settings.RATE_LIMIT_SEARCH == "30/minute"
        assert settings.RATE_LIMIT_WRITE == "60/minute"
        assert settings.RATE_LIMIT_CREATE == "10/minute"
        assert settings.RATE_LIMIT_HEALTH is False
        assert settings.RATE_LIMIT_STORAGE_URI == "memory://"

    def test_embedding_defaults(self):
        """Test embedding configuration defaults."""
        settings = Settings()
        assert settings.EMBEDDING_DIMENSION == 1024
        assert settings.EMBEDDING_MODEL == "embed-english-v3.0"

    def test_input_limit_defaults(self):
        """Test input size limit defaults."""
        settings = Settings()
        assert settings.MAX_CHUNKS_PER_DOCUMENT == 1000
        assert settings.MAX_CHUNK_LENGTH == 10000
        assert settings.MAX_QUERY_LENGTH == 1000
        assert settings.MAX_RESULTS_K == 100
        assert settings.MAX_DOCUMENTS_PER_LIBRARY == 10000


class TestSettingsEnvironmentVariables:
    """Test configuration from environment variables."""

    def test_host_from_env(self):
        """Test HOST can be set from environment."""
        with patch.dict(os.environ, {"HOST": "127.0.0.1"}, clear=False):
            settings = Settings()
            assert settings.HOST == "127.0.0.1"

    def test_port_from_env(self):
        """Test PORT can be set from environment."""
        with patch.dict(os.environ, {"PORT": "9000"}, clear=False):
            settings = Settings()
            assert settings.PORT == 9000

    def test_rate_limit_enabled_from_env(self):
        """Test RATE_LIMIT_ENABLED can be set from environment."""
        with patch.dict(os.environ, {"RATE_LIMIT_ENABLED": "true"}, clear=False):
            settings = Settings()
            assert settings.RATE_LIMIT_ENABLED is True

    def test_gunicorn_workers_from_env(self):
        """Test GUNICORN_WORKERS can be set from environment."""
        with patch.dict(os.environ, {"GUNICORN_WORKERS": "8"}, clear=False):
            settings = Settings()
            assert settings.GUNICORN_WORKERS == 8

    def test_embedding_dimension_from_env(self):
        """Test EMBEDDING_DIMENSION can be set from environment."""
        with patch.dict(os.environ, {"EMBEDDING_DIMENSION": "768"}, clear=False):
            settings = Settings()
            assert settings.EMBEDDING_DIMENSION == 768

    def test_max_chunks_from_env(self):
        """Test MAX_CHUNKS_PER_DOCUMENT can be set from environment."""
        with patch.dict(os.environ, {"MAX_CHUNKS_PER_DOCUMENT": "500"}, clear=False):
            settings = Settings()
            assert settings.MAX_CHUNKS_PER_DOCUMENT == 500

    def test_rate_limit_search_from_env(self):
        """Test RATE_LIMIT_SEARCH can be customized."""
        with patch.dict(os.environ, {"RATE_LIMIT_SEARCH": "100/minute"}, clear=False):
            settings = Settings()
            assert settings.RATE_LIMIT_SEARCH == "100/minute"


class TestSettingsWorkersProperty:
    """Test the workers computed property."""

    def test_workers_with_explicit_value(self):
        """Test workers property returns explicit GUNICORN_WORKERS value."""
        with patch.dict(os.environ, {"GUNICORN_WORKERS": "8"}, clear=False):
            settings = Settings()
            assert settings.workers == 8

    def test_workers_with_zero_clamped_to_one(self):
        """Test workers property clamps 0 to 1."""
        with patch.dict(os.environ, {"GUNICORN_WORKERS": "0"}, clear=False):
            settings = Settings()
            assert settings.workers == 1

    def test_workers_with_negative_clamped_to_one(self):
        """Test workers property clamps negative to 1."""
        with patch.dict(os.environ, {"GUNICORN_WORKERS": "-5"}, clear=False):
            settings = Settings()
            assert settings.workers == 1

    def test_workers_auto_detect_when_not_set(self):
        """Test workers auto-detects CPU count when not explicitly set."""
        # Clear GUNICORN_WORKERS if set
        env = {k: v for k, v in os.environ.items() if k != "GUNICORN_WORKERS"}
        with patch.dict(os.environ, env, clear=True):
            settings = Settings()
            # Should return actual CPU count (at least 1)
            expected = max(1, multiprocessing.cpu_count())
            assert settings.workers == expected

    def test_workers_fallback_on_cpu_detection_failure(self):
        """Test workers uses fallback if CPU detection fails."""
        with patch.dict(os.environ, {}, clear=False):
            settings = Settings()
            # Mock cpu_count to raise NotImplementedError
            with patch("multiprocessing.cpu_count", side_effect=NotImplementedError):
                # Create new instance to test the property
                settings2 = Settings()
                assert settings2.workers == settings2.GUNICORN_WORKERS_FALLBACK


class TestSettingsConvenienceProperties:
    """Test convenience property aliases."""

    def test_rate_limit_document_add_property(self):
        """Test RATE_LIMIT_DOCUMENT_ADD returns RATE_LIMIT_WRITE."""
        settings = Settings()
        assert settings.RATE_LIMIT_DOCUMENT_ADD == settings.RATE_LIMIT_WRITE

    def test_rate_limit_document_add_with_custom_write(self):
        """Test RATE_LIMIT_DOCUMENT_ADD reflects custom RATE_LIMIT_WRITE."""
        with patch.dict(os.environ, {"RATE_LIMIT_WRITE": "200/minute"}, clear=False):
            settings = Settings()
            assert settings.RATE_LIMIT_DOCUMENT_ADD == "200/minute"

    def test_max_text_length_per_chunk_property(self):
        """Test MAX_TEXT_LENGTH_PER_CHUNK returns MAX_CHUNK_LENGTH."""
        settings = Settings()
        assert settings.MAX_TEXT_LENGTH_PER_CHUNK == settings.MAX_CHUNK_LENGTH

    def test_max_search_results_property(self):
        """Test MAX_SEARCH_RESULTS returns MAX_RESULTS_K."""
        settings = Settings()
        assert settings.MAX_SEARCH_RESULTS == settings.MAX_RESULTS_K


class TestSettingsBooleanParsing:
    """Test boolean environment variable parsing."""

    @pytest.mark.parametrize(
        "env_value,expected",
        [
            ("true", True),
            ("True", True),
            ("TRUE", True),
            ("1", True),
            ("yes", True),
            ("false", False),
            ("False", False),
            ("FALSE", False),
            ("0", False),
            ("no", False),
        ],
    )
    def test_boolean_parsing(self, env_value, expected):
        """Test various boolean string formats are parsed correctly."""
        with patch.dict(os.environ, {"RATE_LIMIT_ENABLED": env_value}, clear=False):
            settings = Settings()
            assert settings.RATE_LIMIT_ENABLED == expected

    @pytest.mark.parametrize(
        "env_value,expected",
        [
            ("true", True),
            ("false", False),
        ],
    )
    def test_debug_mode_boolean_parsing(self, env_value, expected):
        """Test DEBUG boolean parsing."""
        with patch.dict(os.environ, {"DEBUG": env_value}, clear=False):
            settings = Settings()
            assert settings.DEBUG == expected


class TestSettingsMultipleInstances:
    """Test that Settings can be instantiated multiple times."""

    def test_multiple_instances_independent(self):
        """Test multiple Settings instances are independent."""
        # First instance with one port
        with patch.dict(os.environ, {"PORT": "8000"}, clear=False):
            settings1 = Settings()
            port1 = settings1.PORT

        # Second instance with different port
        with patch.dict(os.environ, {"PORT": "9000"}, clear=False):
            settings2 = Settings()
            port2 = settings2.PORT

        # They should have captured their own values
        assert port1 == 8000
        assert port2 == 9000


class TestSettingsRateLimitStorageURI:
    """Test rate limit storage URI configuration."""

    def test_default_memory_storage(self):
        """Test default uses in-memory storage."""
        settings = Settings()
        assert settings.RATE_LIMIT_STORAGE_URI == "memory://"

    def test_redis_storage_from_env(self):
        """Test Redis storage can be configured."""
        redis_uri = "redis://localhost:6379"
        with patch.dict(os.environ, {"RATE_LIMIT_STORAGE_URI": redis_uri}, clear=False):
            settings = Settings()
            assert settings.RATE_LIMIT_STORAGE_URI == redis_uri


class TestSettingsCompleteness:
    """Test that all expected configuration exists."""

    def test_has_all_server_config(self):
        """Test all server configuration fields exist."""
        settings = Settings()
        assert hasattr(settings, "HOST")
        assert hasattr(settings, "PORT")

    def test_has_all_worker_config(self):
        """Test all worker configuration fields exist."""
        settings = Settings()
        assert hasattr(settings, "GUNICORN_WORKERS")
        assert hasattr(settings, "GUNICORN_WORKERS_FALLBACK")
        assert hasattr(settings, "GUNICORN_MAX_REQUESTS")
        assert hasattr(settings, "GUNICORN_TIMEOUT")
        assert hasattr(settings, "workers")

    def test_has_all_rate_limit_config(self):
        """Test all rate limiting fields exist."""
        settings = Settings()
        assert hasattr(settings, "RATE_LIMIT_ENABLED")
        assert hasattr(settings, "RATE_LIMIT_SEARCH")
        assert hasattr(settings, "RATE_LIMIT_WRITE")
        assert hasattr(settings, "RATE_LIMIT_CREATE")
        assert hasattr(settings, "RATE_LIMIT_STORAGE_URI")
        assert hasattr(settings, "RATE_LIMIT_DOCUMENT_ADD")

    def test_has_all_limit_config(self):
        """Test all input limit fields exist."""
        settings = Settings()
        assert hasattr(settings, "MAX_CHUNKS_PER_DOCUMENT")
        assert hasattr(settings, "MAX_CHUNK_LENGTH")
        assert hasattr(settings, "MAX_QUERY_LENGTH")
        assert hasattr(settings, "MAX_RESULTS_K")
        assert hasattr(settings, "MAX_TEXT_LENGTH_PER_CHUNK")
        assert hasattr(settings, "MAX_SEARCH_RESULTS")

    def test_has_embedding_config(self):
        """Test embedding configuration fields exist."""
        settings = Settings()
        assert hasattr(settings, "EMBEDDING_DIMENSION")
        assert hasattr(settings, "EMBEDDING_MODEL")


class TestPrintConfigSummary:
    """Test configuration summary printing."""

    def test_print_config_summary_executes(self, capsys):
        """Test that print_config_summary runs without error."""
        from app.config import print_config_summary

        # Should not raise any exceptions
        print_config_summary()

        # Verify it printed something
        captured = capsys.readouterr()
        assert len(captured.out) > 0
        assert "Vector Database API" in captured.out

    def test_print_config_summary_shows_server_info(self, capsys):
        """Test that summary includes server configuration."""
        from app.config import print_config_summary

        print_config_summary()
        captured = capsys.readouterr()

        assert "Server Configuration" in captured.out or "HOST" in captured.out

    def test_print_config_summary_shows_rate_limiting(self, capsys):
        """Test that summary includes rate limiting info."""
        from app.config import print_config_summary

        print_config_summary()
        captured = capsys.readouterr()

        assert "Rate Limiting" in captured.out or "DISABLED" in captured.out
