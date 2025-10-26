"""
Production-grade configuration system for Vector Database API.

This module provides comprehensive configuration management with support for:
- Environment variables
- CLI arguments (via environment)
- Code-level configuration
- Sensible defaults based on industry best practices

Configuration priority (highest to lowest):
1. Environment variables (CLI or shell)
2. .env file
3. Defaults (based on research and best practices)
"""

import multiprocessing
import os
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application configuration with environment variable support.

    All settings can be configured via environment variables with the same name.
    Example:
        RATE_LIMIT_ENABLED=true uvicorn app.api.main:app
        GUNICORN_WORKERS=8 gunicorn app.api.main:app
    """

    # ============================================================
    # Server Configuration
    # ============================================================

    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Worker configuration - defaults to all available CPUs
    # Research shows: CPU-bound tasks benefit from workers = CPU count
    # I/O-bound tasks benefit from workers = (2 * CPU count) + 1
    # Vector search is CPU-bound, so we use CPU count
    GUNICORN_WORKERS: Optional[int] = None  # None = auto-detect

    # Fallback if CPU detection fails
    GUNICORN_WORKERS_FALLBACK: int = 4

    # Max requests per worker before restart (prevents memory leaks)
    GUNICORN_MAX_REQUESTS: int = 10000
    GUNICORN_MAX_REQUESTS_JITTER: int = 1000

    # Timeout for worker requests
    GUNICORN_TIMEOUT: int = 120  # 2 minutes (vector search can be slow)

    @property
    def workers(self) -> int:
        """
        Calculate optimal number of workers.

        Returns actual CPU count, or fallback if detection fails.
        """
        if self.GUNICORN_WORKERS is not None:
            return max(1, self.GUNICORN_WORKERS)

        try:
            cpu_count = multiprocessing.cpu_count()
            # For vector search (CPU-bound), use exactly CPU count
            # Don't over-subscribe (no benefit, just context switching overhead)
            return max(1, cpu_count)
        except NotImplementedError:
            # CPU detection failed (some platforms don't support it)
            return self.GUNICORN_WORKERS_FALLBACK

    # ============================================================
    # Logging Configuration
    # ============================================================

    # Enable structured JSON logging (useful for production log aggregators)
    # If False, uses standard formatted logs (better for development)
    LOG_JSON_FORMAT: bool = False

    # Logging level
    LOG_LEVEL: str = "INFO"

    # ============================================================
    # Multi-Tenancy Configuration
    # ============================================================

    # Enable multi-tenancy (API key authentication required)
    # If False, single-tenant mode (no authentication required)
    MULTI_TENANCY_ENABLED: bool = False

    # Path to store tenant/API key mappings
    TENANTS_DB_PATH: str = "./data/tenants.json"

    # ============================================================
    # Rate Limiting Configuration
    # ============================================================

    # DISABLED BY DEFAULT - user must opt-in
    # Rationale: Development/testing shouldn't have artificial limits
    # Production deployments should enable with appropriate thresholds
    RATE_LIMIT_ENABLED: bool = False

    # Rate limit definitions (format: "requests/period")
    # Research-backed defaults based on typical vector DB usage:

    # Search endpoint: Most expensive operation (embedding + vector search)
    # 30/min = 1 search every 2 seconds (reasonable for interactive use)
    # Production recommendation: 100-500/min depending on infrastructure
    RATE_LIMIT_SEARCH: str = "30/minute"

    # Document creation: Calls embedding API (costs money + time)
    # 60/min = 1 document per second (suitable for bulk imports)
    # Production recommendation: 100-300/min depending on Cohere tier
    RATE_LIMIT_WRITE: str = "60/minute"

    # Library creation: Rare operation, allocates memory structures
    # 10/min = should be plenty for legitimate use
    # Production recommendation: 10-20/min (rarely needs more)
    RATE_LIMIT_CREATE: str = "10/minute"

    # Health checks: No limit (monitoring systems need unrestricted access)
    RATE_LIMIT_HEALTH: bool = False

    # Storage backend for rate limiting (in-memory by default)
    # For production with multiple workers, use Redis: "redis://localhost:6379"
    RATE_LIMIT_STORAGE_URI: str = "memory://"

    # Convenience aliases for API
    @property
    def RATE_LIMIT_DOCUMENT_ADD(self) -> str:
        """Alias for document write rate limit."""
        return self.RATE_LIMIT_WRITE

    # ============================================================
    # Input Size Limits
    # ============================================================

    # Convenience properties to match API model expectations
    @property
    def MAX_TEXT_LENGTH_PER_CHUNK(self) -> int:
        """Alias for max chunk length."""
        return self.MAX_CHUNK_LENGTH

    @property
    def MAX_SEARCH_RESULTS(self) -> int:
        """Alias for max results."""
        return self.MAX_RESULTS_K

    # Document limits
    # 1000 chunks = ~200 pages of text (typical academic paper or book chapter)
    # Production recommendation: 500-2000 depending on use case
    MAX_CHUNKS_PER_DOCUMENT: int = 1000

    # Chunk text length
    # 10,000 chars = ~2-3 pages of text (already implemented in Pydantic model)
    # Production recommendation: 5000-20000 depending on embedding model
    MAX_CHUNK_LENGTH: int = 10000

    # Search query length
    # 1000 chars = ~2-3 paragraphs (reasonable for semantic search)
    # Production recommendation: 500-2000 (longer queries don't help much)
    MAX_QUERY_LENGTH: int = 1000

    # Search results
    # 100 results = reasonable for most use cases
    # Production recommendation: 50-200 (diminishing returns after 50)
    MAX_RESULTS_K: int = 100

    # Library limits
    # 10,000 documents per library = reasonable for in-memory storage
    # Production recommendation: Adjust based on available RAM
    # Estimate: ~500MB + (200MB per 1000 documents)
    MAX_DOCUMENTS_PER_LIBRARY: int = 10000

    # Metadata limits
    MAX_TAGS_PER_DOCUMENT: int = 50
    MAX_TAG_LENGTH: int = 100
    MAX_TITLE_LENGTH: int = 500
    MAX_AUTHOR_LENGTH: int = 200

    # Bulk operations
    # 100 documents per bulk request = balance between efficiency and timeout
    # Production recommendation: 50-500 depending on document size
    MAX_BULK_DOCUMENTS: int = 100

    # ============================================================
    # API Configuration
    # ============================================================

    # API versioning
    API_VERSION: str = "v1"

    # Enable API versioning in URLs (/v1/libraries instead of /libraries)
    API_VERSIONING_ENABLED: bool = True

    # ============================================================
    # Data Storage
    # ============================================================

    # Base directory for all data (vectors, WAL, snapshots)
    VECTOR_DB_DATA_DIR: str = "./data"

    # ============================================================
    # Embedding Service (Cohere)
    # ============================================================

    COHERE_API_KEY: Optional[str] = None
    EMBEDDING_MODEL: str = "embed-english-v3.0"
    EMBEDDING_DIMENSION: int = 1024

    # ============================================================
    # Logging
    # ============================================================

    LOG_LEVEL: str = "INFO"

    # Use JSON structured logging (better for production)
    LOG_JSON_FORMAT: bool = False  # Disabled by default for readability

    # ============================================================
    # Development/Debug
    # ============================================================

    # Enable debug mode (more verbose logging, auto-reload, etc.)
    DEBUG: bool = False

    # Enable API documentation endpoints (/docs, /redoc)
    ENABLE_DOCS: bool = True

    # ============================================================
    # Performance Tuning
    # ============================================================

    # Enable memory-mapped vector storage (for large datasets)
    # Automatically enabled for libraries > 10K documents
    ENABLE_MMAP: bool = True

    # Index rebuild threshold (for dynamic indexes like KD-Tree)
    INDEX_REBUILD_THRESHOLD: int = 100

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"  # Allow extra fields for forward compatibility
    )


# Global settings instance
# This is initialized once at startup and shared across the application
settings = Settings()


def get_settings() -> Settings:
    """
    Dependency injection function for FastAPI.

    Returns:
        Global settings instance.

    Usage in FastAPI:
        from fastapi import Depends
        from app.config import get_settings, Settings

        @app.get("/config")
        def show_config(settings: Settings = Depends(get_settings)):
            return {"workers": settings.workers}
    """
    return settings


def print_config_summary() -> None:
    """
    Print configuration summary for startup logging.

    This helps users verify their configuration is correct.
    """
    print("\n" + "="*60)
    print("Vector Database API - Configuration Summary")
    print("="*60)

    print(f"\n🚀 Server Configuration:")
    print(f"   Host: {settings.HOST}")
    print(f"   Port: {settings.PORT}")
    print(f"   Workers: {settings.workers} (CPU count: {multiprocessing.cpu_count() if hasattr(multiprocessing, 'cpu_count') else 'unknown'})")
    print(f"   Debug Mode: {settings.DEBUG}")

    print(f"\n🔒 Rate Limiting:")
    if settings.RATE_LIMIT_ENABLED:
        print(f"   Status: ENABLED")
        print(f"   Search: {settings.RATE_LIMIT_SEARCH}")
        print(f"   Write: {settings.RATE_LIMIT_WRITE}")
        print(f"   Create: {settings.RATE_LIMIT_CREATE}")
    else:
        print(f"   Status: DISABLED (set RATE_LIMIT_ENABLED=true to enable)")

    print(f"\n📏 Input Limits:")
    print(f"   Max chunks/document: {settings.MAX_CHUNKS_PER_DOCUMENT}")
    print(f"   Max chunk length: {settings.MAX_CHUNK_LENGTH}")
    print(f"   Max query length: {settings.MAX_QUERY_LENGTH}")
    print(f"   Max search results: {settings.MAX_RESULTS_K}")
    print(f"   Max bulk documents: {settings.MAX_BULK_DOCUMENTS}")

    print(f"\n🔧 API Configuration:")
    print(f"   Version: {settings.API_VERSION}")
    print(f"   Versioning: {'ENABLED' if settings.API_VERSIONING_ENABLED else 'DISABLED'}")
    print(f"   Documentation: {'ENABLED' if settings.ENABLE_DOCS else 'DISABLED'} (/docs, /redoc)")

    print(f"\n💾 Storage:")
    print(f"   Data directory: {settings.VECTOR_DB_DATA_DIR}")
    print(f"   Memory-mapped storage: {'ENABLED' if settings.ENABLE_MMAP else 'DISABLED'}")

    print(f"\n🤖 Embedding Service:")
    print(f"   Model: {settings.EMBEDDING_MODEL}")
    print(f"   Dimension: {settings.EMBEDDING_DIMENSION}")
    print(f"   API Key: {'✓ Configured' if settings.COHERE_API_KEY else '✗ MISSING'}")

    print(f"\n📊 Logging:")
    print(f"   Level: {settings.LOG_LEVEL}")
    print(f"   Format: {'JSON' if settings.LOG_JSON_FORMAT else 'TEXT'}")

    print("\n" + "="*60)
    print(f"To customize: Set environment variables or create .env file")
    print(f"Example: RATE_LIMIT_ENABLED=true GUNICORN_WORKERS=8 python run_api.py")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Allow running this file directly to view current configuration
    print_config_summary()
