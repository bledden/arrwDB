"""
Domain-specific exception hierarchy for arrwDB.

Rationale for structured exceptions over generic Exception:
- Allows fine-grained error handling and recovery strategies
- Communicates failure modes clearly to API consumers
- Enables different retry logic for different error types
- Shows understanding of operational failure categories

Exception hierarchy follows domain-driven design:
- Separate read vs write failures (different recovery strategies)
- Distinguish client errors (4xx) from system errors (5xx)
- Identify transient vs permanent failures (retry vs abort)
- Capture operational context (what failed, why, how to recover)
"""


# ============================================================================
# Base Exception Hierarchy
# ============================================================================


class ArrwDBError(Exception):
    """
    Base exception for all arrwDB errors.

    WHY: Single root exception allows catching all domain errors while
    letting system errors (MemoryError, KeyboardInterrupt) propagate.
    """

    def __init__(self, message: str, details: dict = None):
        """
        Initialize exception with message and optional context.

        Args:
            message: Human-readable error description
            details: Additional context for debugging (corpus_id, vector count, etc.)
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message


# ============================================================================
# Operational Error Categories (transient vs permanent)
# ============================================================================


class TransientError(ArrwDBError):
    """
    Transient error that may succeed on retry.

    RETRY STRATEGY: Exponential backoff, typically 3-5 attempts.
    Examples: Lock contention, temporary resource exhaustion, network blip.
    """
    pass


class PermanentError(ArrwDBError):
    """
    Permanent error that will never succeed even with retries.

    ABORT: Don't retry - fix the input or system state first.
    Examples: Invalid vector dimension, corpus not found, corrupted index.
    """
    pass


# ============================================================================
# Resource Not Found Errors (404 class)
# ============================================================================


class ResourceNotFoundError(PermanentError):
    """
    Base for all "resource not found" errors.

    WHY: 404-class errors should never be retried. Client must create resource first.
    """
    pass


class CorpusNotFoundError(ResourceNotFoundError):
    """
    Corpus (document collection) does not exist.

    RECOVERY: Create corpus first, or check corpus_id for typos.
    """
    pass


class DocumentNotFoundError(ResourceNotFoundError):
    """
    Document does not exist in corpus.

    RECOVERY: Verify document_id, or check if it was deleted.
    """
    pass


class ChunkNotFoundError(ResourceNotFoundError):
    """
    Chunk does not exist.

    RECOVERY: Document may have been updated/deleted. Refresh corpus state.
    """
    pass


# ============================================================================
# Validation Errors (400 class - client error)
# ============================================================================


class ValidationError(PermanentError):
    """
    Base for input validation failures.

    WHY: Client sent invalid data. Never retry without fixing input.
    """
    pass


class DimensionMismatchError(ValidationError):
    """
    Vector dimension doesn't match corpus's embedding dimension.

    SEEN IN PROD: Most common error when mixing embedding models.
    Corpus expects 1024D (Cohere), client sends 1536D (OpenAI).

    RECOVERY: Use correct embedding model for this corpus, or create new corpus.
    """
    pass


class InvalidVectorError(ValidationError):
    """
    Vector contains NaN, Inf, or other invalid values.

    SEEN IN PROD: Embedding model returned NaN due to empty input text.

    RECOVERY: Validate embedding model output before insertion.
    """
    pass


class InvalidQueryError(ValidationError):
    """
    Search query is malformed or unsupported.

    Examples: k <= 0, k > 10000, invalid metadata filter syntax.

    RECOVERY: Fix query parameters.
    """
    pass


# ============================================================================
# State Errors (conflicts, precondition failures)
# ============================================================================


class StateError(PermanentError):
    """
    Operation invalid for current system state.

    WHY: Different from validation - input is valid, but state is wrong.
    """
    pass


class CorpusNotEmptyError(StateError):
    """
    Cannot perform operation on non-empty corpus.

    SEEN: Changing index type requires empty corpus (or rebuild).

    RECOVERY: Delete all documents first, or use rebuild operation.
    """
    pass


class IndexNotBuiltError(StateError):
    """
    Index not yet constructed (e.g., no vectors added).

    SEEN: Search on empty corpus with HNSW (graph has no entry point).

    RECOVERY: Add vectors first, or check corpus.document_count.
    """
    pass


# ============================================================================
# Data Corruption Errors (system integrity failures)
# ============================================================================


class CorruptionError(PermanentError):
    """
    Data structure corrupted - internal consistency check failed.

    WHY: These should be RARE. If seen often, indicates serious bug.
    """
    pass


class VectorSpaceCorruption(CorruptionError):
    """
    VectorArena internal state inconsistent.

    SEEN IN PROD: Reference count mismatch after concurrent delete race.

    RECOVERY: Rebuild corpus from source documents. File bug report.
    """
    pass


class IndexFragmentation(CorruptionError):
    """
    Index graph structure corrupted (disconnected components, dangling edges).

    SEEN IN PROD: HNSW graph has disconnected islands after bulk delete.
    Search recall drops to <10%.

    RECOVERY: Rebuild index. Check for concurrent modifications.
    """
    pass


class PersistenceCorruption(CorruptionError):
    """
    On-disk state (WAL, snapshots) corrupted or inconsistent.

    SEEN: Partial write during crash, checksum mismatch on load.

    RECOVERY: Restore from backup. May lose recent writes.
    """
    pass


# ============================================================================
# Resource Exhaustion Errors (429/503 class - system limit)
# ============================================================================


class ResourceExhaustedError(TransientError):
    """
    System resource limit reached.

    RETRY: May succeed after other operations complete.
    """
    pass


class MemoryLimitExceeded(ResourceExhaustedError):
    """
    Memory limit reached - cannot allocate more vectors.

    SEEN IN PROD: Adding 1M vectors to in-memory index on 8GB machine.

    RECOVERY: Use memory-mapped storage, or upgrade instance size.
    """
    pass


class ConcurrencyLimitExceeded(ResourceExhaustedError):
    """
    Too many concurrent operations - lock contention or queue full.

    SEEN: 100+ concurrent searches on single corpus.

    RECOVERY: Retry with backoff, or implement client-side rate limiting.
    """
    pass


class StorageLimitExceeded(ResourceExhaustedError):
    """
    Disk space exhausted.

    RECOVERY: Free space or expand volume. May be transient if logs rotate.
    """
    pass


# ============================================================================
# Performance Degradation Warnings (operational alerts)
# ============================================================================


class PerformanceDegradation(ArrwDBError):
    """
    Operation succeeded but performance degraded significantly.

    WHY: Not a failure, but signals operational issue needing attention.
    Used for alerting/monitoring, not error handling.
    """
    pass


class SearchLatencySpike(PerformanceDegradation):
    """
    Search took >10x longer than p50 latency.

    SEEN: HNSW search on 50M vectors after no rebuild for 6 months.

    ACTION: Schedule index rebuild, investigate graph quality.
    """
    pass


class IndexQualityDegraded(PerformanceDegradation):
    """
    Index quality metrics below threshold (recall <90%).

    SEEN: After 30%+ deletions without rebuild.

    ACTION: Rebuild index to restore quality.
    """
    pass


# ============================================================================
# Service Integration Errors (external dependencies)
# ============================================================================


class IntegrationError(TransientError):
    """
    External service failure (embedding model, webhook delivery, etc.).

    RETRY: External service may recover.
    """
    pass


class EmbeddingServiceError(IntegrationError):
    """
    Embedding model API failed or returned invalid output.

    SEEN: Rate limit (429), timeout (504), model returning NaN.

    RECOVERY: Retry with backoff. May need different embedding provider.
    """
    pass


class WebhookDeliveryFailed(IntegrationError):
    """
    Webhook delivery failed after max retries.

    SEEN: Endpoint down, network partition, TLS cert expired.

    RECOVERY: Check endpoint health. May need manual replay.
    """
    pass


# ============================================================================
# Backward Compatibility Aliases (for migration from old exception names)
# ============================================================================

# Repository layer currently uses these names
LibraryNotFoundError = CorpusNotFoundError  # Deprecated - use CorpusNotFoundError


# ============================================================================
# Exception Helpers
# ============================================================================


def is_retryable(error: Exception) -> bool:
    """
    Check if error is transient and should be retried.

    WHY: Centralize retry logic instead of isinstance checks everywhere.

    Returns:
        True if error is transient, False if permanent.
    """
    return isinstance(error, TransientError)


def get_http_status(error: Exception) -> int:
    """
    Map domain exception to HTTP status code.

    WHY: API layer needs to convert exceptions to HTTP responses.
    Better than littering API code with exception type checks.

    Returns:
        HTTP status code (400, 404, 409, 429, 500, 503).
    """
    if isinstance(error, ResourceNotFoundError):
        return 404
    elif isinstance(error, ValidationError):
        return 400
    elif isinstance(error, StateError):
        return 409  # Conflict
    elif isinstance(error, ConcurrencyLimitExceeded):
        return 429  # Too Many Requests
    elif isinstance(error, ResourceExhaustedError):
        return 503  # Service Unavailable
    elif isinstance(error, CorruptionError):
        return 500  # Internal Server Error
    elif isinstance(error, IntegrationError):
        return 502  # Bad Gateway
    else:
        return 500  # Default to Internal Server Error
