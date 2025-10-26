"""
Custom Prometheus metrics for vector database operations.

These metrics track vector-specific operations beyond standard HTTP metrics.
"""

from prometheus_client import Counter, Histogram, Gauge

# Vector operation counters
vector_searches_total = Counter(
    "vectordb_searches_total",
    "Total number of vector searches performed",
    ["library_id", "index_type"],
)

vectors_added_total = Counter(
    "vectordb_vectors_added_total",
    "Total number of vectors added",
    ["library_id", "index_type"],
)

vectors_deleted_total = Counter(
    "vectordb_vectors_deleted_total",
    "Total number of vectors deleted",
    ["library_id"],
)

documents_added_total = Counter(
    "vectordb_documents_added_total",
    "Total number of documents added",
    ["library_id"],
)

documents_deleted_total = Counter(
    "vectordb_documents_deleted_total",
    "Total number of documents deleted",
)

# Batch operation counters
batch_operations_total = Counter(
    "vectordb_batch_operations_total",
    "Total number of batch operations",
    ["operation_type", "status"],  # operation_type: add/delete, status: success/failure
)

batch_documents_processed = Counter(
    "vectordb_batch_documents_processed",
    "Total number of documents processed in batch operations",
    ["operation_type"],
)

# Index operation counters
index_rebuilds_total = Counter(
    "vectordb_index_rebuilds_total",
    "Total number of index rebuilds",
    ["library_id", "old_type", "new_type"],
)

index_optimizations_total = Counter(
    "vectordb_index_optimizations_total",
    "Total number of index optimizations",
    ["library_id", "index_type"],
)

# Search performance histograms
search_duration_seconds = Histogram(
    "vectordb_search_duration_seconds",
    "Duration of vector search operations",
    ["library_id", "index_type"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

search_results_count = Histogram(
    "vectordb_search_results_count",
    "Number of results returned by search",
    buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000),
)

embedding_duration_seconds = Histogram(
    "vectordb_embedding_duration_seconds",
    "Duration of embedding generation",
    ["batch_size_bucket"],  # small/medium/large
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0),
)

# Index rebuild/optimization performance
index_rebuild_duration_seconds = Histogram(
    "vectordb_index_rebuild_duration_seconds",
    "Duration of index rebuild operations",
    ["library_id", "index_type"],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
)

# Database state gauges
libraries_total = Gauge(
    "vectordb_libraries_total",
    "Total number of libraries",
)

documents_total = Gauge(
    "vectordb_documents_total",
    "Total number of documents across all libraries",
)

vectors_total = Gauge(
    "vectordb_vectors_total",
    "Total number of vectors across all libraries",
)

# Library-specific gauges (updated periodically)
library_size_vectors = Gauge(
    "vectordb_library_size_vectors",
    "Number of vectors in a library",
    ["library_id", "library_name"],
)

library_size_documents = Gauge(
    "vectordb_library_size_documents",
    "Number of documents in a library",
    ["library_id", "library_name"],
)


def get_batch_size_bucket(count: int) -> str:
    """Categorize batch size for metrics."""
    if count <= 10:
        return "small"
    elif count <= 100:
        return "medium"
    else:
        return "large"
