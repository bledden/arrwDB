"""
ann-benchmarks adapter for arrwDB.

This module implements the BaseANN interface required by ann-benchmarks.com.
It wraps arrwDB's RustFastHNSWIndex for direct index-level benchmarking
(no HTTP API, no VectorStore — raw Rust HNSW performance).

To use with ann-benchmarks:
1. Copy this file to ann-benchmarks/ann_benchmarks/algorithms/arrwdb/module.py
2. Add arrwDB to ann-benchmarks/algos.yaml
3. Run: python run.py --algorithm arrwdb --dataset sift-128-euclidean

See: https://github.com/erikbern/ann-benchmarks
"""

import numpy as np
from ann_benchmarks.algorithms.base import BaseANN


class ArrwDB(BaseANN):
    """arrwDB HNSW adapter for ann-benchmarks."""

    def __init__(self, metric, method_param):
        self._metric = metric
        self._m = method_param.get("M", 16)
        self._ef_construction = method_param.get("efConstruction", 200)
        self._ef_search = method_param.get("efSearch", 50)
        self._index = None

    def fit(self, X):
        """Build the HNSW index from training data."""
        from rust_hnsw import RustFastHNSWIndex

        n, dim = X.shape
        X = np.ascontiguousarray(X, dtype=np.float32)

        # Map ann-benchmarks metric names to arrwDB metric names
        metric_map = {
            "angular": "cosine",
            "euclidean": "l2",
        }
        arrwdb_metric = metric_map.get(self._metric, "cosine")

        self._index = RustFastHNSWIndex(
            dimension=dim,
            m=self._m,
            ef_construction=self._ef_construction,
            ef_search=self._ef_search,
            metric=arrwdb_metric,
        )

        # Use build_bulk for fastest construction
        ids = [str(i) for i in range(n)]
        self._index.build_bulk(ids, X.ravel())

        # Build ID mapping for result conversion
        self._n = n

    def set_query_arguments(self, ef_search):
        """Set ef_search for the upcoming queries."""
        self._ef_search = ef_search
        if self._index is not None:
            self._index.set_ef_search(ef_search)

    def query(self, v, n):
        """Search for n nearest neighbors of vector v."""
        results = self._index.search(
            np.ascontiguousarray(v, dtype=np.float32),
            k=n,
            ef_override=self._ef_search,
        )
        # Return integer indices (ann-benchmarks expects this)
        return [int(vid) for vid, dist in results]

    def get_memory_usage(self):
        """Return approximate memory usage in KB."""
        if self._index is None:
            return 0
        stats = self._index.get_statistics()
        n = stats.get("size", 0)
        dim = stats.get("dimension", 0)
        m = stats.get("M", 0)
        # Vectors + graph + overhead
        vector_bytes = n * dim * 4
        graph_bytes = n * m * 2 * 8  # approximate
        return (vector_bytes + graph_bytes) // 1024

    def __str__(self):
        return f"arrwDB(M={self._m}, efC={self._ef_construction}, efS={self._ef_search})"
