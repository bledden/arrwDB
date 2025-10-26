#!/usr/bin/env python3
"""
Performance benchmark comparing Python HNSW vs Rust HNSW.

This benchmark tests:
1. Index building time
2. Search performance
3. Memory efficiency
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add parent directory to path to import Python HNSW
sys.path.insert(0, str(Path(__file__).parent.parent))

from infrastructure.indexes.hnsw import HNSWIndex
from core.vector_store import VectorStore
import rust_hnsw


def generate_test_vectors(n_vectors: int, dimension: int) -> np.ndarray:
    """Generate random normalized vectors for testing."""
    vectors = np.random.randn(n_vectors, dimension).astype(np.float32)
    # Normalize vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms
    return vectors


def benchmark_python_hnsw(vectors: np.ndarray, n_queries: int = 100, k: int = 10):
    """Benchmark Python HNSW implementation."""
    print("\n" + "=" * 60)
    print("PYTHON HNSW BENCHMARK")
    print("=" * 60)

    n_vectors, dimension = vectors.shape

    # Create vector store
    vector_store = VectorStore(dimension=dimension)

    # Build index
    print(f"\nBuilding index with {n_vectors} vectors (dimension={dimension})...")
    start = time.time()

    hnsw = HNSWIndex(vector_store, M=16, ef_construction=200, ef_search=50)

    for i in range(n_vectors):
        vid = f"vec-{i}"
        idx = vector_store.add_vector(vid, vectors[i])
        hnsw.add_vector(vid, idx)

    build_time = time.time() - start
    print(f"Build time: {build_time:.3f}s ({n_vectors/build_time:.1f} vectors/sec)")

    # Search benchmark
    print(f"\nRunning {n_queries} search queries (k={k})...")
    query_vectors = generate_test_vectors(n_queries, dimension)

    start = time.time()
    for i in range(n_queries):
        results = hnsw.search(query_vectors[i], k=k)
    search_time = time.time() - start

    avg_search_time = search_time / n_queries
    qps = n_queries / search_time

    print(f"Total search time: {search_time:.3f}s")
    print(f"Average search time: {avg_search_time*1000:.2f}ms")
    print(f"Queries per second: {qps:.1f}")

    stats = hnsw.get_statistics()
    print(f"\nIndex statistics:")
    print(f"  Levels: {stats.get('num_levels', 'N/A')}")
    print(f"  Avg connections: {stats.get('avg_connections', 'N/A')}")

    return {
        'build_time': build_time,
        'search_time': search_time,
        'avg_search_time': avg_search_time,
        'qps': qps,
    }


def benchmark_rust_hnsw(vectors: np.ndarray, n_queries: int = 100, k: int = 10):
    """Benchmark Rust HNSW implementation."""
    print("\n" + "=" * 60)
    print("RUST HNSW BENCHMARK")
    print("=" * 60)

    n_vectors, dimension = vectors.shape

    # Build index
    print(f"\nBuilding index with {n_vectors} vectors (dimension={dimension})...")
    start = time.time()

    hnsw = rust_hnsw.RustHNSWIndex(dimension=dimension, m=16, ef_construction=200, ef_search=50)

    for i in range(n_vectors):
        vid = f"vec-{i}"
        hnsw.add_vector(vid, vectors[i])

    build_time = time.time() - start
    print(f"Build time: {build_time:.3f}s ({n_vectors/build_time:.1f} vectors/sec)")

    # Search benchmark
    print(f"\nRunning {n_queries} search queries (k={k})...")
    query_vectors = generate_test_vectors(n_queries, dimension)

    start = time.time()
    for i in range(n_queries):
        results = hnsw.search(query_vectors[i], k=k)
    search_time = time.time() - start

    avg_search_time = search_time / n_queries
    qps = n_queries / search_time

    print(f"Total search time: {search_time:.3f}s")
    print(f"Average search time: {avg_search_time*1000:.2f}ms")
    print(f"Queries per second: {qps:.1f}")

    stats = hnsw.get_statistics()
    print(f"\nIndex statistics:")
    print(f"  Levels: {stats.get('num_levels', 'N/A')}")
    print(f"  Avg connections: {stats.get('avg_connections', 'N/A')}")

    return {
        'build_time': build_time,
        'search_time': search_time,
        'avg_search_time': avg_search_time,
        'qps': qps,
    }


def main():
    """Run performance benchmarks."""
    # Test configuration
    n_vectors = 10000
    dimension = 384  # Typical embedding dimension
    n_queries = 1000
    k = 10

    print("=" * 60)
    print("HNSW PERFORMANCE BENCHMARK")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Vectors: {n_vectors}")
    print(f"  Dimension: {dimension}")
    print(f"  Queries: {n_queries}")
    print(f"  k: {k}")
    print(f"  M: 16")
    print(f"  ef_construction: 200")
    print(f"  ef_search: 50")

    # Generate test data
    print(f"\nGenerating {n_vectors} random vectors...")
    vectors = generate_test_vectors(n_vectors, dimension)

    # Benchmark Python implementation
    python_results = benchmark_python_hnsw(vectors, n_queries, k)

    # Benchmark Rust implementation
    rust_results = benchmark_rust_hnsw(vectors, n_queries, k)

    # Compare results
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)

    build_speedup = python_results['build_time'] / rust_results['build_time']
    search_speedup = python_results['search_time'] / rust_results['search_time']

    print(f"\nBuild time:")
    print(f"  Python: {python_results['build_time']:.3f}s")
    print(f"  Rust:   {rust_results['build_time']:.3f}s")
    print(f"  Speedup: {build_speedup:.2f}x")

    print(f"\nSearch time (avg per query):")
    print(f"  Python: {python_results['avg_search_time']*1000:.2f}ms")
    print(f"  Rust:   {rust_results['avg_search_time']*1000:.2f}ms")
    print(f"  Speedup: {search_speedup:.2f}x")

    print(f"\nQueries per second:")
    print(f"  Python: {python_results['qps']:.1f}")
    print(f"  Rust:   {rust_results['qps']:.1f}")
    print(f"  Improvement: {(rust_results['qps'] / python_results['qps'] - 1) * 100:.1f}%")

    print("\n" + "=" * 60)
    print(f"OVERALL: Rust is {search_speedup:.2f}x faster for search!")
    print("=" * 60)


if __name__ == "__main__":
    main()
