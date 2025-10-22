#!/usr/bin/env python3
"""
Validate 100K Vector Performance
Tests HNSW vs Brute Force at large scale to validate "20-30x speedup" claim
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import time
from core.vector_store import VectorStore
from infrastructure.indexes.hnsw import HNSWIndex
from infrastructure.indexes.brute_force import BruteForceIndex

def generate_vectors(n: int, dim: int = 256) -> np.ndarray:
    """Generate random normalized vectors"""
    vectors = np.random.randn(n, dim).astype(np.float32)
    # Normalize
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms

def benchmark_index(index, vector_store, query: np.ndarray, num_queries: int = 50):
    """Benchmark search performance"""
    times = []
    for _ in range(num_queries):
        start = time.perf_counter()
        index.search(query, k=10)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        times.append(elapsed)

    return {
        'avg_ms': np.mean(times),
        'p50_ms': np.percentile(times, 50),
        'p95_ms': np.percentile(times, 95),
        'p99_ms': np.percentile(times, 99),
        'min_ms': np.min(times),
        'max_ms': np.max(times)
    }

def main():
    print("=" * 60)
    print("100K VECTOR PERFORMANCE VALIDATION")
    print("Testing HNSW vs Brute Force at Scale")
    print("=" * 60)

    DIM = 256
    N_VECTORS = 100_000
    N_QUERIES = 50

    print(f"\n📊 Test Configuration:")
    print(f"  • Vectors: {N_VECTORS:,}")
    print(f"  • Dimensions: {DIM}")
    print(f"  • Queries: {N_QUERIES}")
    print(f"  • k (neighbors): 10")

    # Generate data
    print(f"\n🔧 Generating {N_VECTORS:,} random vectors...")
    vectors = generate_vectors(N_VECTORS, DIM)
    query = generate_vectors(1, DIM)[0]

    print("✓ Vectors generated\n")

    # ========================================
    # Test 1: Brute Force @ 100K
    # ========================================
    print("=" * 60)
    print("TEST 1: BRUTE FORCE @ 100K")
    print("=" * 60)

    vector_store_bf = VectorStore(dimension=DIM)
    bf_index = BruteForceIndex(vector_store_bf)

    print("Building Brute Force index...")
    build_start = time.perf_counter()

    from uuid import uuid4
    for i, vec in enumerate(vectors):
        chunk_id = uuid4()
        vector_id = vector_store_bf.add_vector(chunk_id, vec)
        bf_index.add_vector(vector_id, vec)

        if (i + 1) % 10000 == 0:
            print(f"  Added {i + 1:,} vectors...")

    build_time = time.perf_counter() - build_start
    print(f"✓ Build complete: {build_time:.1f}s ({N_VECTORS / build_time:.0f} vec/s)")

    print(f"\nBenchmarking Brute Force search ({N_QUERIES} queries)...")
    bf_results = benchmark_index(bf_index, vector_store_bf, query, N_QUERIES)

    print(f"\n📈 Brute Force Results:")
    print(f"  • Average: {bf_results['avg_ms']:.2f}ms")
    print(f"  • P50: {bf_results['p50_ms']:.2f}ms")
    print(f"  • P95: {bf_results['p95_ms']:.2f}ms")
    print(f"  • P99: {bf_results['p99_ms']:.2f}ms")
    print(f"  • Min: {bf_results['min_ms']:.2f}ms")
    print(f"  • Max: {bf_results['max_ms']:.2f}ms")

    # ========================================
    # Test 2: HNSW @ 100K
    # ========================================
    print("\n" + "=" * 60)
    print("TEST 2: HNSW @ 100K")
    print("=" * 60)

    vector_store_hnsw = VectorStore(dimension=DIM)
    hnsw_index = HNSWIndex(
        dimension=DIM,
        vector_store=vector_store_hnsw,
        M=16,
        ef_construction=200,
        ef_search=50
    )

    print("Building HNSW index (this will take a few minutes)...")
    build_start = time.perf_counter()

    for i, vec in enumerate(vectors):
        chunk_id = uuid4()
        vector_id = vector_store_hnsw.add_vector(chunk_id, vec)
        hnsw_index.add_vector(vector_id, vec)

        if (i + 1) % 10000 == 0:
            elapsed = time.perf_counter() - build_start
            rate = (i + 1) / elapsed
            remaining = (N_VECTORS - i - 1) / rate
            print(f"  Added {i + 1:,}/{N_VECTORS:,} vectors "
                  f"({rate:.0f} vec/s, ~{remaining:.0f}s remaining)")

    build_time = time.perf_counter() - build_start
    print(f"✓ Build complete: {build_time:.1f}s ({N_VECTORS / build_time:.0f} vec/s)")

    print(f"\nBenchmarking HNSW search ({N_QUERIES} queries)...")
    hnsw_results = benchmark_index(hnsw_index, vector_store_hnsw, query, N_QUERIES)

    print(f"\n📈 HNSW Results:")
    print(f"  • Average: {hnsw_results['avg_ms']:.2f}ms")
    print(f"  • P50: {hnsw_results['p50_ms']:.2f}ms")
    print(f"  • P95: {hnsw_results['p95_ms']:.2f}ms")
    print(f"  • P99: {hnsw_results['p99_ms']:.2f}ms")
    print(f"  • Min: {hnsw_results['min_ms']:.2f}ms")
    print(f"  • Max: {hnsw_results['max_ms']:.2f}ms")

    # ========================================
    # Comparison
    # ========================================
    print("\n" + "=" * 60)
    print("COMPARISON: HNSW vs BRUTE FORCE @ 100K")
    print("=" * 60)

    speedup = bf_results['avg_ms'] / hnsw_results['avg_ms']

    print(f"\n🏆 SPEEDUP: {speedup:.1f}x")
    print(f"\n📊 Detailed Comparison:")
    print(f"  • Brute Force avg: {bf_results['avg_ms']:.2f}ms")
    print(f"  • HNSW avg:        {hnsw_results['avg_ms']:.2f}ms")
    print(f"  • Speedup:         {speedup:.1f}x faster")

    print(f"\n⚡ Latency Reduction:")
    print(f"  • Time saved:      {bf_results['avg_ms'] - hnsw_results['avg_ms']:.2f}ms per query")
    print(f"  • Reduction:       {(1 - hnsw_results['avg_ms']/bf_results['avg_ms'])*100:.1f}%")

    # Validation
    print(f"\n✅ CLAIM VALIDATION:")
    if 20 <= speedup <= 30:
        print(f"  ✓ CONFIRMED: {speedup:.1f}x speedup is within 20-30x range")
    elif speedup > 15:
        print(f"  ✓ STRONG: {speedup:.1f}x speedup (close to 20-30x target)")
    else:
        print(f"  ⚠ LOWER: {speedup:.1f}x speedup (expected 20-30x)")

    # Save results
    results = {
        'config': {
            'n_vectors': N_VECTORS,
            'dimensions': DIM,
            'n_queries': N_QUERIES
        },
        'brute_force': bf_results,
        'hnsw': hnsw_results,
        'speedup': speedup
    }

    import json
    output_path = Path(__file__).parent.parent / 'docs' / 'performance_100k_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to: docs/performance_100k_results.json")
    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
