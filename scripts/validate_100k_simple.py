#!/usr/bin/env python3
"""
Simplified 100K Vector Performance Test
Uses LibraryRepository infrastructure (same as 10K test)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import time
from uuid import uuid4
from infrastructure.repositories.library_repository import LibraryRepository
from infrastructure.indexes.hnsw import HNSWIndex
from infrastructure.indexes.brute_force import BruteForceIndex

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

    # Create repositories
    print(f"\n🔧 Setting up repositories...")
    import tempfile
    temp_dir_bf = Path(tempfile.mkdtemp())
    temp_dir_hnsw = Path(tempfile.mkdtemp())
    repo_bf = LibraryRepository(data_dir=temp_dir_bf)
    repo_hnsw = LibraryRepository(data_dir=temp_dir_hnsw)

    lib_id_bf = uuid4()
    lib_id_hnsw = uuid4()

    # Create libraries with different indexes
    from app.models.base import IndexType
    repo_bf.create_library(lib_id_bf, "test_brute_force", DIM, IndexType.BRUTE_FORCE)
    repo_hnsw.create_library(lib_id_hnsw, "test_hnsw", DIM, IndexType.HNSW)

    # Generate test data
    print(f"\n🔧 Generating {N_VECTORS:,} random vectors...")
    vectors = np.random.randn(N_VECTORS, DIM).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms
    query = np.random.randn(DIM).astype(np.float32)
    query = query / np.linalg.norm(query)
    print("✓ Vectors generated\n")

    # ========================================
    # Test 1: Brute Force @ 100K
    # ========================================
    print("=" * 60)
    print("TEST 1: BRUTE FORCE @ 100K")
    print("=" * 60)

    print("Building Brute Force index...")
    build_start = time.perf_counter()

    for i, vec in enumerate(vectors):
        doc_id = uuid4()
        chunk_id = uuid4()
        repo_bf.add_document(lib_id_bf, doc_id, f"doc_{i}", [vec.tolist()], [chunk_id])

        if (i + 1) % 10000 == 0:
            elapsed = time.perf_counter() - build_start
            rate = (i + 1) / elapsed
            print(f"  Added {i + 1:,} vectors ({rate:.0f} vec/s)")

    build_time = time.perf_counter() - build_start
    print(f"✓ Build complete: {build_time:.1f}s ({N_VECTORS / build_time:.0f} vec/s)")

    print(f"\nBenchmarking Brute Force search ({N_QUERIES} queries)...")
    times_bf = []
    for _ in range(N_QUERIES):
        start = time.perf_counter()
        repo_bf.search(lib_id_bf, query.tolist(), k=10)
        elapsed = (time.perf_counter() - start) * 1000
        times_bf.append(elapsed)

    bf_avg = np.mean(times_bf)
    bf_p95 = np.percentile(times_bf, 95)
    bf_p99 = np.percentile(times_bf, 99)

    print(f"\n📈 Brute Force Results:")
    print(f"  • Average: {bf_avg:.2f}ms")
    print(f"  • P95: {bf_p95:.2f}ms")
    print(f"  • P99: {bf_p99:.2f}ms")
    print(f"  • Min: {np.min(times_bf):.2f}ms")
    print(f"  • Max: {np.max(times_bf):.2f}ms")

    # ========================================
    # Test 2: HNSW @ 100K
    # ========================================
    print("\n" + "=" * 60)
    print("TEST 2: HNSW @ 100K")
    print("=" * 60)

    print("Building HNSW index (this will take several minutes)...")
    build_start = time.perf_counter()

    for i, vec in enumerate(vectors):
        doc_id = uuid4()
        chunk_id = uuid4()
        repo_hnsw.add_document(lib_id_hnsw, doc_id, f"doc_{i}", [vec.tolist()], [chunk_id])

        if (i + 1) % 10000 == 0:
            elapsed = time.perf_counter() - build_start
            rate = (i + 1) / elapsed
            remaining = (N_VECTORS - i - 1) / rate
            print(f"  Added {i + 1:,}/{N_VECTORS:,} vectors "
                  f"({rate:.0f} vec/s, ~{remaining/60:.1f}min remaining)")

    build_time = time.perf_counter() - build_start
    print(f"✓ Build complete: {build_time:.1f}s ({N_VECTORS / build_time:.0f} vec/s)")

    print(f"\nBenchmarking HNSW search ({N_QUERIES} queries)...")
    times_hnsw = []
    for _ in range(N_QUERIES):
        start = time.perf_counter()
        repo_hnsw.search(lib_id_hnsw, query.tolist(), k=10)
        elapsed = (time.perf_counter() - start) * 1000
        times_hnsw.append(elapsed)

    hnsw_avg = np.mean(times_hnsw)
    hnsw_p95 = np.percentile(times_hnsw, 95)
    hnsw_p99 = np.percentile(times_hnsw, 99)

    print(f"\n📈 HNSW Results:")
    print(f"  • Average: {hnsw_avg:.2f}ms")
    print(f"  • P95: {hnsw_p95:.2f}ms")
    print(f"  • P99: {hnsw_p99:.2f}ms")
    print(f"  • Min: {np.min(times_hnsw):.2f}ms")
    print(f"  • Max: {np.max(times_hnsw):.2f}ms")

    # ========================================
    # Comparison
    # ========================================
    print("\n" + "=" * 60)
    print("COMPARISON: HNSW vs BRUTE FORCE @ 100K")
    print("=" * 60)

    speedup = bf_avg / hnsw_avg

    print(f"\n🏆 SPEEDUP: {speedup:.1f}x")
    print(f"\n📊 Detailed Comparison:")
    print(f"  • Brute Force avg: {bf_avg:.2f}ms")
    print(f"  • HNSW avg:        {hnsw_avg:.2f}ms")
    print(f"  • Speedup:         {speedup:.1f}x faster")

    print(f"\n⚡ Latency Reduction:")
    print(f"  • Time saved:      {bf_avg - hnsw_avg:.2f}ms per query")
    print(f"  • Reduction:       {(1 - hnsw_avg/bf_avg)*100:.1f}%")

    # Validation
    print(f"\n✅ CLAIM VALIDATION:")
    if 20 <= speedup <= 30:
        print(f"  ✓ CONFIRMED: {speedup:.1f}x speedup is within 20-30x range")
    elif speedup > 15:
        print(f"  ✓ STRONG: {speedup:.1f}x speedup validates significant performance gain")
    else:
        print(f"  ⚠ LOWER: {speedup:.1f}x speedup (expected 20-30x)")

    # Save results
    results = {
        'config': {
            'n_vectors': N_VECTORS,
            'dimensions': DIM,
            'n_queries': N_QUERIES
        },
        'brute_force': {
            'avg_ms': float(bf_avg),
            'p95_ms': float(bf_p95),
            'p99_ms': float(bf_p99),
            'min_ms': float(np.min(times_bf)),
            'max_ms': float(np.max(times_bf))
        },
        'hnsw': {
            'avg_ms': float(hnsw_avg),
            'p95_ms': float(hnsw_p95),
            'p99_ms': float(hnsw_p99),
            'min_ms': float(np.min(times_hnsw)),
            'max_ms': float(np.max(times_hnsw))
        },
        'speedup': float(speedup)
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
