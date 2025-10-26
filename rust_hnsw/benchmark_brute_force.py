#!/usr/bin/env python3
"""Benchmark BruteForce: Python vs Rust"""

import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from infrastructure.indexes.brute_force import BruteForceIndex
from core.vector_store import VectorStore
import rust_hnsw


def generate_vectors(n, dim):
    vecs = np.random.randn(n, dim).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def benchmark_python(vectors, n_queries=1000, k=10):
    print("\n" + "="*60)
    print("PYTHON BRUTEFORCE")
    print("="*60)
    n, dim = vectors.shape

    vector_store = VectorStore(dimension=dim)
    idx = BruteForceIndex(vector_store)

    print(f"\nAdding {n} vectors...")
    start = time.time()
    for i in range(n):
        vid = f"vec-{i}"
        vidx = vector_store.add_vector(vid, vectors[i])
        idx.add_vector(vid, vidx)
    add_time = time.time() - start
    print(f"Add time: {add_time:.3f}s ({n/add_time:.1f} vectors/sec)")

    queries = generate_vectors(n_queries, dim)
    print(f"\nSearching {n_queries} queries (k={k})...")
    start = time.time()
    for q in queries:
        idx.search(q, k=k)
    search_time = time.time() - start
    print(f"Search time: {search_time:.3f}s ({n_queries/search_time:.1f} qps)")

    return {'add': add_time, 'search': search_time, 'qps': n_queries/search_time}


def benchmark_rust(vectors, n_queries=1000, k=10):
    print("\n" + "="*60)
    print("RUST BRUTEFORCE")
    print("="*60)
    n, dim = vectors.shape

    idx = rust_hnsw.RustBruteForceIndex(dimension=dim)

    print(f"\nAdding {n} vectors...")
    start = time.time()
    for i in range(n):
        idx.add_vector(f"vec-{i}", vectors[i])
    add_time = time.time() - start
    print(f"Add time: {add_time:.3f}s ({n/add_time:.1f} vectors/sec)")

    queries = generate_vectors(n_queries, dim)
    print(f"\nSearching {n_queries} queries (k={k})...")
    start = time.time()
    for q in queries:
        idx.search(q, k=k)
    search_time = time.time() - start
    print(f"Search time: {search_time:.3f}s ({n_queries/search_time:.1f} qps)")

    return {'add': add_time, 'search': search_time, 'qps': n_queries/search_time}


print("="*60)
print("BRUTEFORCE BENCHMARK")
print("="*60)

n_vectors = 10000
dim = 384
vectors = generate_vectors(n_vectors, dim)

py_results = benchmark_python(vectors)
rust_results = benchmark_rust(vectors)

print("\n" + "="*60)
print("COMPARISON")
print("="*60)
print(f"\nAdd speedup: {py_results['add']/rust_results['add']:.2f}x")
print(f"Search speedup: {py_results['search']/rust_results['search']:.2f}x")
print(f"QPS improvement: {(rust_results['qps']/py_results['qps']-1)*100:.1f}%")
print("\n" + "="*60)
