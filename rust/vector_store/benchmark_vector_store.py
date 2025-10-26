"""
Benchmark comparison: Python VectorStore vs Rust VectorStore

This benchmark tests the core VectorStore operations:
1. Adding vectors (with deduplication)
2. Retrieving individual vectors
3. Batch vector retrieval
4. Removing vectors
"""

import time
import numpy as np
from uuid import uuid4
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.vector_store import VectorStore as PythonVectorStore
import rust_vector_store

def generate_vectors(n, dimension):
    """Generate n random normalized vectors."""
    vectors = np.random.rand(n, dimension).astype(np.float32)
    # Normalize
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms

def benchmark_python(n_vectors, dimension, batch_size):
    """Benchmark Python VectorStore."""
    print(f"\n{'='*60}")
    print("PYTHON VectorStore")
    print(f"{'='*60}")

    vectors = generate_vectors(n_vectors, dimension)
    chunk_ids = [uuid4() for _ in range(n_vectors)]

    # Create store
    store = PythonVectorStore(dimension=dimension, initial_capacity=n_vectors)

    # Benchmark: Add vectors
    start = time.time()
    indices = []
    for chunk_id, vector in zip(chunk_ids, vectors):
        idx = store.add_vector(chunk_id, vector)
        indices.append(idx)
    add_time = time.time() - start
    print(f"Add {n_vectors} vectors: {add_time:.4f}s ({n_vectors/add_time:.0f} ops/sec)")

    # Add duplicates to test deduplication
    start = time.time()
    dup_ids = [uuid4() for _ in range(100)]
    for dup_id in dup_ids:
        store.add_vector(dup_id, vectors[0])  # Same vector
    dup_time = time.time() - start
    print(f"Add 100 duplicates: {dup_time:.4f}s (deduplication)")
    print(f"  Unique vectors: {store.count}, Total refs: {store.total_references}")

    # Benchmark: Get individual vectors
    start = time.time()
    for chunk_id in chunk_ids[:1000]:
        _ = store.get_vector(chunk_id)
    get_time = time.time() - start
    print(f"Get 1000 individual vectors: {get_time:.4f}s ({1000/get_time:.0f} ops/sec)")

    # Benchmark: Batch retrieval
    batch_indices = indices[:batch_size]
    start = time.time()
    for _ in range(100):
        _ = store.get_vectors_by_indices(batch_indices)
    batch_time = time.time() - start
    print(f"Batch get {batch_size} vectors (100x): {batch_time:.4f}s")

    # Benchmark: Remove vectors
    remove_ids = chunk_ids[:1000]
    start = time.time()
    for chunk_id in remove_ids:
        store.remove_vector(chunk_id)
    remove_time = time.time() - start
    print(f"Remove 1000 vectors: {remove_time:.4f}s ({1000/remove_time:.0f} ops/sec)")

    return {
        'add': add_time,
        'dup': dup_time,
        'get': get_time,
        'batch': batch_time,
        'remove': remove_time,
    }

def benchmark_rust(n_vectors, dimension, batch_size):
    """Benchmark Rust VectorStore."""
    print(f"\n{'='*60}")
    print("RUST VectorStore")
    print(f"{'='*60}")

    vectors = generate_vectors(n_vectors, dimension)
    chunk_ids = [str(uuid4()) for _ in range(n_vectors)]

    # Create store
    store = rust_vector_store.RustVectorStore(dimension=dimension, initial_capacity=n_vectors)

    # Benchmark: Add vectors
    start = time.time()
    indices = []
    for chunk_id, vector in zip(chunk_ids, vectors):
        idx = store.add_vector(chunk_id, vector)
        indices.append(idx)
    add_time = time.time() - start
    print(f"Add {n_vectors} vectors: {add_time:.4f}s ({n_vectors/add_time:.0f} ops/sec)")

    # Add duplicates to test deduplication
    start = time.time()
    dup_ids = [str(uuid4()) for _ in range(100)]
    for dup_id in dup_ids:
        store.add_vector(dup_id, vectors[0])  # Same vector
    dup_time = time.time() - start
    stats = store.get_statistics()
    print(f"Add 100 duplicates: {dup_time:.4f}s (deduplication)")
    print(f"  Unique vectors: {stats['unique_vectors']}, Total refs: {stats['total_references']}")

    # Benchmark: Get individual vectors
    start = time.time()
    for chunk_id in chunk_ids[:1000]:
        _ = store.get_vector(chunk_id)
    get_time = time.time() - start
    print(f"Get 1000 individual vectors: {get_time:.4f}s ({1000/get_time:.0f} ops/sec)")

    # Benchmark: Batch retrieval
    batch_indices = indices[:batch_size]
    start = time.time()
    for _ in range(100):
        _ = store.get_vectors_by_indices(batch_indices)
    batch_time = time.time() - start
    print(f"Batch get {batch_size} vectors (100x): {batch_time:.4f}s")

    # Benchmark: Remove vectors
    remove_ids = chunk_ids[:1000]
    start = time.time()
    for chunk_id in remove_ids:
        store.remove_vector(chunk_id)
    remove_time = time.time() - start
    print(f"Remove 1000 vectors: {remove_time:.4f}s ({1000/remove_time:.0f} ops/sec)")

    return {
        'add': add_time,
        'dup': dup_time,
        'get': get_time,
        'batch': batch_time,
        'remove': remove_time,
    }

def main():
    print("="*60)
    print("VectorStore Benchmark: Python vs Rust")
    print("="*60)
    print(f"Configuration:")
    print(f"  Vectors: 10,000")
    print(f"  Dimension: 384")
    print(f"  Batch size: 1000")

    n_vectors = 10_000
    dimension = 384
    batch_size = 1000

    # Run benchmarks
    py_results = benchmark_python(n_vectors, dimension, batch_size)
    rust_results = benchmark_rust(n_vectors, dimension, batch_size)

    # Print comparison
    print(f"\n{'='*60}")
    print("PERFORMANCE COMPARISON")
    print(f"{'='*60}")
    print(f"{'Operation':<30} {'Python':>12} {'Rust':>12} {'Speedup':>10}")
    print(f"{'-'*60}")

    for op_name, py_time in py_results.items():
        rust_time = rust_results[op_name]
        speedup = py_time / rust_time
        op_display = {
            'add': 'Add 10K vectors',
            'dup': 'Add 100 duplicates',
            'get': 'Get 1K individual vectors',
            'batch': 'Batch get 1K (100x)',
            'remove': 'Remove 1K vectors',
        }[op_name]

        print(f"{op_display:<30} {py_time:>10.4f}s {rust_time:>10.4f}s {speedup:>9.2f}x")

    # Calculate overall speedup
    total_py = sum(py_results.values())
    total_rust = sum(rust_results.values())
    overall_speedup = total_py / total_rust

    print(f"{'-'*60}")
    print(f"{'TOTAL':<30} {total_py:>10.4f}s {total_rust:>10.4f}s {overall_speedup:>9.2f}x")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Rust VectorStore is {overall_speedup:.2f}x faster overall")

    # Find best and worst speedups
    speedups = {k: py_results[k]/rust_results[k] for k in py_results.keys()}
    best = max(speedups, key=speedups.get)
    worst = min(speedups, key=speedups.get)

    best_name = {
        'add': 'Adding vectors',
        'dup': 'Deduplication',
        'get': 'Individual retrieval',
        'batch': 'Batch retrieval',
        'remove': 'Removing vectors',
    }[best]

    worst_name = {
        'add': 'Adding vectors',
        'dup': 'Deduplication',
        'get': 'Individual retrieval',
        'batch': 'Batch retrieval',
        'remove': 'Removing vectors',
    }[worst]

    print(f"Best speedup: {best_name} ({speedups[best]:.2f}x)")
    print(f"Worst speedup: {worst_name} ({speedups[worst]:.2f}x)")

if __name__ == "__main__":
    main()
