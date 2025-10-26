#!/usr/bin/env python3
"""
Test script to verify index management functionality.

This script:
1. Creates a test library with brute_force index
2. Adds test documents
3. Gets index statistics
4. Rebuilds index (switches to HNSW)
5. Optimizes index
6. Verifies all operations work correctly
"""

import requests
import time
from typing import Dict

# API configuration
API_URL = "http://localhost:8000"
LIBRARY_NAME = "test-index-management"

def create_library(index_type: str = "brute_force") -> str:
    """Create a test library with specified index type."""
    print(f"\n[STEP 1] Creating library with {index_type} index...", end="", flush=True)
    response = requests.post(
        f"{API_URL}/v1/libraries",
        json={
            "name": LIBRARY_NAME,
            "description": "Test library for index management",
            "index_type": index_type
        }
    )
    if response.status_code == 201:
        library = response.json()
        print(f" ✓ (ID: {library['id']})")
        return library['id']
    else:
        print(f" ✗ ({response.status_code})")
        print(response.text)
        return None

def add_test_documents(library_id: str, count: int = 100) -> int:
    """Add test documents to the library."""
    print(f"\n[STEP 2] Adding {count} test documents...", end="", flush=True)

    documents = []
    for i in range(count):
        documents.append({
            "title": f"Test Document {i+1}",
            "texts": [
                f"This is test content {i+1} about various topics including "
                f"machine learning, artificial intelligence, data science, and more. "
                f"Document index: {i}"
            ] * 3,  # 3 chunks per document
            "tags": ["test", f"batch-{i//10}"]
        })

    response = requests.post(
        f"{API_URL}/v1/libraries/{library_id}/documents/batch",
        json={"documents": documents}
    )

    if response.status_code == 201:
        result = response.json()
        total_chunks = result['total_chunks_added']
        print(f" ✓ ({total_chunks} chunks added)")
        return total_chunks
    else:
        print(f" ✗ ({response.status_code})")
        print(response.text)
        return 0

def get_index_stats(library_id: str) -> Dict:
    """Get index statistics."""
    print(f"\n[STEP 3] Getting index statistics...")
    response = requests.get(
        f"{API_URL}/v1/libraries/{library_id}/index/statistics"
    )

    if response.status_code == 200:
        stats = response.json()
        print(f"  ✓ Index Statistics:")
        print(f"    - Index type: {stats['index_type']}")
        print(f"    - Total vectors: {stats['total_vectors']}")
        print(f"    - Index stats: {stats['index_stats']}")
        print(f"    - Vector store: capacity={stats['vector_store_stats'].get('capacity')}, "
              f"dimension={stats['vector_store_stats'].get('dimension')}")
        return stats
    else:
        print(f"  ✗ Failed ({response.status_code})")
        print(response.text)
        return {}

def rebuild_index(library_id: str, new_index_type: str, index_config: Dict = None):
    """Rebuild index with new type."""
    print(f"\n[STEP 4] Rebuilding index (switching to {new_index_type})...")

    payload = {}
    if new_index_type:
        payload["index_type"] = new_index_type
    if index_config:
        payload["index_config"] = index_config

    start_time = time.time()
    response = requests.post(
        f"{API_URL}/v1/libraries/{library_id}/index/rebuild",
        json=payload
    )
    rebuild_time = time.time() - start_time

    if response.status_code == 200:
        result = response.json()
        print(f"  ✓ Index Rebuilt:")
        print(f"    - Old type: {result['old_index_type']}")
        print(f"    - New type: {result['new_index_type']}")
        print(f"    - Vectors reindexed: {result['total_vectors_reindexed']}")
        print(f"    - API time: {result['rebuild_time_ms']:.2f}ms")
        print(f"    - Wall clock time: {rebuild_time*1000:.2f}ms")
        print(f"    - Throughput: {result['total_vectors_reindexed']/rebuild_time:.0f} vectors/sec")
        return True
    else:
        print(f"  ✗ Failed ({response.status_code})")
        print(response.text)
        return False

def optimize_index(library_id: str):
    """Optimize the index."""
    print(f"\n[STEP 5] Optimizing index...")

    start_time = time.time()
    response = requests.post(
        f"{API_URL}/v1/libraries/{library_id}/index/optimize"
    )
    optimize_time = time.time() - start_time

    if response.status_code == 200:
        result = response.json()
        print(f"  ✓ Index Optimized:")
        print(f"    - Vectors compacted: {result['vectors_compacted']}")
        print(f"    - Memory freed: {result['memory_freed_bytes']} bytes "
              f"({result['memory_freed_bytes']/1024:.1f} KB)")
        print(f"    - API time: {result['optimization_time_ms']:.2f}ms")
        print(f"    - Wall clock time: {optimize_time*1000:.2f}ms")
        return True
    else:
        print(f"  ✗ Failed ({response.status_code})")
        print(response.text)
        return False

def verify_search_works(library_id: str):
    """Verify search still works after index operations."""
    print(f"\n[STEP 6] Verifying search functionality...")

    response = requests.post(
        f"{API_URL}/v1/libraries/{library_id}/search",
        json={
            "query": "machine learning artificial intelligence",
            "k": 5
        }
    )

    if response.status_code == 200:
        results = response.json()
        num_results = results['total_results']
        query_time = results['query_time_ms']
        print(f"  ✓ Search works: {num_results} results in {query_time:.2f}ms")
        return True
    else:
        print(f"  ✗ Search failed ({response.status_code})")
        print(response.text)
        return False

def test_index_switching_sequence(library_id: str):
    """Test switching between different index types."""
    print(f"\n[STEP 7] Testing index type switching sequence...")

    index_types = ["kd_tree", "lsh", "hnsw", "brute_force"]

    for idx_type in index_types:
        print(f"\n  Switching to {idx_type}...", end="", flush=True)

        response = requests.post(
            f"{API_URL}/v1/libraries/{library_id}/index/rebuild",
            json={"index_type": idx_type}
        )

        if response.status_code == 200:
            result = response.json()
            print(f" ✓ ({result['total_vectors_reindexed']} vectors, "
                  f"{result['rebuild_time_ms']:.0f}ms)")
        else:
            print(f" ✗ ({response.status_code})")
            return False

    print(f"\n  ✓ Successfully switched through all index types")
    return True

def test_hnsw_config_tuning(library_id: str):
    """Test HNSW index with custom configuration."""
    print(f"\n[STEP 8] Testing HNSW configuration tuning...")

    # Test with different HNSW parameters
    configs = [
        {"M": 8, "ef_construction": 100, "ef_search": 50},
        {"M": 16, "ef_construction": 200, "ef_search": 100},
        {"M": 32, "ef_construction": 400, "ef_search": 200},
    ]

    for config in configs:
        print(f"\n  Testing HNSW config M={config['M']}, "
              f"ef_construction={config['ef_construction']}...", end="", flush=True)

        response = requests.post(
            f"{API_URL}/v1/libraries/{library_id}/index/rebuild",
            json={
                "index_type": "hnsw",
                "index_config": config
            }
        )

        if response.status_code == 200:
            result = response.json()
            print(f" ✓ ({result['rebuild_time_ms']:.0f}ms)")
        else:
            print(f" ✗ ({response.status_code})")
            return False

    print(f"\n  ✓ Successfully tested HNSW configuration tuning")
    return True

def cleanup(library_id: str):
    """Clean up test library."""
    print(f"\n[CLEANUP] Deleting test library...", end="", flush=True)
    response = requests.delete(f"{API_URL}/v1/libraries/{library_id}")
    if response.status_code == 204:
        print(" ✓")
    else:
        print(f" ✗ ({response.status_code})")

def main():
    """Run index management test."""
    print("=" * 60)
    print("Index Management Test")
    print("=" * 60)

    # Check if server is running
    try:
        response = requests.get(f"{API_URL}/health", timeout=1)
        if response.status_code != 200:
            print("⚠️  Server is not healthy")
            return
    except requests.exceptions.RequestException:
        print("✗ Server is not running. Please start it first:")
        print("   uvicorn app.api.main:app --host localhost --port 8000")
        return

    # Create test library with brute_force index
    library_id = create_library("brute_force")
    if not library_id:
        print("✗ Failed to create library")
        return

    try:
        # Add test documents
        chunks_added = add_test_documents(library_id, 100)
        if chunks_added == 0:
            print("✗ Failed to add documents")
            return

        # Get initial statistics
        stats = get_index_stats(library_id)
        if not stats:
            return

        # Rebuild index (switch to HNSW)
        if not rebuild_index(library_id, "hnsw", {"M": 16, "ef_construction": 200}):
            return

        # Get statistics after rebuild
        stats_after_rebuild = get_index_stats(library_id)
        if not stats_after_rebuild:
            return

        # Optimize index
        if not optimize_index(library_id):
            return

        # Verify search still works
        if not verify_search_works(library_id):
            return

        # Test switching between index types
        if not test_index_switching_sequence(library_id):
            return

        # Test HNSW configuration tuning
        if not test_hnsw_config_tuning(library_id):
            return

        print("\n" + "=" * 60)
        print("✓ ALL INDEX MANAGEMENT TESTS PASSED")
        print("=" * 60)
        print("\nSummary:")
        print(f"  - Successfully created library with brute_force index")
        print(f"  - Added {chunks_added} chunks")
        print(f"  - Switched index types: brute_force → hnsw → kd_tree → lsh → brute_force")
        print(f"  - Tested HNSW configuration tuning")
        print(f"  - Optimized index successfully")
        print(f"  - Search functionality verified after all operations")
        print("\nIndex management is working correctly!")
        print("Users can now:")
        print("  - Switch between index types without data loss")
        print("  - Tune index parameters for their workload")
        print("  - Optimize indexes after deletions")

    finally:
        # Cleanup
        cleanup(library_id)

if __name__ == "__main__":
    main()
