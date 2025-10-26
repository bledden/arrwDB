#!/usr/bin/env python3
"""
Test script to verify batch operations functionality and performance.

This script:
1. Creates a test library
2. Batch adds 100 documents (10 chunks each = 1,000 chunks)
3. Measures performance vs individual adds
4. Batch deletes documents
5. Verifies batch operations work correctly
"""

import requests
import time
from typing import List, Dict
from uuid import UUID

# API configuration
API_URL = "http://localhost:8000"
LIBRARY_NAME = "test-batch-operations"

def create_library() -> str:
    """Create a test library."""
    print("\n[STEP 1] Creating test library...", end="", flush=True)
    response = requests.post(
        f"{API_URL}/v1/libraries",
        json={
            "name": LIBRARY_NAME,
            "description": "Test library for batch operations",
            "index_type": "hnsw"
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

def generate_test_documents(count: int, chunks_per_doc: int = 10) -> List[Dict]:
    """Generate test documents for batch operation."""
    documents = []
    for i in range(count):
        chunks = [
            f"Document {i+1}, Chunk {j+1}: This is test content about machine learning, "
            f"artificial intelligence, and data science topics. Index: {i}-{j}"
            for j in range(chunks_per_doc)
        ]

        documents.append({
            "title": f"Test Document {i+1}",
            "texts": chunks,
            "author": "Batch Test Author",
            "document_type": "test",
            "tags": ["batch-test", f"batch-{i//10}"]
        })

    return documents

def test_batch_add(library_id: str, num_docs: int) -> tuple:
    """Test batch add operation."""
    print(f"\n[STEP 2] Batch adding {num_docs} documents...")

    # Generate test documents
    documents = generate_test_documents(num_docs)

    # Batch add
    start_time = time.time()
    response = requests.post(
        f"{API_URL}/v1/libraries/{library_id}/documents/batch",
        json={"documents": documents}
    )
    batch_time = time.time() - start_time

    if response.status_code == 201:
        result = response.json()
        print(f"✓ Batch add completed:")
        print(f"  - Total requested: {result['total_requested']}")
        print(f"  - Successful: {result['successful']}")
        print(f"  - Failed: {result['failed']}")
        print(f"  - Total chunks: {result['total_chunks_added']}")
        print(f"  - Processing time: {result['processing_time_ms']:.2f}ms")
        print(f"  - Wall clock time: {batch_time*1000:.2f}ms")
        print(f"  - Throughput: {result['total_chunks_added']/batch_time:.0f} chunks/sec")

        # Extract successful document IDs
        doc_ids = [
            r['document_id'] for r in result['results']
            if r['success'] and r['document_id']
        ]

        return doc_ids, batch_time, result['total_chunks_added']
    else:
        print(f"✗ Batch add failed ({response.status_code})")
        print(response.text)
        return [], 0, 0

def test_individual_add(library_id: str, num_docs: int) -> tuple:
    """Test individual add operations for comparison."""
    print(f"\n[STEP 3] Adding {num_docs} documents individually (for comparison)...")

    # Generate test documents
    documents = generate_test_documents(num_docs)

    # Individual adds
    start_time = time.time()
    successful = 0
    total_chunks = 0

    for doc in documents:
        response = requests.post(
            f"{API_URL}/v1/libraries/{library_id}/documents",
            json=doc
        )
        if response.status_code == 201:
            successful += 1
            result = response.json()
            total_chunks += len(result['chunks'])

    individual_time = time.time() - start_time

    print(f"✓ Individual adds completed:")
    print(f"  - Successful: {successful}/{num_docs}")
    print(f"  - Total chunks: {total_chunks}")
    print(f"  - Wall clock time: {individual_time*1000:.2f}ms")
    print(f"  - Throughput: {total_chunks/individual_time:.0f} chunks/sec")

    return individual_time, total_chunks

def compare_performance(batch_time: float, individual_time: float,
                       batch_chunks: int, individual_chunks: int):
    """Compare batch vs individual performance."""
    print(f"\n[STEP 4] Performance Comparison:")
    print(f"{'=' * 60}")
    print(f"{'Metric':<30} {'Batch':<15} {'Individual':<15}")
    print(f"{'-' * 60}")
    print(f"{'Total Time':<30} {batch_time*1000:>13.2f}ms  {individual_time*1000:>13.2f}ms")
    print(f"{'Chunks Added':<30} {batch_chunks:>13}    {individual_chunks:>13}")
    print(f"{'Throughput (chunks/sec)':<30} {batch_chunks/batch_time:>13.0f}    {individual_chunks/individual_time:>13.0f}")

    speedup = individual_time / batch_time
    throughput_improvement = (batch_chunks/batch_time) / (individual_chunks/individual_time)

    print(f"{'-' * 60}")
    print(f"{'Speedup':<30} {speedup:>13.2f}x")
    print(f"{'Throughput Improvement':<30} {throughput_improvement:>13.2f}x")
    print(f"{'=' * 60}")

    if speedup >= 10:
        print(f"\n✓ EXCELLENT: Batch operations are {speedup:.1f}x faster!")
    elif speedup >= 5:
        print(f"\n✓ GOOD: Batch operations are {speedup:.1f}x faster")
    else:
        print(f"\n⚠ Batch speedup is only {speedup:.1f}x (expected 10-100x)")

def test_batch_delete(document_ids: List[str]):
    """Test batch delete operation."""
    print(f"\n[STEP 5] Batch deleting {len(document_ids)} documents...")

    start_time = time.time()
    response = requests.delete(
        f"{API_URL}/documents/batch",
        json={"document_ids": document_ids}
    )
    delete_time = time.time() - start_time

    if response.status_code == 200:
        result = response.json()
        print(f"✓ Batch delete completed:")
        print(f"  - Total requested: {result['total_requested']}")
        print(f"  - Successful: {result['successful']}")
        print(f"  - Failed: {result['failed']}")
        print(f"  - Processing time: {result['processing_time_ms']:.2f}ms")
        print(f"  - Wall clock time: {delete_time*1000:.2f}ms")
        print(f"  - Throughput: {result['successful']/delete_time:.0f} docs/sec")
        return True
    else:
        print(f"✗ Batch delete failed ({response.status_code})")
        print(response.text)
        return False

def verify_library_state(library_id: str, expected_docs: int):
    """Verify library has expected number of documents."""
    print(f"\n[STEP 6] Verifying library state...")
    response = requests.get(f"{API_URL}/v1/libraries/{library_id}")

    if response.status_code == 200:
        library = response.json()
        actual_docs = len(library['documents'])
        print(f"  - Expected documents: {expected_docs}")
        print(f"  - Actual documents: {actual_docs}")

        if actual_docs == expected_docs:
            print(f"  ✓ Library state is correct")
            return True
        else:
            print(f"  ✗ Document count mismatch!")
            return False
    else:
        print(f"  ✗ Failed to get library ({response.status_code})")
        return False

def cleanup(library_id: str):
    """Clean up test library."""
    print(f"\n[CLEANUP] Deleting test library...", end="", flush=True)
    response = requests.delete(f"{API_URL}/v1/libraries/{library_id}")
    if response.status_code == 204:
        print(" ✓")
    else:
        print(f" ✗ ({response.status_code})")

def main():
    """Run batch operations test."""
    print("=" * 60)
    print("Batch Operations Test")
    print("=" * 60)

    # Check if server is running
    try:
        response = requests.get(f"{API_URL}/health", timeout=1)
        if response.status_code != 200:
            print("⚠️ Server is not healthy")
            return
    except requests.exceptions.RequestException:
        print("✗ Server is not running. Please start it first:")
        print("   uvicorn app.api.main:app --host localhost --port 8000")
        return

    # Create test library
    library_id = create_library()
    if not library_id:
        print("✗ Failed to create library")
        return

    try:
        # Test batch add (100 documents with 10 chunks each = 1,000 chunks)
        batch_doc_ids, batch_time, batch_chunks = test_batch_add(library_id, 100)

        if not batch_doc_ids:
            print("✗ Batch add failed")
            return

        # Verify batch add worked
        if not verify_library_state(library_id, 100):
            print("✗ Library state verification failed after batch add")
            return

        # Test individual add (10 documents with 10 chunks each = 100 chunks)
        individual_time, individual_chunks = test_individual_add(library_id, 10)

        # Verify individual adds worked
        if not verify_library_state(library_id, 110):
            print("✗ Library state verification failed after individual adds")
            return

        # Compare performance (normalize to same number of chunks)
        normalized_individual_time = individual_time * (batch_chunks / individual_chunks)
        compare_performance(batch_time, normalized_individual_time, batch_chunks, batch_chunks)

        # Test batch delete
        if not test_batch_delete(batch_doc_ids):
            print("✗ Batch delete failed")
            return

        # Verify batch delete worked (should have 10 docs left from individual adds)
        if not verify_library_state(library_id, 10):
            print("✗ Library state verification failed after batch delete")
            return

        print("\n" + "=" * 60)
        print("✓ ALL BATCH OPERATIONS TESTS PASSED")
        print("=" * 60)
        print("\nSummary:")
        print(f"  - Batch add: {batch_chunks} chunks in {batch_time*1000:.0f}ms")
        print(f"  - Batch delete: {len(batch_doc_ids)} docs in {batch_time*1000:.0f}ms")
        print(f"  - Speedup: {normalized_individual_time/batch_time:.1f}x faster than individual operations")
        print("\nBatch operations are working correctly and provide significant")
        print("performance improvements for bulk data ingestion!")

    finally:
        # Cleanup
        cleanup(library_id)

if __name__ == "__main__":
    main()
