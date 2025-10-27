"""
Tests for streaming and async endpoints.

Tests:
- Streaming document ingestion (NDJSON)
- Streaming search results
- Document export streaming
"""

import asyncio
import json
import pytest
import requests
from uuid import uuid4

BASE_URL = "http://localhost:8000/v1"


def test_streaming_document_ingestion():
    """Test streaming NDJSON document ingestion."""
    print("\n" + "=" * 80)
    print("TEST: Streaming Document Ingestion")
    print("=" * 80)

    # Create library
    library_response = requests.post(
        f"{BASE_URL}/libraries",
        json={
            "name": f"Streaming Test {uuid4().hex[:8]}",
            "description": "Test library for streaming ingestion",
            "index_type": "hnsw",
        },
    )
    assert library_response.status_code == 201
    library_id = library_response.json()["id"]
    print(f"✓ Created library: {library_id}")

    # Prepare NDJSON data (10 documents)
    ndjson_data = ""
    for i in range(10):
        doc = {
            "title": f"Streaming Document {i+1}",
            "texts": [
                f"This is chunk 1 of document {i+1}. It discusses machine learning and AI.",
                f"This is chunk 2 of document {i+1}. It covers deep learning and neural networks.",
                f"This is chunk 3 of document {i+1}. It explores natural language processing.",
            ],
            "tags": [f"stream-test", f"doc-{i+1}"],
            "author": "Test Author",
        }
        ndjson_data += json.dumps(doc) + "\n"

    print(f"✓ Prepared 10 documents in NDJSON format ({len(ndjson_data)} bytes)")

    # Stream documents
    print("✓ Streaming documents to server...")
    response = requests.post(
        f"{BASE_URL}/libraries/{library_id}/documents/stream",
        headers={"Content-Type": "application/x-ndjson"},
        data=ndjson_data,
        stream=True,  # Enable streaming response
    )

    assert response.status_code == 200
    print(f"✓ Server accepted stream (HTTP {response.status_code})")

    # Parse streaming response
    results = []
    for line in response.iter_lines():
        if line:
            result = json.loads(line.decode("utf-8"))
            results.append(result)

            status = result.get("status")
            if status == "processing":
                print(f"  → Processing: {result['title']}")
            elif status == "completed":
                print(f"  ✓ Completed: {result['title']} ({result['num_chunks']} chunks)")
            elif status == "error":
                print(f"  ✗ Error: {result.get('title', 'Unknown')} - {result['error']}")
            elif status == "summary":
                print(f"\n✓ Summary:")
                print(f"  Total processed: {result['total_processed']}")
                print(f"  Total succeeded: {result['total_succeeded']}")
                print(f"  Total failed: {result['total_failed']}")
                print(f"  Total chunks: {result['total_chunks']}")

    # Verify summary
    summary = results[-1]
    assert summary["status"] == "summary"
    assert summary["total_succeeded"] == 10
    assert summary["total_failed"] == 0
    assert summary["total_chunks"] == 30  # 10 docs * 3 chunks

    # Verify documents are searchable
    search_response = requests.post(
        f"{BASE_URL}/libraries/{library_id}/search",
        json={"query": "machine learning", "k": 5},
    )
    assert search_response.status_code == 200
    search_results = search_response.json()["results"]
    assert len(search_results) >= 5
    print(f"\n✓ Documents are searchable ({len(search_results)} results found)")

    print("\n✅ Streaming ingestion test PASSED")


def test_streaming_search_results():
    """Test streaming search results."""
    print("\n" + "=" * 80)
    print("TEST: Streaming Search Results")
    print("=" * 80)

    # Create library and add documents
    library_response = requests.post(
        f"{BASE_URL}/libraries",
        json={
            "name": f"Stream Search Test {uuid4().hex[:8]}",
            "index_type": "hnsw",
        },
    )
    library_id = library_response.json()["id"]
    print(f"✓ Created library: {library_id}")

    # Add 20 documents via batch
    documents = []
    for i in range(20):
        documents.append({
            "title": f"Document {i+1}",
            "texts": [
                f"Content about artificial intelligence and machine learning topic {i+1}.",
                f"Discussion of neural networks and deep learning systems {i+1}.",
            ],
            "tags": [f"ai", f"ml", f"doc-{i+1}"],
        })

    batch_response = requests.post(
        f"{BASE_URL}/libraries/{library_id}/documents/batch",
        json={"documents": documents},
    )
    assert batch_response.status_code == 200
    print(f"✓ Added 20 documents via batch")

    # Stream search results
    print("✓ Streaming search results...")
    response = requests.post(
        f"{BASE_URL}/libraries/{library_id}/search/stream?query=machine+learning&k=15",
        stream=True,
    )

    assert response.status_code == 200
    print(f"✓ Server started streaming results (HTTP {response.status_code})")

    # Parse streaming results
    results = []
    for line in response.iter_lines():
        if line:
            result = json.loads(line.decode("utf-8"))
            results.append(result)

            if "rank" in result:
                print(f"  #{result['rank']}: score={result['score']:.3f}, text={result['text'][:50]}...")
            elif result.get("status") == "complete":
                print(f"\n✓ Streaming complete: {result['total_results']} results")

    # Verify results
    search_results = [r for r in results if "rank" in r]
    assert len(search_results) == 15
    print(f"✓ Received 15 results via stream")

    # Verify results are sorted by score
    scores = [r["score"] for r in search_results]
    assert scores == sorted(scores, reverse=True)
    print(f"✓ Results are properly sorted by score")

    print("\n✅ Streaming search test PASSED")


def test_stream_all_documents():
    """Test streaming document export."""
    print("\n" + "=" * 80)
    print("TEST: Stream Document Export")
    print("=" * 80)

    # Create library and add documents
    library_response = requests.post(
        f"{BASE_URL}/libraries",
        json={
            "name": f"Export Test {uuid4().hex[:8]}",
            "index_type": "brute_force",
        },
    )
    library_id = library_response.json()["id"]
    print(f"✓ Created library: {library_id}")

    # Add 5 documents
    for i in range(5):
        requests.post(
            f"{BASE_URL}/libraries/{library_id}/documents",
            json={
                "title": f"Export Doc {i+1}",
                "texts": [f"Content {i+1}-1", f"Content {i+1}-2"],
                "tags": ["export-test"],
            },
        )
    print(f"✓ Added 5 documents")

    # Stream export
    print("✓ Streaming document export...")
    response = requests.get(
        f"{BASE_URL}/libraries/{library_id}/documents/stream",
        stream=True,
    )

    assert response.status_code == 200
    print(f"✓ Server started export stream (HTTP {response.status_code})")

    # Parse exported documents
    documents = []
    for line in response.iter_lines():
        if line:
            doc = json.loads(line.decode("utf-8"))
            if "title" in doc:  # Not summary
                documents.append(doc)
                print(f"  → Exported: {doc['title']} ({len(doc['chunks'])} chunks)")
            elif doc.get("status") == "complete":
                print(f"\n✓ Export complete: {doc['total_documents']} documents")

    # Verify export
    assert len(documents) == 5
    print(f"✓ Exported 5 documents")

    # Verify document structure
    for doc in documents:
        assert "id" in doc
        assert "title" in doc
        assert "chunks" in doc
        assert len(doc["chunks"]) == 2  # Each doc has 2 chunks
        for chunk in doc["chunks"]:
            assert "id" in chunk
            assert "text" in chunk
            assert "metadata" in chunk

    print(f"✓ Document structure is complete")
    print("\n✅ Document export test PASSED")


def test_streaming_error_handling():
    """Test error handling in streaming endpoints."""
    print("\n" + "=" * 80)
    print("TEST: Streaming Error Handling")
    print("=" * 80)

    # Create library
    library_response = requests.post(
        f"{BASE_URL}/libraries",
        json={"name": f"Error Test {uuid4().hex[:8]}", "index_type": "hnsw"},
    )
    library_id = library_response.json()["id"]
    print(f"✓ Created library: {library_id}")

    # Test 1: Invalid JSON in stream
    print("\n✓ Test 1: Invalid JSON in stream")
    invalid_ndjson = '{"title": "Valid"}\n{invalid json here}\n{"title": "Also Valid"}\n'

    response = requests.post(
        f"{BASE_URL}/libraries/{library_id}/documents/stream",
        headers={"Content-Type": "application/x-ndjson"},
        data=invalid_ndjson,
        stream=True,
    )

    errors = []
    successes = []
    for line in response.iter_lines():
        if line:
            result = json.loads(line.decode("utf-8"))
            if result.get("status") == "error":
                errors.append(result)
                print(f"  ✓ Caught error: {result['error']}")
            elif result.get("status") == "completed":
                successes.append(result)

    assert len(errors) >= 1  # At least one error for invalid JSON
    print(f"  ✓ {len(errors)} errors caught, {len(successes)} successes")

    # Test 2: Missing required fields
    print("\n✓ Test 2: Missing required fields")
    missing_fields_ndjson = json.dumps({"title": "No texts field"}) + "\n"

    response = requests.post(
        f"{BASE_URL}/libraries/{library_id}/documents/stream",
        headers={"Content-Type": "application/x-ndjson"},
        data=missing_fields_ndjson,
        stream=True,
    )

    errors = []
    for line in response.iter_lines():
        if line:
            result = json.loads(line.decode("utf-8"))
            if result.get("status") == "error":
                errors.append(result)
                print(f"  ✓ Caught error: {result.get('error', 'Unknown')}")

    assert len(errors) >= 1
    print(f"  ✓ Missing field error caught")

    print("\n✅ Error handling test PASSED")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("STREAMING ENDPOINTS TEST SUITE")
    print("=" * 80)
    print("\nMake sure the API server is running: python3 run_api.py\n")

    try:
        # Check API is available
        response = requests.get(f"{BASE_URL}/libraries")
        assert response.status_code in [200, 401]  # 401 if auth is enabled
        print("✓ API server is running\n")

        # Run tests
        test_streaming_document_ingestion()
        test_streaming_search_results()
        test_stream_all_documents()
        test_streaming_error_handling()

        print("\n" + "=" * 80)
        print("✅ ALL STREAMING TESTS PASSED")
        print("=" * 80 + "\n")

    except requests.ConnectionError:
        print("❌ ERROR: Cannot connect to API server")
        print("Please start the server with: python3 run_api.py")
        exit(1)
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
