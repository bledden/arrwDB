#!/usr/bin/env python3
"""
Comprehensive validation tests for Streaming & Real-Time Features (Phases 1-2).

This test suite validates:
1. Streaming document ingestion (NDJSON)
2. Streaming search results
3. Document export streaming
4. Server-Sent Events (SSE) for real-time notifications
5. Event bus statistics

Tests use small data to avoid slow Cohere API calls.
"""

import json
import requests
import time
import threading
from uuid import uuid4
import sys

BASE_URL = "http://localhost:8000/v1"

# Colors for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"


def print_header(text):
    """Print a section header."""
    print(f"\n{BLUE}{'=' * 80}{RESET}")
    print(f"{BLUE}{text.center(80)}{RESET}")
    print(f"{BLUE}{'=' * 80}{RESET}\n")


def print_success(text):
    """Print success message."""
    print(f"{GREEN}✓ {text}{RESET}")


def print_error(text):
    """Print error message."""
    print(f"{RED}✗ {text}{RESET}")


def print_info(text):
    """Print info message."""
    print(f"{YELLOW}→ {text}{RESET}")


def check_api_health():
    """Check if API is running and healthy."""
    print_header("API Health Check")
    try:
        response = requests.get(f"{BASE_URL}/../health", timeout=5)
        if response.status_code == 200:
            print_success("API server is running and healthy")
            health = response.json()
            print_info(f"Status: {health['status']}")
            print_info(f"Version: {health['version']}")
            return True
        else:
            print_error(f"API returned status {response.status_code}")
            return False
    except requests.ConnectionError:
        print_error("Cannot connect to API server")
        print_info("Start the server with: python3 run_api.py")
        return False


def test_streaming_ingestion():
    """Test Phase 1: Streaming document ingestion."""
    print_header("Phase 1 Test: Streaming Document Ingestion")

    # Create test library
    print_info("Creating test library...")
    lib_response = requests.post(
        f"{BASE_URL}/libraries",
        json={
            "name": f"Streaming Test {uuid4().hex[:8]}",
            "description": "Test library for streaming validation",
            "index_type": "brute_force",  # Fastest for testing
        }
    )

    if lib_response.status_code != 201:
        print_error(f"Failed to create library: {lib_response.status_code}")
        print_error(lib_response.text)
        return False

    library_id = lib_response.json()["id"]
    print_success(f"Created library: {library_id}")

    # Prepare small NDJSON data (3 docs with 1 chunk each = fast embeddings)
    print_info("Preparing NDJSON test data (3 small documents)...")
    docs = []
    for i in range(3):
        docs.append({
            "title": f"Stream Doc {i+1}",
            "texts": [f"Test content {i+1}"],  # Single short chunk
            "tags": [f"stream-test-{i+1}"]
        })

    ndjson_data = "\n".join(json.dumps(doc) for doc in docs) + "\n"
    print_success(f"Prepared {len(docs)} documents ({len(ndjson_data)} bytes)")

    # Stream documents
    print_info("Streaming documents to server...")
    start_time = time.time()

    response = requests.post(
        f"{BASE_URL}/libraries/{library_id}/documents/stream",
        headers={"Content-Type": "application/x-ndjson"},
        data=ndjson_data,
        stream=True,
        timeout=60,
    )

    if response.status_code != 200:
        print_error(f"Streaming failed with status {response.status_code}")
        print_error(response.text)
        return False

    print_success("Server accepted stream")

    # Parse streaming response
    results = []
    completed_count = 0
    error_count = 0

    print_info("Processing streaming response...")
    for line in response.iter_lines():
        if line:
            result = json.loads(line.decode("utf-8"))
            results.append(result)

            status = result.get("status")
            if status == "processing":
                print_info(f"  Processing: {result['title']}")
            elif status == "completed":
                completed_count += 1
                print_success(f"  Completed: {result['title']} (ID: {result['id'][:8]}...)")
            elif status == "error":
                error_count += 1
                print_error(f"  Error: {result.get('title', 'Unknown')} - {result['error']}")
            elif status == "summary":
                elapsed = time.time() - start_time
                print_success(f"\nSummary:")
                print_info(f"  Total processed: {result['total_processed']}")
                print_info(f"  Total succeeded: {result['total_succeeded']}")
                print_info(f"  Total failed: {result['total_failed']}")
                print_info(f"  Total chunks: {result['total_chunks']}")
                print_info(f"  Time: {elapsed:.2f}s")

    # Verify summary
    summary = results[-1]
    if summary["status"] != "summary":
        print_error("Missing summary in response")
        return False

    if summary["total_succeeded"] != 3:
        print_error(f"Expected 3 successes, got {summary['total_succeeded']}")
        return False

    if summary["total_failed"] != 0:
        print_error(f"Expected 0 failures, got {summary['total_failed']}")
        return False

    print_success("Streaming ingestion validation PASSED")

    # Verify documents are searchable
    print_info("\nVerifying documents are searchable...")
    search_response = requests.post(
        f"{BASE_URL}/libraries/{library_id}/search",
        json={"query": "test content", "k": 3}
    )

    if search_response.status_code != 200:
        print_error("Search failed")
        return False

    results = search_response.json()["results"]
    if len(results) < 3:
        print_error(f"Expected at least 3 results, got {len(results)}")
        return False

    print_success(f"Documents are searchable ({len(results)} results found)")

    return library_id  # Return for use in other tests


def test_streaming_search(library_id):
    """Test Phase 1: Streaming search results."""
    print_header("Phase 1 Test: Streaming Search Results")

    print_info("Streaming search results...")
    response = requests.post(
        f"{BASE_URL}/libraries/{library_id}/search/stream",
        params={"query": "test content", "k": 3},
        stream=True,
        timeout=30,
    )

    if response.status_code != 200:
        print_error(f"Streaming search failed: {response.status_code}")
        return False

    print_success("Server started streaming results")

    # Parse results
    results = []
    for line in response.iter_lines():
        if line:
            result = json.loads(line.decode("utf-8"))
            results.append(result)

            if "rank" in result:
                print_info(f"  Rank {result['rank']}: score={result['score']:.3f}, text={result['text'][:40]}...")
            elif result.get("status") == "complete":
                print_success(f"Stream complete: {result['total_results']} results")

    # Verify results
    search_results = [r for r in results if "rank" in r]
    if len(search_results) < 3:
        print_error(f"Expected at least 3 results, got {len(search_results)}")
        return False

    # Verify sorted by score
    scores = [r["score"] for r in search_results]
    if scores != sorted(scores, reverse=True):
        print_error("Results not sorted by score")
        return False

    print_success("Streaming search validation PASSED")
    return True


def test_document_export(library_id):
    """Test Phase 1: Document export streaming."""
    print_header("Phase 1 Test: Document Export Streaming")

    print_info("Streaming document export...")
    response = requests.get(
        f"{BASE_URL}/libraries/{library_id}/documents/stream",
        stream=True,
        timeout=30,
    )

    if response.status_code != 200:
        print_error(f"Export streaming failed: {response.status_code}")
        return False

    print_success("Server started export stream")

    # Parse exported documents
    documents = []
    for line in response.iter_lines():
        if line:
            doc = json.loads(line.decode("utf-8"))
            if "title" in doc:  # Not summary
                documents.append(doc)
                print_info(f"  Exported: {doc['title']} ({len(doc['chunks'])} chunks)")
            elif doc.get("status") == "complete":
                print_success(f"Export complete: {doc['total_documents']} documents")

    # Verify export
    if len(documents) != 3:
        print_error(f"Expected 3 documents, got {len(documents)}")
        return False

    # Verify document structure
    for doc in documents:
        if not all(k in doc for k in ["id", "title", "chunks"]):
            print_error("Document missing required fields")
            return False

        for chunk in doc["chunks"]:
            if not all(k in chunk for k in ["id", "text", "metadata"]):
                print_error("Chunk missing required fields")
                return False

    print_success("Document export validation PASSED")
    return True


def test_event_bus_statistics():
    """Test Phase 2: Event bus statistics endpoint."""
    print_header("Phase 2 Test: Event Bus Statistics")

    print_info("Fetching event bus statistics...")
    response = requests.get(f"{BASE_URL}/events/statistics", timeout=10)

    if response.status_code != 200:
        print_error(f"Statistics endpoint failed: {response.status_code}")
        return False

    stats = response.json()
    print_success("Event bus statistics retrieved")

    # Display statistics
    print_info(f"  Total published: {stats.get('total_published', 0)}")
    print_info(f"  Total delivered: {stats.get('total_delivered', 0)}")
    print_info(f"  Total errors: {stats.get('total_errors', 0)}")
    print_info(f"  Pending events: {stats.get('pending_events', 0)}")
    print_info(f"  Subscriber count: {stats.get('subscriber_count', 0)}")
    print_info(f"  Running: {stats.get('running', False)}")

    # Verify structure
    required_keys = ["total_published", "total_delivered", "total_errors",
                     "pending_events", "subscriber_count", "running"]

    for key in required_keys:
        if key not in stats:
            print_error(f"Missing statistic: {key}")
            return False

    print_success("Event bus statistics validation PASSED")
    return True


def test_sse_events():
    """Test Phase 2: Server-Sent Events (SSE) for real-time notifications."""
    print_header("Phase 2 Test: Server-Sent Events (SSE)")

    print_info("This test will:")
    print_info("  1. Subscribe to SSE event stream in background thread")
    print_info("  2. Create a new library (triggers library.created event)")
    print_info("  3. Add a document (triggers document.added event)")
    print_info("  4. Verify events were received")

    # Shared state for event collection
    events_received = []
    sse_error = None

    def sse_listener():
        """Background thread to listen for SSE events."""
        nonlocal sse_error
        try:
            print_info("  SSE listener starting...")
            response = requests.get(
                f"{BASE_URL}/events/stream",
                stream=True,
                timeout=30,
            )

            if response.status_code != 200:
                sse_error = f"SSE connection failed: {response.status_code}"
                return

            print_success("  SSE listener connected")

            # Read events (with timeout)
            start = time.time()
            for line in response.iter_lines():
                if time.time() - start > 20:  # 20 second timeout
                    break

                if line:
                    line_str = line.decode("utf-8")
                    # Parse SSE format
                    if line_str.startswith("event:"):
                        event_type = line_str.split(":", 1)[1].strip()
                    elif line_str.startswith("data:"):
                        data_str = line_str.split(":", 1)[1].strip()
                        try:
                            data = json.loads(data_str)
                            events_received.append({
                                "event": event_type,
                                "data": data
                            })
                            print_info(f"  Received event: {event_type}")
                        except json.JSONDecodeError:
                            pass

        except Exception as e:
            sse_error = str(e)

    # Start SSE listener in background
    listener_thread = threading.Thread(target=sse_listener, daemon=True)
    listener_thread.start()

    # Give listener time to connect
    time.sleep(2)

    if sse_error:
        print_error(f"SSE listener error: {sse_error}")
        return False

    # Create library (should trigger library.created event)
    print_info("\nCreating library to trigger event...")
    lib_response = requests.post(
        f"{BASE_URL}/libraries",
        json={
            "name": f"SSE Test {uuid4().hex[:8]}",
            "index_type": "brute_force",
        }
    )

    if lib_response.status_code != 201:
        print_error("Failed to create library")
        return False

    library_id = lib_response.json()["id"]
    print_success(f"Created library: {library_id}")

    # Wait for event
    time.sleep(2)

    # Add document (should trigger document.added event)
    print_info("Adding document to trigger event...")
    doc_response = requests.post(
        f"{BASE_URL}/libraries/{library_id}/documents",
        json={
            "title": "SSE Test Doc",
            "texts": ["SSE test content"]
        }
    )

    if doc_response.status_code != 200:
        print_error("Failed to add document")
        return False

    print_success("Added document")

    # Wait for events to be received
    time.sleep(2)

    # Verify events
    print_info(f"\nReceived {len(events_received)} events total")

    # Check for library.created event
    library_events = [e for e in events_received if e["event"] == "library.created"]
    if not library_events:
        print_error("Did not receive library.created event")
        return False

    print_success("Received library.created event")
    print_info(f"  Library: {library_events[0]['data'].get('name', 'Unknown')}")

    # Check for document.added event
    document_events = [e for e in events_received if e["event"] == "document.added"]
    if not document_events:
        print_error("Did not receive document.added event")
        return False

    print_success("Received document.added event")
    print_info(f"  Document: {document_events[0]['data'].get('title', 'Unknown')}")

    print_success("SSE event streaming validation PASSED")
    return True


def main():
    """Run all validation tests."""
    print_header("STREAMING & REAL-TIME FEATURES VALIDATION")
    print_info("Testing Phases 1-2 Implementation")
    print_info("This will validate all streaming and event features\n")

    # Check API health
    if not check_api_health():
        print_error("\nAPI server is not running. Please start it with:")
        print_info("  python3 run_api.py")
        sys.exit(1)

    # Track results
    results = {}

    # Phase 1 Tests
    library_id = test_streaming_ingestion()
    results["Streaming Ingestion"] = bool(library_id)

    if library_id:
        results["Streaming Search"] = test_streaming_search(library_id)
        results["Document Export"] = test_document_export(library_id)
    else:
        results["Streaming Search"] = False
        results["Document Export"] = False

    # Phase 2 Tests
    results["Event Bus Statistics"] = test_event_bus_statistics()
    results["SSE Events"] = test_sse_events()

    # Summary
    print_header("VALIDATION SUMMARY")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, passed_test in results.items():
        if passed_test:
            print_success(f"{test_name}: PASSED")
        else:
            print_error(f"{test_name}: FAILED")

    print(f"\n{BLUE}{'=' * 80}{RESET}")
    if passed == total:
        print(f"{GREEN}✓ ALL TESTS PASSED ({passed}/{total}){RESET}")
        print(f"{BLUE}{'=' * 80}{RESET}\n")
        print_success("Streaming & Real-Time features are working correctly!")
        sys.exit(0)
    else:
        print(f"{YELLOW}⚠ SOME TESTS FAILED ({passed}/{total} passed){RESET}")
        print(f"{BLUE}{'=' * 80}{RESET}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
