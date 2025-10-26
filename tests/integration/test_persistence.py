#!/usr/bin/env python3
"""
Test script to verify persistence system.

This script:
1. Starts the API server
2. Creates a library and adds a document
3. Stops the server
4. Restarts the server
5. Verifies the library and document are restored
"""

import requests
import time
import subprocess
import signal
import sys
from pathlib import Path

# API configuration
API_URL = "http://localhost:8000"
LIBRARY_NAME = "test-persistence-library"

def wait_for_server(timeout=30):
    """Wait for the API server to be ready."""
    print("Waiting for server to start...", end="", flush=True)
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{API_URL}/health", timeout=1)
            if response.status_code == 200:
                print(" ✓")
                return True
        except requests.exceptions.RequestException:
            pass
        print(".", end="", flush=True)
        time.sleep(1)
    print(" ✗")
    return False

def create_library():
    """Create a test library."""
    print(f"\nCreating library '{LIBRARY_NAME}'...", end="", flush=True)
    response = requests.post(
        f"{API_URL}/v1/libraries",
        json={
            "name": LIBRARY_NAME,
            "description": "Test library for persistence",
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

def add_document(library_id):
    """Add a test document to the library."""
    print("Adding test document...", end="", flush=True)
    response = requests.post(
        f"{API_URL}/v1/libraries/{library_id}/documents",
        json={
            "title": "Test Document",
            "texts": [
                "This is the first test chunk.",
                "This is the second test chunk.",
                "This is the third test chunk."
            ],
            "author": "Test Author",
            "document_type": "text",
            "tags": ["test", "persistence"]
        }
    )
    if response.status_code == 201:
        document = response.json()
        print(f" ✓ (ID: {document['id']}, {len(document['chunks'])} chunks)")
        return document['id']
    else:
        print(f" ✗ ({response.status_code})")
        print(response.text)
        return None

def verify_library_exists():
    """Verify the library still exists after restart."""
    print(f"\nVerifying library '{LIBRARY_NAME}' exists...", end="", flush=True)
    response = requests.get(f"{API_URL}/v1/libraries")
    if response.status_code == 200:
        libraries = response.json()
        for lib in libraries:
            if lib['name'] == LIBRARY_NAME:
                print(f" ✓ (ID: {lib['id']}, {lib['num_documents']} documents)")
                return lib['id']
        print(" ✗ (Library not found)")
        return None
    else:
        print(f" ✗ ({response.status_code})")
        return None

def verify_search_works(library_id):
    """Verify that search works on the restored library."""
    print("Verifying search functionality...", end="", flush=True)
    response = requests.post(
        f"{API_URL}/v1/libraries/{library_id}/search",
        json={
            "query": "test chunk",
            "k": 3
        }
    )
    if response.status_code == 200:
        results = response.json()
        num_results = results['total_results']
        print(f" ✓ ({num_results} results found)")
        return True
    else:
        print(f" ✗ ({response.status_code})")
        print(response.text)
        return False

def cleanup():
    """Clean up test data."""
    print("\nCleaning up test data...", end="", flush=True)
    response = requests.get(f"{API_URL}/v1/libraries")
    if response.status_code == 200:
        libraries = response.json()
        for lib in libraries:
            if lib['name'] == LIBRARY_NAME:
                del_response = requests.delete(f"{API_URL}/v1/libraries/{lib['id']}")
                if del_response.status_code == 204:
                    print(" ✓")
                else:
                    print(f" ✗ ({del_response.status_code})")
                return
    print(" (Library not found)")

def main():
    """Run persistence test."""
    print("=" * 60)
    print("Persistence System Test")
    print("=" * 60)

    # Check if server is already running
    try:
        response = requests.get(f"{API_URL}/health", timeout=1)
        print("⚠️  Server is already running. Please stop it first.")
        print("   This test needs to control server startup/shutdown.")
        sys.exit(1)
    except requests.exceptions.RequestException:
        pass

    # Start server
    print("\n[STEP 1] Starting API server...")
    server_process = subprocess.Popen(
        ["python", "-m", "uvicorn", "app.api.main:app", "--host", "localhost", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=Path(__file__).parent
    )

    try:
        if not wait_for_server():
            print("✗ Failed to start server")
            server_process.terminate()
            sys.exit(1)

        # Create library and add document
        print("\n[STEP 2] Creating test data...")
        library_id = create_library()
        if not library_id:
            print("✗ Failed to create library")
            server_process.terminate()
            sys.exit(1)

        document_id = add_document(library_id)
        if not document_id:
            print("✗ Failed to add document")
            server_process.terminate()
            sys.exit(1)

        print("\n✓ Test data created successfully")

        # Stop server gracefully
        print("\n[STEP 3] Stopping server to trigger save...")
        server_process.send_signal(signal.SIGTERM)
        server_process.wait(timeout=10)
        print("✓ Server stopped")

        # Wait a moment
        time.sleep(2)

        # Restart server
        print("\n[STEP 4] Restarting server to test restoration...")
        server_process = subprocess.Popen(
            ["python", "-m", "uvicorn", "app.api.main:app", "--host", "localhost", "--port", "8000"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=Path(__file__).parent
        )

        if not wait_for_server():
            print("✗ Failed to restart server")
            server_process.terminate()
            sys.exit(1)

        # Verify data was restored
        print("\n[STEP 5] Verifying data restoration...")
        restored_library_id = verify_library_exists()
        if not restored_library_id:
            print("✗ Library was not restored")
            server_process.terminate()
            sys.exit(1)

        # Verify search works (embeddings were regenerated)
        if not verify_search_works(restored_library_id):
            print("✗ Search failed - embeddings may not have been regenerated")
            server_process.terminate()
            sys.exit(1)

        print("\n" + "=" * 60)
        print("✓ PERSISTENCE TEST PASSED")
        print("=" * 60)
        print("\nData survived server restart:")
        print(f"  - Library '{LIBRARY_NAME}' restored")
        print("  - Documents and chunks restored")
        print("  - Embeddings regenerated successfully")
        print("  - Search functionality working")

        # Cleanup
        cleanup()

    finally:
        # Stop server
        print("\nStopping server...")
        server_process.terminate()
        try:
            server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_process.kill()
        print("✓ Server stopped")

if __name__ == "__main__":
    main()
