#!/usr/bin/env python3
"""
Test IVF index integration with live API server.

Tests:
1. Create library with IVF index
2. Add documents (will trigger index build)
3. Perform search
4. Check statistics

Run with: python3 test_ivf_integration.py
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def print_success(msg):
    print(f"✅ {msg}")

def print_error(msg):
    print(f"❌ {msg}")

def print_info(msg):
    print(f"ℹ️  {msg}")

def main():
    print("=" * 60)
    print("IVF Index Integration Test")
    print("=" * 60)

    # Step 1: Create library with IVF index
    print("\n1. Creating library with IVF index...")
    lib_response = requests.post(
        f"{BASE_URL}/v1/libraries",
        json={
            "name": "IVF Test Library",
            "index_type": "ivf",
            "description": "Testing IVF index integration"
        }
    )

    if lib_response.status_code != 200:
        print_error(f"Failed to create library: {lib_response.status_code}")
        print(lib_response.text)
        return

    library_data = lib_response.json()
    library_id = library_data["id"]
    print_success(f"Created library: {library_id}")
    print_info(f"Index type: {library_data['metadata']['index_type']}")

    # Step 2: Add documents
    print("\n2. Adding documents...")
    docs = [
        {"title": "Machine Learning Basics", "text": "Machine learning is a subset of artificial intelligence that enables systems to learn from data."},
        {"title": "Deep Learning", "text": "Deep learning uses neural networks with multiple layers to learn hierarchical representations."},
        {"title": "Natural Language Processing", "text": "NLP enables computers to understand, interpret and generate human language."},
        {"title": "Computer Vision", "text": "Computer vision trains computers to interpret and understand visual information from images."},
        {"title": "Reinforcement Learning", "text": "Reinforcement learning is about learning what actions to take to maximize cumulative reward."},
    ]

    doc_ids = []
    for doc in docs:
        response = requests.post(
            f"{BASE_URL}/v1/libraries/{library_id}/documents",
            json=doc
        )
        if response.status_code == 200:
            doc_data = response.json()
            doc_ids.append(doc_data["id"])
            print_info(f"Added: {doc['title']}")
        else:
            print_error(f"Failed to add document: {doc['title']}")

    print_success(f"Added {len(doc_ids)} documents")

    # Give index time to build
    print("\n3. Waiting for index to build...")
    time.sleep(2)

    # Step 3: Perform search
    print("\n4. Performing search...")
    search_response = requests.post(
        f"{BASE_URL}/v1/libraries/{library_id}/search",
        json={
            "query": "neural networks and deep learning",
            "k": 3
        }
    )

    if search_response.status_code == 200:
        search_results = search_response.json()
        print_success(f"Search returned {len(search_results['results'])} results")

        for i, result in enumerate(search_results['results'], 1):
            print(f"\n   Rank {i}:")
            print(f"   Distance: {result['distance']:.4f}")
            print(f"   Text: {result['text'][:80]}...")
    else:
        print_error(f"Search failed: {search_response.status_code}")
        print(search_response.text)

    # Step 4: Check statistics
    print("\n5. Checking library statistics...")
    stats_response = requests.get(
        f"{BASE_URL}/v1/libraries/{library_id}/statistics"
    )

    if stats_response.status_code == 200:
        stats = stats_response.json()
        print_success("Statistics retrieved")
        print(f"\n   Library: {stats['library_name']}")
        print(f"   Index Type: {stats['index_type']}")
        print(f"   Total Vectors: {stats.get('total_vectors', 'N/A')}")
        print(f"   Index Stats: {json.dumps(stats.get('index_stats', {}), indent=2)}")
    else:
        print_error(f"Failed to get statistics: {stats_response.status_code}")

    # Step 5: Test optimize endpoint (if implemented)
    print("\n6. Testing index optimization...")
    try:
        optimize_response = requests.post(
            f"{BASE_URL}/v1/libraries/{library_id}/optimize"
        )
        if optimize_response.status_code == 200:
            opt_data = optimize_response.json()
            print_success(f"Optimization complete: {opt_data}")
        else:
            print_info(f"Optimize endpoint not available or failed: {optimize_response.status_code}")
    except Exception as e:
        print_info(f"Optimize endpoint not available: {e}")

    # Cleanup
    print("\n7. Cleaning up...")
    delete_response = requests.delete(f"{BASE_URL}/v1/libraries/{library_id}")
    if delete_response.status_code == 200:
        print_success("Library deleted")
    else:
        print_error(f"Failed to delete library: {delete_response.status_code}")

    print("\n" + "=" * 60)
    print("IVF Integration Test Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
