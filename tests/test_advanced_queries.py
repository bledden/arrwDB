"""
Test advanced query features including hybrid search and reranking.

This script demonstrates:
1. Hybrid search (vector + metadata scoring)
2. Reranking with different functions (recency, position, length)
3. Score breakdown and transparency
4. Comparison with standard vector search
"""

import requests
import time
from datetime import datetime, timedelta

BASE_URL = "http://localhost:8000/v1"


def setup_test_library():
    """Create a test library with documents of varying ages."""
    print("\n" + "=" * 60)
    print("SETUP: Creating test library")
    print("=" * 60)

    # Create library
    response = requests.post(
        f"{BASE_URL}/libraries",
        json={
            "name": "Advanced Query Test Library",
            "description": "Test library for hybrid search and reranking",
            "index_type": "hnsw",
        },
    )

    if response.status_code != 201:
        print(f"✗ Failed to create library: {response.text}")
        return None

    library = response.json()
    library_id = library["id"]
    print(f"✓ Created library: {library_id}")

    # Add test documents with different characteristics
    test_docs = [
        {
            "title": "Recent AI Breakthroughs",
            "texts": [
                "Recent advances in artificial intelligence have revolutionized natural language processing.",
                "Large language models like GPT-4 demonstrate unprecedented capabilities.",
            ],
            "tags": ["AI", "recent", "breakthrough"],
        },
        {
            "title": "Introduction to Machine Learning",
            "texts": [
                "Machine learning is a subset of artificial intelligence that focuses on data-driven learning.",
                "The fundamentals of ML include supervised learning, unsupervised learning, and reinforcement learning.",
                "This chapter introduces basic concepts that are essential for understanding advanced topics.",
            ],
            "tags": ["ML", "introduction", "basics"],
        },
        {
            "title": "Deep Learning Architectures",
            "texts": [
                "Neural networks form the foundation of modern deep learning systems.",
                "Convolutional neural networks excel at computer vision tasks.",
                "Transformers have become the dominant architecture for NLP applications.",
            ],
            "tags": ["deep-learning", "architecture"],
        },
        {
            "title": "Historical Overview of AI",
            "texts": [
                "The history of artificial intelligence dates back to the 1950s.",
                "Early AI systems were rule-based and limited in scope.",
                "The AI winter of the 1980s marked a period of reduced funding and interest.",
                "Recent decades have seen a resurgence in AI research and applications.",
            ],
            "tags": ["history", "overview"],
        },
        {
            "title": "Short Note on ML",
            "texts": ["ML is about learning from data."],
            "tags": ["ML", "short"],
        },
    ]

    print(f"\nAdding {len(test_docs)} test documents...")
    for i, doc in enumerate(test_docs, 1):
        response = requests.post(
            f"{BASE_URL}/libraries/{library_id}/documents",
            json=doc,
        )
        if response.status_code == 201:
            print(f"  ✓ Added document {i}/{len(test_docs)}: {doc['title']}")
        else:
            print(f"  ✗ Failed to add document {i}: {response.text}")

    print(f"\n✓ Test library setup complete")
    return library_id


def test_standard_vector_search(library_id):
    """Baseline: standard vector search."""
    print("\n" + "=" * 60)
    print("TEST 1: Standard Vector Search (Baseline)")
    print("=" * 60)

    response = requests.post(
        f"{BASE_URL}/libraries/{library_id}/search",
        json={"query": "artificial intelligence fundamentals", "k": 5},
    )

    if response.status_code != 200:
        print(f"✗ Failed: {response.text}")
        return

    results = response.json()
    print(f"✓ Found {results['total_results']} results in {results['query_time_ms']:.2f}ms")
    print(f"\nTop results:")
    for i, result in enumerate(results["results"][:3], 1):
        print(f"\n  {i}. {result['document_title']}")
        print(f"     Distance: {result['distance']:.4f}")
        print(f"     Text: {result['chunk']['text'][:80]}...")


def test_hybrid_search_with_recency(library_id):
    """Test hybrid search with recency boost."""
    print("\n" + "=" * 60)
    print("TEST 2: Hybrid Search with Recency Boost")
    print("=" * 60)

    response = requests.post(
        f"{BASE_URL}/libraries/{library_id}/search/hybrid",
        json={
            "query": "artificial intelligence fundamentals",
            "k": 5,
            "vector_weight": 0.6,
            "metadata_weight": 0.4,
            "recency_boost": True,
            "recency_half_life_days": 30.0,
        },
    )

    if response.status_code != 200:
        print(f"✗ Failed: {response.text}")
        return

    results = response.json()
    print(f"✓ Found {results['total_results']} results in {results['query_time_ms']:.2f}ms")
    print(f"\nScoring Config:")
    print(f"  Vector Weight: {results['scoring_config']['vector_weight']}")
    print(f"  Metadata Weight: {results['scoring_config']['metadata_weight']}")
    print(f"  Recency Boost: {results['scoring_config']['recency_boost']}")

    print(f"\nTop results with score breakdown:")
    for i, result in enumerate(results["results"][:3], 1):
        breakdown = result["score_breakdown"]
        print(f"\n  {i}. {result['document_title']}")
        print(f"     Hybrid Score: {result['score']:.4f}")
        print(f"       ├─ Vector Score: {breakdown['vector_score']:.4f} (weight: {breakdown['vector_weight']})")
        print(f"       └─ Metadata Score: {breakdown['metadata_score']:.4f} (weight: {breakdown['metadata_weight']})")
        if breakdown.get("recency_boost"):
            print(f"          └─ Recency Boost: {breakdown['recency_boost']:.4f}")
        print(f"     Text: {result['chunk']['text'][:80]}...")


def test_hybrid_search_pure_vector(library_id):
    """Test hybrid search with 100% vector weight (should match standard search)."""
    print("\n" + "=" * 60)
    print("TEST 3: Hybrid Search (100% Vector, 0% Metadata)")
    print("=" * 60)

    response = requests.post(
        f"{BASE_URL}/libraries/{library_id}/search/hybrid",
        json={
            "query": "artificial intelligence fundamentals",
            "k": 5,
            "vector_weight": 1.0,
            "metadata_weight": 0.0,
            "recency_boost": False,
        },
    )

    if response.status_code != 200:
        print(f"✗ Failed: {response.text}")
        return

    results = response.json()
    print(f"✓ Found {results['total_results']} results in {results['query_time_ms']:.2f}ms")
    print(f"\nTop results (should match standard vector search):")
    for i, result in enumerate(results["results"][:3], 1):
        breakdown = result["score_breakdown"]
        print(f"\n  {i}. {result['document_title']}")
        print(f"     Hybrid Score: {result['score']:.4f}")
        print(f"     Vector Score: {breakdown['vector_score']:.4f}")
        print(f"     Vector Distance: {breakdown['vector_distance']:.4f}")


def test_reranking_by_recency(library_id):
    """Test reranking with recency function."""
    print("\n" + "=" * 60)
    print("TEST 4: Reranking by Recency")
    print("=" * 60)

    response = requests.post(
        f"{BASE_URL}/libraries/{library_id}/search/rerank",
        json={
            "query": "artificial intelligence",
            "k": 5,
            "rerank_function": "recency",
            "rerank_params": {"half_life_days": 30.0},
        },
    )

    if response.status_code != 200:
        print(f"✗ Failed: {response.text}")
        return

    results = response.json()
    print(f"✓ Found {results['total_results']} results in {results['query_time_ms']:.2f}ms")
    print(f"  Rerank Function: {results['rerank_function']}")
    print(f"  Rerank Params: {results['rerank_params']}")

    print(f"\nTop results:")
    for i, result in enumerate(results["results"][:3], 1):
        print(f"\n  {i}. {result['document_title']}")
        print(f"     Reranked Score: {result['score']:.4f}")
        print(f"     Text: {result['chunk']['text'][:80]}...")


def test_reranking_by_position(library_id):
    """Test reranking by chunk position (prefer early chunks)."""
    print("\n" + "=" * 60)
    print("TEST 5: Reranking by Position (Prefer Early)")
    print("=" * 60)

    response = requests.post(
        f"{BASE_URL}/libraries/{library_id}/search/rerank",
        json={
            "query": "machine learning fundamentals",
            "k": 5,
            "rerank_function": "position",
            "rerank_params": {"prefer_early": True},
        },
    )

    if response.status_code != 200:
        print(f"✗ Failed: {response.text}")
        return

    results = response.json()
    print(f"✓ Found {results['total_results']} results in {results['query_time_ms']:.2f}ms")
    print(f"  Rerank Function: {results['rerank_function']}")
    print(f"  Rerank Params: {results['rerank_params']}")

    print(f"\nTop results (should prefer introductory/early chunks):")
    for i, result in enumerate(results["results"][:3], 1):
        chunk_idx = result["chunk"]["metadata"]["chunk_index"]
        print(f"\n  {i}. {result['document_title']}")
        print(f"     Reranked Score: {result['score']:.4f}")
        print(f"     Chunk Index: {chunk_idx} (0 = first chunk)")
        print(f"     Text: {result['chunk']['text'][:80]}...")


def test_reranking_by_length(library_id):
    """Test reranking by text length (prefer longer chunks)."""
    print("\n" + "=" * 60)
    print("TEST 6: Reranking by Length (Prefer Longer)")
    print("=" * 60)

    response = requests.post(
        f"{BASE_URL}/libraries/{library_id}/search/rerank",
        json={
            "query": "machine learning",
            "k": 5,
            "rerank_function": "length",
            "rerank_params": {"prefer_longer": True},
        },
    )

    if response.status_code != 200:
        print(f"✗ Failed: {response.text}")
        return

    results = response.json()
    print(f"✓ Found {results['total_results']} results in {results['query_time_ms']:.2f}ms")
    print(f"  Rerank Function: {results['rerank_function']}")
    print(f"  Rerank Params: {results['rerank_params']}")

    print(f"\nTop results (should prefer longer, more detailed chunks):")
    for i, result in enumerate(results["results"][:3], 1):
        text_length = len(result["chunk"]["text"])
        print(f"\n  {i}. {result['document_title']}")
        print(f"     Reranked Score: {result['score']:.4f}")
        print(f"     Text Length: {text_length} characters")
        print(f"     Text: {result['chunk']['text'][:80]}...")


def test_comparison_standard_vs_hybrid(library_id):
    """Compare standard vector search with hybrid search."""
    print("\n" + "=" * 60)
    print("TEST 7: Comparison - Standard vs Hybrid")
    print("=" * 60)

    query = "recent advances in AI"

    # Standard search
    print("\n📊 Standard Vector Search:")
    response1 = requests.post(
        f"{BASE_URL}/libraries/{library_id}/search",
        json={"query": query, "k": 3},
    )
    if response1.status_code == 200:
        results1 = response1.json()
        for i, r in enumerate(results1["results"][:3], 1):
            print(f"  {i}. {r['document_title']} (distance: {r['distance']:.4f})")

    # Hybrid search with recency
    print("\n🚀 Hybrid Search (with recency boost):")
    response2 = requests.post(
        f"{BASE_URL}/libraries/{library_id}/search/hybrid",
        json={
            "query": query,
            "k": 3,
            "vector_weight": 0.6,
            "metadata_weight": 0.4,
            "recency_boost": True,
            "recency_half_life_days": 30.0,
        },
    )
    if response2.status_code == 200:
        results2 = response2.json()
        for i, r in enumerate(results2["results"][:3], 1):
            print(f"  {i}. {r['document_title']} (hybrid score: {r['score']:.4f})")

    print("\n✓ Notice how hybrid search can reorder results based on recency!")


def main():
    """Run all advanced query tests."""
    print("\n" + "=" * 60)
    print("ADVANCED QUERY FEATURES TEST SUITE")
    print("=" * 60)
    print("\nThis test suite demonstrates:")
    print("  1. Hybrid search (vector + metadata scoring)")
    print("  2. Reranking with different functions")
    print("  3. Score breakdown and transparency")
    print("  4. Comparison with standard vector search")
    print("\nMake sure the API server is running:")
    print("  python3 run_api.py")
    print("\n" + "=" * 60)

    input("\nPress Enter to start tests...")

    # Setup
    library_id = setup_test_library()
    if not library_id:
        print("\n✗ Setup failed, stopping tests")
        return

    # Wait for indexing
    print("\n⏳ Waiting 2 seconds for indexing...")
    time.sleep(2)

    # Run tests
    test_standard_vector_search(library_id)
    test_hybrid_search_with_recency(library_id)
    test_hybrid_search_pure_vector(library_id)
    test_reranking_by_recency(library_id)
    test_reranking_by_position(library_id)
    test_reranking_by_length(library_id)
    test_comparison_standard_vs_hybrid(library_id)

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUITE COMPLETE")
    print("=" * 60)
    print("\n✓ All advanced query features tested successfully!")
    print("\nKey Takeaways:")
    print("  • Hybrid search combines vector similarity with metadata signals")
    print("  • Recency boost helps surface recent content")
    print("  • Reranking provides flexible post-processing")
    print("  • Score breakdowns provide transparency")
    print("\nExplore more in API docs: http://localhost:8000/docs")
    print("Check 'Advanced Queries' tag for endpoints")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
