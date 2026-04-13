#!/usr/bin/env python3
"""
Embedding Provider Benchmark for arrwDB.

Tests HNSW performance with real embeddings from different providers:
- Voyage AI voyage-4-large (1024d, 200M tokens free) — best retrieval quality
- NVIDIA NV-Embed-v2 (4096d, free via NGC)
- Google Gemini Embedding 2 (3072d, free tier)
- Cohere embed-v4 (1536d, $0.12/1M tokens)

Uses a standard text corpus to generate embeddings and then benchmarks
HNSW search recall/latency on them.

Usage:
    # Voyage 4-large benchmark (primary, free)
    python run_embedding_benchmark.py --provider voyage --size 10000

    # All providers comparison
    python run_embedding_benchmark.py --provider all --size 10000
"""

import argparse
import json
import logging
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import List, Tuple
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.vector_store import VectorStore

try:
    from infrastructure.indexes.rust_hnsw_wrapper import RustHNSWIndexWrapper as HNSWIndex
    USING_RUST = True
except ImportError:
    from infrastructure.indexes.hnsw import HNSWIndex
    USING_RUST = False

from app.services.embedding_providers import get_embedding_provider, EmbeddingProvider

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Text corpus
# =============================================================================

# Simple English Wikipedia paragraphs for benchmark
SAMPLE_CORPUS_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def load_text_corpus(cache_dir: Path, target_size: int) -> List[str]:
    """Load or generate a text corpus for embedding.

    Uses a mix of synthetic and downloaded text to create a diverse corpus.
    Each text is a paragraph-length chunk suitable for embedding.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    corpus_file = cache_dir / "text_corpus.json"

    if corpus_file.exists():
        with open(corpus_file, "r") as f:
            corpus = json.load(f)
        if len(corpus) >= target_size:
            logger.info(f"Loaded cached corpus: {len(corpus)} texts")
            return corpus[:target_size]

    logger.info(f"Generating text corpus ({target_size} texts)...")

    # Domain-specific text templates for realistic embeddings
    domains = [
        # Technology
        "Machine learning algorithms can be categorized into supervised, unsupervised, and reinforcement learning. {detail}",
        "Cloud computing platforms like AWS, GCP, and Azure provide scalable infrastructure for modern applications. {detail}",
        "Database systems implement various indexing strategies including B-trees, hash indexes, and inverted indexes. {detail}",
        "Neural networks consist of interconnected layers of artificial neurons that learn hierarchical representations. {detail}",
        "Distributed systems face challenges like network partitions, consistency, and availability trade-offs. {detail}",
        # Science
        "Quantum mechanics describes the behavior of matter and energy at the atomic and subatomic level. {detail}",
        "The human genome contains approximately 3 billion base pairs of DNA organized into 23 chromosome pairs. {detail}",
        "Climate models simulate atmospheric, oceanic, and land surface processes to predict weather patterns. {detail}",
        "Photosynthesis converts carbon dioxide and water into glucose and oxygen using solar energy. {detail}",
        "The periodic table organizes chemical elements by atomic number and electron configuration. {detail}",
        # Business
        "Supply chain management involves coordinating production, inventory, and distribution across organizations. {detail}",
        "Financial markets facilitate the trading of securities, commodities, and currencies between participants. {detail}",
        "Marketing strategies encompass product positioning, pricing, distribution channels, and promotional activities. {detail}",
        "Organizational leadership theories include transformational, servant, and situational approaches. {detail}",
        "Risk management frameworks identify, assess, and mitigate potential threats to business operations. {detail}",
        # History
        "The Industrial Revolution transformed manufacturing processes from hand production to machine-based methods. {detail}",
        "Ancient civilizations developed writing systems including cuneiform, hieroglyphics, and alphabets. {detail}",
        "The Renaissance period saw advances in art, science, and philosophy across Europe. {detail}",
        "Maritime exploration in the 15th century connected previously isolated continents and cultures. {detail}",
        "The development of democratic governance evolved from ancient Greek city-states to modern nations. {detail}",
        # General knowledge
        "Nutrition science studies how dietary components affect human health and disease prevention. {detail}",
        "Architecture combines engineering principles with aesthetic design to create functional living spaces. {detail}",
        "Music theory analyzes the structure, harmony, and rhythm of compositions across genres. {detail}",
        "Environmental conservation efforts aim to protect biodiversity and natural habitats worldwide. {detail}",
        "Psychological research examines cognitive processes, emotional responses, and behavioral patterns. {detail}",
    ]

    details = [
        "This approach has been widely adopted in industry and academia.",
        "Recent advances have significantly improved performance and efficiency.",
        "Researchers continue to explore novel methods and applications.",
        "The theoretical foundations were established in the mid-20th century.",
        "Practical implementations require careful consideration of trade-offs.",
        "Ongoing developments promise to transform the field in coming years.",
        "Cross-disciplinary collaboration has accelerated progress in this area.",
        "Standardization efforts have improved interoperability and adoption.",
        "Educational institutions increasingly incorporate these concepts into curricula.",
        "Industry applications demonstrate significant real-world impact.",
        "The economic implications extend across multiple sectors and regions.",
        "Ethical considerations play an important role in guiding development.",
        "Technological limitations continue to drive innovative solutions.",
        "Historical precedents inform current approaches and strategies.",
        "Comparative analysis reveals strengths and weaknesses of different methods.",
        "Data-driven approaches have complemented traditional analytical techniques.",
        "International cooperation has been essential for advancing this field.",
        "Public awareness and understanding continue to grow through outreach efforts.",
        "Regulatory frameworks are evolving to address emerging challenges.",
        "Future directions include automation, personalization, and scalability.",
    ]

    np.random.seed(42)
    corpus = []
    for i in range(target_size):
        template = domains[i % len(domains)]
        detail = details[i % len(details)]
        # Add variation with a unique identifier
        text = template.format(detail=detail) + f" (Reference: doc-{i:06d})"
        corpus.append(text)

    # Cache for reuse
    with open(corpus_file, "w") as f:
        json.dump(corpus, f)

    logger.info(f"Generated corpus: {len(corpus)} texts")
    return corpus


# =============================================================================
# Embedding generation
# =============================================================================


def generate_embeddings(
    provider: EmbeddingProvider,
    texts: List[str],
    batch_size: int = 20,
    inter_batch_delay: float = 0.5,
) -> Tuple[NDArray[np.float32], float]:
    """Generate embeddings for all texts using the given provider.

    Returns (embeddings_array, time_seconds).
    """
    n = len(texts)
    dim = provider.dimension
    embeddings = np.zeros((n, dim), dtype=np.float32)

    logger.info(f"Generating {n} embeddings with {provider.name}/{provider.model} (dim={dim}, batch={batch_size})...")

    start = time.time()

    for i in range(0, n, batch_size):
        batch = texts[i : i + batch_size]
        batch_embeddings = provider.embed_texts(batch)

        for j, emb in enumerate(batch_embeddings):
            embeddings[i + j] = emb

        done = min(i + batch_size, n)
        if done % 500 == 0 or done == n:
            elapsed = time.time() - start
            rate = done / elapsed if elapsed > 0 else 0
            logger.info(f"  Embedded {done}/{n} ({rate:.0f} texts/s)")

        # Rate limit pacing
        if inter_batch_delay > 0 and (i + batch_size) < n:
            time.sleep(inter_batch_delay)

    embed_time = time.time() - start
    logger.info(f"Embedding complete: {embed_time:.1f}s ({n / embed_time:.0f} texts/s)")

    return embeddings, embed_time


# =============================================================================
# Benchmark
# =============================================================================


def run_embedding_benchmark(
    provider: EmbeddingProvider,
    corpus_size: int,
    num_queries: int,
    cache_dir: Path,
    M: int = 48,
    ef_construction: int = 400,
    ef_search: int = 200,
) -> dict:
    """Run a complete embedding benchmark for one provider."""
    # Load text corpus
    texts = load_text_corpus(cache_dir, corpus_size + num_queries)
    doc_texts = texts[:corpus_size]
    query_texts = texts[corpus_size : corpus_size + num_queries]

    # Generate document embeddings
    doc_embeddings, embed_doc_time = generate_embeddings(provider, doc_texts)

    # Generate query embeddings (switch to query mode)
    if hasattr(provider, "change_input_type"):
        if provider.name == "nvidia":
            provider.change_input_type("query")
        elif provider.name == "voyage":
            provider.change_input_type("query")
        elif provider.name == "gemini":
            provider.change_input_type("query")
        elif provider.name == "cohere":
            provider.change_input_type("search_query")

    query_embeddings, embed_query_time = generate_embeddings(provider, query_texts)

    dim = doc_embeddings.shape[1]

    # Compute ground truth (brute force cosine similarity)
    logger.info("Computing ground truth...")
    k_gt = 100
    ground_truth = np.zeros((num_queries, k_gt), dtype=np.int32)
    for i in range(num_queries):
        sims = doc_embeddings @ query_embeddings[i]
        ground_truth[i] = np.argsort(-sims)[:k_gt]

    # Build HNSW index
    logger.info(f"Building HNSW index (M={M}, ef_construction={ef_construction})...")
    vector_store = VectorStore(dimension=dim, initial_capacity=corpus_size)
    index = HNSWIndex(
        vector_store=vector_store,
        M=M,
        ef_construction=ef_construction,
        ef_search=ef_search,
    )

    idx_to_uuid = {}
    build_start = time.time()
    report_interval = max(1, corpus_size // 10)

    for i in range(corpus_size):
        uid = uuid4()
        vec = doc_embeddings[i]
        vector_index = vector_store.add_vector(uid, vec)
        index.add_vector(uid, vector_index)
        idx_to_uuid[i] = uid

        if (i + 1) % report_interval == 0:
            elapsed = time.time() - build_start
            rate = (i + 1) / elapsed
            logger.info(f"  Indexed {i + 1}/{corpus_size} ({rate:.0f} vec/s)")

    build_time = time.time() - build_start
    uuid_to_idx = {uid: idx for idx, uid in idx_to_uuid.items()}

    # Search benchmark
    logger.info(f"Running {num_queries} search queries (ef_search={ef_search})...")
    latencies = []
    recalls_at_10 = []
    recalls_at_1 = []

    for i in range(num_queries):
        query = query_embeddings[i]

        start = time.time()
        results = index.search(query, k=10)
        latency_ms = (time.time() - start) * 1000
        latencies.append(latency_ms)

        result_indices = [uuid_to_idx.get(uid, -1) for uid, _ in results]

        gt_set = set(ground_truth[i, :10].tolist())
        result_set = set(result_indices[:10])
        recalls_at_10.append(len(gt_set & result_set) / 10)

        gt_top1 = ground_truth[i, 0]
        recalls_at_1.append(1.0 if result_indices and result_indices[0] == gt_top1 else 0.0)

    latencies = np.array(latencies)

    result = {
        "provider": provider.name,
        "model": provider.model,
        "dimension": dim,
        "corpus_size": corpus_size,
        "num_queries": num_queries,
        "backend": "rust" if USING_RUST else "python",
        "hnsw_M": M,
        "hnsw_ef_construction": ef_construction,
        "hnsw_ef_search": ef_search,
        "embedding_time_sec": embed_doc_time,
        "embedding_throughput_texts_per_sec": corpus_size / embed_doc_time,
        "build_time_sec": build_time,
        "build_throughput_vec_per_sec": corpus_size / build_time,
        "search_latency_p50_ms": float(np.percentile(latencies, 50)),
        "search_latency_p95_ms": float(np.percentile(latencies, 95)),
        "search_latency_p99_ms": float(np.percentile(latencies, 99)),
        "search_qps": 1000.0 / float(np.mean(latencies)),
        "recall_at_10": float(np.mean(recalls_at_10)),
        "recall_at_1": float(np.mean(recalls_at_1)),
    }

    # Print results
    print()
    print("=" * 70)
    print(f"  EMBEDDING BENCHMARK — {provider.name}/{provider.model}")
    print("=" * 70)
    print(f"  Dimension:        {dim}")
    print(f"  Corpus:           {corpus_size:,} texts")
    print(f"  Queries:          {num_queries}")
    print(f"  Backend:          {'Rust' if USING_RUST else 'Python'}")
    print("-" * 70)
    print(f"  Embedding:")
    print(f"    Time:           {embed_doc_time:.1f}s ({corpus_size / embed_doc_time:.0f} texts/s)")
    print(f"  Index Build:")
    print(f"    Time:           {build_time:.1f}s ({corpus_size / build_time:.0f} vec/s)")
    print(f"  Search:")
    print(f"    Recall@10:      {result['recall_at_10']:.4f}")
    print(f"    Recall@1:       {result['recall_at_1']:.4f}")
    print(f"    p50:            {result['search_latency_p50_ms']:.2f}ms")
    print(f"    QPS:            {result['search_qps']:.0f}")
    print("=" * 70)

    return result


def main():
    parser = argparse.ArgumentParser(description="Embedding Provider Benchmark")
    parser.add_argument(
        "--provider",
        default="voyage",
        choices=["voyage", "nvidia", "gemini", "cohere", "all"],
        help="Embedding provider to benchmark (default: voyage)",
    )
    parser.add_argument("--size", type=int, default=10000, help="Corpus size")
    parser.add_argument("--queries", type=int, default=100, help="Number of queries")
    parser.add_argument("--M", type=int, default=48, help="HNSW M")
    parser.add_argument("--ef-construction", type=int, default=400, help="ef_construction")
    parser.add_argument("--ef-search", type=int, default=200, help="ef_search")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="~/.cache/arrwdb_bench/embedding",
        help="Cache directory",
    )

    args = parser.parse_args()
    cache_dir = Path(args.cache_dir).expanduser()

    providers = []
    if args.provider == "all":
        # Try each provider, skip if API key not available
        for name in ["voyage", "nvidia", "gemini", "cohere"]:
            try:
                p = get_embedding_provider(name)
                providers.append(p)
            except ValueError as e:
                logger.warning(f"Skipping {name}: {e}")
    else:
        providers.append(get_embedding_provider(args.provider))

    if not providers:
        logger.error("No providers available. Set API keys in environment.")
        sys.exit(1)

    all_results = []
    for provider in providers:
        logger.info(f"\nBenchmarking: {provider.name}/{provider.model}")
        result = run_embedding_benchmark(
            provider=provider,
            corpus_size=args.size,
            num_queries=args.queries,
            cache_dir=cache_dir,
            M=args.M,
            ef_construction=args.ef_construction,
            ef_search=args.ef_search,
        )
        all_results.append(result)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(all_results if len(all_results) > 1 else all_results[0], f, indent=2)
        logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
