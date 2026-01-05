#!/usr/bin/env python3
"""
Standalone benchmark script for arrwDB.

This script runs performance benchmarks against arrwDB without requiring
the full VectorDBBench installation. Useful for quick internal testing.

Usage:
    python run_benchmark.py --url http://localhost:8000 --dataset sift-10k
    python run_benchmark.py --url http://localhost:8000 --dataset random --dim 768 --size 100000

Datasets:
    sift-10k: 10,000 128-dim SIFT vectors (downloads automatically)
    random: Randomly generated vectors (specify --dim and --size)
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np

# Add parent to path for arrwdb import
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "packages" / "arrwdb"))

try:
    from arrwdb import ArrwDBClient
except ImportError:
    print("Error: arrwdb package not found. Install with: pip install -e packages/arrwdb")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    dataset: str
    num_vectors: int
    dimension: int
    index_type: str

    # Insertion metrics
    insert_time_sec: float
    insert_throughput_vec_per_sec: float

    # Search metrics (standard)
    search_latency_p50_ms: float
    search_latency_p95_ms: float
    search_latency_p99_ms: float
    search_qps: float
    recall_at_10: float

    # Temperature search metrics (novel feature)
    temp_search_latency_p50_ms: float | None = None
    temp_search_diversity_score: float | None = None

    # Index Oracle result
    index_recommendation: str | None = None

    # Embedding Health result
    embedding_health_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def download_sift_dataset(cache_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Download and load SIFT-10K dataset.

    Returns:
        Tuple of (base vectors, query vectors, ground truth).
    """
    import urllib.request
    import tarfile

    cache_dir.mkdir(parents=True, exist_ok=True)
    sift_path = cache_dir / "siftsmall"

    if not sift_path.exists():
        logger.info("Downloading SIFT-10K dataset...")
        url = "ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz"
        tar_path = cache_dir / "siftsmall.tar.gz"

        urllib.request.urlretrieve(url, tar_path)

        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(cache_dir)

        tar_path.unlink()

    def read_fvecs(filename: str) -> np.ndarray:
        with open(filename, "rb") as f:
            data = np.frombuffer(f.read(), dtype=np.float32)
        dim = int(data[0])
        return data.reshape(-1, dim + 1)[:, 1:]

    def read_ivecs(filename: str) -> np.ndarray:
        with open(filename, "rb") as f:
            data = np.frombuffer(f.read(), dtype=np.int32)
        dim = int(data[0])
        return data.reshape(-1, dim + 1)[:, 1:]

    base = read_fvecs(str(sift_path / "siftsmall_base.fvecs"))
    query = read_fvecs(str(sift_path / "siftsmall_query.fvecs"))
    gt = read_ivecs(str(sift_path / "siftsmall_groundtruth.ivecs"))

    logger.info(f"Loaded SIFT: {base.shape[0]} base, {query.shape[0]} queries, dim={base.shape[1]}")
    return base, query, gt


def generate_random_dataset(
    num_vectors: int,
    num_queries: int,
    dim: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate random vectors for benchmarking.

    Returns:
        Tuple of (base vectors, query vectors, ground truth).
    """
    np.random.seed(seed)

    # Generate base vectors (normalized for cosine similarity)
    base = np.random.randn(num_vectors, dim).astype(np.float32)
    base /= np.linalg.norm(base, axis=1, keepdims=True)

    # Generate query vectors
    query = np.random.randn(num_queries, dim).astype(np.float32)
    query /= np.linalg.norm(query, axis=1, keepdims=True)

    # Compute ground truth (brute force)
    logger.info("Computing ground truth...")
    k = 100
    gt = np.zeros((num_queries, k), dtype=np.int32)

    for i, q in enumerate(query):
        # Cosine similarity = dot product for normalized vectors
        sims = base @ q
        gt[i] = np.argsort(-sims)[:k]

    logger.info(f"Generated random: {num_vectors} base, {num_queries} queries, dim={dim}")
    return base, query, gt


def compute_recall(results: list[int], ground_truth: np.ndarray, k: int = 10) -> float:
    """Compute recall@k."""
    gt_set = set(ground_truth[:k].tolist())
    result_set = set(results[:k])
    return len(gt_set & result_set) / k


def run_benchmark(
    client: ArrwDBClient,
    base: np.ndarray,
    query: np.ndarray,
    ground_truth: np.ndarray,
    index_type: str = "hnsw",
    dataset_name: str = "unknown",
) -> BenchmarkResult:
    """Run full benchmark suite.

    Args:
        client: arrwDB client.
        base: Base vectors to index.
        query: Query vectors.
        ground_truth: Ground truth for recall calculation.
        index_type: Index type to use.
        dataset_name: Name for results.

    Returns:
        BenchmarkResult with all metrics.
    """
    num_vectors, dim = base.shape
    num_queries = query.shape[0]

    # Create library
    logger.info(f"Creating library with {index_type} index...")
    library = client.create_library(
        name=f"benchmark_{dataset_name}_{int(time.time())}",
        description=f"Benchmark: {num_vectors} vectors, dim={dim}",
        index_type=index_type,
    )
    library_id = library["id"]

    # =========================================================================
    # Insertion benchmark
    # =========================================================================
    logger.info(f"Inserting {num_vectors} vectors...")
    batch_size = 1000
    insert_start = time.time()

    for i in range(0, num_vectors, batch_size):
        batch = base[i:i + batch_size]
        texts = [f"doc_{j}" for j in range(i, min(i + batch_size, num_vectors))]

        # Create chunks with pre-computed embeddings
        chunks = [(text, emb.tolist()) for text, emb in zip(texts, batch)]

        client.add_document_with_embeddings(
            library_id=library_id,
            title=f"batch_{i // batch_size}",
            chunks=chunks,
        )

        if (i + batch_size) % 10000 == 0:
            logger.info(f"  Inserted {i + batch_size}/{num_vectors}")

    insert_time = time.time() - insert_start
    insert_throughput = num_vectors / insert_time
    logger.info(f"Insertion complete: {insert_time:.2f}s ({insert_throughput:.0f} vec/s)")

    # =========================================================================
    # Search benchmark
    # =========================================================================
    logger.info(f"Running {num_queries} search queries...")
    latencies = []
    recalls = []

    for i, q in enumerate(query):
        start = time.time()
        results = client.search_with_embedding(
            library_id=library_id,
            embedding=q.tolist(),
            k=100,
        )
        latency = (time.time() - start) * 1000  # ms
        latencies.append(latency)

        # Extract chunk indices as IDs (since we're using pre-computed embeddings)
        result_ids = []
        for r in results.get("results", []):
            # Result format: {'chunk': {'text': 'doc_N', ...}, 'distance': ...}
            chunk = r.get("chunk", {})
            text = chunk.get("text", "") if isinstance(chunk, dict) else ""
            if text.startswith("doc_"):
                try:
                    result_ids.append(int(text.split("_")[1]))
                except (ValueError, IndexError):
                    result_ids.append(-1)
            else:
                result_ids.append(-1)
        recall = compute_recall(result_ids, ground_truth[i], k=10)
        recalls.append(recall)

    latencies = np.array(latencies)
    search_p50 = np.percentile(latencies, 50)
    search_p95 = np.percentile(latencies, 95)
    search_p99 = np.percentile(latencies, 99)
    search_qps = 1000 / np.mean(latencies)
    mean_recall = np.mean(recalls)

    logger.info(f"Search latency: p50={search_p50:.2f}ms, p95={search_p95:.2f}ms, p99={search_p99:.2f}ms")
    logger.info(f"Search QPS: {search_qps:.0f}, Recall@10: {mean_recall:.3f}")

    # =========================================================================
    # Temperature search benchmark (novel feature)
    # =========================================================================
    logger.info("Running temperature search benchmark...")
    temp_latencies = []
    diversity_scores = []
    temp_p50 = None
    diversity = None

    try:
        # Use text-based temperature search with sample queries
        sample_queries = ["machine learning", "neural networks", "data science", "deep learning", "AI"]
        for query_text in sample_queries[:min(5, num_queries)]:
            start = time.time()
            results = client.temperature_search(
                corpus_id=library_id,
                query_text=query_text,
                k=20,
                temperature=1.5,
            )
            latency = (time.time() - start) * 1000
            temp_latencies.append(latency)

            # Compute diversity (variance of distances)
            distances = [r.get("distance", 0) for r in results.get("results", [])]
            if distances:
                diversity_scores.append(np.std(distances))

        temp_p50 = np.percentile(temp_latencies, 50) if temp_latencies else None
        diversity = np.mean(diversity_scores) if diversity_scores else None
    except Exception as e:
        logger.warning(f"Temperature search benchmark failed: {e}")

    # =========================================================================
    # Index Oracle (novel feature)
    # =========================================================================
    logger.info("Getting Index Oracle recommendation...")
    try:
        recommendation = client.get_index_recommendation(library_id)
        index_rec = recommendation.get("recommended_index", "unknown")
        logger.info(f"Index Oracle recommends: {index_rec}")
    except Exception as e:
        logger.warning(f"Index Oracle failed: {e}")
        index_rec = None

    # =========================================================================
    # Embedding Health (novel feature)
    # =========================================================================
    logger.info("Analyzing embedding health...")
    try:
        health = client.analyze_embedding_health(library_id)
        health_score = 1.0 - health.get("degeneracy_score", 0)
        logger.info(f"Embedding health score: {health_score:.3f}")
    except Exception as e:
        logger.warning(f"Embedding health failed: {e}")
        health_score = None

    # Cleanup
    logger.info("Cleaning up...")
    try:
        client.delete_library(library_id)
    except Exception:
        pass

    return BenchmarkResult(
        dataset=dataset_name,
        num_vectors=num_vectors,
        dimension=dim,
        index_type=index_type,
        insert_time_sec=insert_time,
        insert_throughput_vec_per_sec=insert_throughput,
        search_latency_p50_ms=search_p50,
        search_latency_p95_ms=search_p95,
        search_latency_p99_ms=search_p99,
        search_qps=search_qps,
        recall_at_10=mean_recall,
        temp_search_latency_p50_ms=temp_p50,
        temp_search_diversity_score=diversity,
        index_recommendation=index_rec,
        embedding_health_score=health_score,
    )


def main():
    parser = argparse.ArgumentParser(description="arrwDB Benchmark")
    parser.add_argument("--url", default="http://localhost:8000", help="arrwDB server URL")
    parser.add_argument("--api-key", default=None, help="API key")
    parser.add_argument("--dataset", default="random", choices=["sift-10k", "random"],
                        help="Dataset to use")
    parser.add_argument("--dim", type=int, default=768, help="Vector dimension (random dataset)")
    parser.add_argument("--size", type=int, default=10000, help="Number of vectors (random dataset)")
    parser.add_argument("--queries", type=int, default=100, help="Number of queries")
    parser.add_argument("--index-type", default="hnsw",
                        choices=["hnsw", "ivf", "lsh", "kdtree", "brute_force"],
                        help="Index type to benchmark")
    parser.add_argument("--output", default=None, help="Output JSON file for results")
    parser.add_argument("--cache-dir", default="~/.cache/arrwdb_bench", help="Dataset cache dir")
    parser.add_argument("--timeout", type=int, default=300, help="Request timeout in seconds (default 300 for large datasets)")

    args = parser.parse_args()

    # Create client
    logger.info(f"Connecting to arrwDB at {args.url}...")
    client = ArrwDBClient(base_url=args.url, api_key=args.api_key, timeout=args.timeout)

    # Check health
    try:
        health = client.health_check()
        logger.info(f"Server health: {health}")
    except Exception as e:
        logger.error(f"Failed to connect: {e}")
        sys.exit(1)

    # Load dataset
    cache_dir = Path(args.cache_dir).expanduser()

    if args.dataset == "sift-10k":
        base, query, gt = download_sift_dataset(cache_dir)
    else:
        base, query, gt = generate_random_dataset(
            num_vectors=args.size,
            num_queries=args.queries,
            dim=args.dim,
        )

    # Run benchmark
    result = run_benchmark(
        client=client,
        base=base,
        query=query[:args.queries],
        ground_truth=gt[:args.queries],
        index_type=args.index_type,
        dataset_name=args.dataset,
    )

    # Print results
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Dataset:        {result.dataset}")
    print(f"Vectors:        {result.num_vectors:,}")
    print(f"Dimension:      {result.dimension}")
    print(f"Index Type:     {result.index_type}")
    print("-" * 60)
    print(f"Insert Time:    {result.insert_time_sec:.2f}s")
    print(f"Insert Rate:    {result.insert_throughput_vec_per_sec:,.0f} vec/s")
    print("-" * 60)
    print(f"Search p50:     {result.search_latency_p50_ms:.2f}ms")
    print(f"Search p95:     {result.search_latency_p95_ms:.2f}ms")
    print(f"Search p99:     {result.search_latency_p99_ms:.2f}ms")
    print(f"Search QPS:     {result.search_qps:,.0f}")
    print(f"Recall@10:      {result.recall_at_10:.3f}")
    print("-" * 60)
    print("Novel Features:")
    if result.temp_search_latency_p50_ms:
        print(f"  Temp Search p50:  {result.temp_search_latency_p50_ms:.2f}ms")
    if result.temp_search_diversity_score:
        print(f"  Diversity Score:  {result.temp_search_diversity_score:.4f}")
    if result.index_recommendation:
        print(f"  Index Oracle:     {result.index_recommendation}")
    if result.embedding_health_score:
        print(f"  Health Score:     {result.embedding_health_score:.3f}")
    print("=" * 60)

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
