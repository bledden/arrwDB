#!/usr/bin/env python3
"""
Standard ANN Benchmark Runner for arrwDB.

Downloads standard datasets from ann-benchmarks.com and runs them directly
against the HNSW index (no HTTP API overhead), matching the methodology used
by FAISS, Milvus, Qdrant, Weaviate, and other competitors on the leaderboard.

Supported datasets:
    sift-128-euclidean    1M vectors, 128 dims (normalized for cosine)
    glove-200-angular     1.18M vectors, 200 dims
    deep-96-angular       1M vectors, 96 dims (subset of Deep-10M)

Usage:
    python run_ann_benchmark.py --dataset sift-128-euclidean
    python run_ann_benchmark.py --dataset glove-200-angular --M 48 --ef-construction 400
    python run_ann_benchmark.py --dataset all --output-dir results/
    python run_ann_benchmark.py --dataset sift-128-euclidean --ef-search-sweep 10,20,50,100,200,500

Output:
    JSON results compatible with ann-benchmarks.com comparison charts.
    Includes recall-vs-QPS curves when using --ef-search-sweep.
"""

import argparse
import json
import logging
import os
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.vector_store import VectorStore

# Try Rust HNSW first, fall back to Python
try:
    from infrastructure.indexes.rust_hnsw_wrapper import RustHNSWIndexWrapper as HNSWIndex

    USING_RUST = True
except ImportError:
    from infrastructure.indexes.hnsw import HNSWIndex

    USING_RUST = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Dataset definitions
# =============================================================================

DATASETS = {
    "sift-128-euclidean": {
        "url": "https://ann-benchmarks.com/sift-128-euclidean.hdf5",
        "fallback_url": "https://huggingface.co/datasets/hhy3/ann-datasets/resolve/main/sift-128-euclidean.hdf5",
        "distance": "euclidean",
        "normalize": True,  # Normalize so cosine distance preserves ranking
    },
    "glove-200-angular": {
        "url": "https://ann-benchmarks.com/glove-200-angular.hdf5",
        "distance": "angular",
        "normalize": True,
    },
    "deep-96-angular": {
        "url": "https://ann-benchmarks.com/deep-image-96-angular.hdf5",
        "distance": "angular",
        "normalize": True,
        "max_vectors": 1_000_000,  # Use first 1M of Deep-10M
    },
}


# =============================================================================
# Data classes
# =============================================================================


@dataclass
class AnnBenchmarkConfig:
    """Configuration for a single benchmark run."""

    dataset: str
    M: int = 16
    ef_construction: int = 200
    ef_search: int = 50
    max_level: int = 16


@dataclass
class AnnBenchmarkResult:
    """Results from a single benchmark configuration."""

    dataset: str
    num_vectors: int
    dimension: int
    backend: str  # "rust" or "python"

    # HNSW parameters
    M: int
    ef_construction: int
    ef_search: int

    # Build metrics
    build_time_sec: float
    build_throughput_vec_per_sec: float

    # Search metrics
    search_latency_p50_ms: float
    search_latency_p95_ms: float
    search_latency_p99_ms: float
    search_qps: float
    recall_at_10: float
    recall_at_1: float

    # Memory (approximate)
    index_memory_mb: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RecallQPSPoint:
    """A single point on the recall-vs-QPS curve."""

    ef_search: int
    recall_at_10: float
    search_qps: float
    search_latency_p50_ms: float


@dataclass
class AnnBenchmarkSweepResult:
    """Results from a parameter sweep (recall-vs-QPS curve)."""

    dataset: str
    num_vectors: int
    dimension: int
    backend: str
    M: int
    ef_construction: int
    build_time_sec: float
    points: List[RecallQPSPoint] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["points"] = [asdict(p) for p in self.points]
        return d


# =============================================================================
# Dataset loading
# =============================================================================


def download_dataset(name: str, cache_dir: Path) -> Path:
    """Download an HDF5 dataset if not already cached."""
    info = DATASETS[name]
    cache_dir.mkdir(parents=True, exist_ok=True)

    filename = info["url"].split("/")[-1]
    filepath = cache_dir / filename

    if filepath.exists():
        size_mb = filepath.stat().st_size / (1024 * 1024)
        logger.info(f"Dataset {name} already cached ({size_mb:.0f} MB): {filepath}")
        return filepath

    # Browser User-Agent to bypass Cloudflare bot detection on cloud VMs
    opener = urllib.request.build_opener()
    opener.addheaders = [
        ("User-Agent", "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"),
    ]
    urllib.request.install_opener(opener)

    urls_to_try = [info["url"]]
    if "fallback_url" in info:
        urls_to_try.append(info["fallback_url"])

    for url in urls_to_try:
        try:
            logger.info(f"Downloading {name} from {url}...")
            logger.info("This may take several minutes for large datasets.")

            def progress_hook(count, block_size, total_size):
                if total_size > 0:
                    pct = count * block_size * 100 / total_size
                    mb_done = count * block_size / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    if count % 100 == 0:
                        print(f"\r  {pct:.1f}% ({mb_done:.0f}/{mb_total:.0f} MB)", end="", flush=True)

            urllib.request.urlretrieve(url, filepath, reporthook=progress_hook)
            print()  # newline after progress

            size_mb = filepath.stat().st_size / (1024 * 1024)
            logger.info(f"Downloaded {name}: {size_mb:.0f} MB")
            return filepath
        except urllib.error.HTTPError as e:
            logger.warning(f"Failed to download from {url}: {e}")
            if filepath.exists():
                filepath.unlink()  # Remove partial download
            continue

    raise RuntimeError(f"Failed to download dataset {name} from all sources")


def load_dataset(
    name: str, cache_dir: Path
) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.int32]]:
    """Load a dataset, returning (train, test, ground_truth_neighbors).

    All vectors are L2-normalized for use with cosine distance.
    """
    import h5py

    filepath = download_dataset(name, cache_dir)
    info = DATASETS[name]

    with h5py.File(filepath, "r") as f:
        train = np.array(f["train"], dtype=np.float32)
        test = np.array(f["test"], dtype=np.float32)
        neighbors = np.array(f["neighbors"], dtype=np.int32)

    # Subset if needed (e.g., Deep-10M -> Deep-1M)
    max_vectors = info.get("max_vectors")
    if max_vectors and train.shape[0] > max_vectors:
        logger.info(f"Subsetting {train.shape[0]} -> {max_vectors} vectors")
        train = train[:max_vectors]
        # Recompute ground truth for the subset
        neighbors = recompute_ground_truth(train, test, k=neighbors.shape[1])

    # Normalize vectors for cosine distance
    if info.get("normalize", False):
        norms = np.linalg.norm(train, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)  # avoid division by zero
        train = train / norms

        norms = np.linalg.norm(test, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        test = test / norms

    logger.info(
        f"Loaded {name}: {train.shape[0]} train, {test.shape[0]} test, "
        f"dim={train.shape[1]}, gt_k={neighbors.shape[1]}"
    )
    return train, test, neighbors


def recompute_ground_truth(
    train: NDArray[np.float32],
    test: NDArray[np.float32],
    k: int = 100,
) -> NDArray[np.int32]:
    """Recompute ground truth neighbors using brute-force cosine similarity."""
    logger.info(f"Recomputing ground truth for {test.shape[0]} queries (k={k})...")

    # Normalize for cosine similarity
    train_norm = train / np.maximum(np.linalg.norm(train, axis=1, keepdims=True), 1e-10)
    test_norm = test / np.maximum(np.linalg.norm(test, axis=1, keepdims=True), 1e-10)

    neighbors = np.zeros((test.shape[0], k), dtype=np.int32)

    # Process in batches to manage memory
    batch_size = 100
    for i in range(0, test.shape[0], batch_size):
        batch = test_norm[i : i + batch_size]
        # Cosine similarity = dot product for normalized vectors
        sims = batch @ train_norm.T  # (batch_size, n_train)
        # Get top-k indices
        for j in range(batch.shape[0]):
            neighbors[i + j] = np.argsort(-sims[j])[:k]
        if (i + batch_size) % 1000 == 0:
            logger.info(f"  Ground truth: {i + batch_size}/{test.shape[0]}")

    logger.info("Ground truth computed.")
    return neighbors


# =============================================================================
# Benchmark execution
# =============================================================================


def build_index(
    train: NDArray[np.float32],
    config: AnnBenchmarkConfig,
) -> Tuple[Any, VectorStore, Dict[int, Any], float]:
    """Build HNSW index from training vectors.

    Returns (index, vector_store, idx_to_uuid_map, build_time_seconds).
    """
    dim = train.shape[1]
    n = train.shape[0]

    logger.info(
        f"Building HNSW index: {n} vectors, dim={dim}, "
        f"M={config.M}, ef_construction={config.ef_construction}"
    )

    # Create vector store
    vector_store = VectorStore(dimension=dim, initial_capacity=n)

    # Create index
    index = HNSWIndex(
        vector_store=vector_store,
        M=config.M,
        ef_construction=config.ef_construction,
        ef_search=config.ef_search,
    )

    # Insert vectors
    idx_to_uuid = {}
    start = time.time()
    report_interval = max(1, n // 20)  # Report ~20 times

    for i in range(n):
        uid = uuid4()
        vec = train[i]

        # Add to vector store
        vector_index = vector_store.add_vector(uid, vec)

        # Add to HNSW index
        index.add_vector(uid, vector_index)

        idx_to_uuid[i] = uid

        if (i + 1) % report_interval == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate
            logger.info(
                f"  Inserted {i + 1}/{n} ({(i + 1) / n * 100:.1f}%) "
                f"- {rate:.0f} vec/s, ETA {eta:.0f}s"
            )

    build_time = time.time() - start
    logger.info(
        f"Build complete: {build_time:.1f}s ({n / build_time:.0f} vec/s)"
    )

    return index, vector_store, idx_to_uuid, build_time


def run_search_benchmark(
    index: Any,
    test: NDArray[np.float32],
    ground_truth: NDArray[np.int32],
    idx_to_uuid: Dict[int, Any],
    ef_search: int,
    k: int = 10,
) -> Tuple[float, float, float, float, float, float]:
    """Run search benchmark and return metrics.

    Returns (recall@10, recall@1, p50_ms, p95_ms, p99_ms, qps).
    """
    # Build reverse mapping: UUID -> original index
    uuid_to_idx = {uid: idx for idx, uid in idx_to_uuid.items()}

    # Set ef_search
    index.set_ef_search(ef_search)

    n_queries = test.shape[0]
    latencies = []
    recalls_at_10 = []
    recalls_at_1 = []

    for i in range(n_queries):
        query = test[i]

        start = time.time()
        results = index.search(query, k=k)
        latency_ms = (time.time() - start) * 1000
        latencies.append(latency_ms)

        # Convert UUID results back to original indices
        result_indices = []
        for uid, dist in results:
            orig_idx = uuid_to_idx.get(uid, -1)
            result_indices.append(orig_idx)

        # Compute recall@10
        gt_set_10 = set(ground_truth[i, :10].tolist())
        result_set_10 = set(result_indices[:10])
        recall_10 = len(gt_set_10 & result_set_10) / 10
        recalls_at_10.append(recall_10)

        # Compute recall@1
        gt_top1 = ground_truth[i, 0]
        recall_1 = 1.0 if len(result_indices) > 0 and result_indices[0] == gt_top1 else 0.0
        recalls_at_1.append(recall_1)

    latencies = np.array(latencies)
    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))
    p99 = float(np.percentile(latencies, 99))
    qps = 1000.0 / float(np.mean(latencies))
    mean_recall_10 = float(np.mean(recalls_at_10))
    mean_recall_1 = float(np.mean(recalls_at_1))

    return mean_recall_10, mean_recall_1, p50, p95, p99, qps


def estimate_memory_mb(n: int, dim: int, M: int) -> float:
    """Rough estimate of HNSW memory usage in MB."""
    # Vectors: n * dim * 4 bytes
    vector_bytes = n * dim * 4
    # Graph: ~n * M * 2 * 8 bytes (neighbor lists, average across layers)
    graph_bytes = n * M * 2 * 8
    # Overhead: ~20% for hash maps, metadata
    total = (vector_bytes + graph_bytes) * 1.2
    return total / (1024 * 1024)


def run_single_benchmark(
    train: NDArray[np.float32],
    test: NDArray[np.float32],
    ground_truth: NDArray[np.int32],
    config: AnnBenchmarkConfig,
) -> AnnBenchmarkResult:
    """Run a single benchmark configuration."""
    n, dim = train.shape

    # Build index
    index, vector_store, idx_to_uuid, build_time = build_index(train, config)

    # Search benchmark
    logger.info(f"Running {test.shape[0]} search queries (ef_search={config.ef_search})...")
    recall_10, recall_1, p50, p95, p99, qps = run_search_benchmark(
        index, test, ground_truth, idx_to_uuid, ef_search=config.ef_search
    )

    logger.info(
        f"Results: recall@10={recall_10:.4f}, recall@1={recall_1:.4f}, "
        f"p50={p50:.2f}ms, QPS={qps:.0f}"
    )

    return AnnBenchmarkResult(
        dataset=config.dataset,
        num_vectors=n,
        dimension=dim,
        backend="rust" if USING_RUST else "python",
        M=config.M,
        ef_construction=config.ef_construction,
        ef_search=config.ef_search,
        build_time_sec=build_time,
        build_throughput_vec_per_sec=n / build_time,
        search_latency_p50_ms=p50,
        search_latency_p95_ms=p95,
        search_latency_p99_ms=p99,
        search_qps=qps,
        recall_at_10=recall_10,
        recall_at_1=recall_1,
        index_memory_mb=estimate_memory_mb(n, dim, config.M),
    )


def run_sweep_benchmark(
    train: NDArray[np.float32],
    test: NDArray[np.float32],
    ground_truth: NDArray[np.int32],
    config: AnnBenchmarkConfig,
    ef_search_values: List[int],
) -> AnnBenchmarkSweepResult:
    """Run a parameter sweep to generate recall-vs-QPS curves.

    Builds the index once, then tests multiple ef_search values.
    """
    n, dim = train.shape

    # Build index once
    index, vector_store, idx_to_uuid, build_time = build_index(train, config)

    sweep = AnnBenchmarkSweepResult(
        dataset=config.dataset,
        num_vectors=n,
        dimension=dim,
        backend="rust" if USING_RUST else "python",
        M=config.M,
        ef_construction=config.ef_construction,
        build_time_sec=build_time,
    )

    for ef in sorted(ef_search_values):
        logger.info(f"Sweep: ef_search={ef}")
        recall_10, recall_1, p50, p95, p99, qps = run_search_benchmark(
            index, test, ground_truth, idx_to_uuid, ef_search=ef
        )

        point = RecallQPSPoint(
            ef_search=ef,
            recall_at_10=recall_10,
            search_qps=qps,
            search_latency_p50_ms=p50,
        )
        sweep.points.append(point)

        logger.info(
            f"  ef={ef}: recall@10={recall_10:.4f}, QPS={qps:.0f}, p50={p50:.2f}ms"
        )

    return sweep


# =============================================================================
# Output formatting
# =============================================================================


def print_results(result: AnnBenchmarkResult) -> None:
    """Print benchmark results in a readable format."""
    print()
    print("=" * 70)
    print(f"  ANN BENCHMARK RESULTS — {result.dataset}")
    print("=" * 70)
    print(f"  Backend:          {'Rust (PyO3)' if result.backend == 'rust' else 'Python'}")
    print(f"  Vectors:          {result.num_vectors:,}")
    print(f"  Dimension:        {result.dimension}")
    print(f"  Memory (est.):    {result.index_memory_mb:.0f} MB")
    print("-" * 70)
    print(f"  HNSW Parameters:")
    print(f"    M:              {result.M}")
    print(f"    ef_construction:{result.ef_construction}")
    print(f"    ef_search:      {result.ef_search}")
    print("-" * 70)
    print(f"  Build:")
    print(f"    Time:           {result.build_time_sec:.1f}s")
    print(f"    Throughput:     {result.build_throughput_vec_per_sec:,.0f} vec/s")
    print("-" * 70)
    print(f"  Search:")
    print(f"    Recall@10:      {result.recall_at_10:.4f}")
    print(f"    Recall@1:       {result.recall_at_1:.4f}")
    print(f"    Latency p50:    {result.search_latency_p50_ms:.2f} ms")
    print(f"    Latency p95:    {result.search_latency_p95_ms:.2f} ms")
    print(f"    Latency p99:    {result.search_latency_p99_ms:.2f} ms")
    print(f"    QPS:            {result.search_qps:,.0f}")
    print("=" * 70)


def print_sweep_results(sweep: AnnBenchmarkSweepResult) -> None:
    """Print sweep results as a table."""
    print()
    print("=" * 70)
    print(f"  RECALL vs QPS CURVE — {sweep.dataset}")
    print(f"  M={sweep.M}, ef_construction={sweep.ef_construction}")
    print(f"  Build time: {sweep.build_time_sec:.1f}s")
    print("=" * 70)
    print(f"  {'ef_search':>10}  {'Recall@10':>10}  {'QPS':>10}  {'p50 (ms)':>10}")
    print("-" * 70)
    for p in sweep.points:
        print(f"  {p.ef_search:>10}  {p.recall_at_10:>10.4f}  {p.search_qps:>10.0f}  {p.search_latency_p50_ms:>10.2f}")
    print("=" * 70)


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Standard ANN Benchmark Runner for arrwDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run SIFT-1M with default parameters
  python run_ann_benchmark.py --dataset sift-128-euclidean

  # Run with competition-grade parameters
  python run_ann_benchmark.py --dataset sift-128-euclidean --M 48 --ef-construction 400 --ef-search 200

  # Generate recall-vs-QPS curve
  python run_ann_benchmark.py --dataset sift-128-euclidean --M 48 --ef-construction 400 \\
      --ef-search-sweep 10,20,50,100,200,500,1000

  # Run all datasets
  python run_ann_benchmark.py --dataset all --M 48 --ef-construction 400 --output-dir results/
        """,
    )
    parser.add_argument(
        "--dataset",
        default="sift-128-euclidean",
        choices=list(DATASETS.keys()) + ["all"],
        help="Dataset to benchmark (default: sift-128-euclidean)",
    )
    parser.add_argument("--M", type=int, default=48, help="HNSW M parameter (default: 48)")
    parser.add_argument(
        "--ef-construction", type=int, default=400, help="ef_construction (default: 400)"
    )
    parser.add_argument("--ef-search", type=int, default=200, help="ef_search (default: 200)")
    parser.add_argument(
        "--ef-search-sweep",
        type=str,
        default=None,
        help="Comma-separated ef_search values for recall-vs-QPS curve",
    )
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for multi-dataset runs")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="~/.cache/arrwdb_bench/ann",
        help="Dataset cache directory",
    )

    args = parser.parse_args()
    cache_dir = Path(args.cache_dir).expanduser()

    # Determine which datasets to run
    if args.dataset == "all":
        datasets = list(DATASETS.keys())
    else:
        datasets = [args.dataset]

    # Parse ef_search sweep values
    ef_search_values = None
    if args.ef_search_sweep:
        ef_search_values = [int(x.strip()) for x in args.ef_search_sweep.split(",")]

    logger.info(f"HNSW backend: {'Rust (PyO3)' if USING_RUST else 'Python'}")
    logger.info(f"Datasets: {datasets}")
    logger.info(f"Parameters: M={args.M}, ef_construction={args.ef_construction}, ef_search={args.ef_search}")
    if ef_search_values:
        logger.info(f"ef_search sweep: {ef_search_values}")

    all_results = []

    for dataset_name in datasets:
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting benchmark: {dataset_name}")
        logger.info(f"{'='*60}")

        # Load dataset
        train, test, ground_truth = load_dataset(dataset_name, cache_dir)

        config = AnnBenchmarkConfig(
            dataset=dataset_name,
            M=args.M,
            ef_construction=args.ef_construction,
            ef_search=args.ef_search,
        )

        if ef_search_values:
            # Sweep mode
            sweep = run_sweep_benchmark(train, test, ground_truth, config, ef_search_values)
            print_sweep_results(sweep)
            all_results.append(sweep.to_dict())

            # Save individual result
            if args.output_dir:
                out_dir = Path(args.output_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{dataset_name}_sweep.json"
                with open(out_path, "w") as f:
                    json.dump(sweep.to_dict(), f, indent=2)
                logger.info(f"Saved: {out_path}")
        else:
            # Single run mode
            result = run_single_benchmark(train, test, ground_truth, config)
            print_results(result)
            all_results.append(result.to_dict())

            # Save individual result
            if args.output_dir:
                out_dir = Path(args.output_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{dataset_name}.json"
                with open(out_path, "w") as f:
                    json.dump(result.to_dict(), f, indent=2)
                logger.info(f"Saved: {out_path}")

    # Save combined output
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(all_results if len(all_results) > 1 else all_results[0], f, indent=2)
        logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
