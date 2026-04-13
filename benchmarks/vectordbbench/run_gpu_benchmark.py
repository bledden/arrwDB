#!/usr/bin/env python3
"""
GPU CAGRA Benchmark for arrwDB.

Runs the same SIFT-1M, GloVe-1.2M, Deep-1M datasets through NVIDIA CAGRA
(via FAISS-GPU) and compares against CPU HNSW results.

Requires: conda install -c pytorch -c nvidia -c rapidsai -c conda-forge faiss-gpu-cuvs

Usage:
    python run_gpu_benchmark.py --dataset sift-128-euclidean
    python run_gpu_benchmark.py --dataset all --output-dir /tmp/gpu_results/
    python run_gpu_benchmark.py --dataset sift-128-euclidean --itopk-sweep 32,64,128,256,512
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Reuse dataset loading from the ANN benchmark
sys.path.insert(0, str(Path(__file__).parent))
from run_ann_benchmark import DATASETS, load_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class GPUBenchmarkPoint:
    """A single point on the recall-vs-QPS curve."""
    itopk_size: int
    recall_at_10: float
    recall_at_1: float
    search_qps: float
    search_latency_p50_ms: float


@dataclass
class GPUBenchmarkResult:
    """Results from a GPU CAGRA benchmark."""
    dataset: str
    num_vectors: int
    dimension: int
    algorithm: str  # "cagra"
    graph_degree: int
    intermediate_graph_degree: int
    build_time_sec: float
    build_throughput_vec_per_sec: float
    gpu_name: str
    points: List[GPUBenchmarkPoint] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["points"] = [asdict(p) for p in self.points]
        return d


def get_gpu_name() -> str:
    """Get the GPU device name."""
    try:
        import faiss
        res = faiss.StandardGpuResources()
        # Try to get GPU name via nvidia-smi
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            return result.stdout.strip().split("\n")[0]
    except Exception:
        pass
    return "unknown"


def run_gpu_benchmark(
    train: np.ndarray,
    test: np.ndarray,
    ground_truth: np.ndarray,
    dataset_name: str,
    graph_degree: int = 64,
    intermediate_graph_degree: int = 128,
    itopk_values: List[int] = None,
) -> GPUBenchmarkResult:
    """Run GPU CAGRA benchmark with itopk sweep."""
    import faiss

    if itopk_values is None:
        itopk_values = [32, 64, 128, 256, 512]

    n, d = train.shape
    gpu_name = get_gpu_name()

    logger.info(f"GPU: {gpu_name}")
    logger.info(f"Building CAGRA index: {n} vectors, dim={d}")
    logger.info(f"  graph_degree={graph_degree}, intermediate={intermediate_graph_degree}")

    # For cosine similarity with normalized vectors, use inner product
    metric = faiss.METRIC_INNER_PRODUCT

    config = faiss.GpuIndexCagraConfig()
    config.graph_degree = graph_degree
    config.intermediate_graph_degree = intermediate_graph_degree

    res = faiss.StandardGpuResources()

    # Build
    build_start = time.time()
    index = faiss.GpuIndexCagra(res, d, metric, config)
    index.train(train)
    index.add(train)
    build_time = time.time() - build_start

    logger.info(f"Build complete: {build_time:.2f}s ({n / build_time:.0f} vec/s)")

    result = GPUBenchmarkResult(
        dataset=dataset_name,
        num_vectors=n,
        dimension=d,
        algorithm="cagra",
        graph_degree=graph_degree,
        intermediate_graph_degree=intermediate_graph_degree,
        build_time_sec=build_time,
        build_throughput_vec_per_sec=n / build_time,
        gpu_name=gpu_name,
    )

    # Sweep itopk_size (analogous to ef_search for HNSW)
    k = 10
    for itopk in sorted(itopk_values):
        logger.info(f"Sweep: itopk_size={itopk}")

        search_params = faiss.SearchParametersCagra()
        search_params.itopk_size = itopk

        # Warmup
        _ = index.search(test[:10], k, params=search_params)

        # Timed search
        latencies = []
        all_I = np.zeros((test.shape[0], k), dtype=np.int64)

        for i in range(test.shape[0]):
            q = test[i:i+1]
            start = time.time()
            D, I = index.search(q, k, params=search_params)
            latency_ms = (time.time() - start) * 1000
            latencies.append(latency_ms)
            all_I[i] = I[0]

        latencies = np.array(latencies)
        p50 = float(np.percentile(latencies, 50))
        qps = 1000.0 / float(np.mean(latencies))

        # Recall
        recalls_10 = []
        recalls_1 = []
        for i in range(test.shape[0]):
            gt_set = set(ground_truth[i, :10].tolist())
            result_set = set(all_I[i].tolist())
            recalls_10.append(len(gt_set & result_set) / 10)

            gt_top1 = ground_truth[i, 0]
            recalls_1.append(1.0 if all_I[i][0] == gt_top1 else 0.0)

        recall_10 = float(np.mean(recalls_10))
        recall_1 = float(np.mean(recalls_1))

        point = GPUBenchmarkPoint(
            itopk_size=itopk,
            recall_at_10=recall_10,
            recall_at_1=recall_1,
            search_qps=qps,
            search_latency_p50_ms=p50,
        )
        result.points.append(point)

        logger.info(
            f"  itopk={itopk}: recall@10={recall_10:.4f}, "
            f"QPS={qps:.0f}, p50={p50:.2f}ms"
        )

    # Also run batch search for throughput measurement
    logger.info("Batch search throughput test...")
    search_params = faiss.SearchParametersCagra()
    search_params.itopk_size = 128

    # Warmup
    _ = index.search(test[:100], k, params=search_params)

    batch_start = time.time()
    D_batch, I_batch = index.search(test, k, params=search_params)
    batch_time = time.time() - batch_start
    batch_qps = test.shape[0] / batch_time

    logger.info(f"Batch QPS ({test.shape[0]} queries): {batch_qps:.0f} ({batch_time:.3f}s)")

    return result


def print_gpu_results(result: GPUBenchmarkResult) -> None:
    """Print GPU benchmark results."""
    print()
    print("=" * 70)
    print(f"  GPU CAGRA BENCHMARK — {result.dataset}")
    print(f"  GPU: {result.gpu_name}")
    print(f"  graph_degree={result.graph_degree}, "
          f"intermediate={result.intermediate_graph_degree}")
    print(f"  Build time: {result.build_time_sec:.2f}s "
          f"({result.build_throughput_vec_per_sec:.0f} vec/s)")
    print("=" * 70)
    print(f"  {'itopk':>10}  {'Recall@10':>10}  {'Recall@1':>10}  "
          f"{'QPS':>10}  {'p50 (ms)':>10}")
    print("-" * 70)
    for p in result.points:
        print(f"  {p.itopk_size:>10}  {p.recall_at_10:>10.4f}  "
              f"{p.recall_at_1:>10.4f}  {p.search_qps:>10.0f}  "
              f"{p.search_latency_p50_ms:>10.2f}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="GPU CAGRA Benchmark")
    parser.add_argument(
        "--dataset",
        default="sift-128-euclidean",
        choices=list(DATASETS.keys()) + ["all"],
    )
    parser.add_argument("--graph-degree", type=int, default=64)
    parser.add_argument("--intermediate-graph-degree", type=int, default=128)
    parser.add_argument(
        "--itopk-sweep",
        type=str,
        default="32,64,128,256,512",
        help="Comma-separated itopk_size values",
    )
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="~/.cache/arrwdb_bench/ann",
    )

    args = parser.parse_args()
    cache_dir = Path(args.cache_dir).expanduser()

    itopk_values = [int(x.strip()) for x in args.itopk_sweep.split(",")]

    datasets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]

    all_results = []

    for dataset_name in datasets:
        logger.info(f"\n{'='*60}")
        logger.info(f"GPU CAGRA Benchmark: {dataset_name}")
        logger.info(f"{'='*60}")

        train, test, ground_truth = load_dataset(dataset_name, cache_dir)

        result = run_gpu_benchmark(
            train, test, ground_truth,
            dataset_name=dataset_name,
            graph_degree=args.graph_degree,
            intermediate_graph_degree=args.intermediate_graph_degree,
            itopk_values=itopk_values,
        )
        print_gpu_results(result)
        all_results.append(result.to_dict())

        if args.output_dir:
            out_dir = Path(args.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{dataset_name}_gpu_cagra.json"
            with open(out_path, "w") as f:
                json.dump(result.to_dict(), f, indent=2)
            logger.info(f"Saved: {out_path}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(
                all_results if len(all_results) > 1 else all_results[0],
                f, indent=2,
            )
        logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
