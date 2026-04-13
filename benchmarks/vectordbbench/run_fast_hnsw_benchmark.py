#!/usr/bin/env python3
"""
FastHNSW Benchmark — direct Rust index, no Python VectorStore overhead.

Compares RustFastHNSWIndex against RustHNSWIndex on standard ANN datasets.
Uses the Rust index directly (no VectorStore, no UUID mapping during search)
to measure pure index performance.

Usage:
    python run_fast_hnsw_benchmark.py --dataset sift-128-euclidean
    python run_fast_hnsw_benchmark.py --dataset all
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from run_ann_benchmark import DATASETS, load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def benchmark_index(IndexClass, name, train, test, ground_truth, M, ef_construction, ef_sweep):
    """Run a full benchmark for one index implementation."""
    n, dim = train.shape

    logger.info(f"[{name}] Building: {n} vectors, dim={dim}, M={M}, ef_construction={ef_construction}")

    idx = IndexClass(dimension=dim, m=M, ef_construction=ef_construction, ef_search=50)

    # Build
    build_start = time.time()
    for i in range(n):
        idx.add_vector(f"v{i}", train[i])
        if (i + 1) % (n // 20) == 0:
            elapsed = time.time() - build_start
            rate = (i + 1) / elapsed
            logger.info(f"  [{name}] {i+1}/{n} ({rate:.0f} vec/s, ETA {(n-i-1)/rate:.0f}s)")

    build_time = time.time() - build_start
    logger.info(f"[{name}] Build: {build_time:.1f}s ({n/build_time:.0f} vec/s)")

    # Sweep ef_search
    points = []
    for ef in ef_sweep:
        idx.set_ef_search(ef)

        latencies = []
        recalls_10 = []
        recalls_1 = []

        for i in range(test.shape[0]):
            start = time.time()
            results = idx.search(test[i], k=10, ef_override=ef)
            latencies.append((time.time() - start) * 1000)

            result_ids = set()
            for vid, dist in results:
                try:
                    result_ids.add(int(vid[1:]))  # "v123" -> 123
                except (ValueError, IndexError):
                    pass

            gt10 = set(ground_truth[i, :10].tolist())
            recalls_10.append(len(result_ids & gt10) / 10)

            gt1 = ground_truth[i, 0]
            recalls_1.append(1.0 if gt1 in result_ids else 0.0)

        lat = np.array(latencies)
        point = {
            "ef_search": ef,
            "recall_at_10": float(np.mean(recalls_10)),
            "recall_at_1": float(np.mean(recalls_1)),
            "search_qps": 1000.0 / float(np.mean(lat)),
            "search_latency_p50_ms": float(np.percentile(lat, 50)),
        }
        points.append(point)
        logger.info(
            f"  [{name}] ef={ef}: recall@10={point['recall_at_10']:.4f} "
            f"QPS={point['search_qps']:.0f} p50={point['search_latency_p50_ms']:.2f}ms"
        )

    return {
        "implementation": name,
        "num_vectors": n,
        "dimension": dim,
        "M": M,
        "ef_construction": ef_construction,
        "build_time_sec": build_time,
        "build_throughput_vec_per_sec": n / build_time,
        "points": points,
    }


def main():
    parser = argparse.ArgumentParser(description="FastHNSW vs OldHNSW Benchmark")
    parser.add_argument("--dataset", default="sift-128-euclidean", choices=list(DATASETS.keys()) + ["all"])
    parser.add_argument("--M", type=int, default=48)
    parser.add_argument("--ef-construction", type=int, default=400)
    parser.add_argument("--ef-search-sweep", type=str, default="10,20,50,100,200,400,800")
    parser.add_argument("--output-dir", type=str, default="/tmp/fast_hnsw_results")
    parser.add_argument("--cache-dir", type=str, default="~/.cache/arrwdb_bench/ann")
    parser.add_argument("--fast-only", action="store_true", help="Only run FastHNSW (skip old)")

    args = parser.parse_args()
    cache_dir = Path(args.cache_dir).expanduser()
    ef_sweep = [int(x) for x in args.ef_search_sweep.split(",")]
    datasets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]

    from rust_hnsw import RustFastHNSWIndex, RustHNSWIndex

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    for ds_name in datasets:
        logger.info(f"\n{'='*60}")
        logger.info(f"Dataset: {ds_name}")
        logger.info(f"{'='*60}")

        train, test, gt = load_dataset(ds_name, cache_dir)
        all_results = []

        # FastHNSW
        fast_result = benchmark_index(
            RustFastHNSWIndex, "FastHNSW", train, test, gt,
            args.M, args.ef_construction, ef_sweep,
        )
        all_results.append(fast_result)

        # OldHNSW (skip if --fast-only)
        if not args.fast_only:
            old_result = benchmark_index(
                RustHNSWIndex, "OldHNSW", train, test, gt,
                args.M, args.ef_construction, ef_sweep,
            )
            all_results.append(old_result)

            # Print comparison
            print(f"\n{'='*70}")
            print(f"  {ds_name}: FastHNSW vs OldHNSW")
            print(f"{'='*70}")
            print(f"  Build: Fast={fast_result['build_time_sec']:.1f}s "
                  f"Old={old_result['build_time_sec']:.1f}s "
                  f"({old_result['build_time_sec']/fast_result['build_time_sec']:.1f}x)")
            print(f"  {'ef':>6} {'Fast R@10':>10} {'Fast QPS':>10} {'Old R@10':>10} {'Old QPS':>10} {'QPS Speedup':>12}")
            print(f"  {'-'*64}")
            for fp, op in zip(fast_result["points"], old_result["points"]):
                speedup = fp["search_qps"] / op["search_qps"] if op["search_qps"] > 0 else float("inf")
                print(f"  {fp['ef_search']:>6} {fp['recall_at_10']:>10.4f} {fp['search_qps']:>10.0f} "
                      f"{op['recall_at_10']:>10.4f} {op['search_qps']:>10.0f} {speedup:>11.1f}x")
            print(f"{'='*70}")

        out_path = Path(args.output_dir) / f"{ds_name}_fast_vs_old.json"
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
