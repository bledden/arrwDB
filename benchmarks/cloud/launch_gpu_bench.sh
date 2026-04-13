#!/bin/bash
# Launch GPU CAGRA benchmark — runs in background, safe to disconnect
#
# Runs SIFT-1M, GloVe-1.2M, Deep-1M through NVIDIA CAGRA on GPU
# and generates recall-vs-QPS curves comparable to CPU HNSW results.
#
# Usage:
#   ./launch_gpu_bench.sh                      # All datasets
#   ./launch_gpu_bench.sh sift-128-euclidean   # Single dataset
#
# Monitor:
#   tail -f /tmp/gpu_benchmark.log
#
set -e

# Activate conda env
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate arrwdb-gpu

cd ~/arrwDB

DATASET="${1:-all}"
RESULTS_DIR="/tmp/gpu_results"
LOG_FILE="/tmp/gpu_benchmark.log"

mkdir -p "$RESULTS_DIR"

echo "============================================="
echo "  arrwDB GPU CAGRA Benchmark"
echo "============================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "  Dataset: $DATASET"
echo "  Results: $RESULTS_DIR"
echo "  Log: $LOG_FILE"
echo "============================================="
echo ""

nohup python benchmarks/vectordbbench/run_gpu_benchmark.py \
    --dataset "$DATASET" \
    --graph-degree 64 \
    --intermediate-graph-degree 128 \
    --itopk-sweep "32,64,128,256,512" \
    --output-dir "$RESULTS_DIR" \
    > "$LOG_FILE" 2>&1 &

BENCH_PID=$!
echo "Benchmark PID: $BENCH_PID"
echo ""
echo "Running in background. Safe to disconnect."
echo "Check progress: tail -f $LOG_FILE"
echo "Check results:  ls -la $RESULTS_DIR/"
