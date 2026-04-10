#!/bin/bash
# Launch full ANN benchmark suite - runs in background, safe to disconnect SSH
#
# This runs the standard ann-benchmarks.com datasets directly against the
# HNSW index (no HTTP API overhead), matching competitor methodology.
#
# Datasets:
#   1. SIFT-1M    (128d, 1M vectors)  - Euclidean benchmark standard
#   2. GloVe-1.2M (200d, 1.18M vectors) - NLP embeddings benchmark
#   3. Deep-1M    (96d, 1M vectors)   - Deep learning features benchmark
#
# Each dataset runs a recall-vs-QPS sweep with multiple ef_search values.
#
# Usage:
#   ./launch_ann_suite.sh                    # All datasets
#   ./launch_ann_suite.sh sift-128-euclidean # Single dataset
#
# Monitor:
#   tail -f /tmp/ann_benchmark.log
#
# Results:
#   ls /tmp/ann_results/
#
set -e

cd ~/arrwDB
source venv/bin/activate

DATASET="${1:-all}"
RESULTS_DIR="/tmp/ann_results"
LOG_FILE="/tmp/ann_benchmark.log"
M=48
EF_CONSTRUCTION=400
EF_SWEEP="10,20,50,100,200,400,800"

mkdir -p "$RESULTS_DIR"

echo "============================================="
echo "  arrwDB ANN Benchmark Suite"
echo "============================================="
echo "  Backend: $(python -c 'import sys; sys.path.insert(0,"."); from infrastructure.indexes.rust_hnsw_wrapper import RustHNSWIndexWrapper; print("Rust (PyO3)")' 2>/dev/null || echo 'Python')"
echo "  Dataset: $DATASET"
echo "  M=$M, ef_construction=$EF_CONSTRUCTION"
echo "  ef_search sweep: $EF_SWEEP"
echo "  Results: $RESULTS_DIR"
echo "  Log: $LOG_FILE"
echo "============================================="
echo ""

# Run benchmark
echo "Starting benchmark..."
echo "Monitor: tail -f $LOG_FILE"
echo ""

nohup python benchmarks/vectordbbench/run_ann_benchmark.py \
    --dataset "$DATASET" \
    --M "$M" \
    --ef-construction "$EF_CONSTRUCTION" \
    --ef-search-sweep "$EF_SWEEP" \
    --output-dir "$RESULTS_DIR" \
    > "$LOG_FILE" 2>&1 &

BENCH_PID=$!
echo "Benchmark PID: $BENCH_PID"
echo ""
echo "Running in background. Safe to disconnect."
echo "Check progress: tail -f $LOG_FILE"
echo "Check results:  ls -la $RESULTS_DIR/"
