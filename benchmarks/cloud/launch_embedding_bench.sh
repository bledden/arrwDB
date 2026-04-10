#!/bin/bash
# Launch embedding provider benchmark - runs in background
#
# Tests HNSW with real embeddings from NV-Embed-v2 and Voyage AI.
# Requires API keys set in environment or .env file.
#
# Usage:
#   ./launch_embedding_bench.sh              # All available providers
#   ./launch_embedding_bench.sh nvidia       # NV-Embed-v2 only
#   ./launch_embedding_bench.sh voyage       # Voyage AI only
#
# Monitor:
#   tail -f /tmp/embedding_benchmark.log
#
set -e

cd ~/arrwDB
source venv/bin/activate

# Load .env if present
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

PROVIDER="${1:-all}"
SIZE="${2:-10000}"
RESULTS_DIR="/tmp/embedding_results"
LOG_FILE="/tmp/embedding_benchmark.log"

mkdir -p "$RESULTS_DIR"

echo "============================================="
echo "  arrwDB Embedding Provider Benchmark"
echo "============================================="
echo "  Provider: $PROVIDER"
echo "  Corpus size: $SIZE"
echo "  NGC key: ${NGC_API_KEY:+set}${NGC_API_KEY:-NOT SET}"
echo "  Voyage key: ${VOYAGE_API_KEY:+set}${VOYAGE_API_KEY:-NOT SET}"
echo "  Results: $RESULTS_DIR"
echo "  Log: $LOG_FILE"
echo "============================================="
echo ""

nohup python benchmarks/vectordbbench/run_embedding_benchmark.py \
    --provider "$PROVIDER" \
    --size "$SIZE" \
    --queries 100 \
    --M 48 \
    --ef-construction 400 \
    --ef-search 200 \
    --output "$RESULTS_DIR/embedding_${PROVIDER}_${SIZE}.json" \
    > "$LOG_FILE" 2>&1 &

BENCH_PID=$!
echo "Benchmark PID: $BENCH_PID"
echo ""
echo "Running in background. Safe to disconnect."
echo "Check progress: tail -f $LOG_FILE"
