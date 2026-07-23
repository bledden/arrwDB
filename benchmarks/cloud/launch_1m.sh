#!/bin/bash
# Launch 1M benchmark - runs in background, safe to disconnect SSH
set -e

cd ~/arrwDB
source venv/bin/activate

# Kill any existing server
pkill -f run_api.py 2>/dev/null || true
sleep 2

# Cohere key must come from the environment (or be loaded from .env by the server)
export COHERE_API_KEY="${COHERE_API_KEY:?COHERE_API_KEY must be set in the environment}"

# Start server in background
echo "Starting arrwDB server..."
nohup python run_api.py > /tmp/arrwdb_server.log 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Wait for server to be ready
echo "Waiting for server..."
for i in $(seq 1 30); do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "Server is ready!"
        break
    fi
    sleep 2
done

# Verify
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "ERROR: Server failed to start"
    cat /tmp/arrwdb_server.log
    exit 1
fi

# Kick off benchmark with nohup (safe to disconnect)
echo ""
echo "Starting 1M vector benchmark..."
echo "This will take several hours."
echo "Monitor: tail -f /tmp/benchmark_cloud.log"
echo ""
nohup python benchmarks/vectordbbench/run_benchmark.py \
    --url http://localhost:8000 \
    --dataset random \
    --dim 1024 \
    --size 1000000 \
    --queries 100 \
    --index-type hnsw \
    --timeout 1800 \
    --clear-checkpoint \
    --output /tmp/benchmark_cloud_1m.json > /tmp/benchmark_cloud.log 2>&1 &
BENCH_PID=$!
echo "Benchmark PID: $BENCH_PID"
echo "Server PID: $SERVER_PID"
echo ""
echo "Both processes running in background. Safe to disconnect."
echo "Check progress: tail -f /tmp/benchmark_cloud.log"
echo "Check results:  cat /tmp/benchmark_cloud_1m.json"
