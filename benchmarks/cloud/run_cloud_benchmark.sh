#!/bin/bash
# Cloud Benchmark Runner
set -e

cd ~/arrwDB
source venv/bin/activate

echo "=== arrwDB Cloud Benchmark ==="
echo "Machine: $(nproc) cores, $(free -h | awk '/^Mem:/ {print $2}') RAM"
echo ""

# Start server in background
echo "Starting arrwDB server..."
nohup python run_api.py > /tmp/arrwdb_server.log 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Wait for server to be ready
echo "Waiting for server to start..."
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "Server is ready!"
        break
    fi
    sleep 2
done

# Verify server is running
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "Error: Server failed to start"
    cat /tmp/arrwdb_server.log
    exit 1
fi

# Run 1M vector benchmark
echo ""
echo "Starting 1M vector benchmark..."
echo "This will take several hours. Logs: /tmp/benchmark_cloud.log"
echo ""

python benchmarks/vectordbbench/run_benchmark.py \
    --url http://localhost:8000 \
    --dataset random \
    --dim 1024 \
    --size 1000000 \
    --queries 100 \
    --index-type hnsw \
    --timeout 600 \
    --output /tmp/benchmark_cloud_1m.json 2>&1 | tee /tmp/benchmark_cloud.log

# Print results
echo ""
echo "=== BENCHMARK COMPLETE ==="
cat /tmp/benchmark_cloud_1m.json

# Cleanup
kill $SERVER_PID 2>/dev/null || true
echo ""
echo "Results saved to: /tmp/benchmark_cloud_1m.json"
echo "Copy results with: gcloud compute scp $HOSTNAME:/tmp/benchmark_cloud_1m.json ."
