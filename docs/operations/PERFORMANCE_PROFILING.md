# Performance Profiling Guide for arrwDB

This guide covers performance profiling, load testing, and optimization for arrwDB.

## Table of Contents

- [Overview](#overview)
- [CPU Profiling with py-spy](#cpu-profiling-with-py-spy)
- [Memory Profiling](#memory-profiling)
- [Load Testing with Locust](#load-testing-with-locust)
- [API Performance Testing with K6](#api-performance-testing-with-k6)
- [Database Query Optimization](#database-query-optimization)
- [Caching Strategies](#caching-strategies)
- [Performance Benchmarking](#performance-benchmarking)

## Overview

Performance profiling helps identify:

- **CPU bottlenecks**: Functions consuming most CPU time
- **Memory leaks**: Objects not being garbage collected
- **Slow queries**: Database operations taking too long
- **Concurrency issues**: Lock contention and thread starvation
- **I/O bottlenecks**: Disk and network latency

## CPU Profiling with py-spy

### Installation

```bash
pip install py-spy
```

### Basic CPU Profiling

```bash
# Profile a running process
py-spy top --pid 12345

# Record CPU profile
py-spy record -o profile.svg --pid 12345

# Profile with native extensions
py-spy record -o profile.svg --native --pid 12345

# Profile for specific duration
py-spy record -o profile.svg --duration 60 --pid 12345
```

### Flamegraph Analysis

```bash
# Generate flamegraph
py-spy record -o flamegraph.svg --format flamegraph --pid 12345

# Open in browser
open flamegraph.svg
```

### Profiling During Load Test

```bash
# Start arrwDB
python3 -m uvicorn app.api.main:app --host 0.0.0.0 --port 8000 &
APP_PID=$!

# Wait for startup
sleep 3

# Start profiling in background
py-spy record -o profile.svg --pid $APP_PID --duration 300 &
PROFILER_PID=$!

# Run load test
locust -f loadtest.py --headless -u 100 -r 10 --run-time 5m

# Wait for profiler to finish
wait $PROFILER_PID

# View results
open profile.svg
```

### Python Code Profiling

For in-code profiling, use `cProfile`:

```python
import cProfile
import pstats
from pstats import SortKey

def profile_search():
    """Profile the search function."""
    profiler = cProfile.Profile()
    profiler.enable()

    # Your search code here
    results = library.search(embedding, k=10)

    profiler.disable()

    # Print stats
    stats = pstats.Stats(profiler)
    stats.sort_stats(SortKey.CUMULATIVE)
    stats.print_stats(20)  # Top 20 functions
```

### Line-by-Line Profiling

```bash
pip install line_profiler
```

```python
from line_profiler import LineProfiler

@profile
def search_documents(query: str, k: int = 10):
    """Search documents (line-by-line profiled)."""
    embedding = embed_query(query)
    results = vector_search(embedding, k)
    reranked = rerank_results(results)
    return reranked

# Run with:
# kernprof -l -v your_script.py
```

## Memory Profiling

### Installation

```bash
pip install memory_profiler
```

### Basic Memory Profiling

```python
from memory_profiler import profile

@profile
def process_large_batch(documents: list):
    """Process large batch of documents."""
    embeddings = []
    for doc in documents:
        embedding = generate_embedding(doc)
        embeddings.append(embedding)
    return embeddings

# Run with:
# python -m memory_profiler your_script.py
```

### Memory Growth Tracking

```bash
# Track memory over time
mprof run python3 -m uvicorn app.api.main:app

# Plot memory usage
mprof plot
```

### Finding Memory Leaks

```python
import tracemalloc

# Start tracking
tracemalloc.start()

# Your code here
result = process_data()

# Get snapshot
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')

# Print top 10 memory allocations
for stat in top_stats[:10]:
    print(stat)
```

### Object Tracking with objgraph

```bash
pip install objgraph
```

```python
import objgraph

# Show most common types
objgraph.show_most_common_types(limit=10)

# Find memory leaks
objgraph.show_growth(limit=10)

# Generate reference graph
objgraph.show_refs([my_object], filename='refs.png')
```

## Load Testing with Locust

### Installation

```bash
pip install locust
```

### Basic Load Test

Create `locustfile.py`:

```python
from locust import HttpUser, task, between
import random

class arrwDBUser(HttpUser):
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    host = "http://localhost:8000"

    def on_start(self):
        """Setup - runs once per user."""
        # Create a test library
        response = self.client.post("/v1/libraries", json={
            "name": f"test-library-{self.id}",
            "dimension": 1536,
            "index_type": "hnsw"
        })
        self.library_id = response.json()["id"]

    @task(10)  # Weight: 10x more common than other tasks
    def search_documents(self):
        """Simulate search operations."""
        query = random.choice([
            "machine learning algorithms",
            "vector database performance",
            "neural network architectures",
            "natural language processing",
        ])

        self.client.post(f"/v1/libraries/{self.library_id}/search", json={
            "query": query,
            "k": 10
        })

    @task(5)
    def add_document(self):
        """Simulate document additions."""
        self.client.post(f"/v1/libraries/{self.library_id}/documents", json={
            "title": f"Document {random.randint(1, 10000)}",
            "text": "This is test content for load testing. " * 20,
            "metadata": {"source": "load_test"}
        })

    @task(2)
    def get_library_stats(self):
        """Get library statistics."""
        self.client.get(f"/v1/libraries/{self.library_id}/statistics")

    @task(1)
    def list_documents(self):
        """List documents in library."""
        self.client.get(
            f"/v1/libraries/{self.library_id}/documents",
            params={"limit": 100}
        )
```

### Running Load Tests

```bash
# Web UI (recommended)
locust -f locustfile.py --host=http://localhost:8000

# Headless mode
locust -f locustfile.py --headless \
    --users 100 \
    --spawn-rate 10 \
    --run-time 10m \
    --host=http://localhost:8000

# Export results
locust -f locustfile.py --headless \
    --users 100 \
    --spawn-rate 10 \
    --run-time 5m \
    --html=report.html \
    --csv=results
```

### Distributed Load Testing

```bash
# Start master node
locust -f locustfile.py --master --host=http://api.production.com

# Start worker nodes (run on multiple machines)
locust -f locustfile.py --worker --master-host=master-ip

# Or with Docker
docker run -p 8089:8089 -v $PWD:/mnt/locust locustio/locust \
    -f /mnt/locust/locustfile.py \
    --master \
    --host=http://api.production.com
```

### Advanced Scenarios

```python
from locust import HttpUser, task, between, events
from locust.runners import MasterRunner
import time

class AdvancedUser(HttpUser):
    wait_time = between(0.5, 2)

    @task
    def complex_workflow(self):
        """Simulate a complex user workflow."""

        # 1. Create library
        start_time = time.time()
        library_response = self.client.post("/v1/libraries", json={
            "name": f"workflow-{time.time()}",
            "dimension": 1536,
            "index_type": "hnsw"
        })
        library_id = library_response.json()["id"]

        # 2. Add multiple documents
        for i in range(5):
            self.client.post(f"/v1/libraries/{library_id}/documents", json={
                "title": f"Doc {i}",
                "text": "Content here " * 50
            })

        # 3. Perform searches
        for query in ["test query 1", "test query 2"]:
            self.client.post(f"/v1/libraries/{library_id}/search", json={
                "query": query,
                "k": 10
            })

        # 4. Cleanup
        self.client.delete(f"/v1/libraries/{library_id}")

        # Record custom metric
        total_time = (time.time() - start_time) * 1000
        events.request.fire(
            request_type="workflow",
            name="complete_workflow",
            response_time=total_time,
            response_length=0,
            exception=None,
            context={}
        )

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Runs once before test starts."""
    print("Load test starting...")

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Runs once after test completes."""
    print("Load test finished!")
```

## API Performance Testing with K6

### Installation

```bash
# macOS
brew install k6

# Linux
sudo gpg -k
sudo gpg --no-default-keyring --keyring /usr/share/keyrings/k6-archive-keyring.gpg --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C5AD17C747E3415A3642D57D77C6C491D6AC1D69
echo "deb [signed-by=/usr/share/keyrings/k6-archive-keyring.gpg] https://dl.k6.io/deb stable main" | sudo tee /etc/apt/sources.list.d/k6.list
sudo apt-get update
sudo apt-get install k6
```

### Basic Load Test

Create `k6-test.js`:

```javascript
import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const searchDuration = new Trend('search_duration');

// Test configuration
export let options = {
    stages: [
        { duration: '2m', target: 50 },   // Ramp up to 50 users
        { duration: '5m', target: 50 },   // Stay at 50 users
        { duration: '2m', target: 100 },  // Ramp up to 100 users
        { duration: '5m', target: 100 },  // Stay at 100 users
        { duration: '2m', target: 0 },    // Ramp down to 0
    ],
    thresholds: {
        'http_req_duration': ['p(95)<500'],  // 95% of requests under 500ms
        'errors': ['rate<0.01'],              // Error rate under 1%
    },
};

const BASE_URL = 'http://localhost:8000';
const API_KEY = 'your-api-key-here';

export function setup() {
    // Setup: Create test library
    const response = http.post(`${BASE_URL}/v1/libraries`, JSON.stringify({
        name: `k6-test-${Date.now()}`,
        dimension: 1536,
        index_type: 'hnsw'
    }), {
        headers: {
            'Content-Type': 'application/json',
            'X-API-Key': API_KEY,
        },
    });

    return { libraryId: response.json('id') };
}

export default function(data) {
    const { libraryId } = data;

    // Test 1: Search
    let searchResponse = http.post(
        `${BASE_URL}/v1/libraries/${libraryId}/search`,
        JSON.stringify({
            query: 'test query',
            k: 10
        }),
        {
            headers: {
                'Content-Type': 'application/json',
                'X-API-Key': API_KEY,
            },
            tags: { name: 'search' },
        }
    );

    // Record metrics
    searchDuration.add(searchResponse.timings.duration);
    errorRate.add(searchResponse.status !== 200);

    // Validate response
    check(searchResponse, {
        'status is 200': (r) => r.status === 200,
        'has results': (r) => r.json('results') !== undefined,
        'response time < 500ms': (r) => r.timings.duration < 500,
    });

    // Test 2: Add document
    let addResponse = http.post(
        `${BASE_URL}/v1/libraries/${libraryId}/documents`,
        JSON.stringify({
            title: 'Test Document',
            text: 'This is test content.',
            metadata: { test: true }
        }),
        {
            headers: {
                'Content-Type': 'application/json',
                'X-API-Key': API_KEY,
            },
            tags: { name: 'add_document' },
        }
    );

    check(addResponse, {
        'document added': (r) => r.status === 201,
    });

    sleep(1);  // Wait 1 second between iterations
}

export function teardown(data) {
    // Cleanup: Delete test library
    http.del(`${BASE_URL}/v1/libraries/${data.libraryId}`, null, {
        headers: { 'X-API-Key': API_KEY },
    });
}
```

### Running K6 Tests

```bash
# Run test
k6 run k6-test.js

# Run with more virtual users
k6 run --vus 100 --duration 30s k6-test.js

# Output to InfluxDB
k6 run --out influxdb=http://localhost:8086/k6 k6-test.js

# Cloud execution
k6 cloud k6-test.js

# Generate HTML report
k6 run k6-test.js --out json=results.json
k6 report results.json
```

### Spike Test

```javascript
export let options = {
    stages: [
        { duration: '10s', target: 100 },   // Fast ramp-up
        { duration: '1m', target: 100 },    // Stay at peak
        { duration: '10s', target: 0 },     // Fast ramp-down
    ],
};
```

### Stress Test

```javascript
export let options = {
    stages: [
        { duration: '2m', target: 100 },
        { duration: '5m', target: 100 },
        { duration: '2m', target: 200 },
        { duration: '5m', target: 200 },
        { duration: '2m', target: 300 },
        { duration: '5m', target: 300 },
        { duration: '10m', target: 0 },
    ],
};
```

### Soak Test (Long Duration)

```javascript
export let options = {
    stages: [
        { duration: '5m', target: 50 },    // Ramp up
        { duration: '12h', target: 50 },   // Stay at load for 12 hours
        { duration: '5m', target: 0 },     // Ramp down
    ],
};
```

## Database Query Optimization

### Query Analysis

```python
import time
from functools import wraps

def profile_query(func):
    """Decorator to profile database queries."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start

        if duration > 0.1:  # Log slow queries (>100ms)
            logger.warning(
                f"Slow query detected: {func.__name__}",
                extra={"duration_ms": duration * 1000}
            )

        return result
    return wrapper

@profile_query
def search_vectors(embedding: list, k: int):
    """Search for similar vectors."""
    return index.search(embedding, k)
```

### Index Optimization

```python
# Monitor index performance
def benchmark_index_types():
    """Benchmark different index types."""
    results = {}

    for index_type in ["hnsw", "ivf", "brute_force"]:
        library = create_library(index_type=index_type)

        # Add documents
        start = time.time()
        for i in range(1000):
            library.add_document(generate_random_document())
        add_time = time.time() - start

        # Search
        start = time.time()
        for i in range(100):
            library.search(generate_random_embedding(), k=10)
        search_time = time.time() - start

        results[index_type] = {
            "add_time": add_time,
            "search_time": search_time,
            "avg_search_ms": (search_time / 100) * 1000
        }

    return results
```

## Caching Strategies

### Embedding Cache

```python
from functools import lru_cache
import hashlib

class EmbeddingCache:
    """LRU cache for embeddings."""

    @lru_cache(maxsize=10000)
    def get_embedding(self, text: str) -> list:
        """Get cached embedding or generate new one."""
        text_hash = hashlib.sha256(text.encode()).hexdigest()

        # Check cache
        if text_hash in self.cache:
            return self.cache[text_hash]

        # Generate embedding
        embedding = self.embedding_service.embed(text)

        # Store in cache
        self.cache[text_hash] = embedding

        return embedding
```

### Redis Caching

```python
import redis
import json

class RedisCache:
    """Redis-based caching for search results."""

    def __init__(self):
        self.redis = redis.Redis(host='localhost', port=6379, db=0)
        self.ttl = 3600  # 1 hour

    def get_search_results(self, query_hash: str):
        """Get cached search results."""
        cached = self.redis.get(f"search:{query_hash}")
        if cached:
            return json.loads(cached)
        return None

    def set_search_results(self, query_hash: str, results: list):
        """Cache search results."""
        self.redis.setex(
            f"search:{query_hash}",
            self.ttl,
            json.dumps(results)
        )
```

## Performance Benchmarking

### Automated Benchmark Suite

Create `benchmark.py`:

```python
"""arrwDB performance benchmark suite."""

import time
import statistics
import numpy as np
from typing import List, Dict

class Benchmark:
    """Performance benchmark suite."""

    def __init__(self):
        self.results = {}

    def benchmark_vector_search(self, dimensions: int = 1536, k: int = 10):
        """Benchmark vector search performance."""
        library = create_test_library(dimension=dimensions)

        # Add test data
        for i in range(10000):
            library.add_document(generate_random_document(dimensions))

        # Warmup
        for i in range(100):
            library.search(generate_random_embedding(dimensions), k=k)

        # Benchmark
        durations = []
        for i in range(1000):
            start = time.time()
            library.search(generate_random_embedding(dimensions), k=k)
            durations.append(time.time() - start)

        return {
            "mean_ms": statistics.mean(durations) * 1000,
            "p50_ms": statistics.median(durations) * 1000,
            "p95_ms": np.percentile(durations, 95) * 1000,
            "p99_ms": np.percentile(durations, 99) * 1000,
        }

    def benchmark_document_indexing(self):
        """Benchmark document indexing speed."""
        library = create_test_library()

        documents = [generate_random_document() for _ in range(1000)]

        start = time.time()
        for doc in documents:
            library.add_document(doc)
        total_time = time.time() - start

        return {
            "total_time_s": total_time,
            "docs_per_second": len(documents) / total_time,
            "avg_time_per_doc_ms": (total_time / len(documents)) * 1000,
        }

    def run_all_benchmarks(self):
        """Run all benchmarks."""
        print("Running arrwDB Performance Benchmarks...")

        # Vector search
        print("\n1. Vector Search Performance:")
        for dims in [384, 768, 1536]:
            results = self.benchmark_vector_search(dimensions=dims)
            print(f"  Dimension {dims}:")
            print(f"    Mean: {results['mean_ms']:.2f}ms")
            print(f"    P95: {results['p95_ms']:.2f}ms")
            print(f"    P99: {results['p99_ms']:.2f}ms")

        # Document indexing
        print("\n2. Document Indexing Performance:")
        results = self.benchmark_document_indexing()
        print(f"  Documents/sec: {results['docs_per_second']:.2f}")
        print(f"  Avg time per doc: {results['avg_time_per_doc_ms']:.2f}ms")

if __name__ == "__main__":
    benchmark = Benchmark()
    benchmark.run_all_benchmarks()
```

### Running Benchmarks

```bash
# Run full benchmark suite
python benchmark.py

# Run specific benchmark
python benchmark.py --test vector_search

# Compare before/after optimization
python benchmark.py --output before.json
# ... make optimizations ...
python benchmark.py --output after.json
python compare_benchmarks.py before.json after.json
```

## Performance Optimization Checklist

- [ ] CPU profiling completed (py-spy)
- [ ] Memory profiling completed (memory_profiler)
- [ ] Memory leaks identified and fixed
- [ ] Load tests passing (Locust)
- [ ] API performance tests passing (K6)
- [ ] P95 latency < 500ms
- [ ] P99 latency < 2s
- [ ] Slow queries identified and optimized
- [ ] Embedding caching implemented
- [ ] Search result caching implemented
- [ ] Database indexes optimized
- [ ] N+1 query problems resolved
- [ ] Benchmark suite automated
- [ ] Performance regression tests in CI/CD

## Additional Resources

- [py-spy Documentation](https://github.com/benfred/py-spy)
- [memory_profiler Documentation](https://pypi.org/project/memory-profiler/)
- [Locust Documentation](https://docs.locust.io/)
- [K6 Documentation](https://k6.io/docs/)
- [Python Performance Tips](https://wiki.python.org/moin/PythonSpeed/PerformanceTips)
- [FastAPI Performance](https://fastapi.tiangolo.com/deployment/concepts/)
