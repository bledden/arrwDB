# arrwDB Benchmarks

Performance benchmarking tools for arrwDB vector database.

## VectorDBBench Integration

arrwDB can be benchmarked using [VectorDBBench](https://github.com/zilliztech/VectorDBBench), the industry-standard vector database benchmark tool.

### Setup

1. Install VectorDBBench:
```bash
pip install vectordb-bench
```

2. Copy the arrwDB adapter:
```bash
cp arrwdb_client.py /path/to/vectordb_bench/backend/clients/arrwdb/
```

3. Register in VectorDBBench (edit `vectordb_bench/backend/clients/__init__.py`):
```python
from .arrwdb.arrwdb_client import ArrwDB, ArrwDBConfig
```

4. Run benchmarks:
```bash
python -m vectordb_bench --db arrwdb --url http://localhost:8000
```

## Standalone Benchmark

For quick internal testing without VectorDBBench:

```bash
# With SIFT-10K dataset (auto-downloads)
python run_benchmark.py --url http://localhost:8000 --dataset sift-10k

# With random vectors
python run_benchmark.py --url http://localhost:8000 --dataset random --dim 768 --size 100000

# Test different index types
python run_benchmark.py --url http://localhost:8000 --index-type ivf

# Save results to JSON
python run_benchmark.py --url http://localhost:8000 --output results/hnsw_10k.json
```

### Metrics Collected

| Metric | Description |
|--------|-------------|
| `insert_time_sec` | Total time to insert all vectors |
| `insert_throughput_vec_per_sec` | Vectors inserted per second |
| `search_latency_p50_ms` | 50th percentile search latency |
| `search_latency_p95_ms` | 95th percentile search latency |
| `search_latency_p99_ms` | 99th percentile search latency |
| `search_qps` | Queries per second |
| `recall_at_10` | Recall@10 against ground truth |

### Novel Feature Metrics

| Metric | Description |
|--------|-------------|
| `temp_search_latency_p50_ms` | Temperature search latency (temp=1.5) |
| `temp_search_diversity_score` | Result diversity from temperature search |
| `index_recommendation` | Index Oracle recommendation |
| `embedding_health_score` | Embedding quality score (0-1) |

## Example Results

Benchmark on M2 MacBook Pro (10,000 vectors, 768 dimensions):

```
============================================================
BENCHMARK RESULTS
============================================================
Dataset:        random
Vectors:        10,000
Dimension:      768
Index Type:     hnsw
------------------------------------------------------------
Insert Time:    2.34s
Insert Rate:    4,274 vec/s
------------------------------------------------------------
Search p50:     1.23ms
Search p95:     2.45ms
Search p99:     3.12ms
Search QPS:     812
Recall@10:      0.982
------------------------------------------------------------
Novel Features:
  Temp Search p50:  1.45ms
  Diversity Score:  0.0234
  Index Oracle:     hnsw
  Health Score:     0.987
============================================================
```

## Comparing with Other Databases

To compare arrwDB with other vector databases using VectorDBBench:

```bash
# Run standard benchmark suite
vectordb-bench run --db arrwdb,milvus,qdrant,pinecone \
  --dataset cohere-medium-10m \
  --output-dir results/

# View results
vectordb-bench view results/
```

## CI/CD Integration

Add to your CI pipeline:

```yaml
benchmark:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4

    - name: Start arrwDB
      run: docker-compose up -d arrwdb

    - name: Run benchmarks
      run: |
        pip install numpy
        python benchmarks/vectordbbench/run_benchmark.py \
          --url http://localhost:8000 \
          --dataset random \
          --size 10000 \
          --output benchmark-results.json

    - name: Upload results
      uses: actions/upload-artifact@v4
      with:
        name: benchmark-results
        path: benchmark-results.json
```

## License

Apache 2.0
