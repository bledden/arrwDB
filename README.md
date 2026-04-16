# arrwDB

Vector database with a Rust core, Python API, and GPU acceleration.

## Benchmark Results

Tested on standard ANN datasets (SIFT-1M, GloVe-1.2M, Deep-1M) used by ann-benchmarks.com.

### CPU Search (FastHNSW, Rust, AVX-512/AVX2/FMA)

| Dataset | Recall@10 | QPS | Build Time | Hardware |
|---------|-----------|-----|------------|----------|
| SIFT-1M (128d) | 0.999 | 1,793 | 14 min | r6i.16xlarge |
| Deep-1M (96d) | 0.998 | 2,038 | 13 min | r6i.16xlarge |
| GloVe-1.2M (200d) | 0.953 | 252 | 47 min | r6i.16xlarge |
| Voyage 4-large (1024d, 10K) | 1.000 | 148 | 5 min | n2-highmem-16 |

### GPU Search (CAGRA, NVIDIA L4)

| Dataset | Recall@10 | QPS | Build Time |
|---------|-----------|-----|------------|
| SIFT-1M (128d) | 0.992 | 3,175 | 27 seconds |
| Deep-1M (96d) | 0.999 | 2,763 | 22 seconds |
| GloVe-1.2M (200d) | 0.940 | 2,633 | 47 seconds |

### vs pgvector (SIFT-1M, r6i.16xlarge)

If you're using pgvector for vector search today, here's what changes:

| Metric | pgvector (0.7+) | arrwDB | Difference |
|--------|----------------|--------|------------|
| QPS at 0.99 recall | ~19 | 3,217 | **169x faster** |
| QPS at 0.95 recall | ~35 | 5,735 | **164x faster** |
| p50 latency at 0.99 | ~50ms | 0.32ms | **156x lower** |
| Build time (1M vectors) | ~minutes | 14 min | Comparable |
| Filtered search | SQL WHERE (post-filter) | Bitset (graph-integrated) | No recall loss |
| Hybrid search | Separate FTS + vector | BM25 + vector in one API | Single query |
| GPU acceleration | No | CAGRA (3,175 QPS) | N/A |
| Runs inside Postgres | Yes | No (standalone service) | Trade-off |

arrwDB is not a Postgres extension — it's a standalone vector search service. Use it alongside your existing database (Supabase, RDS, etc.) when pgvector becomes the bottleneck.

### Competitive Context (SIFT-1M, r6i.16xlarge — ann-benchmarks hardware)

| System | QPS at 0.999 recall | Notes |
|--------|---------------------|-------|
| qsgngt (Huawei) | ~7,000 | Quantized graph |
| NGT-qg (Yahoo) | ~4,300 | Quantized graph |
| glass (Zilliz) | ~2,400 | SIMD-optimized graph |
| arrwDB GPU CAGRA | 3,175 | NVIDIA L4 |
| **arrwDB CPU** | **1,793** | **Rust, AVX-512, top-7 on SIFT** |
| hnswlib | ~1,400 | C++, reference HNSW |
| FAISS-HNSW | ~1,200 | C++ |
| Weaviate | ~913 | Go |
| Qdrant | ~572 | Rust |
| pgvector | ~19 | PostgreSQL |

## Architecture

```
Python (FastAPI)          Rust (PyO3)
==================        ============================
REST API                  FastHNSW (integer-indexed)
Embedding providers       BM25 inverted index
Service layer             Distance metrics (cosine/L2/IP)
                          Pre-filtered search (bitset)
                          VectorStorage (contiguous + mmap)
                          RaBitQ quantization (optional)

GPU (FAISS-GPU)
============================
CAGRA index (NVIDIA cuVS)
```

### Index Backends

| Backend | Best For | Requires |
|---------|----------|----------|
| **FastHNSW** (default) | General use, incremental updates | CPU only |
| **GPU CAGRA** | Max throughput, fast builds | NVIDIA GPU |
| **RaBitQ** | Memory-constrained, large datasets | x86_64 CPU |
| **BM25** | Keyword search, hybrid retrieval | CPU only |
| Brute Force | Small datasets, exact results | CPU only |

## Quick Start

### Install

```bash
git clone https://github.com/bledden/arrwDB.git
cd arrwDB
pip install -r requirements.txt

# Build Rust extensions (recommended)
cd rust/indexes && maturin build --release
pip install target/wheels/*.whl
cd ../..

# Start the server
python run_api.py
```

### Python SDK

```python
from arrwdb import ArrwDBClient

with ArrwDBClient("http://localhost:8000") as client:
    # Create a library
    lib = client.create_library(name="docs", index_type="hnsw")

    # Add documents (auto-embeds via configured provider)
    client.add_document(lib["id"], title="Example", texts=["Document text here"])

    # Search
    results = client.search(lib["id"], query="example query", k=10)
```

### Direct Rust Index (no server)

```python
from rust_hnsw import RustFastHNSWIndex
import numpy as np

# Create index
index = RustFastHNSWIndex(dimension=1024, m=48, ef_construction=400, ef_search=200)

# Add vectors
index.add_vector("doc_1", embedding_array)

# Search
results = index.search(query_array, k=10)

# Upsert (update in-place if exists)
was_update = index.upsert_vector("doc_1", new_embedding)

# Filtered search (only return matching IDs)
results = index.search_filtered(query_array, k=10, filter_ids=["doc_1", "doc_3"])

# BM25 keyword search
from rust_hnsw import RustBM25Index
bm25 = RustBM25Index()
bm25.add_document("doc_1", "full text content here")
keyword_results = bm25.search("content", k=10)

# Hybrid search (vector + keyword via RRF)
hybrid = bm25.hybrid_search(
    vector_results=[("doc_1", 0.1), ("doc_2", 0.3)],
    query="content",
    k=10,
)
```

### Embedding Providers

```python
from app.services.embedding_providers import get_embedding_provider

# Voyage 4-large (best retrieval quality, 200M tokens free)
provider = get_embedding_provider("voyage")

# NVIDIA NV-Embed-v2 (free via NGC)
provider = get_embedding_provider("nvidia")

# Google Gemini Embedding 2 (#1 MTEB overall)
provider = get_embedding_provider("gemini")

# Cohere Embed v3
provider = get_embedding_provider("cohere")

embeddings = provider.embed_texts(["hello world"])
```

### Distance Metrics

```python
# Cosine (default, best for normalized embeddings)
index = RustFastHNSWIndex(dimension=128, metric="cosine")

# L2 / Euclidean
index = RustFastHNSWIndex(dimension=128, metric="l2")

# Inner product
index = RustFastHNSWIndex(dimension=128, metric="inner_product")
```

### GPU Acceleration

```python
# Requires: conda install faiss-gpu-cuvs
from infrastructure.indexes.gpu_cagra import GPUCagraIndex

index = GPUCagraIndex(vector_store, graph_degree=64)
# Add vectors, then rebuild() to build GPU index
index.rebuild()
results = index.search(query, k=10)

# Export to CPU HNSW for serving without GPU
cpu_index = index.to_cpu_hnsw()
```

## Features

### Search
- **Vector similarity** with cosine, L2, inner product distance
- **BM25 keyword search** with Okapi BM25 scoring
- **Hybrid search** combining vector + BM25 via Reciprocal Rank Fusion
- **Pre-filtered search** with ID-based filtering in the index scan
- **Temperature search** for diversity-tunable retrieval

### Index
- **FastHNSW** -- Rust, integer-indexed, paper-correct algorithms (Algorithm 4 heuristic selection)
- **GPU CAGRA** -- NVIDIA cuVS via FAISS-GPU, 27-second builds for 1M vectors
- **RaBitQ** -- IVF+RaBitQ quantization, 3-32x memory compression
- **Upsert** -- atomic insert-or-update, in-place vector overwrite with graph reconnection

### Storage
- **In-memory** -- contiguous float32 arrays for cache-friendly access
- **Memory-mapped** -- disk-backed vectors via mmap, OS handles paging
- **Write-ahead log** -- crash recovery
- **Snapshots** -- periodic full backups

### Observability
- **OpenTelemetry** distributed tracing with Sentry integration
- **Prometheus** metrics endpoint
- **Embedding health monitor** -- detects degenerate vectors, outliers, drift
- **Index Oracle** -- auto-recommends optimal index type

### API
- REST (FastAPI) with OpenAPI docs at `/docs`
- Python SDK with typed exceptions and retry logic
- WebSocket for real-time search
- Webhooks with HMAC verification

## Configuration

```bash
# Embedding provider
EMBEDDING_PROVIDER=voyage          # voyage, nvidia, gemini, cohere
VOYAGE_API_KEY=your_key
COHERE_API_KEY=your_key

# Server
HOST=0.0.0.0
PORT=8000

# Telemetry (optional)
SENTRY_DSN=https://...
OTEL_ENABLED=true
```

See [.env.example](.env.example) for all options.

## Rust Crates

| Crate | Purpose | PyO3 Class |
|-------|---------|------------|
| `rust_hnsw` | HNSW index + BM25 + brute force + LSH + KD-tree | `RustFastHNSWIndex`, `RustBM25Index` |
| `rust_vector_store` | Vector storage with deduplication | `RustVectorStore` |
| `rust_wal` | Write-ahead log | `RustWriteAheadLog` |
| `rust_snapshot` | Snapshot management | `RustSnapshotManager` |

All crates use PyO3 0.23 with `abi3-py39` for broad Python compatibility.

## License

Proprietary. All rights reserved.
