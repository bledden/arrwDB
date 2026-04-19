# arrwDB Python SDK

[![PyPI version](https://badge.fury.io/py/arrwdb.svg)](https://badge.fury.io/py/arrwdb)
[![Python](https://img.shields.io/pypi/pyversions/arrwdb.svg)](https://pypi.org/project/arrwdb/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Official Python client library for **arrwDB** - a production-grade vector database with 9 novel features not found in other vector databases.

## Features

- **Multiple Index Types**: HNSW, IVF, LSH, KD-Tree, Brute Force
- **9 Novel Features**:
  - 🔍 **Search Replay** - Debug HNSW traversal paths
  - 🌡️ **Temperature Search** - Exploration vs exploitation control
  - 🧠 **Index Oracle** - Intelligent index recommendations
  - 📊 **Embedding Health** - Outlier/drift detection
  - 🎯 **Vector Clustering** - K-means with auto-k
  - 🔄 **Query Expansion** - Automatic query rewriting
  - 📈 **Vector Drift** - Distribution monitoring
  - 🎓 **Adaptive Reranking** - Feedback-based learning
  - 🔗 **Hybrid Fusion** - Multi-strategy merging
- **Real-time**: WebSocket support, streaming search
- **Webhooks**: Event notifications with HMAC signatures
- **Background Jobs**: Async processing with job queue
- **Rust Optimizations**: 5-10x faster on critical paths

## Installation

```bash
pip install arrwdb

# With async support
pip install arrwdb[async]

# Framework integrations (pick what you need)
pip install arrwdb[langchain]     # LangChain VectorStore adapter
pip install arrwdb[llamaindex]    # LlamaIndex VectorStore adapter
pip install arrwdb[postgres]      # pgvector / Postgres sync helpers
pip install arrwdb[all-integrations]
```

## Integrations

Drop-in adapters so arrwDB plugs into existing AI stacks without migration:

```python
# LangChain — replace PGVector / Pinecone / Qdrant in one line
from arrwdb.integrations.langchain import ArrwDBVectorStore
store = ArrwDBVectorStore.from_texts(texts, embedding,
                                     base_url="http://localhost:8000")

# LlamaIndex
from arrwdb.integrations.llama_index import ArrwDBVectorStore
vector_store = ArrwDBVectorStore(base_url="http://localhost:8000")

# Pull a pgvector table into arrwDB (one-shot or incremental)
from arrwdb.integrations.postgres import sync_from_postgres
sync_from_postgres(
    pg_url="postgresql://user:pass@host/db",
    table="documents", id_column="id",
    text_column="content", embedding_column="embedding",
    library_name="docs",
)
```

See [examples/pgvector-migration](https://github.com/bledden/arrwDB/tree/main/examples/pgvector-migration) for a working docker-compose stack that runs pgvector and arrwDB side-by-side.

## Quick Start

```python
from arrwdb import ArrwDBClient

# Initialize client
client = ArrwDBClient("http://localhost:8000")

# Create a library with HNSW index
library = client.create_library(
    name="Research Papers",
    index_type="hnsw"
)

# Add documents (embeddings generated automatically)
doc = client.add_document(
    library_id=library["id"],
    title="Introduction to Neural Networks",
    texts=[
        "Neural networks are computing systems inspired by biological neural networks.",
        "They learn from data through backpropagation and gradient descent."
    ],
    tags=["AI", "machine learning"]
)

# Search
results = client.search(
    library_id=library["id"],
    query="What are neural networks?",
    k=5
)

for result in results["results"]:
    print(f"Text: {result['text'][:100]}...")
    print(f"Distance: {result['distance']:.4f}\n")
```

## Novel Features

### Temperature Search

Control exploration vs exploitation in search results:

```python
# Greedy - return top results only
results = client.temperature_search(
    corpus_id=library["id"],
    query_text="machine learning",
    temperature=0.0,  # Deterministic
    k=10
)

# Exploratory - diverse, serendipitous results
results = client.temperature_search(
    corpus_id=library["id"],
    query_text="machine learning",
    temperature=1.5,  # High diversity
    k=10
)
```

### Index Oracle

Get intelligent index recommendations:

```python
recommendation = client.get_index_recommendation(library["id"])
print(f"Recommended: {recommendation['recommended_index']}")
print(f"Reason: {recommendation['reasoning']}")
```

### Embedding Health Monitor

Detect quality issues in your embeddings:

```python
health = client.analyze_embedding_health(library["id"])
print(f"Outliers: {health['outlier_count']}")
print(f"Degeneracy score: {health['degeneracy_score']}")
print(f"Drift detected: {health['drift_detected']}")
```

## CLI Usage

```bash
# Check server health
arrwdb health

# List libraries
arrwdb libraries list

# Create library
arrwdb libraries create "My Library" --index-type hnsw

# Search
arrwdb search <library_id> "query text" -k 10

# Temperature search
arrwdb temperature-search <library_id> "query" --temperature 1.5

# Get index recommendation
arrwdb index-oracle <library_id>
```

## Async Client

```python
from arrwdb import AsyncArrwDBClient

async with AsyncArrwDBClient("http://localhost:8000") as client:
    library = await client.create_library("async-lib", index_type="hnsw")
    results = await client.search(library["id"], "query")
```

## Webhooks

```python
# Create webhook for event notifications
webhook = client.create_webhook(
    url="https://yourapp.com/webhooks/arrwdb",
    events=["document.created", "job.completed"],
    description="Production notifications"
)

# Save secret for HMAC verification
webhook_secret = webhook["secret"]
```

## Configuration

```python
client = ArrwDBClient(
    base_url="https://api.yourcompany.com",
    timeout=60,
    verify_ssl=True,
    api_key="your-api-key"  # Optional
)
```

Environment variables:
- `ARRWDB_URL` - Server URL (default: http://localhost:8000)
- `ARRWDB_API_KEY` - API key for authentication

## Documentation

- [Full API Guide](https://github.com/bledden/arrwDB/blob/main/docs/API_GUIDE.md)
- [Novel Features](https://github.com/bledden/arrwDB/blob/main/docs/NOVEL_FEATURES.md)
- [Competitive Analysis](https://github.com/bledden/arrwDB/blob/main/docs/COMPETITIVE_ANALYSIS_2025.md)

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.
