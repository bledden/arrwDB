# LlamaIndex arrwDB Integration

[![PyPI version](https://badge.fury.io/py/llama-index-vector-stores-arrwdb.svg)](https://badge.fury.io/py/llama-index-vector-stores-arrwdb)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

LlamaIndex integration for **arrwDB** - a production-grade vector database with novel features including temperature search, index oracle, and embedding health monitoring.

## Installation

```bash
pip install llama-index-vector-stores-arrwdb
```

## Quick Start

### Basic Usage

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index_arrwdb import ArrwDBVectorStore

# Create vector store
vector_store = ArrwDBVectorStore(
    base_url="http://localhost:8000",
    library_name="my-docs",  # Creates new library
)

# Load documents and create index
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(
    documents,
    vector_store=vector_store,
)

# Query
query_engine = index.as_query_engine()
response = query_engine.query("What is machine learning?")
print(response)
```

### Connect to Existing Library

```python
vector_store = ArrwDBVectorStore(
    base_url="http://localhost:8000",
    library_id="existing-library-id",
)

index = VectorStoreIndex.from_vector_store(vector_store)
```

### Temperature Search (Novel Feature)

Control exploration vs exploitation in retrieval:

```python
# Deterministic retrieval - top-k most similar
results = vector_store.temperature_query(
    "machine learning",
    k=10,
    temperature=0.0,  # Greedy
)

# Exploratory retrieval - diverse, serendipitous results
results = vector_store.temperature_query(
    "machine learning",
    k=10,
    temperature=1.5,  # High diversity
)
```

### Custom Retriever with Temperature

```python
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever

vector_store = ArrwDBVectorStore(
    base_url="http://localhost:8000",
    library_id="my-library",
)

index = VectorStoreIndex.from_vector_store(vector_store)

# Create retriever with temperature
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=10,
)

# Manually query with temperature
results = vector_store.temperature_query(
    "What is deep learning?",
    k=10,
    temperature=1.2,
)
```

## Novel Features

### Index Oracle

Get intelligent index recommendations:

```python
recommendation = vector_store.get_index_recommendation()
print(f"Recommended: {recommendation['recommended_index']}")
print(f"Reason: {recommendation['reasoning']}")
```

### Embedding Health

Monitor embedding quality:

```python
health = vector_store.analyze_embedding_health()
print(f"Outliers: {health['outlier_count']}")
print(f"Degeneracy: {health['degeneracy_score']}")
print(f"Drift detected: {health['drift_detected']}")
```

## Configuration

### Server-Side vs Client-Side Embeddings

```python
# Use arrwDB's built-in embeddings (default)
vector_store = ArrwDBVectorStore(
    base_url="http://localhost:8000",
    library_name="my-docs",
    use_server_embeddings=True,  # Default
)

# Use LlamaIndex embeddings
from llama_index.embeddings.openai import OpenAIEmbedding

vector_store = ArrwDBVectorStore(
    base_url="http://localhost:8000",
    library_name="my-docs",
    use_server_embeddings=False,
)

index = VectorStoreIndex.from_documents(
    documents,
    vector_store=vector_store,
    embed_model=OpenAIEmbedding(),
)
```

### Index Types

arrwDB supports multiple index types:

```python
# HNSW - Best for most use cases (default)
vector_store = ArrwDBVectorStore(
    library_name="hnsw-lib",
    index_type="hnsw",
)

# IVF - Good for large datasets
vector_store = ArrwDBVectorStore(
    library_name="ivf-lib",
    index_type="ivf",
)

# LSH - Fast approximate search
vector_store = ArrwDBVectorStore(
    library_name="lsh-lib",
    index_type="lsh",
)
```

## API Reference

### ArrwDBVectorStore

| Method | Description |
|--------|-------------|
| `add()` | Add nodes to vector store |
| `delete()` | Delete document by ID |
| `query()` | Standard vector store query |
| `temperature_query()` | Exploration-controlled query |
| `get_index_recommendation()` | Index Oracle analysis |
| `analyze_embedding_health()` | Embedding health check |

### Constructor Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `base_url` | str | arrwDB server URL |
| `library_id` | str | Existing library ID |
| `library_name` | str | Name for new library |
| `api_key` | str | Optional API key |
| `index_type` | str | Index type (hnsw, ivf, lsh, kdtree, brute_force) |
| `use_server_embeddings` | bool | Use arrwDB's embeddings |

## License

Apache 2.0
