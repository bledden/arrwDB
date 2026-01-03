# LangChain arrwDB Integration

[![PyPI version](https://badge.fury.io/py/langchain-arrwdb.svg)](https://badge.fury.io/py/langchain-arrwdb)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

LangChain integration for **arrwDB** - a production-grade vector database with novel features including temperature search, index oracle, and embedding health monitoring.

## Installation

```bash
pip install langchain-arrwdb
```

## Quick Start

### Vector Store

```python
from langchain_arrwdb import ArrwDBVectorStore
from langchain_openai import OpenAIEmbeddings

# Create vector store
vectorstore = ArrwDBVectorStore(
    base_url="http://localhost:8000",
    library_name="my-docs",  # Creates new library
    embedding=OpenAIEmbeddings(),
)

# Add documents
vectorstore.add_texts([
    "Neural networks learn from data",
    "Machine learning is a subset of AI",
    "Deep learning uses multiple layers",
])

# Search
results = vectorstore.similarity_search("What is deep learning?", k=3)
for doc in results:
    print(f"{doc.page_content} (distance: {doc.metadata['distance']:.4f})")
```

### Using Server-Side Embeddings

arrwDB can generate embeddings automatically:

```python
# No embedding model needed - arrwDB handles it
vectorstore = ArrwDBVectorStore(
    base_url="http://localhost:8000",
    library_name="my-docs",
    # embedding=None means use server-side embeddings
)
```

### Temperature Search (Novel Feature)

Control exploration vs exploitation in search results:

```python
# Deterministic - top-k most similar
results = vectorstore.temperature_search(
    "machine learning",
    k=10,
    temperature=0.0,  # Greedy
)

# Exploratory - diverse, serendipitous results
results = vectorstore.temperature_search(
    "machine learning",
    k=10,
    temperature=1.5,  # High diversity
)
```

### Retriever for RAG

```python
from langchain_arrwdb import ArrwDBRetriever
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Create retriever
retriever = ArrwDBRetriever(
    base_url="http://localhost:8000",
    library_id="your-library-id",
    k=5,
)

# Build RAG chain
template = """Answer based on the context:

Context: {context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
llm = ChatOpenAI()

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)

response = chain.invoke("What is machine learning?")
```

### Exploratory RAG with Temperature

```python
# Use temperature for diverse context retrieval
retriever = ArrwDBRetriever(
    base_url="http://localhost:8000",
    library_id="your-library-id",
    k=10,
    temperature=1.2,  # Balanced exploration
)
```

## Novel Features

### Index Oracle

Get intelligent index recommendations:

```python
recommendation = vectorstore.get_index_recommendation()
print(f"Recommended: {recommendation['recommended_index']}")
print(f"Reason: {recommendation['reasoning']}")
```

### Embedding Health

Monitor embedding quality:

```python
health = vectorstore.analyze_embedding_health()
print(f"Outliers: {health['outlier_count']}")
print(f"Degeneracy: {health['degeneracy_score']}")
print(f"Drift detected: {health['drift_detected']}")
```

## API Reference

### ArrwDBVectorStore

| Method | Description |
|--------|-------------|
| `add_texts()` | Add texts to vector store |
| `similarity_search()` | Standard similarity search |
| `similarity_search_with_score()` | Search with distance scores |
| `temperature_search()` | Exploration-controlled search |
| `get_index_recommendation()` | Index Oracle analysis |
| `analyze_embedding_health()` | Embedding health check |
| `delete()` | Delete documents |

### ArrwDBRetriever

| Parameter | Description |
|-----------|-------------|
| `k` | Number of documents to retrieve |
| `temperature` | Optional temperature for exploration search |
| `search_kwargs` | Additional search parameters |

## License

Apache 2.0
