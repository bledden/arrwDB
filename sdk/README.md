# arrwDB Python SDK

Official Python client library for arrwDB - Production Vector Database.

## Installation

```bash
pip install arrwdb  # Coming soon to PyPI

# Or install from source
cd /path/to/arrwDB
pip install -e .
```

## Quick Start

```python
from sdk.client import VectorDBClient

# Initialize client
client = VectorDBClient("http://localhost:8000")

# Check health
health = client.health_check()
print(f"API Status: {health['status']}")

# Create a library
library = client.create_library(
    name="Research Papers",
    description="AI and ML research papers",
    index_type="hnsw"
)
library_id = library["id"]

# Add documents
doc = client.add_document(
    library_id=library_id,
    title="Introduction to Neural Networks",
    texts=[
        "Neural networks are computing systems inspired by biological neural networks.",
        "They learn from data through backpropagation and gradient descent."
    ],
    tags=["AI", "machine learning", "neural networks"]
)

# Search
results = client.search(
    library_id=library_id,
    query="What are neural networks?",
    k=5
)

for result in results["results"]:
    print(f"Text: {result['text']}")
    print(f"Distance: {result['distance']}")
    print()
```

## Features

### Library Management

```python
# Create library with specific index type
library = client.create_library(
    name="My Library",
    index_type="hnsw",  # Options: brute_force, kd_tree, lsh, hnsw
    embedding_model="embed-english-v3.0"
)

# List all libraries
libraries = client.list_libraries()

# Get library details
library = client.get_library(library_id)

# Get library statistics
stats = client.get_library_statistics(library_id)
print(f"Documents: {stats['total_documents']}")
print(f"Chunks: {stats['total_chunks']}")

# Delete library
client.delete_library(library_id)
```

### Document Operations

```python
# Add single document (auto-embedding)
doc = client.add_document(
    library_id=library_id,
    title="Document Title",
    texts=["Chunk 1 text", "Chunk 2 text"],
    author="John Doe",
    tags=["tag1", "tag2"]
)

# Add document with pre-computed embeddings
doc = client.add_document_with_embeddings(
    library_id=library_id,
    title="Document Title",
    chunks=[
        ("Chunk 1 text", [0.1, 0.2, ...]),  # (text, embedding)
        ("Chunk 2 text", [0.3, 0.4, ...])
    ]
)

# Get document
doc = client.get_document(document_id)

# Delete document
client.delete_document(document_id)
```

### Search Operations

```python
# Basic semantic search
results = client.search(
    library_id=library_id,
    query="machine learning techniques",
    k=10
)

# Search with distance threshold
results = client.search(
    library_id=library_id,
    query="neural networks",
    k=10,
    distance_threshold=0.5  # Only return results within distance
)

# Search with pre-computed embedding
results = client.search_with_embedding(
    library_id=library_id,
    embedding=[0.1, 0.2, 0.3, ...],
    k=10
)

# Search with metadata filters
results = client.search_with_filters(
    library_id=library_id,
    query="deep learning",
    metadata_filters=[
        {"field": "chunk_index", "operator": "gte", "value": 2},
        {"field": "chunk_index", "operator": "lt", "value": 10}
    ],
    k=10
)
```

### Webhooks

```python
# Create webhook to receive event notifications
webhook = client.create_webhook(
    url="https://yourapp.com/webhooks/arrwdb",
    events=["job.completed", "job.failed", "document.created"],
    description="Production webhook",
    max_retries=3
)

# Save the webhook secret for HMAC verification
webhook_secret = webhook["secret"]

# List all webhooks
webhooks = client.list_webhooks()

# Get webhook details
webhook = client.get_webhook(webhook_id)

# Update webhook
webhook = client.update_webhook(
    webhook_id=webhook_id,
    status="paused",  # Pause webhook deliveries
    events=["job.completed"]  # Update event subscriptions
)

# Get webhook delivery history
deliveries = client.get_webhook_deliveries(
    webhook_id=webhook_id,
    status="failed"  # Filter by status
)

# Get webhook statistics
stats = client.get_webhook_stats(webhook_id)
print(f"Success rate: {stats['success_rate']}%")
print(f"Total deliveries: {stats['total_deliveries']}")

# Test webhook
result = client.test_webhook(webhook_id)
print(f"Test successful: {result['success']}")

# Delete webhook
client.delete_webhook(webhook_id)
```

### Verifying Webhook Signatures

```python
import hmac
import hashlib

def verify_webhook_signature(payload: bytes, signature: str, secret: str) -> bool:
    """
    Verify HMAC signature from webhook delivery.

    Args:
        payload: Raw request body (bytes)
        signature: X-Webhook-Signature header value
        secret: Webhook secret from create_webhook response

    Returns:
        True if signature is valid
    """
    expected_sig = hmac.new(
        secret.encode(),
        payload,
        hashlib.sha256
    ).hexdigest()

    # Remove 'sha256=' prefix if present
    if signature.startswith('sha256='):
        signature = signature[7:]

    return hmac.compare_digest(expected_sig, signature)

# In your webhook handler
from flask import Flask, request

app = Flask(__name__)

@app.route('/webhooks/arrwdb', methods=['POST'])
def handle_webhook():
    # Get signature from header
    signature = request.headers.get('X-Webhook-Signature')

    # Verify signature
    if not verify_webhook_signature(request.data, signature, webhook_secret):
        return {'error': 'Invalid signature'}, 401

    # Process webhook event
    event = request.json
    print(f"Received event: {event['type']}")
    print(f"Event data: {event['data']}")

    return {'received': True}, 200
```

### Background Jobs

```python
# Submit a batch import job
job = client.submit_job(
    job_type="batch_import",
    payload={
        "library_id": library_id,
        "documents": [...]
    }
)
job_id = job["id"]

# Submit and wait for completion
job = client.submit_job(
    job_type="index_rebuild",
    payload={"library_id": library_id},
    wait=True  # Blocks until complete
)

# Check job status
status = client.get_job_status(job_id)
print(f"Status: {status['status']}")
print(f"Progress: {status['progress']}%")

# List all jobs
jobs = client.list_jobs(status="running")

# Cancel a job
client.cancel_job(job_id)
```

### Health Monitoring

```python
# Basic health check
health = client.health_check()
print(f"Status: {health['status']}")
print(f"Uptime: {health['uptime_seconds']}s")

# Readiness check (for K8s)
ready = client.readiness_check()
print(f"Ready: {ready['ready']}")
print(f"Checks: {ready['checks']}")

# Detailed component health
detailed = client.detailed_health()
for component, status in detailed['components'].items():
    print(f"{component}: {status['status']}")
```

## Context Manager

Use the client as a context manager for automatic cleanup:

```python
with VectorDBClient("http://localhost:8000") as client:
    # Create library
    library = client.create_library(name="Test")

    # Add documents
    client.add_document(...)

    # Search
    results = client.search(...)

# Session automatically closed
```

## Error Handling

```python
from sdk.client import VectorDBClient, VectorDBException

client = VectorDBClient()

try:
    library = client.create_library(name="Test")
except VectorDBException as e:
    print(f"API Error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Configuration

```python
# Custom configuration
client = VectorDBClient(
    base_url="https://api.yourcompany.com",
    timeout=60,  # Request timeout in seconds
    verify_ssl=True  # SSL certificate verification
)
```

## Advanced Examples

### RAG Pipeline

```python
# Build a RAG (Retrieval-Augmented Generation) pipeline

# 1. Create library and index documents
library = client.create_library(name="Knowledge Base", index_type="hnsw")

# 2. Add documents
for doc in documents:
    client.add_document(
        library_id=library["id"],
        title=doc["title"],
        texts=doc["chunks"]
    )

# 3. Query and retrieve context
def get_context(question: str, k: int = 5) -> str:
    results = client.search(
        library_id=library["id"],
        query=question,
        k=k
    )

    # Combine top results into context
    context = "\n\n".join([r["text"] for r in results["results"]])
    return context

# 4. Use with LLM
question = "What is machine learning?"
context = get_context(question)

# Send to LLM with context
prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
# ... call your LLM ...
```

### Batch Document Processing

```python
import json

# Prepare NDJSON file
with open('documents.ndjson', 'w') as f:
    for doc in documents:
        f.write(json.dumps({
            "title": doc["title"],
            "texts": doc["texts"],
            "tags": doc["tags"]
        }) + '\n')

# Submit batch import job
job = client.submit_job(
    job_type="batch_import",
    payload={
        "library_id": library_id,
        "file_path": "documents.ndjson"
    },
    wait=True
)

print(f"Imported {job['result']['successful']} documents")
```

## API Reference

See the [API Guide](../docs/API_GUIDE.md) for complete API documentation.

## Support

- Documentation: [/docs](/docs)
- Issues: GitHub Issues
- API Docs: http://localhost:8000/docs (when server is running)

## License

Proprietary - All rights reserved
