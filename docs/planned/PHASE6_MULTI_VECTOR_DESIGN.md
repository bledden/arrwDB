# Phase 6: Multi-Vector Support Per Document - Design Document

**Status**: 📋 **PLANNED** - Design complete, implementation pending
**Target**: Enable cross-modal search and multiple representations per document

---

## Executive Summary

Multi-vector support allows each document to have multiple vector representations, enabling:
- **Cross-modal search**: Text, image, audio, video vectors in same document
- **Multi-lingual**: Same document with vectors in different languages
- **Multiple embeddings**: Different models or strategies per document
- **Hierarchical**: Title vector, section vectors, paragraph vectors

**Use Case Example**: A research paper with:
- Title embedding (semantic search)
- Abstract embedding (summary search)
- Full text embedding (detailed search)
- Figure embeddings (visual search)
- Author embedding (author similarity)

---

## Architecture Design

### Current Model (Single Vector):

```python
Document:
  - id: UUID
  - metadata: {...}
  - chunks: [Chunk]

Chunk:
  - id: UUID
  - text: str
  - embedding: np.ndarray  # Single vector
  - metadata: {...}
```

### Proposed Model (Multi-Vector):

```python
Document:
  - id: UUID
  - metadata: {...}
  - vectors: [DocumentVector]  # NEW: Multiple vectors per document

DocumentVector:
  - id: UUID
  - vector_type: VectorType  # NEW: text, image, audio, etc.
  - embedding: np.ndarray
  - metadata: dict
  - source: str  # Which field/file this came from
```

### Vector Types:

```python
class VectorType(Enum):
    TEXT = "text"           # Text embedding
    IMAGE = "image"         # Image embedding (CLIP, etc.)
    AUDIO = "audio"         # Audio embedding
    VIDEO = "video"         # Video embedding
    CODE = "code"           # Code embedding
    TITLE = "title"         # Document title
    SUMMARY = "summary"     # Document summary
    CUSTOM = "custom"       # User-defined
```

---

## API Design

### 1. Create Document with Multiple Vectors

**Endpoint**: `POST /v1/libraries/{id}/documents/multi-vector`

**Request**:
```json
{
  "title": "Machine Learning Research Paper",
  "vectors": [
    {
      "type": "title",
      "text": "Deep Learning for NLP",
      "metadata": {"lang": "en"}
    },
    {
      "type": "summary",
      "text": "This paper explores transformer architectures...",
      "metadata": {"lang": "en"}
    },
    {
      "type": "image",
      "image_url": "https://example.com/fig1.png",
      "metadata": {"caption": "Model architecture"}
    },
    {
      "type": "text",
      "text": "Full paper content...",
      "metadata": {"section": "introduction"}
    }
  ],
  "metadata": {
    "author": "John Doe",
    "year": 2024
  }
}
```

**Response**:
```json
{
  "id": "doc-uuid",
  "title": "Machine Learning Research Paper",
  "vectors": [
    {
      "id": "vec-uuid-1",
      "type": "title",
      "embedding_model": "text-embedding-3-large",
      "dimensions": 1024
    },
    {
      "id": "vec-uuid-2",
      "type": "summary",
      "embedding_model": "text-embedding-3-large",
      "dimensions": 1024
    },
    {
      "id": "vec-uuid-3",
      "type": "image",
      "embedding_model": "clip-vit-large",
      "dimensions": 768
    },
    {
      "id": "vec-uuid-4",
      "type": "text",
      "embedding_model": "text-embedding-3-large",
      "dimensions": 1024
    }
  ],
  "metadata": {...}
}
```

### 2. Search Across Vector Types

**Endpoint**: `POST /v1/libraries/{id}/search/multi-vector`

**Request**:
```json
{
  "queries": [
    {
      "type": "text",
      "query": "transformer architecture",
      "weight": 0.5
    },
    {
      "type": "image",
      "image_url": "https://example.com/query.png",
      "weight": 0.5
    }
  ],
  "k": 10,
  "vector_types": ["text", "image", "summary"]  # Optional filter
}
```

**Response**:
```json
{
  "results": [
    {
      "document_id": "doc-uuid",
      "title": "Machine Learning Research Paper",
      "matched_vectors": [
        {
          "vector_id": "vec-uuid-1",
          "type": "text",
          "distance": 0.15,
          "weighted_score": 0.425
        },
        {
          "vector_id": "vec-uuid-3",
          "type": "image",
          "distance": 0.20,
          "weighted_score": 0.400
        }
      ],
      "combined_score": 0.825,
      "score_breakdown": {
        "text_score": 0.425,
        "image_score": 0.400
      }
    }
  ],
  "query_time_ms": 45.2
}
```

### 3. Aggregation Strategies

**Late Fusion** (search each type, then combine):
```python
def late_fusion_search(queries, k):
    results_per_type = {}

    for query in queries:
        results = search_single_type(query.type, query.query, k)
        results_per_type[query.type] = results

    # Combine results with weighted scores
    combined = aggregate_results(results_per_type, queries.weights)
    return combined[:k]
```

**Early Fusion** (combine embeddings, then search):
```python
def early_fusion_search(queries, k):
    # Average embeddings with weights
    combined_embedding = sum(q.embedding * q.weight for q in queries)
    combined_embedding /= sum(q.weight for q in queries)

    # Search with combined embedding
    return index.search(combined_embedding, k)
```

---

## Indexing Strategy

### Separate Indices per Vector Type:

```python
class MultiVectorLibrary:
    def __init__(self):
        self.indices = {
            VectorType.TEXT: HNSWIndex(dimensions=1024),
            VectorType.IMAGE: HNSWIndex(dimensions=768),
            VectorType.AUDIO: HNSWIndex(dimensions=512),
            # ...
        }

    def add_document(self, doc):
        for vector in doc.vectors:
            index = self.indices[vector.type]
            index.add(vector.embedding, doc.id)

    def search(self, queries, k):
        results_per_type = {}

        for query in queries:
            index = self.indices[query.type]
            results = index.search(query.embedding, k)
            results_per_type[query.type] = results

        # Aggregate and return
        return self._aggregate_results(results_per_type, queries)
```

### Unified Index with Type Filtering:

```python
class UnifiedMultiVectorIndex:
    def __init__(self):
        # Store all vectors in single index
        self.index = HNSWIndex(dimensions=max_dimensions)

        # Track vector types
        self.vector_types = {}  # vector_id -> VectorType

    def search(self, query, k, vector_types=None):
        # Search index
        candidates = self.index.search(query, k * 10)

        # Filter by vector type
        if vector_types:
            candidates = [
                c for c in candidates
                if self.vector_types[c.id] in vector_types
            ]

        return candidates[:k]
```

---

## Embedding Generation

### Multi-Modal Embedding Models:

**Text Embeddings**:
```python
from openai import OpenAI

def embed_text(text):
    response = OpenAI().embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return response.data[0].embedding
```

**Image Embeddings (CLIP)**:
```python
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

def embed_image(image_path):
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")
    outputs = model.get_image_features(**inputs)
    return outputs.detach().numpy()[0]
```

**Audio Embeddings**:
```python
from transformers import Wav2Vec2Model, Wav2Vec2Processor

model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

def embed_audio(audio_path):
    audio, sr = librosa.load(audio_path, sr=16000)
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
    outputs = model(**inputs).last_hidden_state.mean(dim=1)
    return outputs.detach().numpy()[0]
```

---

## Use Cases

### 1. Cross-Modal Search

**Scenario**: Search research papers with both text and image queries

```python
# User uploads diagram and enters text query
query = {
    "queries": [
        {"type": "text", "query": "transformer attention mechanism", "weight": 0.6},
        {"type": "image", "image": diagram_bytes, "weight": 0.4}
    ]
}

# Find papers matching both text concept and visual similarity
results = library.search_multi_vector(query, k=20)
```

### 2. Multi-Lingual Documents

**Scenario**: Same document in multiple languages

```python
document = {
    "title": "Machine Learning Guide",
    "vectors": [
        {"type": "text", "text": english_text, "metadata": {"lang": "en"}},
        {"type": "text", "text": spanish_text, "metadata": {"lang": "es"}},
        {"type": "text", "text": chinese_text, "metadata": {"lang": "zh"}}
    ]
}

# Search in any language, match in all languages
results = library.search(query="深度学习", k=10)  # Chinese query
# Returns documents with Chinese, English, Spanish vectors
```

### 3. Hierarchical Representations

**Scenario**: Document with title, summary, and full content

```python
document = {
    "vectors": [
        {"type": "title", "text": title, "weight": 2.0},     # Higher priority
        {"type": "summary", "text": summary, "weight": 1.5}, # Medium priority
        {"type": "text", "text": full_text, "weight": 1.0}   # Base priority
    ]
}

# Search prioritizes title matches
results = library.search(query, vector_types=["title", "summary"])
```

### 4. Product Search (E-Commerce)

**Scenario**: Products with image, title, description

```python
product = {
    "title": "Blue Running Shoes",
    "vectors": [
        {"type": "image", "image": product_image},
        {"type": "title", "text": "Nike Air Zoom Pegasus 40"},
        {"type": "text", "text": "Lightweight running shoe with..."}
    ]
}

# User uploads photo of shoes they like
results = library.search_multi_vector({
    "queries": [
        {"type": "image", "image": user_photo, "weight": 0.8},
        {"type": "text", "query": "lightweight", "weight": 0.2}
    ]
})
```

---

## Performance Considerations

### Memory:

**Single Vector per Document**: `N * D * 4 bytes`
**Multi-Vector per Document**: `N * V * D * 4 bytes` (V = vectors per doc)

**Example**:
- 1M documents
- 3 vectors per document (title, summary, content)
- 1024 dimensions

Memory: `1M * 3 * 1024 * 4 = 12GB` (vs 4GB for single vector)

### Search Performance:

**Late Fusion** (separate searches):
```
Time = V * search_time
V = number of vector types searched
```

**Early Fusion** (combined embedding):
```
Time = 1 * search_time + fusion_overhead
Faster but less flexible
```

### Optimization Strategies:

1. **Cached Aggregation**: Pre-compute combined scores
2. **Lazy Loading**: Load vectors only when needed
3. **Type-Specific Indices**: Optimize per vector type
4. **Pruning**: Search only relevant types based on query

---

## Implementation Plan

### Phase 6.1: Data Model (Week 1)
- [ ] Extend Document model with vectors field
- [ ] Create DocumentVector class
- [ ] Add VectorType enum
- [ ] Database schema migration

### Phase 6.2: API Endpoints (Week 2)
- [ ] POST /documents/multi-vector
- [ ] POST /search/multi-vector
- [ ] GET /documents/{id}/vectors
- [ ] PUT /documents/{id}/vectors/{vec_id}

### Phase 6.3: Indexing (Week 3)
- [ ] Multi-index manager
- [ ] Type-specific indices
- [ ] Unified search interface
- [ ] Aggregation strategies

### Phase 6.4: Embedding Integration (Week 4)
- [ ] Text embedding service (existing)
- [ ] Image embedding service (CLIP)
- [ ] Audio embedding service (Wav2Vec2)
- [ ] Custom embedding support

### Phase 6.5: Testing & Optimization (Week 5)
- [ ] Unit tests for multi-vector operations
- [ ] Integration tests for cross-modal search
- [ ] Performance benchmarks
- [ ] Documentation

---

## Testing Strategy

```python
def test_multi_vector_document():
    """Test creating document with multiple vectors"""
    doc = {
        "title": "Test Doc",
        "vectors": [
            {"type": "title", "text": "Title"},
            {"type": "text", "text": "Content"}
        ]
    }

    response = client.post("/v1/libraries/{id}/documents/multi-vector", json=doc)
    assert response.status_code == 201
    assert len(response.json()["vectors"]) == 2

def test_cross_modal_search():
    """Test searching with text and image queries"""
    query = {
        "queries": [
            {"type": "text", "query": "machine learning", "weight": 0.5},
            {"type": "image", "image_url": "test.jpg", "weight": 0.5}
        ],
        "k": 10
    }

    results = client.post("/v1/libraries/{id}/search/multi-vector", json=query)
    assert results.status_code == 200
    assert len(results.json()["results"]) <= 10

def test_vector_type_filtering():
    """Test filtering by vector type"""
    query = {
        "query": "test query",
        "k": 10,
        "vector_types": ["title", "summary"]  # Only search these types
    }

    results = client.post("/v1/libraries/{id}/search/multi-vector", json=query)
    for result in results.json()["results"]:
        for vec in result["matched_vectors"]:
            assert vec["type"] in ["title", "summary"]
```

---

## Comparison to Alternatives

### vs Weaviate Multi-Vector:
- **arrwDB**: More flexible aggregation strategies
- **Weaviate**: More mature, GraphQL API

### vs Milvus Collections:
- **arrwDB**: Simpler API, unified search
- **Milvus**: Better performance, more features

### vs Pinecone Namespaces:
- **arrwDB**: True multi-vector per document
- **Pinecone**: Separate collections, manual merging

---

## Production Considerations

### Scaling:
1. **Horizontal**: Shard by vector type across machines
2. **Vertical**: Use IVF-PQ for each vector type
3. **Caching**: Cache frequently accessed vectors
4. **CDN**: Serve image/audio embeddings from CDN

### Monitoring:
- Track search latency per vector type
- Monitor index sizes per type
- Alert on aggregation bottlenecks
- Measure cross-modal recall

### Cost Optimization:
- Use smaller models for less important vector types
- Cache embeddings for static content
- Lazy-load infrequently accessed vectors
- Compress embeddings with quantization

---

## Conclusion

Multi-vector support transforms arrwDB into a truly multi-modal vector database, enabling:
- ✅ Cross-modal search (text + image + audio)
- ✅ Multi-lingual documents
- ✅ Hierarchical representations
- ✅ Flexible aggregation strategies
- ✅ Production-ready architecture

**Next Steps**: Implement Phase 6.1 (Data Model) and begin integration testing.

---

**Status**: Ready for implementation once Phase 5 (IVF) is complete! 🚀
