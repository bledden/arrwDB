# CLI Usage Examples - Vector Database API

This guide shows how to interact with the Vector Database API using command-line tools (curl). These are the same operations you can do in the Swagger UI at http://localhost:8000/docs, but from the terminal.

---

## Prerequisites

1. **Start the API server:**
   ```bash
   python3 run_api.py
   ```

2. **Verify it's running:**
   ```bash
   curl http://localhost:8000/health
   ```
   Expected: `{"status":"healthy","version":"1.0.0",...}`

3. **Set up environment:**
   ```bash
   # Optional: Install jq for pretty JSON output
   brew install jq  # macOS
   # sudo apt install jq  # Linux
   ```

---

## Quick Start - Complete Workflow

Here's a complete example from creating a library to searching:

```bash
# 1. Create a library
LIBRARY_ID=$(curl -s -X POST "http://localhost:8000/v1/libraries" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Tech Articles",
    "description": "Technology and AI articles",
    "index_type": "hnsw"
  }' | jq -r '.id')

echo "Created library: $LIBRARY_ID"

# 2. Add a document
DOC_ID=$(curl -s -X POST "http://localhost:8000/v1/libraries/$LIBRARY_ID/documents" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Introduction to Machine Learning",
    "texts": [
      "Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed. It uses statistical techniques to give computers the ability to learn patterns and make decisions."
    ],
    "author": "Tech Blog"
  }' | jq -r '.id')

echo "Added document: $DOC_ID"

# 3. Add more documents
curl -s -X POST "http://localhost:8000/v1/libraries/$LIBRARY_ID/documents" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Deep Learning Explained",
    "texts": [
      "Deep learning is a type of machine learning based on artificial neural networks with multiple layers. It has revolutionized fields like computer vision and natural language processing."
    ],
    "author": "Tech Blog"
  }' | jq -r '.id'

curl -s -X POST "http://localhost:8000/v1/libraries/$LIBRARY_ID/documents" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "NLP Overview",
    "texts": [
      "Natural language processing enables computers to understand, interpret, and generate human language. It powers applications like chatbots, translation, and sentiment analysis."
    ],
    "author": "Tech Blog"
  }' | jq -r '.id'

# 4. Search for similar content
curl -s -X POST "http://localhost:8000/v1/libraries/$LIBRARY_ID/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is AI and how does it learn?",
    "k": 3
  }' | jq '.'

# 5. Get library statistics
curl -s "http://localhost:8000/v1/libraries/$LIBRARY_ID/statistics" | jq '.'
```

---

## Detailed Examples

### 1. Health Check

**Check API status:**
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2025-10-21T12:00:00Z"
}
```

---

### 2. Create Libraries

**Create a library with BruteForce (100% accurate):**
```bash
curl -X POST "http://localhost:8000/v1/libraries" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Research Papers",
    "description": "AI and ML research papers",
    "index_type": "brute_force"
  }'
```

**Create a library with HNSW (recommended):**
```bash
curl -X POST "http://localhost:8000/v1/libraries" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Product Catalog",
    "description": "E-commerce product descriptions",
    "index_type": "hnsw"
  }'
```

**Response:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "Product Catalog",
  "description": "E-commerce product descriptions",
  "index_type": "hnsw",
  "created_at": "2025-10-21T12:00:00Z"
}
```

**Index Types:**
- `brute_force` - 100% accurate, slower for large datasets
- `kd_tree` - Fast exact search, good for structured data
- `lsh` - Approximate search, very fast
- `hnsw` - Best balance (recommended)

---

### 3. List All Libraries

**Get all libraries:**
```bash
curl http://localhost:8000/v1/libraries | jq '.'
```

**Response:**
```json
{
  "libraries": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "name": "Research Papers",
      "description": "AI and ML research papers",
      "index_type": "brute_force",
      "created_at": "2025-10-21T12:00:00Z"
    }
  ],
  "total": 1
}
```

---

### 4. Get Specific Library

**Get library by ID:**
```bash
LIBRARY_ID="550e8400-e29b-41d4-a716-446655440000"
curl "http://localhost:8000/v1/libraries/$LIBRARY_ID" | jq '.'
```

---

### 5. Add Documents (Automatic Embedding)

**Add a single document:**
```bash
curl -X POST "http://localhost:8000/v1/libraries/$LIBRARY_ID/documents" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [ "Transformers are a neural network architecture that has revolutionized natural language processing. They use self-attention mechanisms to process sequential data in parallel, making them much faster than recurrent neural networks.",
    "}
      "title": "Understanding Transformers",
      "author": "Jane Doe",
      "category": "Deep Learning",
      "published": "2024-03-15"
    }
  }' | jq '.'
```

**Response:**
```json
{
  "id": "a1b2c3d4-e5f6-4a5b-8c9d-0e1f2a3b4c5d",
  "library_id": "550e8400-e29b-41d4-a716-446655440000",
  "}
    "title": "Understanding Transformers",
    "author": "Jane Doe",
    "category": "Deep Learning",
    "published": "2024-03-15"
  },
  "chunks": [
    {
      "id": "chunk-001",
      "texts": [ "Transformers are a neural network architecture...",
      "chunk_index": 0
    }
  ],
  "created_at": "2025-10-21T12:00:00Z"
}
```

**Add a long document (auto-chunking):**
```bash
curl -X POST "http://localhost:8000/v1/libraries/$LIBRARY_ID/documents" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [ "'"$(cat long_article.txt)"'",
    "}
      "title": "Long Article",
      "source": "file"
    }
  }' | jq '.'
```

---

### 6. Search by Text Query

**Basic search:**
```bash
curl -X POST "http://localhost:8000/v1/libraries/$LIBRARY_ID/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "neural networks for language understanding",
    "k": 5
  }' | jq '.'
```

**Response:**
```json
{
  "results": [
    {
      "chunk_id": "chunk-001",
      "document_id": "a1b2c3d4-e5f6-4a5b-8c9d-0e1f2a3b4c5d",
      "similarity": 0.8542,
      "texts": [ "Transformers are a neural network architecture...",
      "}
        "title": "Understanding Transformers",
        "author": "Jane Doe"
      }
    },
    {
      "chunk_id": "chunk-002",
      "document_id": "b2c3d4e5-f6a7-4b8c-9d0e-1f2a3b4c5d6e",
      "similarity": 0.7891,
      "texts": [ "Natural language processing has been transformed...",
      "}
        "title": "NLP Revolution"
      }
    }
  ],
  "query": "neural networks for language understanding",
  "k": 5,
  "total_results": 2
}
```

**Search with distance threshold (only high similarity):**
```bash
curl -X POST "http://localhost:8000/v1/libraries/$LIBRARY_ID/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning algorithms",
    "k": 10,
    "distance_threshold": 0.7
  }' | jq '.'
```

---

### 7. Search with Pre-computed Embeddings

If you already have embeddings from another source:

```bash
curl -X POST "http://localhost:8000/v1/libraries/$LIBRARY_ID/search/embedding" \
  -H "Content-Type: application/json" \
  -d '{
    "embedding": [0.123, -0.456, 0.789, ...],
    "k": 5
  }' | jq '.'
```

**Note**: Embedding must be 1024-dimensional array to match Cohere's model.

---

### 8. Get Document Details

**Retrieve a specific document:**
```bash
DOC_ID="a1b2c3d4-e5f6-4a5b-8c9d-0e1f2a3b4c5d"
curl "http://localhost:8000/v1/documents/$DOC_ID" | jq '.'
```

**Response:**
```json
{
  "id": "a1b2c3d4-e5f6-4a5b-8c9d-0e1f2a3b4c5d",
  "library_id": "550e8400-e29b-41d4-a716-446655440000",
  "}
    "title": "Understanding Transformers",
    "author": "Jane Doe"
  },
  "chunks": [
    {
      "id": "chunk-001",
      "texts": [ "Transformers are a neural network architecture...",
      "chunk_index": 0
    }
  ],
  "created_at": "2025-10-21T12:00:00Z"
}
```

---

### 9. Delete Documents

**Delete a document:**
```bash
curl -X DELETE "http://localhost:8000/v1/documents/$DOC_ID"
```

**Response:**
```json
{
  "message": "Document deleted successfully",
  "document_id": "a1b2c3d4-e5f6-4a5b-8c9d-0e1f2a3b4c5d"
}
```

---

### 10. Get Library Statistics

**View library stats:**
```bash
curl "http://localhost:8000/v1/libraries/$LIBRARY_ID/statistics" | jq '.'
```

**Response:**
```json
{
  "library_id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "Research Papers",
  "index_type": "hnsw",
  "num_documents": 15,
  "num_chunks": 42,
  "num_vectors": 38,
  "index_size": 1024,
  "created_at": "2025-10-21T12:00:00Z"
}
```

**Note**: `num_vectors` can be less than `num_chunks` due to deduplication (identical text → same vector).

---

### 11. Delete Libraries

**Delete a library:**
```bash
curl -X DELETE "http://localhost:8000/v1/libraries/$LIBRARY_ID"
```

**Response:**
```json
{
  "message": "Library deleted successfully",
  "library_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

---

## Real-World Examples

### Example 1: Building a Knowledge Base

```bash
#!/bin/bash
# Create a company knowledge base

# 1. Create library
KB_ID=$(curl -s -X POST "http://localhost:8000/v1/libraries" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Company Knowledge Base",
    "description": "Internal documentation and FAQs",
    "index_type": "hnsw"
  }' | jq -r '.id')

# 2. Add FAQ entries
curl -s -X POST "http://localhost:8000/v1/libraries/$KB_ID/documents" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [ "To reset your password, go to the login page and click Forgot Password. Enter your email address and you will receive a reset link within 5 minutes.",
    "}
      "category": "Account",
      "topic": "Password Reset",
      "last_updated": "2024-10-15"
    }
  }'

curl -s -X POST "http://localhost:8000/v1/libraries/$KB_ID/documents" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [ "Our support team is available Monday-Friday 9am-5pm EST. You can reach us via email at support@company.com or through the live chat on our website.",
    "}
      "category": "Support",
      "topic": "Contact Information"
    }
  }'

# 3. Search the knowledge base
curl -s -X POST "http://localhost:8000/v1/libraries/$KB_ID/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do I change my password?",
    "k": 3
  }' | jq '.results[] | {similarity, title: .metadata.topic, text: .text}'
```

---

### Example 2: Semantic Code Search

```bash
#!/bin/bash
# Index code documentation

# 1. Create library
CODE_ID=$(curl -s -X POST "http://localhost:8000/v1/libraries" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Code Documentation",
    "description": "API and library documentation",
    "index_type": "hnsw"
  }' | jq -r '.id')

# 2. Add function documentation
curl -s -X POST "http://localhost:8000/v1/libraries/$CODE_ID/documents" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [ "def calculate_similarity(vec1, vec2): Computes the cosine similarity between two vectors. Returns a float between 0 and 1, where 1 means identical and 0 means orthogonal.",
    "}
      "function": "calculate_similarity",
      "file": "vector_utils.py",
      "type": "function"
    }
  }'

# 3. Search for similar functions
curl -s -X POST "http://localhost:8000/v1/libraries/$CODE_ID/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "compare two vectors",
    "k": 5
  }' | jq '.results[] | {similarity, function: .metadata.function, file: .metadata.file}'
```

---

### Example 3: Content Recommendation

```bash
#!/bin/bash
# Build a content recommendation system

# 1. Create library
CONTENT_ID=$(curl -s -X POST "http://localhost:8000/v1/libraries" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Blog Articles",
    "description": "Tech blog content",
    "index_type": "hnsw"
  }' | jq -r '.id')

# 2. Add articles
for article in articles/*.txt; do
  TITLE=$(basename "$article" .txt)
  TEXT=$(cat "$article")

  curl -s -X POST "http://localhost:8000/v1/libraries/$CONTENT_ID/documents" \
    -H "Content-Type: application/json" \
    -d "{
      \"text\": \"$TEXT\",
      \"metadata\": {
        \"title\": \"$TITLE\",
        \"source\": \"blog\"
      }
    }"
done

# 3. Find similar articles
curl -s -X POST "http://localhost:8000/v1/libraries/$CONTENT_ID/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning deployment best practices",
    "k": 5
  }' | jq '.results[] | {similarity, title: .metadata.title}'
```

---

## Helper Scripts

### Save Library ID for Reuse

```bash
# Save to file
echo "LIBRARY_ID=550e8400-e29b-41d4-a716-446655440000" > .env.library

# Load from file
source .env.library
curl "http://localhost:8000/v1/libraries/$LIBRARY_ID" | jq '.'
```

### Batch Add Documents from CSV

```bash
#!/bin/bash
# batch_import.sh - Import documents from CSV

LIBRARY_ID=$1
CSV_FILE=$2

tail -n +2 "$CSV_FILE" | while IFS=',' read -r title text author; do
  curl -s -X POST "http://localhost:8000/v1/libraries/$LIBRARY_ID/documents" \
    -H "Content-Type: application/json" \
    -d "{
      \"text\": \"$text\",
      \"metadata\": {
        \"title\": \"$title\",
        \"author\": \"$author\"
      }
    }" > /dev/null
  echo "Added: $title"
done
```

Usage:
```bash
./batch_import.sh $LIBRARY_ID documents.csv
```

### Pretty Search Results

```bash
#!/bin/bash
# search.sh - Pretty search output

LIBRARY_ID=$1
QUERY=$2

curl -s -X POST "http://localhost:8000/v1/libraries/$LIBRARY_ID/search" \
  -H "Content-Type: application/json" \
  -d "{
    \"query\": \"$QUERY\",
    \"k\": 5
  }" | jq -r '.results[] | "[\(.similarity * 100 | floor)%] \(.metadata.title // "Untitled")\n    \(.text[0:100])...\n"'
```

Usage:
```bash
./search.sh $LIBRARY_ID "neural networks"
```

Output:
```
[85%] Understanding Transformers
    Transformers are a neural network architecture that has revolutionized natural language processi...

[78%] Deep Learning Basics
    Neural networks are computing systems inspired by biological neural networks. They consist of la...
```

---

## Error Handling

**404 - Library Not Found:**
```bash
curl "http://localhost:8000/v1/libraries/invalid-id"
```
```json
{
  "detail": "Library not found"
}
```

**400 - Invalid Input:**
```bash
curl -X POST "http://localhost:8000/v1/libraries" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "",
    "index_type": "invalid"
  }'
```
```json
{
  "detail": [
    {
      "loc": ["body", "name"],
      "msg": "ensure this value has at least 1 character",
      "type": "value_error.any_str.min_length"
    }
  ]
}
```

**503 - Service Unavailable (Cohere API):**
```json
{
  "detail": "Embedding service temporarily unavailable"
}
```

---

## Tips & Best Practices

### 1. Save IDs for Later Use
```bash
# Create and save
LIBRARY_ID=$(curl -s -X POST ... | jq -r '.id')
echo $LIBRARY_ID > library_id.txt

# Load later
LIBRARY_ID=$(cat library_id.txt)
```

### 2. Use jq for Clean Output
```bash
# Pretty print
curl ... | jq '.'

# Extract specific fields
curl ... | jq '.results[] | {similarity, title: .metadata.title}'

# Filter by similarity
curl ... | jq '.results[] | select(.similarity > 0.8)'
```

### 3. Check Response Status
```bash
# Get HTTP status code
curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health
```

### 4. Use Variables for Readability
```bash
API_URL="http://localhost:8000"
HEADERS="-H Content-Type: application/json"

curl -s -X POST "$API_URL/libraries" $HEADERS -d '{...}'
```

### 5. Debug with Verbose Mode
```bash
curl -v -X POST ... # Shows request/response headers
```

---

## Next Steps

- **Try the examples** - Copy and paste commands to see results
- **Explore Swagger UI** - http://localhost:8000/docs for interactive testing
- **Read API docs** - [INDEX.md](INDEX.md) for complete endpoint reference
- **Run tests** - See [RUN_TESTS.md](../testing/RUN_TESTS.md) for test suite

---

## Questions?

**Q: How do I add embeddings I already have?**
A: Use `POST /libraries/{id}/documents/with-embeddings` endpoint with your 1024-dimensional vectors.

**Q: Can I search across multiple libraries?**
A: Not currently - each search is scoped to one library. Create separate libraries for different collections.

**Q: What's the max document size?**
A: No hard limit, but very long documents are automatically chunked for better search granularity.

**Q: How do I back up my data?**
A: Data is stored in `./data/` directory. The API uses WAL + snapshots for persistence.

**Q: Can I use this in production?**
A: Yes! 96% test coverage, thread-safe, persistent storage. See [README.md](../../README.md) for deployment options.

---

*For more examples and use cases, see the demo script at `/Users/bledden/Documents/SAI_demo_script.md`*
