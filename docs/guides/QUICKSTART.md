# Quick Start Guide ⚡

**Your Vector Database API is already set up and ready to use!**

## ✅ What's Already Done

- ✅ Python dependencies installed
- ✅ Cohere API key configured
- ✅ `.env` file created
- ✅ All tests passing

## 🚀 Start in 3 Steps

### 1. Test It Works

```bash
cd /Users/bledden/Documents/SAI
python3 test_basic_functionality.py
```

Expected output:
```
✓ ALL TESTS PASSED SUCCESSFULLY!
```

### 2. Start the API

```bash
python3 run_api.py
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 3. Try It Out

Open your browser:
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

Or use curl:
```bash
curl http://localhost:8000/health
```

## 💡 First API Calls

### Using Python SDK

```python
from sdk import VectorDBClient

# Connect
client = VectorDBClient("http://localhost:8000")

# Create library
library = client.create_library(
    name="My First Library",
    index_type="hnsw"  # Best for most use cases
)

# Add document (embeddings auto-generated)
doc = client.add_document(
    library_id=library["id"],
    title="Introduction to ML",
    texts=[
        "Machine learning is a subset of AI.",
        "It uses data to make predictions.",
        "Deep learning uses neural networks."
    ]
)

# Search
results = client.search(
    library_id=library["id"],
    query="What is machine learning?",
    k=2
)

# Print results
for r in results["results"]:
    print(f"Score: {1 - r['distance']:.2f}")
    print(f"Text: {r['chunk']['text']}\n")
```

### Using curl

```bash
# Create library
curl -X POST http://localhost:8000/libraries \
  -H "Content-Type: application/json" \
  -d '{"name": "Test", "index_type": "hnsw"}'

# Copy the "id" from response, then add document:
curl -X POST http://localhost:8000/libraries/YOUR_LIBRARY_ID/documents \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Test Doc",
    "texts": ["This is a test document."]
  }'

# Search:
curl -X POST http://localhost:8000/libraries/YOUR_LIBRARY_ID/search \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "k": 5}'
```

## 📚 Index Types

Choose based on your needs:

| Index | Best For | Speed | Accuracy |
|-------|----------|-------|----------|
| `brute_force` | < 100K vectors | Slow | 100% |
| `kd_tree` | < 20 dimensions | Medium | 100% |
| `hnsw` | Production (recommended) | Fast | ~98% |
| `lsh` | > 10M vectors | Very Fast | ~90% |

## 🛠️ Common Commands

```bash
# Start API
python3 run_api.py

# Stop API
# Press Ctrl+C

# Run tests
python3 test_basic_functionality.py

# Check health
curl http://localhost:8000/health

# View docs
open http://localhost:8000/docs  # macOS
xdg-open http://localhost:8000/docs  # Linux
```

## 🐛 Quick Troubleshooting

**"Address already in use"**
```bash
# Kill existing server
pkill -f "python3 run_api.py"
```

**"No module named..."**
```bash
pip3 install -r requirements.txt
```

**"COHERE_API_KEY not found"**
```bash
# Check .env file exists
ls -la .env
cat .env | grep COHERE_API_KEY
```

## 📖 More Information

- **Full Documentation**: [README.md](README.md)
- **Installation Guide**: [INSTALLATION.md](INSTALLATION.md)
- **Implementation Details**: [IMPLEMENTATION_COMPLETE.md](IMPLEMENTATION_COMPLETE.md)

## 🎯 What's Possible

With this Vector Database you can:

✅ Store and search text documents by semantic meaning
✅ Build RAG (Retrieval-Augmented Generation) applications
✅ Create semantic search engines
✅ Find similar documents automatically
✅ Process millions of documents with HNSW index
✅ Run everything locally or in Docker
✅ Integrate with any application via REST API or Python SDK

## 🚀 Ready to Build!

Your Vector Database is production-ready. Start building your AI applications!

For help, check the docs or run:
```bash
python3 -c "from sdk import VectorDBClient; help(VectorDBClient)"
```
