# Installation Guide

## Prerequisites

- **Python 3.9+** (3.11+ recommended)
- **pip** (Python package manager)
- **Cohere API Key** (free at [dashboard.cohere.com](https://dashboard.cohere.com/api-keys))

## Local Installation (Tested & Working)

### Step 1: Verify Python

```bash
python3 --version  # Should be 3.9 or higher
```

If you need to install Python:
- macOS: `brew install python@3.11`
- Ubuntu/Debian: `sudo apt install python3.11 python3-pip`
- Windows: Download from [python.org](https://www.python.org/downloads/)

### Step 2: Install Dependencies

```bash
cd /Users/bledden/Documents/SAI

# Install all required packages
pip3 install -r requirements.txt
```

**Required packages (already in requirements.txt):**
- fastapi==0.104.1
- uvicorn[standard]==0.24.0
- pydantic==2.5.0
- numpy==1.26.2
- cohere==4.37
- temporalio==1.5.1
- tenacity==8.2.3
- python-dotenv==1.0.0
- requests==2.31.0

### Step 3: Configure API Key

The `.env` file is already configured with your Cohere API key:

```bash
# Verify .env file exists
cat .env | grep COHERE_API_KEY
```

If you need to update it:
```bash
nano .env
# Or use any text editor to edit COHERE_API_KEY
```

### Step 4: Create Data Directory

```bash
mkdir -p data/vectors data/wal data/snapshots
```

### Step 5: Test the Installation

```bash
# Run the basic functionality test
python3 test_basic_functionality.py
```

You should see:
```
============================================================
✓ ALL TESTS PASSED SUCCESSFULLY!
============================================================
```

### Step 6: Start the API Server

```bash
python3 run_api.py
```

You should see:
```
INFO:     Started server process [XXXXX]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### Step 7: Verify API is Running

Open a new terminal and test:

```bash
# Health check
curl http://localhost:8000/health

# Should return:
# {"status":"healthy","version":"1.0.0","timestamp":"..."}
```

Or open in your browser:
- **API Documentation**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

### Step 8: Try a Quick Test

Using the Python SDK:

```bash
python3 << 'EOF'
from sdk import VectorDBClient

# Create client
client = VectorDBClient("http://localhost:8000")

# Check health
health = client.health_check()
print(f"API Status: {health['status']}")

# Create a library
library = client.create_library(
    name="Test Library",
    index_type="brute_force"
)
print(f"Created library: {library['id']}")

# Add a document
doc = client.add_document(
    library_id=library['id'],
    title="Test Document",
    texts=["This is a test document about machine learning."]
)
print(f"Added document: {doc['id']}")

# Search
results = client.search(
    library_id=library['id'],
    query="machine learning",
    k=1
)
print(f"Search found {len(results['results'])} results")
print(f"Query time: {results['query_time_ms']}ms")

print("\n✓ All operations successful!")
EOF
```

## Docker Installation (Alternative)

### Prerequisites

- **Docker**: https://docs.docker.com/get-docker/
- **Docker Compose**: Usually included with Docker Desktop

### For macOS/Linux:

```bash
# Install Docker Desktop (includes docker-compose)
# macOS: Download from docker.com
# Linux:
sudo apt install docker.io docker-compose  # Ubuntu/Debian
brew install docker docker-compose         # macOS with Homebrew
```

### Using Docker Compose v2 (Built into Docker):

Modern Docker includes compose as a subcommand:

```bash
# Instead of: docker-compose up -d
# Use: docker compose up -d (no hyphen)

docker compose up -d
```

### Verify Docker Installation:

```bash
docker --version
docker compose version  # or: docker-compose --version
```

### Start All Services:

```bash
cd /Users/bledden/Documents/SAI

# Using Docker Compose v2 (recommended)
docker compose up -d

# Or using Docker Compose v1
docker-compose up -d
```

This starts:
- Vector DB API (port 8000)
- Temporal Server (port 7233)
- Temporal Worker
- Temporal Web UI (port 8080)
- PostgreSQL (port 5432)

### Check Docker Services:

```bash
docker compose ps
docker compose logs -f vector-db-api
```

### Stop Docker Services:

```bash
docker compose down
```

## Troubleshooting

### Issue: "command not found: python3"

**Solution**: Install Python 3 or use `python` instead:
```bash
python --version  # Check if python works
```

### Issue: "No module named 'fastapi'"

**Solution**: Install requirements:
```bash
pip3 install -r requirements.txt
```

### Issue: "COHERE_API_KEY environment variable must be set"

**Solution**: Make sure `.env` file exists and has your key:
```bash
cat .env | grep COHERE_API_KEY
# Should show: COHERE_API_KEY=pa6s...
```

### Issue: "Address already in use" (Port 8000)

**Solution**: Either kill the process using port 8000:
```bash
# Find process
lsof -i :8000

# Kill it
kill -9 <PID>
```

Or change the port in `.env`:
```bash
API_PORT=8001
```

### Issue: "docker-compose: command not found"

**Solutions**:

1. **Use Docker Compose v2** (recommended):
```bash
docker compose up -d  # Note: no hyphen
```

2. **Install Docker Compose v1**:
```bash
# macOS
brew install docker-compose

# Linux
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

3. **Use Python pip**:
```bash
pip3 install docker-compose
```

### Issue: Cannot connect to Cohere API

**Check**:
1. Internet connection
2. API key is valid
3. Not rate limited (free tier: 100 calls/minute)

### Issue: Tests failing

**Debug**:
```bash
# Run with verbose output
python3 test_basic_functionality.py 2>&1 | tee test_output.log

# Check data directory permissions
ls -la data/

# Check if port 8000 is available
lsof -i :8000
```

## Environment Variables

All configuration is in the `.env` file:

```bash
# Required
COHERE_API_KEY=your_key_here

# Optional (with defaults)
VECTOR_DB_DATA_DIR=./data           # Data storage location
API_HOST=0.0.0.0                    # API host
API_PORT=8000                       # API port
API_WORKERS=1                       # Number of workers
API_RELOAD=false                    # Auto-reload on code changes
EMBEDDING_MODEL=embed-english-v3.0  # Cohere model
EMBEDDING_DIMENSION=1024            # Embedding dimension
TEMPORAL_HOST=localhost:7233        # Temporal server
TEMPORAL_NAMESPACE=default          # Temporal namespace
TEMPORAL_TASK_QUEUE=vector-db-task-queue  # Task queue name
```

## Development Setup

For development with auto-reload:

```bash
# Edit .env
API_RELOAD=true
API_WORKERS=1  # Must be 1 with reload

# Start server
python3 run_api.py

# Server will restart on code changes
```

## Performance Tips

1. **For production**: Increase workers in `.env`:
   ```
   API_WORKERS=4
   ```

2. **For large datasets**: Use memory-mapped storage automatically kicks in at >10K documents

3. **Index selection**:
   - Small datasets (< 100K): `brute_force`
   - Low dimensions (< 20D): `kd_tree`
   - Large datasets: `hnsw` (recommended)
   - Extreme scale: `lsh`

## Verification Checklist

- [ ] Python 3.9+ installed
- [ ] All dependencies installed (`pip3 install -r requirements.txt`)
- [ ] `.env` file configured with COHERE_API_KEY
- [ ] Data directories created
- [ ] Basic test passes (`python3 test_basic_functionality.py`)
- [ ] API server starts (`python3 run_api.py`)
- [ ] Health check responds (`curl http://localhost:8000/health`)
- [ ] Can access docs at http://localhost:8000/docs

## Next Steps

Once everything is installed and running:

1. **Read the README.md** for usage examples
2. **Try the Python SDK** (see examples in README)
3. **Explore the API docs** at http://localhost:8000/docs
4. **Run your own tests** with your data
5. **Check IMPLEMENTATION_COMPLETE.md** for feature list

## Support

If you encounter issues:

1. Check this troubleshooting section
2. Verify all prerequisites are met
3. Check the logs for specific errors
4. Ensure API key is valid and has quota remaining
5. Try the basic test first before running the full API

## System Requirements

**Minimum**:
- CPU: 2 cores
- RAM: 4 GB
- Disk: 1 GB free space
- Network: Internet connection for Cohere API

**Recommended**:
- CPU: 4+ cores
- RAM: 8+ GB
- Disk: 10+ GB for large datasets
- SSD for better performance

## Installation Complete!

If all steps completed successfully, you now have a fully functional Vector Database API running locally! 🎉
