# Scripts

Utility scripts for development and testing.

## Available Scripts

### test_basic_functionality.py
Basic functionality test that runs without the API server.

Tests the core functionality by directly using the service layer, repository, and indexes.

**Usage**:
```bash
# Set up environment
export COHERE_API_KEY="your_key_here"

# Run the test
python scripts/test_basic_functionality.py
```

**What it tests**:
- Library creation
- Document addition with text chunks
- Vector embedding generation
- k-NN similarity search
- Index operations

**Use this when**:
- You want to test core functionality without starting the API server
- You're debugging the service or repository layers
- You want a quick smoke test

## Main Entry Points

For normal use, use the main entry points:
- `run_api.py` - Start the REST API server
- `pytest tests/` - Run the full test suite
