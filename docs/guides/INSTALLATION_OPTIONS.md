# Installation Options Guide

This guide helps you choose the best installation method based on your use case.

## TL;DR - Quick Decision

| Use Case | Installation Type | Repository Size | Command |
|----------|------------------|-----------------|---------|
| **Development & Testing** | Full | ~10,600 lines | `git clone https://github.com/bledden/SAI.git` |
| **Production Deployment** | Lightweight | ~2,100 lines (80% smaller) | See [Lightweight Installation](#lightweight-installation-production-only) |
| **Learning/Reading Code** | Lightweight | ~2,100 lines (80% smaller) | See [Lightweight Installation](#lightweight-installation-production-only) |
| **Contributing** | Full | ~10,600 lines | `git clone https://github.com/bledden/SAI.git` |

---

## Why Multiple Installation Options?

The Vector Database API has a **4:1 test-to-code ratio**:
- **Production Code**: 2,096 lines (app, core, infrastructure)
- **Test Code**: 8,482 lines (tests directory)
- **Total**: 10,578 lines

If you only need to **run the API** or **study the code**, you can save 80% disk space and clone time by excluding tests.

---

## Full Installation (Default)

### When to Use
- ✅ You want to run the test suite
- ✅ You plan to contribute or modify code
- ✅ You want to verify all 458 tests pass
- ✅ You need the full development experience

### Installation Steps

```bash
# Clone the entire repository
git clone https://github.com/bledden/SAI.git
cd SAI

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env and add your COHERE_API_KEY

# Run tests to verify
pytest tests/ -v

# Start the API
python run_api.py
```

### What You Get
- ✅ Complete codebase (2,096 lines)
- ✅ Full test suite (458 tests, 8,482 lines)
- ✅ Test fixtures and utilities
- ✅ Coverage reporting tools
- ✅ Can contribute and run tests

### Repository Structure
```
SAI/
├── app/                    # 745 lines - REST API
├── core/                   # 202 lines - Core logic
├── infrastructure/         # 1,149 lines - Indexes, persistence
├── tests/                  # 8,482 lines - Test suite ⭐
│   ├── unit/              # 7,371 lines
│   ├── integration/       # 543 lines
│   └── test_edge_cases.py # 320 lines
├── temporal/              # Workflows
├── sdk/                   # Python client
└── docs/                  # Documentation
```

---

## Lightweight Installation (Production Only)

### When to Use
- ✅ You only want to run the API
- ✅ You're deploying to production
- ✅ You want to study the code
- ✅ You want faster clone and minimal disk space
- ✅ You don't need to run tests

### Installation Steps

**Method 1: Sparse Checkout (Recommended - No Bandwidth Waste!)**
```bash
# Clone with sparse checkout - tests never downloaded!
git clone --filter=blob:none --sparse https://github.com/bledden/SAI.git
cd SAI

# Configure to checkout everything except tests
git sparse-checkout set '/*' '!tests'

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install only production dependencies
pip install fastapi uvicorn pydantic numpy cohere python-dotenv slowapi

# Set up environment
cp .env.example .env
# Edit .env and add your COHERE_API_KEY

# Start the API
python run_api.py
```

**Why This Is Better**: The tests directory is **never downloaded**, saving bandwidth and disk space. No wasted downloads!

**Method 2: Partial Clone with Blobless Filter (Alternative)**
```bash
# Clone without downloading file blobs initially
git clone --filter=blob:none --no-checkout https://github.com/bledden/SAI.git
cd SAI

# Enable sparse checkout
git sparse-checkout init --cone

# Checkout everything except tests
git sparse-checkout set '/*' '!tests'

# Now checkout the working tree
git checkout main

# Continue with installation...
pip install fastapi uvicorn pydantic numpy cohere python-dotenv slowapi
```

**Method 3: Simple Clone + Remove (Fallback for Old Git)**
```bash
# For Git versions < 2.25 that don't support sparse checkout
git clone --depth 1 https://github.com/bledden/SAI.git
cd SAI
rm -rf tests/  # Less efficient - tests already downloaded

# Continue with installation...
pip install fastapi uvicorn pydantic numpy cohere python-dotenv slowapi
```

**Note**: Method 1 is most efficient - it never downloads the 8,482 lines of test code!

### What You Get
- ✅ Complete production code (2,096 lines)
- ✅ All 4 index algorithms
- ✅ Full REST API
- ✅ Temporal workflows
- ✅ Python SDK
- ✅ Documentation
- ❌ No test suite (can't run pytest)
- ❌ No coverage reporting

### Repository Structure
```
SAI/
├── app/                    # 745 lines - REST API
├── core/                   # 202 lines - Core logic
├── infrastructure/         # 1,149 lines - Indexes, persistence
├── temporal/              # Workflows
├── sdk/                   # Python client
└── docs/                  # Documentation
```

### Disk Space & Bandwidth Comparison
| Installation | Lines of Code | Disk Space | Bandwidth Used | Clone Time* |
|--------------|---------------|------------|----------------|-------------|
| Full | 10,578 | ~15 MB | ~15 MB | ~5-10 sec |
| Lightweight (sparse) | 2,096 | ~3 MB | ~3 MB | ~2-3 sec |
| Lightweight (rm -rf) | 2,096 | ~3 MB | ~15 MB (wasted!) | ~5-10 sec |
| **Savings (sparse)** | **80% less** | **80% less** | **80% less** | **60% faster** |

*Approximate, depends on network speed

**Important**: Use sparse checkout (Method 1) to avoid downloading tests entirely. The `rm -rf` method still wastes bandwidth downloading files you'll delete!

---

## Production Dependencies Only

If you're using the lightweight installation, you only need these packages:

```bash
# Minimal production dependencies
pip install \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    pydantic==2.5.0 \
    numpy==1.24.3 \
    cohere==4.37 \
    python-dotenv==1.0.0 \
    slowapi==0.1.9
```

**Optional for Temporal workflows:**
```bash
pip install temporalio==1.4.0
```

**Optional for development:**
```bash
pip install \
    pytest==7.4.3 \
    pytest-asyncio==0.21.1 \
    pytest-cov==4.1.0 \
    httpx==0.25.2
```

---

## Docker Installation

Docker provides the same API regardless of which git installation you use.

```bash
# Clone (full or lightweight, doesn't matter for Docker)
git clone https://github.com/bledden/SAI.git
cd SAI

# Configure
cp .env.example .env
# Edit .env and add COHERE_API_KEY

# Build and run
docker-compose up -d

# Access API
curl http://localhost:8000/health
```

Docker image size: ~450 MB (production dependencies only, no test files included)

---

## Comparison Matrix

| Feature | Full Installation | Lightweight | Docker |
|---------|------------------|-------------|--------|
| **Repository Size** | 10,578 lines | 2,096 lines | 2,096 lines |
| **Disk Space** | ~15 MB | ~3 MB | ~450 MB (image) |
| **Clone Time** | ~5-10 sec | ~2-3 sec | ~5-10 sec |
| **Can Run API** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Can Run Tests** | ✅ Yes (458 tests) | ❌ No | ❌ No |
| **Code Coverage** | ✅ Yes (96%) | ❌ No | ❌ No |
| **For Development** | ✅ Recommended | ⚠️ Limited | ⚠️ Limited |
| **For Production** | ✅ Yes | ✅ Yes | ✅ Recommended |
| **For Learning** | ✅ Yes | ✅ Recommended | ⚠️ Harder |
| **Can Contribute** | ✅ Yes | ❌ No | ❌ No |

---

## Recommendations

### Development Team
```bash
# Full installation - you need tests
git clone https://github.com/bledden/SAI.git
pip install -r requirements.txt
pytest tests/ -v
```

### Production Deployment
```bash
# Docker is easiest
docker-compose up -d
```

### Learning/Reading Code
```bash
# Lightweight - faster and cleaner
git clone --depth 1 https://github.com/bledden/SAI.git
rm -rf tests/
pip install fastapi uvicorn pydantic numpy cohere
```

### Contributing
```bash
# Full installation required
git clone https://github.com/bledden/SAI.git
pip install -r requirements.txt
# Make changes
pytest tests/ -v  # Verify tests pass
```

---

## FAQ

**Q: Will the lightweight installation work with the API?**
A: Yes! The API runs identically. Tests are only for verification.

**Q: Can I switch between installations?**
A: Yes. Just delete and re-clone, or run `git checkout tests/` to restore tests.

**Q: What's the test-to-code ratio?**
A: 4:1 - We have 8,482 lines of tests for 2,096 lines of production code. This is exceptional coverage!

**Q: Why so many tests?**
A: High coverage (96%), comprehensive edge case testing, and zero mocking means tests verify real behavior.

**Q: Do I need tests to deploy?**
A: No. Tests verify correctness during development. Production only needs the code.

**Q: Can I run some tests without the full suite?**
A: Not easily. Tests have dependencies in `tests/conftest.py`. Use Docker for production.

---

## Next Steps

After installation, see:
- [Quick Start Guide](QUICKSTART.md) - Get started in 5 minutes
- [API Reference](INDEX.md) - Complete endpoint documentation
- [Main README](../../README.md) - Project overview

---

## Support

For issues or questions:
1. Check [Installation Guide](INSTALLATION.md) for troubleshooting
2. Review [FAQ section](#faq) above
3. See [documentation index](../README.md)
