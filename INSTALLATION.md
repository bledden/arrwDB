# Installation Guide

Choose the installation method that best fits your use case.

## Quick Start (Lightweight - Recommended)

**Most users should use this - excludes test files (80% smaller):**

```bash
# 1. Clone without tests
git clone --filter=blob:none --sparse https://github.com/bledden/SAI.git
cd SAI
git sparse-checkout set '/*' '!tests'

# 2. Install
pip install -e .

# 3. Configure
cp .env.example .env
# Edit .env and add your COHERE_API_KEY

# 4. Run
python run_api.py
```

**What you get**: Full API (2,096 lines), no test files, no test dependencies

---

## Full Clone (Alternative)

**Use this if you want test files on disk (even if not running them):**

```bash
# 1. Clone everything
git clone https://github.com/bledden/SAI.git
cd SAI

# 2. Install (production dependencies only)
pip install -e .

# 3. Configure and run
cp .env.example .env
# Edit .env and add your COHERE_API_KEY
python run_api.py
```

**What you get**: Full repo (10,578 lines including tests), but test tools NOT installed

---

## Development Setup (For Contributors)

**Use this if you want to run tests or contribute:**

```bash
# 1. Clone everything
git clone https://github.com/bledden/SAI.git
cd SAI

# 2. Install with test dependencies
pip install -e ".[dev]"

# 3. Run tests
pytest tests/

# 4. Run API
python run_api.py
```

**What you get**: Everything including 458 tests, coverage tools, etc.

---

## Installation Options Comparison

| Method | Repo Size | Install Time | Test Suite | Best For |
|--------|-----------|--------------|------------|----------|
| **Production** | Full repo | Fast | ❌ No | Running API, learning |
| **Development** | Full repo | Slower | ✅ Yes | Contributing, testing |
| **Lightweight** | 80% smaller | Fast | ❌ No | Minimal setup |

---

## What Gets Installed?

### Production Install (`pip install -e .`)

**Installed:**
- FastAPI, Uvicorn (API framework)
- Pydantic (validation)
- NumPy (vector operations)
- Cohere (embeddings)
- All 4 index algorithms
- Full API functionality

**NOT Installed:**
- pytest, coverage tools
- Development dependencies

### Development Install (`pip install -e ".[dev]"`)

**Everything from production PLUS:**
- pytest (testing framework)
- pytest-cov (coverage reporting)
- pytest-asyncio (async tests)
- black, ruff, mypy (code quality)
- httpx (test client)

---

## Optional: Temporal Workflows

```bash
# Add Temporal support
pip install -e ".[temporal]"

# Or with dev dependencies
pip install -e ".[dev,temporal]"
```

---

## Docker Installation

**Simplest for production deployment:**

```bash
git clone https://github.com/bledden/SAI.git
cd SAI
cp .env.example .env
# Edit .env with your COHERE_API_KEY
docker-compose up -d
```

Docker handles all dependencies automatically.

---

## Verification

After installation, verify it works:

```bash
# Test imports
python3 -c "import app.api.main; print('✅ Installation successful')"

# Start API
python run_api.py

# Visit http://localhost:8000/docs
```

---

## Troubleshooting

**Issue**: `ModuleNotFoundError: No module named 'app'`
- **Solution**: Make sure you ran `pip install -e .` (note the dot!)

**Issue**: `pip: command not found`
- **Solution**: Use `python3 -m pip install -e .`

**Issue**: Tests fail after `pip install -e .`
- **Expected**: Test dependencies not installed in production mode
- **Solution**: Use `pip install -e ".[dev]"` to get test dependencies

---

## Why This Approach?

✅ **Tests are opt-in** - Users don't download 8,482 lines of test code unless needed
✅ **Faster installs** - Production install skips pytest, coverage tools, etc.
✅ **Professional** - Follows Python packaging best practices (PEP 518)
✅ **Flexible** - Easy to switch between production and development modes

---

## Next Steps

- [Quick Start Guide](docs/guides/QUICKSTART.md)
- [API Documentation](docs/guides/INDEX.md)
- [Contributing Guide](CONTRIBUTING.md) *(if exists)*
