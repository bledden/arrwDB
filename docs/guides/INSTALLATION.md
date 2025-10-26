# Installation Guide

Choose the installation method that best fits your use case.

## Quick Start (Lightweight - Recommended)

**Most users should use this - excludes test files (80% smaller):**

```bash
# 1. Clone without tests
git clone --filter=blob:none --sparse https://github.com/bledden/arrwDB.git
cd arrwDB
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
git clone https://github.com/bledden/arrwDB.git
cd arrwDB

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
git clone https://github.com/bledden/arrwDB.git
cd arrwDB

# 2. Install with test dependencies
pip install -e ".[dev]"

# 3. Run API
python run_api.py

# 4. Run tests
python pytest tests/
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

### Installing Docker

**First-time Docker users need to install Docker:**

#### macOS (Apple Silicon / M1/M2/M3)

```bash
# Option 1: Download Docker Desktop (Recommended - includes GUI)
# Visit: https://desktop.docker.com/mac/main/arm64/Docker.dmg
# Or open in browser:
open https://desktop.docker.com/mac/main/arm64/Docker.dmg

# Option 2: Install via Homebrew
brew install --cask docker
```

#### macOS (Intel)

```bash
# Option 1: Download Docker Desktop
# Visit: https://desktop.docker.com/mac/main/amd64/Docker.dmg

# Option 2: Install via Homebrew
brew install --cask docker
```

#### Windows 10/11

```bash
# Download Docker Desktop for Windows
# Visit: https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe

# Requirements:
# - Windows 10/11 64-bit: Pro, Enterprise, or Education
# - Enable WSL 2 (Windows Subsystem for Linux)
```

#### Linux (Ubuntu/Debian)

```bash
# Install Docker Engine
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add your user to docker group (no sudo needed)
sudo usermod -aG docker $USER
newgrp docker

# Install Docker Compose
sudo apt-get update
sudo apt-get install docker-compose-plugin
```

#### Linux (Fedora/RHEL)

```bash
# Install Docker
sudo dnf -y install dnf-plugins-core
sudo dnf config-manager --add-repo https://download.docker.com/linux/fedora/docker-ce.repo
sudo dnf install docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Start Docker
sudo systemctl start docker
sudo systemctl enable docker
```

**Verify Docker Installation:**

```bash
docker --version
docker compose version
```

### Running the API with Docker

**Once Docker is installed:**

```bash
# 1. Clone the repository
git clone https://github.com/bledden/arrwDB.git
cd arrwDB

# 2. Configure environment
cp .env.example .env
# Edit .env with your COHERE_API_KEY

# 3. Start the API
docker-compose up -d

# 4. Verify it's running
curl http://localhost:8000/health

# 5. View logs
docker-compose logs -f vector-db-api

# 6. Stop the services
docker-compose down
```

**What Gets Deployed:**
- Vector Database API (port 8000)
- Temporal Server (workflow orchestration)
- PostgreSQL (Temporal's database)
- Temporal Web UI (port 8088)
- Temporal Worker (background jobs)

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
