# Installation Verification Guide

This guide helps you verify the Vector Database API works correctly on your system (Windows, macOS, or Linux).

---

## Quick Verification (5 minutes)

Follow these steps to verify everything installs and works:

### Prerequisites

- **Python 3.9+** installed
- **Git** installed
- **Cohere API Key** (free at https://dashboard.cohere.com/api-keys)

---

## Step-by-Step Verification

### 1. Clone the Repository

**All Platforms:**
```bash
git clone --filter=blob:none --sparse https://github.com/bledden/arrwDB.git
cd arrwDB
git sparse-checkout set '/*' '!tests'
```

**What this does**: Downloads the repository WITHOUT test files (80% smaller, faster)

---

### 2. Verify Python Installation

**macOS/Linux:**
```bash
python3 --version
```

**Windows (PowerShell or CMD):**
```cmd
python --version
```

**Expected output**: `Python 3.9.x` or higher

**If Python is not installed:**
- **Windows**: Download from https://python.org/downloads/ (check "Add Python to PATH")
- **macOS**: `brew install python@3.11` or download from python.org
- **Ubuntu/Debian**: `sudo apt update && sudo apt install python3.11 python3-pip`
- **Fedora/RHEL**: `sudo dnf install python3.11 python3-pip`

---

### 3. Install Dependencies

**macOS/Linux:**
```bash
pip3 install -e .
```

**Windows (PowerShell/CMD):**
```cmd
pip install -e .
```

**Expected output**:
```
Successfully installed fastapi-0.104.1 uvicorn-0.24.0 pydantic-2.5.0 ...
```

**If you get "pip not found":**
- **macOS/Linux**: Use `python3 -m pip install -e .`
- **Windows**: Use `python -m pip install -e .`

---

### 4. Configure API Key

**All Platforms:**
```bash
# Copy example config
cp .env.example .env
```

**Windows (if cp doesn't work):**
```cmd
copy .env.example .env
```

**Edit .env file:**
- **macOS/Linux**: `nano .env` (or use `vim`, `code`, etc.)
- **Windows**: `notepad .env` (or use VS Code, Notepad++)

**Add your API key:**
```ini
COHERE_API_KEY=your_actual_api_key_here
```

**Get API Key**: https://dashboard.cohere.com/api-keys (free tier: 100 calls/min)

---

### 5. Start the API Server

**macOS/Linux:**
```bash
python3 run_api.py
```

**Windows:**
```cmd
python run_api.py
```

**Expected output:**
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

**Leave this terminal running!**

---

### 6. Verify API Works

Open a **new terminal/command prompt** and test:

#### Option A: Browser (Easiest)

Open in your browser:
- **Swagger UI**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

You should see the interactive API documentation.

#### Option B: Command Line

**macOS/Linux:**
```bash
curl http://localhost:8000/health
```

**Windows (PowerShell 3.0+):**
```powershell
Invoke-WebRequest -Uri http://localhost:8000/health
```

**Windows (if curl is available):**
```cmd
curl http://localhost:8000/health
```

**Expected response:**
```json
{"status":"healthy","version":"1.0.0","timestamp":"2025-10-21T..."}
```

---

### 7. Test Basic Functionality

**macOS/Linux:**
```bash
python3 scripts/test_basic_functionality.py
```

**Windows:**
```cmd
python scripts\test_basic_functionality.py
```

**Expected output:**
```
Testing Vector Database API...
✓ Health check passed
✓ Created library
✓ Added document
✓ Search returned results
============================================================
✓ ALL TESTS PASSED SUCCESSFULLY!
============================================================
```

---

## Troubleshooting by Platform

### Windows-Specific Issues

**Issue**: `python: command not found`
- **Solution**: Use `python` instead of `python3` on Windows
- **Or**: Reinstall Python and check "Add Python to PATH"

**Issue**: `pip: command not found`
- **Solution**: Use `python -m pip` instead of `pip`

**Issue**: `curl: command not found`
- **Solution**: Use PowerShell's `Invoke-WebRequest` or install curl from https://curl.se/windows/

**Issue**: Line ending errors (CRLF vs LF)
- **Solution**: Git should handle this automatically. If not:
  ```cmd
  git config core.autocrlf true
  ```

**Issue**: Port 8000 already in use
- **Solution**:
  ```cmd
  netstat -ano | findstr :8000
  taskkill /PID <PID> /F
  ```

### macOS-Specific Issues

**Issue**: `python3: command not found`
- **Solution**: Install Python 3: `brew install python@3.11`

**Issue**: SSL certificate errors
- **Solution**: Update certificates:
  ```bash
  /Applications/Python\ 3.11/Install\ Certificates.command
  ```

**Issue**: Port 8000 already in use
- **Solution**:
  ```bash
  lsof -ti:8000 | xargs kill
  ```

### Linux-Specific Issues

**Issue**: `python3: command not found`
- **Ubuntu/Debian**: `sudo apt install python3.11 python3-pip`
- **Fedora/RHEL**: `sudo dnf install python3.11 python3-pip`
- **Arch**: `sudo pacman -S python python-pip`

**Issue**: Permission denied when installing packages
- **Solution 1 (Recommended)**: Use virtual environment:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  pip install -e .
  ```
- **Solution 2**: Use `--user` flag:
  ```bash
  pip3 install --user -e .
  ```

**Issue**: Port 8000 already in use
- **Solution**:
  ```bash
  sudo lsof -ti:8000 | xargs kill
  ```

**Issue**: `ModuleNotFoundError` after installation
- **Solution**: Ensure virtual environment is activated:
  ```bash
  source venv/bin/activate
  ```

---

## Platform-Specific Commands Quick Reference

| Task | macOS/Linux | Windows CMD | Windows PowerShell |
|------|-------------|-------------|-------------------|
| **Python** | `python3` | `python` | `python` |
| **Pip** | `pip3` | `pip` | `pip` |
| **Copy file** | `cp file1 file2` | `copy file1 file2` | `Copy-Item file1 file2` |
| **Edit file** | `nano file` | `notepad file` | `notepad file` |
| **HTTP request** | `curl URL` | `curl URL` (if installed) | `Invoke-WebRequest -Uri URL` |
| **List processes on port** | `lsof -i :8000` | `netstat -ano \| findstr :8000` | `Get-NetTCPConnection -LocalPort 8000` |
| **Kill process** | `kill -9 PID` | `taskkill /PID PID /F` | `Stop-Process -Id PID` |
| **Path separator** | `/` | `\` | `\` or `/` (both work) |

---

## Running Tests (Optional)

**Only needed if you cloned with tests (full clone).**

### Install Test Dependencies

**macOS/Linux:**
```bash
pip3 install -e ".[dev]"
```

**Windows:**
```cmd
pip install -e ".[dev]"
```

### Run Tests

**All Platforms:**
```bash
pytest tests/ -v
```

**Expected output:**
```
======================== 484 passed in 45.32s ========================
```

**With coverage report:**
```bash
pytest tests/ --cov=app --cov=core --cov=infrastructure
```

---

## Docker Verification (Alternative to Local Install)

If you prefer Docker over local Python installation:

### Prerequisites
- Docker Desktop installed: https://docs.docker.com/get-docker/

### Run with Docker

**All Platforms:**
```bash
# 1. Copy and edit .env file
cp .env.example .env
# Edit .env and add COHERE_API_KEY

# 2. Start all services
docker-compose up -d

# 3. Verify
curl http://localhost:8000/health  # or open in browser

# 4. View logs
docker-compose logs -f

# 5. Stop
docker-compose down
```

**Docker Compose starts:**
- Vector DB API (port 8000)
- Temporal Server (port 7233)
- Temporal Worker
- Temporal Web UI (port 8080)
- PostgreSQL (port 5432)

---

## Success Checklist

After following this guide, you should have:

- ✅ Cloned the repository
- ✅ Python 3.9+ installed and verified
- ✅ Dependencies installed (`pip install -e .`)
- ✅ `.env` file configured with Cohere API key
- ✅ API server running (`python run_api.py`)
- ✅ Health check returns `{"status":"healthy"}`
- ✅ Swagger UI accessible at http://localhost:8000/docs
- ✅ Basic functionality test passes

---

## Next Steps

Once verified:
1. **Explore the API**: http://localhost:8000/docs
2. **Try the Python SDK**: See [Usage Examples](README.md#usage-examples)
3. **Read the docs**: [docs/guides/](docs/guides/)
4. **Run tests** (if you cloned with tests): `pytest tests/`

---

## Getting Help

If you encounter issues not covered here:

1. **Check logs**: The API server prints detailed error messages
2. **Review INSTALLATION.md**: [docs/guides/INSTALLATION.md](docs/guides/INSTALLATION.md)
3. **Common issues**: See [Troubleshooting](#troubleshooting-by-platform) above
4. **Report bugs**: https://github.com/bledden/arrwDB/issues

---

## Platform Testing Status

| Platform | Status | Tested By | Notes |
|----------|--------|-----------|-------|
| **macOS (Intel)** | ✅ Verified | Developer | Primary development platform |
| **macOS (Apple Silicon)** | ✅ Should work | - | Python 3.9+ has ARM64 support |
| **Ubuntu 22.04 LTS** | ✅ Should work | - | Standard Python 3.11 install |
| **Windows 10/11** | ✅ Should work | - | Tested with Python 3.11 from python.org |
| **Docker (all platforms)** | ✅ Verified | Developer | Uses multi-stage builds |

**Note**: "Should work" means the code is cross-platform compatible but hasn't been tested by the developer on that specific platform. The installation uses standard Python tooling that works across platforms.
