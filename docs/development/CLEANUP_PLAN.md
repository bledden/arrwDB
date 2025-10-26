# arrwDB Cleanup Plan

## Overview
This document outlines the cleanup and reorganization of the arrwDB codebase to improve structure and maintainability.

---

## ✅ To Execute Immediately

### 1. Clean Up Cache Files
```bash
# Remove Python cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete
find . -name "*.pyo" -delete

# Remove Rust target directories from version control (keep for builds)
echo "rust_*/target/" >> .gitignore
git rm -r --cached rust_hnsw/target/
git rm -r --cached rust_vector_store/target/ 2>/dev/null
git rm -r --cached rust_wal/target/ 2>/dev/null
git rm -r --cached rust_snapshot/target/ 2>/dev/null
```

### 2. Move Test Files to Proper Location
```bash
# Move integration tests
mkdir -p tests/integration
mv test_batch_operations.py tests/integration/
mv test_index_management.py tests/integration/
mv test_persistence.py tests/integration/
```

### 3. Organize Documentation
```bash
# Create docs structure
mkdir -p docs/{design,implementation,competitive}

# Move design docs
# (Already in docs/: QUANTIZATION_DESIGN.md, PERSISTENCE_IMPLEMENTATION.md)

# Move competitive analysis
mv COMPETITIVE_GAPS_ANALYSIS.md docs/competitive/
```

### 4. Organize Benchmarks
```bash
# Create benchmarks directory
mkdir -p benchmarks/{indexes,vector_store,wal,snapshot}

# Move benchmark scripts from rust_hnsw
mv rust_hnsw/benchmark.py benchmarks/indexes/
mv rust_hnsw/benchmark_brute_force.py benchmarks/indexes/
mv rust_hnsw/benchmark_kd_tree.py benchmarks/indexes/
mv rust_hnsw/benchmark_lsh.py benchmarks/indexes/

# Move other benchmark scripts
mv rust_vector_store/benchmark_vector_store.py benchmarks/vector_store/ 2>/dev/null
mv rust_wal/benchmark_wal.py benchmarks/wal/ 2>/dev/null
```

---

## 📁 Rust Code Consolidation (Option 1 - RECOMMENDED)

### Current Structure (4 separate directories):
```
rust_hnsw/
rust_vector_store/
rust_wal/
rust_snapshot/
```

### Proposed Structure:
```
rust/
├── indexes/          # From rust_hnsw
│   ├── Cargo.toml
│   ├── src/
│   │   ├── brute_force.rs
│   │   ├── hnsw.rs
│   │   ├── kd_tree.rs
│   │   └── lsh.rs
│   └── README.md
├── vector_store/     # From rust_vector_store
│   ├── Cargo.toml
│   └── src/
├── wal/              # From rust_wal
│   ├── Cargo.toml
│   └── src/
└── snapshot/         # From rust_snapshot
    ├── Cargo.toml
    └── src/
```

### Migration Commands:
```bash
# Create new rust directory structure
mkdir -p rust/{indexes,vector_store,wal,snapshot}

# Move rust_hnsw → rust/indexes
mv rust_hnsw/* rust/indexes/
rmdir rust_hnsw

# Move rust_vector_store → rust/vector_store
mv rust_vector_store/* rust/vector_store/
rmdir rust_vector_store

# Move rust_wal → rust/wal
mv rust_wal/* rust/wal/
rmdir rust_wal

# Move rust_snapshot → rust/snapshot
mv rust_snapshot/* rust/snapshot/
rmdir rust_snapshot
```

### Update Import Paths:
Files that need updating:
- `infrastructure/indexes/rust_brute_force_wrapper.py`
- `infrastructure/indexes/rust_hnsw_wrapper.py`
- `infrastructure/indexes/rust_kd_tree_wrapper.py`
- `infrastructure/indexes/rust_lsh_wrapper.py`
- `core/rust_vector_store_wrapper.py`

Change imports from:
```python
import rust_hnsw
```
To:
```python
import rust.indexes as rust_indexes
```

---

## 🎯 Final Directory Structure

After cleanup:
```
arrwDB/
├── app/                      # Python application code
│   ├── api/
│   ├── auth/
│   ├── models/
│   ├── services/
│   └── utils/
├── core/                     # Core abstractions
├── infrastructure/           # Infrastructure implementations
│   ├── indexes/
│   ├── persistence/
│   └── repositories/
├── rust/                     # Rust performance modules (NEW)
│   ├── indexes/
│   ├── vector_store/
│   ├── wal/
│   └── snapshot/
├── tests/                    # All tests
│   ├── unit/
│   └── integration/
├── benchmarks/               # Performance benchmarks (NEW)
│   ├── indexes/
│   ├── vector_store/
│   ├── wal/
│   └── snapshot/
├── docs/                     # Documentation
│   ├── design/
│   ├── implementation/
│   └── competitive/
└── temporal/                 # Temporal workflows
```

---

## 🗑️ Files to Delete (After Verification)

None - all files are being reorganized rather than deleted.

---

## 📝 .gitignore Additions

Add to `.gitignore`:
```
# Python cache
__pycache__/
*.py[cod]
*$py.class
*.so

# Rust build artifacts
rust/*/target/
rust/*/Cargo.lock  # Only for libraries, keep for binaries

# IDE
.vscode/
.idea/
*.swp

# Test artifacts
.pytest_cache/
htmlcov/
.coverage
tests/test_results/

# Environment
.env
venv/
env/
```

---

## ⚠️ Important Notes

1. **Rust modules ARE actively used** - the wrappers in `infrastructure/indexes/` import them
2. **Don't delete rust_* directories** - they contain working code that's imported
3. **Test after moving** - ensure all imports work after reorganization
4. **Update documentation** - reflect new structure in README

---

## Execution Order

1. ✅ Clean cache files (safe, no dependencies)
2. ✅ Move test files (safe, pytest will find them)
3. ✅ Organize documentation (safe, just moving files)
4. ✅ Move benchmarks (safe, standalone scripts)
5. ⚠️ Consolidate Rust code (requires import updates)

---

## Verification Checklist

After cleanup:
- [ ] Run `python3 -m pytest` - all tests pass
- [ ] Run `python3 run_api.py` - API starts without errors
- [ ] Import test: `python3 -c "from infrastructure.indexes.rust_hnsw_wrapper import RustHNSWIndex"`
- [ ] Check git status - no unintended changes
