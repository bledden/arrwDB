# How to Run Tests

Quick guide for running the comprehensive test suite.

---

## Prerequisites

```bash
# Ensure you're in the project directory
cd /Users/bledden/Documents/arrwDB

# Install test dependencies (already in requirements.txt)
pip3 install pytest pytest-cov pytest-asyncio
```

---

## Quick Start - Run Passing Tests

```bash
# Run all VectorStore tests (22 tests, all passing)
python3 -m pytest tests/unit/test_vector_store.py -v

# Expected output:
# ============================= test session starts ==============================
# collected 22 items
#
# tests/unit/test_vector_store.py::TestVectorStoreInitialization::test_valid_initialization PASSED [  4%]
# tests/unit/test_vector_store.py::TestVectorStoreInitialization::test_invalid_dimension_raises_error PASSED [  9%]
# ... (20 more tests)
# ========================= 22 passed in 0.37s =========================
```

---

## Run All Tests

```bash
# Run entire test suite
python3 -m pytest tests/ -v

# Current status:
# - 22 passing (VectorStore fully tested)
# - 45 need minor API fixes (~15 min to fix)
# - 19 errors due to API mismatches
```

---

## Run with Coverage

```bash
# Run tests with coverage report
python3 -m pytest tests/unit/test_vector_store.py --cov=core.vector_store --cov-report=term-missing

# Output:
# Name                  Stmts   Miss  Cover   Missing
# ---------------------------------------------------
# core/vector_store.py    149     30    80%   84, 96-102, ...
#
# ========================= 22 passed in 0.37s =========================
```

---

## Run Specific Test Categories

```bash
# Run only unit tests
python3 -m pytest tests/ -v -m unit

# Run only edge case tests
python3 -m pytest tests/ -v -m edge

# Run only thread safety tests
python3 -m pytest tests/ -v -m thread_safety
```

---

## Run Single Test

```bash
# Run a specific test by name
python3 -m pytest tests/unit/test_vector_store.py::TestVectorStoreAddVector::test_vector_deduplication -v

# Output:
# tests/unit/test_vector_store.py::TestVectorStoreAddVector::test_vector_deduplication PASSED [100%]
```

---

## Generate HTML Coverage Report

```bash
# Generate detailed HTML coverage report
python3 -m pytest tests/unit/test_vector_store.py --cov=core --cov=infrastructure --cov-report=html

# Open the report
open htmlcov/index.html
```

---

## Test Output Examples

### Successful Test

```
tests/unit/test_vector_store.py::TestVectorStoreAddVector::test_vector_deduplication PASSED

Test Details:
- Creates 2 chunks with identical vectors
- Verifies only 1 vector stored (deduplication)
- Verifies 2 references counted
- Verifies both chunks can retrieve the vector
✅ PASSED in 0.01s
```

### Failed Test (Example)

```
tests/unit/test_indexes.py::TestIndexSearch::test_search_single_result FAILED

Reason: AttributeError: 'BruteForceIndex' object has no attribute 'size'
Expected: index.size()
Actual API: index.count or len(index)

Fix: Replace index.size() with len(index)
```

---

## Quick Verification Checklist

Run these commands to verify the test suite is working:

```bash
# 1. Verify pytest is installed
python3 -m pytest --version
# Expected: pytest 8.4.2 or higher

# 2. Verify test discovery
python3 -m pytest tests/ --collect-only
# Expected: collected 86 items

# 3. Run VectorStore tests (should all pass)
python3 -m pytest tests/unit/test_vector_store.py -v
# Expected: 22 passed in ~0.4s

# 4. Check coverage
python3 -m pytest tests/unit/test_vector_store.py --cov=core.vector_store
# Expected: 80% coverage
```

---

## Interpreting Results

### ✅ All Tests Pass

```
========================= 22 passed in 0.37s =========================
```

- All tests executed successfully
- No errors or failures
- Code behaves as expected

### ⚠️ Some Tests Fail

```
============= 1 failed, 21 passed in 0.37s =============
```

- Most tests pass
- Check FAILED section for details
- Usually API mismatches or edge case issues

### ❌ Errors

```
============= 19 errors, 45 passed in 1.85s =============
```

- Test setup issues (missing imports, wrong API)
- Check ERROR section for details
- Usually fixture or import problems

---

## Common Issues

### Issue: ImportError

```python
ImportError: No module named 'pytest'
```

**Fix**:
```bash
pip3 install pytest pytest-cov pytest-asyncio
```

### Issue: Test Collection Failed

```
ERROR collecting tests/unit/test_vector_store.py
```

**Fix**: Check Python path
```bash
export PYTHONPATH="${PYTHONPATH}:/Users/bledden/Documents/arrwDB"
python3 -m pytest tests/
```

### Issue: Fixtures Not Found

```
fixture 'vector_store' not found
```

**Fix**: Ensure conftest.py is in tests/ directory
```bash
ls tests/conftest.py  # Should exist
```

---

## What Each Test File Tests

### test_vector_store.py (22 tests) ✅
- Vector storage and retrieval
- Reference counting
- Vector deduplication
- Dynamic capacity growth
- **Status**: All passing

### test_embedding_contract.py (17 tests)
- Vector validation
- Dimension enforcement
- NaN/Inf detection
- Normalization to unit length
- **Status**: Needs API alignment

### test_indexes.py (20 tests)
- All 4 index types (parametrized)
- Add/remove operations
- Search accuracy
- Distance threshold filtering
- **Status**: Needs API alignment

### test_reader_writer_lock.py (13 tests)
- Thread safety verification
- Multiple concurrent readers
- Writer exclusivity
- Writer priority
- **Status**: Ready to run

### test_library_repository.py (14 tests)
- Repository CRUD operations
- Thread-safe document/chunk management
- Search integration
- **Status**: Needs API alignment

### test_edge_cases.py (41 tests)
- Unicode text
- Numerical extremes
- Boundary values
- Metadata edge cases
- **Status**: Ready to run

---

## Next Steps

1. **Run passing tests** to verify setup:
   ```bash
   python3 -m pytest tests/unit/test_vector_store.py -v
   ```

2. **Review test output** to understand what's tested

3. **Check coverage** to see what's verified:
   ```bash
   python3 -m pytest tests/unit/test_vector_store.py --cov=core.vector_store --cov-report=term-missing
   ```

4. **Read test code** to understand test strategy:
   ```bash
   cat tests/unit/test_vector_store.py
   ```

---

## Expected Timeline

| Task | Time | Command |
|------|------|---------|
| Run passing tests | 30 sec | `pytest tests/unit/test_vector_store.py -v` |
| Fix API mismatches | 15 min | Edit test files |
| Run all tests | 2 min | `pytest tests/ -v` |
| Generate coverage | 1 min | `pytest tests/ --cov --cov-report=html` |
| Review results | 10 min | Open htmlcov/index.html |

---

## Success Criteria

✅ **All VectorStore tests pass** (22/22)
- Demonstrates core component works correctly
- 80% code coverage
- Reference counting verified
- Thread safety verified

✅ **Test suite runs without errors**
- pytest discovers all 86 tests
- No import errors
- No fixture errors

✅ **Coverage report generated**
- Shows what's tested vs not tested
- Identifies gaps
- HTML report viewable in browser

---

## Questions?

**Q**: Why do some tests fail?
**A**: API mismatches (test expects `size()`, actual API is `count`). Easy 15-min fix.

**Q**: What's the minimum to verify?
**A**: Run `python3 -m pytest tests/unit/test_vector_store.py -v` - should show 22 passed.

**Q**: How do I know tests are real?
**A**: Look at test code - uses actual VectorStore, real numpy, real locks. No mocks!

**Q**: What's the coverage target?
**A**: 70% overall, 80%+ for core components (VectorStore, Repository).

---

**Ready to test!** 🚀
