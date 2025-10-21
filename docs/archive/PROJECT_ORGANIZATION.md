# Project Organization

**Last Updated**: 2025-10-20

## Overview

This document describes the final organization of the Vector Database project after cleanup and restructuring.

## Root Directory

The root directory contains only essential files:

```
/
├── README.md                 # Main entry point & documentation hub
├── run_api.py               # API server entry point
├── .env.example             # Environment variable template
├── .gitignore              # Git exclusions
├── requirements.txt         # Python dependencies
├── pyproject.toml          # Project metadata
├── pytest.ini              # Test configuration
├── docker-compose.yml      # Docker orchestration
├── Dockerfile              # Docker image definition
└── [source directories]     # app/, core/, infrastructure/, etc.
```

**Only 1 markdown file in root**: `README.md`

## Source Code Structure

```
app/                        # REST API Layer
├── api/                   # FastAPI endpoints & routes
│   ├── main.py           # Main FastAPI application
│   ├── dependencies.py   # Dependency injection
│   └── models.py         # API request/response models
├── models/               # Domain models
│   └── base.py          # Pydantic models (Library, Document, Chunk)
└── services/            # Business logic layer
    ├── library_service.py      # Library management
    └── embedding_service.py    # Embedding generation

core/                      # Core Domain Logic
├── vector_store.py       # Vector storage & deduplication
└── embedding_contract.py # Validation & normalization

infrastructure/           # Technical Infrastructure
├── indexes/             # Search algorithms
│   ├── brute_force.py   # O(n) exact search
│   ├── kd_tree.py       # O(log n) tree-based
│   ├── lsh.py          # Locality-sensitive hashing
│   └── hnsw.py         # Hierarchical navigable small world
├── repositories/        # Data access layer
│   └── library_repository.py
├── concurrency/         # Thread safety
│   └── rw_lock.py      # Reader-writer lock
└── persistence/         # Durability
    ├── wal.py          # Write-ahead log
    └── snapshot.py     # Snapshot management

temporal/                # Temporal Workflows
├── workflows/          # Workflow definitions
├── activities/         # Activity implementations
└── worker.py          # Workflow worker

sdk/                    # Python Client SDK
└── client.py          # High-level client

tests/                  # Test Suite (131 tests)
├── unit/              # Unit tests (86 tests)
│   ├── test_vector_store.py
│   ├── test_embedding_contract.py
│   ├── test_indexes.py
│   ├── test_library_repository.py
│   └── test_reader_writer_lock.py
├── integration/       # Integration tests (23 tests)
│   └── test_api.py
├── test_edge_cases.py # Edge case tests (22 tests)
└── conftest.py       # Test fixtures

scripts/               # Utility Scripts
└── test_basic_functionality.py
```

## Documentation Structure

```
docs/
├── README.md                       # Documentation index
│
├── guides/                        # User Guides
│   ├── INSTALLATION.md           # Setup instructions
│   ├── QUICKSTART.md            # 5-minute quick start
│   └── INDEX.md                 # API endpoint reference
│
├── testing/                       # Test Documentation
│   ├── FINAL_TEST_REPORT.md     # Current test status (131/131 passing)
│   ├── TEST_STATUS_FINAL.md     # Detailed coverage (74%)
│   ├── TEST_RESULTS_FINAL.md    # Test result details
│   ├── TEST_RESULTS_UPDATED.md  # Updated results
│   ├── TEST_SUMMARY.md          # Test suite overview
│   ├── TESTING_COMPLETE.md      # Testing milestones
│   └── RUN_TESTS.md            # How to run tests
│
├── planning/                      # Historical Planning (Archived)
│   ├── PLAN1.md                  # Initial planning
│   ├── PLAN1_REVISED.md         # Revision 1
│   ├── PLAN1_PERFECT.md         # Revision 2
│   ├── PLAN1_COMPLETE.md        # Revision 3
│   ├── FINAL_PLAN1.md           # Final plan
│   ├── REQUIREMENTS_VALIDATION.md
│   ├── IMPLEMENTATION_COMPLETE.md
│   └── STATUS.md
│
├── CODE_QUALITY_ASSESSMENT.md     # Code quality analysis
├── HIRING_REVIEW.md               # Code review checklist
├── LEADER_FOLLOWER_DESIGN.md      # Architecture design
├── REAL_VS_MOCKED.md             # Testing philosophy
├── REQUIREMENTS_VERIFICATION.md   # Requirements compliance
├── CLEANUP_SUMMARY.md            # This cleanup process
└── PROJECT_ORGANIZATION.md        # This document
```

## Hidden/Generated Files

These are in `.gitignore` and should not be committed:

```
.env                   # Environment variables (API keys)
.coverage              # Coverage data
.pytest_cache/         # Pytest cache
htmlcov/              # HTML coverage reports
__pycache__/          # Python bytecode
*.pyc                 # Compiled Python files
data/                 # Runtime data directory
```

## Entry Points

### Starting the API Server
```bash
python run_api.py
```

### Running Tests
```bash
pytest tests/                      # All tests
pytest tests/unit/                 # Unit tests only
pytest tests/integration/          # Integration tests
```

### Quick Smoke Test
```bash
python scripts/test_basic_functionality.py
```

## Navigation Guide

**For new developers**:
1. Start with `README.md`
2. Read `docs/guides/INSTALLATION.md`
3. Try `docs/guides/QUICKSTART.md`
4. Check `docs/testing/FINAL_TEST_REPORT.md` for current status

**For code review**:
1. Check `docs/CODE_QUALITY_ASSESSMENT.md`
2. Review `docs/HIRING_REVIEW.md`
3. See `docs/testing/TEST_STATUS_FINAL.md`

**For architecture understanding**:
1. Read `docs/LEADER_FOLLOWER_DESIGN.md`
2. Check `docs/REQUIREMENTS_VERIFICATION.md`
3. Review `README.md` architecture section

## Key Files

| File | Purpose | Location |
|------|---------|----------|
| Main README | Project overview & entry point | `/README.md` |
| API Server | Start the REST API | `/run_api.py` |
| Test Runner | Run test suite | `/pytest.ini` |
| Environment | API key configuration | `/.env.example` |
| Dependencies | Python packages | `/requirements.txt` |
| Test Report | Current test status | `/docs/testing/FINAL_TEST_REPORT.md` |
| Installation | Setup guide | `/docs/guides/INSTALLATION.md` |
| Quick Start | Getting started | `/docs/guides/QUICKSTART.md` |

## Benefits of This Organization

1. **Clean Root Directory**
   - Only essential files visible
   - Clear entry points
   - Professional appearance

2. **Logical Documentation**
   - Grouped by purpose (guides, testing, planning)
   - Easy to find information
   - Historical context preserved

3. **Developer Friendly**
   - Quick navigation
   - Clear structure
   - Good discoverability

4. **Maintainable**
   - Separation of concerns
   - Easy to update
   - Scalable structure

## Maintenance

When adding new documentation:
- User guides → `docs/guides/`
- Test documentation → `docs/testing/`
- Technical specs → `docs/`
- Scripts → `scripts/`

Keep the root directory minimal - only essential entry points.
