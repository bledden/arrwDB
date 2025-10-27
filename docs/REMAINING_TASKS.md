# Remaining Tasks - arrwDB

## Overview

This document lists all remaining test coverage tasks and feature development work for arrwDB.

---

## Part 1: Remaining Test Coverage Tasks

### Phase 2: Streaming Tests (Not Started)

**Priority**: MEDIUM
**Estimated Time**: 3-4 hours
**Coverage Gain**: +6-8%

#### Test File 1: `tests/unit/test_streaming.py`

**Target Module**: `app/api/streaming.py` (currently 0% coverage)

**Test Scenarios Needed**:
1. `test_ndjson_document_ingestion_success` - Valid NDJSON upload
2. `test_ndjson_document_ingestion_partial_failure` - Some docs fail
3. `test_ndjson_empty_lines_handling` - Skip empty lines
4. `test_ndjson_malformed_json` - Handle JSON parsing errors
5. `test_ndjson_missing_required_fields` - Handle validation errors
6. `test_search_results_streaming_success` - Valid search query
7. `test_search_results_streaming_with_filters` - Search with distance threshold
8. `test_search_results_streaming_empty_results` - No matches found
9. `test_sse_event_streaming_subscribe_all` - Subscribe to all events
10. `test_sse_event_streaming_filter_by_library` - Library-specific events
11. `test_sse_client_disconnect_cleanup` - Proper cleanup on disconnect
12. `test_sse_event_format_validation` - Correct SSE message format

**Dependencies**:
- Library service (mocked)
- Event bus integration
- SSE starlette library

#### Test File 2: `tests/unit/test_streaming_fixed.py`

**Target Module**: `app/api/streaming_fixed.py` (currently 0% coverage)

**Test Scenarios Needed**:
1. `test_fixed_ndjson_ingestion_success` - Simplified ingestion
2. `test_fixed_ndjson_response_format` - JSON response validation
3. `test_fixed_search_response_format` - Search result format

**Status**: ❌ Not started

---

### Phase 3: Auth & Authorization Tests (Partial)

**Priority**: HIGH
**Estimated Time**: 3-4 hours
**Coverage Gain**: +5-7%

#### Test File 1: `tests/unit/test_auth_middleware.py` ✅ (NEEDS FIXING)

**Target Module**: `app/auth/middleware.py` (currently 22% coverage)

**Status**: ⚠️ Created but 17/24 tests failing due to Tenant dataclass structure

**Tasks**:
1. Fix Tenant dataclass instantiation in tests
2. Ensure all 24 tests pass
3. Target 90%+ coverage

#### Test File 2: `tests/unit/test_api_key_management.py`

**Target Module**: `app/auth/api_keys.py` (currently 32% coverage)

**Test Scenarios Needed**:
1. `test_create_api_key_for_tenant` - Generate new API key
2. `test_api_key_format_validation` - Verify arrw_ prefix
3. `test_validate_api_key_success` - Valid key returns tenant
4. `test_validate_api_key_invalid` - Invalid key returns None
5. `test_validate_api_key_expired` - Expired key returns None
6. `test_revoke_api_key` - Mark key as inactive
7. `test_rotate_api_key` - Generate new key, revoke old
8. `test_list_tenant_api_keys` - Get all keys for tenant
9. `test_get_api_key_details` - Get key metadata
10. `test_api_key_expiration_handling` - Expiration logic
11. `test_api_key_rate_limiting` - Rate limit integration
12. `test_api_key_persistence` - Save/load from storage

**Status**: ❌ Not started

---

### Phase 4: Library Service & Search Tests (Partial)

**Priority**: MEDIUM
**Estimated Time**: 4-5 hours
**Coverage Gain**: +7-9%

#### Test File 1: `tests/unit/test_hybrid_search.py`

**Target Module**: `app/services/hybrid_search.py` (currently 0% coverage)

**Test Scenarios Needed**:
1. `test_hybrid_search_vector_only` - Pure vector search
2. `test_hybrid_search_keyword_only` - Pure keyword search
3. `test_hybrid_search_combined` - Vector + keyword with fusion
4. `test_hybrid_search_with_metadata_filters` - Apply filters
5. `test_hybrid_search_reranking` - Result reranking logic
6. `test_hybrid_search_score_normalization` - Normalize scores
7. `test_hybrid_search_empty_results` - No matches
8. `test_hybrid_search_performance` - Benchmark throughput

**Status**: ❌ Not started

**Note**: Library service already has tests (`test_library_service_core.py`, `test_library_service_advanced.py`) with some coverage. May need additional edge case tests.

---

## Part 2: Feature Analysis & Integration Tasks

### Requested Features Analysis

Based on previous conversations and roadmap, here are features that may have been requested or are beneficial:

#### Feature 1: Advanced Metadata Filtering ✅ COMPLETE

**Status**: ✅ **Already implemented**
- Metadata filtering exists in search APIs
- JSON path queries supported
- Documentation exists in `docs/features/METADATA_FILTERING.md`

**No action needed**.

---

#### Feature 2: Batch Operations ✅ COMPLETE

**Status**: ✅ **Already implemented**
- Batch document import via Job Queue
- Batch delete operations
- Batch export functionality
- Job handlers exist in `app/jobs/handlers.py`

**No action needed**.

---

#### Feature 3: Real-time Event Streaming ✅ COMPLETE

**Status**: ✅ **Already implemented**
- SSE event streaming via `/events/stream`
- WebSocket support for real-time updates
- Event Bus for CDC (Change Data Capture)

**No action needed**.

---

#### Feature 4: Multi-tenancy & API Key Management ✅ COMPLETE

**Status**: ✅ **Already implemented**
- Multi-tenancy support via API keys
- Tenant isolation
- API key validation middleware

**Action needed**: Complete test coverage (see Phase 3 above)

---

#### Feature 5: Hybrid Search (Vector + Keyword) ✅ COMPLETE

**Status**: ✅ **Already implemented**
- Hybrid search in `app/services/hybrid_search.py`
- Vector + BM25 keyword fusion
- Reranking support

**Action needed**: Complete test coverage (see Phase 4 above)

---

#### Feature 6: Performance Monitoring & Metrics ⚠️ PARTIAL

**Status**: ⚠️ **Partially implemented**

**What exists**:
- Basic metrics endpoint at `/metrics`
- Event bus statistics
- Job queue statistics
- WebSocket connection stats

**What's missing**:
- Prometheus metrics export
- Grafana dashboard templates
- Query performance tracking
- Slow query logging
- Resource utilization metrics

**Tasks to complete**:
1. Add Prometheus metrics exporter
2. Create Grafana dashboard JSON
3. Add query performance tracking decorator
4. Implement slow query logging
5. Add memory/CPU utilization tracking
6. Document metrics setup in deployment guide

**Priority**: MEDIUM
**Estimated Time**: 4-6 hours

---

#### Feature 7: Deployment & Infrastructure ⚠️ PARTIAL

**Status**: ⚠️ **Partially documented**

**What exists**:
- Docker deployment guide
- Security hardening guide
- HTTPS/TLS configuration guide
- Performance benchmarks

**What's missing**:
- Docker Compose file for full stack
- Kubernetes deployment manifests
- Helm chart
- CI/CD pipeline configuration
- Automated backup scripts
- Health check endpoints

**Tasks to complete**:
1. Create `docker-compose.yml` for local development
2. Create `docker-compose.prod.yml` for production
3. Create Kubernetes manifests (`k8s/`)
4. Create Helm chart (`helm/arrwdb/`)
5. Add GitHub Actions CI/CD workflow
6. Add health check endpoint `/health`
7. Add readiness probe endpoint `/ready`
8. Create backup automation script
9. Document deployment process

**Priority**: HIGH (for production readiness)
**Estimated Time**: 8-10 hours

---

#### Feature 8: Rust HNSW Integration ⚠️ IN PROGRESS

**Status**: ⚠️ **In development**

**What exists**:
- Rust HNSW wrapper stubs (`core/rust_vector_store_wrapper.py`)
- Rust HNSW implementation (`rust_hnsw/`)
- Python bindings structure

**What's missing**:
- Complete Rust implementation compilation
- Python binding finalization
- Performance testing vs pure Python
- Integration tests
- Fallback mechanism if Rust unavailable

**Tasks to complete**:
1. Complete Rust HNSW implementation
2. Build and test Rust library
3. Finalize Python bindings with PyO3
4. Add Rust compilation to CI/CD
5. Create performance comparison benchmarks
6. Add fallback to Python HNSW if Rust unavailable
7. Document Rust setup requirements

**Priority**: LOW (Python HNSW works well)
**Estimated Time**: 10-15 hours

---

#### Feature 9: Advanced Index Types ✅ COMPLETE

**Status**: ✅ **Already implemented**

Multiple index types supported:
- HNSW (Hierarchical Navigable Small World)
- IVF (Inverted File Index)
- KD-Tree
- LSH (Locality-Sensitive Hashing)
- Brute Force

**No action needed**.

---

#### Feature 10: Data Persistence & WAL ✅ COMPLETE

**Status**: ✅ **Already implemented**
- Write-Ahead Log (WAL) for durability
- Snapshot functionality
- Recovery mechanisms
- Comprehensive tests exist

**No action needed**.

---

## Part 3: Prioritized Task List

### Critical Path (Must Complete)

1. **Fix Auth Middleware Tests** (1-2 hours)
   - Fix Tenant dataclass issues
   - Get all 24 tests passing
   - Achieve 90%+ coverage on `app/auth/middleware.py`

2. **Create API Key Management Tests** (2-3 hours)
   - Write 12 test scenarios
   - Achieve 90%+ coverage on `app/auth/api_keys.py`

3. **Add Deployment Infrastructure** (8-10 hours)
   - Docker Compose files
   - Health check endpoints
   - CI/CD workflow
   - Deployment documentation

### High Priority (Should Complete)

4. **Create Streaming Tests** (3-4 hours)
   - NDJSON ingestion tests
   - Search streaming tests
   - SSE event streaming tests

5. **Create Hybrid Search Tests** (3-4 hours)
   - Hybrid search test scenarios
   - Performance benchmarks

6. **Add Monitoring & Metrics** (4-6 hours)
   - Prometheus metrics
   - Grafana dashboards
   - Query performance tracking

### Medium Priority (Nice to Have)

7. **Complete Rust HNSW Integration** (10-15 hours)
   - Rust implementation
   - Python bindings
   - Performance testing

### Low Priority (Future Enhancements)

8. **Additional Performance Optimizations**
9. **Advanced Analytics Features**
10. **Machine Learning Pipeline Integration**

---

## Part 4: Task Breakdown by Type

### Testing Tasks (10-15 hours total)

| Task | File | Status | Time | Priority |
|------|------|--------|------|----------|
| Fix auth middleware tests | `test_auth_middleware.py` | ⚠️ Fixing | 1-2h | CRITICAL |
| API key management tests | `test_api_key_management.py` | ❌ Not started | 2-3h | CRITICAL |
| Streaming tests | `test_streaming.py` | ❌ Not started | 3-4h | HIGH |
| Hybrid search tests | `test_hybrid_search.py` | ❌ Not started | 3-4h | HIGH |

### Infrastructure Tasks (8-10 hours total)

| Task | File | Status | Time | Priority |
|------|------|--------|------|----------|
| Docker Compose | `docker-compose.yml` | ❌ Not started | 2h | CRITICAL |
| Health endpoints | `app/api/health.py` | ❌ Not started | 1h | CRITICAL |
| CI/CD workflow | `.github/workflows/ci.yml` | ❌ Not started | 2-3h | CRITICAL |
| K8s manifests | `k8s/*.yaml` | ❌ Not started | 3-4h | HIGH |

### Monitoring Tasks (4-6 hours total)

| Task | File | Status | Time | Priority |
|------|------|--------|------|----------|
| Prometheus metrics | `app/api/prometheus.py` | ❌ Not started | 2-3h | MEDIUM |
| Grafana dashboards | `monitoring/grafana/*.json` | ❌ Not started | 2-3h | MEDIUM |

### Feature Tasks (10-15 hours total)

| Task | Directory | Status | Time | Priority |
|------|-----------|--------|------|----------|
| Rust HNSW complete | `rust_hnsw/` | ⚠️ In progress | 10-15h | LOW |

---

## Part 5: Execution Plan

### Week 1: Critical Path
- Day 1-2: Fix auth tests + API key tests (3-5 hours)
- Day 3-4: Deployment infrastructure (8-10 hours)
- Day 5: Testing and validation (2-3 hours)

### Week 2: High Priority
- Day 1-2: Streaming tests (3-4 hours)
- Day 3-4: Hybrid search tests (3-4 hours)
- Day 5: Monitoring & metrics (4-6 hours)

### Future: Medium/Low Priority
- Rust HNSW completion (when needed)
- Additional optimizations
- Advanced features

---

## Summary

### What's Complete ✅
- Async infrastructure (Event Bus, Job Queue, WebSocket): 95-100% coverage
- Performance testing suite: All targets exceeded by 2-142x
- Core features: Multi-tenancy, batch ops, real-time events, hybrid search
- Comprehensive documentation

### What's In Progress ⚠️
- Auth middleware tests: Created, needs fixing
- Rust HNSW: Implementation in progress

### What's Missing ❌
- 3 test files (streaming, API keys, hybrid search)
- Deployment infrastructure (Docker Compose, CI/CD, K8s)
- Monitoring setup (Prometheus, Grafana)

### Total Estimated Time
- **Critical tasks**: 12-15 hours
- **High priority**: 10-14 hours
- **Medium priority**: 4-6 hours
- **Low priority**: 10-15 hours
- **Total**: 36-50 hours

### Current Test Coverage
- **Overall**: 23% (goal: 80%)
- **Async infrastructure**: 95-100% ✅
- **Auth**: 22-91% (needs completion)
- **Services**: 10-30% (needs tests)

---

## Next Steps

**Immediate actions**:
1. Fix auth middleware tests (1-2 hours)
2. Create API key management tests (2-3 hours)
3. Add deployment infrastructure (8-10 hours)

**Then proceed with**:
4. Streaming tests
5. Hybrid search tests
6. Monitoring setup

This provides a clear path to production-ready arrwDB with comprehensive test coverage and deployment infrastructure.
