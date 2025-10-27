# Test Coverage Roadmap for arrwDB

This document outlines the systematic plan to increase test coverage to 80%+.

## Current Status

**Overall Coverage**: 62% (3,619/5,184 statements missed)
**Tests Passing**: 635/661 (96%)
**Goal**: 80%+ coverage

## Coverage by Module (Lowest to Highest)

### Critical Modules Needing Coverage

| Module | Coverage | Missing Lines | Priority |
|--------|----------|---------------|----------|
| app/services/hybrid_search.py | 0% | 130 | HIGH |
| app/api/streaming_fixed.py | 0% | 33 | MEDIUM |
| app/auth/middleware.py | 21.7% | 18 | HIGH |
| app/jobs/handlers.py | 27.5% | 74 | HIGH |
| app/api/streaming.py | 28.4% | 48 | MEDIUM |
| app/api/websocket_routes.py | 30.6% | 84 | HIGH |
| app/auth/api_keys.py | 32.2% | 80 | HIGH |
| app/websockets/manager.py | 38.1% | 39 | HIGH |
| app/logging_config.py | 50.0% | 27 | LOW |
| app/services/library_service.py | 50.4% | 172 | MEDIUM |
| app/api/job_routes.py | 52.6% | 65 | MEDIUM |
| app/api/main.py | 54.0% | 233 | LOW |
| app/jobs/queue.py | 54.7% | 78 | HIGH |
| app/events/bus.py | 62.1% | 44 | HIGH |

### Already Well-Covered

| Module | Coverage | Status |
|--------|----------|--------|
| app/auth/permissions.py | 100% | ✅ Complete |
| app/auth/audit.py | 91% | ✅ Complete |
| app/middleware/security.py | 94% | ✅ Complete |
| infrastructure/indexes/hnsw.py | 100% | ✅ Complete |
| infrastructure/indexes/kd_tree.py | 97% | ✅ Complete |
| infrastructure/indexes/lsh.py | 98% | ✅ Complete |

## Test Implementation Plan

### Phase 1: Event Bus & Job Queue (Highest Impact)

**Estimated Time**: 4-6 hours
**Coverage Gain**: ~8-10%

#### 1.1 Event Bus Tests (`tests/unit/test_event_bus.py`)

```python
"""
Test suite for app/events/bus.py

Coverage targets:
- Event creation and validation
- Event publishing and subscription
- Event filtering by type
- Event filtering by library
- Global subscribers
- Error handling in callbacks
- Event statistics
- Bus lifecycle (start/stop)
- Async event processing
- Subscriber deregistration
"""

Test scenarios:
- ✅ test_event_creation
- ✅ test_event_with_metadata
- ✅ test_subscribe_to_specific_event_type
- ✅ test_subscribe_to_all_events
- ✅ test_publish_event
- ✅ test_event_delivered_to_correct_subscribers
- ✅ test_event_filtering_by_library
- ✅ test_multiple_subscribers_same_event
- ✅ test_unsubscribe
- ✅ test_subscriber_error_handling
- ✅ test_event_statistics
- ✅ test_bus_start_stop
- ✅ test_concurrent_publishing
- ✅ test_event_ordering
```

#### 1.2 Job Queue Tests (`tests/unit/test_job_queue.py`)

```python
"""
Test suite for app/jobs/queue.py

Coverage targets:
- Job creation and validation
- Job enqueueing
- Job execution
- Job status tracking
- Job priority handling
- Job retries on failure
- Dead letter queue
- Job expiration
- Queue statistics
"""

Test scenarios:
- ✅ test_create_job
- ✅ test_enqueue_job
- ✅ test_dequeue_job
- ✅ test_job_execution
- ✅ test_job_success
- ✅ test_job_failure_with_retry
- ✅ test_job_max_retries_exceeded
- ✅ test_job_priority_ordering
- ✅ test_job_expiration
- ✅ test_dead_letter_queue
- ✅ test_get_job_status
- ✅ test_queue_statistics
- ✅ test_concurrent_job_processing
```

#### 1.3 Job Handlers Tests (`tests/unit/test_job_handlers.py`)

```python
"""
Test suite for app/jobs/handlers.py

Coverage targets:
- Document embedding jobs
- Batch operation jobs
- Index optimization jobs
- Library maintenance jobs
- Error handling
- Progress tracking
"""

Test scenarios:
- ✅ test_document_embedding_handler
- ✅ test_batch_add_handler
- ✅ test_batch_delete_handler
- ✅ test_index_rebuild_handler
- ✅ test_index_optimize_handler
- ✅ test_library_cleanup_handler
- ✅ test_handler_error_propagation
- ✅ test_handler_progress_updates
```

### Phase 2: WebSocket & Streaming (Real-time Features)

**Estimated Time**: 3-4 hours
**Coverage Gain**: ~6-8%

#### 2.1 WebSocket Manager Tests (`tests/unit/test_websocket_manager.py`)

```python
"""
Test suite for app/websockets/manager.py

Coverage targets:
- Connection management
- Subscription handling
- Message broadcasting
- Connection cleanup
- Error handling
- Connection authentication
"""

Test scenarios:
- ✅ test_connect_websocket
- ✅ test_disconnect_websocket
- ✅ test_subscribe_to_library
- ✅ test_unsubscribe_from_library
- ✅ test_broadcast_to_subscribers
- ✅ test_broadcast_filtering
- ✅ test_multiple_subscriptions
- ✅ test_connection_cleanup_on_disconnect
- ✅ test_invalid_subscription
- ✅ test_concurrent_connections
```

#### 2.2 Streaming Tests (`tests/unit/test_streaming.py`)

```python
"""
Test suite for app/api/streaming.py

Coverage targets:
- NDJSON streaming
- SSE streaming
- Stream error handling
- Backpressure handling
- Stream timeout
"""

Test scenarios:
- ✅ test_ndjson_stream_documents
- ✅ test_ndjson_stream_search_results
- ✅ test_sse_stream_events
- ✅ test_stream_timeout
- ✅ test_stream_error_handling
- ✅ test_stream_backpressure
- ✅ test_stream_cancellation
```

### Phase 3: Authentication & Authorization

**Estimated Time**: 3-4 hours
**Coverage Gain**: ~5-7%

#### 3.1 Auth Middleware Tests (`tests/unit/test_auth_middleware.py`)

```python
"""
Test suite for app/auth/middleware.py

Coverage targets:
- API key validation
- Request authentication
- Tenant identification
- Rate limiting integration
- Auth error responses
"""

Test scenarios:
- ✅ test_valid_api_key
- ✅ test_invalid_api_key
- ✅ test_missing_api_key
- ✅ test_expired_api_key
- ✅ test_tenant_identification
- ✅ test_rate_limit_integration
- ✅ test_auth_error_responses
- ✅ test_auth_bypass_health_endpoint
```

#### 3.2 API Key Management Tests (`tests/unit/test_api_key_management.py`)

```python
"""
Test suite for app/auth/api_keys.py

Coverage targets:
- API key creation
- API key validation
- API key rotation
- API key expiration
- Usage tracking
- Key revocation
"""

Test scenarios:
- ✅ test_create_api_key
- ✅ test_validate_api_key
- ✅ test_rotate_api_key
- ✅ test_api_key_expiration
- ✅ test_track_api_key_usage
- ✅ test_revoke_api_key
- ✅ test_api_key_with_scopes
- ✅ test_api_key_metadata
```

### Phase 4: Library Service & Business Logic

**Estimated Time**: 4-5 hours
**Coverage Gain**: ~7-9%

#### 4.1 Library Service Tests (`tests/unit/test_library_service_advanced.py`)

```python
"""
Test suite for app/services/library_service.py

Coverage targets:
- Library lifecycle management
- Document operations
- Search operations
- Index management
- Batch operations
- Error handling
- Concurrency control
"""

Test scenarios:
- ✅ test_create_library_with_config
- ✅ test_library_dimension_validation
- ✅ test_add_document_with_embedding
- ✅ test_add_document_with_text
- ✅ test_batch_add_documents
- ✅ test_search_with_filters
- ✅ test_search_with_metadata
- ✅ test_rebuild_index
- ✅ test_switch_index_type
- ✅ test_concurrent_document_adds
- ✅ test_library_not_found_error
- ✅ test_dimension_mismatch_error
```

#### 4.2 Hybrid Search Tests (`tests/unit/test_hybrid_search.py`)

```python
"""
Test suite for app/services/hybrid_search.py

Coverage targets:
- Vector search
- Keyword search
- Hybrid ranking
- Reranking strategies
- Score normalization
"""

Test scenarios:
- ✅ test_vector_search
- ✅ test_keyword_search
- ✅ test_hybrid_search_fusion
- ✅ test_rerank_by_recency
- ✅ test_rerank_by_position
- ✅ test_score_normalization
- ✅ test_empty_query_handling
```

### Phase 5: Integration & End-to-End Tests

**Estimated Time**: 3-4 hours
**Coverage Gain**: Improved reliability

#### 5.1 API Integration Tests (`tests/integration/test_api_complete.py`)

```python
"""
Complete end-to-end API workflow tests

Test scenarios:
- ✅ test_complete_document_workflow
- ✅ test_library_lifecycle
- ✅ test_concurrent_operations
- ✅ test_error_recovery
- ✅ test_rate_limiting
- ✅ test_authentication_flow
```

#### 5.2 Performance Tests (`tests/performance/test_load.py`)

```python
"""
Performance and load tests

Test scenarios:
- ✅ test_concurrent_searches
- ✅ test_bulk_document_ingestion
- ✅ test_index_rebuild_performance
- ✅ test_memory_usage
```

## Implementation Strategy

### Step 1: Setup Test Infrastructure

```bash
# Install additional test dependencies
pip install pytest-asyncio pytest-timeout pytest-mock freezegun

# Create test directory structure
mkdir -p tests/unit/events
mkdir -p tests/unit/jobs
mkdir -p tests/unit/websockets
mkdir -p tests/unit/auth
mkdir -p tests/performance
```

### Step 2: Write Tests in Order of Impact

1. **Event Bus** (HIGH impact, enables testing of dependent features)
2. **Job Queue** (HIGH impact, core async functionality)
3. **WebSocket Manager** (HIGH impact, real-time features)
4. **Auth Middleware** (HIGH impact, security)
5. **Library Service** (MEDIUM impact, business logic)
6. **Hybrid Search** (MEDIUM impact, search functionality)

### Step 3: Run Coverage After Each Phase

```bash
# Run tests with coverage
pytest --cov=app --cov-report=term-missing --cov-report=html

# View coverage report
open htmlcov/index.html

# Check coverage threshold
pytest --cov=app --cov-fail-under=80
```

### Step 4: Continuous Integration

Add to `.github/workflows/tests.yml`:

```yaml
- name: Run tests with coverage
  run: |
    pytest --cov=app --cov-report=xml --cov-fail-under=75

- name: Upload coverage to Codecov
  uses: codecov/codecov-action@v3
  with:
    files: ./coverage.xml
```

## Test Quality Guidelines

### 1. Test Organization

```python
class TestEventBus:
    """Group related tests in classes."""

    @pytest.fixture
    def event_bus(self):
        """Provide fresh event bus for each test."""
        return EventBus()

    def test_specific_behavior(self, event_bus):
        """Test names should describe what they test."""
        pass
```

### 2. Test Independence

- Each test should be independent
- Use fixtures for setup/teardown
- Don't rely on test execution order
- Clean up resources after tests

### 3. Test Coverage vs Test Quality

- Aim for meaningful tests, not just coverage numbers
- Test edge cases and error conditions
- Test concurrent/async behavior
- Test integration points

### 4. Async Test Patterns

```python
@pytest.mark.asyncio
async def test_async_operation():
    """Test async operations properly."""
    result = await async_function()
    assert result is not None
```

### 5. Mock External Dependencies

```python
@pytest.mark.asyncio
async def test_with_mock(mocker):
    """Mock external services."""
    mock_embed = mocker.patch('app.services.embedding_service.embed')
    mock_embed.return_value = [0.1] * 1536
    # Test code here
```

## Success Criteria

- [ ] Overall coverage ≥ 80%
- [ ] All critical modules ≥ 70% coverage
- [ ] Event Bus coverage ≥ 85%
- [ ] Job Queue coverage ≥ 85%
- [ ] WebSocket Manager coverage ≥ 80%
- [ ] Auth middleware coverage ≥ 90%
- [ ] All tests passing in CI/CD
- [ ] No flaky tests
- [ ] Test execution time < 5 minutes

## Estimated Timeline

| Phase | Time | Coverage Gain | Cumulative |
|-------|------|---------------|------------|
| Current | - | - | 62% |
| Phase 1: Event Bus & Jobs | 4-6h | +8-10% | 70-72% |
| Phase 2: WebSocket & Streaming | 3-4h | +6-8% | 76-80% |
| Phase 3: Auth & Authorization | 3-4h | +5-7% | 81-87% |
| Phase 4: Library Service | 4-5h | +7-9% | 88-96% |
| **Total** | **14-19h** | **+26-34%** | **88-96%** |

## Next Steps

1. ✅ Create this roadmap document
2. ⏳ Implement Phase 1: Event Bus & Job Queue tests
3. ⏳ Implement Phase 2: WebSocket & Streaming tests
4. ⏳ Implement Phase 3: Auth & Authorization tests
5. ⏳ Implement Phase 4: Library Service tests
6. ⏳ Set up CI/CD coverage reporting
7. ⏳ Document test best practices for team

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Pytest-asyncio](https://pytest-asyncio.readthedocs.io/)
- [Coverage.py](https://coverage.readthedocs.io/)
- [Python Testing Best Practices](https://docs.python-guide.org/writing/tests/)
