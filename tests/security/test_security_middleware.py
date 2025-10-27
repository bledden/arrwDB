"""
Tests for security middleware (CORS, security headers, request size limits).

Tests verify:
- CORS headers are correctly applied
- Security headers follow OWASP recommendations
- Request size limits prevent DoS
- Request ID tracking works
"""

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from app.middleware.security import (
    SecurityHeadersMiddleware,
    RequestSizeLimitMiddleware,
    RequestIDMiddleware,
    get_request_id,
    get_request_latency,
)
from app.config import settings


# ============================================================
# Test Fixtures
# ============================================================

@pytest.fixture
def test_app():
    """Create a test FastAPI app with security middleware."""
    app = FastAPI()

    # Add middleware in correct order (reverse of execution order)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RequestSizeLimitMiddleware)
    app.add_middleware(RequestIDMiddleware)

    @app.get("/test")
    async def test_endpoint(request: Request):
        return {
            "message": "success",
            "request_id": get_request_id(request),
            "latency": get_request_latency(request),
        }

    @app.post("/test-post")
    async def test_post(request: Request):
        body = await request.body()
        return {"size": len(body), "request_id": get_request_id(request)}

    return app


@pytest.fixture
def client(test_app):
    """Create a test client."""
    return TestClient(test_app)


# ============================================================
# Security Headers Tests
# ============================================================

def test_security_headers_present(client):
    """Test that all OWASP security headers are present."""
    response = client.get("/test")

    assert response.status_code == 200

    # Check all required security headers
    assert "X-Content-Type-Options" in response.headers
    assert response.headers["X-Content-Type-Options"] == "nosniff"

    assert "X-Frame-Options" in response.headers
    assert response.headers["X-Frame-Options"] == "DENY"

    assert "X-XSS-Protection" in response.headers
    assert response.headers["X-XSS-Protection"] == "1; mode=block"

    assert "Strict-Transport-Security" in response.headers
    assert "max-age=31536000" in response.headers["Strict-Transport-Security"]
    assert "includeSubDomains" in response.headers["Strict-Transport-Security"]

    assert "Content-Security-Policy" in response.headers
    assert "default-src 'self'" in response.headers["Content-Security-Policy"]

    assert "Referrer-Policy" in response.headers
    assert response.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"

    assert "Permissions-Policy" in response.headers
    assert "geolocation=()" in response.headers["Permissions-Policy"]


def test_hsts_header_includes_preload(client):
    """Test that HSTS header includes preload directive."""
    response = client.get("/test")
    hsts = response.headers.get("Strict-Transport-Security", "")

    assert "preload" in hsts
    assert "includeSubDomains" in hsts
    assert "max-age=31536000" in hsts  # 1 year


def test_csp_blocks_inline_scripts(client):
    """Test that CSP header restricts script sources."""
    response = client.get("/test")
    csp = response.headers.get("Content-Security-Policy", "")

    # Should have default-src 'self'
    assert "default-src 'self'" in csp

    # Should restrict frames
    assert "frame-ancestors 'none'" in csp


# ============================================================
# Request Size Limit Tests
# ============================================================

def test_request_size_limit_allows_small_requests(client):
    """Test that small requests pass through."""
    small_data = "x" * 1000  # 1KB
    response = client.post("/test-post", content=small_data)

    assert response.status_code == 200
    assert response.json()["size"] == 1000


def test_request_size_limit_rejects_large_requests(client, monkeypatch):
    """Test that requests exceeding size limit are rejected."""
    # Temporarily set a small limit for testing
    monkeypatch.setattr(settings, "MAX_REQUEST_SIZE", 1024)  # 1KB limit

    large_data = "x" * (2 * 1024)  # 2KB (exceeds limit)
    response = client.post(
        "/test-post",
        content=large_data,
        headers={"Content-Length": str(len(large_data))},
    )

    assert response.status_code == 413  # Payload Too Large
    assert "error" in response.json()
    assert "too large" in response.json()["error"].lower()


def test_request_size_limit_returns_helpful_error(client, monkeypatch):
    """Test that size limit error includes helpful details."""
    monkeypatch.setattr(settings, "MAX_REQUEST_SIZE", 1024)

    large_data = "x" * (2 * 1024)
    response = client.post(
        "/test-post",
        content=large_data,
        headers={"Content-Length": str(len(large_data))},
    )

    data = response.json()
    assert "max_size_bytes" in data
    assert "actual_size_bytes" in data
    assert data["max_size_bytes"] == 1024
    assert data["actual_size_bytes"] == 2048


def test_request_size_limit_only_checks_post_put_patch(client):
    """Test that GET requests are not checked for size."""
    # GET requests don't have a body, so no size check
    response = client.get("/test")
    assert response.status_code == 200


# ============================================================
# Request ID Tests
# ============================================================

def test_request_id_added_to_response(client):
    """Test that request ID is added to response headers."""
    response = client.get("/test")

    assert "X-Request-ID" in response.headers
    request_id = response.headers["X-Request-ID"]

    # Should be a valid UUID format
    assert len(request_id) == 36  # UUID format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
    assert request_id.count("-") == 4


def test_request_id_preserves_client_id(client):
    """Test that client-provided request ID is preserved."""
    client_request_id = "test-request-123"
    response = client.get("/test", headers={"X-Request-ID": client_request_id})

    assert response.headers["X-Request-ID"] == client_request_id


def test_request_id_accessible_in_endpoint(client):
    """Test that request ID is accessible within endpoint."""
    response = client.get("/test")

    assert response.status_code == 200
    data = response.json()

    # Should be same as header
    assert data["request_id"] == response.headers["X-Request-ID"]


def test_response_time_header_present(client):
    """Test that response time header is added."""
    response = client.get("/test")

    assert "X-Response-Time" in response.headers
    response_time = response.headers["X-Response-Time"]

    # Should be in format "0.XXXs"
    assert response_time.endswith("s")
    assert float(response_time[:-1]) >= 0


def test_request_latency_tracking(client):
    """Test that request latency is tracked correctly."""
    response = client.get("/test")

    assert response.status_code == 200
    data = response.json()

    # Latency should be positive
    assert data["latency"] > 0

    # Should be reasonable (< 1 second for simple endpoint)
    assert data["latency"] < 1.0


# ============================================================
# CORS Tests (Integration)
# ============================================================

def test_cors_preflight_request():
    """Test CORS preflight (OPTIONS) request handling."""
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["https://example.com"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
        max_age=3600,
    )

    @app.get("/test")
    async def test():
        return {"message": "success"}

    client = TestClient(app)

    # Send preflight request
    response = client.options(
        "/test",
        headers={
            "Origin": "https://example.com",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code == 200
    assert "Access-Control-Allow-Origin" in response.headers


def test_cors_actual_request():
    """Test actual CORS request with allowed origin."""
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["https://example.com"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.get("/test")
    async def test():
        return {"message": "success"}

    client = TestClient(app)

    response = client.get("/test", headers={"Origin": "https://example.com"})

    assert response.status_code == 200
    assert response.headers["Access-Control-Allow-Origin"] == "https://example.com"


def test_cors_wildcard_origin():
    """Test CORS with wildcard origin."""
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.get("/test")
    async def test():
        return {"message": "success"}

    client = TestClient(app)

    response = client.get("/test", headers={"Origin": "https://anything.com"})

    assert response.status_code == 200
    assert "Access-Control-Allow-Origin" in response.headers


# ============================================================
# Combined Middleware Tests
# ============================================================

def test_all_middleware_together(client):
    """Test that all middleware work together correctly."""
    response = client.get("/test")

    # Should have request ID
    assert "X-Request-ID" in response.headers

    # Should have response time
    assert "X-Response-Time" in response.headers

    # Should have security headers
    assert "X-Content-Type-Options" in response.headers
    assert "X-Frame-Options" in response.headers

    # Response should be successful
    assert response.status_code == 200


def test_middleware_order_preserved(client):
    """Test that middleware execute in correct order."""
    # Request ID should be added first (outermost middleware)
    # Security headers should be added last (innermost middleware)

    response = client.get("/test")

    # All headers should be present
    assert "X-Request-ID" in response.headers
    assert "X-Content-Type-Options" in response.headers

    # Response should contain request ID from middleware
    data = response.json()
    assert "request_id" in data
    assert data["request_id"] == response.headers["X-Request-ID"]
