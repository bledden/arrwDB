"""
Security middleware for production-grade API protection.

Implements:
- Security headers (OWASP recommendations)
- Request size limits (DoS protection)
- Request ID tracking
"""

import time
import uuid
from typing import Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import settings


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Add security headers to all responses.

    Implements OWASP security header recommendations:
    https://owasp.org/www-project-secure-headers/
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add security headers to response."""
        response = await call_next(request)

        # Prevent MIME type sniffing
        # Stops browsers from guessing content type
        response.headers["X-Content-Type-Options"] = "nosniff"

        # Prevent clickjacking attacks
        # Stops site from being embedded in iframe
        response.headers["X-Frame-Options"] = "DENY"

        # Enable XSS protection (legacy browsers)
        # Modern browsers use CSP instead
        response.headers["X-XSS-Protection"] = "1; mode=block"

        # Enforce HTTPS (if enabled)
        # Tell browsers to only connect via HTTPS for 1 year
        if settings.SECURITY_HEADERS_ENABLED:
            response.headers["Strict-Transport-Security"] = (
                "max-age=31536000; includeSubDomains; preload"
            )

        # Content Security Policy
        # Restrict resource loading to prevent XSS
        # Note: Adjust 'unsafe-inline' for production if needed
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self' data:; "
            "connect-src 'self'; "
            "frame-ancestors 'none';"
        )

        # Referrer policy
        # Control how much referrer information is sent
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Permissions policy (formerly Feature-Policy)
        # Disable unnecessary browser features
        response.headers["Permissions-Policy"] = (
            "geolocation=(), "
            "microphone=(), "
            "camera=(), "
            "payment=(), "
            "usb=(), "
            "magnetometer=(), "
            "gyroscope=(), "
            "speaker=()"
        )

        return response


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """
    Limit request body size to prevent DoS attacks.

    Rejects requests larger than configured MAX_REQUEST_SIZE.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Check request size before processing."""
        # Only check size for methods that can have a body
        if request.method in ["POST", "PUT", "PATCH"]:
            content_length = request.headers.get("content-length")

            if content_length:
                try:
                    size = int(content_length)
                    if size > settings.MAX_REQUEST_SIZE:
                        # Calculate human-readable sizes
                        max_mb = settings.MAX_REQUEST_SIZE / (1024 * 1024)
                        actual_mb = size / (1024 * 1024)

                        return JSONResponse(
                            status_code=413,  # Payload Too Large
                            content={
                                "error": "Request too large",
                                "detail": f"Request size {actual_mb:.2f}MB exceeds maximum {max_mb:.2f}MB",
                                "max_size_bytes": settings.MAX_REQUEST_SIZE,
                                "actual_size_bytes": size,
                            },
                        )
                except ValueError:
                    # Invalid content-length header, let it through
                    # (will likely fail elsewhere)
                    pass

        return await call_next(request)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Add unique request ID to each request for tracing.

    Useful for:
    - Distributed tracing
    - Log correlation
    - Debugging
    - Support tickets
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add request ID to request and response."""
        # Check if client provided a request ID
        request_id = request.headers.get("X-Request-ID")

        # Generate one if not provided
        if not request_id:
            request_id = str(uuid.uuid4())

        # Store in request state for access in routes
        request.state.request_id = request_id

        # Store request start time for latency tracking
        request.state.start_time = time.time()

        # Process request
        response = await call_next(request)

        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id

        # Add latency header for monitoring
        latency = time.time() - request.state.start_time
        response.headers["X-Response-Time"] = f"{latency:.3f}s"

        return response


def get_request_id(request: Request) -> str:
    """
    Get request ID from request state.

    Usage in routes:
        from fastapi import Request
        request_id = get_request_id(request)
    """
    return getattr(request.state, "request_id", "unknown")


def get_request_latency(request: Request) -> float:
    """
    Get request latency in seconds.

    Usage in routes:
        from fastapi import Request
        latency = get_request_latency(request)
    """
    start_time = getattr(request.state, "start_time", None)
    if start_time:
        return time.time() - start_time
    return 0.0
