"""
Gunicorn configuration for production deployment.

This configuration provides:
- Auto-detection of optimal worker count based on CPU cores
- Graceful worker restarts to prevent memory leaks
- Production-grade logging and monitoring
- Configurable via environment variables

Usage:
    gunicorn app.api.main:app -c gunicorn_conf.py
    GUNICORN_WORKERS=8 gunicorn app.api.main:app -c gunicorn_conf.py
"""

import multiprocessing
import os
from app.config import settings

# ============================================================
# Server Socket
# ============================================================

bind = f"{settings.HOST}:{settings.PORT}"
backlog = 2048  # Maximum number of pending connections

# ============================================================
# Worker Processes
# ============================================================

# Number of worker processes
# For CPU-bound tasks (vector search), use CPU count
# For I/O-bound tasks, typically use (2 * CPU count) + 1
workers = settings.workers

# Worker class - use Uvicorn workers for ASGI support
worker_class = "uvicorn.workers.UvicornWorker"

# Number of pending connections each worker can handle
worker_connections = 1000

# ============================================================
# Worker Lifecycle
# ============================================================

# Restart workers after serving this many requests
# This prevents memory leaks from accumulating
max_requests = settings.GUNICORN_MAX_REQUESTS

# Randomize max_requests to avoid all workers restarting simultaneously
max_requests_jitter = settings.GUNICORN_MAX_REQUESTS_JITTER

# Timeout for worker requests (in seconds)
# Vector search can take time, especially for large libraries
timeout = settings.GUNICORN_TIMEOUT

# Time to wait for workers to finish serving requests during graceful shutdown
graceful_timeout = 30

# Workers silent for more than this time are killed and restarted
keepalive = 2

# ============================================================
# Logging
# ============================================================

# Access log - write to stdout for Docker/cloud logging
accesslog = "-"

# Error log - write to stderr
errorlog = "-"

# Log level
loglevel = settings.LOG_LEVEL.lower()

# Access log format
# Shows: client IP, timestamp, request, status, response time
access_log_format = (
    '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'
)

# ============================================================
# Process Naming
# ============================================================

proc_name = "vector_db_api"

# ============================================================
# Server Mechanics
# ============================================================

# Run in foreground (not as daemon)
daemon = False

# Disable PID file (not needed in containers)
pidfile = None

# User/group to run workers as (None = don't change)
user = None
group = None

# Directory for temporary files
tmp_upload_dir = None

# ============================================================
# Server Hooks
# ============================================================


def on_starting(server):
    """
    Called just before the master process is initialized.

    Useful for logging configuration summary.
    """
    from app.config import print_config_summary

    print_config_summary()


def when_ready(server):
    """
    Called just after the server is started.

    Useful for health check endpoints or initialization tasks.
    """
    print(f"\n✓ Server is ready! Workers: {workers}")
    print(f"✓ Listening on: http://{settings.HOST}:{settings.PORT}")
    if settings.ENABLE_DOCS:
        print(f"✓ API docs: http://{settings.HOST}:{settings.PORT}/docs")
    print()


def worker_int(worker):
    """
    Called when a worker receives SIGINT or SIGQUIT signal.

    Useful for cleanup tasks.
    """
    worker.log.info(f"Worker {worker.pid} received shutdown signal")


def worker_abort(worker):
    """
    Called when a worker times out and is killed.

    Useful for debugging timeout issues.
    """
    worker.log.error(
        f"Worker {worker.pid} timed out (timeout={timeout}s). "
        "Consider increasing GUNICORN_TIMEOUT if vector searches are slow."
    )


# ============================================================
# SSL Configuration (optional)
# ============================================================

# Uncomment and configure for HTTPS
# keyfile = "/path/to/key.pem"
# certfile = "/path/to/cert.pem"

# ============================================================
# Performance Tuning
# ============================================================

# Pre-load application code before forking workers
# This saves memory by sharing code across workers
# Only enable if your code is safe to pre-load (no side effects on import)
preload_app = False  # Disabled by default for safety

# Enable sendfile() for static files (not applicable for API)
sendfile = False

# ============================================================
# Development Mode Overrides
# ============================================================

if settings.DEBUG:
    # In debug mode, use fewer workers and enable auto-reload
    workers = 1
    reload = True
    loglevel = "debug"
    print("\n⚠️  DEBUG MODE ENABLED - Using 1 worker with auto-reload\n")
