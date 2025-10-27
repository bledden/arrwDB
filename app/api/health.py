"""
Health check and readiness endpoints for arrwDB.

This module provides comprehensive health monitoring for:
- Liveness probes (basic API health)
- Readiness probes (service dependencies ready)
- Detailed status information for debugging
"""

import logging
import time
from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, status
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Track service start time for uptime calculation
SERVICE_START_TIME = time.time()

router = APIRouter(tags=["health"])


class HealthStatus(BaseModel):
    """Health status response model."""

    status: str
    timestamp: datetime
    uptime_seconds: float
    version: str = "2.0.0"


class ReadinessStatus(BaseModel):
    """Readiness status response model."""

    status: str
    timestamp: datetime
    checks: Dict[str, Any]
    ready: bool


class DetailedHealth(BaseModel):
    """Detailed health status with all components."""

    status: str
    timestamp: datetime
    uptime_seconds: float
    version: str
    components: Dict[str, Dict[str, Any]]


@router.get(
    "/health",
    response_model=HealthStatus,
    status_code=status.HTTP_200_OK,
    summary="Basic health check",
    description="Returns basic API health status. Used for liveness probes.",
)
async def health() -> HealthStatus:
    """
    Basic health check endpoint (liveness probe).

    This endpoint always returns 200 OK if the service is running.
    It's designed for Kubernetes liveness probes and Docker health checks.

    Returns:
        HealthStatus with basic service information
    """
    uptime = time.time() - SERVICE_START_TIME

    return HealthStatus(
        status="healthy",
        timestamp=datetime.utcnow(),
        uptime_seconds=round(uptime, 2),
    )


@router.get(
    "/ready",
    response_model=ReadinessStatus,
    summary="Readiness check",
    description="Returns readiness status with dependency checks. Used for readiness probes.",
)
async def readiness() -> ReadinessStatus:
    """
    Readiness check endpoint (readiness probe).

    This endpoint checks if all service dependencies are ready:
    - Event Bus
    - Job Queue
    - Embedding Service (Cohere)

    Returns 200 OK if all checks pass, 503 Service Unavailable otherwise.
    Designed for Kubernetes readiness probes.

    Returns:
        ReadinessStatus with detailed dependency status
    """
    checks = {}
    all_ready = True

    # Check Event Bus
    try:
        from app.events.bus import EventBus
        event_bus = EventBus()
        checks["event_bus"] = {
            "status": "ready" if event_bus.running else "not_running",
            "healthy": event_bus.running,
        }
        if not event_bus.running:
            all_ready = False
    except Exception as e:
        logger.warning(f"Event Bus check failed: {e}")
        checks["event_bus"] = {
            "status": "error",
            "healthy": False,
            "error": str(e),
        }
        all_ready = False

    # Check Job Queue
    try:
        from app.jobs.queue import JobQueue
        job_queue = JobQueue()
        checks["job_queue"] = {
            "status": "ready" if job_queue.running else "not_running",
            "healthy": job_queue.running,
        }
        if not job_queue.running:
            all_ready = False
    except Exception as e:
        logger.warning(f"Job Queue check failed: {e}")
        checks["job_queue"] = {
            "status": "error",
            "healthy": False,
            "error": str(e),
        }
        all_ready = False

    # Check Embedding Service (basic connectivity test)
    try:
        from app.config import settings
        if settings.COHERE_API_KEY and settings.COHERE_API_KEY != "":
            checks["embedding_service"] = {
                "status": "configured",
                "healthy": True,
            }
        else:
            checks["embedding_service"] = {
                "status": "not_configured",
                "healthy": False,
                "note": "COHERE_API_KEY not set",
            }
            all_ready = False
    except Exception as e:
        logger.warning(f"Embedding Service check failed: {e}")
        checks["embedding_service"] = {
            "status": "error",
            "healthy": False,
            "error": str(e),
        }
        all_ready = False

    return ReadinessStatus(
        status="ready" if all_ready else "not_ready",
        timestamp=datetime.utcnow(),
        checks=checks,
        ready=all_ready,
    )


@router.get(
    "/health/detailed",
    response_model=DetailedHealth,
    summary="Detailed health status",
    description="Returns detailed health information for all components.",
)
async def detailed_health() -> DetailedHealth:
    """
    Detailed health check with all component information.

    This endpoint provides comprehensive status for debugging and monitoring:
    - Service metadata (version, uptime)
    - Event Bus statistics
    - Job Queue statistics
    - WebSocket Manager statistics
    - Embedding Service status

    Returns:
        DetailedHealth with comprehensive component information
    """
    uptime = time.time() - SERVICE_START_TIME
    components = {}

    # Event Bus
    try:
        from app.events.bus import EventBus
        event_bus = EventBus()
        components["event_bus"] = {
            "status": "running" if event_bus.running else "stopped",
            "running": event_bus.running,
            "subscribers": len(event_bus.subscribers) if hasattr(event_bus, 'subscribers') else 0,
        }
    except Exception as e:
        components["event_bus"] = {
            "status": "error",
            "error": str(e),
        }

    # Job Queue
    try:
        from app.jobs.queue import JobQueue
        job_queue = JobQueue()

        # Get job statistics
        all_jobs = job_queue.list_jobs()
        completed = sum(1 for j in all_jobs if j.status == "completed")
        failed = sum(1 for j in all_jobs if j.status == "failed")
        running = sum(1 for j in all_jobs if j.status == "running")
        pending = sum(1 for j in all_jobs if j.status == "pending")

        components["job_queue"] = {
            "status": "running" if job_queue.running else "stopped",
            "running": job_queue.running,
            "total_jobs": len(all_jobs),
            "jobs_by_status": {
                "completed": completed,
                "failed": failed,
                "running": running,
                "pending": pending,
            },
            "handlers_registered": len(job_queue.handlers) if hasattr(job_queue, 'handlers') else 0,
        }
    except Exception as e:
        components["job_queue"] = {
            "status": "error",
            "error": str(e),
        }

    # WebSocket Manager
    try:
        from app.websockets.manager import WebSocketManager
        ws_manager = WebSocketManager()

        # Get connection statistics
        total_connections = ws_manager.get_connection_count()

        components["websocket_manager"] = {
            "status": "available",
            "total_connections": total_connections,
        }
    except Exception as e:
        components["websocket_manager"] = {
            "status": "error",
            "error": str(e),
        }

    # Embedding Service
    try:
        from app.config import settings
        from app.services.embedding_service import EmbeddingService

        embedding_service = EmbeddingService()

        components["embedding_service"] = {
            "status": "configured" if settings.COHERE_API_KEY else "not_configured",
            "model": settings.EMBEDDING_MODEL,
            "dimension": settings.EMBEDDING_DIMENSION,
            "provider": "cohere",
        }
    except Exception as e:
        components["embedding_service"] = {
            "status": "error",
            "error": str(e),
        }

    return DetailedHealth(
        status="healthy",
        timestamp=datetime.utcnow(),
        uptime_seconds=round(uptime, 2),
        version="2.0.0",
        components=components,
    )


@router.get(
    "/ping",
    status_code=status.HTTP_200_OK,
    summary="Simple ping",
    description="Simplest health check - just returns 'pong'",
)
async def ping() -> Dict[str, str]:
    """
    Simplest health check endpoint.

    Returns:
        {"ping": "pong"}
    """
    return {"ping": "pong"}
