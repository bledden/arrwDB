"""
Event Bus API endpoints for real-time change data capture.

Provides endpoints to monitor event statistics and subscribe to event streams.
"""

from typing import Dict, Any
from fastapi import APIRouter, Depends

from app.events.bus import EventBus, get_event_bus

router = APIRouter(prefix="/v1/events", tags=["events"])


@router.get("/stats", response_model=Dict[str, Any])
async def get_event_stats(event_bus: EventBus = Depends(get_event_bus)) -> Dict[str, Any]:
    """
    Get event bus statistics.

    Returns:
        Statistics about event publishing and subscriptions:
        - total_published: Total events published since startup
        - total_subscriptions: Number of active subscribers
        - queue_size: Current number of pending events
        - is_running: Whether the event bus worker is running
    """
    stats = event_bus.get_statistics()
    return stats
