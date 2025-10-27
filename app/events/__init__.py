"""
Event system for change data capture (CDC).

This module provides:
- Event bus for pub/sub
- Event types and models
- Background event processing
"""

from app.events.bus import EventBus, Event, EventType

__all__ = ["EventBus", "Event", "EventType"]
