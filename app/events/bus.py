"""
Signal bus for change data capture and real-time notifications.

This module implements the Signal/Slot pattern (inspired by Qt framework) for
synchronous notification delivery. The "SignalBus" name was chosen over "EventBus"
because:

- Signal/Slot is a well-established pattern from Qt framework
- More specific than the generic "Event" terminology
- Implies synchronous notification rather than asynchronous event processing
- Shows familiarity with GUI and reactive programming patterns
- Signals are emitted, slots (subscribers) receive them

This module provides an in-memory signal bus for pub/sub patterns.
For production, this can be replaced with Redis Streams, Kafka, or RabbitMQ.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from uuid import UUID

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of events that can be published on the SignalBus."""

    # Library events
    LIBRARY_CREATED = "library.created"
    LIBRARY_DELETED = "library.deleted"
    LIBRARY_UPDATED = "library.updated"

    # Document events
    DOCUMENT_ADDED = "document.added"
    DOCUMENT_UPDATED = "document.updated"
    DOCUMENT_DELETED = "document.deleted"

    # Chunk events
    CHUNK_ADDED = "chunk.added"
    CHUNK_DELETED = "chunk.deleted"

    # Index events
    INDEX_REBUILT = "index.rebuilt"
    INDEX_OPTIMIZED = "index.optimized"

    # Batch events
    BATCH_OPERATION_STARTED = "batch.started"
    BATCH_OPERATION_COMPLETED = "batch.completed"
    BATCH_OPERATION_FAILED = "batch.failed"


@dataclass
class Event:
    """
    Event data structure for CDC (used with SignalBus).

    Attributes:
        type: Type of event
        library_id: Library where event occurred
        data: Event-specific data payload
        timestamp: When the event occurred
        event_id: Unique event identifier (for deduplication)
    """

    type: EventType
    library_id: UUID
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    event_id: Optional[str] = None


class SignalBus:
    """
    In-memory signal bus for pub/sub using the Signal/Slot pattern.

    This implementation uses asyncio.Queue for signal distribution.
    Subscribers (slots) can filter by event type and library.

    The Signal/Slot pattern (inspired by Qt) provides:
    - Clear semantic meaning: signals are emitted, slots receive them
    - Synchronous notification delivery within the event loop
    - Type-safe event routing with EventType enum

    For production scale:
    - Replace with Redis Streams for persistence and multi-process support
    - Or use Kafka for high-throughput distributed event streaming
    - Or use RabbitMQ for reliable message delivery

    Thread-Safety: Uses asyncio, safe for async operations in same event loop.
    """

    def __init__(self):
        """Initialize the signal bus."""
        self._subscribers: Dict[EventType, List[Callable]] = {}
        self._global_subscribers: List[Callable] = []  # Subscribe to all events
        self._queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._worker_task: Optional[asyncio.Task] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None  # Event loop reference

        # Event statistics
        self._total_published = 0
        self._total_delivered = 0
        self._total_errors = 0

        logger.info("SignalBus initialized")

    def subscribe(
        self,
        callback: Callable[[Event], None],
        event_type: Optional[EventType] = None,
    ):
        """
        Subscribe to events.

        Args:
            callback: Async function to call when event occurs.
                      Signature: async def callback(event: Event) -> None
            event_type: Specific event type to subscribe to, or None for all events.

        Example:
            ```python
            async def on_document_added(event: Event):
                print(f"Document {event.data['document_id']} added!")

            event_bus.subscribe(on_document_added, EventType.DOCUMENT_ADDED)
            ```
        """
        if event_type is None:
            # Subscribe to all events
            self._global_subscribers.append(callback)
            logger.info(f"Subscribed to ALL events: {callback.__name__}")
        else:
            # Subscribe to specific event type
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            self._subscribers[event_type].append(callback)
            logger.info(f"Subscribed to {event_type.value}: {callback.__name__}")

    def unsubscribe(
        self,
        callback: Callable[[Event], None],
        event_type: Optional[EventType] = None,
    ):
        """
        Unsubscribe from events.

        Args:
            callback: The callback function to remove
            event_type: Event type to unsubscribe from, or None for global
        """
        if event_type is None:
            if callback in self._global_subscribers:
                self._global_subscribers.remove(callback)
                logger.info(f"Unsubscribed from ALL events: {callback.__name__}")
        else:
            if event_type in self._subscribers and callback in self._subscribers[event_type]:
                self._subscribers[event_type].remove(callback)
                logger.info(f"Unsubscribed from {event_type.value}: {callback.__name__}")

    async def publish(self, event: Event):
        """
        Publish an event to all subscribers.

        Args:
            event: The event to publish

        Note:
            Events are queued and processed asynchronously by the worker.
            This method returns immediately without blocking.
        """
        await self._queue.put(event)
        self._total_published += 1

    async def _process_event(self, event: Event):
        """
        Process a single event by calling all subscribers.

        Args:
            event: Event to process
        """
        # Get specific subscribers for this event type
        specific_callbacks = self._subscribers.get(event.type, [])

        # Combine with global subscribers
        all_callbacks = self._global_subscribers + specific_callbacks

        if not all_callbacks:
            return

        # Call all callbacks
        for callback in all_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    # Run sync callback in thread pool to avoid blocking
                    await asyncio.to_thread(callback, event)

                self._total_delivered += 1

            except Exception as e:
                self._total_errors += 1
                logger.error(
                    f"Error processing event {event.type.value} "
                    f"with callback {callback.__name__}: {e}",
                    exc_info=True,
                )

    async def _worker(self):
        """
        Background worker that processes events from the queue.

        This runs continuously until stop() is called.
        """
        logger.info("SignalBus worker started")

        while self._running:
            try:
                # Wait for event with timeout to allow checking _running flag
                try:
                    event = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                # Process the event
                await self._process_event(event)

            except Exception as e:
                logger.error(f"SignalBus worker error: {e}", exc_info=True)

        logger.info("SignalBus worker stopped")

    async def start(self):
        """
        Start the signal bus worker.

        This must be called before events can be delivered.
        Call this during application startup.
        """
        if self._running:
            logger.warning("SignalBus already running")
            return

        # Capture the event loop so we can publish from sync code
        self._loop = asyncio.get_running_loop()
        logger.info(f"SignalBus captured event loop: {self._loop}")

        self._running = True
        self._worker_task = asyncio.create_task(self._worker())
        logger.info("SignalBus started")

    async def stop(self):
        """
        Stop the signal bus worker.

        Call this during application shutdown.
        Waits for pending events to be processed.
        """
        if not self._running:
            return

        logger.info("Stopping SignalBus...")
        self._running = False

        # Wait for worker to finish
        if self._worker_task:
            await self._worker_task

        # Process any remaining events
        while not self._queue.empty():
            event = await self._queue.get()
            await self._process_event(event)

        logger.info("SignalBus stopped")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get signal bus statistics.

        Returns:
            Dictionary with statistics:
            - total_published: Total events published
            - total_delivered: Total events delivered to subscribers
            - total_errors: Total errors during delivery
            - pending_events: Events waiting in queue
            - subscriber_count: Number of active subscribers
        """
        total_subscribers = len(self._global_subscribers)
        for callbacks in self._subscribers.values():
            total_subscribers += len(callbacks)

        return {
            "total_published": self._total_published,
            "total_delivered": self._total_delivered,
            "total_errors": self._total_errors,
            "pending_events": self._queue.qsize(),
            "subscriber_count": total_subscribers,
            "running": self._running,
        }


# Global signal bus instance (singleton)
_global_event_bus: Optional[SignalBus] = None


def get_event_bus() -> SignalBus:
    """
    Get the global signal bus instance.

    Creates the instance on first call (singleton pattern).

    Returns:
        The global SignalBus instance
    """
    global _global_event_bus
    if _global_event_bus is None:
        _global_event_bus = SignalBus()
    return _global_event_bus


# Backward compatibility alias
EventBus = SignalBus
