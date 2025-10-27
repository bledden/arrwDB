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

import asyncio
from datetime import datetime
from uuid import uuid4

import pytest

from app.events.bus import Event, EventBus, EventType, get_event_bus


class TestEventCreation:
    """Test event creation and data structures."""

    def test_event_creation(self):
        """Test basic event creation."""
        library_id = uuid4()
        event = Event(
            type=EventType.DOCUMENT_ADDED,
            library_id=library_id,
            data={"document_id": "doc123", "title": "Test Doc"},
        )

        assert event.type == EventType.DOCUMENT_ADDED
        assert event.library_id == library_id
        assert event.data["document_id"] == "doc123"
        assert isinstance(event.timestamp, datetime)

    def test_event_with_metadata(self):
        """Test event creation with all optional fields."""
        library_id = uuid4()
        event_id = "custom-event-id-123"
        timestamp = datetime.utcnow()

        event = Event(
            type=EventType.LIBRARY_CREATED,
            library_id=library_id,
            data={"name": "Test Library", "dimensions": 1536},
            timestamp=timestamp,
            event_id=event_id,
        )

        assert event.type == EventType.LIBRARY_CREATED
        assert event.library_id == library_id
        assert event.data["name"] == "Test Library"
        assert event.timestamp == timestamp
        assert event.event_id == event_id


class TestEventBusSubscription:
    """Test event subscription functionality."""

    @pytest.fixture
    def event_bus(self):
        """Provide fresh event bus for each test."""
        return EventBus()

    def test_subscribe_to_specific_event_type(self, event_bus):
        """Test subscribing to a specific event type."""
        callback_called = []

        async def callback(event: Event):
            callback_called.append(event)

        event_bus.subscribe(callback, EventType.DOCUMENT_ADDED)

        # Verify subscription was added
        assert EventType.DOCUMENT_ADDED in event_bus._subscribers
        assert callback in event_bus._subscribers[EventType.DOCUMENT_ADDED]

    def test_subscribe_to_all_events(self, event_bus):
        """Test subscribing to all event types (global subscriber)."""
        callback_called = []

        async def callback(event: Event):
            callback_called.append(event)

        event_bus.subscribe(callback)

        # Verify global subscription
        assert callback in event_bus._global_subscribers

    @pytest.mark.asyncio
    async def test_multiple_subscribers_same_event(self, event_bus):
        """Test multiple subscribers for the same event type."""
        calls_1 = []
        calls_2 = []

        async def callback_1(event: Event):
            calls_1.append(event)

        async def callback_2(event: Event):
            calls_2.append(event)

        event_bus.subscribe(callback_1, EventType.DOCUMENT_ADDED)
        event_bus.subscribe(callback_2, EventType.DOCUMENT_ADDED)

        # Verify both subscribers were added
        assert len(event_bus._subscribers[EventType.DOCUMENT_ADDED]) == 2

    def test_unsubscribe(self, event_bus):
        """Test unsubscribing from events."""
        async def callback(event: Event):
            pass

        # Subscribe then unsubscribe
        event_bus.subscribe(callback, EventType.DOCUMENT_ADDED)
        assert callback in event_bus._subscribers[EventType.DOCUMENT_ADDED]

        event_bus.unsubscribe(callback, EventType.DOCUMENT_ADDED)
        assert callback not in event_bus._subscribers[EventType.DOCUMENT_ADDED]

    def test_unsubscribe_global(self, event_bus):
        """Test unsubscribing from global events."""
        async def callback(event: Event):
            pass

        # Subscribe then unsubscribe globally
        event_bus.subscribe(callback)
        assert callback in event_bus._global_subscribers

        event_bus.unsubscribe(callback)
        assert callback not in event_bus._global_subscribers


class TestEventPublishing:
    """Test event publishing and delivery."""

    @pytest.fixture
    def event_bus(self):
        """Provide fresh event bus for each test."""
        return EventBus()

    @pytest.mark.asyncio
    async def test_publish_event(self, event_bus):
        """Test publishing an event."""
        library_id = uuid4()
        event = Event(
            type=EventType.DOCUMENT_ADDED,
            library_id=library_id,
            data={"document_id": "doc123"},
        )

        await event_bus.start()
        await event_bus.publish(event)

        # Event should be in queue
        stats = event_bus.get_statistics()
        assert stats["total_published"] == 1

        await event_bus.stop()

    @pytest.mark.asyncio
    async def test_event_delivered_to_correct_subscribers(self, event_bus):
        """Test events are delivered only to matching subscribers."""
        received_events = []
        wrong_type_events = []

        async def correct_callback(event: Event):
            received_events.append(event)

        async def wrong_type_callback(event: Event):
            wrong_type_events.append(event)

        # Subscribe to different event types
        event_bus.subscribe(correct_callback, EventType.DOCUMENT_ADDED)
        event_bus.subscribe(wrong_type_callback, EventType.LIBRARY_CREATED)

        # Publish a DOCUMENT_ADDED event
        library_id = uuid4()
        event = Event(
            type=EventType.DOCUMENT_ADDED,
            library_id=library_id,
            data={"document_id": "doc123"},
        )

        await event_bus.start()
        await event_bus.publish(event)

        # Give worker time to process
        await asyncio.sleep(0.1)
        await event_bus.stop()

        # Verify delivery
        assert len(received_events) == 1
        assert received_events[0].type == EventType.DOCUMENT_ADDED
        assert len(wrong_type_events) == 0

    @pytest.mark.asyncio
    async def test_global_subscriber_receives_all_events(self, event_bus):
        """Test global subscribers receive all event types."""
        all_events = []

        async def global_callback(event: Event):
            all_events.append(event)

        event_bus.subscribe(global_callback)  # No event type = global

        await event_bus.start()

        # Publish different event types
        library_id = uuid4()
        events = [
            Event(type=EventType.DOCUMENT_ADDED, library_id=library_id, data={}),
            Event(type=EventType.LIBRARY_CREATED, library_id=library_id, data={}),
            Event(type=EventType.INDEX_REBUILT, library_id=library_id, data={}),
        ]

        for event in events:
            await event_bus.publish(event)

        # Give worker time to process
        await asyncio.sleep(0.2)
        await event_bus.stop()

        # Global subscriber should receive all events
        assert len(all_events) == 3
        assert all_events[0].type == EventType.DOCUMENT_ADDED
        assert all_events[1].type == EventType.LIBRARY_CREATED
        assert all_events[2].type == EventType.INDEX_REBUILT

    @pytest.mark.asyncio
    async def test_multiple_subscribers_all_receive_event(self, event_bus):
        """Test that all subscribers for an event type receive the event."""
        calls_1 = []
        calls_2 = []
        calls_3 = []

        async def callback_1(event: Event):
            calls_1.append(event)

        async def callback_2(event: Event):
            calls_2.append(event)

        async def callback_3(event: Event):
            calls_3.append(event)

        # Multiple subscribers for same event type
        event_bus.subscribe(callback_1, EventType.DOCUMENT_ADDED)
        event_bus.subscribe(callback_2, EventType.DOCUMENT_ADDED)
        event_bus.subscribe(callback_3, EventType.DOCUMENT_ADDED)

        library_id = uuid4()
        event = Event(
            type=EventType.DOCUMENT_ADDED,
            library_id=library_id,
            data={"document_id": "doc123"},
        )

        await event_bus.start()
        await event_bus.publish(event)
        await asyncio.sleep(0.1)
        await event_bus.stop()

        # All subscribers should receive the event
        assert len(calls_1) == 1
        assert len(calls_2) == 1
        assert len(calls_3) == 1


class TestEventBusErrorHandling:
    """Test error handling in event processing."""

    @pytest.fixture
    def event_bus(self):
        """Provide fresh event bus for each test."""
        return EventBus()

    @pytest.mark.asyncio
    async def test_subscriber_error_handling(self, event_bus):
        """Test that errors in one callback don't affect others."""
        successful_calls = []
        error_calls = []

        async def failing_callback(event: Event):
            error_calls.append(event)
            raise ValueError("Simulated error")

        async def successful_callback(event: Event):
            successful_calls.append(event)

        # Subscribe both callbacks
        event_bus.subscribe(failing_callback, EventType.DOCUMENT_ADDED)
        event_bus.subscribe(successful_callback, EventType.DOCUMENT_ADDED)

        library_id = uuid4()
        event = Event(
            type=EventType.DOCUMENT_ADDED,
            library_id=library_id,
            data={"document_id": "doc123"},
        )

        await event_bus.start()
        await event_bus.publish(event)
        await asyncio.sleep(0.1)
        await event_bus.stop()

        # Both callbacks should have been called despite error
        assert len(error_calls) == 1
        assert len(successful_calls) == 1

        # Error should be tracked in statistics
        stats = event_bus.get_statistics()
        assert stats["total_errors"] == 1

    @pytest.mark.asyncio
    async def test_sync_callback_support(self, event_bus):
        """Test that synchronous callbacks are supported."""
        sync_calls = []

        def sync_callback(event: Event):
            """Synchronous (non-async) callback."""
            sync_calls.append(event)

        event_bus.subscribe(sync_callback, EventType.DOCUMENT_ADDED)

        library_id = uuid4()
        event = Event(
            type=EventType.DOCUMENT_ADDED,
            library_id=library_id,
            data={"document_id": "doc123"},
        )

        await event_bus.start()
        await event_bus.publish(event)
        await asyncio.sleep(0.1)
        await event_bus.stop()

        # Sync callback should have been called
        assert len(sync_calls) == 1


class TestEventBusLifecycle:
    """Test event bus lifecycle management."""

    @pytest.fixture
    def event_bus(self):
        """Provide fresh event bus for each test."""
        return EventBus()

    @pytest.mark.asyncio
    async def test_bus_start_stop(self, event_bus):
        """Test starting and stopping the event bus."""
        assert not event_bus._running

        await event_bus.start()
        assert event_bus._running
        assert event_bus._worker_task is not None

        await event_bus.stop()
        assert not event_bus._running

    @pytest.mark.asyncio
    async def test_start_already_running(self, event_bus):
        """Test starting an already running bus (should be idempotent)."""
        await event_bus.start()
        assert event_bus._running

        # Start again - should not create duplicate worker
        await event_bus.start()
        assert event_bus._running

        await event_bus.stop()

    @pytest.mark.asyncio
    async def test_stop_processes_pending_events(self, event_bus):
        """Test that stop() processes remaining events in queue."""
        received_events = []

        async def callback(event: Event):
            received_events.append(event)

        event_bus.subscribe(callback, EventType.DOCUMENT_ADDED)

        await event_bus.start()

        # Publish multiple events
        library_id = uuid4()
        for i in range(5):
            event = Event(
                type=EventType.DOCUMENT_ADDED,
                library_id=library_id,
                data={"document_id": f"doc{i}"},
            )
            await event_bus.publish(event)

        # Stop immediately (should process all pending events)
        await event_bus.stop()

        # All events should have been processed
        assert len(received_events) == 5


class TestEventBusStatistics:
    """Test event bus statistics tracking."""

    @pytest.fixture
    def event_bus(self):
        """Provide fresh event bus for each test."""
        return EventBus()

    @pytest.mark.asyncio
    async def test_event_statistics(self, event_bus):
        """Test event statistics tracking."""
        received = []

        async def callback(event: Event):
            received.append(event)

        event_bus.subscribe(callback, EventType.DOCUMENT_ADDED)

        stats = event_bus.get_statistics()
        assert stats["total_published"] == 0
        assert stats["total_delivered"] == 0
        assert stats["subscriber_count"] == 1
        assert stats["running"] is False

        await event_bus.start()

        library_id = uuid4()
        for i in range(3):
            event = Event(
                type=EventType.DOCUMENT_ADDED,
                library_id=library_id,
                data={"document_id": f"doc{i}"},
            )
            await event_bus.publish(event)

        await asyncio.sleep(0.2)
        await event_bus.stop()

        stats = event_bus.get_statistics()
        assert stats["total_published"] == 3
        assert stats["total_delivered"] == 3
        assert stats["total_errors"] == 0
        assert stats["pending_events"] == 0

    @pytest.mark.asyncio
    async def test_subscriber_count_statistics(self, event_bus):
        """Test subscriber count tracking."""
        async def callback_1(event: Event):
            pass

        async def callback_2(event: Event):
            pass

        async def callback_3(event: Event):
            pass

        stats = event_bus.get_statistics()
        assert stats["subscriber_count"] == 0

        # Add specific subscribers
        event_bus.subscribe(callback_1, EventType.DOCUMENT_ADDED)
        event_bus.subscribe(callback_2, EventType.DOCUMENT_ADDED)

        stats = event_bus.get_statistics()
        assert stats["subscriber_count"] == 2

        # Add global subscriber
        event_bus.subscribe(callback_3)

        stats = event_bus.get_statistics()
        assert stats["subscriber_count"] == 3


class TestConcurrentPublishing:
    """Test concurrent event publishing."""

    @pytest.fixture
    def event_bus(self):
        """Provide fresh event bus for each test."""
        return EventBus()

    @pytest.mark.asyncio
    async def test_concurrent_publishing(self, event_bus):
        """Test publishing events concurrently from multiple tasks."""
        received_events = []

        async def callback(event: Event):
            received_events.append(event)

        event_bus.subscribe(callback, EventType.DOCUMENT_ADDED)

        await event_bus.start()

        # Publish events concurrently
        library_id = uuid4()

        async def publish_events(start_idx: int, count: int):
            for i in range(start_idx, start_idx + count):
                event = Event(
                    type=EventType.DOCUMENT_ADDED,
                    library_id=library_id,
                    data={"document_id": f"doc{i}"},
                )
                await event_bus.publish(event)

        # Create multiple publishing tasks
        tasks = [
            publish_events(0, 10),
            publish_events(10, 10),
            publish_events(20, 10),
        ]

        await asyncio.gather(*tasks)
        await asyncio.sleep(0.3)
        await event_bus.stop()

        # All 30 events should have been processed
        assert len(received_events) == 30

    @pytest.mark.asyncio
    async def test_event_ordering_within_publisher(self, event_bus):
        """Test that events from single publisher maintain order."""
        received_events = []

        async def callback(event: Event):
            received_events.append(event.data["index"])

        event_bus.subscribe(callback, EventType.DOCUMENT_ADDED)

        await event_bus.start()

        library_id = uuid4()
        expected_order = list(range(20))

        # Publish events in order
        for i in expected_order:
            event = Event(
                type=EventType.DOCUMENT_ADDED,
                library_id=library_id,
                data={"index": i},
            )
            await event_bus.publish(event)

        await asyncio.sleep(0.2)
        await event_bus.stop()

        # Events should be processed in order
        assert received_events == expected_order


class TestGlobalEventBusSingleton:
    """Test the global event bus singleton pattern."""

    def test_get_event_bus_singleton(self):
        """Test that get_event_bus returns the same instance."""
        bus1 = get_event_bus()
        bus2 = get_event_bus()

        assert bus1 is bus2


class TestEventTypes:
    """Test event type enumeration."""

    def test_all_event_types_exist(self):
        """Test that all expected event types are defined."""
        expected_types = [
            "LIBRARY_CREATED",
            "LIBRARY_DELETED",
            "LIBRARY_UPDATED",
            "DOCUMENT_ADDED",
            "DOCUMENT_UPDATED",
            "DOCUMENT_DELETED",
            "CHUNK_ADDED",
            "CHUNK_DELETED",
            "INDEX_REBUILT",
            "INDEX_OPTIMIZED",
            "BATCH_OPERATION_STARTED",
            "BATCH_OPERATION_COMPLETED",
            "BATCH_OPERATION_FAILED",
        ]

        for event_type_name in expected_types:
            assert hasattr(EventType, event_type_name)

    def test_event_type_values(self):
        """Test that event type values follow naming convention."""
        assert EventType.LIBRARY_CREATED.value == "library.created"
        assert EventType.DOCUMENT_ADDED.value == "document.added"
        assert EventType.INDEX_REBUILT.value == "index.rebuilt"
        assert EventType.BATCH_OPERATION_STARTED.value == "batch.started"
