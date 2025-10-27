# Event Bus & Change Data Capture (CDC) Guide

Complete guide to arrwDB's event system: real-time event bus, change data capture, and pub/sub patterns for reactive applications.

---

## Table of Contents

1. [Overview](#overview)
2. [Event Types](#event-types)
3. [Event Bus Architecture](#event-bus-architecture)
4. [Subscribing to Events](#subscribing-to-events)
5. [Publishing Events](#publishing-events)
6. [Monitoring & Statistics](#monitoring--statistics)
7. [Use Cases](#use-cases)
8. [Best Practices](#best-practices)

---

## Overview

The arrwDB Event Bus provides a lightweight pub/sub system for change data capture (CDC). It allows applications to react to data changes in real-time without polling.

### Key Features

- **Asynchronous Event Processing** - Non-blocking event delivery
- **Type-Safe Events** - Pydantic models for event validation
- **Automatic CDC** - Library and document changes trigger events automatically
- **WebSocket Integration** - Events forwarded to connected WebSocket clients
- **Statistics & Monitoring** - Built-in metrics for event throughput

### Performance

- **Throughput**: 4-8 events/second
- **Delivery Latency**: <10ms
- **Queue Capacity**: Unlimited (memory-bound)
- **Workers**: 1 async worker (expandable)

---

## Event Types

### Library Events

#### `library.created`
Triggered when a new library is created.

```python
{
  "type": "library.created",
  "library_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2025-10-27T06:00:00.000000",
  "data": {
    "name": "My Library",
    "index_type": "hnsw",
    "embedding_dimension": 1024
  }
}
```

#### `library.deleted`
Triggered when a library is deleted.

```python
{
  "type": "library.deleted",
  "library_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2025-10-27T06:05:00.000000",
  "data": {
    "name": "My Library"
  }
}
```

### Document Events

#### `document.added`
Triggered when a document is added to a library.

```python
{
  "type": "document.added",
  "library_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2025-10-27T06:01:00.000000",
  "data": {
    "document_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "title": "My Document",
    "num_chunks": 3
  }
}
```

#### `document.deleted`
Triggered when a document is deleted from a library.

```python
{
  "type": "document.deleted",
  "library_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2025-10-27T06:03:00.000000",
  "data": {
    "document_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
  }
}
```

### Search Events

#### `search.performed`
Triggered when a search is executed (optional, can be noisy).

```python
{
  "type": "search.performed",
  "library_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2025-10-27T06:02:00.000000",
  "data": {
    "query": "machine learning",
    "k": 10,
    "results_count": 8,
    "duration_ms": 145
  }
}
```

---

## Event Bus Architecture

### Components

```
┌─────────────────────────────────────────────────────────┐
│                      Event Publishers                    │
│  (LibraryRepository, DocumentService, SearchService)    │
└────────────────────┬────────────────────────────────────┘
                     │
                     │ publish()
                     ▼
┌─────────────────────────────────────────────────────────┐
│                     Event Bus                            │
│  - Queue: asyncio.Queue                                 │
│  - Worker: async task processing events                │
│  - Subscribers: Dict[str, List[Callable]]               │
└────────────────────┬────────────────────────────────────┘
                     │
                     │ deliver()
                     ▼
┌─────────────────────────────────────────────────────────┐
│                   Event Subscribers                      │
│  - WebSocket Manager (forwards to clients)              │
│  - Custom Handlers (application-specific logic)         │
│  - Analytics Services (tracking, logging)               │
└─────────────────────────────────────────────────────────┘
```

### Event Flow

1. **Action Occurs** - User creates document via API
2. **Event Published** - Repository calls `event_bus.publish(event)`
3. **Event Queued** - Event added to async queue
4. **Worker Processes** - Background worker picks up event
5. **Subscribers Notified** - All registered subscribers receive event
6. **WebSocket Delivery** - Connected clients receive event notification

### Thread Safety

The Event Bus uses:
- **asyncio.Queue** for thread-safe event queuing
- **Event loop stored at startup** for cross-thread publishing
- **asyncio.run_coroutine_threadsafe()** for sync→async event publishing

---

## Subscribing to Events

### Method 1: Direct Subscription

Subscribe to all events:

```python
from app.events.bus import get_event_bus

async def my_event_handler(event):
    """Handle all events."""
    print(f"Received event: {event.type} for library {event.library_id}")

# Subscribe
event_bus = get_event_bus()
await event_bus.subscribe(my_event_handler)
```

Subscribe to specific event types:

```python
async def document_added_handler(event):
    """Handle only document.added events."""
    if event.type == EventType.DOCUMENT_ADDED:
        doc_id = event.data.get("document_id")
        print(f"New document added: {doc_id}")

await event_bus.subscribe(document_added_handler, event_type=EventType.DOCUMENT_ADDED)
```

### Method 2: WebSocket Subscription

Events are automatically forwarded to WebSocket clients connected to a library:

```javascript
// Connect to library WebSocket
const ws = new WebSocket(`ws://localhost:8000/v1/libraries/${libraryId}/ws`);

// Receive events
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);

    if (data.type === 'event') {
        console.log('Event received:', data.event_type);
        handleEvent(data);
    }
};
```

### Method 3: Custom Service Integration

```python
class AnalyticsService:
    """Track library usage via events."""

    def __init__(self, event_bus):
        self.event_bus = event_bus
        self.stats = {
            "documents_added": 0,
            "searches_performed": 0
        }

    async def start(self):
        """Subscribe to events."""
        await self.event_bus.subscribe(self.handle_event)

    async def handle_event(self, event):
        """Process events for analytics."""
        if event.type == EventType.DOCUMENT_ADDED:
            self.stats["documents_added"] += 1

        elif event.type == EventType.SEARCH_PERFORMED:
            self.stats["searches_performed"] += 1

        # Store to database, send to analytics service, etc.
```

---

## Publishing Events

### Automatic Publishing

Events are automatically published by the system for:
- Library creation/deletion
- Document addition/deletion
- Search operations

No manual publishing needed for standard operations.

### Manual Event Publishing

For custom events or application-specific changes:

```python
from app.events.bus import get_event_bus, Event, EventType
from datetime import datetime

async def custom_operation(library_id):
    """Perform custom operation and publish event."""
    event_bus = get_event_bus()

    # Do work...
    result = perform_work()

    # Publish custom event
    event = Event(
        type=EventType.CUSTOM,  # Or any event type
        library_id=library_id,
        data={"result": result, "status": "completed"},
        timestamp=datetime.utcnow()
    )

    await event_bus.publish(event)
```

### Publishing from Sync Code

The Event Bus supports publishing from synchronous code (e.g., repository methods):

```python
from app.events.bus import Event, EventType
import asyncio

def sync_operation(self, library_id):
    """Sync method that publishes events."""
    # Do sync work...

    # Publish event using stored event loop
    event = Event(
        type=EventType.DOCUMENT_ADDED,
        library_id=library_id,
        data={"document_id": str(doc_id)},
        timestamp=datetime.utcnow()
    )

    # Get event loop from event bus
    event_loop = self._event_bus._loop
    if event_loop and not event_loop.is_closed():
        future = asyncio.run_coroutine_threadsafe(
            self._event_bus.publish(event),
            event_loop
        )
        # Optional: wait for completion
        # future.result(timeout=5)
```

---

## Monitoring & Statistics

### Get Event Bus Statistics

```bash
curl http://localhost:8000/v1/events/stats
```

Response:
```json
{
  "total_published": 142,
  "total_delivered": 142,
  "active_subscribers": 3,
  "queue_size": 0,
  "events_by_type": {
    "library.created": 5,
    "library.deleted": 2,
    "document.added": 85,
    "document.deleted": 12,
    "search.performed": 38
  },
  "average_delivery_latency_ms": 8.5,
  "uptime_seconds": 3600
}
```

### Statistics Fields

- **total_published**: Total events published since startup
- **total_delivered**: Total events delivered to subscribers
- **active_subscribers**: Number of active event subscribers
- **queue_size**: Current event queue depth (0 = all processed)
- **events_by_type**: Breakdown by event type
- **average_delivery_latency_ms**: Average time from publish to delivery
- **uptime_seconds**: Event bus uptime

### Programmatic Access

```python
from app.events.bus import get_event_bus

event_bus = get_event_bus()
stats = event_bus.get_statistics()

print(f"Published: {stats['total_published']}")
print(f"Queue size: {stats['queue_size']}")
print(f"Subscribers: {stats['active_subscribers']}")
```

### Health Checks

Monitor event bus health:

```python
def is_event_bus_healthy():
    """Check if event bus is healthy."""
    stats = event_bus.get_statistics()

    # Check queue isn't backing up
    if stats['queue_size'] > 1000:
        return False

    # Check events are being delivered
    if stats['total_published'] > 0 and stats['total_delivered'] == 0:
        return False

    # Check worker is running
    if not event_bus._running:
        return False

    return True
```

---

## Use Cases

### 1. Real-Time Dashboard

Update dashboard in real-time as documents are added:

```javascript
class LibraryDashboard {
    constructor(libraryId) {
        this.libraryId = libraryId;
        this.documentCount = 0;
        this.connectWebSocket();
    }

    connectWebSocket() {
        this.ws = new WebSocket(`ws://localhost:8000/v1/libraries/${this.libraryId}/ws`);

        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);

            if (data.type === 'event') {
                this.handleEvent(data);
            }
        };
    }

    handleEvent(eventData) {
        switch(eventData.event_type) {
            case 'document.added':
                this.documentCount++;
                this.updateUI();
                break;

            case 'document.deleted':
                this.documentCount--;
                this.updateUI();
                break;

            case 'search.performed':
                this.recordSearchActivity(eventData);
                break;
        }
    }

    updateUI() {
        document.getElementById('doc-count').textContent = this.documentCount;
    }
}
```

### 2. Audit Logging

Log all changes to documents for compliance:

```python
class AuditLogger:
    """Log all library changes for audit trail."""

    def __init__(self, event_bus, db_connection):
        self.event_bus = event_bus
        self.db = db_connection

    async def start(self):
        await self.event_bus.subscribe(self.log_event)

    async def log_event(self, event):
        """Write event to audit log."""
        await self.db.insert_audit_log({
            "timestamp": event.timestamp,
            "event_type": event.type.value,
            "library_id": str(event.library_id),
            "user_id": event.data.get("user_id"),
            "details": event.data
        })
```

### 3. Cache Invalidation

Invalidate caches when data changes:

```python
class CacheManager:
    """Invalidate caches on data changes."""

    def __init__(self, event_bus, cache):
        self.event_bus = event_bus
        self.cache = cache

    async def start(self):
        await self.event_bus.subscribe(self.handle_event)

    async def handle_event(self, event):
        """Invalidate cache on relevant events."""
        if event.type in [EventType.DOCUMENT_ADDED, EventType.DOCUMENT_DELETED]:
            # Invalidate search cache for this library
            cache_key = f"search_cache:{event.library_id}"
            await self.cache.delete(cache_key)

        elif event.type == EventType.LIBRARY_DELETED:
            # Invalidate all caches for this library
            pattern = f"*:{event.library_id}:*"
            await self.cache.delete_pattern(pattern)
```

### 4. Notification System

Send notifications to users:

```python
class NotificationService:
    """Send notifications for important events."""

    async def handle_event(self, event):
        """Send notifications based on event type."""
        if event.type == EventType.DOCUMENT_ADDED:
            # Notify library owner
            await self.send_notification(
                user_id=event.data.get("owner_id"),
                message=f"New document added: {event.data['title']}"
            )

        elif event.type == EventType.SEARCH_PERFORMED:
            # Track search activity for recommendations
            await self.update_user_interests(
                user_id=event.data.get("user_id"),
                query=event.data.get("query")
            )
```

### 5. Analytics & Metrics

Track usage patterns:

```python
class MetricsCollector:
    """Collect metrics from events."""

    def __init__(self, event_bus, metrics_client):
        self.event_bus = event_bus
        self.metrics = metrics_client

    async def start(self):
        await self.event_bus.subscribe(self.collect_metrics)

    async def collect_metrics(self, event):
        """Send metrics to monitoring system."""
        # Increment event counter
        self.metrics.increment(f"events.{event.type.value}")

        # Track latency for searches
        if event.type == EventType.SEARCH_PERFORMED:
            duration = event.data.get("duration_ms")
            self.metrics.histogram("search.latency", duration)

        # Track document growth
        if event.type == EventType.DOCUMENT_ADDED:
            self.metrics.increment(f"documents.added.{event.library_id}")
```

### 6. Workflow Triggers

Trigger workflows on specific events:

```python
class WorkflowEngine:
    """Trigger workflows based on events."""

    async def handle_event(self, event):
        """Execute workflows for events."""
        if event.type == EventType.DOCUMENT_ADDED:
            # Auto-tag documents
            await self.trigger_workflow("auto_tagging", event.data)

            # Extract entities
            await self.trigger_workflow("entity_extraction", event.data)

            # Generate summary
            await self.trigger_workflow("summarization", event.data)

        elif event.type == EventType.LIBRARY_CREATED:
            # Initialize library with templates
            await self.trigger_workflow("library_setup", event.data)
```

---

## Best Practices

### 1. Event Handler Performance

Keep handlers lightweight:

```python
# Good: Quick processing, async operations
async def good_handler(event):
    """Lightweight, non-blocking handler."""
    await log_to_queue(event)  # Queue for background processing
    await update_counter(event.type)

# Bad: Heavy processing blocks event delivery
async def bad_handler(event):
    """Heavy processing blocks event bus."""
    # DON'T: Expensive operations in handler
    await expensive_ai_processing(event.data)
    await slow_database_query()
```

### 2. Error Handling

Handle errors gracefully:

```python
async def resilient_handler(event):
    """Handler with error handling."""
    try:
        await process_event(event)
    except Exception as e:
        logger.error(f"Error processing event {event.type}: {e}")
        # Don't re-raise - prevents blocking other subscribers
        await send_to_dead_letter_queue(event, error=str(e))
```

### 3. Idempotency

Handle duplicate events:

```python
async def idempotent_handler(event):
    """Handler that safely handles duplicates."""
    event_id = f"{event.type}:{event.library_id}:{event.timestamp}"

    # Check if already processed
    if await cache.exists(event_id):
        logger.debug(f"Skipping duplicate event: {event_id}")
        return

    # Process event
    await process_event(event)

    # Mark as processed
    await cache.set(event_id, "processed", expire=3600)
```

### 4. Selective Subscription

Only subscribe to relevant events:

```python
# Good: Specific event types
async def document_handler(event):
    """Handle only document events."""
    if event.type not in [EventType.DOCUMENT_ADDED, EventType.DOCUMENT_DELETED]:
        return

    await process_document_event(event)

# Better: Filter at subscription
await event_bus.subscribe(
    handler,
    event_types=[EventType.DOCUMENT_ADDED, EventType.DOCUMENT_DELETED]
)
```

### 5. Resource Cleanup

Unsubscribe when done:

```python
class EventListener:
    """Properly managed event listener."""

    async def start(self):
        """Start listening."""
        self.handler_id = await event_bus.subscribe(self.handle_event)

    async def stop(self):
        """Stop listening and cleanup."""
        if self.handler_id:
            await event_bus.unsubscribe(self.handler_id)

    async def handle_event(self, event):
        """Process events."""
        pass
```

### 6. Monitoring

Monitor event bus health:

```python
async def monitor_event_bus():
    """Periodic health check."""
    while True:
        stats = event_bus.get_statistics()

        # Alert if queue is backing up
        if stats['queue_size'] > 1000:
            await send_alert("Event bus queue backing up")

        # Alert if no events in last hour (might indicate issue)
        if stats['total_published'] == last_check_count:
            await send_alert("No events published in last check")

        last_check_count = stats['total_published']
        await asyncio.sleep(300)  # Check every 5 minutes
```

---

## Troubleshooting

### Issue: Events Not Being Delivered

**Symptoms**: Published events don't reach subscribers

**Solutions**:
- Check event bus is started: `event_bus._running == True`
- Verify subscribers registered: `event_bus.get_statistics()['active_subscribers'] > 0`
- Check worker task is running: `event_bus._worker_task` exists
- Look for errors in event handler logs

### Issue: Event Queue Backing Up

**Symptoms**: `queue_size` keeps growing

**Solutions**:
- Check for slow event handlers (add timeouts)
- Increase worker count if needed
- Move heavy processing to background jobs
- Check for errors preventing event processing

### Issue: Duplicate Events

**Symptoms**: Same event received multiple times

**Solutions**:
- Implement idempotency in handlers
- Use event IDs for deduplication
- Check for multiple subscriber registrations

### Issue: Events Missing After Restart

**Symptoms**: Events lost on server restart

**Expected Behavior**: Events are in-memory only and won't persist across restarts. For durable events, implement event sourcing or use external message queue.

---

## Advanced Topics

### Event Sourcing

For durable event history:

```python
class EventStore:
    """Persist events for event sourcing."""

    async def handle_event(self, event):
        """Store event to database."""
        await self.db.insert_event({
            "id": str(uuid4()),
            "type": event.type.value,
            "library_id": str(event.library_id),
            "timestamp": event.timestamp,
            "data": event.data
        })

    async def replay_events(self, library_id, from_timestamp=None):
        """Replay events to rebuild state."""
        events = await self.db.query_events(library_id, from_timestamp)
        for event in events:
            await self.process_event(event)
```

### Event Filtering

Filter events before delivery:

```python
class EventFilter:
    """Filter events based on criteria."""

    def __init__(self, event_bus):
        self.event_bus = event_bus

    async def subscribe_filtered(self, handler, filter_fn):
        """Subscribe with custom filter."""
        async def filtered_handler(event):
            if filter_fn(event):
                await handler(event)

        await self.event_bus.subscribe(filtered_handler)

# Usage
await filter.subscribe_filtered(
    my_handler,
    lambda e: e.data.get("priority") == "high"
)
```

---

## Next Steps

- [Streaming & WebSocket Guide](STREAMING_WEBSOCKET_GUIDE.md) - Real-time data ingestion
- [API Reference](API_GUIDE.md) - Complete REST API documentation
- [Deployment Guide](DEPLOYMENT.md) - Production deployment

---

**Last Updated**: October 27, 2025
