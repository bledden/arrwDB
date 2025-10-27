# Streaming & Real-Time Updates Design

**Date**: 2025-10-26
**Status**: Design Phase
**Goal**: Enable streaming data ingestion, real-time index updates, and change data capture

---

## Current State

### ✅ What Already Works (Real-Time!)

**Incremental Index Updates**:
- HNSW, LSH, BruteForce all support `add_vector()` with **immediate** effect
- No rebuild needed - vectors are searchable instantly
- Only KD-Tree requires periodic rebuilds (and it auto-rebuilds when threshold is hit)

**Evidence**:
```python
# infrastructure/repositories/library_repository.py:298
index.add_vector(chunk.id, vector_index)  # Immediate, no rebuild
```

**Batch Operations**:
- 100-1000x faster than individual operations
- Single lock acquisition
- Batched embedding generation

### ❌ What's Missing (Actual Gaps)

1. **Async/Streaming Endpoints**
   - Current: All endpoints are synchronous (blocking)
   - Gap: Can't stream large datasets without blocking
   - Impact: Large ingestion jobs block API

2. **Change Data Capture (CDC)**
   - Current: No events published when data changes
   - Gap: Apps can't subscribe to updates
   - Impact: No real-time notifications, polling required

3. **WebSocket Support**
   - Current: HTTP-only (request-response)
   - Gap: No bidirectional streaming
   - Impact: Can't push updates to clients

4. **Background Job Queue**
   - Current: All operations run synchronously
   - Gap: Heavy operations block API responses
   - Impact: Poor UX for large operations

---

## Design: Streaming & Real-Time Architecture

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                         Client App                           │
└──────┬──────────────────────────────────────────┬───────────┘
       │                                           │
       │ HTTP/REST                                 │ WebSocket
       │                                           │
┌──────▼──────────────────────────────────────────▼───────────┐
│                      FastAPI Server                          │
│  ┌────────────────┐  ┌────────────────┐  ┌───────────────┐  │
│  │  Sync Routes   │  │  Async Routes  │  │   WebSocket   │  │
│  │  (existing)    │  │    (NEW)       │  │   Handler     │  │
│  └────────────────┘  └────────┬───────┘  └───────┬───────┘  │
└──────────────────────────────┬┼───────────────────┼──────────┘
                               ││                   │
                        ┌──────▼▼──────┐    ┌───────▼──────┐
                        │  Background   │    │   Event      │
                        │  Job Queue    │    │   Bus        │
                        │  (asyncio)    │    │  (pub/sub)   │
                        └──────┬────────┘    └───────┬──────┘
                               │                     │
                        ┌──────▼─────────────────────▼──────┐
                        │     Library Repository            │
                        │     (Vector Store + Index)        │
                        └───────────────────────────────────┘
```

---

## Implementation Plan

### Phase 1: Async Streaming Endpoints (Days 1-2)

**Goal**: Allow clients to stream large datasets without blocking

**New Endpoints**:

1. **POST `/v1/libraries/{id}/documents/stream`** - Streaming document ingestion
   ```python
   @router.post("/libraries/{id}/documents/stream")
   async def stream_documents(
       library_id: UUID,
       request: Request,
       service: LibraryService = Depends(get_library_service)
   ):
       """
       Stream documents via chunked transfer encoding.

       Accepts newline-delimited JSON (NDJSON):
       {"title": "Doc 1", "texts": ["chunk1", "chunk2"]}
       {"title": "Doc 2", "texts": ["chunk1", "chunk2"]}
       ...

       Returns streaming response with status updates.
       """
       async def process_stream():
           async for line in request.stream():
               doc_data = json.loads(line)
               # Process in background
               yield json.dumps({"status": "processing", "title": doc_data["title"]})
               result = await asyncio.to_thread(
                   service.add_document_with_text,
                   library_id,
                   **doc_data
               )
               yield json.dumps({"status": "completed", "id": str(result.id)})

       return StreamingResponse(
           process_stream(),
           media_type="application/x-ndjson"
       )
   ```

2. **POST `/v1/libraries/{id}/search/stream`** - Streaming search results
   ```python
   @router.post("/libraries/{id}/search/stream")
   async def stream_search(
       library_id: UUID,
       query: str,
       k: int = 10,
       service: LibraryService = Depends(get_library_service)
   ):
       """
       Return search results as they're found (for large k).

       Returns NDJSON stream of results.
       """
       async def result_stream():
           results = await asyncio.to_thread(
               service.search,
               library_id,
               query,
               k=k
           )
           for result in results:
               yield json.dumps({
                   "id": str(result["chunk_id"]),
                   "score": result["score"],
                   "text": result["text"]
               }) + "\n"

       return StreamingResponse(
           result_stream(),
           media_type="application/x-ndjson"
       )
   ```

**Implementation**:
- Create `app/api/streaming.py` module
- Use FastAPI `StreamingResponse`
- Leverage `asyncio.to_thread()` to run sync code async
- Support NDJSON format (newline-delimited JSON)

---

### Phase 2: Change Data Capture (CDC) (Days 3-4)

**Goal**: Publish events when data changes

**Event Bus Design**:

```python
# app/events/bus.py
from typing import Callable, Dict, List
from dataclasses import dataclass
from enum import Enum
import asyncio

class EventType(Enum):
    LIBRARY_CREATED = "library.created"
    LIBRARY_DELETED = "library.deleted"
    DOCUMENT_ADDED = "document.added"
    DOCUMENT_UPDATED = "document.updated"
    DOCUMENT_DELETED = "document.deleted"
    CHUNK_ADDED = "chunk.added"
    CHUNK_DELETED = "chunk.deleted"
    INDEX_REBUILT = "index.rebuilt"

@dataclass
class Event:
    type: EventType
    library_id: UUID
    data: dict
    timestamp: datetime

class EventBus:
    """
    Simple in-memory event bus for pub/sub.

    For production: Use Redis Streams, Kafka, or RabbitMQ.
    """

    def __init__(self):
        self._subscribers: Dict[EventType, List[Callable]] = {}
        self._queue: asyncio.Queue = asyncio.Queue()

    def subscribe(self, event_type: EventType, callback: Callable):
        """Subscribe to events of a specific type."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)

    async def publish(self, event: Event):
        """Publish an event to all subscribers."""
        await self._queue.put(event)

    async def start(self):
        """Start processing events."""
        while True:
            event = await self._queue.get()
            callbacks = self._subscribers.get(event.type, [])
            for callback in callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(event)
                    else:
                        await asyncio.to_thread(callback, event)
                except Exception as e:
                    logger.error(f"Error processing event {event.type}: {e}")
```

**Integration with Repository**:

```python
# infrastructure/repositories/library_repository.py
class LibraryRepository:
    def __init__(self, event_bus: Optional[EventBus] = None):
        # ... existing init ...
        self._event_bus = event_bus

    def add_document(self, library_id: UUID, document: Document) -> Document:
        with self._lock.write():
            # ... existing logic ...

            # Publish event
            if self._event_bus:
                asyncio.create_task(self._event_bus.publish(Event(
                    type=EventType.DOCUMENT_ADDED,
                    library_id=library_id,
                    data={
                        "document_id": str(document.id),
                        "num_chunks": len(document.chunks)
                    },
                    timestamp=datetime.utcnow()
                )))

            return document
```

**New Endpoints**:

1. **GET `/v1/events/stream`** - SSE (Server-Sent Events)
   ```python
   @router.get("/events/stream")
   async def stream_events(
       library_id: Optional[UUID] = None,
       event_bus: EventBus = Depends(get_event_bus)
   ):
       """
       Stream events via Server-Sent Events (SSE).

       Clients can listen to real-time changes:
       - Document additions
       - Index rebuilds
       - Deletions
       """
       async def event_generator():
           queue = asyncio.Queue()

           def callback(event: Event):
               if library_id is None or event.library_id == library_id:
                   asyncio.create_task(queue.put(event))

           # Subscribe to all event types
           for event_type in EventType:
               event_bus.subscribe(event_type, callback)

           while True:
               event = await queue.get()
               yield {
                   "event": event.type.value,
                   "data": json.dumps({
                       "library_id": str(event.library_id),
                       **event.data
                   })
               }

       return EventSourceResponse(event_generator())
   ```

---

### Phase 3: WebSocket Support (Days 5-6)

**Goal**: Bidirectional real-time communication

**WebSocket Endpoints**:

1. **WS `/v1/libraries/{id}/ws`** - Library operations via WebSocket
   ```python
   @router.websocket("/libraries/{library_id}/ws")
   async def library_websocket(
       websocket: WebSocket,
       library_id: UUID,
       service: LibraryService = Depends(get_library_service)
   ):
       """
       WebSocket for real-time library operations.

       Commands:
       - {"action": "add_document", "data": {...}}
       - {"action": "search", "query": "...", "k": 10}
       - {"action": "subscribe", "events": ["document.added"]}
       """
       await websocket.accept()

       try:
           while True:
               # Receive command
               data = await websocket.receive_json()
               action = data.get("action")

               if action == "add_document":
                   # Add document
                   result = await asyncio.to_thread(
                       service.add_document_with_text,
                       library_id,
                       **data["data"]
                   )
                   await websocket.send_json({
                       "status": "success",
                       "id": str(result.id)
                   })

               elif action == "search":
                   # Search and stream results
                   results = await asyncio.to_thread(
                       service.search,
                       library_id,
                       data["query"],
                       k=data.get("k", 10)
                   )
                   await websocket.send_json({
                       "status": "success",
                       "results": results
                   })

               elif action == "subscribe":
                   # Subscribe to events (handled by event bus)
                   pass

       except WebSocketDisconnect:
           logger.info(f"WebSocket disconnected for library {library_id}")
   ```

**Use Cases**:
- Real-time collaborative search
- Live document ingestion with progress
- Event subscriptions (like database triggers)

---

### Phase 4: Background Job Queue (Days 7-8)

**Goal**: Offload heavy operations to background workers

**Job Queue Design**:

```python
# app/jobs/queue.py
from typing import Callable, Any
from dataclasses import dataclass
from uuid import UUID, uuid4
import asyncio

@dataclass
class Job:
    id: UUID
    task: Callable
    args: tuple
    kwargs: dict
    status: str = "pending"  # pending, running, completed, failed
    result: Any = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

class JobQueue:
    """
    Simple async job queue.

    For production: Use Celery, RQ, or Dramatiq.
    """

    def __init__(self, num_workers: int = 4):
        self._queue: asyncio.Queue = asyncio.Queue()
        self._jobs: Dict[UUID, Job] = {}
        self._num_workers = num_workers
        self._workers: List[asyncio.Task] = []

    async def submit(self, task: Callable, *args, **kwargs) -> UUID:
        """Submit a job to the queue."""
        job = Job(
            id=uuid4(),
            task=task,
            args=args,
            kwargs=kwargs
        )
        self._jobs[job.id] = job
        await self._queue.put(job)
        return job.id

    async def get_status(self, job_id: UUID) -> Job:
        """Get job status."""
        return self._jobs.get(job_id)

    async def _worker(self):
        """Worker that processes jobs."""
        while True:
            job = await self._queue.get()
            job.status = "running"

            try:
                if asyncio.iscoroutinefunction(job.task):
                    result = await job.task(*job.args, **job.kwargs)
                else:
                    result = await asyncio.to_thread(job.task, *job.args, **job.kwargs)

                job.status = "completed"
                job.result = result
                job.completed_at = datetime.utcnow()

            except Exception as e:
                job.status = "failed"
                job.error = str(e)
                job.completed_at = datetime.utcnow()
                logger.error(f"Job {job.id} failed: {e}")

    async def start(self):
        """Start worker pool."""
        self._workers = [
            asyncio.create_task(self._worker())
            for _ in range(self._num_workers)
        ]
```

**New Endpoints**:

1. **POST `/v1/libraries/{id}/documents/batch-async`** - Async batch ingestion
   ```python
   @router.post("/libraries/{id}/documents/batch-async")
   async def batch_documents_async(
       library_id: UUID,
       request: BatchDocumentsRequest,
       service: LibraryService = Depends(get_library_service),
       job_queue: JobQueue = Depends(get_job_queue)
   ):
       """
       Submit batch ingestion job and return immediately.

       Returns job ID for status polling.
       """
       job_id = await job_queue.submit(
           service.add_documents_batch,
           library_id,
           request.documents
       )

       return {
           "job_id": str(job_id),
           "status": "submitted",
           "status_url": f"/v1/jobs/{job_id}"
       }

   @router.get("/jobs/{job_id}")
   async def get_job_status(
       job_id: UUID,
       job_queue: JobQueue = Depends(get_job_queue)
   ):
       """Get job status and results."""
       job = await job_queue.get_status(job_id)
       if not job:
           raise HTTPException(404, "Job not found")

       return {
           "job_id": str(job.id),
           "status": job.status,
           "created_at": job.created_at.isoformat(),
           "completed_at": job.completed_at.isoformat() if job.completed_at else None,
           "result": job.result,
           "error": job.error
       }
   ```

---

## Technology Stack

### In-Memory (MVP - Simple)
- **Event Bus**: In-process `asyncio.Queue`
- **Job Queue**: In-process `asyncio.Queue` + worker pool
- **Limitations**: Single process only, no persistence

### Production (Scalable)
- **Event Bus**: Redis Streams, Kafka, or RabbitMQ
- **Job Queue**: Celery (Redis backend) or Dramatiq
- **Benefits**: Multi-process, persistence, distributed

---

## API Examples

### Streaming Ingestion (NDJSON)

```bash
# Stream documents via curl
cat documents.ndjson | curl -X POST \
  http://localhost:8000/v1/libraries/{id}/documents/stream \
  -H "Content-Type: application/x-ndjson" \
  --data-binary @-
```

### Server-Sent Events (SSE)

```javascript
// JavaScript client for real-time events
const eventSource = new EventSource('/v1/events/stream?library_id=...');

eventSource.addEventListener('document.added', (e) => {
  const data = JSON.parse(e.data);
  console.log('New document:', data.document_id);
});

eventSource.addEventListener('index.rebuilt', (e) => {
  console.log('Index rebuilt!');
});
```

### WebSocket

```javascript
// JavaScript client for WebSocket
const ws = new WebSocket('ws://localhost:8000/v1/libraries/{id}/ws');

// Add document
ws.send(JSON.stringify({
  action: 'add_document',
  data: {
    title: 'New Doc',
    texts: ['chunk 1', 'chunk 2']
  }
}));

// Receive confirmation
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Document added:', data.id);
};

// Search
ws.send(JSON.stringify({
  action: 'search',
  query: 'machine learning',
  k: 10
}));
```

### Background Jobs

```bash
# Submit large batch job
curl -X POST http://localhost:8000/v1/libraries/{id}/documents/batch-async \
  -H "Content-Type: application/json" \
  -d '{"documents": [...1000 documents...]}'

# Response: {"job_id": "...", "status_url": "/v1/jobs/..."}

# Poll job status
curl http://localhost:8000/v1/jobs/{job_id}

# Response: {"status": "completed", "result": {...}}
```

---

## Performance Impact

### Memory
- **Event Bus**: ~10 MB per 10K events in queue (negligible)
- **Job Queue**: ~50 MB per 1K pending jobs (manageable)

### CPU
- **Async overhead**: <5% (worth it for scalability)
- **Event processing**: <1ms per event

### Latency
- **Streaming ingestion**: 10-50ms per document (vs 100-500ms blocking)
- **WebSocket**: <1ms round-trip (vs 10-100ms HTTP)
- **Background jobs**: Instant response (vs minutes blocking)

---

## Migration Path

### Phase 1: Add Async Endpoints (Days 1-2)
- Coexist with existing sync endpoints
- No breaking changes
- Gradual adoption

### Phase 2: Add Events (Days 3-4)
- Optional event bus
- Backward compatible
- Enable gradually per library

### Phase 3: Add WebSocket (Days 5-6)
- New protocol, doesn't affect REST
- Opt-in for clients

### Phase 4: Background Jobs (Days 7-8)
- New `-async` endpoints
- Existing endpoints unchanged

---

## Testing Strategy

### Unit Tests
- Event bus publish/subscribe
- Job queue submit/status
- WebSocket message handling

### Integration Tests
- Stream 10K documents, verify all indexed
- Subscribe to events, verify delivery
- Submit background job, poll until complete

### Load Tests
- 1000 concurrent WebSocket connections
- 10K events/second through event bus
- 100 background jobs in parallel

---

## Future Enhancements

1. **Event Persistence** - Store events in WAL for replay
2. **Event Filtering** - Subscribe to specific document types
3. **Batch Event Publishing** - Reduce overhead for bulk operations
4. **Job Priorities** - High-priority jobs first
5. **Job Scheduling** - Cron-like scheduled jobs
6. **Dead Letter Queue** - Failed jobs for manual retry

---

## Summary

**Current State**: Indexes ALREADY support real-time updates! No rebuild needed (except KD-Tree).

**What We're Adding**:
1. **Async/Streaming Endpoints** - Non-blocking ingestion
2. **Change Data Capture** - Event notifications
3. **WebSocket Support** - Bidirectional streaming
4. **Background Jobs** - Async heavy operations

**Impact**:
- ✅ Better UX (no blocking on large operations)
- ✅ Real-time notifications (SSE, WebSocket)
- ✅ Scalability (background workers)
- ✅ Competitive with Qdrant/Pinecone real-time features

**Effort**: 7-8 days (MVP with in-memory event bus/job queue)

**Future**: Swap to Redis/Kafka for production scale
