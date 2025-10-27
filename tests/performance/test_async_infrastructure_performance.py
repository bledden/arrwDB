"""
Performance tests for async infrastructure components.

This module tests performance characteristics of:
- Event Bus throughput and latency
- Job Queue processing capacity
- WebSocket Manager scalability
- Concurrent operations

Performance targets:
- Event Bus: >10,000 events/sec
- Job Queue: >1,000 jobs/sec
- WebSocket: >500 concurrent connections
- Latency: <10ms for most operations
"""

import asyncio
import time
from statistics import mean, median, stdev
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from app.events.bus import Event, EventBus, EventType
from app.jobs.queue import Job, JobQueue, JobType
from app.websockets.manager import ConnectionManager


class TestEventBusPerformance:
    """Performance tests for Event Bus."""

    @pytest.mark.asyncio
    async def test_event_publishing_throughput(self):
        """Test event publishing throughput (target: >10,000/sec)."""
        event_bus = EventBus()
        await event_bus.start()

        library_id = uuid4()
        num_events = 10000

        start_time = time.time()

        # Publish events
        for i in range(num_events):
            event = Event(
                type=EventType.DOCUMENT_ADDED,
                library_id=library_id,
                data={"doc_id": f"doc{i}"},
            )
            await event_bus.publish(event)

        elapsed = time.time() - start_time
        throughput = num_events / elapsed

        await event_bus.stop()

        print(f"\n  Event publishing throughput: {throughput:.0f} events/sec")
        print(f"  Time for {num_events} events: {elapsed:.3f}s")

        # Target: >10,000 events/sec
        assert throughput > 10000, f"Throughput {throughput:.0f}/sec below target"

    @pytest.mark.asyncio
    async def test_event_delivery_throughput(self):
        """Test event delivery throughput with subscribers."""
        event_bus = EventBus()
        received_count = []

        async def fast_subscriber(event: Event):
            received_count.append(1)

        # Register multiple subscribers
        for _ in range(5):
            event_bus.subscribe(fast_subscriber, EventType.DOCUMENT_ADDED)

        await event_bus.start()

        library_id = uuid4()
        num_events = 5000

        start_time = time.time()

        for i in range(num_events):
            event = Event(
                type=EventType.DOCUMENT_ADDED,
                library_id=library_id,
                data={"doc_id": f"doc{i}"},
            )
            await event_bus.publish(event)

        # Wait for delivery
        await asyncio.sleep(1.0)
        await event_bus.stop()

        elapsed = time.time() - start_time
        delivery_rate = len(received_count) / elapsed

        print(f"\n  Event delivery rate: {delivery_rate:.0f} deliveries/sec")
        print(f"  Total deliveries: {len(received_count)} (5 subscribers × {num_events} events)")

        # Should deliver to all subscribers
        assert len(received_count) == num_events * 5

    @pytest.mark.asyncio
    async def test_event_latency(self):
        """Test event delivery latency (target: <10ms p99)."""
        event_bus = EventBus()
        latencies = []

        async def latency_subscriber(event: Event):
            receive_time = time.time()
            publish_time = event.data.get("publish_time")
            if publish_time:
                latency = (receive_time - publish_time) * 1000  # ms
                latencies.append(latency)

        event_bus.subscribe(latency_subscriber, EventType.DOCUMENT_ADDED)
        await event_bus.start()

        library_id = uuid4()
        num_events = 1000

        for i in range(num_events):
            event = Event(
                type=EventType.DOCUMENT_ADDED,
                library_id=library_id,
                data={"doc_id": f"doc{i}", "publish_time": time.time()},
            )
            await event_bus.publish(event)
            await asyncio.sleep(0.001)  # Small delay between publishes

        await asyncio.sleep(0.5)
        await event_bus.stop()

        if latencies:
            p50 = sorted(latencies)[len(latencies) // 2]
            p95 = sorted(latencies)[int(len(latencies) * 0.95)]
            p99 = sorted(latencies)[int(len(latencies) * 0.99)]
            avg = mean(latencies)

            print(f"\n  Event latency:")
            print(f"    Average: {avg:.2f}ms")
            print(f"    P50: {p50:.2f}ms")
            print(f"    P95: {p95:.2f}ms")
            print(f"    P99: {p99:.2f}ms")

            # Target: p99 < 10ms
            assert p99 < 10, f"P99 latency {p99:.2f}ms exceeds 10ms target"


class TestJobQueuePerformance:
    """Performance tests for Job Queue."""

    @pytest.mark.asyncio
    async def test_job_submission_throughput(self):
        """Test job submission throughput (target: >1,000/sec)."""
        job_queue = JobQueue(num_workers=4)

        num_jobs = 5000

        start_time = time.time()

        for i in range(num_jobs):
            await job_queue.submit(
                JobType.CUSTOM,
                {"index": i},
            )

        elapsed = time.time() - start_time
        throughput = num_jobs / elapsed

        print(f"\n  Job submission throughput: {throughput:.0f} jobs/sec")
        print(f"  Time for {num_jobs} jobs: {elapsed:.3f}s")

        # Target: >1,000 jobs/sec
        assert throughput > 1000, f"Throughput {throughput:.0f}/sec below target"

    @pytest.mark.asyncio
    async def test_job_execution_throughput(self):
        """Test job execution throughput with fast jobs."""
        job_queue = JobQueue(num_workers=8)
        completed_count = []

        async def fast_handler(job: Job):
            completed_count.append(1)
            return {"success": True}

        job_queue.register_handler(JobType.CUSTOM, fast_handler)
        await job_queue.start()

        num_jobs = 1000

        start_time = time.time()

        for i in range(num_jobs):
            await job_queue.submit(JobType.CUSTOM, {"index": i})

        # Wait for all jobs to complete
        max_wait = 10  # seconds
        waited = 0
        while len(completed_count) < num_jobs and waited < max_wait:
            await asyncio.sleep(0.1)
            waited += 0.1

        elapsed = time.time() - start_time
        await job_queue.stop()

        throughput = len(completed_count) / elapsed

        print(f"\n  Job execution throughput: {throughput:.0f} jobs/sec")
        print(f"  Completed: {len(completed_count)}/{num_jobs} in {elapsed:.3f}s")
        print(f"  Workers: 8")

        assert len(completed_count) == num_jobs, "Not all jobs completed"

    @pytest.mark.asyncio
    async def test_job_queue_scalability(self):
        """Test job queue scalability with different worker counts."""
        num_jobs = 500
        worker_counts = [1, 2, 4, 8]
        results = []

        async def work_handler(job: Job):
            await asyncio.sleep(0.01)  # Simulate 10ms of work
            return {"done": True}

        for num_workers in worker_counts:
            job_queue = JobQueue(num_workers=num_workers)
            job_queue.register_handler(JobType.CUSTOM, work_handler)
            await job_queue.start()

            completed = []

            async def track_handler(job: Job):
                result = await work_handler(job)
                completed.append(1)
                return result

            job_queue.register_handler(JobType.CUSTOM, track_handler)

            start_time = time.time()

            for i in range(num_jobs):
                await job_queue.submit(JobType.CUSTOM, {"index": i})

            # Wait for completion
            max_wait = 30
            waited = 0
            while len(completed) < num_jobs and waited < max_wait:
                await asyncio.sleep(0.1)
                waited += 0.1

            elapsed = time.time() - start_time
            await job_queue.stop()

            throughput = num_jobs / elapsed
            results.append((num_workers, throughput))

            print(f"\n  Workers: {num_workers}, Throughput: {throughput:.0f} jobs/sec")

        # Verify scaling improves throughput
        assert results[1][1] > results[0][1], "2 workers should be faster than 1"
        assert results[2][1] > results[1][1], "4 workers should be faster than 2"


class TestWebSocketManagerPerformance:
    """Performance tests for WebSocket Manager."""

    @pytest.mark.asyncio
    async def test_connection_throughput(self):
        """Test connection acceptance throughput."""
        manager = ConnectionManager()
        library_id = uuid4()
        num_connections = 1000

        start_time = time.time()

        for i in range(num_connections):
            ws = Mock()
            ws.accept = AsyncMock()
            ws.send_json = AsyncMock()
            await manager.connect(ws, library_id)

        elapsed = time.time() - start_time
        throughput = num_connections / elapsed

        print(f"\n  Connection throughput: {throughput:.0f} connections/sec")
        print(f"  Time for {num_connections} connections: {elapsed:.3f}s")

        assert len(manager._connections) == num_connections

    @pytest.mark.asyncio
    async def test_broadcast_performance(self):
        """Test broadcast performance with many connections."""
        manager = ConnectionManager()
        library_id = uuid4()
        num_connections = 500

        # Create connections
        for i in range(num_connections):
            ws = Mock()
            ws.accept = AsyncMock()
            ws.send_json = AsyncMock()
            await manager.connect(ws, library_id)

        message = {"type": "test", "data": "performance"}

        start_time = time.time()

        # Broadcast 100 messages
        num_broadcasts = 100
        for i in range(num_broadcasts):
            await manager.broadcast_to_library(library_id, message)

        elapsed = time.time() - start_time

        total_messages = num_broadcasts * num_connections
        throughput = total_messages / elapsed

        print(f"\n  Broadcast performance:")
        print(f"    Connections: {num_connections}")
        print(f"    Broadcasts: {num_broadcasts}")
        print(f"    Total messages: {total_messages}")
        print(f"    Throughput: {throughput:.0f} messages/sec")
        print(f"    Time: {elapsed:.3f}s")

    @pytest.mark.asyncio
    async def test_websocket_scalability(self):
        """Test WebSocket manager with increasing connection counts."""
        connection_counts = [100, 250, 500]
        results = []

        for num_connections in connection_counts:
            manager = ConnectionManager()
            library_id = uuid4()

            # Create connections
            for i in range(num_connections):
                ws = Mock()
                ws.accept = AsyncMock()
                ws.send_json = AsyncMock()
                await manager.connect(ws, library_id)

            message = {"type": "test"}

            # Measure broadcast time
            start_time = time.time()
            await manager.broadcast_to_library(library_id, message)
            elapsed = time.time() - start_time

            latency_per_connection = (elapsed * 1000) / num_connections  # ms

            results.append((num_connections, latency_per_connection))

            print(f"\n  Connections: {num_connections}")
            print(f"    Broadcast latency: {elapsed*1000:.2f}ms")
            print(f"    Per-connection: {latency_per_connection:.3f}ms")

        # Verify it scales reasonably (not exponentially)
        # Latency per connection should stay relatively constant
        assert results[-1][1] < results[0][1] * 3, "Scaling degrades too much"


class TestConcurrentOperations:
    """Test concurrent operations across components."""

    @pytest.mark.asyncio
    async def test_mixed_workload_performance(self):
        """Test mixed workload of events, jobs, and WebSockets."""
        # Setup
        event_bus = EventBus()
        job_queue = JobQueue(num_workers=4)
        ws_manager = ConnectionManager()

        event_count = []
        job_count = []

        async def event_handler(event: Event):
            event_count.append(1)

        async def job_handler(job: Job):
            job_count.append(1)
            return {"done": True}

        event_bus.subscribe(event_handler, EventType.DOCUMENT_ADDED)
        job_queue.register_handler(JobType.CUSTOM, job_handler)

        await event_bus.start()
        await job_queue.start()

        # Create WebSocket connections
        library_id = uuid4()
        num_ws = 100
        for i in range(num_ws):
            ws = Mock()
            ws.accept = AsyncMock()
            ws.send_json = AsyncMock()
            await ws_manager.connect(ws, library_id)

        # Run mixed workload
        start_time = time.time()

        async def publish_events():
            for i in range(500):
                event = Event(
                    type=EventType.DOCUMENT_ADDED,
                    library_id=library_id,
                    data={"doc": f"doc{i}"},
                )
                await event_bus.publish(event)

        async def submit_jobs():
            for i in range(500):
                await job_queue.submit(JobType.CUSTOM, {"index": i})

        async def send_broadcasts():
            for i in range(50):
                await ws_manager.broadcast_to_library(
                    library_id, {"update": f"msg{i}"}
                )
                await asyncio.sleep(0.01)

        # Run all concurrently
        await asyncio.gather(
            publish_events(),
            submit_jobs(),
            send_broadcasts(),
        )

        # Wait for processing
        await asyncio.sleep(2.0)

        elapsed = time.time() - start_time

        await event_bus.stop()
        await job_queue.stop()

        print(f"\n  Mixed workload results ({elapsed:.2f}s):")
        print(f"    Events processed: {len(event_count)}/500")
        print(f"    Jobs completed: {len(job_count)}/500")
        print(f"    WebSocket connections: {len(ws_manager._connections)}")

        # Verify all work completed
        assert len(event_count) >= 450, "Most events should be processed"
        assert len(job_count) >= 450, "Most jobs should complete"


class TestMemoryEfficiency:
    """Test memory efficiency under load."""

    @pytest.mark.asyncio
    async def test_event_bus_memory_cleanup(self):
        """Test that Event Bus cleans up completed events."""
        event_bus = EventBus()
        await event_bus.start()

        library_id = uuid4()

        # Publish many events
        for i in range(10000):
            event = Event(
                type=EventType.DOCUMENT_ADDED,
                library_id=library_id,
                data={"doc": f"doc{i}"},
            )
            await event_bus.publish(event)

        # Wait for processing
        await asyncio.sleep(2.0)
        await event_bus.stop()

        # Queue should be empty
        stats = event_bus.get_statistics()
        print(f"\n  Event Bus after 10k events:")
        print(f"    Pending: {stats['pending_events']}")
        print(f"    Published: {stats['total_published']}")

        assert stats["pending_events"] == 0, "Queue should be empty"

    @pytest.mark.asyncio
    async def test_job_queue_memory_cleanup(self):
        """Test that Job Queue manages memory efficiently."""
        job_queue = JobQueue(num_workers=4)

        async def noop_handler(job: Job):
            return {}

        job_queue.register_handler(JobType.CUSTOM, noop_handler)
        await job_queue.start()

        # Submit many jobs
        job_ids = []
        for i in range(5000):
            job_id = await job_queue.submit(JobType.CUSTOM, {"i": i})
            job_ids.append(job_id)

        # Wait for completion
        await asyncio.sleep(5.0)
        await job_queue.stop()

        stats = job_queue.get_statistics()
        print(f"\n  Job Queue after 5k jobs:")
        print(f"    Total: {stats['total_jobs']}")
        print(f"    Completed: {stats['completed_jobs']}")
        print(f"    Pending: {stats['pending_jobs']}")

        # Most jobs should complete
        assert stats["completed_jobs"] > 4500, "Most jobs should complete"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
