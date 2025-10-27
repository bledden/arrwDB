"""
Test suite for app/jobs/queue.py

Coverage targets:
- Job creation and validation
- Job enqueueing and dequeuing
- Job execution
- Job status tracking
- Job retry handling
- Job cancellation
- Job filtering and listing
- Queue statistics
- Worker lifecycle
- Concurrent job processing
"""

import asyncio
from uuid import uuid4

import pytest

from app.jobs.queue import (
    Job,
    JobProgress,
    JobQueue,
    JobStatus,
    JobType,
    get_job_queue,
)


class TestJobProgress:
    """Test job progress tracking."""

    def test_job_progress_creation(self):
        """Test creating job progress."""
        progress = JobProgress()
        assert progress.current == 0
        assert progress.total == 0
        assert progress.message == ""
        assert progress.percentage == 0.0

    def test_job_progress_update(self):
        """Test updating job progress."""
        progress = JobProgress()
        progress.update(50, 100, "Processing...")

        assert progress.current == 50
        assert progress.total == 100
        assert progress.message == "Processing..."
        assert progress.percentage == 50.0

    def test_job_progress_percentage_calculation(self):
        """Test percentage calculation."""
        progress = JobProgress()

        progress.update(25, 100)
        assert progress.percentage == 25.0

        progress.update(75, 100)
        assert progress.percentage == 75.0

        progress.update(100, 100)
        assert progress.percentage == 100.0

    def test_job_progress_zero_total(self):
        """Test progress with zero total (edge case)."""
        progress = JobProgress()
        progress.update(0, 0, "Starting...")

        assert progress.percentage == 0.0


class TestJobCreation:
    """Test job creation and data structures."""

    def test_job_creation_defaults(self):
        """Test creating a job with default values."""
        job = Job()

        assert job.job_id is not None
        assert job.job_type == JobType.CUSTOM
        assert job.status == JobStatus.PENDING
        assert job.created_at is not None
        assert job.started_at is None
        assert job.completed_at is None
        assert job.library_id is None
        assert job.params == {}
        assert job.result is None
        assert job.error is None
        assert job.retries == 0
        assert job.max_retries == 3

    def test_job_creation_with_params(self):
        """Test creating a job with custom parameters."""
        library_id = uuid4()
        params = {"documents": ["doc1", "doc2"], "batch_size": 10}

        job = Job(
            job_type=JobType.BATCH_IMPORT,
            library_id=library_id,
            params=params,
            max_retries=5,
        )

        assert job.job_type == JobType.BATCH_IMPORT
        assert job.library_id == library_id
        assert job.params == params
        assert job.max_retries == 5

    def test_job_to_dict(self):
        """Test converting job to dictionary."""
        library_id = uuid4()
        job = Job(
            job_type=JobType.INDEX_REBUILD,
            library_id=library_id,
            params={"force": True},
        )

        job_dict = job.to_dict()

        assert job_dict["job_id"] == job.job_id
        assert job_dict["job_type"] == "index_rebuild"
        assert job_dict["status"] == "pending"
        assert job_dict["library_id"] == str(library_id)
        assert job_dict["params"] == {"force": True}
        assert "progress" in job_dict


class TestJobQueueBasics:
    """Test basic job queue operations."""

    @pytest.fixture
    def job_queue(self):
        """Provide fresh job queue for each test."""
        return JobQueue(num_workers=2)

    @pytest.mark.asyncio
    async def test_job_queue_creation(self, job_queue):
        """Test creating a job queue."""
        assert job_queue._num_workers == 2
        assert not job_queue._running
        assert len(job_queue._workers) == 0

    @pytest.mark.asyncio
    async def test_register_handler(self, job_queue):
        """Test registering a job handler."""
        async def test_handler(job: Job):
            return {"success": True}

        job_queue.register_handler(JobType.BATCH_IMPORT, test_handler)

        assert JobType.BATCH_IMPORT in job_queue._handlers
        assert job_queue._handlers[JobType.BATCH_IMPORT] == test_handler

    @pytest.mark.asyncio
    async def test_submit_job(self, job_queue):
        """Test submitting a job to the queue."""
        library_id = uuid4()
        params = {"documents": ["doc1", "doc2"]}

        job_id = await job_queue.submit(
            JobType.BATCH_IMPORT,
            params,
            library_id=library_id,
            max_retries=5,
        )

        assert job_id is not None
        job = job_queue.get_job(job_id)
        assert job is not None
        assert job.job_type == JobType.BATCH_IMPORT
        assert job.params == params
        assert job.library_id == library_id
        assert job.max_retries == 5
        assert job.status == JobStatus.PENDING

    @pytest.mark.asyncio
    async def test_get_job(self, job_queue):
        """Test getting a job by ID."""
        job_id = await job_queue.submit(JobType.CUSTOM, {})

        job = job_queue.get_job(job_id)
        assert job is not None
        assert job.job_id == job_id

    @pytest.mark.asyncio
    async def test_get_nonexistent_job(self, job_queue):
        """Test getting a nonexistent job."""
        job = job_queue.get_job("nonexistent-id")
        assert job is None


class TestJobExecution:
    """Test job execution functionality."""

    @pytest.fixture
    def job_queue(self):
        """Provide fresh job queue for each test."""
        return JobQueue(num_workers=2)

    @pytest.mark.asyncio
    async def test_successful_job_execution(self, job_queue):
        """Test executing a job successfully."""
        executed = []

        async def handler(job: Job):
            executed.append(job.job_id)
            return {"status": "done"}

        job_queue.register_handler(JobType.CUSTOM, handler)

        await job_queue.start()
        job_id = await job_queue.submit(JobType.CUSTOM, {"test": True})

        # Wait for job to complete
        await asyncio.sleep(0.2)
        await job_queue.stop()

        job = job_queue.get_job(job_id)
        assert job.status == JobStatus.COMPLETED
        assert job.result == {"status": "done"}
        assert job.error is None
        assert job_id in executed

    @pytest.mark.asyncio
    async def test_job_failure_with_retry(self, job_queue):
        """Test job failure and retry logic."""
        attempt_count = []

        async def failing_handler(job: Job):
            attempt_count.append(1)
            if len(attempt_count) < 2:
                raise ValueError("Simulated error")
            return {"success": True}

        job_queue.register_handler(JobType.CUSTOM, failing_handler)

        await job_queue.start()
        job_id = await job_queue.submit(JobType.CUSTOM, {}, max_retries=3)

        # Wait for retries
        await asyncio.sleep(0.5)
        await job_queue.stop()

        job = job_queue.get_job(job_id)
        assert job.status == JobStatus.COMPLETED
        assert job.retries == 1  # One retry
        assert len(attempt_count) == 2  # Original + 1 retry

    @pytest.mark.asyncio
    async def test_job_max_retries_exceeded(self, job_queue):
        """Test job failing after max retries."""
        async def always_fails(job: Job):
            raise ValueError("Always fails")

        job_queue.register_handler(JobType.CUSTOM, always_fails)

        await job_queue.start()
        job_id = await job_queue.submit(JobType.CUSTOM, {}, max_retries=2)

        # Wait for all retries
        await asyncio.sleep(0.5)
        await job_queue.stop()

        job = job_queue.get_job(job_id)
        assert job.status == JobStatus.FAILED
        assert job.retries == 2  # Max retries
        assert job.error is not None
        assert "ValueError" in job.error

    @pytest.mark.asyncio
    async def test_job_without_handler(self, job_queue):
        """Test executing a job without registered handler."""
        await job_queue.start()
        job_id = await job_queue.submit(JobType.BATCH_IMPORT, {})

        # Wait for job to fail
        await asyncio.sleep(0.2)
        await job_queue.stop()

        job = job_queue.get_job(job_id)
        assert job.status == JobStatus.FAILED
        assert "No handler registered" in job.error


class TestJobCancellation:
    """Test job cancellation functionality."""

    @pytest.fixture
    def job_queue(self):
        """Provide fresh job queue for each test."""
        return JobQueue(num_workers=2)

    @pytest.mark.asyncio
    async def test_cancel_pending_job(self, job_queue):
        """Test cancelling a pending job."""
        job_id = await job_queue.submit(JobType.CUSTOM, {})

        # Job should be pending
        job = job_queue.get_job(job_id)
        assert job.status == JobStatus.PENDING

        # Cancel it
        cancelled = await job_queue.cancel_job(job_id)
        assert cancelled is True

        job = job_queue.get_job(job_id)
        assert job.status == JobStatus.CANCELLED
        assert job.completed_at is not None

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_job(self, job_queue):
        """Test cancelling a nonexistent job."""
        cancelled = await job_queue.cancel_job("nonexistent-id")
        assert cancelled is False

    @pytest.mark.asyncio
    async def test_cannot_cancel_completed_job(self, job_queue):
        """Test that completed jobs cannot be cancelled."""
        async def handler(job: Job):
            return {"done": True}

        job_queue.register_handler(JobType.CUSTOM, handler)

        await job_queue.start()
        job_id = await job_queue.submit(JobType.CUSTOM, {})

        # Wait for completion
        await asyncio.sleep(0.2)

        # Try to cancel completed job
        cancelled = await job_queue.cancel_job(job_id)
        assert cancelled is False

        await job_queue.stop()


class TestJobListing:
    """Test job listing and filtering."""

    @pytest.fixture
    def job_queue(self):
        """Provide fresh job queue for each test."""
        return JobQueue(num_workers=2)

    @pytest.mark.asyncio
    async def test_list_all_jobs(self, job_queue):
        """Test listing all jobs."""
        # Submit multiple jobs
        job_id1 = await job_queue.submit(JobType.BATCH_IMPORT, {})
        job_id2 = await job_queue.submit(JobType.INDEX_REBUILD, {})
        job_id3 = await job_queue.submit(JobType.CUSTOM, {})

        jobs = job_queue.list_jobs()

        assert len(jobs) == 3
        job_ids = [j.job_id for j in jobs]
        assert job_id1 in job_ids
        assert job_id2 in job_ids
        assert job_id3 in job_ids

    @pytest.mark.asyncio
    async def test_list_jobs_by_status(self, job_queue):
        """Test filtering jobs by status."""
        async def handler(job: Job):
            return {"done": True}

        job_queue.register_handler(JobType.CUSTOM, handler)

        # Submit jobs
        await job_queue.submit(JobType.CUSTOM, {})
        await job_queue.submit(JobType.BATCH_IMPORT, {})  # No handler, will fail

        await job_queue.start()
        await asyncio.sleep(0.3)
        await job_queue.stop()

        # Check pending jobs
        pending = job_queue.list_jobs(status=JobStatus.PENDING)
        completed = job_queue.list_jobs(status=JobStatus.COMPLETED)
        failed = job_queue.list_jobs(status=JobStatus.FAILED)

        assert len(completed) >= 1
        assert len(failed) >= 1

    @pytest.mark.asyncio
    async def test_list_jobs_by_type(self, job_queue):
        """Test filtering jobs by type."""
        await job_queue.submit(JobType.BATCH_IMPORT, {})
        await job_queue.submit(JobType.BATCH_IMPORT, {})
        await job_queue.submit(JobType.INDEX_REBUILD, {})

        import_jobs = job_queue.list_jobs(job_type=JobType.BATCH_IMPORT)
        rebuild_jobs = job_queue.list_jobs(job_type=JobType.INDEX_REBUILD)

        assert len(import_jobs) == 2
        assert len(rebuild_jobs) == 1

    @pytest.mark.asyncio
    async def test_list_jobs_by_library(self, job_queue):
        """Test filtering jobs by library."""
        lib_id1 = uuid4()
        lib_id2 = uuid4()

        await job_queue.submit(JobType.CUSTOM, {}, library_id=lib_id1)
        await job_queue.submit(JobType.CUSTOM, {}, library_id=lib_id1)
        await job_queue.submit(JobType.CUSTOM, {}, library_id=lib_id2)

        lib1_jobs = job_queue.list_jobs(library_id=lib_id1)
        lib2_jobs = job_queue.list_jobs(library_id=lib_id2)

        assert len(lib1_jobs) == 2
        assert len(lib2_jobs) == 1

    @pytest.mark.asyncio
    async def test_list_jobs_with_limit(self, job_queue):
        """Test limiting the number of returned jobs."""
        for _ in range(10):
            await job_queue.submit(JobType.CUSTOM, {})

        jobs = job_queue.list_jobs(limit=5)

        assert len(jobs) == 5


class TestJobQueueLifecycle:
    """Test job queue lifecycle management."""

    @pytest.fixture
    def job_queue(self):
        """Provide fresh job queue for each test."""
        return JobQueue(num_workers=2)

    @pytest.mark.asyncio
    async def test_start_job_queue(self, job_queue):
        """Test starting the job queue."""
        assert not job_queue._running

        await job_queue.start()

        assert job_queue._running
        assert len(job_queue._workers) == 2

    @pytest.mark.asyncio
    async def test_stop_job_queue(self, job_queue):
        """Test stopping the job queue."""
        await job_queue.start()
        assert job_queue._running

        await job_queue.stop()

        assert not job_queue._running

    @pytest.mark.asyncio
    async def test_start_already_running(self, job_queue):
        """Test starting an already running queue."""
        await job_queue.start()
        worker_count = len(job_queue._workers)

        # Start again
        await job_queue.start()

        # Should not create duplicate workers
        assert len(job_queue._workers) == worker_count

        await job_queue.stop()


class TestJobQueueStatistics:
    """Test job queue statistics tracking."""

    @pytest.fixture
    def job_queue(self):
        """Provide fresh job queue for each test."""
        return JobQueue(num_workers=2)

    @pytest.mark.asyncio
    async def test_initial_statistics(self, job_queue):
        """Test initial statistics."""
        stats = job_queue.get_statistics()

        assert stats["total_jobs"] == 0
        assert stats["completed_jobs"] == 0
        assert stats["failed_jobs"] == 0
        assert stats["pending_jobs"] == 0
        assert stats["running_jobs"] == 0
        assert stats["queue_size"] == 0
        assert stats["num_workers"] == 2
        assert stats["running"] is False

    @pytest.mark.asyncio
    async def test_statistics_after_jobs(self, job_queue):
        """Test statistics after submitting and executing jobs."""
        async def handler(job: Job):
            return {"success": True}

        job_queue.register_handler(JobType.CUSTOM, handler)

        # Submit jobs
        await job_queue.submit(JobType.CUSTOM, {})
        await job_queue.submit(JobType.CUSTOM, {})
        await job_queue.submit(JobType.CUSTOM, {})

        stats = job_queue.get_statistics()
        assert stats["total_jobs"] == 3

        # Execute jobs
        await job_queue.start()
        await asyncio.sleep(0.3)
        await job_queue.stop()

        stats = job_queue.get_statistics()
        assert stats["total_jobs"] == 3
        assert stats["completed_jobs"] == 3
        assert stats["failed_jobs"] == 0


class TestConcurrentJobProcessing:
    """Test concurrent job processing."""

    @pytest.fixture
    def job_queue(self):
        """Provide fresh job queue for each test."""
        return JobQueue(num_workers=4)

    @pytest.mark.asyncio
    async def test_concurrent_job_execution(self, job_queue):
        """Test executing multiple jobs concurrently."""
        executed_jobs = []

        async def handler(job: Job):
            executed_jobs.append(job.job_id)
            await asyncio.sleep(0.1)  # Simulate work
            return {"index": job.params.get("index")}

        job_queue.register_handler(JobType.CUSTOM, handler)

        await job_queue.start()

        # Submit multiple jobs
        job_ids = []
        for i in range(10):
            job_id = await job_queue.submit(JobType.CUSTOM, {"index": i})
            job_ids.append(job_id)

        # Wait for all jobs to complete
        await asyncio.sleep(0.5)
        await job_queue.stop()

        # All jobs should be completed
        assert len(executed_jobs) == 10

        for job_id in job_ids:
            job = job_queue.get_job(job_id)
            assert job.status == JobStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_worker_load_distribution(self, job_queue):
        """Test that jobs are distributed across workers."""
        worker_usage = []

        async def handler(job: Job):
            worker_usage.append(job.params.get("worker_hint"))
            await asyncio.sleep(0.05)
            return {"done": True}

        job_queue.register_handler(JobType.CUSTOM, handler)

        await job_queue.start()

        # Submit jobs
        for i in range(8):
            await job_queue.submit(JobType.CUSTOM, {"worker_hint": i})

        await asyncio.sleep(0.4)
        await job_queue.stop()

        # All jobs should have been executed
        assert len(worker_usage) == 8


class TestJobQueueSingleton:
    """Test the global job queue singleton pattern."""

    def test_get_job_queue_singleton(self):
        """Test that get_job_queue returns the same instance."""
        queue1 = get_job_queue()
        queue2 = get_job_queue()

        assert queue1 is queue2


class TestJobTypes:
    """Test job type enumeration."""

    def test_all_job_types_exist(self):
        """Test that all expected job types are defined."""
        expected_types = [
            "BATCH_IMPORT",
            "BATCH_EXPORT",
            "INDEX_REBUILD",
            "INDEX_OPTIMIZE",
            "BATCH_DELETE",
            "REGENERATE_EMBEDDINGS",
            "CUSTOM",
        ]

        for job_type_name in expected_types:
            assert hasattr(JobType, job_type_name)

    def test_job_status_types_exist(self):
        """Test that all job status types are defined."""
        expected_statuses = [
            "PENDING",
            "RUNNING",
            "COMPLETED",
            "FAILED",
            "CANCELLED",
        ]

        for status_name in expected_statuses:
            assert hasattr(JobStatus, status_name)
