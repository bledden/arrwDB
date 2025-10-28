"""
Background reactor for long-running async operations.

This module implements the Reactor pattern - an event-driven architecture that
actively responds to and processes background jobs. Unlike a passive queue that
simply stores items, a reactor monitors events and dispatches them to appropriate
handlers for processing.

The Reactor pattern (popularized by Node.js, Twisted, and other async frameworks)
emphasizes:
- Event-driven processing: Jobs are events that trigger reactive handlers
- Non-blocking I/O: Async workers process jobs without blocking
- Event demultiplexing: Multiple workers handle different jobs concurrently
- Active response: The system actively reacts to job submission, not passive storage

Provides:
- Async worker pool for parallel job execution (reactive processing)
- Job status tracking and progress reporting
- Result persistence and retrieval
- Cancellation and retry support
- Event-driven job dispatch to registered handlers
"""

import asyncio
import logging
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from uuid import UUID, uuid4

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Status of a background job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobType(Enum):
    """Types of background jobs."""

    BATCH_IMPORT = "batch_import"
    BATCH_EXPORT = "batch_export"
    INDEX_REBUILD = "index_rebuild"
    INDEX_OPTIMIZE = "index_optimize"
    BATCH_DELETE = "batch_delete"
    REGENERATE_EMBEDDINGS = "regenerate_embeddings"
    CUSTOM = "custom"


@dataclass
class JobProgress:
    """Progress information for a running job."""

    current: int = 0
    total: int = 0
    message: str = ""
    percentage: float = 0.0

    def update(self, current: int, total: int, message: str = ""):
        """Update progress."""
        self.current = current
        self.total = total
        self.message = message
        self.percentage = (current / total * 100) if total > 0 else 0.0


@dataclass
class Job:
    """
    Background job representation.

    Attributes:
        job_id: Unique job identifier
        job_type: Type of job
        status: Current status
        created_at: When job was created
        started_at: When job started running
        completed_at: When job finished
        library_id: Associated library (if applicable)
        params: Job-specific parameters
        result: Job result (if completed)
        error: Error message (if failed)
        progress: Progress information
        retries: Number of retry attempts
        max_retries: Maximum retries allowed
    """

    job_id: str = field(default_factory=lambda: str(uuid4()))
    job_type: JobType = JobType.CUSTOM
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    library_id: Optional[UUID] = None
    params: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Any] = None
    error: Optional[str] = None
    progress: JobProgress = field(default_factory=JobProgress)
    retries: int = 0
    max_retries: int = 3

    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary."""
        return {
            "job_id": self.job_id,
            "job_type": self.job_type.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "library_id": str(self.library_id) if self.library_id else None,
            "params": self.params,
            "result": self.result,
            "error": self.error,
            "progress": {
                "current": self.progress.current,
                "total": self.progress.total,
                "message": self.progress.message,
                "percentage": self.progress.percentage,
            },
            "retries": self.retries,
            "max_retries": self.max_retries,
        }


class BackgroundReactor:
    """
    Background reactor implementing the Reactor pattern for async job processing.

    The Reactor pattern is a well-established event-driven architecture (used in
    Node.js, Twisted, nginx) that actively responds to events rather than passively
    storing them. This reactor:

    1. **Event Registration**: Handlers register for specific job types (events)
    2. **Event Demultiplexing**: Worker pool monitors and dispatches jobs concurrently
    3. **Event Handling**: Dispatches jobs to appropriate handlers for processing
    4. **Non-blocking I/O**: Async/await ensures reactive, non-blocking execution

    Why "Reactor" over "Queue":
    - "Queue" implies passive storage of items waiting to be retrieved
    - "Reactor" emphasizes active, event-driven response to job submissions
    - Better reflects the async, event-driven architecture of the system
    - Aligns with established patterns from async frameworks (Twisted, Node.js)
    - Communicates reactive processing vs. passive buffering

    Features:
    - Parallel job execution with configurable workers (event demultiplexing)
    - Job status tracking and progress reporting
    - Automatic retry on failure
    - Cancellation support
    - Result persistence
    - Event-driven handler registration and dispatch
    """

    def __init__(self, num_workers: int = 4):
        """
        Initialize background reactor.

        Args:
            num_workers: Number of worker tasks for event demultiplexing
        """
        self._num_workers = num_workers
        self._queue: asyncio.Queue = asyncio.Queue()
        self._jobs: Dict[str, Job] = {}
        self._handlers: Dict[JobType, Callable] = {}
        self._workers: List[asyncio.Task] = []
        self._running = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Statistics
        self._total_jobs = 0
        self._completed_jobs = 0
        self._failed_jobs = 0

        logger.info(f"BackgroundReactor initialized with {num_workers} workers")

    def register_handler(self, job_type: JobType, handler: Callable):
        """
        Register an event handler for a job type (Reactor pattern: Event Registration).

        This implements the event registration phase of the Reactor pattern, where
        handlers subscribe to specific job types (events) they want to process.

        Args:
            job_type: Type of job (event) to handle
            handler: Async function to execute when job type is dispatched
                     Signature: async def handler(job: Job) -> Any

        Example:
            ```python
            async def handle_batch_import(job: Job):
                documents = job.params["documents"]
                library_id = job.library_id
                # ... import documents ...
                return {"imported": len(documents)}

            reactor.register_handler(JobType.BATCH_IMPORT, handle_batch_import)
            ```
        """
        self._handlers[job_type] = handler
        logger.info(f"Registered handler for {job_type.value}")

    async def submit(
        self,
        job_type: JobType,
        params: Dict[str, Any],
        library_id: Optional[UUID] = None,
        max_retries: int = 3,
    ) -> str:
        """
        Submit a job event to the reactor for processing.

        This triggers the reactor's event-driven processing pipeline.

        Args:
            job_type: Type of job (event type)
            params: Job parameters (event data)
            library_id: Associated library (optional)
            max_retries: Maximum retry attempts

        Returns:
            job_id: Job identifier for tracking

        Example:
            ```python
            job_id = await reactor.submit(
                JobType.BATCH_IMPORT,
                {"documents": [...]},
                library_id=lib_id
            )
            ```
        """
        job = Job(
            job_type=job_type,
            params=params,
            library_id=library_id,
            max_retries=max_retries,
        )

        self._jobs[job.job_id] = job
        self._total_jobs += 1

        await self._queue.put(job.job_id)

        logger.info(f"Job {job.job_id} ({job_type.value}) submitted to reactor for processing")

        return job.job_id

    def get_job(self, job_id: str) -> Optional[Job]:
        """
        Get job status and details.

        Args:
            job_id: Job identifier

        Returns:
            Job object or None if not found
        """
        return self._jobs.get(job_id)

    def list_jobs(
        self,
        status: Optional[JobStatus] = None,
        job_type: Optional[JobType] = None,
        library_id: Optional[UUID] = None,
        limit: int = 100,
    ) -> List[Job]:
        """
        List jobs with optional filtering.

        Args:
            status: Filter by status
            job_type: Filter by job type
            library_id: Filter by library
            limit: Maximum number of jobs to return

        Returns:
            List of jobs
        """
        jobs = list(self._jobs.values())

        # Apply filters
        if status:
            jobs = [j for j in jobs if j.status == status]
        if job_type:
            jobs = [j for j in jobs if j.job_type == job_type]
        if library_id:
            jobs = [j for j in jobs if j.library_id == library_id]

        # Sort by creation time (newest first)
        jobs.sort(key=lambda j: j.created_at, reverse=True)

        return jobs[:limit]

    async def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a pending or running job.

        Args:
            job_id: Job identifier

        Returns:
            True if cancelled, False if not found or already finished
        """
        job = self._jobs.get(job_id)
        if not job:
            return False

        if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            return False

        job.status = JobStatus.CANCELLED
        job.completed_at = datetime.utcnow()

        logger.info(f"Job {job_id} cancelled")

        return True

    async def _worker(self, worker_id: int):
        """
        Background worker that reactively processes jobs (Reactor pattern: Event Demultiplexing).

        Workers continuously monitor for job events and dispatch them to handlers,
        implementing the event demultiplexing phase of the Reactor pattern.

        Args:
            worker_id: Worker identifier
        """
        logger.info(f"Worker {worker_id} started")

        while self._running:
            try:
                # Wait for job with timeout to check _running flag
                try:
                    job_id = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                job = self._jobs.get(job_id)
                if not job:
                    continue

                # Check if cancelled
                if job.status == JobStatus.CANCELLED:
                    continue

                # Reactively execute job (dispatch to handler)
                await self._execute_job(job, worker_id)

            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}", exc_info=True)

        logger.info(f"Worker {worker_id} stopped")

    async def _execute_job(self, job: Job, worker_id: int):
        """
        Execute a single job (Reactor pattern: Event Handling).

        Dispatches the job event to its registered handler for processing,
        implementing the event handling phase of the Reactor pattern.

        Args:
            job: Job to execute
            worker_id: Worker identifier
        """
        logger.info(f"Worker {worker_id} executing job {job.job_id} ({job.job_type.value})")

        # Update status
        job.status = JobStatus.RUNNING
        job.started_at = datetime.utcnow()

        try:
            # Get handler
            handler = self._handlers.get(job.job_type)
            if not handler:
                raise ValueError(f"No handler registered for {job.job_type.value}")

            # Execute handler
            start_time = time.time()
            result = await handler(job)
            execution_time = time.time() - start_time

            # Mark as completed
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            job.result = result

            self._completed_jobs += 1

            logger.info(
                f"Job {job.job_id} completed in {execution_time:.2f}s "
                f"(worker {worker_id})"
            )

        except Exception as e:
            # Job failed
            error_msg = f"{type(e).__name__}: {str(e)}"
            job.error = error_msg

            logger.error(
                f"Job {job.job_id} failed: {error_msg}\n{traceback.format_exc()}"
            )

            # Retry if possible
            if job.retries < job.max_retries:
                job.retries += 1
                job.status = JobStatus.PENDING
                job.started_at = None

                # Requeue
                await self._queue.put(job.job_id)

                logger.info(
                    f"Job {job.job_id} requeued (retry {job.retries}/{job.max_retries})"
                )
            else:
                # Max retries exceeded
                job.status = JobStatus.FAILED
                job.completed_at = datetime.utcnow()
                self._failed_jobs += 1

                logger.error(
                    f"Job {job.job_id} failed permanently after {job.retries} retries"
                )

    async def start(self):
        """
        Start the background reactor workers.

        Activates the reactor's event processing loop, starting worker tasks
        that will reactively process incoming job events.

        Call this during application startup.
        """
        if self._running:
            logger.warning("BackgroundReactor already running")
            return

        # Capture event loop (reactor event loop)
        self._loop = asyncio.get_running_loop()
        logger.info(f"BackgroundReactor captured event loop: {self._loop}")

        self._running = True

        # Start workers (event demultiplexing pool)
        for i in range(self._num_workers):
            worker_task = asyncio.create_task(self._worker(i))
            self._workers.append(worker_task)

        logger.info(f"BackgroundReactor started with {self._num_workers} workers")

    async def stop(self):
        """
        Stop the background reactor workers.

        Gracefully shuts down the reactor's event processing loop.

        Call this during application shutdown.
        Waits for running jobs to complete.
        """
        if not self._running:
            return

        logger.info("Stopping BackgroundReactor...")
        self._running = False

        # Wait for workers to finish
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)

        logger.info("BackgroundReactor stopped")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get background reactor statistics.

        Returns:
            Statistics dictionary with reactor metrics
        """
        pending = sum(1 for j in self._jobs.values() if j.status == JobStatus.PENDING)
        running = sum(1 for j in self._jobs.values() if j.status == JobStatus.RUNNING)

        return {
            "total_jobs": self._total_jobs,
            "completed_jobs": self._completed_jobs,
            "failed_jobs": self._failed_jobs,
            "pending_jobs": pending,
            "running_jobs": running,
            "queue_size": self._queue.qsize(),
            "num_workers": self._num_workers,
            "running": self._running,
        }


# Global background reactor instance
_global_job_queue: Optional[BackgroundReactor] = None


def get_job_queue() -> BackgroundReactor:
    """
    Get the global background reactor instance.

    Creates the instance on first call (singleton pattern).

    Returns:
        The global BackgroundReactor instance
    """
    global _global_job_queue
    if _global_job_queue is None:
        _global_job_queue = BackgroundReactor(num_workers=4)
    return _global_job_queue


# Backward compatibility alias
# This allows existing code to continue using JobQueue while we migrate to BackgroundReactor
JobQueue = BackgroundReactor
