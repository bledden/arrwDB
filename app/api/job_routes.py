"""
REST endpoints for background job management.

Provides:
- Job submission endpoints
- Job status polling
- Job cancellation
- Job listing and filtering
"""

import logging
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel

from app.api.dependencies import get_library_service
from app.jobs.queue import JobStatus, JobType, get_job_queue
from app.services.library_service import LibraryService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/jobs", tags=["jobs"])


# Request/Response Models


class SubmitBatchImportRequest(BaseModel):
    """Request to submit batch import job."""

    library_id: UUID
    documents: List[dict]


class SubmitIndexRebuildRequest(BaseModel):
    """Request to submit index rebuild job."""

    library_id: UUID
    index_type: Optional[str] = None
    index_config: Optional[dict] = None


class SubmitIndexOptimizeRequest(BaseModel):
    """Request to submit index optimization job."""

    library_id: UUID


class SubmitBatchExportRequest(BaseModel):
    """Request to submit batch export job."""

    library_id: UUID
    format: str = "json"
    include_embeddings: bool = False


class SubmitBatchDeleteRequest(BaseModel):
    """Request to submit batch delete job."""

    document_ids: List[str]


class JobSubmitResponse(BaseModel):
    """Response after submitting a job."""

    job_id: str
    status: str
    message: str


class JobStatusResponse(BaseModel):
    """Job status response."""

    job_id: str
    job_type: str
    status: str
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    library_id: Optional[str]
    result: Optional[dict]
    error: Optional[str]
    progress: dict
    retries: int
    max_retries: int


class JobListResponse(BaseModel):
    """List of jobs response."""

    jobs: List[JobStatusResponse]
    total: int


class JobQueueStatsResponse(BaseModel):
    """Job queue statistics response."""

    total_jobs: int
    completed_jobs: int
    failed_jobs: int
    pending_jobs: int
    running_jobs: int
    queue_size: int
    num_workers: int
    running: bool


# Endpoints


@router.post("/batch-import", response_model=JobSubmitResponse)
async def submit_batch_import_job(
    request: SubmitBatchImportRequest,
    library_service: LibraryService = Depends(get_library_service),
):
    """
    Submit a batch document import job.

    This runs in the background and returns immediately with a job_id.
    Use GET /jobs/{job_id} to check status and result.

    **Benefits**:
    - Non-blocking API response
    - Progress tracking
    - Retry on failure
    - Result persistence

    **Example**:
    ```bash
    curl -X POST http://localhost:8000/v1/jobs/batch-import \\
      -H "Content-Type: application/json" \\
      -d '{
        "library_id": "uuid",
        "documents": [
          {"title": "Doc 1", "texts": ["text"]},
          {"title": "Doc 2", "texts": ["text"]}
        ]
      }'
    ```

    Returns job_id for tracking.
    """
    queue = get_job_queue()

    # Verify library exists
    library = library_service.get_library(request.library_id)
    if not library:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Library {request.library_id} not found",
        )

    job_id = await queue.submit(
        JobType.BATCH_IMPORT,
        params={"documents": request.documents},
        library_id=request.library_id,
    )

    return JobSubmitResponse(
        job_id=job_id,
        status="pending",
        message=f"Batch import job submitted ({len(request.documents)} documents)",
    )


@router.post("/index-rebuild", response_model=JobSubmitResponse)
async def submit_index_rebuild_job(
    request: SubmitIndexRebuildRequest,
    library_service: LibraryService = Depends(get_library_service),
):
    """
    Submit an index rebuild job.

    Rebuilding an index can take time for large libraries,
    so it runs in the background.

    **Example**:
    ```bash
    curl -X POST http://localhost:8000/v1/jobs/index-rebuild \\
      -H "Content-Type: application/json" \\
      -d '{
        "library_id": "uuid",
        "index_type": "hnsw"
      }'
    ```
    """
    queue = get_job_queue()

    # Verify library exists
    library = library_service.get_library(request.library_id)
    if not library:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Library {request.library_id} not found",
        )

    job_id = await queue.submit(
        JobType.INDEX_REBUILD,
        params={
            "index_type": request.index_type,
            "index_config": request.index_config,
        },
        library_id=request.library_id,
    )

    return JobSubmitResponse(
        job_id=job_id,
        status="pending",
        message=f"Index rebuild job submitted for library {request.library_id}",
    )


@router.post("/index-optimize", response_model=JobSubmitResponse)
async def submit_index_optimize_job(
    request: SubmitIndexOptimizeRequest,
    library_service: LibraryService = Depends(get_library_service),
):
    """
    Submit an index optimization job.

    Optimizes the index by compacting and removing fragmentation.

    **Example**:
    ```bash
    curl -X POST http://localhost:8000/v1/jobs/index-optimize \\
      -H "Content-Type: application/json" \\
      -d '{"library_id": "uuid"}'
    ```
    """
    queue = get_job_queue()

    # Verify library exists
    library = library_service.get_library(request.library_id)
    if not library:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Library {request.library_id} not found",
        )

    job_id = await queue.submit(
        JobType.INDEX_OPTIMIZE,
        params={},
        library_id=request.library_id,
    )

    return JobSubmitResponse(
        job_id=job_id,
        status="pending",
        message=f"Index optimization job submitted for library {request.library_id}",
    )


@router.post("/batch-export", response_model=JobSubmitResponse)
async def submit_batch_export_job(
    request: SubmitBatchExportRequest,
    library_service: LibraryService = Depends(get_library_service),
):
    """
    Submit a batch export job.

    Exports all documents in a library to specified format.

    **Example**:
    ```bash
    curl -X POST http://localhost:8000/v1/jobs/batch-export \\
      -H "Content-Type: application/json" \\
      -d '{
        "library_id": "uuid",
        "format": "json",
        "include_embeddings": false
      }'
    ```
    """
    queue = get_job_queue()

    # Verify library exists
    library = library_service.get_library(request.library_id)
    if not library:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Library {request.library_id} not found",
        )

    job_id = await queue.submit(
        JobType.BATCH_EXPORT,
        params={
            "format": request.format,
            "include_embeddings": request.include_embeddings,
        },
        library_id=request.library_id,
    )

    return JobSubmitResponse(
        job_id=job_id,
        status="pending",
        message=f"Batch export job submitted for library {request.library_id}",
    )


@router.post("/batch-delete", response_model=JobSubmitResponse)
async def submit_batch_delete_job(request: SubmitBatchDeleteRequest):
    """
    Submit a batch delete job.

    Deletes multiple documents in the background.

    **Example**:
    ```bash
    curl -X POST http://localhost:8000/v1/jobs/batch-delete \\
      -H "Content-Type: application/json" \\
      -d '{
        "document_ids": ["uuid1", "uuid2", "uuid3"]
      }'
    ```
    """
    queue = get_job_queue()

    job_id = await queue.submit(
        JobType.BATCH_DELETE,
        params={"document_ids": request.document_ids},
    )

    return JobSubmitResponse(
        job_id=job_id,
        status="pending",
        message=f"Batch delete job submitted ({len(request.document_ids)} documents)",
    )


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Get the status and details of a job.

    **Returns**:
    - Current status (pending, running, completed, failed, cancelled)
    - Progress information
    - Result (if completed)
    - Error message (if failed)

    **Example**:
    ```bash
    curl http://localhost:8000/v1/jobs/{job_id}
    ```
    """
    queue = get_job_queue()
    job = queue.get_job(job_id)

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )

    return JobStatusResponse(**job.to_dict())


@router.get("", response_model=JobListResponse)
async def list_jobs(
    status_filter: Optional[str] = Query(None, alias="status"),
    job_type: Optional[str] = Query(None, alias="type"),
    library_id: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
):
    """
    List jobs with optional filtering.

    **Query Parameters**:
    - `status`: Filter by status (pending, running, completed, failed, cancelled)
    - `type`: Filter by job type
    - `library_id`: Filter by library
    - `limit`: Maximum number of jobs to return (default: 100)

    **Example**:
    ```bash
    # Get all running jobs
    curl http://localhost:8000/v1/jobs?status=running

    # Get completed jobs for a library
    curl http://localhost:8000/v1/jobs?status=completed&library_id=uuid
    ```
    """
    queue = get_job_queue()

    # Parse filters
    status_enum = None
    if status_filter:
        try:
            status_enum = JobStatus(status_filter)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status: {status_filter}",
            )

    job_type_enum = None
    if job_type:
        try:
            job_type_enum = JobType(job_type)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid job type: {job_type}",
            )

    library_uuid = None
    if library_id:
        try:
            library_uuid = UUID(library_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid library_id: {library_id}",
            )

    # List jobs
    jobs = queue.list_jobs(
        status=status_enum,
        job_type=job_type_enum,
        library_id=library_uuid,
        limit=limit,
    )

    job_responses = [JobStatusResponse(**job.to_dict()) for job in jobs]

    return JobListResponse(jobs=job_responses, total=len(job_responses))


@router.delete("/{job_id}")
async def cancel_job(job_id: str):
    """
    Cancel a pending or running job.

    **Returns**:
    - 204 No Content if cancelled successfully
    - 404 Not Found if job doesn't exist
    - 400 Bad Request if job already finished

    **Example**:
    ```bash
    curl -X DELETE http://localhost:8000/v1/jobs/{job_id}
    ```
    """
    queue = get_job_queue()
    success = await queue.cancel_job(job_id)

    if not success:
        job = queue.get_job(job_id)
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found",
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Job {job_id} cannot be cancelled (status: {job.status.value})",
            )

    return {"message": f"Job {job_id} cancelled"}


@router.get("/stats/queue", response_model=JobQueueStatsResponse)
async def get_queue_statistics():
    """
    Get job queue statistics.

    **Returns**:
    - Total jobs submitted
    - Completed/failed counts
    - Current pending/running counts
    - Queue size and worker count

    **Example**:
    ```bash
    curl http://localhost:8000/v1/jobs/stats/queue
    ```
    """
    queue = get_job_queue()
    stats = queue.get_statistics()

    return JobQueueStatsResponse(**stats)
