"""
Job handlers for background operations.

Each handler is an async function that executes a specific job type.
"""

import asyncio
import logging
from typing import Any, Dict
from uuid import UUID

from app.jobs.queue import Job
from app.services.library_service import LibraryService

logger = logging.getLogger(__name__)


async def handle_batch_import(job: Job, library_service: LibraryService) -> Dict[str, Any]:
    """
    Handle batch document import job.

    Job params:
        - documents: List of document dicts with title, texts, etc.
        - library_id: Target library UUID

    Returns:
        - imported: Number of documents imported
        - failed: Number of failures
        - total_chunks: Total chunks created
    """
    library_id = job.library_id
    documents_data = job.params.get("documents", [])

    logger.info(f"Batch import job {job.job_id}: {len(documents_data)} documents")

    # Update progress
    job.progress.update(0, len(documents_data), "Starting batch import")

    # Run in executor to avoid blocking
    def do_import():
        return library_service.add_documents_batch(library_id, documents_data)

    successful, failed, total_chunks = await asyncio.get_event_loop().run_in_executor(
        None, do_import
    )

    # Update progress
    job.progress.update(len(successful), len(documents_data), "Import complete")

    return {
        "imported": len(successful),
        "failed": len(failed),
        "total_chunks": total_chunks,
        "library_id": str(library_id),
    }


async def handle_index_rebuild(job: Job, library_service: LibraryService) -> Dict[str, Any]:
    """
    Handle index rebuild job.

    Job params:
        - library_id: Target library UUID
        - index_type: New index type (optional)
        - index_config: Index configuration (optional)

    Returns:
        - old_index_type: Previous index type
        - new_index_type: New index type
        - vectors_reindexed: Number of vectors
    """
    library_id = job.library_id
    index_type = job.params.get("index_type")
    index_config = job.params.get("index_config")

    logger.info(f"Index rebuild job {job.job_id}: library {library_id}")

    # Update progress
    job.progress.update(0, 100, "Rebuilding index")

    # Run in executor
    def do_rebuild():
        return library_service.rebuild_index(library_id, index_type, index_config)

    old_type, new_type, vectors = await asyncio.get_event_loop().run_in_executor(
        None, do_rebuild
    )

    # Update progress
    job.progress.update(100, 100, "Index rebuild complete")

    return {
        "old_index_type": old_type,
        "new_index_type": new_type,
        "vectors_reindexed": vectors,
        "library_id": str(library_id),
    }


async def handle_index_optimize(job: Job, library_service: LibraryService) -> Dict[str, Any]:
    """
    Handle index optimization job.

    Job params:
        - library_id: Target library UUID

    Returns:
        - vectors_compacted: Number of vectors
        - memory_freed: Bytes freed
    """
    library_id = job.library_id

    logger.info(f"Index optimize job {job.job_id}: library {library_id}")

    # Update progress
    job.progress.update(0, 100, "Optimizing index")

    # Run in executor
    def do_optimize():
        return library_service.optimize_index(library_id)

    vectors, memory_freed = await asyncio.get_event_loop().run_in_executor(
        None, do_optimize
    )

    # Update progress
    job.progress.update(100, 100, "Optimization complete")

    return {
        "vectors_compacted": vectors,
        "memory_freed_bytes": memory_freed,
        "library_id": str(library_id),
    }


async def handle_regenerate_embeddings(
    job: Job, library_service: LibraryService
) -> Dict[str, Any]:
    """
    Handle embedding regeneration job.

    Job params:
        - library_id: Target library UUID

    Returns:
        - chunks_reembedded: Number of chunks processed
    """
    library_id = job.library_id

    logger.info(f"Regenerate embeddings job {job.job_id}: library {library_id}")

    # Update progress
    job.progress.update(0, 100, "Regenerating embeddings")

    # Run in executor
    def do_regenerate():
        return library_service.regenerate_embeddings(library_id)

    chunks = await asyncio.get_event_loop().run_in_executor(None, do_regenerate)

    # Update progress
    job.progress.update(100, 100, "Regeneration complete")

    return {
        "chunks_reembedded": chunks,
        "library_id": str(library_id),
    }


async def handle_batch_delete(job: Job, library_service: LibraryService) -> Dict[str, Any]:
    """
    Handle batch document deletion job.

    Job params:
        - document_ids: List of document UUIDs to delete

    Returns:
        - deleted: Number of documents deleted
        - failed: Number of failures
    """
    document_ids = [UUID(doc_id) for doc_id in job.params.get("document_ids", [])]

    logger.info(f"Batch delete job {job.job_id}: {len(document_ids)} documents")

    # Update progress
    job.progress.update(0, len(document_ids), "Starting batch delete")

    # Run in executor
    def do_delete():
        return library_service.delete_documents_batch(document_ids)

    successful, failed = await asyncio.get_event_loop().run_in_executor(None, do_delete)

    # Update progress
    job.progress.update(len(successful), len(document_ids), "Delete complete")

    return {
        "deleted": len(successful),
        "failed": len(failed),
    }


async def handle_batch_export(job: Job, library_service: LibraryService) -> Dict[str, Any]:
    """
    Handle batch document export job.

    Job params:
        - library_id: Target library UUID
        - format: Export format (json, ndjson, csv)
        - include_embeddings: Include embeddings (default: false)

    Returns:
        - documents_exported: Number of documents
        - format: Export format used
        - file_size: Size in bytes
    """
    library_id = job.library_id
    export_format = job.params.get("format", "json")
    include_embeddings = job.params.get("include_embeddings", False)

    logger.info(f"Batch export job {job.job_id}: library {library_id}")

    # Update progress
    job.progress.update(0, 100, f"Exporting to {export_format}")

    # Get library
    def get_library():
        return library_service.get_library(library_id)

    library = await asyncio.get_event_loop().run_in_executor(None, get_library)

    if not library:
        raise ValueError(f"Library {library_id} not found")

    # Count documents
    num_docs = len(library.documents)

    # Export (simplified - in production would write to file/stream)
    import json

    documents_data = []
    for doc in library.documents:
        doc_dict = {
            "id": str(doc.id),
            "title": doc.metadata.title,
            "author": doc.metadata.author,
            "chunks": [],
        }

        for chunk in doc.chunks:
            chunk_dict = {
                "id": str(chunk.id),
                "text": chunk.text,
                "metadata": chunk.metadata.model_dump() if hasattr(chunk.metadata, 'model_dump') else {},
            }

            if include_embeddings:
                chunk_dict["embedding"] = chunk.embedding.tolist() if chunk.embedding is not None else None

            doc_dict["chunks"].append(chunk_dict)

        documents_data.append(doc_dict)

    # Calculate size
    exported_json = json.dumps(documents_data)
    file_size = len(exported_json.encode("utf-8"))

    # Update progress
    job.progress.update(100, 100, "Export complete")

    return {
        "documents_exported": num_docs,
        "format": export_format,
        "file_size_bytes": file_size,
        "library_id": str(library_id),
    }


def register_default_handlers(job_queue, library_service: LibraryService):
    """
    Register default job handlers.

    Args:
        job_queue: JobQueue instance
        library_service: LibraryService instance
    """
    from app.jobs.queue import JobType

    # Create wrapped handlers that inject library_service
    async def batch_import_handler(job: Job):
        return await handle_batch_import(job, library_service)

    async def index_rebuild_handler(job: Job):
        return await handle_index_rebuild(job, library_service)

    async def index_optimize_handler(job: Job):
        return await handle_index_optimize(job, library_service)

    async def regenerate_embeddings_handler(job: Job):
        return await handle_regenerate_embeddings(job, library_service)

    async def batch_delete_handler(job: Job):
        return await handle_batch_delete(job, library_service)

    async def batch_export_handler(job: Job):
        return await handle_batch_export(job, library_service)

    # Register handlers
    job_queue.register_handler(JobType.BATCH_IMPORT, batch_import_handler)
    job_queue.register_handler(JobType.INDEX_REBUILD, index_rebuild_handler)
    job_queue.register_handler(JobType.INDEX_OPTIMIZE, index_optimize_handler)
    job_queue.register_handler(JobType.REGENERATE_EMBEDDINGS, regenerate_embeddings_handler)
    job_queue.register_handler(JobType.BATCH_DELETE, batch_delete_handler)
    job_queue.register_handler(JobType.BATCH_EXPORT, batch_export_handler)

    logger.info("Registered all default job handlers")
