"""
Test suite for app/jobs/handlers.py

Coverage targets:
- Batch import handler
- Index rebuild handler
- Index optimize handler
- Regenerate embeddings handler
- Batch delete handler
- Batch export handler
- Handler registration
- Progress tracking in handlers
- Error handling in handlers
"""

from unittest.mock import MagicMock, Mock
from uuid import uuid4

import numpy as np
import pytest

from app.jobs.handlers import (
    handle_batch_delete,
    handle_batch_export,
    handle_batch_import,
    handle_index_optimize,
    handle_index_rebuild,
    handle_regenerate_embeddings,
    register_default_handlers,
)
from app.jobs.queue import Job, JobQueue, JobType


class TestBatchImportHandler:
    """Test batch import job handler."""

    @pytest.mark.asyncio
    async def test_batch_import_success(self):
        """Test successful batch import."""
        library_id = uuid4()
        documents_data = [
            {"title": "Doc 1", "texts": ["text1", "text2"]},
            {"title": "Doc 2", "texts": ["text3"]},
        ]

        # Create job
        job = Job(
            job_type=JobType.BATCH_IMPORT,
            library_id=library_id,
            params={"documents": documents_data},
        )

        # Mock library service
        mock_service = Mock()
        mock_service.add_documents_batch.return_value = (
            ["doc1", "doc2"],  # successful
            [],  # failed
            5,  # total chunks
        )

        # Execute handler
        result = await handle_batch_import(job, mock_service)

        # Verify results
        assert result["imported"] == 2
        assert result["failed"] == 0
        assert result["total_chunks"] == 5
        assert result["library_id"] == str(library_id)

        # Verify service was called
        mock_service.add_documents_batch.assert_called_once_with(
            library_id, documents_data
        )

        # Verify progress was updated
        assert job.progress.current == 2
        assert job.progress.total == 2

    @pytest.mark.asyncio
    async def test_batch_import_with_failures(self):
        """Test batch import with some failures."""
        library_id = uuid4()
        documents_data = [
            {"title": "Doc 1", "texts": ["text1"]},
            {"title": "Doc 2", "texts": ["text2"]},
            {"title": "Doc 3", "texts": ["text3"]},
        ]

        job = Job(
            job_type=JobType.BATCH_IMPORT,
            library_id=library_id,
            params={"documents": documents_data},
        )

        # Mock service with some failures
        mock_service = Mock()
        mock_service.add_documents_batch.return_value = (
            ["doc1", "doc2"],  # successful
            ["doc3"],  # failed
            4,  # total chunks
        )

        result = await handle_batch_import(job, mock_service)

        assert result["imported"] == 2
        assert result["failed"] == 1
        assert result["total_chunks"] == 4


class TestIndexRebuildHandler:
    """Test index rebuild job handler."""

    @pytest.mark.asyncio
    async def test_index_rebuild_success(self):
        """Test successful index rebuild."""
        library_id = uuid4()

        job = Job(
            job_type=JobType.INDEX_REBUILD,
            library_id=library_id,
            params={"index_type": "hnsw", "index_config": {"m": 16, "ef": 200}},
        )

        # Mock library service
        mock_service = Mock()
        mock_service.rebuild_index.return_value = (
            "brute_force",  # old type
            "hnsw",  # new type
            1000,  # vectors reindexed
        )

        result = await handle_index_rebuild(job, mock_service)

        assert result["old_index_type"] == "brute_force"
        assert result["new_index_type"] == "hnsw"
        assert result["vectors_reindexed"] == 1000
        assert result["library_id"] == str(library_id)

        # Verify service was called with correct params
        mock_service.rebuild_index.assert_called_once_with(
            library_id, "hnsw", {"m": 16, "ef": 200}
        )

        # Verify progress
        assert job.progress.percentage == 100.0

    @pytest.mark.asyncio
    async def test_index_rebuild_without_params(self):
        """Test index rebuild without explicit type/config."""
        library_id = uuid4()

        job = Job(
            job_type=JobType.INDEX_REBUILD,
            library_id=library_id,
            params={},
        )

        mock_service = Mock()
        mock_service.rebuild_index.return_value = ("hnsw", "hnsw", 500)

        result = await handle_index_rebuild(job, mock_service)

        assert result["vectors_reindexed"] == 500
        # Should call with None for optional params
        mock_service.rebuild_index.assert_called_once_with(library_id, None, None)


class TestIndexOptimizeHandler:
    """Test index optimization job handler."""

    @pytest.mark.asyncio
    async def test_index_optimize_success(self):
        """Test successful index optimization."""
        library_id = uuid4()

        job = Job(
            job_type=JobType.INDEX_OPTIMIZE,
            library_id=library_id,
            params={},
        )

        # Mock library service
        mock_service = Mock()
        mock_service.optimize_index.return_value = (
            5000,  # vectors compacted
            1024 * 1024,  # memory freed (1MB)
        )

        result = await handle_index_optimize(job, mock_service)

        assert result["vectors_compacted"] == 5000
        assert result["memory_freed_bytes"] == 1024 * 1024
        assert result["library_id"] == str(library_id)

        mock_service.optimize_index.assert_called_once_with(library_id)

        # Verify progress
        assert job.progress.percentage == 100.0


class TestRegenerateEmbeddingsHandler:
    """Test embedding regeneration job handler."""

    @pytest.mark.asyncio
    async def test_regenerate_embeddings_success(self):
        """Test successful embedding regeneration."""
        library_id = uuid4()

        job = Job(
            job_type=JobType.REGENERATE_EMBEDDINGS,
            library_id=library_id,
            params={},
        )

        # Mock library service
        mock_service = Mock()
        mock_service.regenerate_embeddings.return_value = 250  # chunks reembedded

        result = await handle_regenerate_embeddings(job, mock_service)

        assert result["chunks_reembedded"] == 250
        assert result["library_id"] == str(library_id)

        mock_service.regenerate_embeddings.assert_called_once_with(library_id)

        # Verify progress
        assert job.progress.percentage == 100.0


class TestBatchDeleteHandler:
    """Test batch delete job handler."""

    @pytest.mark.asyncio
    async def test_batch_delete_success(self):
        """Test successful batch delete."""
        doc_ids = [str(uuid4()), str(uuid4()), str(uuid4())]

        job = Job(
            job_type=JobType.BATCH_DELETE,
            params={"document_ids": doc_ids},
        )

        # Mock library service
        mock_service = Mock()
        mock_service.delete_documents_batch.return_value = (
            [doc_ids[0], doc_ids[1]],  # successful
            [doc_ids[2]],  # failed
        )

        result = await handle_batch_delete(job, mock_service)

        assert result["deleted"] == 2
        assert result["failed"] == 1

        # Verify progress
        assert job.progress.current == 2
        assert job.progress.total == 3

    @pytest.mark.asyncio
    async def test_batch_delete_all_successful(self):
        """Test batch delete with all documents deleted successfully."""
        doc_ids = [str(uuid4()), str(uuid4())]

        job = Job(
            job_type=JobType.BATCH_DELETE,
            params={"document_ids": doc_ids},
        )

        mock_service = Mock()
        mock_service.delete_documents_batch.return_value = (
            [doc_ids[0], doc_ids[1]],  # all successful
            [],  # no failures
        )

        result = await handle_batch_delete(job, mock_service)

        assert result["deleted"] == 2
        assert result["failed"] == 0


class TestBatchExportHandler:
    """Test batch export job handler."""

    @pytest.fixture
    def mock_library(self):
        """Create a mock library with documents."""
        # Create mock chunks
        chunk1 = Mock()
        chunk1.id = uuid4()
        chunk1.text = "Test chunk 1"
        chunk1.metadata = Mock()
        chunk1.metadata.model_dump.return_value = {"page": 1}
        chunk1.embedding = np.array([0.1, 0.2, 0.3])

        chunk2 = Mock()
        chunk2.id = uuid4()
        chunk2.text = "Test chunk 2"
        chunk2.metadata = Mock()
        chunk2.metadata.model_dump.return_value = {"page": 2}
        chunk2.embedding = np.array([0.4, 0.5, 0.6])

        # Create mock document
        doc = Mock()
        doc.id = uuid4()
        doc.metadata = Mock()
        doc.metadata.title = "Test Document"
        doc.metadata.author = "Test Author"
        doc.chunks = [chunk1, chunk2]

        # Create mock library
        library = Mock()
        library.documents = [doc]

        return library

    @pytest.mark.asyncio
    async def test_batch_export_json(self, mock_library):
        """Test batch export in JSON format."""
        library_id = uuid4()

        job = Job(
            job_type=JobType.BATCH_EXPORT,
            library_id=library_id,
            params={"format": "json", "include_embeddings": False},
        )

        # Mock library service
        mock_service = Mock()
        mock_service.get_library.return_value = mock_library

        result = await handle_batch_export(job, mock_service)

        assert result["documents_exported"] == 1
        assert result["format"] == "json"
        assert result["file_size_bytes"] > 0
        assert result["library_id"] == str(library_id)

        mock_service.get_library.assert_called_once_with(library_id)

        # Verify progress
        assert job.progress.percentage == 100.0

    @pytest.mark.asyncio
    async def test_batch_export_with_embeddings(self, mock_library):
        """Test batch export including embeddings."""
        library_id = uuid4()

        job = Job(
            job_type=JobType.BATCH_EXPORT,
            library_id=library_id,
            params={"format": "json", "include_embeddings": True},
        )

        mock_service = Mock()
        mock_service.get_library.return_value = mock_library

        result = await handle_batch_export(job, mock_service)

        assert result["documents_exported"] == 1
        # File size should be larger with embeddings
        assert result["file_size_bytes"] > 100

    @pytest.mark.asyncio
    async def test_batch_export_library_not_found(self):
        """Test batch export with nonexistent library."""
        library_id = uuid4()

        job = Job(
            job_type=JobType.BATCH_EXPORT,
            library_id=library_id,
            params={"format": "json"},
        )

        # Mock service returning None (library not found)
        mock_service = Mock()
        mock_service.get_library.return_value = None

        # Should raise ValueError
        with pytest.raises(ValueError, match="Library .* not found"):
            await handle_batch_export(job, mock_service)

    @pytest.mark.asyncio
    async def test_batch_export_default_format(self, mock_library):
        """Test batch export with default format."""
        library_id = uuid4()

        job = Job(
            job_type=JobType.BATCH_EXPORT,
            library_id=library_id,
            params={},  # No format specified
        )

        mock_service = Mock()
        mock_service.get_library.return_value = mock_library

        result = await handle_batch_export(job, mock_service)

        # Should default to json
        assert result["format"] == "json"


class TestHandlerRegistration:
    """Test handler registration functionality."""

    @pytest.mark.asyncio
    async def test_register_default_handlers(self):
        """Test registering all default handlers."""
        job_queue = JobQueue(num_workers=2)
        mock_service = Mock()

        # Register handlers
        register_default_handlers(job_queue, mock_service)

        # Verify all handlers are registered
        assert JobType.BATCH_IMPORT in job_queue._handlers
        assert JobType.INDEX_REBUILD in job_queue._handlers
        assert JobType.INDEX_OPTIMIZE in job_queue._handlers
        assert JobType.REGENERATE_EMBEDDINGS in job_queue._handlers
        assert JobType.BATCH_DELETE in job_queue._handlers
        assert JobType.BATCH_EXPORT in job_queue._handlers

    @pytest.mark.asyncio
    async def test_registered_handlers_work(self):
        """Test that registered handlers are callable."""
        job_queue = JobQueue(num_workers=2)
        mock_service = Mock()
        mock_service.add_documents_batch.return_value = ([], [], 0)

        register_default_handlers(job_queue, mock_service)

        await job_queue.start()

        # Submit a job
        library_id = uuid4()
        job_id = await job_queue.submit(
            JobType.BATCH_IMPORT,
            {"documents": []},
            library_id=library_id,
        )

        # Wait for job to complete
        import asyncio
        await asyncio.sleep(0.2)
        await job_queue.stop()

        # Verify job was executed
        job = job_queue.get_job(job_id)
        assert job.status.value in ["completed", "failed"]  # Should have been processed


class TestHandlerProgressTracking:
    """Test progress tracking in handlers."""

    @pytest.mark.asyncio
    async def test_batch_import_progress_updates(self):
        """Test that batch import updates progress correctly."""
        library_id = uuid4()
        documents_data = [{"title": f"Doc {i}"} for i in range(10)]

        job = Job(
            job_type=JobType.BATCH_IMPORT,
            library_id=library_id,
            params={"documents": documents_data},
        )

        mock_service = Mock()
        mock_service.add_documents_batch.return_value = (
            ["doc" + str(i) for i in range(10)],
            [],
            20,
        )

        # Check initial progress
        assert job.progress.current == 0
        assert job.progress.total == 0

        await handle_batch_import(job, mock_service)

        # Progress should be updated
        assert job.progress.current == 10
        assert job.progress.total == 10
        assert job.progress.percentage == 100.0
        assert "complete" in job.progress.message.lower()

    @pytest.mark.asyncio
    async def test_index_rebuild_progress_updates(self):
        """Test that index rebuild updates progress correctly."""
        library_id = uuid4()

        job = Job(
            job_type=JobType.INDEX_REBUILD,
            library_id=library_id,
            params={},
        )

        mock_service = Mock()
        mock_service.rebuild_index.return_value = ("old", "new", 100)

        await handle_index_rebuild(job, mock_service)

        # Progress should reach 100%
        assert job.progress.current == 100
        assert job.progress.total == 100
        assert job.progress.percentage == 100.0


class TestHandlerErrorHandling:
    """Test error handling in handlers."""

    @pytest.mark.asyncio
    async def test_batch_import_service_error(self):
        """Test batch import when service raises an error."""
        library_id = uuid4()

        job = Job(
            job_type=JobType.BATCH_IMPORT,
            library_id=library_id,
            params={"documents": [{"title": "Test"}]},
        )

        # Mock service that raises an error
        mock_service = Mock()
        mock_service.add_documents_batch.side_effect = Exception("Database error")

        # Should propagate the exception
        with pytest.raises(Exception, match="Database error"):
            await handle_batch_import(job, mock_service)

    @pytest.mark.asyncio
    async def test_index_rebuild_service_error(self):
        """Test index rebuild when service raises an error."""
        library_id = uuid4()

        job = Job(
            job_type=JobType.INDEX_REBUILD,
            library_id=library_id,
            params={},
        )

        mock_service = Mock()
        mock_service.rebuild_index.side_effect = ValueError("Invalid index type")

        with pytest.raises(ValueError, match="Invalid index type"):
            await handle_index_rebuild(job, mock_service)


class TestHandlerIntegration:
    """Integration tests for handlers with job queue."""

    @pytest.mark.asyncio
    async def test_end_to_end_batch_import(self):
        """Test batch import from submission to completion."""
        job_queue = JobQueue(num_workers=1)
        mock_service = Mock()
        mock_service.add_documents_batch.return_value = (["doc1"], [], 2)

        register_default_handlers(job_queue, mock_service)

        await job_queue.start()

        library_id = uuid4()
        job_id = await job_queue.submit(
            JobType.BATCH_IMPORT,
            {"documents": [{"title": "Test"}]},
            library_id=library_id,
        )

        # Wait for completion
        import asyncio
        await asyncio.sleep(0.3)
        await job_queue.stop()

        job = job_queue.get_job(job_id)
        assert job.status.value == "completed"
        assert job.result["imported"] == 1
