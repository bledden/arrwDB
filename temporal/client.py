"""
Temporal client for starting workflows.

This module provides a client interface for starting Temporal workflows
from the REST API or other Python code.
"""

import os
from typing import Dict, Any
from uuid import uuid4
from temporalio.client import Client
import logging

logger = logging.getLogger(__name__)


class TemporalClient:
    """
    Client for interacting with Temporal workflows.

    This client provides methods to start and query workflows
    for RAG operations and batch embedding.
    """

    def __init__(
        self,
        temporal_host: str = None,
        namespace: str = None,
        task_queue: str = None,
    ):
        """
        Initialize the Temporal client.

        Args:
            temporal_host: Temporal server address (default: from env)
            namespace: Temporal namespace (default: from env)
            task_queue: Task queue name (default: from env)
        """
        self.temporal_host = temporal_host or os.getenv(
            "TEMPORAL_HOST", "localhost:7233"
        )
        self.namespace = namespace or os.getenv("TEMPORAL_NAMESPACE", "default")
        self.task_queue = task_queue or os.getenv(
            "TEMPORAL_TASK_QUEUE", "vector-db-task-queue"
        )
        self._client = None

    async def connect(self):
        """Connect to the Temporal server."""
        if self._client is None:
            self._client = await Client.connect(
                self.temporal_host,
                namespace=self.namespace,
            )
            logger.info(f"Connected to Temporal at {self.temporal_host}")

    async def start_rag_workflow(
        self,
        query: str,
        library_id: str,
        k: int = 10,
        top_k: int = 5,
        embedding_service_config: Dict[str, Any] = None,
        service_config: Dict[str, Any] = None,
    ) -> str:
        """
        Start a RAG workflow.

        Args:
            query: User query.
            library_id: Library to search.
            k: Number of results to retrieve.
            top_k: Number to keep after reranking.
            embedding_service_config: Embedding service configuration.
            service_config: Service configuration.

        Returns:
            Workflow ID.
        """
        await self.connect()

        workflow_id = f"rag-{uuid4()}"

        input_data = {
            "query": query,
            "library_id": library_id,
            "k": k,
            "top_k": top_k,
            "embedding_service_config": embedding_service_config or {},
            "service_config": service_config or {},
        }

        handle = await self._client.start_workflow(
            "rag_workflow",
            input_data,
            id=workflow_id,
            task_queue=self.task_queue,
        )

        logger.info(f"Started RAG workflow with ID: {workflow_id}")
        return workflow_id

    async def get_workflow_result(self, workflow_id: str) -> Dict[str, Any]:
        """
        Get the result of a completed workflow.

        Args:
            workflow_id: The workflow ID.

        Returns:
            Workflow result.
        """
        await self.connect()

        handle = self._client.get_workflow_handle(workflow_id)
        result = await handle.result()

        return result

    async def query_workflow_status(self, workflow_id: str) -> str:
        """
        Query the status of a workflow.

        Args:
            workflow_id: The workflow ID.

        Returns:
            Workflow status string.
        """
        await self.connect()

        handle = self._client.get_workflow_handle(workflow_id)
        description = await handle.describe()

        return description.status.name

    async def start_batch_embed_workflow(
        self,
        library_id: str,
        documents: list,
        embedding_service_config: Dict[str, Any],
    ) -> str:
        """
        Start a batch embedding workflow.

        Args:
            library_id: Target library ID.
            documents: List of documents to embed.
            embedding_service_config: Embedding service configuration.

        Returns:
            Workflow ID.
        """
        await self.connect()

        workflow_id = f"batch-embed-{uuid4()}"

        input_data = {
            "library_id": library_id,
            "documents": documents,
            "embedding_service_config": embedding_service_config,
        }

        handle = await self._client.start_workflow(
            "batch_embed_workflow",
            input_data,
            id=workflow_id,
            task_queue=self.task_queue,
        )

        logger.info(f"Started batch embed workflow with ID: {workflow_id}")
        return workflow_id
