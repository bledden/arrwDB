"""
Temporal worker for executing workflows and activities.

This worker connects to a Temporal server and processes workflow and
activity tasks from the task queue.
"""

import asyncio
import logging
import os
from temporalio.client import Client
from temporalio.worker import Worker

from temporal.workflows import RAGWorkflow, BatchEmbedWorkflow
from temporal.activities import (
    preprocess_query,
    embed_query,
    retrieve_chunks,
    rerank_results,
    generate_answer,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """
    Start the Temporal worker.

    The worker:
    - Connects to the Temporal server
    - Registers workflows and activities
    - Polls for and executes tasks
    """
    # Get configuration from environment
    temporal_host = os.getenv("TEMPORAL_HOST", "localhost:7233")
    temporal_namespace = os.getenv("TEMPORAL_NAMESPACE", "default")
    task_queue = os.getenv("TEMPORAL_TASK_QUEUE", "vector-db-task-queue")

    logger.info(f"Connecting to Temporal at {temporal_host}")
    logger.info(f"Namespace: {temporal_namespace}")
    logger.info(f"Task Queue: {task_queue}")

    # Create client
    client = await Client.connect(
        temporal_host,
        namespace=temporal_namespace,
    )

    logger.info("Connected to Temporal server")

    # Create worker
    worker = Worker(
        client,
        task_queue=task_queue,
        workflows=[RAGWorkflow, BatchEmbedWorkflow],
        activities=[
            preprocess_query,
            embed_query,
            retrieve_chunks,
            rerank_results,
            generate_answer,
        ],
    )

    logger.info("Worker registered, starting to process tasks...")
    logger.info("Press Ctrl+C to stop")

    # Run the worker
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
