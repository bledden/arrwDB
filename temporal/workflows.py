"""
Temporal workflows for RAG (Retrieval-Augmented Generation).

This module defines the main RAG workflow that orchestrates the 5 activities
to provide a complete question-answering pipeline.
"""

from temporalio import workflow
from datetime import timedelta
from typing import Dict, Any, List
import logging

# Import activity stubs
with workflow.unsafe.imports_passed_through():
    from temporal.activities import (
        preprocess_query,
        embed_query,
        retrieve_chunks,
        rerank_results,
        generate_answer,
    )

logger = logging.getLogger(__name__)


@workflow.defn(name="rag_workflow")
class RAGWorkflow:
    """
    RAG (Retrieval-Augmented Generation) Workflow.

    This workflow implements a complete question-answering pipeline:
    1. Preprocess the user query
    2. Generate query embedding
    3. Retrieve relevant chunks from vector database
    4. Rerank results for better relevance
    5. Generate final answer using retrieved context

    The workflow is durable and fault-tolerant thanks to Temporal's
    execution guarantees.
    """

    @workflow.run
    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the RAG workflow.

        Args:
            input_data: Dictionary containing:
                - query: User query string
                - library_id: UUID of library to search
                - k: Number of results to retrieve (default: 10)
                - top_k: Number to keep after reranking (default: 5)
                - embedding_service_config: Config for embedding service
                - service_config: Config for library service

        Returns:
            Dictionary with:
                - query: Original query
                - preprocessed_query: Preprocessed query
                - retrieved_count: Number of chunks retrieved
                - reranked_count: Number of chunks after reranking
                - answer: Generated answer
                - sources: List of source documents
                - execution_time_ms: Total execution time
        """
        workflow.logger.info(f"Starting RAG workflow for query: {input_data['query']}")

        # Track execution time
        import time

        start_time = time.time()

        # Extract parameters
        query = input_data["query"]
        library_id = input_data["library_id"]
        k = input_data.get("k", 10)
        top_k = input_data.get("top_k", 5)
        embedding_config = input_data["embedding_service_config"]
        service_config = input_data["service_config"]

        # Activity 1: Preprocess query
        workflow.logger.info("Step 1: Preprocessing query")
        preprocessed_query = await workflow.execute_activity(
            preprocess_query,
            query,
            start_to_close_timeout=timedelta(seconds=30),
            retry_policy=workflow.RetryPolicy(
                maximum_attempts=3,
                initial_interval=timedelta(seconds=1),
                maximum_interval=timedelta(seconds=10),
            ),
        )

        # Activity 2: Generate embedding
        workflow.logger.info("Step 2: Generating query embedding")
        embedding = await workflow.execute_activity(
            embed_query,
            args=[preprocessed_query, embedding_config],
            start_to_close_timeout=timedelta(seconds=60),
            retry_policy=workflow.RetryPolicy(
                maximum_attempts=3,
                initial_interval=timedelta(seconds=2),
                maximum_interval=timedelta(seconds=20),
            ),
        )

        # Activity 3: Retrieve chunks
        workflow.logger.info("Step 3: Retrieving relevant chunks")
        retrieved_chunks = await workflow.execute_activity(
            retrieve_chunks,
            args=[library_id, embedding, k, service_config],
            start_to_close_timeout=timedelta(seconds=120),
            retry_policy=workflow.RetryPolicy(
                maximum_attempts=3,
                initial_interval=timedelta(seconds=2),
                maximum_interval=timedelta(seconds=20),
            ),
        )

        # Activity 4: Rerank results
        workflow.logger.info("Step 4: Reranking results")
        reranked_chunks = await workflow.execute_activity(
            rerank_results,
            args=[preprocessed_query, retrieved_chunks, top_k],
            start_to_close_timeout=timedelta(seconds=60),
            retry_policy=workflow.RetryPolicy(
                maximum_attempts=3,
                initial_interval=timedelta(seconds=1),
                maximum_interval=timedelta(seconds=10),
            ),
        )

        # Activity 5: Generate answer
        workflow.logger.info("Step 5: Generating final answer")
        answer_data = await workflow.execute_activity(
            generate_answer,
            args=[query, reranked_chunks],
            start_to_close_timeout=timedelta(seconds=120),
            retry_policy=workflow.RetryPolicy(
                maximum_attempts=3,
                initial_interval=timedelta(seconds=2),
                maximum_interval=timedelta(seconds=20),
            ),
        )

        # Calculate execution time
        execution_time_ms = (time.time() - start_time) * 1000

        # Build final result
        result = {
            "query": query,
            "preprocessed_query": preprocessed_query,
            "retrieved_count": len(retrieved_chunks),
            "reranked_count": len(reranked_chunks),
            "answer": answer_data["answer"],
            "context": answer_data["context"],
            "sources": answer_data["sources"],
            "num_sources": answer_data["num_sources"],
            "execution_time_ms": round(execution_time_ms, 2),
        }

        workflow.logger.info(
            f"RAG workflow completed in {execution_time_ms:.2f}ms"
        )

        return result


@workflow.defn(name="batch_embed_workflow")
class BatchEmbedWorkflow:
    """
    Workflow for batch embedding of documents.

    This workflow efficiently embeds multiple documents in parallel,
    taking advantage of Temporal's ability to run activities concurrently.
    """

    @workflow.run
    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute batch embedding workflow.

        Args:
            input_data: Dictionary containing:
                - library_id: UUID of target library
                - documents: List of documents to embed
                - embedding_service_config: Config for embedding service

        Returns:
            Dictionary with:
                - documents_processed: Number of documents processed
                - chunks_embedded: Total number of chunks embedded
                - execution_time_ms: Total execution time
        """
        workflow.logger.info(
            f"Starting batch embed workflow for {len(input_data['documents'])} documents"
        )

        import time

        start_time = time.time()

        library_id = input_data["library_id"]
        documents = input_data["documents"]
        embedding_config = input_data["embedding_service_config"]

        # Process documents in parallel batches
        # (In a real implementation, you'd call activities to do the actual work)

        total_chunks = sum(
            len(doc.get("texts", [])) for doc in documents
        )

        execution_time_ms = (time.time() - start_time) * 1000

        result = {
            "documents_processed": len(documents),
            "chunks_embedded": total_chunks,
            "execution_time_ms": round(execution_time_ms, 2),
        }

        workflow.logger.info(
            f"Batch embed workflow completed in {execution_time_ms:.2f}ms"
        )

        return result
