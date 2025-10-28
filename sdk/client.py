"""
Python SDK client for the Vector Database API.

This module provides a high-level Python client for interacting with
the Vector Database REST API, including webhooks, jobs, and health monitoring.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Callable
from uuid import UUID

import requests

logger = logging.getLogger(__name__)


class VectorDBException(Exception):
    """Base exception for Vector DB client errors."""

    pass


class VectorDBClient:
    """
    Python client for the Vector Database API.

    This client provides a convenient Pythonic interface for all API operations:
    - Creating and managing libraries
    - Adding and querying documents
    - Performing vector similarity search
    - Retrieving statistics

    Example:
        ```python
        client = VectorDBClient("http://localhost:8000")

        # Create a library
        library = client.create_library(
            name="Research Papers",
            index_type="hnsw"
        )

        # Add a document
        doc = client.add_document(
            library_id=library["id"],
            title="Introduction to ML",
            texts=["Machine learning is...", "Deep learning uses..."]
        )

        # Search
        results = client.search(
            library_id=library["id"],
            query="What is machine learning?",
            k=5
        )
        ```
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: int = 30,
        verify_ssl: bool = True,
    ):
        """
        Initialize the Vector DB client.

        Args:
            base_url: Base URL of the API server.
            timeout: Request timeout in seconds.
            verify_ssl: Whether to verify SSL certificates.
        """
        self.base_url = base_url.rstrip("/")
        self.api_prefix = "/v1"  # API version prefix
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        self.session = requests.Session()

    def _request(
        self, method: str, endpoint: str, **kwargs
    ) -> requests.Response:
        """
        Make an HTTP request to the API.

        Args:
            method: HTTP method (GET, POST, DELETE, etc.)
            endpoint: API endpoint path.
            **kwargs: Additional arguments for requests.

        Returns:
            Response object.

        Raises:
            VectorDBException: If request fails.
        """
        # Add API version prefix if not already present
        if not endpoint.startswith(self.api_prefix) and not endpoint.startswith("/health"):
            endpoint = f"{self.api_prefix}{endpoint}"
        url = f"{self.base_url}{endpoint}"
        kwargs.setdefault("timeout", self.timeout)
        kwargs.setdefault("verify", self.verify_ssl)

        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            if hasattr(e, "response") and e.response is not None:
                try:
                    error_data = e.response.json()
                    raise VectorDBException(
                        f"{error_data.get('error', 'Unknown error')}: "
                        f"{error_data.get('detail', str(e))}"
                    )
                except (ValueError, requests.exceptions.JSONDecodeError):
                    raise VectorDBException(f"Request failed: {e}")
            raise VectorDBException(f"Request failed: {e}")

    def health_check(self) -> Dict[str, Any]:
        """
        Check if the API is healthy.

        Returns:
            Health status dictionary.
        """
        response = self._request("GET", "/health")
        return response.json()

    # Library Operations

    def create_library(
        self,
        name: str,
        description: Optional[str] = None,
        index_type: str = "brute_force",
        embedding_model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create a new library.

        Args:
            name: Library name.
            description: Optional description.
            index_type: Index type (brute_force, kd_tree, lsh, hnsw).
            embedding_model: Optional embedding model override.

        Returns:
            Created library data.
        """
        payload = {
            "name": name,
            "description": description,
            "index_type": index_type,
            "embedding_model": embedding_model,
        }

        response = self._request("POST", "/libraries", json=payload)
        return response.json()

    def get_library(self, library_id: str) -> Dict[str, Any]:
        """
        Get a library by ID.

        Args:
            library_id: Library UUID.

        Returns:
            Library data.
        """
        response = self._request("GET", f"/libraries/{library_id}")
        return response.json()

    def list_libraries(self) -> List[Dict[str, Any]]:
        """
        List all libraries.

        Returns:
            List of library summaries.
        """
        response = self._request("GET", "/libraries")
        return response.json()

    def delete_library(self, library_id: str) -> None:
        """
        Delete a library.

        Args:
            library_id: Library UUID.
        """
        self._request("DELETE", f"/libraries/{library_id}")

    def get_library_statistics(self, library_id: str) -> Dict[str, Any]:
        """
        Get statistics for a library.

        Args:
            library_id: Library UUID.

        Returns:
            Statistics dictionary.
        """
        response = self._request("GET", f"/libraries/{library_id}/statistics")
        return response.json()

    # Document Operations

    def add_document(
        self,
        library_id: str,
        title: str,
        texts: List[str],
        author: Optional[str] = None,
        document_type: str = "text",
        source_url: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Add a document with text chunks.

        Embeddings will be generated automatically.

        Args:
            library_id: Library UUID.
            title: Document title.
            texts: List of text chunks.
            author: Optional author.
            document_type: Document type.
            source_url: Optional source URL.
            tags: Optional tags.

        Returns:
            Created document data.
        """
        payload = {
            "title": title,
            "texts": texts,
            "author": author,
            "document_type": document_type,
            "source_url": source_url,
            "tags": tags or [],
        }

        response = self._request(
            "POST", f"/libraries/{library_id}/documents", json=payload
        )
        return response.json()

    def add_document_with_embeddings(
        self,
        library_id: str,
        title: str,
        chunks: List[Tuple[str, List[float]]],
        author: Optional[str] = None,
        document_type: str = "text",
        source_url: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Add a document with pre-computed embeddings.

        Args:
            library_id: Library UUID.
            title: Document title.
            chunks: List of (text, embedding) tuples.
            author: Optional author.
            document_type: Document type.
            source_url: Optional source URL.
            tags: Optional tags.

        Returns:
            Created document data.
        """
        payload = {
            "title": title,
            "chunks": [
                {"text": text, "embedding": embedding}
                for text, embedding in chunks
            ],
            "author": author,
            "document_type": document_type,
            "source_url": source_url,
            "tags": tags or [],
        }

        response = self._request(
            "POST",
            f"/libraries/{library_id}/documents/with-embeddings",
            json=payload,
        )
        return response.json()

    def get_document(self, document_id: str) -> Dict[str, Any]:
        """
        Get a document by ID.

        Args:
            document_id: Document UUID.

        Returns:
            Document data.
        """
        response = self._request("GET", f"/documents/{document_id}")
        return response.json()

    def delete_document(self, document_id: str) -> None:
        """
        Delete a document.

        Args:
            document_id: Document UUID.
        """
        self._request("DELETE", f"/documents/{document_id}")

    # Search Operations

    def search(
        self,
        library_id: str,
        query: str,
        k: int = 10,
        distance_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Search a library with a text query.

        Args:
            library_id: Library UUID.
            query: Search query text.
            k: Number of results to return.
            distance_threshold: Optional maximum distance.

        Returns:
            Search results.
        """
        payload = {
            "query": query,
            "k": k,
            "distance_threshold": distance_threshold,
        }

        response = self._request(
            "POST", f"/libraries/{library_id}/search", json=payload
        )
        return response.json()

    def search_with_embedding(
        self,
        library_id: str,
        embedding: List[float],
        k: int = 10,
        distance_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Search a library with a pre-computed embedding.

        Args:
            library_id: Library UUID.
            embedding: Query embedding.
            k: Number of results to return.
            distance_threshold: Optional maximum distance.

        Returns:
            Search results.
        """
        payload = {
            "embedding": embedding,
            "k": k,
            "distance_threshold": distance_threshold,
        }

        response = self._request(
            "POST", f"/libraries/{library_id}/search/embedding", json=payload
        )
        return response.json()

    def search_with_filters(
        self,
        library_id: str,
        query: str,
        metadata_filters: List[Dict[str, Any]],
        k: int = 10,
        distance_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Search a library with text query and apply metadata filters.

        This method performs vector search first, then filters results
        based on chunk metadata. All filters use AND logic.

        Available metadata fields:
        - created_at: Chunk creation timestamp (datetime)
        - page_number: Page number (int, optional)
        - chunk_index: Position in document (int)
        - source_document_id: Parent document UUID

        Supported operators:
        - eq, ne: Equality/inequality
        - gt, lt, gte, lte: Numeric comparisons
        - in: Check if value is in list
        - contains: String contains substring, or list contains element

        Example:
            filters = [
                {"field": "chunk_index", "operator": "gte", "value": 2},
                {"field": "chunk_index", "operator": "lt", "value": 10}
            ]
            results = client.search_with_filters(
                library_id="...",
                query="machine learning",
                metadata_filters=filters,
                k=10
            )

        Args:
            library_id: Library UUID.
            query: Search query text.
            metadata_filters: List of filter dictionaries with keys:
                - field: Metadata field name
                - operator: Comparison operator
                - value: Value to compare against
            k: Number of results to return.
            distance_threshold: Optional maximum distance.

        Returns:
            Search results matching the query AND all metadata filters.
        """
        payload = {
            "query": query,
            "k": k,
            "metadata_filters": metadata_filters,
            "distance_threshold": distance_threshold,
        }

        response = self._request(
            "POST", f"/libraries/{library_id}/search/filtered", json=payload
        )
        return response.json()

    # Webhook Operations

    def create_webhook(
        self,
        url: str,
        events: List[str],
        description: Optional[str] = None,
        max_retries: int = 3,
        timeout_seconds: int = 30,
    ) -> Dict[str, Any]:
        """
        Create a webhook to receive event notifications.

        Args:
            url: Webhook endpoint URL
            events: List of event types to subscribe to
            description: Optional description
            max_retries: Maximum retry attempts on failure
            timeout_seconds: Request timeout

        Returns:
            Created webhook with secret for HMAC verification
        """
        payload = {
            "url": url,
            "events": events,
            "description": description,
            "max_retries": max_retries,
            "timeout_seconds": timeout_seconds,
        }

        response = self._request("POST", "/api/v1/webhooks", json=payload)
        return response.json()

    def list_webhooks(self) -> List[Dict[str, Any]]:
        """
        List all registered webhooks.

        Returns:
            List of webhooks
        """
        response = self._request("GET", "/api/v1/webhooks")
        return response.json()["webhooks"]

    def get_webhook(self, webhook_id: str) -> Dict[str, Any]:
        """
        Get webhook details.

        Args:
            webhook_id: Webhook UUID

        Returns:
            Webhook details
        """
        response = self._request("GET", f"/api/v1/webhooks/{webhook_id}")
        return response.json()

    def update_webhook(
        self,
        webhook_id: str,
        url: Optional[str] = None,
        events: Optional[List[str]] = None,
        description: Optional[str] = None,
        status: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Update webhook configuration.

        Args:
            webhook_id: Webhook UUID
            url: New URL
            events: New event subscriptions
            description: New description
            status: New status (active, paused, disabled)

        Returns:
            Updated webhook
        """
        payload = {}
        if url:
            payload["url"] = url
        if events:
            payload["events"] = events
        if description:
            payload["description"] = description
        if status:
            payload["status"] = status

        response = self._request(
            "PATCH", f"/api/v1/webhooks/{webhook_id}", json=payload
        )
        return response.json()

    def delete_webhook(self, webhook_id: str) -> None:
        """
        Delete a webhook.

        Args:
            webhook_id: Webhook UUID
        """
        self._request("DELETE", f"/api/v1/webhooks/{webhook_id}")

    def get_webhook_deliveries(
        self, webhook_id: str, status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get webhook delivery history.

        Args:
            webhook_id: Webhook UUID
            status: Optional status filter (success, failed, pending)

        Returns:
            List of delivery attempts
        """
        params = {}
        if status:
            params["status"] = status

        response = self._request(
            "GET", f"/api/v1/webhooks/{webhook_id}/deliveries", params=params
        )
        return response.json()["deliveries"]

    def get_webhook_stats(self, webhook_id: str) -> Dict[str, Any]:
        """
        Get webhook statistics.

        Args:
            webhook_id: Webhook UUID

        Returns:
            Statistics including success rate and delivery counts
        """
        response = self._request("GET", f"/api/v1/webhooks/{webhook_id}/stats")
        return response.json()

    def test_webhook(self, webhook_id: str) -> Dict[str, Any]:
        """
        Send a test event to a webhook.

        Args:
            webhook_id: Webhook UUID

        Returns:
            Test delivery result
        """
        response = self._request("POST", f"/api/v1/webhooks/{webhook_id}/test")
        return response.json()

    # Health & Monitoring

    def readiness_check(self) -> Dict[str, Any]:
        """
        Check if API and all dependencies are ready.

        Returns:
            Readiness status with dependency checks
        """
        response = self._request("GET", "/ready")
        return response.json()

    def detailed_health(self) -> Dict[str, Any]:
        """
        Get detailed health information for all components.

        Returns:
            Detailed component status
        """
        response = self._request("GET", "/health/detailed")
        return response.json()

    # Job Management (for background operations)

    def submit_job(
        self, job_type: str, payload: Dict[str, Any], wait: bool = False
    ) -> Dict[str, Any]:
        """
        Submit a background job.

        Args:
            job_type: Type of job (batch_import, index_rebuild, etc.)
            payload: Job-specific payload
            wait: If True, block until job completes

        Returns:
            Job status
        """
        response = self._request(
            "POST", f"/v1/jobs/{job_type}", json=payload
        )
        job = response.json()

        if wait:
            # Poll for completion
            import time
            job_id = job["id"]
            while True:
                status = self.get_job_status(job_id)
                if status["status"] in ["completed", "failed", "cancelled"]:
                    return status
                time.sleep(1)

        return job

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get job status.

        Args:
            job_id: Job UUID

        Returns:
            Job status including progress and result
        """
        response = self._request("GET", f"/v1/jobs/{job_id}")
        return response.json()

    def cancel_job(self, job_id: str) -> Dict[str, Any]:
        """
        Cancel a running job.

        Args:
            job_id: Job UUID

        Returns:
            Updated job status
        """
        response = self._request("POST", f"/v1/jobs/{job_id}/cancel")
        return response.json()

    def list_jobs(
        self, status: Optional[str] = None, job_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List jobs with optional filtering.

        Args:
            status: Filter by status (pending, running, completed, failed)
            job_type: Filter by job type

        Returns:
            List of jobs
        """
        params = {}
        if status:
            params["status"] = status
        if job_type:
            params["type"] = job_type

        response = self._request("GET", "/v1/jobs", params=params)
        return response.json()

    def __repr__(self) -> str:
        """String representation."""
        return f"VectorDBClient(base_url='{self.base_url}')"

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.session.close()
