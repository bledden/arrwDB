"""
Synchronous Python client for arrwDB API.

This module provides a high-level Python client for interacting with
the arrwDB REST API, including libraries, documents, search, webhooks,
and background jobs.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import requests

logger = logging.getLogger(__name__)


class ArrwDBException(Exception):
    """Base exception for arrwDB client errors."""

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class ArrwDBClient:
    """
    Python client for the arrwDB API.

    This client provides a convenient Pythonic interface for all API operations:
    - Creating and managing libraries
    - Adding and querying documents
    - Performing vector similarity search
    - Managing webhooks
    - Background job operations
    - Novel features (Temperature Search, Index Oracle, etc.)

    Example:
        >>> client = ArrwDBClient("http://localhost:8000")
        >>>
        >>> # Create a library
        >>> library = client.create_library(
        ...     name="Research Papers",
        ...     index_type="hnsw"
        ... )
        >>>
        >>> # Add a document
        >>> doc = client.add_document(
        ...     library_id=library["id"],
        ...     title="Introduction to ML",
        ...     texts=["Machine learning is...", "Deep learning uses..."]
        ... )
        >>>
        >>> # Search
        >>> results = client.search(
        ...     library_id=library["id"],
        ...     query="What is machine learning?",
        ...     k=5
        ... )

    Attributes:
        base_url: The base URL of the arrwDB server.
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: int = 30,
        verify_ssl: bool = True,
        api_key: Optional[str] = None,
    ) -> None:
        """
        Initialize the arrwDB client.

        Args:
            base_url: Base URL of the API server.
            timeout: Request timeout in seconds.
            verify_ssl: Whether to verify SSL certificates.
            api_key: Optional API key for authentication.
        """
        self.base_url = base_url.rstrip("/")
        self.api_prefix = "/v1"
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        self.api_key = api_key
        self._session: Optional[requests.Session] = None

    @property
    def session(self) -> requests.Session:
        """Lazy-initialized session."""
        if self._session is None:
            self._session = requests.Session()
            if self.api_key:
                self._session.headers["Authorization"] = f"Bearer {self.api_key}"
        return self._session

    def _request(
        self,
        method: str,
        endpoint: str,
        **kwargs: Any,
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
            ArrwDBException: If request fails.
        """
        # Add API version prefix if not already present
        if not endpoint.startswith(self.api_prefix) and not endpoint.startswith("/health"):
            if not endpoint.startswith("/api/"):
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
            status_code = None
            if hasattr(e, "response") and e.response is not None:
                status_code = e.response.status_code
                try:
                    error_data = e.response.json()
                    raise ArrwDBException(
                        f"{error_data.get('error', 'Unknown error')}: "
                        f"{error_data.get('detail', str(e))}",
                        status_code=status_code,
                    )
                except (ValueError, requests.exceptions.JSONDecodeError):
                    pass
            raise ArrwDBException(f"Request failed: {e}", status_code=status_code)

    # =========================================================================
    # Health & Monitoring
    # =========================================================================

    def health_check(self) -> Dict[str, Any]:
        """
        Check if the API is healthy.

        Returns:
            Health status dictionary with 'status' and 'uptime_seconds'.
        """
        response = self._request("GET", "/health")
        return response.json()

    def readiness_check(self) -> Dict[str, Any]:
        """
        Check if API and all dependencies are ready.

        Returns:
            Readiness status with dependency checks.
        """
        response = self._request("GET", "/ready")
        return response.json()

    def detailed_health(self) -> Dict[str, Any]:
        """
        Get detailed health information for all components.

        Returns:
            Detailed component status.
        """
        response = self._request("GET", "/health/detailed")
        return response.json()

    # =========================================================================
    # Library Operations
    # =========================================================================

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
            index_type: Index type (brute_force, kd_tree, lsh, hnsw, ivf).
            embedding_model: Optional embedding model override.

        Returns:
            Created library data with 'id', 'name', etc.
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
            Statistics dictionary with document/chunk counts.
        """
        response = self._request("GET", f"/libraries/{library_id}/statistics")
        return response.json()

    # =========================================================================
    # Document Operations
    # =========================================================================

    def add_document(
        self,
        library_id: str,
        title: str,
        texts: List[str],
        author: Optional[str] = None,
        document_type: str = "text",
        source_url: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Add a document with text chunks (embeddings generated automatically).

        Args:
            library_id: Library UUID.
            title: Document title.
            texts: List of text chunks to embed.
            author: Optional author.
            document_type: Document type.
            source_url: Optional source URL.
            tags: Optional tags list.
            metadata: Optional custom metadata.

        Returns:
            Created document data.
        """
        payload: Dict[str, Any] = {
            "title": title,
            "texts": texts,
            "author": author,
            "document_type": document_type,
            "source_url": source_url,
            "tags": tags or [],
        }
        if metadata:
            payload["metadata"] = metadata

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

    # =========================================================================
    # Search Operations
    # =========================================================================

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
            Search results with 'results' list.
        """
        payload: Dict[str, Any] = {
            "query": query,
            "k": k,
        }
        if distance_threshold is not None:
            payload["distance_threshold"] = distance_threshold

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
            embedding: Query embedding vector.
            k: Number of results to return.
            distance_threshold: Optional maximum distance.

        Returns:
            Search results.
        """
        payload: Dict[str, Any] = {
            "embedding": embedding,
            "k": k,
        }
        if distance_threshold is not None:
            payload["distance_threshold"] = distance_threshold

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
        Search with metadata filters.

        Args:
            library_id: Library UUID.
            query: Search query text.
            metadata_filters: List of filter dicts with 'field', 'operator', 'value'.
            k: Number of results to return.
            distance_threshold: Optional maximum distance.

        Returns:
            Filtered search results.

        Example:
            >>> filters = [
            ...     {"field": "chunk_index", "operator": "gte", "value": 2},
            ...     {"field": "tags", "operator": "contains", "value": "ml"}
            ... ]
            >>> results = client.search_with_filters(lib_id, "query", filters)
        """
        payload: Dict[str, Any] = {
            "query": query,
            "k": k,
            "metadata_filters": metadata_filters,
        }
        if distance_threshold is not None:
            payload["distance_threshold"] = distance_threshold

        response = self._request(
            "POST", f"/libraries/{library_id}/search/filtered", json=payload
        )
        return response.json()

    def hybrid_search(
        self,
        library_id: str,
        query: str,
        k: int = 10,
        vector_weight: float = 0.7,
        metadata_weight: float = 0.3,
        field_boosts: Optional[Dict[str, float]] = None,
        recency_boost_enabled: bool = False,
        recency_half_life_days: int = 30,
    ) -> Dict[str, Any]:
        """
        Hybrid search combining vector similarity with metadata signals.

        Args:
            library_id: Library UUID.
            query: Search query.
            k: Number of results.
            vector_weight: Weight for vector similarity (0-1).
            metadata_weight: Weight for metadata scoring (0-1).
            field_boosts: Field-specific boost factors.
            recency_boost_enabled: Enable recency boosting.
            recency_half_life_days: Half-life for recency decay.

        Returns:
            Hybrid search results.
        """
        payload: Dict[str, Any] = {
            "query": query,
            "k": k,
            "scoring_config": {
                "vector_weight": vector_weight,
                "metadata_weight": metadata_weight,
                "field_boosts": field_boosts or {},
                "recency_boost_enabled": recency_boost_enabled,
                "recency_half_life_days": recency_half_life_days,
            },
        }
        response = self._request(
            "POST", f"/libraries/{library_id}/search/hybrid", json=payload
        )
        return response.json()

    # =========================================================================
    # Novel Features
    # =========================================================================

    def temperature_search(
        self,
        corpus_id: str,
        query_text: str,
        k: int = 10,
        temperature: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Temperature-controlled search for exploration vs exploitation.

        Args:
            corpus_id: Corpus/library UUID.
            query_text: Search query.
            k: Number of results.
            temperature: 0.0=greedy (top results), 2.0=exploratory (diverse).

        Returns:
            Search results with temperature-based sampling.
        """
        payload = {
            "query_text": query_text,
            "k": k,
            "temperature": temperature,
        }
        response = self._request(
            "POST", f"/temperature-search/corpora/{corpus_id}/search", json=payload
        )
        return response.json()

    def get_index_recommendation(self, corpus_id: str) -> Dict[str, Any]:
        """
        Get intelligent index type recommendation based on workload.

        Args:
            corpus_id: Corpus/library UUID.

        Returns:
            Recommendation with suggested index type and reasoning.
        """
        response = self._request(
            "GET", f"/index-oracle/corpora/{corpus_id}/analyze"
        )
        return response.json()

    def analyze_embedding_health(self, corpus_id: str) -> Dict[str, Any]:
        """
        Analyze embedding quality (outliers, degeneracy, drift).

        Args:
            corpus_id: Corpus/library UUID.

        Returns:
            Health analysis with statistical metrics.
        """
        response = self._request(
            "GET", f"/embedding-health/corpora/{corpus_id}/analyze"
        )
        return response.json()

    def cluster_vectors(
        self,
        corpus_id: str,
        n_clusters: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        K-means clustering of vectors with auto cluster estimation.

        Args:
            corpus_id: Corpus/library UUID.
            n_clusters: Number of clusters (auto-estimated if None).

        Returns:
            Clustering results with assignments and centroids.
        """
        payload: Dict[str, Any] = {}
        if n_clusters is not None:
            payload["n_clusters"] = n_clusters

        response = self._request(
            "POST", f"/clustering/corpora/{corpus_id}/cluster", json=payload
        )
        return response.json()

    def expand_query(
        self,
        query: str,
        strategy: str = "synonym",
    ) -> Dict[str, Any]:
        """
        Automatic query expansion/rewriting.

        Args:
            query: Original query.
            strategy: Expansion strategy (synonym, semantic, hybrid).

        Returns:
            Expanded queries with fusion weights.
        """
        payload = {
            "query": query,
            "strategy": strategy,
        }
        response = self._request("POST", "/query-expansion/expand", json=payload)
        return response.json()

    def detect_vector_drift(self, corpus_id: str) -> Dict[str, Any]:
        """
        Detect distribution drift in vectors over time.

        Args:
            corpus_id: Corpus/library UUID.

        Returns:
            Drift analysis with KS test statistics.
        """
        response = self._request(
            "GET", f"/vector-drift/corpora/{corpus_id}/analyze"
        )
        return response.json()

    # =========================================================================
    # Webhooks
    # =========================================================================

    def create_webhook(
        self,
        url: str,
        events: List[str],
        description: Optional[str] = None,
        max_retries: int = 3,
        timeout_seconds: int = 30,
    ) -> Dict[str, Any]:
        """
        Create a webhook for event notifications.

        Args:
            url: Webhook endpoint URL.
            events: Event types to subscribe to.
            description: Optional description.
            max_retries: Max retry attempts.
            timeout_seconds: Request timeout.

        Returns:
            Created webhook with 'id' and 'secret' for HMAC verification.
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
        """List all registered webhooks."""
        response = self._request("GET", "/api/v1/webhooks")
        return response.json().get("webhooks", [])

    def get_webhook(self, webhook_id: str) -> Dict[str, Any]:
        """Get webhook details."""
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
        """Update webhook configuration."""
        payload: Dict[str, Any] = {}
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
        """Delete a webhook."""
        self._request("DELETE", f"/api/v1/webhooks/{webhook_id}")

    def get_webhook_deliveries(
        self,
        webhook_id: str,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get webhook delivery history."""
        params = {}
        if status:
            params["status"] = status

        response = self._request(
            "GET", f"/api/v1/webhooks/{webhook_id}/deliveries", params=params
        )
        return response.json().get("deliveries", [])

    def get_webhook_stats(self, webhook_id: str) -> Dict[str, Any]:
        """Get webhook statistics."""
        response = self._request("GET", f"/api/v1/webhooks/{webhook_id}/stats")
        return response.json()

    def test_webhook(self, webhook_id: str) -> Dict[str, Any]:
        """Send a test event to a webhook."""
        response = self._request("POST", f"/api/v1/webhooks/{webhook_id}/test")
        return response.json()

    # =========================================================================
    # Background Jobs
    # =========================================================================

    def submit_job(
        self,
        job_type: str,
        payload: Dict[str, Any],
        wait: bool = False,
    ) -> Dict[str, Any]:
        """
        Submit a background job.

        Args:
            job_type: Type of job (batch_import, index_rebuild, etc.)
            payload: Job-specific payload.
            wait: If True, block until job completes.

        Returns:
            Job status.
        """
        response = self._request("POST", f"/jobs/{job_type}", json=payload)
        job = response.json()

        if wait:
            import time

            job_id = job["id"]
            while True:
                status = self.get_job_status(job_id)
                if status["status"] in ["completed", "failed", "cancelled"]:
                    return status
                time.sleep(1)

        return job

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get job status."""
        response = self._request("GET", f"/jobs/{job_id}")
        return response.json()

    def cancel_job(self, job_id: str) -> Dict[str, Any]:
        """Cancel a running job."""
        response = self._request("POST", f"/jobs/{job_id}/cancel")
        return response.json()

    def list_jobs(
        self,
        status: Optional[str] = None,
        job_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List jobs with optional filtering."""
        params = {}
        if status:
            params["status"] = status
        if job_type:
            params["type"] = job_type

        response = self._request("GET", "/jobs", params=params)
        return response.json()

    # =========================================================================
    # Context Manager Support
    # =========================================================================

    def close(self) -> None:
        """Close the client session."""
        if self._session is not None:
            self._session.close()
            self._session = None

    def __enter__(self) -> "ArrwDBClient":
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_val: Any,
        exc_tb: Any,
    ) -> None:
        """Context manager exit."""
        self.close()

    def __repr__(self) -> str:
        """String representation."""
        return f"ArrwDBClient(base_url='{self.base_url}')"
