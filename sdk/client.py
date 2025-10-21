"""
Python SDK client for the Vector Database API.

This module provides a high-level Python client for interacting with
the Vector Database REST API.
"""

import requests
from typing import List, Optional, Dict, Any, Tuple
from uuid import UUID
import logging

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
                except:
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

    def __repr__(self) -> str:
        """String representation."""
        return f"VectorDBClient(base_url='{self.base_url}')"

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.session.close()
