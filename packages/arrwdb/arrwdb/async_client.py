"""
Asynchronous Python client for arrwDB API.

Requires: pip install arrwdb[async]
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

try:
    import aiohttp
except ImportError:
    raise ImportError(
        "Async client requires aiohttp. Install with: pip install arrwdb[async]"
    )

logger = logging.getLogger(__name__)


class ArrwDBException(Exception):
    """Base exception for arrwDB client errors."""

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class AsyncArrwDBClient:
    """
    Async Python client for the arrwDB API.

    Example:
        >>> async with AsyncArrwDBClient("http://localhost:8000") as client:
        ...     library = await client.create_library("my-lib", index_type="hnsw")
        ...     results = await client.search(library["id"], "query")
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: int = 30,
        api_key: Optional[str] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_prefix = "/v1"
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.api_key = api_key
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            self._session = aiohttp.ClientSession(
                timeout=self.timeout,
                headers=headers,
            )
        return self._session

    async def _request(
        self,
        method: str,
        endpoint: str,
        **kwargs: Any,
    ) -> Any:
        if not endpoint.startswith(self.api_prefix) and not endpoint.startswith("/health"):
            if not endpoint.startswith("/api/"):
                endpoint = f"{self.api_prefix}{endpoint}"

        url = f"{self.base_url}{endpoint}"
        session = await self._get_session()

        try:
            async with session.request(method, url, **kwargs) as response:
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientError as e:
            logger.error(f"Request failed: {e}")
            raise ArrwDBException(f"Request failed: {e}")

    # =========================================================================
    # Health
    # =========================================================================

    async def health_check(self) -> Dict[str, Any]:
        return await self._request("GET", "/health")

    async def readiness_check(self) -> Dict[str, Any]:
        return await self._request("GET", "/ready")

    # =========================================================================
    # Libraries
    # =========================================================================

    async def create_library(
        self,
        name: str,
        description: Optional[str] = None,
        index_type: str = "brute_force",
        embedding_model: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload = {
            "name": name,
            "description": description,
            "index_type": index_type,
            "embedding_model": embedding_model,
        }
        return await self._request("POST", "/libraries", json=payload)

    async def get_library(self, library_id: str) -> Dict[str, Any]:
        return await self._request("GET", f"/libraries/{library_id}")

    async def list_libraries(self) -> List[Dict[str, Any]]:
        return await self._request("GET", "/libraries")

    async def delete_library(self, library_id: str) -> None:
        await self._request("DELETE", f"/libraries/{library_id}")

    # =========================================================================
    # Documents
    # =========================================================================

    async def add_document(
        self,
        library_id: str,
        title: str,
        texts: List[str],
        author: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        payload = {
            "title": title,
            "texts": texts,
            "author": author,
            "tags": tags or [],
        }
        return await self._request(
            "POST", f"/libraries/{library_id}/documents", json=payload
        )

    async def get_document(self, document_id: str) -> Dict[str, Any]:
        return await self._request("GET", f"/documents/{document_id}")

    async def delete_document(self, document_id: str) -> None:
        await self._request("DELETE", f"/documents/{document_id}")

    # =========================================================================
    # Search
    # =========================================================================

    async def search(
        self,
        library_id: str,
        query: str,
        k: int = 10,
        distance_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"query": query, "k": k}
        if distance_threshold is not None:
            payload["distance_threshold"] = distance_threshold
        return await self._request(
            "POST", f"/libraries/{library_id}/search", json=payload
        )

    async def temperature_search(
        self,
        corpus_id: str,
        query_text: str,
        k: int = 10,
        temperature: float = 1.0,
    ) -> Dict[str, Any]:
        payload = {
            "query_text": query_text,
            "k": k,
            "temperature": temperature,
        }
        return await self._request(
            "POST", f"/temperature-search/corpora/{corpus_id}/search", json=payload
        )

    # =========================================================================
    # Context Manager
    # =========================================================================

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None

    async def __aenter__(self) -> "AsyncArrwDBClient":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()
