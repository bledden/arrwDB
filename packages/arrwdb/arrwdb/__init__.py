"""
arrwDB Python SDK - Production Vector Database Client

A Python client library for arrwDB, featuring:
- Semantic search with multiple index types (HNSW, IVF, LSH, KD-Tree, Brute Force)
- 9 novel features: Search Replay, Temperature Search, Index Oracle, and more
- WebSocket support for real-time search
- Webhooks with HMAC verification
- Background job management
- Full async support (optional)

Quick Start:
    >>> from arrwdb import ArrwDBClient
    >>> client = ArrwDBClient("http://localhost:8000")
    >>> library = client.create_library("my-library", index_type="hnsw")
    >>> client.add_document(library["id"], "Title", ["Text chunk 1", "Text chunk 2"])
    >>> results = client.search(library["id"], "query text", k=10)

For async usage:
    >>> from arrwdb import AsyncArrwDBClient
    >>> async with AsyncArrwDBClient("http://localhost:8000") as client:
    ...     results = await client.search(library_id, "query")
"""

from arrwdb.client import ArrwDBClient, ArrwDBException
from arrwdb.version import __version__

__all__ = [
    "ArrwDBClient",
    "ArrwDBException",
    "__version__",
]

# Optional async client (requires arrwdb[async])
try:
    from arrwdb.async_client import AsyncArrwDBClient
    __all__.append("AsyncArrwDBClient")
except ImportError:
    pass
