"""
VectorDBBench adapter for arrwDB.

This module provides the integration between VectorDBBench benchmarking tool
and arrwDB vector database, allowing standardized performance comparisons.

To use with VectorDBBench:
1. Install VectorDBBench: pip install vectordb-bench
2. Copy this file to vectordb_bench/backend/clients/arrwdb/
3. Register in vectordb_bench/backend/clients/__init__.py
4. Run benchmarks: python run.py --db arrwdb

See: https://github.com/zilliztech/VectorDBBench
"""

from contextlib import contextmanager
from typing import Any, Generator
import logging
import time

# VectorDBBench imports (when integrated)
try:
    from vectordb_bench.backend.clients.api import VectorDB, DBCaseConfig
except ImportError:
    # Standalone mode - define base class for development
    from abc import ABC, abstractmethod

    class DBCaseConfig:
        """Placeholder for VectorDBBench config."""
        pass

    class VectorDB(ABC):
        """Base class placeholder for standalone development."""

        @abstractmethod
        def __init__(self, dim: int, db_config: dict, db_case_config: DBCaseConfig | None,
                     collection_name: str, drop_old: bool = False, **kwargs) -> None:
            pass

        @abstractmethod
        @contextmanager
        def init(self) -> Generator[None, None, None]:
            pass

        @abstractmethod
        def insert_embeddings(self, embeddings: list[list[float]], metadata: list[int],
                              labels_data: list[str] | None = None, **kwargs) -> tuple[int, Exception | None]:
            pass

        @abstractmethod
        def search_embedding(self, query: list[float], k: int = 100) -> list[int]:
            pass

        @abstractmethod
        def optimize(self, data_size: int | None = None) -> None:
            pass

# arrwDB imports
from arrwdb import ArrwDBClient

logger = logging.getLogger(__name__)


class ArrwDBConfig:
    """Configuration for arrwDB connection."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: str | None = None,
        index_type: str = "hnsw",
        hnsw_m: int = 16,
        hnsw_ef_construction: int = 200,
        hnsw_ef_search: int = 50,
        **kwargs: Any,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.index_type = index_type
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construction = hnsw_ef_construction
        self.hnsw_ef_search = hnsw_ef_search
        self.extra = kwargs


class ArrwDB(VectorDB):
    """VectorDBBench adapter for arrwDB vector database.

    arrwDB is a production-grade vector database with 9 novel features:
    - Temperature Search: Exploration vs exploitation control
    - Index Oracle: Intelligent index recommendations
    - Embedding Health: Quality monitoring and drift detection
    - Search Replay: HNSW traversal debugging
    - Vector Clustering: K-means with auto-k
    - Query Expansion: Automatic query rewriting
    - Vector Drift: Distribution monitoring
    - Adaptive Reranking: Feedback-based learning
    - Hybrid Fusion: Multi-strategy merging

    Supports multiple index types: HNSW, IVF, LSH, KD-Tree, Brute Force
    """

    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: DBCaseConfig | None,
        collection_name: str = "vectordbbench",
        drop_old: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize arrwDB adapter.

        Args:
            dim: Vector dimension.
            db_config: Database configuration dict.
            db_case_config: Benchmark case configuration.
            collection_name: Name for the library/collection.
            drop_old: Whether to drop existing collection.
            **kwargs: Additional arguments.
        """
        self.dim = dim
        self.collection_name = collection_name
        self.drop_old = drop_old

        # Parse config
        if isinstance(db_config, dict):
            self.config = ArrwDBConfig(**db_config)
        else:
            self.config = db_config

        self.db_case_config = db_case_config

        # Client and library ID (initialized in init())
        self._client: ArrwDBClient | None = None
        self._library_id: str | None = None

        # Document ID tracking for metadata mapping
        self._id_map: dict[int, str] = {}

    @contextmanager
    def init(self) -> Generator[None, None, None]:
        """Initialize connection and collection.

        Creates the arrwDB client, drops old collection if requested,
        and creates a new library with configured index type.
        """
        try:
            # Create client
            self._client = ArrwDBClient(
                base_url=self.config.base_url,
                api_key=self.config.api_key,
            )

            # Check health
            health = self._client.health_check()
            logger.info(f"arrwDB health: {health}")

            # Drop old collection if requested
            if self.drop_old:
                try:
                    libraries = self._client.list_libraries()
                    for lib in libraries:
                        if lib.get("name") == self.collection_name:
                            self._client.delete_library(lib["id"])
                            logger.info(f"Dropped existing library: {self.collection_name}")
                            break
                except Exception as e:
                    logger.warning(f"Failed to drop old library: {e}")

            # Create new library
            index_config = {}
            if self.config.index_type == "hnsw":
                index_config = {
                    "m": self.config.hnsw_m,
                    "ef_construction": self.config.hnsw_ef_construction,
                    "ef_search": self.config.hnsw_ef_search,
                }

            library = self._client.create_library(
                name=self.collection_name,
                description=f"VectorDBBench benchmark - dim={self.dim}",
                index_type=self.config.index_type,
                index_config=index_config,
            )
            self._library_id = library["id"]
            logger.info(f"Created library: {self._library_id} with index {self.config.index_type}")

            yield

        finally:
            # Cleanup
            self._client = None
            self._library_id = None
            self._id_map.clear()

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
        **kwargs: Any,
    ) -> tuple[int, Exception | None]:
        """Insert embeddings with metadata.

        Args:
            embeddings: List of embedding vectors.
            metadata: List of integer IDs for each embedding.
            labels_data: Optional labels/text for each embedding.
            **kwargs: Additional arguments.

        Returns:
            Tuple of (inserted count, exception or None).
        """
        if self._client is None or self._library_id is None:
            return 0, RuntimeError("Client not initialized")

        try:
            inserted = 0
            batch_size = kwargs.get("batch_size", 100)

            # Insert in batches
            for i in range(0, len(embeddings), batch_size):
                batch_embeddings = embeddings[i:i + batch_size]
                batch_metadata = metadata[i:i + batch_size]
                batch_labels = labels_data[i:i + batch_size] if labels_data else None

                # Prepare texts (use labels or generate placeholder)
                if batch_labels:
                    texts = batch_labels
                else:
                    texts = [f"doc_{mid}" for mid in batch_metadata]

                # Add document with pre-computed embeddings
                doc = self._client.add_document(
                    library_id=self._library_id,
                    title=f"batch_{i}",
                    texts=texts,
                    embeddings=batch_embeddings,
                    metadata=[{"id": mid} for mid in batch_metadata],
                )

                # Track ID mapping
                doc_id = doc["id"]
                for j, mid in enumerate(batch_metadata):
                    self._id_map[mid] = f"{doc_id}:{j}"

                inserted += len(batch_embeddings)

            logger.info(f"Inserted {inserted} embeddings")
            return inserted, None

        except Exception as e:
            logger.error(f"Insert failed: {e}")
            return 0, e

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        **kwargs: Any,
    ) -> list[int]:
        """Search for similar embeddings.

        Args:
            query: Query embedding vector.
            k: Number of results to return.
            **kwargs: Additional search parameters.

        Returns:
            List of metadata IDs for top-k results.
        """
        if self._client is None or self._library_id is None:
            return []

        try:
            # Search by vector
            results = self._client.search_by_vector(
                library_id=self._library_id,
                vector=query,
                k=k,
            )

            # Extract metadata IDs from results
            ids = []
            for r in results.get("results", []):
                meta = r.get("metadata", {})
                if "id" in meta:
                    ids.append(meta["id"])

            return ids

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def optimize(self, data_size: int | None = None) -> None:
        """Optimize index after insertions.

        Waits for arrwDB to complete any background indexing operations.

        Args:
            data_size: Optional hint about total data size.
        """
        if self._client is None or self._library_id is None:
            return

        try:
            # Wait for indexing to complete
            # arrwDB indexes synchronously, but we poll stats to confirm
            max_wait = 300  # 5 minutes max
            poll_interval = 1

            start = time.time()
            while time.time() - start < max_wait:
                stats = self._client.get_library_stats(self._library_id)

                # Check if indexed count matches expected
                indexed = stats.get("total_chunks", 0)
                if data_size and indexed >= data_size:
                    break

                # Check if index is ready
                if stats.get("index_ready", True):
                    break

                time.sleep(poll_interval)

            logger.info(f"Optimization complete: {stats}")

        except Exception as e:
            logger.warning(f"Optimize check failed: {e}")

    # =========================================================================
    # Novel Features - arrwDB specific benchmarks
    # =========================================================================

    def temperature_search(
        self,
        query: list[float],
        k: int = 100,
        temperature: float = 1.0,
    ) -> list[int]:
        """Temperature-controlled search (arrwDB novel feature).

        Args:
            query: Query embedding vector.
            k: Number of results.
            temperature: Exploration parameter (0=greedy, >1=exploratory).

        Returns:
            List of metadata IDs.
        """
        if self._client is None or self._library_id is None:
            return []

        try:
            results = self._client.temperature_search_by_vector(
                corpus_id=self._library_id,
                vector=query,
                k=k,
                temperature=temperature,
            )

            ids = []
            for r in results.get("results", []):
                meta = r.get("metadata", {})
                if "id" in meta:
                    ids.append(meta["id"])

            return ids

        except Exception as e:
            logger.error(f"Temperature search failed: {e}")
            return []

    def get_index_recommendation(self) -> dict:
        """Get Index Oracle recommendation (arrwDB novel feature)."""
        if self._client is None or self._library_id is None:
            return {}

        return self._client.get_index_recommendation(self._library_id)

    def get_embedding_health(self) -> dict:
        """Get embedding health analysis (arrwDB novel feature)."""
        if self._client is None or self._library_id is None:
            return {}

        return self._client.analyze_embedding_health(self._library_id)


# Registration for VectorDBBench
def get_db_class():
    """Return the database class for VectorDBBench registration."""
    return ArrwDB


def get_db_config_class():
    """Return the config class for VectorDBBench registration."""
    return ArrwDBConfig
