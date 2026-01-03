"""arrwDB VectorStore implementation for LlamaIndex."""

from __future__ import annotations

import uuid
from typing import Any, List, Optional

from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)

from arrwdb import ArrwDBClient


class ArrwDBVectorStore(BasePydanticVectorStore):
    """arrwDB vector store for LlamaIndex.

    arrwDB is a production-grade vector database with 9 novel features
    including temperature search, index oracle, and embedding health monitoring.

    Example:
        .. code-block:: python

            from llama_index.core import VectorStoreIndex
            from llama_index_arrwdb import ArrwDBVectorStore

            vector_store = ArrwDBVectorStore(
                base_url="http://localhost:8000",
                library_id="my-library",
            )

            index = VectorStoreIndex.from_vector_store(vector_store)
            query_engine = index.as_query_engine()
            response = query_engine.query("What is machine learning?")
    """

    stores_text: bool = True
    flat_metadata: bool = True

    _client: ArrwDBClient
    _library_id: str
    _use_server_embeddings: bool

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        library_id: Optional[str] = None,
        library_name: Optional[str] = None,
        api_key: Optional[str] = None,
        index_type: str = "hnsw",
        use_server_embeddings: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize arrwDB vector store.

        Args:
            base_url: arrwDB server URL.
            library_id: Existing library ID to use.
            library_name: Name for new library (if library_id not provided).
            api_key: Optional API key for authentication.
            index_type: Index type for new library (hnsw, ivf, lsh, kdtree, brute_force).
            use_server_embeddings: Whether to use arrwDB's built-in embedding.
        """
        super().__init__()

        self._client = ArrwDBClient(base_url=base_url, api_key=api_key)
        self._use_server_embeddings = use_server_embeddings

        # Get or create library
        if library_id:
            self._library_id = library_id
        elif library_name:
            library = self._client.create_library(
                name=library_name,
                index_type=index_type,
            )
            self._library_id = library["id"]
        else:
            raise ValueError("Either library_id or library_name must be provided")

    @property
    def client(self) -> ArrwDBClient:
        """Return the arrwDB client."""
        return self._client

    @property
    def library_id(self) -> str:
        """Return the library ID."""
        return self._library_id

    def add(
        self,
        nodes: List[BaseNode],
        **kwargs: Any,
    ) -> List[str]:
        """Add nodes to vector store.

        Args:
            nodes: List of nodes to add.
            **kwargs: Additional arguments.

        Returns:
            List of node IDs.
        """
        if not nodes:
            return []

        ids = []
        for node in nodes:
            text = node.get_content()
            node_id = node.node_id or str(uuid.uuid4())

            title = kwargs.get("title", f"Node {node_id[:8]}")
            tags = list(node.metadata.keys()) if node.metadata else []

            if self._use_server_embeddings:
                # Let arrwDB generate embeddings
                doc = self._client.add_document(
                    library_id=self._library_id,
                    title=title,
                    texts=[text],
                    tags=tags,
                )
            else:
                # Use pre-computed embedding from node
                embedding = node.get_embedding()
                doc = self._client.add_document(
                    library_id=self._library_id,
                    title=title,
                    texts=[text],
                    tags=tags,
                    embeddings=[embedding] if embedding else None,
                )

            ids.append(doc["id"])

        return ids

    def delete(self, ref_doc_id: str, **kwargs: Any) -> None:
        """Delete a document by ID.

        Args:
            ref_doc_id: Document ID to delete.
            **kwargs: Additional arguments.
        """
        self._client.delete_document(self._library_id, ref_doc_id)

    def query(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """Query the vector store.

        Args:
            query: Vector store query.
            **kwargs: Additional arguments (temperature for exploration search).

        Returns:
            Query results.
        """
        k = query.similarity_top_k or 10
        temperature = kwargs.get("temperature")

        # Use temperature search if specified
        if temperature is not None:
            if query.query_str:
                results = self._client.temperature_search(
                    corpus_id=self._library_id,
                    query_text=query.query_str,
                    k=k,
                    temperature=temperature,
                )
            else:
                raise ValueError("Temperature search requires query_str")
        else:
            # Standard search
            if query.query_str:
                results = self._client.search(
                    library_id=self._library_id,
                    query=query.query_str,
                    k=k,
                )
            elif query.query_embedding:
                # Search with pre-computed embedding
                results = self._client.search_by_vector(
                    library_id=self._library_id,
                    vector=query.query_embedding,
                    k=k,
                )
            else:
                raise ValueError("Either query_str or query_embedding required")

        # Convert to LlamaIndex format
        nodes = []
        similarities = []
        ids = []

        for r in results.get("results", []):
            node = TextNode(
                text=r["text"],
                id_=r.get("document_id", str(uuid.uuid4())),
                metadata={
                    "chunk_index": r.get("chunk_index"),
                    **(r.get("metadata") or {}),
                },
            )
            nodes.append(node)

            # Convert distance to similarity (assuming cosine distance)
            distance = r.get("distance", 0.0)
            similarity = 1.0 - distance
            similarities.append(similarity)

            ids.append(r.get("document_id", ""))

        return VectorStoreQueryResult(
            nodes=nodes,
            similarities=similarities,
            ids=ids,
        )

    def temperature_query(
        self,
        query_str: str,
        k: int = 10,
        temperature: float = 1.0,
    ) -> VectorStoreQueryResult:
        """Query with temperature control for exploration vs exploitation.

        This is a novel arrwDB feature. Temperature controls randomness:
        - 0.0: Deterministic, returns top-k most similar
        - 1.0: Balanced exploration/exploitation
        - >1.0: More diverse, serendipitous results

        Args:
            query_str: Query string.
            k: Number of results.
            temperature: Temperature parameter (0.0 to 2.0+).

        Returns:
            Query results.
        """
        query = VectorStoreQuery(
            query_str=query_str,
            similarity_top_k=k,
        )
        return self.query(query, temperature=temperature)

    def get_index_recommendation(self) -> dict:
        """Get intelligent index type recommendation from Index Oracle.

        This is a novel arrwDB feature that analyzes your data
        and workload to recommend the optimal index type.

        Returns:
            Recommendation with reasoning.
        """
        return self._client.get_index_recommendation(self._library_id)

    def analyze_embedding_health(self) -> dict:
        """Analyze embedding quality and detect issues.

        This is a novel arrwDB feature that detects:
        - Outlier embeddings
        - Embedding degeneracy
        - Distribution drift

        Returns:
            Health analysis results.
        """
        return self._client.analyze_embedding_health(self._library_id)
