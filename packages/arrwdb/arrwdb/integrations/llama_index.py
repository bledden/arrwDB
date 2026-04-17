"""
LlamaIndex VectorStore adapter for arrwDB.

Usage:
    from llama_index.core import VectorStoreIndex, StorageContext
    from arrwdb.integrations.llama_index import ArrwDBVectorStore

    vector_store = ArrwDBVectorStore(
        base_url="http://localhost:8000",
        library_name="my-rag-library",
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)

    query_engine = index.as_query_engine()
    response = query_engine.query("your question")
"""

from __future__ import annotations

from typing import Any, List, Optional
from uuid import uuid4

try:
    from llama_index.core.schema import BaseNode, MetadataMode, TextNode
    from llama_index.core.vector_stores.types import (
        BasePydanticVectorStore,
        VectorStoreQuery,
        VectorStoreQueryResult,
    )
except ImportError as e:
    raise ImportError(
        "LlamaIndex is required for this integration. "
        "Install with: pip install arrwdb[llamaindex]"
    ) from e

from arrwdb.client import ArrwDBClient


class ArrwDBVectorStore(BasePydanticVectorStore):
    """arrwDB vector store adapter for LlamaIndex.

    This adapter stores nodes as single-chunk documents in arrwDB and
    accepts pre-computed embeddings from LlamaIndex's embedding pipeline.
    """

    stores_text: bool = True
    flat_metadata: bool = True

    # Pydantic model fields
    base_url: str = "http://localhost:8000"
    library_id: Optional[str] = None
    library_name: Optional[str] = None
    index_type: str = "hnsw"

    # Private state (non-serialized)
    _client: ArrwDBClient
    _node_id_to_doc_id: dict

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        library_id: Optional[str] = None,
        library_name: Optional[str] = None,
        index_type: str = "hnsw",
        client: Optional[ArrwDBClient] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            base_url=base_url,
            library_id=library_id,
            library_name=library_name,
            index_type=index_type,
            **kwargs,
        )
        self._client = client or ArrwDBClient(base_url=base_url)
        self._node_id_to_doc_id = {}

        if self.library_id is None:
            name = self.library_name or f"llamaindex-{uuid4().hex[:8]}"
            lib = self._client.create_library(name=name, index_type=index_type)
            self.library_id = lib["id"]

    @property
    def client(self) -> ArrwDBClient:
        return self._client

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add(self, nodes: List[BaseNode], **kwargs: Any) -> List[str]:
        ids: List[str] = []
        for node in nodes:
            if node.embedding is None:
                raise ValueError(
                    f"Node {node.node_id} has no embedding. LlamaIndex "
                    "should populate embeddings before calling add()."
                )
            text = node.get_content(metadata_mode=MetadataMode.NONE)
            meta = node.metadata or {}
            title = (meta.get("title") or meta.get("file_name") or text[:80]).strip()
            doc = self._client.add_document_with_embeddings(
                library_id=self.library_id,
                title=title,
                chunks=[(text, list(node.embedding))],
                tags=meta.get("tags"),
            )
            self._node_id_to_doc_id[node.node_id] = doc["id"]
            ids.append(node.node_id)
        return ids

    def delete(self, ref_doc_id: str, **kwargs: Any) -> None:
        doc_id = self._node_id_to_doc_id.pop(ref_doc_id, None)
        if doc_id is not None:
            self._client.delete_document(doc_id)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        if query.query_embedding is None:
            raise ValueError(
                "query.query_embedding is required — LlamaIndex should "
                "provide a pre-computed embedding."
            )

        response = self._client.search_with_embedding(
            library_id=self.library_id,
            embedding=list(query.query_embedding),
            k=query.similarity_top_k,
        )

        nodes: List[TextNode] = []
        similarities: List[float] = []
        ids: List[str] = []
        for r in response["results"]:
            chunk = r["chunk"]
            node = TextNode(
                text=chunk["text"],
                id_=str(chunk["id"]),
                metadata={
                    "document_id": str(r["document_id"]),
                    "document_title": r["document_title"],
                },
            )
            nodes.append(node)
            # arrwDB returns distance; convert to similarity (1 - distance for cosine)
            similarities.append(1.0 - float(r["distance"]))
            ids.append(str(chunk["id"]))

        return VectorStoreQueryResult(
            nodes=nodes, similarities=similarities, ids=ids
        )
