"""
LangChain VectorStore adapter for arrwDB.

Drop-in replacement for PGVector / Pinecone / Weaviate / Qdrant in
any LangChain pipeline.

Usage:
    from langchain_openai import OpenAIEmbeddings
    from arrwdb.integrations.langchain import ArrwDBVectorStore

    embeddings = OpenAIEmbeddings()
    store = ArrwDBVectorStore.from_texts(
        texts=["doc 1", "doc 2", "doc 3"],
        embedding=embeddings,
        base_url="http://localhost:8000",
        library_name="my-rag-library",
    )

    results = store.similarity_search("query", k=5)

Switching from pgvector:
    # Before
    from langchain_community.vectorstores import PGVector
    store = PGVector.from_texts(texts, embedding, ...)

    # After
    from arrwdb.integrations.langchain import ArrwDBVectorStore
    store = ArrwDBVectorStore.from_texts(texts, embedding, ...)
"""

from __future__ import annotations

from typing import Any, Iterable, List, Optional, Tuple
from uuid import uuid4

try:
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    from langchain_core.vectorstores import VectorStore
except ImportError as e:
    raise ImportError(
        "LangChain is required for this integration. "
        "Install with: pip install arrwdb[langchain]"
    ) from e

from arrwdb.client import ArrwDBClient


class ArrwDBVectorStore(VectorStore):
    """
    arrwDB vector store adapter for LangChain.

    Each LangChain text becomes one chunk in a dedicated arrwDB document.
    LangChain metadata is stored on the document. Embeddings are computed
    by the LangChain ``Embeddings`` instance and passed to arrwDB, so no
    server-side embedding provider configuration is required.

    Args:
        embedding: LangChain Embeddings instance.
        library_id: Existing arrwDB library UUID. If not provided, pass
            ``library_name`` and one will be created.
        library_name: Name for a new library (used when ``library_id`` is
            not provided).
        base_url: arrwDB server URL. Defaults to http://localhost:8000.
        index_type: arrwDB index backend ("hnsw", "ivf", etc.). Defaults
            to "hnsw".
        client: Optional pre-configured ``ArrwDBClient``. If provided,
            ``base_url`` is ignored.
    """

    def __init__(
        self,
        embedding: Embeddings,
        library_id: Optional[str] = None,
        library_name: Optional[str] = None,
        base_url: str = "http://localhost:8000",
        index_type: str = "hnsw",
        client: Optional[ArrwDBClient] = None,
    ) -> None:
        self._embedding = embedding
        self._client = client or ArrwDBClient(base_url=base_url)

        if library_id is None:
            if library_name is None:
                library_name = f"langchain-{uuid4().hex[:8]}"
            lib = self._client.create_library(
                name=library_name, index_type=index_type
            )
            self._library_id = lib["id"]
        else:
            self._library_id = library_id

    # ------------------------------------------------------------------
    # Properties required by the VectorStore interface
    # ------------------------------------------------------------------

    @property
    def embeddings(self) -> Embeddings:
        return self._embedding

    @property
    def library_id(self) -> str:
        return self._library_id

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Embed ``texts`` with the configured LangChain embeddings and
        insert them into arrwDB.

        Returns a list of arrwDB document IDs (one per input text).
        """
        texts_list = list(texts)
        if not texts_list:
            return []

        vectors = self._embedding.embed_documents(texts_list)
        metadatas = metadatas or [{}] * len(texts_list)

        ids: List[str] = []
        for text, vector, meta in zip(texts_list, vectors, metadatas):
            title = (meta.get("title") or text[:80]).strip()
            doc = self._client.add_document_with_embeddings(
                library_id=self._library_id,
                title=title,
                chunks=[(text, list(vector))],
                tags=meta.get("tags"),
            )
            ids.append(doc["id"])
        return ids

    def add_documents(
        self, documents: List[Document], **kwargs: Any
    ) -> List[str]:
        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]
        return self.add_texts(texts, metadatas, **kwargs)

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        docs_and_scores = self.similarity_search_with_score(query, k=k, **kwargs)
        return [doc for doc, _ in docs_and_scores]

    def similarity_search_by_vector(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        response = self._client.search_with_embedding(
            library_id=self._library_id, embedding=embedding, k=k
        )
        return [self._result_to_document(r) for r in response["results"]]

    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        vector = self._embedding.embed_query(query)
        response = self._client.search_with_embedding(
            library_id=self._library_id, embedding=vector, k=k
        )
        return [
            (self._result_to_document(r), r["distance"])
            for r in response["results"]
        ]

    @staticmethod
    def _result_to_document(result: dict) -> Document:
        chunk = result["chunk"]
        return Document(
            page_content=chunk["text"],
            metadata={
                "document_id": str(result["document_id"]),
                "document_title": result["document_title"],
                "chunk_id": str(chunk["id"]),
                "distance": result["distance"],
            },
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        library_id: Optional[str] = None,
        library_name: Optional[str] = None,
        base_url: str = "http://localhost:8000",
        index_type: str = "hnsw",
        client: Optional[ArrwDBClient] = None,
        **kwargs: Any,
    ) -> "ArrwDBVectorStore":
        store = cls(
            embedding=embedding,
            library_id=library_id,
            library_name=library_name,
            base_url=base_url,
            index_type=index_type,
            client=client,
        )
        store.add_texts(texts, metadatas)
        return store
