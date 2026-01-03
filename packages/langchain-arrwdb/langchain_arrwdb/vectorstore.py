"""arrwDB VectorStore implementation for LangChain."""

from __future__ import annotations

import uuid
from typing import Any, Iterable, List, Optional, Tuple, Type

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from arrwdb import ArrwDBClient


class ArrwDBVectorStore(VectorStore):
    """arrwDB vector store integration for LangChain.

    arrwDB is a production-grade vector database with 9 novel features
    including temperature search, index oracle, and embedding health monitoring.

    Example:
        .. code-block:: python

            from langchain_arrwdb import ArrwDBVectorStore
            from langchain_openai import OpenAIEmbeddings

            vectorstore = ArrwDBVectorStore(
                base_url="http://localhost:8000",
                library_id="my-library",
                embedding=OpenAIEmbeddings(),
            )

            # Add documents
            vectorstore.add_texts(["Hello world", "Goodbye world"])

            # Search
            results = vectorstore.similarity_search("hello", k=5)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        library_id: Optional[str] = None,
        library_name: Optional[str] = None,
        embedding: Optional[Embeddings] = None,
        api_key: Optional[str] = None,
        index_type: str = "hnsw",
        **kwargs: Any,
    ) -> None:
        """Initialize arrwDB vector store.

        Args:
            base_url: arrwDB server URL.
            library_id: Existing library ID to use.
            library_name: Name for new library (if library_id not provided).
            embedding: Embedding model for generating vectors.
                       If None, arrwDB will use its built-in embedding.
            api_key: Optional API key for authentication.
            index_type: Index type for new library (hnsw, ivf, lsh, kdtree, brute_force).
        """
        self._client = ArrwDBClient(base_url=base_url, api_key=api_key)
        self._embedding = embedding
        self._use_server_embeddings = embedding is None

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
    def embeddings(self) -> Optional[Embeddings]:
        """Return the embedding model."""
        return self._embedding

    @property
    def client(self) -> ArrwDBClient:
        """Return the arrwDB client."""
        return self._client

    @property
    def library_id(self) -> str:
        """Return the library ID."""
        return self._library_id

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add texts to the vector store.

        Args:
            texts: Texts to add.
            metadatas: Optional metadata for each text.
            ids: Optional IDs for each text.
            **kwargs: Additional arguments (title, tags).

        Returns:
            List of document IDs.
        """
        texts_list = list(texts)
        if not texts_list:
            return []

        title = kwargs.get("title", f"Document {uuid.uuid4().hex[:8]}")
        tags = kwargs.get("tags", [])

        # If using server embeddings, send texts directly
        if self._use_server_embeddings:
            doc = self._client.add_document(
                library_id=self._library_id,
                title=title,
                texts=texts_list,
                tags=tags,
            )
            return [doc["id"]]

        # Otherwise, generate embeddings locally
        embeddings = self._embedding.embed_documents(texts_list)

        # Add with pre-computed embeddings
        doc = self._client.add_document(
            library_id=self._library_id,
            title=title,
            texts=texts_list,
            tags=tags,
            embeddings=embeddings,
        )
        return [doc["id"]]

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Document]:
        """Search for similar documents.

        Args:
            query: Query text.
            k: Number of results to return.
            **kwargs: Additional search parameters.

        Returns:
            List of similar documents.
        """
        results = self._client.search(
            library_id=self._library_id,
            query=query,
            k=k,
        )

        return [
            Document(
                page_content=r["text"],
                metadata={
                    "distance": r.get("distance"),
                    "document_id": r.get("document_id"),
                    "chunk_index": r.get("chunk_index"),
                    **(r.get("metadata") or {}),
                },
            )
            for r in results.get("results", [])
        ]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Search for similar documents with scores.

        Args:
            query: Query text.
            k: Number of results to return.
            **kwargs: Additional search parameters.

        Returns:
            List of (document, score) tuples.
        """
        results = self._client.search(
            library_id=self._library_id,
            query=query,
            k=k,
        )

        return [
            (
                Document(
                    page_content=r["text"],
                    metadata={
                        "document_id": r.get("document_id"),
                        "chunk_index": r.get("chunk_index"),
                        **(r.get("metadata") or {}),
                    },
                ),
                r.get("distance", 0.0),
            )
            for r in results.get("results", [])
        ]

    def temperature_search(
        self,
        query: str,
        k: int = 4,
        temperature: float = 1.0,
        **kwargs: Any,
    ) -> List[Document]:
        """Search with temperature control for exploration vs exploitation.

        This is a novel arrwDB feature. Temperature controls randomness:
        - 0.0: Deterministic, returns top-k most similar
        - 1.0: Balanced exploration/exploitation
        - >1.0: More diverse, serendipitous results

        Args:
            query: Query text.
            k: Number of results to return.
            temperature: Temperature parameter (0.0 to 2.0+).
            **kwargs: Additional search parameters.

        Returns:
            List of documents.
        """
        results = self._client.temperature_search(
            corpus_id=self._library_id,
            query_text=query,
            k=k,
            temperature=temperature,
        )

        return [
            Document(
                page_content=r["text"],
                metadata={
                    "distance": r.get("distance"),
                    "document_id": r.get("document_id"),
                    "chunk_index": r.get("chunk_index"),
                    **(r.get("metadata") or {}),
                },
            )
            for r in results.get("results", [])
        ]

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

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete documents by ID.

        Args:
            ids: Document IDs to delete.
            **kwargs: Additional arguments.

        Returns:
            True if successful.
        """
        if not ids:
            return None

        for doc_id in ids:
            self._client.delete_document(self._library_id, doc_id)

        return True

    @classmethod
    def from_texts(
        cls: Type["ArrwDBVectorStore"],
        texts: List[str],
        embedding: Optional[Embeddings] = None,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> "ArrwDBVectorStore":
        """Create vector store from texts.

        Args:
            texts: Texts to add.
            embedding: Optional embedding model.
            metadatas: Optional metadata.
            **kwargs: Additional arguments (base_url, library_name, etc.).

        Returns:
            ArrwDBVectorStore instance.
        """
        base_url = kwargs.pop("base_url", "http://localhost:8000")
        library_name = kwargs.pop("library_name", f"langchain-{uuid.uuid4().hex[:8]}")
        api_key = kwargs.pop("api_key", None)
        index_type = kwargs.pop("index_type", "hnsw")

        store = cls(
            base_url=base_url,
            library_name=library_name,
            embedding=embedding,
            api_key=api_key,
            index_type=index_type,
        )

        store.add_texts(texts, metadatas=metadatas, **kwargs)
        return store
