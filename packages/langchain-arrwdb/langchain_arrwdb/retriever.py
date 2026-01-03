"""arrwDB Retriever implementation for LangChain."""

from __future__ import annotations

from typing import Any, List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from langchain_arrwdb.vectorstore import ArrwDBVectorStore


class ArrwDBRetriever(BaseRetriever):
    """arrwDB retriever for LangChain RAG pipelines.

    Supports both standard similarity search and temperature-based
    exploration search.

    Example:
        .. code-block:: python

            from langchain_arrwdb import ArrwDBRetriever

            retriever = ArrwDBRetriever(
                base_url="http://localhost:8000",
                library_id="my-library",
                k=5,
            )

            # Use in RAG chain
            docs = retriever.invoke("What is machine learning?")

            # Or with temperature for diverse results
            retriever_exploratory = ArrwDBRetriever(
                base_url="http://localhost:8000",
                library_id="my-library",
                k=10,
                temperature=1.5,  # Enable temperature search
            )
    """

    vectorstore: ArrwDBVectorStore
    """The underlying vector store."""

    k: int = 4
    """Number of documents to retrieve."""

    temperature: Optional[float] = None
    """Optional temperature for exploration search. If None, uses standard search."""

    search_kwargs: dict = {}
    """Additional search parameters."""

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        vectorstore: Optional[ArrwDBVectorStore] = None,
        base_url: str = "http://localhost:8000",
        library_id: Optional[str] = None,
        k: int = 4,
        temperature: Optional[float] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize retriever.

        Args:
            vectorstore: Existing ArrwDBVectorStore instance.
            base_url: arrwDB server URL (if creating new vectorstore).
            library_id: Library ID (if creating new vectorstore).
            k: Number of documents to retrieve.
            temperature: Optional temperature for exploration search.
            api_key: Optional API key.
            **kwargs: Additional arguments.
        """
        if vectorstore is None:
            if library_id is None:
                raise ValueError("Either vectorstore or library_id must be provided")
            vectorstore = ArrwDBVectorStore(
                base_url=base_url,
                library_id=library_id,
                api_key=api_key,
            )

        super().__init__(
            vectorstore=vectorstore,
            k=k,
            temperature=temperature,
            **kwargs,
        )

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """Get relevant documents for a query.

        Args:
            query: Query string.
            run_manager: Callback manager.

        Returns:
            List of relevant documents.
        """
        if self.temperature is not None:
            # Use temperature search for exploration
            return self.vectorstore.temperature_search(
                query=query,
                k=self.k,
                temperature=self.temperature,
                **self.search_kwargs,
            )
        else:
            # Standard similarity search
            return self.vectorstore.similarity_search(
                query=query,
                k=self.k,
                **self.search_kwargs,
            )

    @classmethod
    def from_vectorstore(
        cls,
        vectorstore: ArrwDBVectorStore,
        k: int = 4,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> "ArrwDBRetriever":
        """Create retriever from existing vectorstore.

        Args:
            vectorstore: ArrwDBVectorStore instance.
            k: Number of documents to retrieve.
            temperature: Optional temperature for exploration.
            **kwargs: Additional arguments.

        Returns:
            ArrwDBRetriever instance.
        """
        return cls(
            vectorstore=vectorstore,
            k=k,
            temperature=temperature,
            **kwargs,
        )
